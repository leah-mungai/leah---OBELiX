from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pymatgen.core import Structure
from tqdm import tqdm
from functools import partial
import lightning as pl
from sklearn.model_selection import KFold
from pytorch_lightning.loggers import CSVLogger
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_graph
from matgl.models import M3GNet
from matgl.utils.training import ModelLightningModule
import numpy as np
from torch.utils.data import Subset

# Load configuration
with open('config_m3gnet.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Set environment variable to avoid threading issues
os.environ["OMP_NUM_THREADS"] = "1"

# Function to dynamically get the latest version folder
def get_latest_version(logs_dir):
    versions = [d for d in os.listdir(logs_dir) if d.startswith("version_")]
    if versions:
        latest_version = max(versions, key=lambda x: int(x.split('_')[-1]))
        return latest_version
    return "version_0"

# Step 1: Dataset Preparation
def load_dataset() -> tuple[list[Structure], list[str], list[float]]:
    updated_data = pd.read_csv(config["data"]["csv_file"])
    updated_data['ID'] = updated_data['ID'].str.replace(r'^="|"$', '', regex=True)
    print(f"Number of rows in the CSV: {len(updated_data)}")

    structures = []
    ids = []
    for i, row in tqdm(updated_data.iterrows(), total=len(updated_data)):
        try:
            struct = Structure.from_file(
                os.path.join(config["data"]["cif_directory"], f'{row["ID"]}.cif'))
            structures.append(struct)
            ids.append(row["ID"])
        except OSError as e:
            print(f"Error loading structure for ID {row['ID']}: {e}")
        
    ionic_conductivities = np.log10(updated_data[config["data"]["ionic_conductivity_column"]].values)

    print(f"Number of structures after loading: {len(structures)}")
    print(f"Number of IDs: {len(ids)}")
    print(f"Number of Ionic Conductivities: {len(ionic_conductivities)}")
    
    return structures, ids, ionic_conductivities

structures, ids, ionic_conductivities = load_dataset()

# Step 2: Convert dataset to MGLDataset format
elem_list = get_element_list(structures)
converter = Structure2Graph(
    element_types=elem_list, cutoff=config["graph"]["cutoff"])

my_dataset = MGLDataset(
    threebody_cutoff=config["graph"]["threebody_cutoff"],
    structures=structures,
    converter=converter,
    labels={"ionic_conductivity": ionic_conductivities},
    include_line_graph=True,
    raw_dir=config["output"]["raw_data_dir"]
)

print(f"Total number of entries in the dataset: {len(my_dataset)}")

# Step 3: Apply K-Fold Cross Validation
kf = KFold(n_splits=config["split"]["k_folds"], 
           shuffle=config["split"]["shuffle"], 
           random_state=config["split"]["random_state"])

mae_train_folds = []
mae_val_folds = []
loss_train_folds = []
loss_val_folds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(my_dataset), start=1):
    print(f"\nFold {fold}/{config['split']['k_folds']}")
    train_data = Subset(my_dataset, train_idx)
    val_data = Subset(my_dataset, val_idx)

    print(f"Training set size: {len(train_idx)} entries")
    print(f"Validation set size: {len(val_idx)} entries")
    
    my_collate_fn = partial(collate_fn_graph, include_line_graph=True)
    train_loader, val_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        collate_fn=my_collate_fn,
        batch_size=config["dataloader"]["batch_size"],
        num_workers=config["dataloader"]["num_workers"]
    )
    
    model = M3GNet(
        element_types=elem_list,
        is_intensive=config["model"]["is_intensive"],
        readout_type=config["model"]["readout_type"]
    )
    lit_module = ModelLightningModule(model=model, include_line_graph=True, lr=config["training"]["lr"])

    logger = CSVLogger(config["logger"]["save_dir"], name=config["logger"]["name"])
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator=config["training"]["accelerator"],
        logger=logger,
        log_every_n_steps=config["training"]["log_every_n_steps"]
    )
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    logs_dir = os.path.join(config["logger"]["save_dir"], config["logger"]["name"])
    version_dir = get_latest_version(logs_dir)
    metrics_path = os.path.join(logs_dir, version_dir, config["output"]["metrics_file"])
    metrics = pd.read_csv(metrics_path)
    
    metrics['epoch'] = metrics['epoch'] + 1

    val_metrics = metrics.iloc[::2].reset_index(drop=True)
    train_metrics = metrics.iloc[1::2].reset_index(drop=True)

    mae_train_folds.append(train_metrics["train_MAE"].values)
    mae_val_folds.append(val_metrics["val_MAE"].values)
    loss_train_folds.append(train_metrics["train_Total_Loss"].values)
    loss_val_folds.append(val_metrics["val_Total_Loss"].values)

    final_train_mae = np.mean(train_metrics["train_MAE"].values)
    final_val_mae = np.mean(val_metrics["val_MAE"].values)
    final_train_loss = np.mean(train_metrics["train_Total_Loss"].values)
    final_val_loss = np.mean(val_metrics["val_Total_Loss"].values)

    print(f"Fold {fold}: Final Train MAE: {final_train_mae:.4f}, Final Validation MAE: {final_val_mae:.4f}")
    print(f"Fold {fold}: Final Train Loss: {final_train_loss:.4f}, Final Validation Loss: {final_val_loss:.4f}")

    plt.figure(figsize=tuple(config["plot"]["fig_size"]))
    plt.plot(train_metrics["epoch"], train_metrics["train_MAE"], label="Train MAE", linewidth=2, marker='o')
    plt.plot(val_metrics["epoch"], val_metrics["val_MAE"], label="Validation MAE", linewidth=2, marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title(f"Fold {fold}: MAE vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"MAE_vs_Epochs_Fold_{fold}.png", dpi=config["plot"]["dpi"])

    plt.figure(figsize=tuple(config["plot"]["fig_size"]))
    plt.plot(train_metrics["epoch"], train_metrics["train_Total_Loss"], label="Train Loss", linewidth=2, marker='o')
    plt.plot(val_metrics["epoch"], val_metrics["val_Total_Loss"], label="Validation Loss", linewidth=2, marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Total Loss")
    plt.title(f"Fold {fold}: Loss vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Loss_vs_Epochs_Fold_{fold}.png", dpi=config["plot"]["dpi"])

# Step 4: Results 
epochs_range = train_metrics['epoch'].values
mae_train_avg = np.mean(mae_train_folds, axis=0)
mae_val_avg = np.mean(mae_val_folds, axis=0)
loss_train_avg = np.mean(loss_train_folds, axis=0)
loss_val_avg = np.mean(loss_val_folds, axis=0)

# Calculate standard deviations for MAE and Loss
mae_train_std = np.std(mae_train_folds, axis=0)
mae_val_std = np.std(mae_val_folds, axis=0)
loss_train_std = np.std(loss_train_folds, axis=0)
loss_val_std = np.std(loss_val_folds, axis=0)

plt.figure(figsize=tuple(config["plot"]["fig_size"]))
plt.plot(epochs_range, mae_train_avg, label="Avg Train MAE", linewidth=2, marker='o')
plt.plot(epochs_range, mae_val_avg, label="Avg Validation MAE", linewidth=2, marker='o')
plt.fill_between(epochs_range, mae_train_avg - mae_train_std, mae_train_avg + mae_train_std, alpha=0.3)
plt.fill_between(epochs_range, mae_val_avg - mae_val_std, mae_val_avg + mae_val_std, alpha=0.3)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Mean Absolute Error (MAE)", fontsize=14)
plt.title("Average MAE vs Epochs Across Folds", fontsize=16)
plt.legend()
plt.grid(True)
plt.savefig("Average_MAE_vs_Epochs.png", dpi=config["plot"]["dpi"])

plt.figure(figsize=tuple(config["plot"]["fig_size"]))
plt.plot(epochs_range, loss_train_avg, label="Avg Train Loss", linewidth=2, marker='o')
plt.plot(epochs_range, loss_val_avg, label="Avg Validation Loss", linewidth=2, marker='o')
plt.fill_between(epochs_range, loss_train_avg - loss_train_std, loss_train_avg + loss_train_std, alpha=0.3)
plt.fill_between(epochs_range, loss_val_avg - loss_val_std, loss_val_avg + loss_val_std, alpha=0.3)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Total Loss", fontsize=14)
plt.title("Average Loss vs Epochs Across Folds", fontsize=16)
plt.legend()
plt.grid(True)
plt.savefig("Average_Loss_vs_Epochs.png", dpi=config["plot"]["dpi"])

results_df = pd.DataFrame({
    "Epoch": epochs_range,
    "Train_MAE_Avg": mae_train_avg,
    "Train_MAE_Std": mae_train_std,
    "Val_MAE_Avg": mae_val_avg,
    "Val_MAE_Std": mae_val_std,
    "Train_Loss_Avg": loss_train_avg,
    "Train_Loss_Std": loss_train_std,
    "Val_Loss_Avg": loss_val_avg,
    "Val_Loss_Std": loss_val_std,
})
results_df.to_csv(config["output"]["results_csv"], index=False)

print(f"Average Train MAE: {np.mean(mae_train_avg):.4f} ± {np.std(mae_train_avg):.4f}")
print(f"Average Validation MAE: {np.mean(mae_val_avg):.4f} ± {np.std(mae_val_avg):.4f}")
print(f"Average Train Loss: {np.mean(loss_train_avg):.4f} ± {np.std(loss_train_avg):.4f}")
print(f"Average Validation Loss: {np.mean(loss_val_avg):.4f} ± {np.std(loss_val_avg):.4f}")
