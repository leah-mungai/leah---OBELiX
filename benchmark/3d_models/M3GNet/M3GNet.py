from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from tqdm import tqdm
from functools import partial
import lightning as pl
from dgl.data.utils import split_dataset
import numpy as np
import yaml
from pytorch_lightning.loggers import CSVLogger
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_graph
from matgl.models import M3GNet
from matgl.utils.training import ModelLightningModule

# Load configuration from config.yaml
with open('config_m3gnet.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Set environment variable to avoid any threading issues
os.environ["OMP_NUM_THREADS"] = "1"

# Step 1: Dataset Preparation
def load_dataset() -> tuple[list[Structure], list[str], list[float]]:
    updated_data = pd.read_csv(config['data']['csv_file'])
    updated_data['ID'] = updated_data['ID'].str.replace(r'^="|"$', '', regex=True)

    structures = []
    ids = []
    for i, row in tqdm(updated_data.iterrows(), total=len(updated_data)):
        try:
            # Load structure from the cleaned CIF file path
            struct = Structure.from_file(os.path.join(config['data']['cif_directory'], f'{row["ID"]}.cif'))
            structures.append(struct)
            ids.append(row["ID"])
        except OSError as e:
            print(f"Error loading structure for ID {row['ID']}: {e}")

    ionic_conductivities = np.log10(updated_data[config['data']['ionic_conductivity_column']].values)

    return structures, ids, ionic_conductivities

structures, ids, ionic_conductivities = load_dataset()

# Step 2: Convert dataset to MGLDataset format
elem_list = get_element_list(structures)
converter = Structure2Graph(element_types=elem_list, cutoff=config['graph']['cutoff'])

# Create the MGLDataset
dataset = MGLDataset(
    threebody_cutoff=config['graph']['threebody_cutoff'],
    structures=structures,
    converter=converter,
    labels={"ionic_conductivity": ionic_conductivities},
    include_line_graph=True,
)

# Step 3: Split dataset into train, validation, and test sets
train_data, val_data, test_data = split_dataset(
    dataset,
    frac_list=config['split']['frac_list'],
    shuffle=config['split']['shuffle'],
    random_state=config['split']['random_state']
)

# Collate function for creating batches
my_collate_fn = partial(collate_fn_graph, include_line_graph=True)

# Create DataLoaders for train, validation, and test datasets
train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=my_collate_fn,
    batch_size=config['dataloader']['batch_size'],
    num_workers=config['dataloader']['num_workers'],
)

# Step 4: Setup M3GNet Model and Training Module
model = M3GNet(
    element_types=elem_list,
    is_intensive=config['model']['is_intensive'],
    readout_type=config['model']['readout_type'],
)

# Lightning module for training the M3GNet model
lit_module = ModelLightningModule(model=model, include_line_graph=True, lr=config['lr'] )

# Step 5: Training the Model
logger = CSVLogger(save_dir=config['logger']['save_dir'], name=config['logger']['name'])

trainer = pl.Trainer(
    max_epochs=config['training']['max_epochs'],
    accelerator=config['training']['accelerator'],
    logger=logger,
    log_every_n_steps=config['training']['log_every_n_steps']
)

trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Function to handle dynamic versioning of the logs directory
def get_latest_version(logs_dir):
    versions = [d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]
    latest_version = max(versions, key=lambda x: int(x.split('_')[-1]))
    return latest_version

# Step 6: Loading the metrics CSV and Handling Epochs
logs_dir = os.path.join(config['logger']['save_dir'], config['logger']['name'])
latest_version = get_latest_version(logs_dir)
metrics_path = os.path.join(logs_dir, latest_version, "metrics.csv")
metrics = pd.read_csv(metrics_path)

# Adjust epoch index 
metrics['epoch'] = metrics['epoch'] + 1

# Create separate arrays for validation and training data by alternating rows
val_metrics = metrics.iloc[::2].reset_index(drop=True)  # even rows for validation
train_metrics = metrics.iloc[1::2].reset_index(drop=True)  # odd rows for training

# Step 7: Plotting MAE vs Epochs for both train and validation
plt.figure(figsize=tuple(config['plot']['fig_size']))
plt.plot(train_metrics["epoch"], train_metrics["train_MAE"], label="Train MAE", linewidth=2, marker='o')
plt.plot(val_metrics["epoch"], val_metrics["val_MAE"], label="Validation MAE", linewidth=2, marker='o')
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Mean Absolute Error (MAE)", fontsize=14)
plt.title("MAE vs Epochs", fontsize=16)
plt.legend()
plt.grid(True)
plt.savefig("MAE_vs_Epochs.png", dpi=config['plot']['dpi'])

# Step 8: Plotting Total Loss vs Epochs for both train and validation
plt.figure(figsize=tuple(config['plot']['fig_size']))
plt.plot(train_metrics["epoch"], train_metrics["train_Total_Loss"], label="Train Total Loss", linewidth=2, marker='o')
plt.plot(val_metrics["epoch"], val_metrics["val_Total_Loss"], label="Validation Total Loss", linewidth=2, marker='o')
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Total Loss", fontsize=14)
plt.title("Total Loss vs Epochs", fontsize=16)
plt.legend()
plt.grid(True)
plt.savefig("Total_Loss_vs_Epochs.png", dpi=config['plot']['dpi'])

plt.show()
