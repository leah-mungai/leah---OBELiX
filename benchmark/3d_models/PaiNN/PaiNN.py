# Import configuration
import yaml
with open('config_painn.yaml', 'r') as file:
    config = yaml.safe_load(file)

####### step 1: Set Random Seed

import torch
import numpy as np
import random

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True)
set_random_seed(config["seed"])  # Seed value from config.yaml


####### step 2: Clean up runtime files

import os

runtime_files = config["runtime_files"]  # List of runtime files from config.yaml

for file_path in runtime_files:
    if os.path.exists(file_path):
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        elif os.path.isdir(file_path):
            os.rmdir(file_path)
            print(f"Deleted directory: {file_path}")


####### step 3: Dataset Preprocessing

import warnings
from ase.io import read
import pandas as pd
import os
import numpy as np
from schnetpack.data import ASEAtomsData

warnings.filterwarnings('ignore')

file_prefix = config["file_prefix"]  # Prefix from config.yaml
updated_file_path = config["updated_file_path"]  # Path from config.yaml
updated_data = pd.read_csv(updated_file_path, dtype={'ID': str})

updated_data['ID'] = updated_data['ID'].str.replace(r'^="|"$', '', regex=True)

atoms_list, property_list, id_list, ionic_conductivity_list = [], [], [], []

for idx, row in updated_data.iterrows():
    cif_file_path = os.path.join(config["cif_file_dir"], f'{row["ID"]}.cif')  # Path from config.yaml
    
    try:
        atoms = read(cif_file_path, 
                     index=config['cif_read_options']['index'], 
                     store_tags=config['cif_read_options']['store_tags'], 
                     primitive_cell=config['cif_read_options']['primitive_cell'], 
                     subtrans_included=config['cif_read_options']['subtrans_included'], 
                     fractional_occupancies=config['cif_read_options']['fractional_occupancies'], 
                     reader=config['cif_read_options']['reader'])

        symbols = list(atoms.symbols)
        coords = list(atoms.get_positions())
        occupancies = [1] * len(symbols)  # Default occupancies to 1
        occ_info = atoms.info.get('occupancy')
        kinds = atoms.arrays.get('spacegroup_kinds')
        
        if occ_info is not None and kinds is not None:
            for i, kind in enumerate(kinds):
                occ_info_kind = occ_info.get(str(kind), {})
                symbol = symbols[i]
                if symbol not in occ_info_kind:
                    raise ValueError(f'Occupancy info for "{symbol}" not found')
                occupancies[i] = occ_info_kind.get(symbol, 1)

        atoms.set_pbc(True)

        ionic_conductivity = np.log10(row['Ionic conductivity (S cm-1)'])
        properties = {'ionic_conductivity': np.array([ionic_conductivity], dtype=np.float32)}
        property_list.append(properties)
        atoms_list.append(atoms)
        id_list.append(row['ID'])
        ionic_conductivity_list.append(ionic_conductivity)
    
    except Exception as e:
        print(f'Failed to process CIF file for {row["ID"]}: {e}')

db_path = config["db_path"]
if os.path.exists(db_path):
    os.remove(db_path)

new_dataset = ASEAtomsData.create(
    db_path,
    distance_unit='Ang',
    property_unit_dict={'ionic_conductivity': 'S/cm'}
)
new_dataset.add_systems(property_list, atoms_list)


####### step 4: Data Splitting and Loading

import schnetpack as spk
import schnetpack.transform as trn

custom_data = spk.data.AtomsDataModule(
    db_path,
    batch_size=config["batch_size"],
    distance_unit='Ang',
    num_train=config["num_train"],
    num_val=config["num_val"],
    num_test=config["num_test"],
    transforms=[
        trn.ASENeighborList(cutoff=config["cutoff"]),
        trn.RemoveOffsets("ionic_conductivity", remove_mean=config["remove_mean"], remove_atomrefs=config["remove_atomrefs"], is_extensive=False, propery_mean=torch.tensor([2.5])),
        trn.CastTo32()
    ],
    num_workers=config["num_workers"],
    pin_memory=config["pin_memory"]
)
custom_data.prepare_data()
custom_data.setup()

train_loader = custom_data.train_dataloader()

train_ionic_conductivity, train_ids = [], []
for batch in train_loader:
    for i in range(batch['ionic_conductivity'].shape[0]):
        index = batch['_idx'][i].item()
        train_ionic_conductivity.append(batch['ionic_conductivity'][i].item())
        train_ids.append(id_list[index])


####### step 5: Plotting Ionic Conductivities

import matplotlib.pyplot as plt

matched_original_conductivity, matched_transformed_conductivity = [], []
for i, train_id in enumerate(train_ids):
    if train_id in id_list:
        idx = id_list.index(train_id)
        matched_original_conductivity.append(ionic_conductivity_list[idx])
        matched_transformed_conductivity.append(train_ionic_conductivity[i])

plt.figure(figsize=(8, 6))
plt.scatter(matched_original_conductivity, matched_transformed_conductivity, color='b', label='Train Set', alpha=0.7)
plt.xlabel('Log-transformed Ionic Conductivity (Original)', fontsize=12)
plt.ylabel('Ionic Conductivity (Transformed)', fontsize=12)
plt.title('Original vs. Transformed Ionic Conductivity for Training Set', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(config["plot_save_path"], dpi=config["high_dpi"])
plt.show()


####### step 6: Model Definition

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.nn import GaussianRBF, CosineCutoff

cutoff = config["cutoff"]
n_atom_basis = config["n_atom_basis"]
n_interactions = config["n_interactions"]

pairwise_distance = spk.atomistic.PairwiseDistances()
radial_basis = GaussianRBF(n_rbf=20, cutoff=cutoff)

painn = spk.representation.PaiNN(
    n_atom_basis=n_atom_basis,
    n_interactions=n_interactions,
    radial_basis=radial_basis,
    cutoff_fn=CosineCutoff(cutoff),
    dropout_rate=0.1
)

pred_ionic_conductivity = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='ionic_conductivity', aggregation_mode="avg")

nnpot_painn = spk.model.NeuralNetworkPotential(
    representation=painn,
    input_modules=[pairwise_distance],
    output_modules=[pred_ionic_conductivity],
    postprocessors=[trn.CastTo64()]
)


####### step 7: Model Training and Logging

import torch
import torchmetrics
import pytorch_lightning as pl
import schnetpack as spk


output_ionic_conductivity = spk.task.ModelOutput(
    name='ionic_conductivity',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.0,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)


task_painn = spk.task.AtomisticTask(
    model=nnpot_painn,
    outputs=[output_ionic_conductivity],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": config["learning_rate"], "weight_decay": config["weight_decay"]}
)

logger_painn = pl.loggers.TensorBoardLogger(save_dir=config["logger_painn_dir"])

metrics_df = pd.DataFrame(columns=["Epoch", "Train Loss", "Val Loss", "Train MAE", "Val MAE"])


class MetricTracker(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        train_loss = trainer.callback_metrics.get("train_loss")
        val_loss = trainer.callback_metrics.get("val_loss")
        train_mae = trainer.callback_metrics.get("train_ionic_conductivity_MAE")
        val_mae = trainer.callback_metrics.get("val_ionic_conductivity_MAE")

        # Extract scalar values from tensors
        if train_loss is not None:
            train_loss = train_loss.item()
        if val_loss is not None:
            val_loss = val_loss.item()
        if train_mae is not None:
            train_mae = train_mae.item()
        if val_mae is not None:
            val_mae = val_mae.item()

        # Save values into pl_module for later access
        if hasattr(pl_module, 'train_losses'):
            pl_module.train_losses.append(train_loss)
            pl_module.val_losses.append(val_loss)
            pl_module.train_maes.append(train_mae)
            pl_module.val_maes.append(val_mae)
        else:
            pl_module.train_losses = [train_loss]
            pl_module.val_losses = [val_loss]
            pl_module.train_maes = [train_mae]
            pl_module.val_maes = [val_mae]

        print(f"Epoch {epoch + 1} - Train Loss: {train_loss}, Val Loss: {val_loss}, Train MAE: {train_mae}, Val MAE: {val_mae}")

        global metrics_df
        metrics_df = pd.concat([metrics_df, pd.DataFrame([{
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "Train MAE": train_mae,
            "Val MAE": val_mae
        }])], ignore_index=True)

        # Save the dataframe to a CSV file after each epoch
        metrics_df.to_csv("metrics_output.csv", index=False)


callbacks = [
    spk.train.ModelCheckpoint(
        model_path=config["model_checkpoint_path"],
        save_top_k=config["save_top_k"],
        monitor=config["monitor_metric"]
    ),
    MetricTracker(),
]

trainer_painn = pl.Trainer(
    callbacks=callbacks,
    logger=logger_painn,
    default_root_dir=config["painn_root_dir"],
    max_epochs=config["max_epochs"],
    deterministic=True,
    benchmark=False
)

trainer_painn.fit(task_painn, datamodule=custom_data)


####### step 8: Test MAE Evaluation

from sklearn.metrics import mean_absolute_error

def evaluate_model_on_test(model, test_loader, id_list):
    test_ids, actual_test_ionic_conductivities, predicted_test_ionic_conductivities = [], [], []
    
    # Put model in evaluation mode
    model.eval()
    
    # No need for gradients during evaluation
    with torch.no_grad():
        for batch in test_loader:
            actual_test_ionic_conductivities.extend(batch['ionic_conductivity'].cpu().numpy())
            predictions = model(batch)['ionic_conductivity'].cpu().detach().numpy()
            
            for i in range(len(predictions)):
                index = batch['_idx'][i].item()
                predicted_test_ionic_conductivities.append(predictions[i])
                test_ids.append(id_list[index])
    
    # Calculate MAE for test set
    mae_test = mean_absolute_error(actual_test_ionic_conductivities, predicted_test_ionic_conductivities)
    print(f"Test MAE: {mae_test}")
    
    # Save predictions to CSV
    df = pd.DataFrame({
        'ID': test_ids,
        'Actual Ionic Conductivity (log-transformed)': actual_test_ionic_conductivities,
        'Predicted Ionic Conductivity': predicted_test_ionic_conductivities
    })
    df.to_csv(config["test_pred_painn_csv"], index=False)
    print(f"Test predictions saved to {config['test_pred_painn_csv']}")
    
    return mae_test

# After training is complete, evaluate the model on the test set
test_loader = custom_data.test_dataloader()
test_mae = evaluate_model_on_test(task_painn, test_loader, id_list)
print(f"Final Test MAE: {test_mae}")


####### step 9: Save Predictions to CSV

import torch
import pandas as pd

def save_predictions_to_csv(model, dataloader, id_list, file_name):
    ids, actual_ionic_conductivities, predicted_ionic_conductivities = [], [], []
    for batch in dataloader:
        actual_ionic_conductivities.extend(batch['ionic_conductivity'].cpu().numpy())
        predictions = model(batch)['ionic_conductivity'].cpu().detach().numpy()
        for i in range(len(predictions)):
            index = batch['_idx'][i].item()
            predicted_ionic_conductivities.append(predictions[i])
            ids.append(id_list[index])
    df = pd.DataFrame({
        'ID': ids,
        'Actual Ionic Conductivity (log-transformed)': actual_ionic_conductivities,
        'Predicted Ionic Conductivity': predicted_ionic_conductivities
    })
    df.to_csv(file_name, index=False)
    print(f'Saved {file_name}')

# Save predictions for PaiNN model
save_predictions_to_csv(task_painn, train_loader, id_list, config["train_pred_painn_csv"])
save_predictions_to_csv(task_painn, custom_data.val_dataloader(), id_list, config["val_pred_painn_csv"])
save_predictions_to_csv(task_painn, custom_data.test_dataloader(), id_list, config["test_pred_painn_csv"])


####### step 10: Actual vs Predicted Plot

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

def plot_actual_vs_predicted(csv_file, model_name, save_path, dpi=config["high_dpi"]):
    df = pd.read_csv(csv_file)
    r2 = r2_score(df['Actual Ionic Conductivity (log-transformed)'], df['Predicted Ionic Conductivity'])
    
    plt.figure(figsize=(6, 6))
    plt.scatter(df['Actual Ionic Conductivity (log-transformed)'], df['Predicted Ionic Conductivity'], alpha=0.5)
    
    min_val = min(df['Actual Ionic Conductivity (log-transformed)'].min(), df['Predicted Ionic Conductivity'].min())
    max_val = max(df['Actual Ionic Conductivity (log-transformed)'].max(), df['Predicted Ionic Conductivity'].max())
    
    # Diagonal line indicating perfect prediction
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Ionic Conductivity (log-transformed)')
    plt.ylabel('Predicted Ionic Conductivity')
    plt.title(f'Actual vs Predicted Ionic Conductivity ({model_name})\nR² = {r2:.4f}')
    
    # Save the plot as PNG with high resolution
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()
    
    plt.close()
    print(f"Plot saved as: {save_path}")

# Save and plot predictions for PaiNN model
plot_actual_vs_predicted(
    config["test_pred_painn_csv"], 
    'PaiNN - Test set', 
    'painn_actual_vs_predicted.png',  
    dpi=config["high_dpi"]
)


####### step 11: Loss & MAE

import matplotlib.pyplot as plt

# Access the trained models to retrieve the logged metrics
trained_model_painn = trainer_painn.model

plt.figure()

plt.plot(range(1, len(trained_model_painn.train_losses) + 1), trained_model_painn.train_losses, 
         label='Train Loss - PaiNN', color='blue', linestyle='--')
plt.plot(range(1, len(trained_model_painn.val_losses) + 1), trained_model_painn.val_losses, 
         label='Validation Loss - PaiNN', color='orange', linestyle='--')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs - PaiNN')
plt.legend()

plt.savefig('loss_painn.png', dpi=config["high_dpi"])

plt.show()


plt.figure()


plt.plot(range(1, len(trained_model_painn.train_maes) + 1), trained_model_painn.train_maes, 
         label='Train MAE - PaiNN', color='blue', linestyle='--')
plt.plot(range(1, len(trained_model_painn.val_maes) + 1), trained_model_painn.val_maes, 
         label='Validation MAE - PaiNN', color='orange', linestyle='--')

plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('MAE vs. Epochs - PaiNN')
plt.legend()

plt.savefig('mae_painn.png', dpi=config["high_dpi"])

plt.show()
