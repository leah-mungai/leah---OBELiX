from __future__ import annotations
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from tqdm import tqdm
from functools import partial
import lightning as pl
from pytorch_lightning.loggers import CSVLogger
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, collate_fn_graph
from matgl.models import M3GNet, SO3Net
from matgl.utils.training import ModelLightningModule
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import gc
import sys
import itertools
import random

def load_config(file: str = "config.yaml"):
    with open(file, "r") as stream:
        config = yaml.safe_load(stream)
    return config

def load_dataset(csv_file, cif_dir, dataset_type, conductivity_column):
    data = pd.read_csv(csv_file)
    data['ID'] = data['ID'].str.replace(r'^="|"$', '', regex=True)
    structures = []
    ids = []
    print(f"\nLoading {dataset_type} dataset...")
    for i, row in tqdm(data.iterrows(), total=len(data)):
        try:
            struct = Structure.from_file(os.path.join(cif_dir, f'{row["ID"]}.cif'))
            structures.append(struct)
            ids.append(row["ID"])
        except OSError as e:
            print(f"Error loading structure for ID {row['ID']}: {e}")
    ionic_conductivities = np.log10(data[conductivity_column].values)
    return structures, ionic_conductivities, ids

def MGLDataLoaderNoVal(train_data, collate_fn, **kwargs):
    """Custom DataLoader without validation."""
    return DataLoader(train_data, shuffle=True, collate_fn=collate_fn, **kwargs)

def get_model(config, model_type, train_structures, params):
    """Instantiate the appropriate model (M3GNet or SO3Net) based on config."""
    elem_list = get_element_list(train_structures)
    if model_type == "m3gnet":
        model = M3GNet(
            element_types=elem_list,
            is_intensive=config['m3gnet']['is_intensive'],
            readout_type=params['readout_type'],
            nblocks=params['nblocks'],
            threebody_cutoff=params['threebody_cutoff'],
            dim_node_embedding=params['dim_node_embedding'],
            dim_edge_embedding=params['dim_edge_embedding'],
            units=params['units'],
        )
    elif model_type == "so3net":
        model = SO3Net(
            element_types=elem_list,
            is_intensive=config['so3net']['is_intensive'],
            readout_type=params['readout_type'],
            nmax=params['nmax'],
            lmax=params['lmax'],
            nblocks=params['nblocks'],
            target_property=config['so3net']['target_property'],
            dim_node_embedding=params['dim_node_embedding'],
            units=params['units'],
        )
    return model
    
def get_latest_version(log_dir):
    """Helper function to get the latest version folder."""
    versions = [d for d in os.listdir(log_dir) if d.startswith("version_")]
    if versions:
        return max(versions, key=lambda x: int(x.split('_')[-1]))
    return "version_0"

def run_cross_validation(config, model_type, params, train_dataset, train_structures):
    """Run k-fold cross-validation with random hyperparameter selection."""
    kf = KFold(n_splits=config['split']['k_folds'], 
    shuffle=config['split']['shuffle'], 
    random_state=config['split']['random_state'])
    
    mae_train_folds = []
    mae_val_folds = []
    loss_train_folds = []
    loss_val_folds = []
    epochs_range = range(1, config['training']['max_epochs'] + 1)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset), start=1):
        print(f"\nFold {fold}/{config['split']['k_folds']}")
        train_data = Subset(train_dataset, train_idx)
        val_data = Subset(train_dataset, val_idx)

        my_collate_fn = partial(collate_fn_graph, include_line_graph=True)
        train_loader = MGLDataLoaderNoVal(
            train_data=train_data, 
            collate_fn=my_collate_fn, 
            batch_size=config['dataloader']['batch_size'], 
            num_workers=config['dataloader']['num_workers']
        )
        val_loader = MGLDataLoaderNoVal(
            train_data=val_data, 
            collate_fn=my_collate_fn, 
            batch_size=config['dataloader']['batch_size'], 
            num_workers=config['dataloader']['num_workers']
        )

        model = get_model(config, model_type, train_structures, params)
        lit_module = ModelLightningModule(model=model, include_line_graph=True, lr=config['training']['lr'])


        log_dir = os.path.join(config['logger']['save_dir'], f"{model_type}_training_fold_{fold}_params_{params}")
        logger = CSVLogger(save_dir=log_dir, name="", version=None)

        trainer = pl.Trainer(
            max_epochs=config['training']['max_epochs'],
            accelerator=config['training']['accelerator'],
            logger=logger,
        )

        try:
            print(f"Starting training for fold {fold}")
            trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        except torch.cuda.OutOfMemoryError as e:
            print(f"Out of memory error in fold {fold}, params {params}: {e}")
            continue  # Skip to the next fold.......
        except RuntimeError as e:
            print(f"Error encountered during training in fold {fold}: {e}")
            continue  # Skip to the next fold.......

        #### Clear memory and caches between folds ####
        torch.cuda.empty_cache()
        gc.collect()

        # Fetch only the latest version folder
        version_dir = get_latest_version(log_dir)
        metrics_path = os.path.join(log_dir, version_dir, "metrics.csv")
        
        if not os.path.exists(metrics_path):
            print(f"Metrics file for fold {fold} not found at {metrics_path}")
            continue
        
        metrics = pd.read_csv(metrics_path)
        val_metrics = metrics.iloc[::2].reset_index(drop=True)
        train_metrics = metrics.iloc[1::2].reset_index(drop=True)

        mae_train_folds.append(train_metrics["train_MAE"].values)
        mae_val_folds.append(val_metrics["val_MAE"].values)
        loss_train_folds.append(train_metrics["train_Total_Loss"].values)
        loss_val_folds.append(val_metrics["val_Total_Loss"].values)

    # Average and standard deviation for each epoch
    mae_train_avg = np.mean(mae_train_folds, axis=0)
    mae_train_std = np.std(mae_train_folds, axis=0)
    mae_val_avg = np.mean(mae_val_folds, axis=0)
    mae_val_std = np.std(mae_val_folds, axis=0)
    loss_train_avg = np.mean(loss_train_folds, axis=0)
    loss_train_std = np.std(loss_train_folds, axis=0)
    loss_val_avg = np.mean(loss_val_folds, axis=0)
    loss_val_std = np.std(loss_val_folds, axis=0)

    # Plot average loss vs. epochs
    plt.figure(figsize=config['plot']['fig_size'])
    plt.plot(epochs_range, loss_train_avg, label="Avg Train Loss", linewidth=2, marker='o')
    plt.plot(epochs_range, loss_val_avg, label="Avg Validation Loss", linewidth=2, marker='o')
    plt.fill_between(epochs_range, loss_train_avg - loss_train_std, loss_train_avg + loss_train_std, alpha=0.3)
    plt.fill_between(epochs_range, loss_val_avg - loss_val_std, loss_val_avg + loss_val_std, alpha=0.3)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Total Loss", fontsize=14)
    plt.title("Average Loss vs Epochs Across Folds", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Average_Loss_vs_Epochs_params_{params}.png", dpi=config['plot']['dpi'])

    # Plot average MAE vs. epochs
    plt.figure(figsize=config['plot']['fig_size'])
    plt.plot(epochs_range, mae_train_avg, label="Avg Train MAE", linewidth=2, marker='o')
    plt.plot(epochs_range, mae_val_avg, label="Avg Validation MAE", linewidth=2, marker='o')
    plt.fill_between(epochs_range, mae_train_avg - mae_train_std, mae_train_avg + mae_train_std, alpha=0.3)
    plt.fill_between(epochs_range, mae_val_avg - mae_val_std, mae_val_avg + mae_val_std, alpha=0.3)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("MAE", fontsize=14)
    plt.title("Average MAE vs Epochs Across Folds", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Average_MAE_vs_Epochs_params_{params}.png", dpi=config['plot']['dpi'])

    # Save results to CSV
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
    results_df.to_csv(f"cross_validation_results_params_{params}.csv", index=False)

    # Print average and standard deviation
    print(f"Avg Train MAE: {np.mean(mae_train_avg):.4f} ± {np.std(mae_train_avg):.4f}")
    print(f"Avg Validation MAE: {np.mean(mae_val_avg):.4f} ± {np.std(mae_val_avg):.4f}")
    print(f"Avg Train Loss: {np.mean(loss_train_avg):.4f} ± {np.std(loss_train_avg):.4f}")
    print(f"Avg Validation Loss: {np.mean(loss_val_avg):.4f} ± {np.std(loss_val_avg):.4f}")

    avg_val_mae = np.mean(mae_val_avg)
    return avg_val_mae

def run_hyperparameter_search(config, model_type, train_dataset, train_structures):
    """Run hyperparameter search with random selection of cases."""
    hyperparameters = generate_random_hyperparams(config, model_type)
    best_val_mae = float('inf')
    best_hyperparameters = None
    all_mae_values = []

    for params in hyperparameters:
        print(f"Running cross-validation for params={params}")
        avg_val_mae = run_cross_validation(config, model_type, params, train_dataset, train_structures)
        print(f"Avg Validation MAE for params={params}: {avg_val_mae:.4f}")
        
        all_mae_values.append({'params': params, 'avg_val_mae': avg_val_mae})

        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_hyperparameters = params

    plot_hyperparameter_comparison(all_mae_values)

    return best_hyperparameters, best_val_mae

def generate_random_hyperparams(config, model_type):
    """Generate random hyperparameter combinations based on the config file."""
    if model_type == "m3gnet":
        param_grid = {
            'readout_type': config['m3gnet']['readout_type'],
            'max_n': config['m3gnet']['max_n'],
            'max_l': config['m3gnet']['max_l'],
            'nblocks': config['m3gnet']['nblocks'],
            'dim_node_embedding': config['m3gnet']['dim_node_embedding'],
            'dim_edge_embedding': config['m3gnet']['dim_edge_embedding'],
            'units': config['m3gnet']['units'],
            'threebody_cutoff': config['m3gnet']['threebody_cutoff'],
            'cutoff': config['m3gnet']['cutoff']
        }
    else:
        param_grid = {
            'readout_type': config['so3net']['readout_type'],
            'nmax': config['so3net']['nmax'],
            'lmax': config['so3net']['lmax'],
            'nblocks': config['so3net']['nblocks'],
            'dim_node_embedding': config['so3net']['dim_node_embedding'],
            'units': config['so3net']['units'],
            'cutoff': config['so3net']['cutoff'],
            'nlayers_readout': config['so3net']['nlayers_readout']
        }

    all_combinations = list(itertools.product(*param_grid.values()))
    random.shuffle(all_combinations)
    num_cases = config['hyperparameters']['num_cases']
    selected_combinations = all_combinations[:num_cases]  # Select random combinations based on config
    return [dict(zip(param_grid.keys(), comb)) for comb in selected_combinations]

def plot_hyperparameter_comparison(all_mae_values):
    """Creative plot: Validation MAE for Different Hyperparameter Cases."""
    mae_list = [item['avg_val_mae'] for item in all_mae_values]
    params_list = [str(item['params']) for item in all_mae_values]

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(range(len(mae_list)), mae_list, c=mae_list, cmap='coolwarm', s=100)
    plt.colorbar(scatter, label='Validation MAE')
    ax.set_xticks(range(len(params_list)))
    ax.set_xticklabels(params_list, rotation=90, fontsize=8)
    ax.set_xlabel('Hyperparameter Combination')
    ax.set_ylabel('Validation MAE')
    ax.set_title('Validation MAE for Different Hyperparameter Combinations')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Hyperparameter_Comparison_MAE.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    config = load_config()

    # Load training and test datasets
    train_structures, train_ionic_conductivities, train_ids = load_dataset(
    config['data']['csv_train_file'], 
    config['data']['cif_directory'], 
    "training", 
    config['data']['ionic_conductivity_column'])
    
    
    test_structures, test_ionic_conductivities, test_ids = load_dataset(
    config['data']['csv_test_file'], 
    config['data']['cif_directory'], 
    "testing", 
    config['data']['ionic_conductivity_column'])

    # Convert datasets to MGLDataset format
    converter = Structure2Graph(
        element_types=get_element_list(train_structures), 
        cutoff=config['graph']['cutoff']
    )
    
    train_dataset = MGLDataset(
        threebody_cutoff=config['graph']['threebody_cutoff'], 
        structures=train_structures, 
        converter=converter, 
        labels={"ionic_conductivity": train_ionic_conductivities}, 
        include_line_graph=True, 
        raw_dir=config['output']['raw_data_dir_train']
    )
    
    test_dataset = MGLDataset(
        threebody_cutoff=config['graph']['threebody_cutoff'], 
        structures=test_structures, 
        converter=converter, 
        labels={"ionic_conductivity": test_ionic_conductivities}, 
        include_line_graph=True, 
        raw_dir=config['output']['raw_data_dir_test']
    )

    # Pass model type from the command line
    model_type = sys.argv[1] if len(sys.argv) > 1 else "m3gnet"

    # Run hyperparameter search
    best_hyperparameters, best_val_mae = run_hyperparameter_search(config, model_type, train_dataset, train_structures)

    print(f"Best Hyperparameters for {model_type}: {best_hyperparameters}")

    # Load the best model using the best hyperparameters
    best_model = get_model(config, model_type, train_structures, best_hyperparameters)
    best_lit_module = ModelLightningModule(model=best_model, include_line_graph=True, lr=config['training']['lr'])

    # Prepare the test data loader
    my_collate_fn = partial(collate_fn_graph, include_line_graph=True)
    test_loader = MGLDataLoaderNoVal(
        train_data=test_dataset, 
        collate_fn=my_collate_fn, 
        batch_size=config['dataloader']['batch_size'], 
        num_workers=config['dataloader']['num_workers']
    )

    # Test the model
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['training']['accelerator'],
        logger=False  # No logger for final test evaluation
    )

    print("Evaluating on the test set with the best hyperparameters...")
    test_results = trainer.test(model=best_lit_module, dataloaders=test_loader)
    test_mae = test_results[0]["test_MAE"] if "test_MAE" in test_results[0] else None

    # Print the results
    print(f"Average Validation MAE: {best_val_mae:.4f} (best hyperparameters)")
    print(f"Final Test MAE: {test_mae:.4f}")
