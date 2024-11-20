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

# Load the YAML config file
def load_config(file: str = "config.yaml"):
    """Load configuration from a YAML file."""
    with open(file, "r") as stream:
        config = yaml.safe_load(stream)
    return config

# Load the dataset from CSV and CIF files
def load_dataset(csv_file, cif_dir, dataset_type, conductivity_column):
    """Load datasets from a CSV file and corresponding CIF files."""
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

# Custom DataLoader without validation
def MGLDataLoaderNoVal(train_data, collate_fn, shuffle=False, **kwargs):
    """Custom DataLoader without validation."""
    return DataLoader(train_data, shuffle=shuffle, collate_fn=collate_fn, **kwargs)


# Create the model (M3GNet or SO3Net) with optional pretrained model loading
def get_model(config, model_type, train_structures, load_pretrained=False):
    elem_list = get_element_list(train_structures)
    
    # Initialize the model based on the model_type
    if model_type == "m3gnet":
        model = M3GNet(
            element_types=elem_list,
            is_intensive=config['m3gnet']['is_intensive'],
            readout_type=config['m3gnet']['readout_type'],
            nblocks=config['m3gnet']['nblocks'],
            threebody_cutoff=config['m3gnet']['threebody_cutoff'],
            dim_node_embedding=config['m3gnet']['dim_node_embedding'],
            dim_edge_embedding=config['m3gnet']['dim_edge_embedding'],
            units=config['m3gnet']['units'],
        )
    elif model_type == "so3net":
        model = SO3Net(
            element_types=elem_list,
            is_intensive=config['so3net']['is_intensive'],
            readout_type=config['so3net']['readout_type'],
            lmax=config['so3net']['lmax'],
            nmax=config['so3net']['nmax'],
            target_property=config['so3net']['target_property'],
        )
    
    # Load pretrained weights if specified
    if load_pretrained:
        pretrained_dir = config.get('pretrained_model_dir', None)
        if pretrained_dir:
            model_path = os.path.join(pretrained_dir, 'model.pt')
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  
            
            # Extract the 'model' key from the checkpoint
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Load the state dict into the model
            model.load_state_dict(state_dict, strict=False)
            print(f"Pretrained model loaded from {model_path}")
    
    return model


# Get the latest version folder
def get_latest_version(log_dir):
    """Helper function to get the latest version folder."""
    versions = [d for d in os.listdir(log_dir) if d.startswith("version_")]
    if versions:
        return max(versions, key=lambda x: int(x.split('_')[-1]))
    return "version_0"

# Set the random seed for reproducibility
def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Get predictions vs actual values
def get_pred_vs_targets(model, dataloader):
    actual_ionic_conductivities, predicted_ionic_conductivities = [], []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(f"Processing batch {i}...")

            try:
                # Unpack the batch components
                graph_data = batch[0]
                lattice_data = batch[1]
                line_graph_data = batch[2]
                node_features = batch[3]
                labels = batch[4]

                # Skip invalid batches
                if labels is None or labels.numel() == 0:
                    print(f"Skipping batch {i}: No labels found.")
                    continue

                # Make predictions
                predictions = model(graph_data, lattice_data, line_graph_data, node_features)

                # Ensure predictions are valid
                pred_ionic_conductivity = (
                    predictions.get("ionic_conductivity", None) if isinstance(predictions, dict) else predictions
                )
                if pred_ionic_conductivity is None or pred_ionic_conductivity.numel() == 0:
                    print(f"Skipping batch {i}: No predictions found.")
                    continue

                # Convert predictions and labels to numpy arrays
                pred_np = pred_ionic_conductivity.cpu().numpy()
                labels_np = labels.cpu().numpy()

                # Ensure correct dimensions for concatenation
                if pred_np.ndim == 0:  # If scalar, expand dims
                    pred_np = np.expand_dims(pred_np, axis=0)
                if labels_np.ndim == 0:  # If scalar, expand dims
                    labels_np = np.expand_dims(labels_np, axis=0)

                # Append to results
                predicted_ionic_conductivities.append(pred_np)
                actual_ionic_conductivities.append(labels_np)

            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue

    # Ensure valid outputs before concatenation
    if len(predicted_ionic_conductivities) == 0 or len(actual_ionic_conductivities) == 0:
        print("No valid predictions or targets. Returning empty arrays.")
        return np.array([]), np.array([])

    # Concatenate results
    try:
        predictions_array = np.concatenate(predicted_ionic_conductivities, axis=0)
        actuals_array = np.concatenate(actual_ionic_conductivities, axis=0)
    except ValueError as e:
        print(f"Error concatenating arrays: {e}")
        print(f"Predicted shapes: {[arr.shape for arr in predicted_ionic_conductivities]}")
        print(f"Actual shapes: {[arr.shape for arr in actual_ionic_conductivities]}")
        return np.array([]), np.array([])

    return predictions_array, actuals_array


# Plot actual vs predicted values for training and validation
def plot_actual_vs_predicted(pred_train, target_train, pred_val, target_val, model_name, save_path, config):
    """Plot predicted vs actual values for train and validation."""
    if len(pred_train) == 0 or len(pred_val) == 0 or len(target_train) == 0 or len(target_val) == 0:
        print("No valid data to plot.")
        return

    plt.figure(figsize=tuple(config['plot']['fig_size']), dpi=config['plot']['dpi'])
    plt.scatter(target_train, pred_train, label='Train', s=10)
    plt.scatter(target_val, pred_val, label='Validation', s=10)

    min_val = min(np.min(target_train), np.min(target_val))
    max_val = max(np.max(target_train), np.max(target_val))

    # Diagonal line for perfect prediction
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')


    plt.xlabel('Actual Ionic Conductivity (log-transformed)')
    plt.ylabel('Predicted Ionic Conductivity')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=config['plot']['dpi'])
    plt.show()

# Run cross-validation with proper file naming
def run_cross_validation(config, model_type, max_epochs, lr, batch_size, train_dataset, train_structures):
    """Run k-fold cross-validation."""
    kf = KFold(n_splits=config['split']['k_folds'], shuffle=config['split']['shuffle'], random_state=config['split']['random_state'])
   
    mae_train_folds = []
    mae_val_folds = []
    loss_train_folds = []
    loss_val_folds = []
    epochs_range = range(1, max_epochs + 1)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset), start=1):
        print(f"\nFold {fold}/{config['split']['k_folds']}")

        train_data = Subset(train_dataset, train_idx)
        val_data = Subset(train_dataset, val_idx)


        my_collate_fn = partial(collate_fn_graph, include_line_graph=True)
        train_loader = MGLDataLoaderNoVal(
            train_data=train_data, 
            collate_fn=my_collate_fn,
            shuffle=True,            
            batch_size=batch_size,  
            num_workers=config['dataloader']['num_workers']
        )
        val_loader = MGLDataLoaderNoVal(
            train_data=val_data, 
            shuffle=False,
            collate_fn=my_collate_fn, 
            batch_size=batch_size,  
            num_workers=config['dataloader']['num_workers']
        )

        model = get_model(config, model_type, train_structures)
        lit_module = ModelLightningModule(model=model, include_line_graph=True, lr=lr)

        # Set log directory dynamically based on fold and hyperparameters
        log_dir = os.path.join(config['logger']['save_dir'], f"{model_type}_fold_{fold}_lr_{lr}_epochs_{max_epochs}")
        logger = CSVLogger(save_dir=log_dir, name="", version=None)

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=config['training']['accelerator'],
            logger=logger,
        )

        try:
            print(f"Starting training for fold {fold}")
            trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        except RuntimeError as e:
            print(f"Error encountered during training in fold {fold}: {e}")
            break

        # Clear memory and caches between folds
        torch.cuda.empty_cache()
        gc.collect()

       # Fetch the latest version folder
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

        # Calculate and plot actual vs predicted for this fold
        pred_train, target_train = get_pred_vs_targets(trainer.model, train_loader)
        pred_val, target_val = get_pred_vs_targets(trainer.model, val_loader)


        if model_type == "m3gnet":
            file_suffix = (
                f"_m3gnet_rt_{config['m3gnet']['readout_type']}_nb_{config['m3gnet']['nblocks']}_"
                f"dimnode_{config['m3gnet']['dim_node_embedding']}_dimedge_{config['m3gnet']['dim_edge_embedding']}_"
                f"units_{config['m3gnet']['units']}_cutoff_{config['m3gnet']['cutoff']}_epochs_{max_epochs}_lr_{lr}"
            )
        elif model_type == "so3net":
            file_suffix = (
                f"_so3net_rt_{config['so3net']['readout_type']}_nb_{config['so3net']['nblocks']}_"
                f"dimnode_{config['so3net']['dim_node_embedding']}_units_{config['so3net']['units']}_"
                f"nmax_{config['so3net']['nmax']}_lmax_{config['so3net']['lmax']}_cutoff_{config['so3net']['cutoff']}_"
                f"epochs_{max_epochs}_lr_{lr}"
            )

        plot_filename = f"pred_vs_actual_fold_{fold}{file_suffix}.png"

        if pred_train.size > 0 and pred_val.size > 0:
            plot_actual_vs_predicted(
                pred_train, target_train, pred_val, target_val,
                f"Fold {fold} (lr={lr}, epochs={max_epochs})",
                plot_filename,  
                config
            )
        else:
            print(f"Skipping plotting for fold {fold} due to empty predictions or targets.")
              
        
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
    plt.figure(figsize=tuple(config['plot']['fig_size']))
    plt.plot(epochs_range, loss_train_avg, label="Avg Train Loss", linewidth=2, marker='o')
    plt.plot(epochs_range, loss_val_avg, label="Avg Validation Loss", linewidth=2, marker='o')
    plt.fill_between(epochs_range, loss_train_avg - loss_train_std, loss_train_avg + loss_train_std, alpha=0.3)
    plt.fill_between(epochs_range, loss_val_avg - loss_val_std, loss_val_avg + loss_val_std, alpha=0.3)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Total Loss", fontsize=14)
    plt.title(f"Average Loss vs Epochs (lr={lr}, epochs={max_epochs})", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Average_Loss_vs_Epochs{file_suffix}.png", dpi=config['plot']['dpi'])

    # Plot Average Validation MAE vs. Epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, mae_val_avg, label="Avg Validation MAE", linewidth=2, marker='o')
    plt.fill_between(epochs_range, mae_val_avg - mae_val_std, mae_val_avg + mae_val_std, alpha=0.3)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Validation MAE", fontsize=14)
    plt.title(f"Average Validation MAE vs Epochs (lr={lr}, epochs={max_epochs})", fontsize=16)
    plt.legend()
    plt.grid(True)

    # Save Average Validation MAE vs Epochs plot with dynamic name
    plt.savefig(f"Average_Val_MAE_vs_Epochs{file_suffix}.png", dpi=config['plot']['dpi'])


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
    results_df.to_csv(f"cross_validation_results{file_suffix}.csv", index=False)



    print(f"Avg Train MAE: {np.mean(mae_train_avg):.4f} ± {np.std(mae_train_avg):.4f}")
    print(f"Avg Validation MAE: {np.mean(mae_val_avg):.4f} ± {np.std(mae_val_avg):.4f}")
    print(f"Avg Train Loss: {np.mean(loss_train_avg):.4f} ± {np.std(loss_train_avg):.4f}")
    print(f"Avg Validation Loss: {np.mean(loss_val_avg):.4f} ± {np.std(loss_val_avg):.4f}")

    avg_val_mae = np.mean(mae_val_avg)
    return avg_val_mae

# Search for the best hyperparameter combination
def run_hyperparameter_search(config, model_type, train_dataset, train_structures):
    """Search for the best hyperparameter combination and return best hyperparameters and validation MAE."""
    if model_type == "m3gnet":
        hyperparameters = itertools.product(
            config['m3gnet']['readout_type'],
            config['m3gnet']['nblocks'],
            config['m3gnet']['dim_node_embedding'],
            config['m3gnet']['dim_edge_embedding'],
            config['m3gnet']['units'],
            config['m3gnet']['threebody_cutoff'],
            config['m3gnet']['cutoff'],
            config['hyperparameters']['batch_size'],
            config['hyperparameters']['lr'],
            config['hyperparameters']['max_epochs']
        )
    elif model_type == "so3net":
        hyperparameters = itertools.product(
            config['so3net']['readout_type'],
            config['so3net']['nblocks'],
            config['so3net']['dim_node_embedding'],
            config['so3net']['units'],
            config['so3net']['nmax'],
            config['so3net']['lmax'],
            config['so3net']['cutoff'],
            config['hyperparameters']['batch_size'],
            config['hyperparameters']['lr'],
            config['hyperparameters']['max_epochs']
        )

    best_val_mae = float('inf')
    best_hyperparameters = None
    all_mae_values = []

    # Shuffle and pick random combinations based on num_cases
    random_hyperparameters = random.sample(list(hyperparameters), config['hyperparameters']['num_cases'])

    for params in random_hyperparameters:
        if model_type == "m3gnet":
            readout_type, nblocks, dim_node_embedding, dim_edge_embedding, units, threebody_cutoff, cutoff, batch_size, lr, max_epochs = params
            print(f"Running cross-validation with: readout_type={readout_type}, nblocks={nblocks}, dim_node_embedding={dim_node_embedding}, "
                  f"dim_edge_embedding={dim_edge_embedding}, units={units}, threebody_cutoff={threebody_cutoff}, cutoff={cutoff}, "
                  f"batch_size={batch_size}, lr={lr}, max_epochs={max_epochs}")
            
            # Update the config with current hyperparameters
            config['m3gnet']['readout_type'] = readout_type
            config['m3gnet']['nblocks'] = nblocks
            config['m3gnet']['dim_node_embedding'] = dim_node_embedding
            config['m3gnet']['dim_edge_embedding'] = dim_edge_embedding
            config['m3gnet']['units'] = units
            config['m3gnet']['threebody_cutoff'] = threebody_cutoff
            config['m3gnet']['cutoff'] = cutoff
        
        elif model_type == "so3net":
            readout_type, nblocks, dim_node_embedding, units, nmax, lmax, cutoff, batch_size, lr, max_epochs = params
            print(f"Running cross-validation with: readout_type={readout_type}, nblocks={nblocks}, dim_node_embedding={dim_node_embedding}, "
                  f"units={units}, nmax={nmax}, lmax={lmax}, cutoff={cutoff}, batch_size={batch_size}, lr={lr}, max_epochs={max_epochs}")
            
            # Update the config with current hyperparameters
            config['so3net']['readout_type'] = readout_type
            config['so3net']['nblocks'] = nblocks
            config['so3net']['dim_node_embedding'] = dim_node_embedding
            config['so3net']['units'] = units
            config['so3net']['nmax'] = nmax
            config['so3net']['lmax'] = lmax
            config['so3net']['cutoff'] = cutoff

        avg_val_mae = run_cross_validation(config, model_type, max_epochs, lr, batch_size, train_dataset, train_structures)
        print(f"Avg Validation MAE for current config: {avg_val_mae:.4f}")

        # Add hyperparameters and results to all_mae_values
        all_mae_values.append({
            'readout_type': readout_type, 
            'nblocks': nblocks,
            'dim_node_embedding': dim_node_embedding, 
            'dim_edge_embedding': dim_edge_embedding if model_type == "m3gnet" else None,
            'units': units, 
            'threebody_cutoff': threebody_cutoff if model_type == "m3gnet" else None,
            'cutoff': cutoff, 
            'batch_size': batch_size,
            'lr': lr,
            'max_epochs': max_epochs,
            'avg_val_mae': avg_val_mae
        })

        # Update best hyperparameters
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_hyperparameters = {
                'readout_type': readout_type, 
                'nblocks': nblocks,
                'dim_node_embedding': dim_node_embedding, 
                'dim_edge_embedding': dim_edge_embedding if model_type == "m3gnet" else None,
                'units': units, 
                'threebody_cutoff': threebody_cutoff if model_type == "m3gnet" else None,
                'cutoff': cutoff,
                'batch_size': batch_size,
                'lr': lr,
                'max_epochs': max_epochs
            }


    # Plot the comparison of hyperparameter combinations
    plot_hyperparameter_comparison(all_mae_values, model_type)

    print(f"Best Hyperparameters for {model_type}: {best_hyperparameters}")
    print(f"Lowest Validation MAE: {best_val_mae:.4f}")
    
    # Return both best hyperparameters and best validation MAE
    return best_hyperparameters, best_val_mae



def plot_hyperparameter_comparison(all_mae_values, model_type):
    """Creative plot: Validation MAE for Different Hyperparameter Cases."""
    mae_list = [item['avg_val_mae'] for item in all_mae_values]

    # Dynamically set the parameter string based on the model type
    if model_type == "m3gnet":
        params_list = [
            f"rt: {item['readout_type']}, nb: {item['nblocks']}, dim_node: {item['dim_node_embedding']}, dim_edge: {item['dim_edge_embedding']}, units: {item['units']}, cutoff: {item['cutoff']}"
            for item in all_mae_values
        ]
    elif model_type == "so3net":
        params_list = [
            f"rt: {item['readout_type']}, nb: {item['nblocks']}, dim_node: {item['dim_node_embedding']}, units: {item['units']}, nmax: {item['nmax']}, lmax: {item['lmax']}, cutoff: {item['cutoff']}"
            for item in all_mae_values
        ]

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(range(len(mae_list)), mae_list, c=mae_list, cmap='coolwarm', s=100)
    plt.colorbar(scatter, label='Validation MAE')

    # Set the parameter combinations as labels on the x-axis
    ax.set_xticks(range(len(params_list)))
    ax.set_xticklabels(params_list, rotation=90, fontsize=8)

    ax.set_xlabel('Hyperparameter Combination')
    ax.set_ylabel('Validation MAE')
    ax.set_title(f'Validation MAE for Different Hyperparameter Combinations ({model_type})')
    
    plt.grid(True)
    plt.tight_layout()

    # Save the plot as a high-resolution image
    plt.savefig(f"Hyperparameter_Comparison_MAE_{model_type}.png", dpi=300)
    plt.show()


# Retrain the model on the full dataset using the best hyperparameters
def retrain_on_full_dataset(config, model_type, best_hyperparameters, train_dataset, train_structures, load_pretrained=False):
    """Retrain the model on the full training dataset using the best hyperparameters and optionally load pretrained weights."""
    my_collate_fn = partial(collate_fn_graph, include_line_graph=True)
    full_train_loader = MGLDataLoaderNoVal(
        train_data=train_dataset, 
        collate_fn=my_collate_fn, 
        batch_size=best_hyperparameters['batch_size'],  # Use best hyperparameter batch size
        num_workers=config['dataloader']['num_workers']
    )

    # Instantiate the model with the best hyperparameters
    if model_type == "m3gnet":
        config['m3gnet'].update(best_hyperparameters)  # Update M3GNet configuration with best hyperparameters
    elif model_type == "so3net":
        config['so3net'].update(best_hyperparameters)  # Update SO3Net configuration with best hyperparameters

    model = get_model(config, model_type, train_structures, load_pretrained=load_pretrained)
    
    lit_module = ModelLightningModule(
        model=model, 
        include_line_graph=True, 
        lr=best_hyperparameters['lr']  # Use best hyperparameter learning rate
    )
    
    logger = CSVLogger(save_dir=config['logger']['save_dir'], name=f"{model_type}_full_training")

    trainer = pl.Trainer(
        max_epochs=best_hyperparameters['max_epochs'],  # Use best hyperparameter max_epochs
        accelerator=config['training']['accelerator'], 
        logger=logger
    )

    # Retrain the model on the full training set
    trainer.fit(model=lit_module, train_dataloaders=full_train_loader)
    return lit_module


# Test the model with test set and best hyperparameters
def run_test_evaluation(config, model_type, lit_module, test_dataset, train_structures):
    """Evaluate the model configuration on the test set."""
    my_collate_fn = partial(collate_fn_graph, include_line_graph=True)
    test_loader = MGLDataLoaderNoVal(
        train_data=test_dataset, 
        collate_fn=my_collate_fn, 
        batch_size=best_hyperparameters['batch_size'], 
        num_workers=config['dataloader']['num_workers']
    )

    # Test on the test set
    test_results = lit_module.trainer.test(model=lit_module, dataloaders=test_loader)
    test_mae = test_results[0]['test_MAE']
    print(f"Test MAE: {test_mae:.4f}")

    return test_mae

# Main function
if __name__ == "__main__":
    config = load_config()

    # Load training and test datasets
    train_structures, train_ionic_conductivities, train_ids = load_dataset(
        config['data']['csv_train_file'], 
        config['data']['cif_directory'], 
        "training", 
        config['data']['ionic_conductivity_column']
    )
    test_structures, test_ionic_conductivities, test_ids = load_dataset(
        config['data']['csv_test_file'], 
        config['data']['cif_directory'], 
        "testing", 
        config['data']['ionic_conductivity_column']
    )

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

    # Run hyperparameter search to get the best hyperparameters and validation MAE
    best_hyperparameters, best_val_mae = run_hyperparameter_search(config, model_type, train_dataset, train_structures)

    # Retrain the model on the full dataset with the best hyperparameters
    lit_module = retrain_on_full_dataset(
        config=config,
        model_type=model_type,
        best_hyperparameters=best_hyperparameters,
        train_dataset=train_dataset,
        train_structures=train_structures,
        load_pretrained=False
    )
    # Evaluate the model on the test set
    test_mae = run_test_evaluation(config, model_type, lit_module, test_dataset, train_structures)

    print(f"Before Pretraining:")
    print(f"Average Validation MAE: {best_val_mae:.4f} (best hyperparameters)")
    print(f"Final Test MAE: {test_mae:.4f}")

    # Now apply pretraining
    print("Applying Pretrained Model...")

    lit_module_pretrained = retrain_on_full_dataset(config, model_type, best_hyperparameters, train_dataset, train_structures, load_pretrained=True)

    # Re-evaluate the model on the test set after pretraining
    test_mae_pretrained = run_test_evaluation(config, model_type, lit_module_pretrained, test_dataset, train_structures)

    print(f"After Pretraining:")
    print(f"Final Test MAE (with pretraining): {test_mae_pretrained:.4f}")