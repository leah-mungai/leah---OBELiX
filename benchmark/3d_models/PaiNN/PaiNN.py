# Import configuration
import yaml
import pickle
import torch
import numpy as np
import random
import os
import warnings
from ase.io import read
import pandas as pd
import schnetpack.transform as trn
import schnetpack as spk
from schnetpack.nn import GaussianRBF, CosineCutoff
import torchmetrics
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from schnetpack.data import ASEAtomsData
from copy import deepcopy
from collections import OrderedDict
from numpy.random import choice
from pathlib import Path

metrics_df = 0


@classmethod
def switch_getitem(cls, new_getitem=None):
    if new_getitem is None:
        cls.__getitem__ = cls.original_getitem
    else:
        cls.__getitem__ = new_getitem

        
ASEAtomsData.switch_getitem = switch_getitem
ASEAtomsData.original_getitem = deepcopy(ASEAtomsData.__getitem__)

def random_product(n, *args):
    sizes = torch.tensor([len(arg) for arg in args])
    if torch.prod(sizes) > n:
        Ns = choice(torch.prod(sizes), n, replace=False)
    else:
        Ns = np.arange(torch.prod(sizes))
    combs = []
    for N in Ns:
        comb = []
        for i, arg in enumerate(args):
            comb.append(arg[N//int(torch.prod(sizes[i+1:]))])
            N = N%int(torch.prod(sizes[i+1:]))
        combs.append(comb)

    return combs

def generate_configs(config):

    adjustable_params = OrderedDict()
    for k, v in config.items():
        if isinstance(v, list) and k not in (config["actual_list"] + ["actual_list"]):
            adjustable_params[k] = v

    print("Grid seach through the following  hyper-parameters:")
    for k, v in adjustable_params.items():
        print(f"{k}: {v}")

    list_params = random_product(config["n_param_combs"], *adjustable_params.values())

    possible_configs = []
    for params in list_params:
        tmp_config = deepcopy(config)
        for i, k in enumerate(adjustable_params.keys()):
            tmp_config[k] = params[i]
        possible_configs.append(tmp_config)
        
    param_grid = pd.DataFrame(np.array(list_params), columns=list(adjustable_params.keys()))
        
    return possible_configs, param_grid

    
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
        metrics_df.to_csv(trainer.default_root_dir + "/metrics_output.csv", index=False)

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_pred_vs_targets(model, dataloader):
    actual_ionic_conductivities, predicted_ionic_conductivities = [], []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            actual_ionic_conductivities.append(batch['ionic_conductivity'].cpu().numpy())
            predicted_ionic_conductivities.append(model(batch)['ionic_conductivity'].cpu().detach().numpy())
    return np.concatenate(predicted_ionic_conductivities), np.concatenate(actual_ionic_conductivities)

def plot_actual_vs_predicted(pred_train, target_train, pred_val, target_val, model_name, save_path):
    
    plt.figure(figsize=(6, 6))
    plt.scatter(target_train, pred_train, label='Train', s=10)
    plt.scatter(target_val, pred_val, label='Validation', s=10)

    min_val = min(np.min(target_train), np.min(target_val))
    max_val = max(np.max(target_train), np.max(target_val))

    # Diagonal line indicating perfect prediction
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.xlabel('Actual Ionic Conductivity (log-transformed)')
    plt.ylabel('Predicted Ionic Conductivity')
    plt.title(f'Actual vs Predicted Ionic Conductivity ({model_name})')

    plt.savefig(save_path)

def prepare_data(db_path, train_data, config):
    
    if not os.path.exists(db_path) or config["overwrite_db"]:

        if os.path.exists(db_path):
            os.remove(db_path)

        atoms_list, property_list, ionic_conductivity_list, id_list = [], [], [], []

        print("Reading CIFs...", end='', flush=True)
        for idx, row in train_data.iterrows():

            cif_file_path = os.path.join(config["cif_file_dir"], f'{idx}.cif')  # Path from config.yam

            if row['CIF'].lower() != "no match":
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
                    ionic_conductivity_list.append(ionic_conductivity)
                
                except Exception as e:
                    print(f'Failed to process CIF file for {idx}: {e}')

        print(f"Processed {len(atoms_list)} CIF files")
        new_dataset = ASEAtomsData.create(
            db_path,
            distance_unit='Ang',
            property_unit_dict={'ionic_conductivity': 'S/cm'},
            transforms=[
            trn.ASENeighborList(cutoff=config["cutoff"]),
            trn.RemoveOffsets("ionic_conductivity", remove_mean=config["remove_mean"], remove_atomrefs=config["remove_atomrefs"], is_extensive=False, property_mean=torch.tensor([2.5])),
            trn.CastTo32()])
        new_dataset.add_systems(property_list, atoms_list)

        print("Preprocessing properties...", end='', flush=True)
        ASEAtomsData.switch_getitem()
        props = [new_dataset[i] for i in range(len(new_dataset))]
        print("Done")
        
        pickle.dump(props, open("pre-transformed.pkl", 'wb'))

    else:
        props = pickle.load(open("pre-transformed.pkl", 'rb'))

    return props

def setup_dataloader(props, config, split_file, db_path):

    custom_data = spk.data.AtomsDataModule(
        db_path,
        batch_size=config["batch_size"],
        distance_unit='Ang',
        split_file=split_file,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )
    custom_data.prepare_data()
    custom_data.setup()

    def new_getitem(self, idx):
        if self.subset_idx is not None:
            idx = self.subset_idx[idx]
        return props[idx]

    ASEAtomsData.switch_getitem(new_getitem)
    
    return custom_data
    
def train(props, split_file, db_path, config, savedir):
    ####### step 4: Data Splitting and Loading

    print("Total number of datapoints: ", len(props))

    custom_data = setup_dataloader(props, config, split_file, db_path)

    cutoff = config["cutoff"]
    n_atom_basis = config["n_atom_basis"]
    n_interactions = config["n_interactions"]

    pairwise_distance = spk.atomistic.PairwiseDistances()
    radial_basis = GaussianRBF(n_rbf=20, cutoff=cutoff)

    if config["representation"].lower() == "painn":
        rep = spk.representation.PaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            radial_basis=radial_basis,
            cutoff_fn=CosineCutoff(cutoff)
        )
    elif config["representation"].lower() == "schnet":
        rep = spk.representation.SchNet(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            radial_basis=radial_basis,
            cutoff_fn=CosineCutoff(cutoff)
        )
    else:
        raise ValueError("Unknown representation")
        
    pred_ionic_conductivity = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='ionic_conductivity', aggregation_mode="avg")

    nnpot = spk.model.NeuralNetworkPotential(
        representation=rep,
        input_modules=[pairwise_distance],
        output_modules=[pred_ionic_conductivity],
        postprocessors=[trn.CastTo64()]
    )

    ####### step 7: Model Training and Logging

    output_ionic_conductivity = spk.task.ModelOutput(
        name='ionic_conductivity',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.0,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_ionic_conductivity],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": config["learning_rate"], "weight_decay": config["weight_decay"]}
    )

    logger = pl.loggers.TensorBoardLogger(save_dir=savedir)

    global metrics_df
    metrics_df = pd.DataFrame(columns=["Epoch", "Train Loss", "Val Loss", "Train MAE", "Val MAE"])


    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=savedir + "/best_model.pth",
            save_top_k=config["save_top_k"],
            monitor=config["monitor_metric"]
        ),
        MetricTracker(),
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=savedir,
        max_epochs=config["max_epochs"],
        deterministic=True,
        benchmark=False
    )

    trainer.fit(task, datamodule=custom_data)
    
    return trainer, custom_data

def crossval(props, db_path, train_data, config, rng, savedir):
    
    shuffled_ids = rng.permutation(len(props))
    splits = np.array_split(shuffled_ids, config["num_folds"])

    if config["extra_plots"]:
        plt.figure(savedir)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs. Epochs')
        plt.legend()

    train_maes = []
    val_maes = []
    for i in range(config["num_folds"]):
        
        train_idx = np.concatenate(splits[:i] + splits[i+1:])
        val_idx = splits[i]
        
        split_file = config["db_dir"] + "/split.npz"
        
        np.savez(split_file, train_idx=train_idx, val_idx=val_idx, test_idx=[])

        traindir = savedir + f"/fold_{i}"
        Path(traindir).mkdir(parents=True, exist_ok=True)
        
        trainer, custom_data = train(props, split_file, db_path, config, traindir)
        
        # Final predictions
        pred_train, target_train = get_pred_vs_targets(trainer.model, custom_data.train_dataloader())
        pred_val, target_val = get_pred_vs_targets(trainer.model, custom_data.val_dataloader())
        
        train_maes.append(np.mean(np.abs(pred_train - target_train)))
        val_maes.append(np.mean(np.abs(pred_val - target_val)))
        
        if config["extra_plots"]:
            plt.figure(savedir)
            plt.plot(range(1, len(trainer.model.train_losses) + 1), trainer.model.train_losses,
                     label='Train Loss', color='blue', linestyle='--')
            plt.plot(range(1, len(trainer.model.val_losses) + 1), trainer.model.val_losses,
                     label='Validation Loss', color='orange', linestyle='--')
            
            plot_actual_vs_predicted(pred_train, target_train,
                                     pred_val, target_val,
                                     f"fold {i}", traindir + "/pred_vs_target.svg")

    if config["extra_plots"]:
        plt.figure(savedir)
        plt.savefig(savedir + "/loss_vs_epochs.svg")

    print()
    print("--------------------------------------------------------")
    print(f"Results for {config['num_folds']}-fold cross-validation:")
    print(f"Train MAE: {np.mean(train_maes)} +/- {np.std(train_maes)}")
    print(f"Validation MAE: {np.mean(val_maes)} +/- {np.std(val_maes)}")

    return np.mean(val_maes), np.std(val_maes)

def main():

    with open('config_painn.yaml', 'r') as file:
        config = yaml.safe_load(file)

    ####### step 1: Set Random Seed
    torch.use_deterministic_algorithms(True)
    set_random_seed(config["seed"])  # Seed value from config.yaml
    rng = np.random.default_rng(seed=config["seed"])

    ####### step 2: Clean up runtime files

    runtime_files = config["runtime_files"] # List of runtime files from config.yaml

    for file_path in runtime_files:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
                print(f"Deleted directory: {file_path}")


    ####### step 3: Dataset Preprocessing

    Path(config["out_dir"]).mkdir(parents=True, exist_ok=True)
    
    warnings.filterwarnings('ignore')

    data = pd.read_csv(config["database_path"], index_col="ID")
    test = pd.read_csv(config["test_path"], index_col="ID")
    train_data = data[~data.index.isin(test.index)]
    test_data = data[data.index.isin(test.index)]
    
    possible_configs, param_grid = generate_configs(config)
    
    best_val_mae = np.inf
    maes = []
    for i, tmp_config in enumerate(possible_configs):

        print()
        print("--------------------------------------------------------")
        print(f"Running crossval for configuration {i + 1}/{len(possible_configs)}")
        print(param_grid.iloc[i])
        
        db_path = tmp_config["db_dir"] + "/train.db"
        props = prepare_data(db_path, train_data, tmp_config)
        props = props[:config["n_train_data"]]

        savedir = config["out_dir"] + f"/config_{i}"
        Path(savedir).mkdir(parents=True, exist_ok=True)
        
        val_mae, val_std = crossval(props, db_path, train_data, tmp_config, rng, savedir)

        maes.append(val_mae)
        yaml.dump(tmp_config, open(savedir + "/config.yaml", 'w'))
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_std = val_std
            best_config = tmp_config
    maes = np.array(maes)
            
    print()
    print("========================================================")
    print("Best configuration found:")
    print(f"Validation MAE: {best_val_mae} +/- {best_val_std}")

    plt.figure()
    cm = plt.cm.get_cmap('winter')
    for i, params in param_grid.iterrows():
        plt.plot(params.index, params.to_numpy(), color=cm((maes[i] - maes.min()) / (maes.max() - maes.min())))
    plt.xlabel('Configuration')
    plt.ylabel('Hyper-parameter value')
    plt.colorbar(plt.cm.ScalarMappable(cmap=cm), label='Validation MAE', ax=plt.gca())
    plt.savefig(config["out_dir"] + '/sweep.svg')   

    yaml.dump(best_config, open(config["out_dir"] + "/best_config.yaml", 'w'))

    print("Retraining model with best configuration...")
    split_file = config["db_dir"] + "/split.npz"  
    np.savez(split_file, train_idx=np.arange(len(props)), val_idx=[], test_idx=[])
    trainer, custom_data = train(props, split_file, db_path, best_config, config["out_dir"] + "/final_training") 

    # Final predictions
    pred_train, target_train = get_pred_vs_targets(trainer.model, custom_data.train_dataloader())

    # --------------------------------------------------------------------------------  
    # Evaluate model on test set
    print("Evaluating model on test set...")
    test_db_path = config["db_dir"] + "/test.db"
    test_props = prepare_data(test_db_path, test_data, best_config)
    split_file = config["db_dir"] + "/split.npz"    
    np.savez(split_file, train_idx=[], val_idx=[], test_idx=np.arange(len(test_props)))
    test_data = setup_dataloader(test_props, best_config, split_file, test_db_path)

    pred_test, target_test = get_pred_vs_targets(trainer.model, test_data.test_dataloader())

    plot_actual_vs_predicted(pred_train, target_train,
                             pred_test, target_test,
                             "final model vs test", config["out_dir"] + "/target_vs_train.svg")

    if config["show"]:
        plt.show()

if __name__ == '__main__':
    main()
