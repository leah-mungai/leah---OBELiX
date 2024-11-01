import argparse
import os
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from random import sample
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from cgcnn.data import CIFData, collate_pool
from cgcnn.model import CrystalGraphConvNet

# Argument parsing
parser = argparse.ArgumentParser(description='CGCNN Training Workflow')
parser.add_argument('data_root', metavar='DIR', help='Path to data root directory containing CIF files and id_prop files')
parser.add_argument('--pretrained_dir', default='pre-trained', type=str, help='Directory containing pre-trained files')
parser.add_argument('--task', choices=['regression', 'classification'], default='regression', help='Task type (default: regression)')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='Number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=5, type=int, metavar='N', help='Number of epochs (default: 50)')
parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='Batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='Learning rate (default: 0.001)')
parser.add_argument('--lr-milestones', default=[30, 50], nargs='+', type=int, metavar='N', help='Scheduler milestones (default: [30, 50])')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='Path to latest checkpoint')
parser.add_argument('--n-folds', default=3, type=int, help='Number of folds for cross-validation')
parser.add_argument('--seed', default=42, type=int, help='Random seed')
parser.add_argument('--num-cases', default=1, type=int, help='Number of hyperparameter cases to test')

args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()

if args.task == 'regression':
    best_mae_error = float('inf')
else:
    best_mae_error = 0.

# Main function
def main():
    global args, best_mae_error
    set_random_seed(args.seed)

    # Load datasets for training and testing
    train_file = os.path.join(args.data_root, 'train', 'id_prop_train.csv')
    test_file = os.path.join(args.data_root, 'test', 'id_prop_test.csv')
    dataset = CIFData(root_dir=args.data_root, id_prop_file=train_file)
    test_dataset = CIFData(root_dir=args.data_root, id_prop_file=test_file)


    best_hyperparams, best_val_mae = hyperparameter_search(args, dataset)

    # Train the best model on the full training set and evaluate on the test set
    model = train_best_model(args, best_hyperparams, dataset)
    test_mae = evaluate_on_test_set(model, test_dataset)

    print(f"\nSummary before pretraining:")
    print(f"Validation MAE (before pretraining): {best_val_mae:.4f}")
    print(f"Test MAE (before pretraining): {test_mae:.4f}")


    run_pretrained_evaluation(args, best_hyperparams, dataset, test_dataset, best_val_mae, test_mae)

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Hyperparameter search function
def hyperparameter_search(args, dataset):
    hyperparameter_space = {
        'atom_fea_len': [32, 64, 128],
        'h_fea_len': [64, 128, 256],
        'n_conv': [2, 3, 4],
        'n_h': [1, 2, 3]
    }

    all_combinations = [dict(zip(hyperparameter_space, v)) for v in itertools.product(*hyperparameter_space.values())]
    selected_combinations = sample(all_combinations, args.num_cases)

    best_val_mae, best_hyperparams = float('inf'), None

    for idx, hyperparams in enumerate(selected_combinations):
        print(f"Testing combination {idx + 1}/{args.num_cases}: {hyperparams}")
        val_mae, val_mae_std = cross_validate(args, dataset, hyperparams)
        if val_mae < best_val_mae:
            best_val_mae, best_hyperparams = val_mae, hyperparams

    print(f"Best Hyperparameters: {best_hyperparams}\nLowest Val MAE: {best_val_mae:.4f}")
    return best_hyperparams, best_val_mae

# Cross-validation function
def cross_validate(args, dataset, hyperparams):
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    val_mae_list, train_mae_list = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Starting fold {fold + 1}/{args.n_folds}")
        
        train_subset, val_subset = Subset(dataset, train_idx), Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pool)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, collate_fn=collate_pool)

        model = build_model(hyperparams, dataset)
        criterion, optimizer = nn.MSELoss(), optim.Adam(model.parameters(), lr=args.lr)
        scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

        # Track losses...
        train_loss_list = []

        for epoch in range(args.epochs):
            train_loss, train_mae = train(train_loader, model, criterion, optimizer)
            scheduler.step()
            train_loss_list.append(train_loss)
            train_mae_list.append(train_mae)


        val_loss, val_mae, preds, targets = validate(val_loader, model, criterion)
        val_mae_list.append(val_mae)
        
        # Print fold metrics
        print(f"Fold {fold + 1}/{args.n_folds} - Avg Train Loss: {np.mean(train_loss_list):.4f}, Train MAE: {np.mean(train_mae_list):.4f}, Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}")
        plot_pred_vs_actual(preds, targets, fold, args, hyperparams)

    # Calculate mean and standard deviation of MAE across folds
    val_mae_mean, val_mae_std = np.mean(val_mae_list), np.std(val_mae_list)
    train_mae_mean, train_mae_std = np.mean(train_mae_list), np.std(train_mae_list)
    print(f"\nCross-validation results - Avg Validation MAE: {val_mae_mean:.4f} ± {val_mae_std:.4f}, Avg Train MAE: {train_mae_mean:.4f} ± {train_mae_std:.4f}")

    return val_mae_mean, val_mae_std  

# Model builder
def build_model(hyperparams, dataset):
    structures, _, _ = dataset[0]
    orig_atom_fea_len, nbr_fea_len = structures[0].shape[-1], structures[1].shape[-1]

    model = CrystalGraphConvNet(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        atom_fea_len=hyperparams['atom_fea_len'],
        h_fea_len=hyperparams['h_fea_len'],
        n_conv=hyperparams['n_conv'],
        n_h=hyperparams['n_h']
    )
    return model.cuda() if torch.cuda.is_available() else model

def train(train_loader, model, criterion, optimizer):
    model.train()
    losses, mae_errors = [], []

    for inputs, targets, _ in train_loader:
        inputs = [input.cuda() for input in inputs] if args.cuda else inputs
        targets = targets.cuda() if args.cuda else targets

        optimizer.zero_grad()
        outputs = model(*inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        
        mae_errors.append(mae(outputs.detach().cpu().numpy(), targets.cpu().numpy()))

    avg_loss = np.mean(losses)
    avg_mae = np.mean(mae_errors)
    print(f"Train Loss: {avg_loss:.4f}, Train MAE: {avg_mae:.4f}")
    return avg_loss, avg_mae


def validate(val_loader, model, criterion):
    model.eval()
    losses, mae_errors, preds, targets = [], [], [], []

    with torch.no_grad():
        for inputs, target, _ in val_loader:
            inputs = [input.cuda() for input in inputs] if args.cuda else inputs
            target = target.cuda() if args.cuda else target

            output = model(*inputs)
            loss = criterion(output, target)
            losses.append(loss.item())

            preds.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())
            mae_errors.append(mae(output.cpu().numpy(), target.cpu().numpy()))

    avg_loss = np.mean(losses)
    avg_mae = np.mean(mae_errors)
    return avg_loss, avg_mae, np.array(preds), np.array(targets)

def mae(preds, targets):
    return np.mean(np.abs(preds - targets))

def plot_pred_vs_actual(preds, targets, fold, args, hyperparams=None):
    plt.figure(figsize=(6, 6))
    
    # Create a unique label and filename
    label = 'Test' if fold == "Test" else f'Fold {fold + 1}'
    if hyperparams:
        filename = f"pred_vs_actual_{label}_{hyperparams['atom_fea_len']}_{hyperparams['h_fea_len']}_{hyperparams['n_conv']}_{hyperparams['n_h']}.png"
    else:
        filename = f"pred_vs_actual_{label}_test.png"  

    plt.scatter(targets, preds, alpha=0.5, label=label)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predicted vs Actual - {label}')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close()

# Train best model on full dataset
def train_best_model(args, best_hyperparams, dataset, pretrained_file=None):
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pool)
    
    # Build model using the given hyperparameters
    model = build_model(best_hyperparams, dataset)
    
    # Load pre-trained model if specified
    if pretrained_file:
        pretrained_path = os.path.join(args.pretrained_dir, pretrained_file)
        if args.cuda:
            checkpoint = torch.load(pretrained_path)
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'))
        
        # Get the pre-trained state_dict and filter layers that match
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()

        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        
        # Update the current model's state_dict with compatible pre-trained weights
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded pre-trained model from {pretrained_file} with matching layers only.")

    criterion, optimizer = nn.MSELoss(), optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

    for epoch in range(args.epochs):
        train(train_loader, model, criterion, optimizer)
        scheduler.step()

    return model




# Evaluate on test set
def evaluate_on_test_set(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_pool)
    val_loss, val_mae, preds, targets = validate(test_loader, model, nn.MSELoss())
    plot_pred_vs_actual(preds, targets, "Test", args)  # hyperparams not needed here
    return val_mae

# Evaluate each pre-trained model
def run_pretrained_evaluation(args, best_hyperparams, dataset, test_dataset, val_mae_before, test_mae_before):
    pretrained_files = [
        "band-gap.pth.tar",
        "bulk-moduli.pth.tar",
        "final-energy-per-atom.pth.tar",
        "formation-energy-per-atom.pth.tar",
        "shear-moduli.pth.tar",
        "efermi.pth.tar"
    ]
    
    print("\nEvaluating with pre-trained models:\n")
    
    results = []
    for pretrained_file in pretrained_files:
        model = train_best_model(args, best_hyperparams, dataset, pretrained_file)
        val_mae_after = cross_validate(args, dataset, best_hyperparams)[0]
        test_mae_after = evaluate_on_test_set(model, test_dataset)
        
        print(f"\nResults with {pretrained_file}:")
        print(f"Validation MAE (after pretraining): {val_mae_after:.4f}")
        print(f"Test MAE (after pretraining): {test_mae_after:.4f}")
        results.append((pretrained_file, val_mae_after, test_mae_after))

    # Summary comparison
    print("\nMAE Comparison Summary:\n")
    print(f"{'Pretrained File':<30} {'Val MAE Before':<15} {'Val MAE After':<15} {'Test MAE Before':<15} {'Test MAE After':<15}")
    for pretrained_file, val_mae_after, test_mae_after in results:
        print(f"{pretrained_file:<30} {val_mae_before:<15.4f} {val_mae_after:<15.4f} {test_mae_before:<15.4f} {test_mae_after:<15.4f}")

if __name__ == '__main__':
    main()
