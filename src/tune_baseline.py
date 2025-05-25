import os
import optuna
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import load_metadata, create_fairness_df
from costum_image import CustomImageDataset
from model import get_model, get_training_setup
from train import train_model
from utils import train_transform, eval_transform, set_seed, stratified_split
from config import PLOT_DIR

def objective(trial):
    set_seed(42)

    # Sample hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    step_size = trial.suggest_int("step_size", 5, 8)

    # Load and split data
    metadata_df = load_metadata()
    fairness_df = create_fairness_df(metadata_df)

    # First stratified split to get a small dataset
    fairness_df = fairness_df.sample(frac=0.5, random_state=42).reset_index(drop=True)
    train_df, val_df, _ = stratified_split(fairness_df)

    train_dataset = CustomImageDataset(train_df, transform=train_transform)
    val_dataset = CustomImageDataset(val_df, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=7, pretrained=True).to(device)
    criterion, optimizer, scheduler = get_training_setup(model, lr=lr, wd=weight_decay, ss=step_size)

    _, val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=15
    )

    return val_acc


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=8)

    print("\nBest trial:")
    print(f"  Value: {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")

    # Optional: Save best config
    os.makedirs(PLOT_DIR, exist_ok=True)
    with open(os.path.join(PLOT_DIR, "best_hyperparams.txt"), "w") as f:
        for k, v in study.best_trial.params.items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    main()