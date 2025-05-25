import os
import optuna
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import load_metadata, create_fairness_df
from costum_image import CustomImageDataset
from model import get_model
from train_rw import train_reweighted
from fairness import get_reweighted_training_setup
from utils import train_transform, eval_transform, set_seed, stratified_split
from evaluation import evaluate_model
from config import PLOT_DIR

if __name__ ==  "__main__":
    set_seed(42)
    
    metadata_df = load_metadata()
    fairness_df = create_fairness_df(metadata_df)

    train_df, valid_df, test_df = stratified_split(fairness_df)

    train_dataset = CustomImageDataset(train_df, transform=train_transform)
    val_dataset = CustomImageDataset(valid_df, transform=eval_transform)
    test_dataset = CustomImageDataset(test_df, transform=eval_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(num_classes=7, pretrained=True).to(device)

    # Get reweighting setup with trial-sampled hyperparams
    criterion, optimizer, scheduler, weights_dict = get_reweighted_training_setup(
        model, train_df, lr=0.000892, wd=6.57676e-05, ss=5
    )

    best_model_state, best_val_acc = train_reweighted(
        model,
        (train_dataloader, val_dataloader),
        criterion,
        optimizer,
        scheduler,
        weights_dict,
        device,
        epochs=30
    )

    model_path = os.path.join(PLOT_DIR, f"best_rw_model_{best_val_acc:.2f}.pt")
    torch.save(best_model_state, model_path)
    print(f"\nBest rw model saved to: {model_path} (Val Acc: {best_val_acc:.2f}%)")

    model.load_state_dict(torch.load(model_path))
    test_accuracy = evaluate_model(model, test_dataloader, device, save_path=os.path.join(PLOT_DIR, "test_results.csv"))
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")