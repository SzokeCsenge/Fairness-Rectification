from dataset import load_metadata, create_fairness_df
import torch
import pandas as pd
from utils import stratified_split
from torch.utils.data import DataLoader
from costum_image import CustomImageDataset
from utils import train_transform, eval_transform, set_seed
from model import get_model, get_training_setup
from train import train_model
from evaluation import evaluate_model
from config import PLOT_DIR, AGE_GROUP_DICT
import os

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

    baseline_model = get_model(num_classes=7, pretrained=True).to(device)
    criterion, optimizer, scheduler = get_training_setup(baseline_model, lr=0.000156, wd=0.00010688, ss=5)

    best_model_state, best_val_acc = train_model(
        baseline_model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, num_epochs=30
    )

    model_path = os.path.join(PLOT_DIR, f"best_baseline_{best_val_acc:.2f}.pt")
    torch.save(best_model_state, model_path)
    print(f"\nBest model saved to: {model_path} (Val Acc: {best_val_acc:.2f}%)")

    baseline_model.load_state_dict(torch.load(model_path))
    test_accuracy = evaluate_model(baseline_model, test_dataloader, device, save_path=os.path.join(PLOT_DIR, "baseline_results.csv"))
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")