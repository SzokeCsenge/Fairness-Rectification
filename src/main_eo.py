import os
import optuna
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import load_metadata, create_fairness_df
from costum_image import CustomImageDataset
from model import get_model
from train_eo import train_equalized_odds
from fairness import get_equalized_odds_training_setup
from utils import train_transform, eval_transform, set_seed, stratified_split
from evaluation import evaluate_model
from config import PLOT_DIR, AGE_GROUP_DICT

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

    eo_model = get_model(num_classes=7, pretrained=True).to(device)
    eo_criterion, eo_optimizer, eo_scheduler = get_equalized_odds_training_setup(
        eo_model, lr=0.002703, wd=0.006847, ss=4, lambda_fairness=1.05168
        )

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    best_eo_state, best_eo_acc = train_equalized_odds(
        eo_model, dataloaders, eo_criterion, eo_optimizer, eo_scheduler,
        age_group_mapping=AGE_GROUP_DICT, device=device, epochs=30
    )

    eo_model_path = os.path.join(PLOT_DIR, f"equalized_odds_model_{best_eo_acc:.2f}.pt")
    torch.save(best_eo_state, eo_model_path)
    print(f"EO model saved to: {eo_model_path} (Val Acc: {best_eo_acc:.2f}%)")

    eo_model.load_state_dict(torch.load(eo_model_path))
    eo_test_acc = evaluate_model(eo_model, test_dataloader, device, save_path=os.path.join(PLOT_DIR, "equalized_odds_results.csv"))
    print(f"Equalized Odds Test Accuracy: {eo_test_acc:.2f}%")