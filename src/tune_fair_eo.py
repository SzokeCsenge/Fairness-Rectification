import os
import optuna
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset import load_metadata, create_fairness_df
from costum_image import CustomImageDataset
from model import get_model
from train_eo import train_equalized_odds
from utils import train_transform, eval_transform, set_seed, stratified_split
from config import PLOT_DIR, AGE_GROUP_DICT
from fairness import get_equalized_odds_training_setup
from evaluation import evaluate_model
from fairness_metrics import equalized_odds, tpr_fpr_mean_diff

def objective(trial):
    set_seed(42)

    # Sample hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 5e-2, log=True)
    step_size = trial.suggest_int("step_size", 3, 7)
    lambda_fairness = trial.suggest_float("lambda_fairness", 0.5, 3.5)

    metadata_df = load_metadata()
    fairness_df = create_fairness_df(metadata_df)
    fairness_df = fairness_df.sample(frac=0.35, random_state=42).reset_index(drop=True)

    train_df, val_df, test_df = stratified_split(fairness_df)

    train_dataset = CustomImageDataset(train_df, transform=train_transform)
    val_dataset = CustomImageDataset(val_df, transform=eval_transform)
    test_dataset = CustomImageDataset(test_df, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eo_model = get_model(num_classes=7, pretrained=True).to(device)

    # Setup with fairness regularization
    eo_criterion, eo_optimizer, eo_scheduler = get_equalized_odds_training_setup(
        eo_model, lr=lr, wd=weight_decay, ss=step_size, lambda_fairness=lambda_fairness
    )

    dataloaders = {'train': train_loader, 'val': val_loader}

    best_state, val_acc = train_equalized_odds(
        eo_model, dataloaders, eo_criterion, eo_optimizer, eo_scheduler,
        age_group_mapping=AGE_GROUP_DICT, device=device, epochs=10
    )

    eo_model.load_state_dict(best_state)

    results_path = os.path.join(PLOT_DIR, "eo_test_results.csv")
    test_acc = evaluate_model(eo_model, test_loader, device, save_path=results_path)
    pred_df = pd.read_csv(results_path)

    # Fairness across Age_Group
    tpr_dev_list = []
    fpr_dev_list = []
    for cls in range(7):
        eo = equalized_odds(pred_df, class_id=cls, group_col="Age_Group")
        mean_diff_tpr, mean_diff_fpr = tpr_fpr_mean_diff(eo)
        tpr_dev_list.append(mean_diff_tpr)
        fpr_dev_list.append(mean_diff_fpr)

    fairness_score = np.mean(tpr_dev_list + fpr_dev_list)
    print(fairness_score)

    alpha = 1.5
    beta = 10
    objective_value = alpha * (1 - val_acc / 100) + beta * fairness_score

    return -objective_value 

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("\nBest trial:")
    print(f"  Value: {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")

    with open(os.path.join(PLOT_DIR, "2_best_fair_equalized_odds_hyperparams.txt"), "w") as f:
        for k, v in study.best_trial.params.items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    main()
