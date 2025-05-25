import os
import optuna
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset import load_metadata, create_fairness_df
from costum_image import CustomImageDataset
from model import get_model
from train_ad import train_adversarial, test_adversarial
from utils import train_transform, eval_transform, set_seed, stratified_split
from config import PLOT_DIR, AGE_GROUP_DICT
from fairness import AdversarialDebiasingModel
from fairness_metrics import equalized_odds, tpr_fpr_mean_diff

def objective(trial):
    set_seed(42)

    # Sample hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 5e-6, 1e-2, log=True)
    step_size = trial.suggest_int("step_size", 3, 9)
    lambda_adv = trial.suggest_float("lambda_adv", 1.0, 8.0, log=True)

    # Load and subset data
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
    model = AdversarialDebiasingModel(num_classes=7, num_sensitive=5, lambda_adv=lambda_adv).to(device)

    # Train model
    best_state, val_task_acc, val_adv_acc = train_adversarial(
        model, train_loader, val_loader, device,
        learning_rate=lr,
        weight_decay=weight_decay,
        step_size=step_size,
        epochs=10
    )

    model.load_state_dict(best_state)

    # Evaluate on test set
    results_path = os.path.join(PLOT_DIR, "ad_test_results.csv")
    test_task_acc, test_adv_acc = test_adversarial(model, test_loader, device, save_path=results_path)

    pred_df = pd.read_csv(results_path)

    # Compute fairness deviations from predictions
    tpr_dev_list, fpr_dev_list = [], []
    for cls in range(7):
        eo = equalized_odds(pred_df, class_id=cls, group_col="Age_Group")
        mean_diff_tpr, mean_diff_fpr = tpr_fpr_mean_diff(eo)
        tpr_dev_list.append(mean_diff_tpr)
        fpr_dev_list.append(mean_diff_fpr)

    fairness_score = np.mean(tpr_dev_list + fpr_dev_list)

    # Composite objective
    task_term = 1 - (val_task_acc / 100)   
    adv_term = val_adv_acc / 100           
    fair_term = fairness_score 
    print(fairness_score)        

    alpha = 1.5
    beta = 3
    gamma = 10                
    objective_value = alpha * task_term + beta * adv_term + gamma * fair_term

    return -objective_value 

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("\nBest trial:")
    print(f"  Value: {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")

    with open(os.path.join(PLOT_DIR, "0new_adversarial_debiasing_hyperparams.txt"), "w") as f:
        for k, v in study.best_trial.params.items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    main()

