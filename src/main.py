from dataset import load_metadata, create_fairness_df
import torch
import pandas as pd
from plots import plot_age_distribution, plot_class_distribution, plot_diagnosis_by_age
from utils import stratified_split
from torch.utils.data import DataLoader
from costum_image import CustomImageDataset
from utils import train_transform, eval_transform
from model import get_model, get_training_setup
from train import train_model
from evaluation import evaluate_model
from config import PLOT_DIR, AGE_GROUP_DICT
import os
from fairness import get_reweighted_training_setup, get_equalized_odds_training_setup, AdversarialDebiasingModel
from train_rw import train_reweighted
from train_eo import train_equalized_odds
from train_ad import train_adversarial, test_adversarial
from fairness_metrics import calculate_fairness_metrics


def main():
    # Load metadata and image paths
    metadata_df = load_metadata()

    # Create the fairness DataFrame with additional demographic attributes
    fairness_df = create_fairness_df(metadata_df)

    # Show basic information about the dataset
    """print("Fairness DataFrame created:")
    print(fairness_df.head())
    print("\nClass distribution:")
    print(fairness_df['cell_type'].value_counts())
    print("\nAge group distribution:")
    print(fairness_df['age_group'].value_counts())"""

    # plot_age_distribution(fairness_df)
    # plot_class_distribution(fairness_df)
    # plot_diagnosis_by_age(fairness_df)

    train_df, valid_df, test_df = stratified_split(fairness_df)
    # print(f"\nSplit complete: Train={len(train_df)}, Val={len(valid_df)}, Test={len(test_df)}")

    train_dataset = CustomImageDataset(train_df, transform=train_transform)
    val_dataset = CustomImageDataset(valid_df, transform=eval_transform)
    test_dataset = CustomImageDataset(test_df, transform=eval_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    # print(f"Train/Val/Test sizes: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # BASELINE MODEL

    baseline_model = get_model(num_classes=7, pretrained=True).to(device)
    criterion, optimizer, scheduler = get_training_setup(baseline_model, lr=0.01, wd=1e-3)

    best_model_state, best_val_acc = train_model(
        baseline_model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, num_epochs=5
    )

    model_path = os.path.join(PLOT_DIR, f"best_baseline_{best_val_acc:.2f}.pt")
    torch.save(best_model_state, model_path)
    print(f"\nBest model saved to: {model_path} (Val Acc: {best_val_acc:.2f}%)")

    baseline_model.load_state_dict(torch.load(model_path))
    test_accuracy = evaluate_model(baseline_model, test_dataloader, device, save_path=os.path.join(PLOT_DIR, "baseline_results.csv"))
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    # REWEIGHTING MODEL

    print("\nStarting Reweighting Fairness Training...")
    reweight_model = get_model(num_classes=7, pretrained=True).to(device)
    reweight_criterion, reweight_optimizer, reweight_scheduler, weights_dict = get_reweighted_training_setup(reweight_model, train_df)

    best_rw_state, best_rw_acc = train_reweighted(
        reweight_model,
        (train_dataloader, val_dataloader),
        reweight_criterion,
        reweight_optimizer,
        reweight_scheduler,
        weights_dict,
        device,
        epochs=5
    )

    reweight_model_path = os.path.join(PLOT_DIR, f"reweighting_model_{best_rw_acc:.2f}.pt")
    torch.save(best_rw_state, reweight_model_path)
    print(f"Reweighting model saved to: {reweight_model_path} (Val Acc: {best_rw_acc:.2f}%)")

    reweight_model.load_state_dict(torch.load(reweight_model_path))
    rw_test_acc = evaluate_model(reweight_model, test_dataloader, device, save_path=os.path.join(PLOT_DIR, "reweighting_results.csv"))
    print(f"Reweighting Test Accuracy: {rw_test_acc:.2f}%")

    # EQUALIZED ODDS MODEL

    print("\n Starting Fairness-Aware Training with Equalized Odds...")
    eo_model = get_model(num_classes=7, pretrained=True).to(device)
    eo_criterion, eo_optimizer, eo_scheduler = get_equalized_odds_training_setup(eo_model)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    best_eo_state, best_eo_acc = train_equalized_odds(
        eo_model, dataloaders, eo_criterion, eo_optimizer, eo_scheduler,
        age_group_mapping=AGE_GROUP_DICT, device=device, epochs=5
    )

    eo_model_path = os.path.join(PLOT_DIR, f"equalized_odds_model_{best_eo_acc:.2f}.pt")
    torch.save(best_eo_state, eo_model_path)
    print(f"EO model saved to: {eo_model_path} (Val Acc: {best_eo_acc:.2f}%)")

    eo_model.load_state_dict(torch.load(eo_model_path))
    eo_test_acc = evaluate_model(eo_model, test_dataloader, device, save_path=os.path.join(PLOT_DIR, "equalized_odds_results.csv"))
    print(f"Equalized Odds Test Accuracy: {eo_test_acc:.2f}%")

    # ADVERSARIAL MODEL

    print("\nStarting Adversarial Debiasing Training...")
    adv_model = AdversarialDebiasingModel(num_classes=7, num_sensitive=5, lambda_adv=1.0).to(device)

    best_adv_state, best_adv_val_acc = train_adversarial(adv_model, train_dataloader, val_dataloader, device, epochs=5)

    adv_model_path = os.path.join(PLOT_DIR, f"adversarial_model_{best_adv_val_acc:.2f}.pt")
    torch.save(best_adv_state, adv_model_path)
    print(f"Adversarial model saved to: {adv_model_path} (Val Acc: {best_adv_val_acc:.2f}%)")

    adv_model.load_state_dict(torch.load(adv_model_path))
    adv_test_task_acc, adv_test_adv_acc = test_adversarial(adv_model, test_dataloader, device, save_path=os.path.join(PLOT_DIR, "adversarial_results.csv"))
    print(f"Adversarial Test Task Accuracy: {adv_test_task_acc:.2f}%")
    print(f"Adversarial Test Adversary Accuracy: {adv_test_adv_acc:.2f}%")

    # COMPARISON

    print("\n\n=== FAIRNESS COMPARISON ===")

    results = {
    "Baseline": os.path.join(PLOT_DIR, "baseline_results.csv"),
    "Reweighting": os.path.join(PLOT_DIR, "reweighting_results.csv"),
    "Equalized Odds": os.path.join(PLOT_DIR, "equalized_odds_results.csv"),
    "Adversarial": os.path.join(PLOT_DIR, "adversarial_results.csv"),
    }

    for name, path in results.items():
        print(f"\n>> Fairness Metrics for {name} model:")
        df = pd.read_csv(path)
        calculate_fairness_metrics(df, num_classes=7, group_col="Age_Group")



if __name__ == "__main__":
    main()
