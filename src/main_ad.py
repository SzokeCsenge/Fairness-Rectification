import os
import optuna
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import load_metadata, create_fairness_df
from costum_image import CustomImageDataset
from model import get_model
from train_eo import train_equalized_odds
from utils import train_transform, eval_transform, set_seed, stratified_split
from config import PLOT_DIR, AGE_GROUP_DICT
from fairness import AdversarialDebiasingModel2
from train_ad import train_adversarial, test_adversarial

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

    adv_model = AdversarialDebiasingModel2(num_classes=7, num_sensitive=5, lambda_adv=7.09589).to(device)

    best_adv_state, best_adv_val_acc, _ = train_adversarial(
        adv_model, train_dataloader, val_dataloader, device, learning_rate=1.48709e-05, weight_decay=1.17312e-05, step_size=9, epochs=10
        )

    adv_model_path = os.path.join(PLOT_DIR, f"adversarial_model_{best_adv_val_acc:.2f}.pt")
    torch.save(best_adv_state, adv_model_path)
    print(f"Adversarial model saved to: {adv_model_path} (Val Acc: {best_adv_val_acc:.2f}%)")

    adv_model.load_state_dict(torch.load(adv_model_path))
    adv_test_task_acc, adv_test_adv_acc = test_adversarial(adv_model, test_dataloader, device, save_path=os.path.join(PLOT_DIR, "14_adversarial_fair_results.csv"))
    print(f"Adversarial Test Task Accuracy: {adv_test_task_acc:.2f}%")
    print(f"Adversarial Test Adversary Accuracy: {adv_test_adv_acc:.2f}%")