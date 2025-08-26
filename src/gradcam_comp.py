import numpy as np
import pandas as pd
from collections import defaultdict
import json
from torchcam.methods import GradCAM
import torch.nn.functional as F
import torch
import os
from torchvision import transforms
from model import get_model
from utils import set_seed, stratified_split
from dataset import load_metadata, create_fairness_df
from costum_image import CustomImageDataset 
from config import PLOT_DIR
from fairness import AdversarialDebiasingModel2
from collections import defaultdict


def main():
    baseline_cam = np.load("/scratch/s5139090/Fairness-Rectification/gradcam_outputs/correct/baseline_correct_class_1_cam.npy")
    fair_cam = np.load("/scratch/s5139090/Fairness-Rectification/gradcam_outputs/correct/ad_correct_class_1_cam.npy")
    consistency_score = np.mean(np.abs(baseline_cam - fair_cam))
    print(consistency_score)


def something():
    # Load CSVs
    baseline_df = pd.read_csv("../Outputs/baseline_results.csv")
    rw_df = pd.read_csv("../Outputs/reweighting_results.csv")
    eo_df = pd.read_csv("../Outputs/equalized_odds_results.csv")
    adv_df = pd.read_csv("../Outputs/adversarial_debiasing_results.csv")

    # Stack all predictions
    dfs = [baseline_df, rw_df, eo_df, adv_df]

    # Find rows where all models predicted correctly
    common_correct_indices = []

    for idx in range(len(baseline_df)):
        true_label = baseline_df.loc[idx, 'Label']
        if all(df.loc[idx, 'Prediction'] == true_label for df in dfs):
            common_correct_indices.append((idx, true_label))

    # Group by class and pick one per class
    class_to_index = {}
    for idx, label in common_correct_indices:
        if label not in class_to_index:
            class_to_index[label] = idx
        if len(class_to_index) == 7:
            break

    # Save for later use
    selected_indices = list(class_to_index.values())

    output_path = "../gradcam_outputs/consistent_correct_indices.json"
    with open(output_path, "w") as f:
        json.dump(selected_indices, f)

    print("Selected indices:", selected_indices)


def something2():
    # Load CSVs
    baseline_df = pd.read_csv("../Outputs/baseline_results.csv")
    rw_df = pd.read_csv("../Outputs/reweighting_results.csv")
    eo_df = pd.read_csv("../Outputs/equalized_odds_results.csv")
    adv_df = pd.read_csv("../Outputs/adversarial_debiasing_results.csv")

    # Indices to exclude (already selected)
    excluded_indices = {1, 2, 15, 253, 266, 282, 485, 486, 518, 519, 616, 1974, 1982}

    # Stack all predictions
    dfs = [baseline_df, rw_df, adv_df]

    # Find rows where all models predicted correctly
    common_correct_indices = []

    for idx in range(len(baseline_df)):
        if idx in excluded_indices:
            continue
        true_label = baseline_df.loc[idx, 'Label']
        if all(df.loc[idx, 'Prediction'] == true_label for df in dfs):
            common_correct_indices.append((idx, true_label))

    # Group by class and pick one per class (excluding previously used indices)
    class_to_index = {}
    for idx, label in common_correct_indices:
        if label not in class_to_index:
            class_to_index[label] = idx
        if len(class_to_index) == 7:
            break

    # Save for later use
    selected_indices = list(class_to_index.values())

    output_path = "../gradcam_outputs/consistent_correct_indices_3.json"
    with open(output_path, "w") as f:
        json.dump(selected_indices, f)

    print("Second set of selected indices:", selected_indices)


def compute_consistency_across_testset(baseline_model, fair_model, dataset, target_layer="layer4", save_path=None):
    baseline_model.eval()
    fair_model.eval()

    baseline_model.to(device)
    fair_model.to(device)

    baseline_cam_extractor = GradCAM(baseline_model, target_layer=target_layer)
    fair_cam_extractor = GradCAM(fair_model.backbone, target_layer=target_layer)

    consistency_scores = []

    for idx in range(len(dataset)):
        img_tensor, label, *_ = dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        # Forward pass through both models
        with torch.enable_grad():
            output_baseline = baseline_model(input_tensor)
            #output_fair = fair_model(input_tensor)
            output_fair_raw = fair_model(input_tensor)
            if isinstance(output_fair_raw, tuple):
                output_fair = output_fair_raw[0]
            else:
                output_fair = output_fair_raw

        pred_baseline = output_baseline.argmax().item()
        pred_fair = output_fair.argmax().item()

        # Compute Grad-CAM for the predicted class of each model
        cam_base = baseline_cam_extractor(pred_baseline, output_baseline)[0].cpu()
        cam_fair = fair_cam_extractor(pred_fair, output_fair)[0].cpu()

        # Resize to match input size
        cam_base = F.interpolate(cam_base.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)[0]
        cam_fair = F.interpolate(cam_fair.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)[0]

        # Normalize
        cam_base = (cam_base - cam_base.min()) / (cam_base.max() - cam_base.min() + 1e-6)
        cam_fair = (cam_fair - cam_fair.min()) / (cam_fair.max() - cam_fair.min() + 1e-6)

        # Compute consistency
        consistency_score = torch.mean(torch.abs(cam_base - cam_fair)).item()

        consistency_scores.append({
            "index": idx,
            "label": int(label),
            "pred_baseline": pred_baseline,
            "pred_fair": pred_fair,
            "consistency": consistency_score
        })

        if idx % 100 == 0:
            print(f"[{idx}/{len(dataset)}] Consistency: {consistency_score:.4f}")

    average_score = sum(entry["consistency"] for entry in consistency_scores) / len(consistency_scores)
    print(f"Average consistency score: {average_score:.4f}")
    # Optionally save
    if save_path:
        import json
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(consistency_scores, f, indent=2)

    return consistency_scores

def compute_consistency_age(baseline_model, fair_model, dataset, target_layer="layer4", save_path=None):
    baseline_model.eval()
    fair_model.eval()

    baseline_model.to(device)
    fair_model.to(device)

    baseline_cam_extractor = GradCAM(baseline_model, target_layer=target_layer)
    fair_cam_extractor = GradCAM(fair_model.backbone, target_layer=target_layer)

    consistency_scores = []
    groupwise_consistency = defaultdict(list)

    for idx in range(len(dataset)):
        img_tensor, label, _, age_group, _ = dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.enable_grad():
            output_baseline = baseline_model(input_tensor)
            output_fair_raw = fair_model(input_tensor)
            output_fair = output_fair_raw[0] if isinstance(output_fair_raw, tuple) else output_fair_raw

        pred_baseline = output_baseline.argmax().item()
        pred_fair = output_fair.argmax().item()

        # Grad-CAM extraction
        cam_base = baseline_cam_extractor(pred_baseline, output_baseline)[0].cpu()
        cam_fair = fair_cam_extractor(pred_fair, output_fair)[0].cpu()

        cam_base = F.interpolate(cam_base.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)[0]
        cam_fair = F.interpolate(cam_fair.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)[0]

        cam_base = (cam_base - cam_base.min()) / (cam_base.max() - cam_base.min() + 1e-6)
        cam_fair = (cam_fair - cam_fair.min()) / (cam_fair.max() - cam_fair.min() + 1e-6)

        consistency_score = torch.mean(torch.abs(cam_base - cam_fair)).item()

        consistency_scores.append({
            "index": idx,
            "label": int(label),
            "pred_baseline": pred_baseline,
            "pred_fair": pred_fair,
            "age_group": age_group,
            "consistency": consistency_score
        })

        groupwise_consistency[age_group].append(consistency_score)

        if idx % 100 == 0:
            print(f"[{idx}/{len(dataset)}] Age Group: {age_group} | Consistency: {consistency_score:.4f}")

    # Report average per group
    print("\n--- Average Consistency by Age Group ---")
    for group, scores in sorted(groupwise_consistency.items()):
        avg_score = sum(scores) / len(scores)
        print(f"Age Group {group}: {avg_score:.4f}")

    # Overall average
    overall_avg = sum(entry["consistency"] for entry in consistency_scores) / len(consistency_scores)
    print(f"\nOverall Average Consistency: {overall_avg:.4f}")

    # Optionally save
    if save_path:
        import json
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(consistency_scores, f, indent=2)

    return consistency_scores, groupwise_consistency

if __name__ ==  "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    baseline_model = get_model(num_classes=7)
    baseline_model.load_state_dict(torch.load("../Outputs/equalized_odds_model_75.98.pt", map_location=device))

    fair_model = AdversarialDebiasingModel2(num_classes=7, num_sensitive=5, lambda_adv=1.0)
    fair_model.load_state_dict(torch.load("../Outputs/adversarial_model_82.58.pt", map_location=device))

    metadata_df = load_metadata()
    fairness_df = create_fairness_df(metadata_df)
    train_df, valid_df, test_df = stratified_split(fairness_df)
    test_dataset = CustomImageDataset(test_df, transform=evaluation_transform)

    consistency_results = compute_consistency_age(
        baseline_model,
        fair_model,
        test_dataset,
        target_layer="layer4",
        save_path="../gradcam_outputs/consistency_scores_age_eo_vs_ad.json"
    )

