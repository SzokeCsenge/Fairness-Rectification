import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import json
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import GradCAM
import torch.nn.functional as F
from torchvision import transforms
from model import get_model
from utils import set_seed, stratified_split
from dataset import load_metadata, create_fairness_df
from costum_image import CustomImageDataset 
from config import PLOT_DIR
from fairness import AdversarialDebiasingModel2

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

evaluation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

model_path = os.path.join(PLOT_DIR, "equalized_odds_model_75.98.pt")
model = get_model(num_classes=7)
#model = AdversarialDebiasingModel2(num_classes=7, num_sensitive=5, lambda_adv=1.0)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

metadata_df = load_metadata()
fairness_df = create_fairness_df(metadata_df)

train_df, valid_df, test_df = stratified_split(fairness_df)
test_dataset = CustomImageDataset(test_df, transform=evaluation_transform)

with open("../gradcam_outputs/consistent_correct_indices_2.json", "r") as f:
    selected_indices = json.load(f)

consistent_samples = {}

for idx in selected_indices:
    img, label, *_ = test_dataset[idx]
    label = int(label)
    print(f"Index: {idx}, Label: {label}")
    consistent_samples[label] = img

cam_extractor = GradCAM(model, target_layer="layer4")
os.makedirs("../gradcam_outputs", exist_ok=True)

def generate_and_save_cam(samples_dict, save_subdir, label_type):
    save_dir = os.path.join("..", "gradcam_outputs", save_subdir)
    os.makedirs(save_dir, exist_ok=True)

    for class_idx, img_tensor in samples_dict.items():
        input_tensor = img_tensor.unsqueeze(0).to(device)

        output = model(input_tensor)
        pred = output.argmax().item()

        # Grad-CAM for predicted class
        cam = cam_extractor(pred, output)[0].cpu()
        cam = F.interpolate(cam.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # Save raw heatmap as .npy
        np.save(os.path.join(save_dir, f"eo2_{label_type}_class_{class_idx}_cam.npy"), cam.squeeze(0).numpy())

        unnorm = transforms.Normalize(
            mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
            std=[1/s for s in [0.229, 0.224, 0.225]]
        )
        img_unnorm = unnorm(img_tensor).clamp(0, 1)  # Prevent out-of-range values
        original_img = to_pil_image(img_unnorm)

        heatmap_array = cam.squeeze(0).numpy()

        # Create side-by-side plot
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        # Left: original image
        axs[0].imshow(original_img)
        axs[0].set_title("Original")
        axs[0].axis('off')

        # Right: overlay
        axs[1].imshow(original_img)
        axs[1].imshow(heatmap_array, cmap='jet', alpha=0.5)
        axs[1].set_title("Grad-CAM")
        axs[1].axis('off')

        # Save side-by-side figure
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"eo2_{label_type}_class_{class_idx}_side_by_side.png"))
        plt.close()

generate_and_save_cam(consistent_samples, "correct", "correct")
#generate_and_save_cam(incorrect_samples, "incorrect", "incorrect")

print("Grad-CAM images saved to 'gradcam_outputs/'")
