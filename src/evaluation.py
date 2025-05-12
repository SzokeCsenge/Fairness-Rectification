import pandas as pd
import torch
from torcheval.metrics import MulticlassAccuracy

@torch.no_grad()
def evaluate_model(model, dataloader, device, save_path="test_results.csv"):
    model.eval()
    accuracy_counter = MulticlassAccuracy(num_classes=7).to(device)
    results = []

    with torch.no_grad():
        for images, labels, age, age_group, gender in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            accuracy_counter.update(outputs, labels)

            results.append({
                "Prediction": predicted.cpu().numpy(),
                "Label": labels.cpu().numpy(),
                "Age_Group": age_group,
                "Gender": gender
            })

    accuracy = accuracy_counter.compute().item() * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    df = pd.DataFrame([{
        "Prediction": p, "Label": l, "Age_Group": ag, "Gender": g
    } for batch in results for p, l, ag, g in zip(*[batch[k] for k in ["Prediction", "Label", "Age_Group", "Gender"]])])

    df.to_csv(save_path, index=False)
    return accuracy
