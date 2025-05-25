# train_equalized_odds.py

import torch
from torch.cuda.amp import autocast, GradScaler
from torcheval.metrics import Mean, MulticlassAccuracy
from tqdm import tqdm


def train_equalized_odds(model, dataloaders, criterion, optimizer, scheduler, age_group_mapping, device, epochs=30):
    scaler = GradScaler()
    best_val_acc = 0.0
    best_model_state = None

    accuracy_counter = MulticlassAccuracy(num_classes=7).to(device)
    loss_counter = Mean().to(device)
    val_acc_counter = MulticlassAccuracy(num_classes=7).to(device)

    for epoch in range(epochs):
        model.train()
        train_loader = tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{epochs}")
        loss_counter.reset()
        accuracy_counter.reset()

        for images, labels, _, age_group, _ in train_loader:
            age_group_indices = [age_group_mapping[age] for age in age_group]
            age_group_tensor = torch.tensor(age_group_indices, dtype=torch.long).to(device)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels, age_group_tensor)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            accuracy_counter.update(outputs, labels)
            loss_counter.update(loss.detach())

            train_loader.set_postfix(loss=loss_counter.compute().item(),
                                     acc=accuracy_counter.compute().item())

        print(f"\nTrain Loss: {loss_counter.compute():.4f}, Accuracy: {accuracy_counter.compute().item() * 100:.2f}%")

        # Validation
        model.eval()
        val_acc_counter.reset()

        with torch.no_grad():
            for images, labels, _, _, _ in dataloaders['val']:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_acc_counter.update(outputs, labels)

        val_acc = val_acc_counter.compute().item() * 100
        print(f"Validation Accuracy: {val_acc:.2f}%")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

    return best_model_state, best_val_acc
