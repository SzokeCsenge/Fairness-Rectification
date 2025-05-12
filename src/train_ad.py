import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torcheval.metrics import Mean, MulticlassAccuracy
from tqdm import tqdm
import pandas as pd
from config import AGE_GROUP_DICT


def train_adversarial(model, train_dataloader, val_dataloader, device, epochs=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.007)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    criterion_task = nn.CrossEntropyLoss()
    criterion_adv = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        task_accuracy = MulticlassAccuracy(num_classes=7).to(device)
        adv_accuracy = MulticlassAccuracy(num_classes=5).to(device)
        task_loss_mean = Mean().to(device)
        adv_loss_mean = Mean().to(device)

        train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, labels, _, age_groups, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            age_group_tensor = torch.tensor([AGE_GROUP_DICT[age] for age in age_groups],
                                            dtype=torch.long).to(device)

            optimizer.zero_grad()
            with autocast():
                task_outputs, adv_outputs = model(images)
                task_loss = criterion_task(task_outputs, labels)
                adv_loss = criterion_adv(adv_outputs, age_group_tensor)
                loss = task_loss + adv_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            task_loss_mean.update(task_loss.detach())
            adv_loss_mean.update(adv_loss.detach())
            task_accuracy.update(task_outputs, labels)
            adv_accuracy.update(adv_outputs, age_group_tensor)

            train_loader.set_postfix(task_loss=task_loss.item(),
                                     adv_loss=adv_loss.item(),
                                     task_acc=task_accuracy.compute().item() * 100,
                                     adv_acc=adv_accuracy.compute().item() * 100)

        print(f"""
        Train - Task Loss: {task_loss_mean.compute():.4f}, Adv Loss: {adv_loss_mean.compute():.4f}
        Task Acc: {task_accuracy.compute().item() * 100:.2f}%, Adv Acc: {adv_accuracy.compute().item() * 100:.2f}%
        """)

        # Validation phase
        model.eval()
        val_task_acc = MulticlassAccuracy(num_classes=7).to(device)
        val_adv_acc = MulticlassAccuracy(num_classes=5).to(device)
        val_task_loss = Mean().to(device)
        val_adv_loss = Mean().to(device)

        val_loader = tqdm(val_dataloader, desc="Validation")
        with torch.no_grad():
            for images, labels, _, age_groups, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                age_group_tensor = torch.tensor([AGE_GROUP_DICT[age] for age in age_groups],
                                                dtype=torch.long).to(device)

                with autocast():
                    task_outputs, adv_outputs = model(images)
                    t_loss = criterion_task(task_outputs, labels)
                    a_loss = criterion_adv(adv_outputs, age_group_tensor)

                val_task_loss.update(t_loss.detach())
                val_adv_loss.update(a_loss.detach())
                val_task_acc.update(task_outputs, labels)
                val_adv_acc.update(adv_outputs, age_group_tensor)

        print(f"""
        Eval - Task Loss: {val_task_loss.compute():.4f}, Adv Loss: {val_adv_loss.compute():.4f}
        Task Acc: {val_task_acc.compute().item() * 100:.2f}%, Adv Acc: {val_adv_acc.compute().item() * 100:.2f}%
        """)
        val_acc = val_task_acc.compute().item() * 100

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            print("New best model found!")

        scheduler.step()

    return best_model_state, best_val_acc


def test_adversarial(model, test_dataloader, device, save_path="adversarial_results.csv"):
    model.eval()
    criterion_task = nn.CrossEntropyLoss()
    criterion_adv = nn.CrossEntropyLoss()

    task_acc = MulticlassAccuracy(num_classes=7).to(device)
    adv_acc = MulticlassAccuracy(num_classes=5).to(device)
    task_loss_mean = Mean().to(device)
    adv_loss_mean = Mean().to(device)
    results = []

    test_loader = tqdm(test_dataloader, desc="Testing")
    with torch.no_grad():
        for images, labels, _, age_groups, gender in test_loader:
            images, labels = images.to(device), labels.to(device)
            age_group_tensor = torch.tensor([AGE_GROUP_DICT[age] for age in age_groups],
                                            dtype=torch.long).to(device)

            with autocast():
                task_outputs, adv_outputs = model(images)
                task_loss = criterion_task(task_outputs, labels)
                adv_loss = criterion_adv(adv_outputs, age_group_tensor)

            task_loss_mean.update(task_loss.detach())
            adv_loss_mean.update(adv_loss.detach())
            task_acc.update(task_outputs, labels)
            adv_acc.update(adv_outputs, age_group_tensor)

            _, predicted = torch.max(task_outputs, dim=1)
            results.extend(zip(predicted.cpu().numpy(),
                               labels.cpu().numpy(),
                               age_groups,
                               gender))

    print(f"""
    Test - Task Loss: {task_loss_mean.compute():.4f}, Adv Loss: {adv_loss_mean.compute():.4f}
    Task Acc: {task_acc.compute().item() * 100:.2f}%, Adv Acc: {adv_acc.compute().item() * 100:.2f}%
    """)
    final_task_acc = task_acc.compute().item() * 100
    final_adv_acc = adv_acc.compute().item() * 100

    df_results = pd.DataFrame(results, columns=["Prediction", "Label", "Age_Group", "Gender"])
    df_results.to_csv(save_path, index=False)

    return final_task_acc, final_adv_acc
