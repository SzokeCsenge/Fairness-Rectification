import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torcheval.metrics import MulticlassAccuracy, Mean

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=60):
    scaler = GradScaler()
    best_val_acc = 0.0
    best_model_state = None

    acc_train = MulticlassAccuracy(num_classes=7).to(device)
    acc_val = MulticlassAccuracy(num_classes=7).to(device)
    loss_train = Mean().to(device)
    loss_val = Mean().to(device)

    for epoch in range(num_epochs):
        model.train()
        loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        acc_train.reset()
        loss_train.reset()

        for images, labels, *_ in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc_train.update(outputs, labels)
            loss_train.update(loss.detach())

            loader.set_postfix(loss=loss_train.compute().item(), acc=acc_train.compute().item())

        print(f"Train Loss: {loss_train.compute():.4f}, Acc: {acc_train.compute().item()*100:.2f}%")

        model.eval()
        acc_val.reset()
        loss_val.reset()
        with torch.no_grad():
            for images, labels, *_ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_val.update(loss.detach())
                acc_val.update(outputs, labels)

        val_acc = acc_val.compute().item() * 100
        print(f"Val Acc: {val_acc:.2f}%, Val Loss: {loss_val.compute():.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

        scheduler.step()

    return best_model_state, best_val_acc
