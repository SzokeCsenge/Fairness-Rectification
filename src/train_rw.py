import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torcheval.metrics import MulticlassAccuracy, Mean

def train_reweighted(model, dataloaders, criterion, optimizer, scheduler, weights_dict, device, epochs=30):
    scaler = GradScaler()
    train_loader, val_loader = dataloaders
    top_acc = 0
    best_state = None

    loss_counter = Mean().to(device)
    acc_counter = MulticlassAccuracy(num_classes=7).to(device)
    val_loss = Mean().to(device)
    val_acc = MulticlassAccuracy(num_classes=7).to(device)

    for epoch in range(epochs):
        model.train()

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        loss_counter.reset()
        acc_counter.reset()
        for images, labels, _, age_group, _ in loop:
            images, labels = images.to(device), labels.to(device)
            batch_weights = [weights_dict[(label.item(), age)] for label, age in zip(labels, age_group)]
            batch_weights = torch.tensor(batch_weights).float().to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels, batch_weights)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc_counter.update(outputs, labels)
            loss_counter.update(loss)
            loop.set_postfix(loss=loss_counter.compute().item(), acc=acc_counter.compute().item())

        print(f"Train Loss: {loss_counter.compute():.4f}, Acc: {acc_counter.compute().item()*100:.2f}%")

        # Validation
        model.eval()
        val_loss.reset()
        val_acc.reset()
        with torch.no_grad():
            for images, labels, _, _, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                neutral_weights = torch.ones_like(labels, dtype=torch.float32, device=device)
                loss = criterion(outputs, labels, neutral_weights)
                val_loss.update(loss)
                val_acc.update(outputs, labels)

        acc = val_acc.compute().item() * 100
        print(f"Val Loss: {val_loss.compute():.4f}, Val Acc: {acc:.2f}%")

        if acc > top_acc:
            top_acc = acc
            best_state = model.state_dict()
            torch.save(best_state, f"weight_model_{top_acc:.2f}.pth")
            print("New best model saved!")

        scheduler.step()

    return best_state, top_acc
