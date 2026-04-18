import torch
from src.datasets.corruption_engine import apply_corruption_batch



def train_model(model, train_loader, optimizer, criterion, device, isMLP):
    model.train()

    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # MLP handling
        if isMLP:
            images = images.view(images.size(0), -1) # (B, 3*32*32) for MLP, we need to flatten the input

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)



def evaluate_model(model, loader, device, isMLP, corruption_type=None,severity=0):
    model.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Apply corruption ONLY at evaluation
            if corruption_type is not None:
                images = apply_corruption_batch(images.cpu(), corruption_type, severity).to(device)

            # MLP handling
            if isMLP:
                images = images.view(images.size(0), -1)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total




