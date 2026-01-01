import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm

from model import PlantDiseaseCNN
from dataset import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def main():
    # ---------------- CONFIG ----------------
    BATCH_SIZE = 32
    EPOCHS = 15
    LR = 1e-3
    MODEL_PATH = "plant_disease_cnn.pth"
    HISTORY_PATH = "experiments/history.json"

    os.makedirs("experiments", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------- DATA ----------------
    train_loader, val_loader, _, class_names = get_dataloaders(
        batch_size=BATCH_SIZE
    )

    # ---------------- MODEL ----------------
    model = PlantDiseaseCNN(num_classes=len(class_names))
    model.to(device)

    # ---------------- TRAIN SETUP ----------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ---------------- HISTORY ----------------
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}\n"
            f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}"
        )

    # ---------------- SAVE MODEL ----------------
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")

    # ---------------- SAVE HISTORY ----------------
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs
    }

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f)

    print(f"✅ Training history saved to {HISTORY_PATH}")


if __name__ == "__main__":
    main()
