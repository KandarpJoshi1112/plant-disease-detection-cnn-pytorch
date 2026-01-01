import os
import sys

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Ensure project root is on the path so `src` imports resolve when run directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import PlantDiseaseCNN
from src.dataset import get_dataloaders


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use num_workers=0 to avoid Windows multiprocessing guard errors
    _, _, test_loader, class_names = get_dataloaders(batch_size=32, num_workers=0)

    model = PlantDiseaseCNN(num_classes=len(class_names))
    state_dict = torch.load(
        "plant_disease_cnn.pth", map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu())
            all_labels.extend(labels.cpu())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("experiments/confusion_matrix.png")

    print("✅ Confusion matrix saved")


if __name__ == "__main__":
    main()
