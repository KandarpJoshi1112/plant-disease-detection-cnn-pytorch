import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from model import PlantDiseaseCNN
from dataset import get_dataloaders


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, class_names = get_dataloaders(batch_size=32)

    model = PlantDiseaseCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load("plant_disease_cnn.pth"))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n📊 Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix Shape:", cm.shape)


if __name__ == "__main__":
    evaluate()
