import argparse
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

# Ensure project root is on sys.path so local imports resolve
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import PlantDiseaseCNN
from src.dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# num_workers=0 to avoid multiprocessing spawn guard issues on Windows for quick predictions
_, _, _, class_names = get_dataloaders(num_workers=0)

model = PlantDiseaseCNN(num_classes=len(class_names))
state_dict = torch.load("plant_disease_cnn.pth", map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)

    return class_names[pred.item()]


def main():
    parser = argparse.ArgumentParser(description="Predict plant disease from an image")
    parser.add_argument("image", help="Path to the image to classify")
    args = parser.parse_args()

    prediction = predict(args.image)
    print(prediction)


if __name__ == "__main__":
    main()
