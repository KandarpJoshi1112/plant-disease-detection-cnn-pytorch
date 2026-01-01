import torch
from model import PlantDiseaseCNN

model = PlantDiseaseCNN(num_classes=38)

dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)

print("Output shape:", output.shape)
