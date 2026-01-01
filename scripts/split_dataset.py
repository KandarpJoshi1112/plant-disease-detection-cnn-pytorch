import os
import shutil
import random
from tqdm import tqdm

RAW_DIR = "data/raw/PlantVillage/plantvillage dataset/color"
OUT_DIR = "data/processed"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

random.seed(42)

def split_class(class_name):
    class_path = os.path.join(RAW_DIR, class_name)
    images = os.listdir(class_path)
    random.shuffle(images)

    n = len(images)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, imgs in splits.items():
        split_dir = os.path.join(OUT_DIR, split, class_name)
        os.makedirs(split_dir, exist_ok=True)

        for img in imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_dir, img)
            shutil.copy2(src, dst)

if __name__ == "__main__":
    classes = os.listdir(RAW_DIR)

    for cls in tqdm(classes, desc="Splitting dataset"):
        split_class(cls)

    print("✅ Dataset split completed!")
