import os
import random
import shutil

DATA_DIR = "data"
TRAIN_IMG = os.path.join(DATA_DIR, "images", "train")
VAL_IMG = os.path.join(DATA_DIR, "images", "val")
TRAIN_LAB = os.path.join(DATA_DIR, "labels", "train")
VAL_LAB = os.path.join(DATA_DIR, "labels", "val")

os.makedirs(VAL_IMG, exist_ok=True)
os.makedirs(VAL_LAB, exist_ok=True)

images = [f for f in os.listdir(TRAIN_IMG) if f.lower().endswith(('.png','.jpg','.jpeg'))]

# 20% en validation
val_count = max(1, int(len(images) * 0.2))
val_images = random.sample(images, val_count)

for img in val_images:
    label = img.replace(".jpg", ".txt").replace(".png", ".txt")

    shutil.move(os.path.join(TRAIN_IMG, img), os.path.join(VAL_IMG, img))
    shutil.move(os.path.join(TRAIN_LAB, label), os.path.join(VAL_LAB, label))

print("✔ Split dataset terminé !")
print("Images validation :", len(val_images))
