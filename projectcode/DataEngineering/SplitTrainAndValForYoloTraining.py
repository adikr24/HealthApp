import os
import random
import shutil
from pathlib import Path

# === CONFIGURATION ===
#dataset_root = '/app/mediaFiles/images/YoloTrainingImages/yolotrainingReadyImages/YoloIterations/iterationI'
## location of yolo trained images and labels
dataset_root = '/app/mediaFiles/images/YoloTrainingImages/ImagesYoloTrainedOnAlready'
images_dir = os.path.join(dataset_root, "images")
labels_dir = os.path.join(dataset_root, "labels")

output_dir = os.path.join(dataset_root, "split_dataset")
train_ratio = 0.8  # 80% train, 20% val
seed = 42

# === OUTPUT STRUCTURE ===
for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

# === COLLECT FILES ===
image_exts = [".jpg", ".jpeg", ".png"]
image_files = [f for f in os.listdir(images_dir) if Path(f).suffix.lower() in image_exts]

random.seed(seed)
random.shuffle(image_files)

split_idx = int(len(image_files) * train_ratio)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

def move_pairs(file_list, split_type):
    for img_name in file_list:
        label_name = Path(img_name).stem + ".txt"
        src_img = os.path.join(images_dir, img_name)
        src_label = os.path.join(labels_dir, label_name)
        dst_img = os.path.join(output_dir, f"images/{split_type}", img_name)
        dst_label = os.path.join(output_dir, f"labels/{split_type}", label_name)

        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)

print(f"Total images found: {len(image_files)}")
print(f"Training: {len(train_files)} | Validation: {len(val_files)}")

move_pairs(train_files, "train")
move_pairs(val_files, "val")

print(f"✅ Dataset split complete. Saved under:\n{output_dir}")
