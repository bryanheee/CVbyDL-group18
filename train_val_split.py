import os
import shutil

import cv2
from sklearn.model_selection import train_test_split

target_resolution = (960, 540)


def resize_and_save(image_path, save_path, target_resolution):
    image = cv2.imread(image_path)
    if image.shape[:2] != target_resolution:
        image_resized = cv2.resize(image, target_resolution)
        cv2.imwrite(save_path, image_resized)
    else:
        shutil.copy(image_path, save_path)


dir = "DATASETS_LOW_RES"

# Paths
dataset_path = "archive/Warp-D/train/images"
labels_path = "archive/Warp-D/train/labels"

new_train_path = dir + "/train/images"
new_val_path = dir + "/val/images"
new_train_labels_path = dir + "/train/labels"
new_val_labels_path = dir + "/val/labels"
new_test_path = dir + "/test/images"
new_test_labels_path = dir + "/test/labels"

# Create directories for new train and val sets
os.makedirs(new_train_path, exist_ok=True)
os.makedirs(new_val_path, exist_ok=True)
os.makedirs(new_test_path, exist_ok=True)

os.makedirs(new_train_labels_path, exist_ok=True)
os.makedirs(new_val_labels_path, exist_ok=True)
os.makedirs(new_test_labels_path, exist_ok=True)

# Get list of all image files
image_files = [
    f
    for f in os.listdir(dataset_path)
    if os.path.isfile(os.path.join(dataset_path, f))
]

# Split the dataset
train_files, val_files = train_test_split(
    image_files, test_size=0.2, random_state=42
)

print(f"Training set size: {len(train_files)}")
# Copy training files
for file in train_files:
    resize_and_save(
        os.path.join(dataset_path, file),
        os.path.join(new_train_path, file),
        target_resolution,
    )
    label_file = file.replace(".jpg", ".txt").replace(".png", ".txt")
    shutil.copy(
        os.path.join(labels_path, label_file),
        os.path.join(new_train_labels_path, label_file),
    )

print(f"Validation set size: {len(val_files)}")
# Copy validation files
for file in val_files:
    resize_and_save(
        os.path.join(dataset_path, file),
        os.path.join(new_val_path, file),
        target_resolution,
    )
    label_file = file.replace(".jpg", ".txt").replace(".png", ".txt")
    shutil.copy(
        os.path.join(labels_path, label_file),
        os.path.join(new_val_labels_path, label_file),
    )

test_files = [
    f
    for f in os.listdir("archive/Warp-D/test/images")
    if os.path.isfile(os.path.join("archive/Warp-D/test/images", f))
]

print(f"Test set size: {len(test_files)}")
for file in test_files:
    resize_and_save(
        os.path.join("archive/Warp-D/test/images", file),
        os.path.join(new_test_path, file),
        target_resolution,
    )
    label_file = file.replace(".jpg", ".txt").replace(".png", ".txt")
    shutil.copy(
        os.path.join("archive/Warp-D/test/labels", label_file),
        os.path.join(new_test_labels_path, label_file),
    )
