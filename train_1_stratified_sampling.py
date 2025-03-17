import os
import shutil
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# Define the base directories
BASE_DIR_1 = "/mnt/shared-storage/yolov11L_Image_training_set_400/BT5_IMG_10K_infer_IT5/BT5_all/Training"
BASE_DIR_2 = "/mnt/shared-storage/yolov11L_Image_training_set_400"

# Define subdirectories dynamically
IMAGE_DIR = os.path.join(BASE_DIR_1, "final_images")
TXT_DIR = os.path.join(BASE_DIR_1, "final_txt")
TRAIN_IMG_DIR = os.path.join(BASE_DIR_2, "images/train")
TRAIN_TXT_DIR = os.path.join(BASE_DIR_2, "labels/train")
VAL_IMG_DIR = os.path.join(BASE_DIR_2, "images/val")
VAL_TXT_DIR = os.path.join(BASE_DIR_2, "labels/val")
TRAIN_CSV = os.path.join(BASE_DIR_2, "train.txt")
VAL_CSV = os.path.join(BASE_DIR_2, "val.txt")

# Create train/val directories if they don't exist
for directory in [TRAIN_IMG_DIR, TRAIN_TXT_DIR, VAL_IMG_DIR, VAL_TXT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Collect all txt annotation files and images
txt_files = [f for f in os.listdir(TXT_DIR) if f.endswith(".txt")]
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg", ".png"))]

# Print initial summary
print("📊 Initial Dataset Summary:")
print(f"📂 Total annotation files (TXT): {len(txt_files)}")
print(f"🖼 Total image files (JPG + PNG): {len(image_files)}")
print("🚀 Starting dataset processing...\n")

# Store label distributions
label_distribution = []
valid_txt_files = []
invalid_txt_files = []

# Step 1: Verify labels and collect valid txt files
for txt_file in txt_files:
    txt_path = os.path.join(TXT_DIR, txt_file)
    
    with open(txt_path, "r") as f:
        lines = f.readlines()
    
    if not lines:
        continue  # Skip empty annotation files

    # Collect classes from annotations
    labels = [int(line.split()[0]) for line in lines]

    # Check if all labels are within the range 0-4
    if all(0 <= label <= 4 for label in labels):  # CHANGE HERE IF LABELS CHANGE
        valid_txt_files.append(txt_file)
        label_distribution.extend(labels)
    else:
        invalid_txt_files.append(txt_file)

# Step 1.1: If all class labels are valid, print a success message
if not invalid_txt_files:
    print("✅ All annotation files have valid class labels (0-4).")
else:
    print("❌ Error: The following annotation files contain invalid class labels (not between 0-4):")
    for file in invalid_txt_files:
        print(file)
    exit(1)

# Step 2: Perform stratified sampling
# Convert file names to image names
valid_image_files = [f.replace(".txt", ".jpg") if os.path.exists(os.path.join(IMAGE_DIR, f.replace(".txt", ".jpg"))) 
                     else f.replace(".txt", ".png") for f in valid_txt_files]

# Map file names to their dominant label
image_to_label = {valid_image_files[i]: label_distribution[i] for i in range(len(valid_image_files))}

# Prepare dataset for stratified split
X = valid_image_files  # Image names
y = [image_to_label[f] for f in valid_image_files]  # Corresponding labels

# Stratified split (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Step 3: Copy files to corresponding directories
def copy_files(file_list, src_img_dir, src_txt_dir, dest_img_dir, dest_txt_dir):
    for file in file_list:
        txt_file = file.replace(".jpg", ".txt").replace(".png", ".txt")
        src_img_path = os.path.join(src_img_dir, file)
        src_txt_path = os.path.join(src_txt_dir, txt_file)
        
        dest_img_path = os.path.join(dest_img_dir, file)
        dest_txt_path = os.path.join(dest_txt_dir, txt_file)
        
        # Copy files if they exist
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dest_img_path)
        if os.path.exists(src_txt_path):
            shutil.copy(src_txt_path, dest_txt_path)

# Copy train files
copy_files(X_train, IMAGE_DIR, TXT_DIR, TRAIN_IMG_DIR, TRAIN_TXT_DIR)

# Copy validation files
copy_files(X_val, IMAGE_DIR, TXT_DIR, VAL_IMG_DIR, VAL_TXT_DIR)

# Step 4: Check for duplicate images between train and val
train_images = set(os.listdir(TRAIN_IMG_DIR))
val_images = set(os.listdir(VAL_IMG_DIR))
duplicates = train_images.intersection(val_images)

if duplicates:
    print(f"❌ Error: Found {len(duplicates)} duplicated images between train and val sets!")
    for dup in duplicates:
        print(dup)
    exit(1)
else:
    print("✅ No duplicate images found between train and validation sets.")

# Step 5: Create train.txt and val.txt with filenames of labels (with extensions)
train_txt_files = [f for f in os.listdir(TRAIN_TXT_DIR) if f.endswith(".txt")]
val_txt_files = [f for f in os.listdir(VAL_TXT_DIR) if f.endswith(".txt")]

# Save as CSV (each file name on a new line)
with open(TRAIN_CSV, "w") as f:
    for file in train_txt_files:
        f.write(file + "\n")

with open(VAL_CSV, "w") as f:
    for file in val_txt_files:
        f.write(file + "\n")

# Final report
print("\n✅ Dataset successfully split using stratified sampling!")
print(f"📊 Final Train Set: {len(X_train)} images")
print(f"📊 Final Validation Set: {len(X_val)} images")
print("✅ train.txt and val.txt created with label filenames (including extensions).")
