import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# Define base directories
BASE_DIR_1 = "/mnt/shared-storage/yolov11L_Image_training_set_400/BT5_IMG_10K_infer_IT5/BT5_all/Training"
BASE_DIR_2 = "/mnt/shared-storage/yolov11L_Image_training_set_400"

# Define subdirectories dynamically
IMAGE_DIR = os.path.join(BASE_DIR_1, "final_images")
TXT_DIR = os.path.join(BASE_DIR_1, "final_txt")
TRAIN_IMG_DIR = os.path.join(BASE_DIR_2, "images/train")
TRAIN_TXT_DIR = os.path.join(BASE_DIR_2, "labels/train")
VAL_IMG_DIR = os.path.join(BASE_DIR_2, "images/val")
VAL_TXT_DIR = os.path.join(BASE_DIR_2, "labels/val")

# Ensure output directories exist
for directory in [TRAIN_IMG_DIR, TRAIN_TXT_DIR, VAL_IMG_DIR, VAL_TXT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Logging setup
LOG_FILE = os.path.join(BASE_DIR_2, "images", "documentation.txt")
def log_message(message):
    print(message)
    with open(LOG_FILE, "a") as log:
        log.write(message + "\n")

log_message("🚀 Starting YOLO dataset processing...")

# Collect image and annotation files
txt_files = [f for f in os.listdir(TXT_DIR) if f.endswith(".txt")]
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg", ".png"))]

log_message(f"📂 Found {len(txt_files)} annotation files and {len(image_files)} image files.")

# Step 1: Validate annotation files
valid_txt_files = []
valid_img_files = []
invalid_txt_files = []
missing_image_files = []
missing_txt_files = []
duplicate_txt_files = set()
file_label_mapping = {}

for txt_file in txt_files:
    txt_path = os.path.join(TXT_DIR, txt_file)
    img_file_jpg = txt_file.replace(".txt", ".jpg")
    img_file_png = txt_file.replace(".txt", ".png")
    
    if img_file_jpg in image_files:
        valid_img_files.append(img_file_jpg)
    elif img_file_png in image_files:
        valid_img_files.append(img_file_png)
    else:
        missing_image_files.append(txt_file)
        continue  # Skip if image is missing

    with open(txt_path, "r") as f:
        lines = f.readlines()
    
    labels = [int(line.split()[0]) for line in lines] if lines else []
    
    if not labels:  # Include empty txt files as background images
        valid_txt_files.append(txt_file)
        file_label_mapping[txt_file] = -1  # Assign -1 for background-only images
        continue
    
    if all(0 <= label <= 4 for label in labels):
        if txt_file in valid_txt_files:
            duplicate_txt_files.add(txt_file)
        valid_txt_files.append(txt_file)
        most_frequent_label = max(set(labels), key=labels.count)  # Assign most frequent label to the file
        file_label_mapping[txt_file] = most_frequent_label
    else:
        invalid_txt_files.append(txt_file)

# Check for missing txt files
for img_file in image_files:
    txt_file = img_file.replace(".jpg", ".txt").replace(".png", ".txt")
    if txt_file not in txt_files:
        missing_txt_files.append(txt_file)
        valid_txt_files.append(txt_file)  # Include images without annotations
        file_label_mapping[txt_file] = -1  # Background-only images

if not invalid_txt_files:
    log_message("✅ All class IDs in annotation files are valid.")
else:
    log_message("❌ Some annotation files contain invalid class IDs.")
    log_message("\n".join(invalid_txt_files))

if not duplicate_txt_files:
    log_message("✅ No duplicate annotation files detected.")
else:
    log_message("⚠️ Duplicate annotation files detected:")
    log_message("\n".join(duplicate_txt_files))

log_message("✅ All annotations verified. Proceeding with dataset split...")

# 📊 Class Distribution Before Splitting
class_counts_before = Counter(file_label_mapping.values())
log_message("\n📊 Class Distribution Before Splitting:")
for class_id, count in sorted(class_counts_before.items()):
    log_message(f"🔹 Class {class_id}: {count} instances")

# Step 2: Perform stratified 80-20 split
X = list(file_label_mapping.keys())  # Annotation file names
y = list(file_label_mapping.values())  # Corresponding labels (1 per file)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 📊 Class Distribution After Splitting
train_class_counts = Counter(y_train)
val_class_counts = Counter(y_val)
log_message("\n📊 Class Distribution After Splitting:")
log_message("🔹 Training Set:")
for class_id, count in sorted(train_class_counts.items()):
    log_message(f"   🟢 Class {class_id}: {count} instances")
log_message("🔹 Validation Set:")
for class_id, count in sorted(val_class_counts.items()):
    log_message(f"   🔵 Class {class_id}: {count} instances")

log_message(f"📊 Train Set: {len(X_train)} images | Validation Set: {len(X_val)} images")

# Copy images and labels to their respective train/val directories
def copy_files(file_list, src_img_dir, src_txt_dir, dest_img_dir, dest_txt_dir):
    for file in file_list:
        txt_file = file.replace(".jpg", ".txt").replace(".png", ".txt")
        img_file = file.replace(".txt", ".jpg") if file.replace(".txt", ".jpg") in image_files else file.replace(".txt", ".png")
        
        img_src = os.path.join(src_img_dir, img_file)
        txt_src = os.path.join(src_txt_dir, txt_file)
        img_dest = os.path.join(dest_img_dir, img_file)
        txt_dest = os.path.join(dest_txt_dir, txt_file)
        
        if os.path.exists(img_src):
            shutil.copy(img_src, img_dest)
        if os.path.exists(txt_src):
            shutil.copy(txt_src, txt_dest)

copy_files(X_train, IMAGE_DIR, TXT_DIR, TRAIN_IMG_DIR, TRAIN_TXT_DIR)
copy_files(X_val, IMAGE_DIR, TXT_DIR, VAL_IMG_DIR, VAL_TXT_DIR)

# Save filenames to CSV
train_csv_path = os.path.join(BASE_DIR_2, "images", "train.csv")
val_csv_path = os.path.join(BASE_DIR_2, "images", "val.csv")
pd.DataFrame(X_train, columns=["filename"]).to_csv(train_csv_path, index=False)
pd.DataFrame(X_val, columns=["filename"]).to_csv(val_csv_path, index=False)

log_message("✅ Dataset split completed successfully! Images and annotations copied, and CSV files saved.")
