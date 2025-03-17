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

# CSV and Documentation Paths
IMAGES_FOLDER = os.path.join(BASE_DIR_2, "images")
os.makedirs(IMAGES_FOLDER, exist_ok=True)
TRAIN_CSV = os.path.join(IMAGES_FOLDER, "train.csv")
VAL_CSV = os.path.join(IMAGES_FOLDER, "val.csv")
DOC_FILE = os.path.join(IMAGES_FOLDER, "documentation.txt")

# Ensure train/val directories exist
for directory in [TRAIN_IMG_DIR, TRAIN_TXT_DIR, VAL_IMG_DIR, VAL_TXT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Open documentation.txt to store logs
with open(DOC_FILE, "w") as doc_file:

    # Collect all images and annotations
    txt_files = [f for f in os.listdir(TXT_DIR) if f.endswith(".txt")]
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg", ".png"))]

    # Save initial dataset summary
    summary = (
        f"\n📊 Initial Dataset Summary:\n"
        f"📂 Total annotation files (TXT): {len(txt_files)}\n"
        f"🖼 Total image files (JPG + PNG): {len(image_files)}\n"
        f"🚀 Starting dataset processing...\n"
    )
    print(summary)
    doc_file.write(summary)

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

        labels = [int(line.split()[0]) for line in lines]

        if all(0 <= label <= 4 for label in labels):
            valid_txt_files.append(txt_file)
            label_distribution.extend(labels)
        else:
            invalid_txt_files.append(txt_file)

    # If labels are valid, print success message
    if not invalid_txt_files:
        validation_msg = "✅ All annotation files have valid class labels (0-4).\n"
        print(validation_msg)
        doc_file.write(validation_msg)
    else:
        error_msg = "❌ Error: Invalid class labels found:\n"
        print(error_msg)
        doc_file.write(error_msg)
        for file in invalid_txt_files:
            print(file)
            doc_file.write(file + "\n")
        exit(1)

    # **Class Distribution Before Splitting**
    class_counts_before = Counter(label_distribution)
    class_distribution_msg = "\n📊 Class Distribution Before Splitting:\n"
    for class_id, count in sorted(class_counts_before.items()):
        class_distribution_msg += f"🔹 Class {class_id}: {count} instances\n"
    print(class_distribution_msg)
    doc_file.write(class_distribution_msg)

    # Step 2: Perform stratified sampling
    valid_image_files = [f.replace(".txt", ".jpg") if os.path.exists(os.path.join(IMAGE_DIR, f.replace(".txt", ".jpg"))) 
                        else f.replace(".txt", ".png") for f in valid_txt_files]

    image_to_label = {valid_image_files[i]: label_distribution[i] for i in range(len(valid_image_files))}
    X = valid_image_files
    y = [image_to_label[f] for f in valid_image_files]

    # Stratified split (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # **Class Distribution After Splitting**
    class_distribution_after_msg = "\n📊 Class Distribution After Splitting:\n"
    train_class_counts = Counter(y_train)
    val_class_counts = Counter(y_val)

    class_distribution_after_msg += "🔹 Training Set:\n"
    for class_id, count in sorted(train_class_counts.items()):
        class_distribution_after_msg += f"   🟢 Class {class_id}: {count} instances\n"

    class_distribution_after_msg += "🔹 Validation Set:\n"
    for class_id, count in sorted(val_class_counts.items()):
        class_distribution_after_msg += f"   🔵 Class {class_id}: {count} instances\n"

    print(class_distribution_after_msg)
    doc_file.write(class_distribution_after_msg)

    # **Train/Val Split Ratio Confirmation**
    train_percentage = (len(X_train) / len(y)) * 100
    val_percentage = (len(X_val) / len(y)) * 100
    split_ratio_msg = f"\n📊 Train/Val Split Ratio: {train_percentage:.2f}% Train | {val_percentage:.2f}% Val\n"
    print(split_ratio_msg)
    doc_file.write(split_ratio_msg)

    # **Copy Images and Labels to Train/Val Folders**
    def copy_files(file_list, src_img_dir, src_txt_dir, dest_img_dir, dest_txt_dir):
        for file in file_list:
            txt_file = file.replace(".jpg", ".txt").replace(".png", ".txt")
            src_img_path = os.path.join(src_img_dir, file)
            src_txt_path = os.path.join(src_txt_dir, txt_file)
            
            dest_img_path = os.path.join(dest_img_dir, file)
            dest_txt_path = os.path.join(dest_txt_dir, txt_file)
            
            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, dest_img_path)
            if os.path.exists(src_txt_path):
                shutil.copy(src_txt_path, dest_txt_path)

    # Copy train and validation files
    copy_files(X_train, IMAGE_DIR, TXT_DIR, TRAIN_IMG_DIR, TRAIN_TXT_DIR)
    copy_files(X_val, IMAGE_DIR, TXT_DIR, VAL_IMG_DIR, VAL_TXT_DIR)

    # **Save CSV Files for Train & Val**
    pd.DataFrame(X_train, columns=["filename"]).to_csv(TRAIN_CSV, index=False)
    pd.DataFrame(X_val, columns=["filename"]).to_csv(VAL_CSV, index=False)

    # Final report
    final_msg = (
        "\n✅ Dataset successfully split using stratified sampling!\n"
        f"📊 Final Train Set: {len(X_train)} images\n"
        f"📊 Final Validation Set: {len(X_val)} images\n"
        "✅ train.csv and val.csv saved with image filenames.\n"
    )
    print(final_msg)
    doc_file.write(final_msg)
