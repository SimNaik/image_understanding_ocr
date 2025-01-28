import os
import random
import shutil
import csv

# Source directories for images and annotations
image_dir = '/mnt/shared-storage/yolov11L_Image_training_set_400/final_images'
if they don't exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Set the split ratio (e.g., 20% of the data for validation)
split_ratio = 0.2

# List all images in the dataset (assuming they are .jpg or .png files)
all_images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Shuffle the list of images
random.shuffle(all_images)

# Calculate the number of validation samples
val_count = int(len(all_images) * split_ratio)

# Split the images into validation and training sets
val_images = set(all_images[:val_count])
train_images = set(all_images[val_count:])

# Check for duplicates in split lists
duplicates = val_images.intersection(train_images)
if duplicates:
    raise ValueError(f"Duplicate images found: {duplicates}")

# Function to copy files
def copy_files(file_list, source_dir, target_dir, label_source_dir, label_target_dir):
    for file_name in file_list:
        # Copy the image
        try:
            shutil.copy2(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))
        except FileNotFoundError:
            print(f"Image file {file_name} not found in source directory.")
            continue
        
        # Copy the corresponding label (assume .txt extension)
        label_name = file_name.replace('.png', '.txt').replace('.jpg', '.txt')
        label_path = os.path.join(label_source_dir, label_name)
        
        if os.path.exists(label_path):
            try:
                shutil.copy2(label_path, os.path.join(label_target_dir, label_name))
            except FileNotFoundError:
                print(f"Label file {label_name} not found in source directory.")
        else:
            print(f"Label file {label_name} not found for image {file_name}.")

# Copy training files
copy_files(train_images, image_dir, train_image_dir, label_dir, train_label_dir)

# Copy validation files
copy_files(val_images, image_dir, val_image_dir, label_dir, val_label_dir)

# Function to create a CSV file with filenames
def create_csv(file_list, csv_path):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['filename'])
        # Write the filenames
        for file_name in file_list:
            writer.writerow([file_name])

# Paths for the train and validation CSV files
train_csv_path = os.path.join(os.path.dirname(train_image_dir), 'train.csv')
val_csv_path = os.path.join(os.path.dirname(val_image_dir), 'val.csv')

# Create train.csv
create_csv(train_images, train_csv_path)

# Create val.csv
create_csv(val_images, val_csv_path)

# Function to check for duplicates in directories
def check_for_duplicates(dir1, dir2):
    files_in_dir1 = set(os.listdir(dir1))
    files_in_dir2 = set(os.listdir(dir2))
    duplicates = files_in_dir1.intersection(files_in_dir2)
    return duplicates

# Check for duplicates in image directories
image_duplicates = check_for_duplicates(train_image_dir, val_image_dir)
if image_duplicates:
    print(f"Duplicate images found between train and val directories: {image_duplicates}")
else:
    print("No duplicate images found between train and val directories.")

# Check for duplicates in label directories
label_duplicates = check_for_duplicates(train_label_dir, val_label_dir)
if label_duplicates:
    print(f"Duplicate labels found between train and val directories: {label_duplicates}")
else:
    print("No duplicate labels found between train and val directories.")

# Print the number of files in each directory after the copy
print(f"Training images: {len(os.listdir(train_image_dir))}")
print(f"Validation images: {len(os.listdir(val_image_dir))}")
print(f"Training labels: {len(os.listdir(train_label_dir))}")
print(f"Validation labels: {len(os.listdir(val_label_dir))}")

# Print completion message for CSV generation
print(f"CSV files created:\n- Train CSV: {train_csv_path}\n- Validation CSV: {val_csv_path}")
