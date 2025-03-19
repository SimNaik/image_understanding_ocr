import os
import shutil

# Ask user for class_id and confidence threshold
class_id = int(input("Enter Class ID: "))
conf_threshold = float(input("Enter Confidence Threshold (e.g., 0.8): "))

# Run evaluate_model
report, items = evaluate_model(
    prediction_df=model_pres_df,
    ground_truth_df=gt_df,
    conf_threshold=conf_threshold,
    iou_threshold=0.3,
    class_id=class_id
)

# Define paths
val_images_dir = "/mnt/shared-storage/yolov11L_Image_training_set_400/BT5_IMG_10K_infer_IT5/BT5_all/Training/images/val"
threshold_folder = f"/mnt/shared-storage/yolov11L_Image_training_set_400/BT5_IMG_10K_infer_IT5/BT5_all/report/{conf_threshold}"

# Define categories
categories = ["extra_predictions", "missed_predictions", "misclassified_annotations"]
missing_images_dict = {category: set() for category in categories}  # Track missing images using sets

# Function to get image name from item (handles multiple extensions)
def get_image_name(category, entry, valid_extensions=("jpg", "png", "jpeg")):
    if category == "extra_predictions":
        return entry['pred_box']['image_name']  # Already includes extension

    elif category == "missed_predictions":
        file_id = entry['gt_box']['file_id']
        
        # Check for the correct extension in val_images_dir
        for ext in valid_extensions:
            potential_filename = f"{file_id}.{ext}"
            if os.path.exists(os.path.join(val_images_dir, potential_filename)):
                return potential_filename
        
        # Default to PNG if no extension is found
        return f"{file_id}.png"

    return None

# Function to process each category
def copy_images(category):
    destination_dir = os.path.join(threshold_folder, category)
    os.makedirs(destination_dir, exist_ok=True)

    all_images = [get_image_name(category, entry) for entry in items.get(category, []) if get_image_name(category, entry)]
    unique_images = set(all_images)  # Remove duplicates within the category

    copied_count = 0

    for image_name in unique_images:
        source_path = os.path.join(val_images_dir, image_name)
        destination_path = os.path.join(destination_dir, image_name)

        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)  # Copy instead of move
            copied_count += 1
        else:
            missing_images_dict[category].add(image_name)  # Track missing images

    # Print summary for each category
    print(f"\nCategory: {category}")
    print(f"Total images (before removing duplicates): {len(all_images)}")
    print(f"Unique images copied: {copied_count}")
    print(f"Missing images: {len(missing_images_dict[category])}")

    if missing_images_dict[category]:
        print("\nMissing Images:")
        for img in missing_images_dict[category]:
            print(img)

# Process extra_predictions and missed_predictions
for category in ["extra_predictions", "missed_predictions"]:
    copy_images(category)

# Handle misclassified_annotations separately
misclassified_dir = os.path.join(threshold_folder, "misclassified_annotations")
os.makedirs(misclassified_dir, exist_ok=True)

misclassified_count = len(items.get("misclassified_annotations", []))

if misclassified_count == 0:
    # Create "no_images_found.txt" file if there are no misclassified images
    no_images_file = os.path.join(misclassified_dir, "no_images_found.txt")
    with open(no_images_file, "w") as f:
        f.write("No images found in misclassified_annotations.")

    print("\nCategory: misclassified_annotations")
    print("No images found. Created 'no_images_found.txt'.")
else:
    copy_images("misclassified_annotations")
