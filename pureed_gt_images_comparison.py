import os
import shutil
import cv2

# Ask user for class_id and confidence threshold
class_id = int(input("Enter Class ID: "))
conf_threshold = float(input("Enter Confidence Threshold (e.g., 0.8): "))

# Class ID Mapping
type_mapping = {
    0: "Dg_Diag",
    1: "Hw_Diag",
    2: "Annotations",
    3: "Table",
    4: "oc"
}
class_label = type_mapping.get(class_id, f"Class_{class_id}")

# Run evaluate_model
report, items = evaluate_model(
    prediction_df=model_pres_df,
    ground_truth_df=gt_df,
    conf_threshold=conf_threshold,
    iou_threshold=0.3,
    class_id=class_id
)

# Define paths
val_images_dir = "/Users/simrannaik/Desktop/Image_Yolo/BT5_all/test_val/version1_val/val_images"
val_labels_dir = "/Users/simrannaik/Desktop/Image_Yolo/BT5_all/test_val/version1_val/val"
third_images_dir = "/Users/simrannaik/Desktop/Image_Yolo/BT5_all/test_val/test_predictions_y/predictions/unsorted_annotations_images_val_2k_BT5_v2"
threshold_folder = f"/Users/simrannaik/Desktop/Image_Yolo/BT5_all/test_val/report/{conf_threshold}_thres_{class_label}"

# Define categories
categories = ["extra_predictions", "missed_predictions", "misclassified_annotations"]
gt_categories = [f"{cat}_gt" for cat in categories]
missing_images_dict = {category: set() for category in categories + gt_categories}

# Function to get image name
def get_image_name(category, entry, valid_extensions=("jpg", "png", "jpeg")):
    if category == "extra_predictions":
        return entry['pred_box']['image_name']
    elif category == "missed_predictions":
        file_id = entry['gt_box']['file_id']
        for ext in valid_extensions:
            potential_filename = f"{file_id}.{ext}"
            if os.path.exists(os.path.join(val_images_dir, potential_filename)):
                return potential_filename
        return f"{file_id}.png"
    return None

# Function to draw bounding boxes on an image (only showing confidence for prediction images)
def draw_boxes(image_path, boxes, output_path, color=(0, 255, 0), confidences=None, labels=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return
    
    height, width, _ = image.shape
    font_color = (255, 0, 0)  # Dark Blue in BGR format
    font_scale = 1.0  # Increase font size
    font_thickness = 2  # Font thickness
    
    for i, box in enumerate(boxes):
        x_center, y_center, w, h = box
        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)

        # Draw the rectangle for the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Only display the confidence number (without the word "Conf:")
        if confidences:
            text = f"{confidences[i]:.2f}"  # Display only the confidence value
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add class label text for GT images (not used for prediction images)
        if labels:
            label = labels[i]
            cv2.putText(image, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
    
    # Save the resulting image with confidence (no labels)
    cv2.imwrite(output_path, image)

# Function to process predictions and ground truth images
def process_images(category, is_gt=False):
    source_images = val_images_dir
    source_labels = val_labels_dir
    destination_dir = os.path.join(threshold_folder, f"{category}_gt" if is_gt else category)
    os.makedirs(destination_dir, exist_ok=True)
    
    all_images = [get_image_name(category, entry) for entry in items.get(category, []) if get_image_name(category, entry)]
    unique_images = set(all_images)
    copied_count = 0
    
    for image_name in unique_images:
        source_path = os.path.join(source_images, image_name)
        destination_path = os.path.join(destination_dir, image_name)
        
        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
            copied_count += 1
            
            if is_gt:
                label_file = os.path.join(source_labels, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))
                if os.path.exists(label_file):
                    boxes = []
                    labels = []
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_id, x_center, y_center, w, h = map(float, parts)
                                boxes.append((x_center, y_center, w, h))
                                labels.append(type_mapping.get(int(class_id), "Unknown"))
                    if boxes:
                        # For GT, pass boxes and labels
                        draw_boxes(destination_path, boxes, destination_path, color=(0, 255, 0), confidences=None, labels=labels)
                else:
                    print(f"Missing label file: {label_file}")
            else:
                # For predictions, only pass boxes and confidences (no labels)
                pred_boxes = []
                confidences = []
                for entry in items.get(category, []):
                    if get_image_name(category, entry) == image_name:
                        pred_box = entry.get("pred_box")
                        if pred_box:
                            # Extract class_id and map to label (we won't use the label here)
                            class_id = pred_box.get("class_id", 0)
                            label = type_mapping.get(class_id, "Unknown")  # Not used here
                            x_center, y_center, w, h = pred_box["x_center"], pred_box["y_center"], pred_box["width"], pred_box["height"]
                            pred_boxes.append((x_center, y_center, w, h))
                            confidences.append(pred_box["confidence"])
                if pred_boxes:
                    # Only show confidence, no label (remove 'labels' argument)
                    draw_boxes(destination_path, pred_boxes, destination_path, color=(0, 0, 255), confidences=confidences, labels=None)
        else:
            missing_images_dict[category].add(image_name)
    
    print(f"\nCategory: {category} ({'GT' if is_gt else 'Predictions'})")
    print(f"Total images: {len(all_images)}")
    print(f"Copied: {copied_count}")
    print(f"Missing: {len(missing_images_dict[category])}")
    if missing_images_dict[category]:
        print("\nMissing Images:")
        for img in missing_images_dict[category]:
            print(img)


# New function to stitch images together for GT, Predictions, and Third Image
def stitch_images_with_third(category):
    gt_dir = os.path.join(threshold_folder, f"{category}_gt")
    pred_dir = os.path.join(threshold_folder, category)
    third_dir = third_images_dir  # Directory with the third image
    combined_dir = os.path.join(threshold_folder, f"{category}_all")
    os.makedirs(combined_dir, exist_ok=True)

    gt_images = set(os.listdir(gt_dir))
    pred_images = set(os.listdir(pred_dir))
    
    missing_images = gt_images.symmetric_difference(pred_images)
    
    for image_name in gt_images.intersection(pred_images):
        gt_image_path = os.path.join(gt_dir, image_name)
        pred_image_path = os.path.join(pred_dir, image_name)
        third_image_path = os.path.join(third_dir, image_name)

        # Read images
        gt_image = cv2.imread(gt_image_path)
        pred_image = cv2.imread(pred_image_path)
        third_image = cv2.imread(third_image_path)

        if gt_image is None or pred_image is None or third_image is None:
            print(f"Error loading one of the images: {image_name}")
            continue

        # Combine images side by side (GT + Prediction + Third image)
        combined_image = cv2.hconcat([gt_image, pred_image, third_image])
        
        # Save the combined image
        combined_image_path = os.path.join(combined_dir, image_name)
        cv2.imwrite(combined_image_path, combined_image)
    
    if missing_images:
        with open(os.path.join(combined_dir, "missing_images.txt"), "w") as f:
            f.write("\n".join(missing_images))

# Process all categories and add third image stitching
for category in categories:
    process_images(category)
    process_images(category, is_gt=True)

    # Add stitching for combined images with the third image
    stitch_images_with_third(category)

# Handle misclassified_annotations separately
misclassified_dir = os.path.join(threshold_folder, "misclassified_annotations")
os.makedirs(misclassified_dir, exist_ok=True)
if not items.get("misclassified_annotations", []):
    with open(os.path.join(misclassified_dir, "no_images_found.txt"), "w") as f:
        f.write("No images found in misclassified_annotations.")
    print("\nCategory: misclassified_annotations")
    print("No images found. Created 'no_images_found.txt'.")
else:
    process_images("misclassified_annotations")
    process_images("misclassified_annotations", is_gt=True)
    stitch_images_with_third("misclassified_annotations")
