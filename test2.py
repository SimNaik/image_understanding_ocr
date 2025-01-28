import os
import shutil
import csv
import ast  # To safely evaluate tensor strings

# File paths and directories
csv_file = '/mnt/shared-storage/yolov11L_Image_training_set_400/test_predictions/predictions/annotations.csv'
images_dir = '/mnt/shared-storage/yolov11L_Image_training_set_400/BT4_IMG_400+1.8+1.8+1.8+2K_2Kinfer_IT4/testing/diagrams_oc_neg_2284'
output_dir = '/mnt/shared-storage/yolov11L_Image_training_set_400/test_predictions/predictions'
top_1000_images_dir = os.path.join(output_dir, "top_2000_images_labelimg")
top_1000_annotations_dir_with_conf = os.path.join(output_dir, "top_2000_annotations_with_conf")
top_1000_annotations_dir_without_conf = os.path.join(output_dir, "top_2000_annotations_without_conf")

# Create necessary directories
os.makedirs(top_1000_images_dir, exist_ok=True)
os.makedirs(top_1000_annotations_dir_with_conf, exist_ok=True)
os.makedirs(top_1000_annotations_dir_without_conf, exist_ok=True)

# Function to calculate the product of confidence scores for bounding boxes with confidence >= 0.2
def calculate_confidence_product(confidences):
    product = 1
    for confidence in confidences:
        if confidence >= 0.2:
            product *= confidence
    return product

# Step 1: Read data from the CSV file and prepare image confidence list
image_confidence_list = []

with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header if present
    current_image = None
    current_confidences = []
    rows_for_image = []

    for row in reader:
        image_name = row[0]
        
        if image_name != current_image and current_image is not None:
            # Calculate confidence product for the previous image
            product = calculate_confidence_product(current_confidences)
            image_confidence_list.append((current_image, rows_for_image, product))

            # Reset for the next image
            current_confidences = []
            rows_for_image = []

        current_image = image_name
        if row[2]:  # If there's a confidence score, process the row
            class_id = row[1]
            
            # Extract confidence value safely
            try:
                confidence_str = row[2]
                if 'tensor' in confidence_str:
                    confidence = float(ast.literal_eval(confidence_str).item())
                else:
                    confidence = float(confidence_str)
            except ValueError:
                print(f"Skipping invalid confidence: {row[2]}")
                continue
            
            x_center = row[3]
            y_center = row[4]
            width = row[5]
            height = row[6]
            current_confidences.append(confidence)
            rows_for_image.append((class_id, confidence, x_center, y_center, width, height))

    # Handle the last image in the CSV file
    if current_image:
        product = calculate_confidence_product(current_confidences)
        image_confidence_list.append((current_image, rows_for_image, product))

# Step 2: Sort the images by their confidence product in ascending order
sorted_images = sorted(image_confidence_list, key=lambda x: x[2])

# Step 3: Process and save only the top 1000 images and their annotations
for i, (image_name, annotations, product) in enumerate(sorted_images[:2000], start=1):
    # Copy the top 1000 images
    src_image_path = os.path.join(images_dir, image_name)
    dst_image_path = os.path.join(top_1000_images_dir, image_name)
    if os.path.exists(src_image_path):  # Ensure image exists before copying
        shutil.copy(src_image_path, dst_image_path)

    # Write annotations with confidence scores
    annotation_with_conf_path = os.path.join(top_1000_annotations_dir_with_conf, os.path.splitext(image_name)[0] + '.txt')
    with open(annotation_with_conf_path, 'w') as f_with_conf:
        for class_id, confidence, x_center, y_center, width, height in annotations:
            f_with_conf.write(f"{class_id} {confidence} {x_center} {y_center} {width} {height}\n")

    # Write annotations without confidence scores
    annotation_without_conf_path = os.path.join(top_1000_annotations_dir_without_conf, os.path.splitext(image_name)[0] + '.txt')
    with open(annotation_without_conf_path, 'w') as f_without_conf:
        for class_id, confidence, x_center, y_center, width, height in annotations:
            f_without_conf.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

print(f"Top 1000 images have been copied to {top_1000_images_dir}")
print(f"Annotations with confidence saved in {top_1000_annotations_dir_with_conf}")
print(f"Annotations without confidence saved in {top_1000_annotations_dir_without_conf}")
