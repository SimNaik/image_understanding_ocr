import os
import shutil
import cv2

def copy_misclassified_images_with_bbox(conf_threshold, class_id, misclassified_annotations_per_conf,
                                         val_images_dir, val_labels_dir, target_dir, gt_target_dir, draw_bbox=True, draw_gt=True):
    # Extract the misclassified annotations for the given confidence threshold
    misclassified_images = misclassified_annotations_per_conf.get(conf_threshold, [])
    
    # Extract image paths from the misclassified annotations
    image_paths = [m['image_id'] for m in misclassified_images]

    # Get unique image paths (remove duplicates)
    unique_images = set(image_paths)

    # Initialize counters for unique and duplicate images
    unique_image_count = 0
    duplicate_image_count = 0

    # Make sure the target directory exists
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(gt_target_dir, exist_ok=True)

    # Process each unique image
    for image_name in unique_images:
        # Find all misclassified annotations for the current image
        image_annotations = [m for m in misclassified_images if m['image_id'] == image_name]

        # Determine the image path (check for png or jpg)
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate_path = os.path.join(val_images_dir, f"{image_name}{ext}")
            if os.path.exists(candidate_path):
                image_path = candidate_path
                break

        if image_path:
            # If the image exists, copy it to the target directory
            target_image_path = os.path.join(target_dir, f"{image_name}.jpg")
            shutil.copy(image_path, target_image_path)
            
            # Copy image for ground truth bounding boxes
            gt_image_path = os.path.join(gt_target_dir, f"{image_name}.jpg")
            shutil.copy(image_path, gt_image_path)

            # If draw_bbox is True, draw bounding boxes for misclassified annotations
            if draw_bbox:
                img = cv2.imread(target_image_path)

                # Draw bounding boxes for each misclassification
                for annotation in image_annotations:
                    if annotation['gt_class_id'] == class_id and annotation['pred_class_id'] != class_id:
                        # Get the bounding box coordinates
                        gt_box = annotation['gt_box']
                        x_center = gt_box['x_center']
                        y_center = gt_box['y_center']
                        width = gt_box['width']
                        height = gt_box['height']

                        # Convert to pixel coordinates (assuming image size 1.0 for simplicity)
                        img_height, img_width, _ = img.shape
                        x_min = int((x_center - width / 2) * img_width)
                        y_min = int((y_center - height / 2) * img_height)
                        x_max = int((x_center + width / 2) * img_width)
                        y_max = int((y_center + height / 2) * img_height)

                        # Draw the rectangle on the image
                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                        # Add confidence and predicted class ID text to the bounding box
                        confidence = annotation['confidence']
                        pred_class_id = annotation['pred_class_id']
                        
                        text = f"Conf: {confidence:.2f}, Pred Class: {pred_class_id}"
                        
                        # Set the position of the text
                        text_position = (x_min, y_min - 10)  # Place text above the bounding box
                        
                        # Draw the text on the image
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img, text, text_position, font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

                # Save the image with misclassification bounding boxes and text
                cv2.imwrite(target_image_path, img)

            # If draw_gt is True, draw bounding boxes for ground truth annotations (class 4)
            if draw_gt:
                img_gt = cv2.imread(gt_image_path)

                # Read the corresponding label file for the current image
                label_file_path = os.path.join(val_labels_dir, f"{image_name}.txt")
                if os.path.exists(label_file_path):
                    with open(label_file_path, 'r') as label_file:
                        lines = label_file.readlines()

                    for line in lines:
                        parts = line.strip().split()
                        pred_class_id = int(parts[0])  # The class ID
                        x_center = float(parts[1])  # Normalized x_center
                        y_center = float(parts[2])  # Normalized y_center
                        width = float(parts[3])  # Normalized width
                        height = float(parts[4])  # Normalized height

                        # Only draw bounding boxes for class 4
                        if pred_class_id == 4:
                            # Convert normalized coordinates to pixels
                            img_height, img_width, _ = img_gt.shape
                            x_min = int((x_center - width / 2) * img_width)
                            y_min = int((y_center - height / 2) * img_height)
                            x_max = int((x_center + width / 2) * img_width)
                            y_max = int((y_center + height / 2) * img_height)

                            # Draw the rectangle on the image
                            cv2.rectangle(img_gt, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                            # Add class ID text to the bounding box
                            text = f"Class: {pred_class_id}"
                            
                            # Set the position of the text
                            text_position = (x_min, y_min - 10)  # Place text above the bounding box
                            
                            # Draw the text on the image
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(img_gt, text, text_position, font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

                # Save the image with ground truth bounding boxes and class IDs
                cv2.imwrite(gt_image_path, img_gt)

            # Increment the counter for unique images
            unique_image_count += 1
        else:
            # If the image is not found, count it as a duplicate
            duplicate_image_count += 1

    # Return details about the process
    return {
        'conf_threshold': conf_threshold,
        'class_id': class_id,
        'misclassified_annotations': len(misclassified_images),
        'unique_images_copied': unique_image_count,
        'duplicate_images': duplicate_image_count
    }


# Example usage:
val_images_dir = "/mnt/shared-storage/yolov11L_Image_training_set_400/BT7_IMG_11K+GB_INFER_IT6_8LA/Training/images/val"
val_labels_dir = "/mnt/shared-storage/yolov11L_Image_training_set_400/BT7_IMG_11K+GB_INFER_IT6_8LA/Training/labels/val"
target_dir = "/mnt/shared-storage/yolov11L_Image_training_set_400/BT5_IMG_10K_infer_IT5/BT5_all/Training/test_predictions_v2/report_oc/misclassified_annotation_pred"
gt_target_dir = "/mnt/shared-storage/yolov11L_Image_training_set_400/BT5_IMG_10K_infer_IT5/BT5_all/Training/test_predictions_v2/report_oc/misclassified_annotations_gt"

conf_threshold = float(input("Enter the confidence threshold (0.1 to 0.9): "))
class_id = int(input("Enter the class ID: "))

# Assuming `misclassified_annotations_per_conf` is the dictionary from the previous steps
details = copy_misclassified_images_with_bbox(
    conf_threshold=conf_threshold,
    class_id=class_id,
    misclassified_annotations_per_conf=misclassified_annotations_per_conf,
    val_images_dir=val_images_dir,
    val_labels_dir=val_labels_dir,
    target_dir=target_dir,
    gt_target_dir=gt_target_dir
)

# Print the details
print(details)
