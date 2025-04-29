import pandas as pd
import shutil
import os

# ————————————————————————————————————————————————————————————————
def check_misclassifications(prediction_df, ground_truth_df,
                             iou_threshold=0.3, class_id=4):
    """
    Returns a dict mapping each confidence threshold (0.1…0.9)
    to the count of ground-truth boxes of `class_id` that were
    misclassified (i.e. GT=class_id, Pred≠class_id).
    """
    misclassified_counts = {}
    misclassified_annotations_per_conf = {}  # To store misclassified annotations for each confidence threshold

    for conf in [round(x * 0.1, 1) for x in range(1, 10)]:
        report, items = evaluate_model(
            prediction_df=prediction_df,
            ground_truth_df=ground_truth_df,
            conf_threshold=conf,
            iou_threshold=iou_threshold,
            class_id=None
        )
        # Filter misclassified annotations based on class_id
        misclassified_annotations = [
            m for m in items["misclassified_annotations"]
            if (m["gt_class_id"] == class_id and m["pred_class_id"] != class_id)
        ]
        misclassified_counts[conf] = len(misclassified_annotations)
        misclassified_annotations_per_conf[conf] = misclassified_annotations

    return misclassified_counts, misclassified_annotations_per_conf

# ————————————————————————————————————————————————————————————————
def copy_misclassified_images(conf_threshold, class_id, misclassified_annotations_per_conf, 
                              val_images_dir, target_dir):
    """
    Copies misclassified images for the given confidence threshold and class_id to the target directory.
    """
    misclassified_images = misclassified_annotations_per_conf.get(conf_threshold, [])
    
    # Check if the misclassified_images list is empty
    if not misclassified_images:
        print(f"No misclassified images found for threshold {conf_threshold} and class_id {class_id}.")
        return {}
    
    # Extract image file names from the misclassified annotations
    image_paths = [m['image_id'] + '.png' for m in misclassified_images]  # Assuming images are PNG format
    
    # If there are no valid image paths, return
    if not image_paths:
        print("No valid image paths found in misclassified annotations.")
        return {}
    
    # Get unique image paths (remove duplicates)
    unique_images = set(image_paths)
    duplicate_images = len(image_paths) - len(unique_images)
    
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Copy unique images to the target directory
    for image_id in unique_images:
        source_image_path = os.path.join(val_images_dir, image_id)
        target_image_path = os.path.join(target_dir, image_id)

        # Ensure the image exists in the source directory before copying
        if os.path.exists(source_image_path):
            shutil.copy(source_image_path, target_image_path)
    
    details = {
        "conf_threshold": conf_threshold,
        "class_id": class_id,
        "misclassified_annotations": len(misclassified_images),
        "unique_images_copied": len(unique_images),
        "duplicate_images": duplicate_images
    }
    
    return details

# ————————————————————————————————————————————————————————————————
# Example of using the functions with a prediction DataFrame and ground truth DataFrame
# Replace `model_pres_df` and `gt_df` with your actual DataFrames:
misclassified_counts, misclassified_annotations_per_conf = check_misclassifications(
    prediction_df=model_pres_df,
    ground_truth_df=gt_df,
    iou_threshold=0.3,
    class_id=4
)

# Set the directories for images
val_images_dir = "/Users/simrannaik/Desktop/Image_Yolo/model_check/data_val_images_txt/Training/images/val"
target_dir = "/Users/simrannaik/Desktop/Image_Yolo/model_check/misclassified_annotation_pred"

# Call the function with the desired confidence threshold (0.8) and class_id (4)
details = copy_misclassified_images(
    conf_threshold=0.8,
    class_id=4,
    misclassified_annotations_per_conf=misclassified_annotations_per_conf,
    val_images_dir=val_images_dir,
    target_dir=target_dir
)

# Print the result
print(details)
