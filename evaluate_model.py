import pandas as pd

def evaluate_model(prediction_df, ground_truth_df,conf_threshold=0.1, iou_threshold=0.5, class_id=None):
    """
    Evaluate the model's predictions using IoU.

    Parameters:
        prediction_df (pd.DataFrame): DataFrame containing predicted bounding boxes.
        ground_truth_df (pd.DataFrame): DataFrame containing ground truth bounding boxes.
        iou_threshold (float): IoU threshold to determine a match.
        class_id (int, optional): If specified, evaluates only for a particular class.

    Returns:
        dict: Summary of evaluation metrics.
    """
    if conf_threshold is not None:
        prediction_df = prediction_df[prediction_df['confidence'] >= conf_threshold]
    if class_id is not None:
        prediction_df = prediction_df[prediction_df['class_ID'] == class_id]
        ground_truth_df = ground_truth_df[ground_truth_df['class_ID'] == class_id]

    # Group predictions and ground truths by image_name
    pred_dict = {image_name.split('.')[0]: img_df for image_name, img_df in prediction_df.groupby('image_name')}
    gt_dict = {image_name: img_df for image_name, img_df in ground_truth_df.groupby('file_id')}

    item_ids = set(gt_dict.keys()).union(set(ground_truth_df.keys()))  # Get all unique image IDs
    good_matches = 0
    total_images = len(item_ids)
    total_annotations = len(ground_truth_df)

    good_match_list = []  # Matches found with IoU > threshold
    extra_predictions = []  # Predictions that do not match any ground truth
    missed_predictions = []  # Ground truth annotations without corresponding predictions
    misclassified_annotations = []  # Annotations with incorrect class predictions

    for image_id in item_ids:
        pred_df = pred_dict.get(image_id, pd.DataFrame(columns=prediction_df.columns))
        gt_df = gt_dict.get(image_id, pd.DataFrame(columns=ground_truth_df.columns))

        # Convert DataFrame rows to dictionaries for easy processing
        pred_box_list = pred_df.to_dict('records')
        gt_box_list = gt_df.to_dict('records')

        # Track assigned matches
        gt_assigned_idx = set()
        pred_assigned_idx = set()

        for gt_idx, gt_box in enumerate(gt_box_list):
            best_match = None
            best_iou = 0.0
            best_pred_idx = None

            for pred_idx, pred_box in enumerate(pred_box_list):
                iou = compute_iou(gt_box, pred_box)

                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_match = pred_box
                    best_pred_idx = pred_idx

            if best_match:
                good_matches += 1
                gt_assigned_idx.add(gt_idx)
                pred_assigned_idx.add(best_pred_idx)

                good_match_list.append({
                    "image_id": image_id,
                    "class_id": gt_box["class_ID"],
                    "confidence": best_match["confidence"],
                    "pred_box": best_match,
                    "gt_box": gt_box
                })

                # Check for misclassification
                if best_match["class_ID"] != gt_box["class_ID"]:
                    misclassified_annotations.append({
                        "image_id": image_id,
                        "gt_class_id": gt_box["class_ID"],
                        "pred_class_id": best_match["class_ID"],
                        "confidence": best_match["confidence"],
                        "gt_box": gt_box
                    })

        # Process unmatched ground truth boxes (missed detections)
        for gt_idx, gt_box in enumerate(gt_box_list):
            if gt_idx not in gt_assigned_idx:
                missed_predictions.append({
                    "image_id": image_id,
                    "class_id": gt_box["class_ID"],
                    "gt_box": gt_box
                })

        # Process unmatched predictions (extra detections)
        for pred_idx, pred_box in enumerate(pred_box_list):
            if pred_idx not in pred_assigned_idx:
                extra_predictions.append({
                    "image_id": image_id,
                    "class_id": pred_box["class_ID"],
                    "confidence": pred_box["confidence"],
                    "pred_box": pred_box
                })

    # Compute final statistics
    precision = good_matches / (good_matches + len(extra_predictions)) if (good_matches + len(extra_predictions)) > 0 else 0
    recall = good_matches / total_annotations if total_annotations > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    report = {
       
        "total_images": total_images,
        "total_annotations": total_annotations,
        "good_matches": good_matches,
        "missed_predictions": len(missed_predictions),
        "extra_predictions": len(extra_predictions),
        "misclassified_annotations": len(misclassified_annotations),
        "conf_threshold": conf_threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }
    items = {
        "good_matches": good_match_list,
        "extra_predictions": extra_predictions,
        "missed_predictions": missed_predictions,
        "misclassified_annotations": misclassified_annotations
    }
        

    return report, items
