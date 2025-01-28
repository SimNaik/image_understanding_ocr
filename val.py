import os
from ultralytics import YOLO

def evaluate_yolo_model(model_path, path_to_validation_data, output_dir):
    """
    Evaluate a YOLOv8 model on a validation dataset, generate PR curves,
    and compute mAP scores.

    :param model_path: str - Path to the trained YOLO .pt model file.
    :param path_to_validation_data: str - Path to the YOLO-format data.yaml for validation.
    :param output_dir: str - Directory to store output plots, JSON, metrics, etc.
    :return: dict containing evaluation metrics such as mAP, precision, recall, etc.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load the model
    model = YOLO(model_path)

    # 2. Run validation
    #    - 'plots=True' saves plots, including PR curves per class, confusion matrix, etc.
    #    - 'save=True' saves predictions JSON (coco format) and label predictions in the output folder.
    results = model.val(
        data=path_to_validation_data,
        project=output_dir,
        name="val_results",
        save=True,   # save predictions
        plots=True   # generate and save PR curves
    )

    # 3. Retrieve overall metrics from the validation results
    # 'results' is usually a list or an object containing info on metrics per class and overall stats.
    # The exact structure may change slightly with library updates, so check with `print(results)`
    # or the official docs to see what attributes are available.

    # The below is a typical structure in YOLOv8. 
    # You can also inspect:
    #   results.box.map   - mAP
    #   results.box.maps  - array of per-class AP
    #   results.speed     - dictionary of inference times
    #   results.metrics   - dictionary with precision, recall, etc.
    try:
        metrics_summary = {
            "model_path": model_path,
            "val_map50": results.box.map,        # mAP at IoU=0.5 (overall)
            "val_map50_95": results.box.map50_95,  # mAP at IoU=0.5:0.95 (COCO standard)
            "precision": results.box.pr,         # overall precision
            "recall": results.box.re,            # overall recall
            "class_wise_ap": results.box.maps,   # array of per-class AP
        }
    except AttributeError:
        # If library changes cause differences in attribute naming, fallback might be needed
        # or you can debug by printing the 'results' structure.
        metrics_summary = {
            "warning": "Unexpected results structure. Please check results object."
        }

    # 4. Return the metrics so the user can handle them programmatically
    return metrics_summary


if __name__ == "__main__":
    # EXAMPLE USAGE:
    model_path_example = "/mnt/shared-storage/yolov11L_Image_training_set_400/B2_img_400+1.8kT_6244/1819kinfer_IT2/Training/logs/Yolo_Organic_Model/weights/best.pt"
    data_path_example = "/mnt/shared-storage/yolov11L_Image_training_set_400/dataset.yaml"
    output_dir_example = "/mnt/shared-storage/yolov11L_Image_training_set_400/pr_curves_check"

    metrics = evaluate_yolo_model(model_path_example, data_path_example, output_dir_example)
    print("Evaluation Metrics:", metrics)
