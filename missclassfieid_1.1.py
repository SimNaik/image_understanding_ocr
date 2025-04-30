import pandas as pd
import numpy as np

# ————————————————————————————————————————————————————————————————
def check_misclassifications(prediction_df, ground_truth_df,
                             iou_threshold=0.3, class_id=4):
    """
    Returns a dict mapping each confidence threshold (0.1…0.9)
    to the count of ground-truth boxes of `class_id` that were
    misclassified (i.e. GT=class_id, Pred≠class_id).
    """
    misclassified_counts = {}
    for conf in [round(x * 0.1, 1) for x in range(1, 10)]:
        report, items = evaluate_model(
            prediction_df=prediction_df,
            ground_truth_df=ground_truth_df,
            conf_threshold=conf,
            iou_threshold=iou_threshold,
            class_id=None
        )
        # count only those misclassifications for our class of interest
        bad = [
            m for m in items["misclassified_annotations"]
            if (m["gt_class_id"] == class_id and m["pred_class_id"] != class_id)
        ]
        misclassified_counts[conf] = len(bad)
    return misclassified_counts

# ————————————————————————————————————————————————————————————————
def check_missed_extra_predictions(prediction_df, ground_truth_df,
                                   iou_threshold=0.3, class_id=4):
    """
    Returns a list of dicts, one per confidence threshold,
    containing total_images, good_matches, missed/extra predictions, etc.
    """
    out = []
    for conf in np.arange(0.1, 1.0, 0.1):
        report, _ = evaluate_model(
            prediction_df=prediction_df,
            ground_truth_df=ground_truth_df,
            conf_threshold=conf,
            iou_threshold=iou_threshold,
            class_id=class_id
        )
        entry = {
            "conf_threshold": round(conf, 1),
            "total_images":      report["total_images"],
            "total_annotations": report["total_annotations"],
            "good_matches":      report["good_matches"],
            "missed_predictions":report["missed_predictions"],
            "extra_predictions": report["extra_predictions"],
            "precision":         report["precision"],
            "recall":            report["recall"],
            "f1_score":          report["f1_score"],
            # this is the full list—will drop before merge
            "misclassified_annotations": report["misclassified_annotations"],
        }
        out.append(entry)
    return out

# ————————————————————————————————————————————————————————————————
# Replace `model_pres_df` and `gt_df` with your actual DataFrames:
misclassified_counts = check_misclassifications(
    prediction_df=model_pres_df,
    ground_truth_df=gt_df,
    iou_threshold=0.3,
    class_id=4
)

missed_extra_counts = check_missed_extra_predictions(
    prediction_df=model_pres_df,
    ground_truth_df=gt_df,
    iou_threshold=0.3,
    class_id=4
)

# ————————————————————————————————————————————————————————————————
# Build DataFrames
missed_extra_df = pd.DataFrame(missed_extra_counts).drop(
    columns=["misclassified_annotations"]
)

misclassified_df = (
    pd.DataFrame.from_dict(
        misclassified_counts,
        orient="index",
        columns=["misclassified_annotations"]
    )
    .reset_index()
    .rename(columns={"index": "conf_threshold"})
)

# ————————————————————————————————————————————————————————————————
# Merge & aggregate
final_df = (
    pd.merge(
        missed_extra_df,
        misclassified_df,
        on="conf_threshold",
        how="outer"
    )
    .fillna(0)
    .groupby("conf_threshold", as_index=False)
    .agg({
        "misclassified_annotations": "max",
        "total_images":              "first",
        "total_annotations":         "first",
        "good_matches":              "first",
        "missed_predictions":        "first",
        "extra_predictions":         "first",
        "precision":                 "first",
        "recall":                    "first",
        "f1_score":                  "first",
    })
    .loc[:, [
        "misclassified_annotations", "conf_threshold",
        "total_images", "total_annotations", "good_matches",
        "missed_predictions", "extra_predictions",
        "precision", "recall", "f1_score"
    ]]
)

# Reorder columns so that 'misclassified_annotations' comes before 'conf_threshold'
final_df = final_df[[  'total_images', 'total_annotations', 'good_matches', 
                     'missed_predictions', 'extra_predictions', 'misclassified_annotations','conf_threshold','precision', 'recall', 'f1_score']]

# Optionally, print the final DataFrame to see the result
pd.DataFrame(final_df)
