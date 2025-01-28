from ultralytics import YOLO
import torch
import yaml
import os
from tensorboard import program

# Check if CUDA (NVIDIA GPU) is available and set the device accordingly
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load the dataset configuration
yaml_file_path = "/mnt/shared-storage/yolov11L_Image_training_set_400/dataset.yaml"
try:
    with open(yaml_file_path, 'r') as file:
        yaml_args = yaml.safe_load(file)
except FileNotFoundError:
    print(f"YAML file not found: {yaml_file_path}")
    exit(1)
except yaml.YAMLError as e:
    print(f"Error loading YAML file: {e}")
    exit(1)

# Define the path to the pre-trained model weights
model_path = "/mnt/shared-storage/yolov11L_Image_training_set_400/B2_img_400+1.8kT_6244/1819kinfer_IT2/Training/logs/Yolo_Organic_Model/weights/best.pt"

# Load the YOLO model
model = YOLO(model_path).to(device)

# Define custom directory for saving validation results
custom_save_dir = "/mnt/shared-storage/yolov11L_Image_training_set_400/B2_img_400+1.8kT_6244/1819kinfer_IT2/Training/logs"

# Run validation and save results to the custom directory
results = model.val(
    data="/mnt/shared-storage/yolov11L_Image_training_set_400/dataset.yaml",
    save=True,
    device=device,
    project=custom_save_dir,  # Custom directory for saving results
    name="validation_results"  # Name of the folder for this validation run
)

# Save evaluation results to a text file
with open(f"{results.save_dir}/results.txt", "w") as f:
    f.write("Evaluation Results:\n")
    f.write(str(results))
print("Results saved to results.txt")


# Print evaluation results
print("Evaluation Results:")
print(results)

# Validation results saved in the custom directory
print(f"Validation complete. Predictions saved in: {results.save_dir}")


