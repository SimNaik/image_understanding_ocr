from ultralytics import YOLO
import torch
from tensorboard import program
import yaml
import os

#  Check if CUDA (NVIDIA GPU) is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#  Load parameters from YAML file
yaml_file_path = "/mnt/shared-storage/yolov11L_Image_training_set_400/args.yaml"
try:
    with open(yaml_file_path, 'r') as file:
        yaml_args = yaml.safe_load(file) or {}  # Ensure it’s a dictionary
except FileNotFoundError:
    print(f" YAML file not found: {yaml_file_path}")
    exit(1)
except yaml.YAMLError as e:
    print(f" Error loading YAML file: {e}")
    exit(1)

# Define training arguments with custom defaults (these will take priority)
train_args = {
    'epochs': 500,
    'batch': 42,  # Can go up to 42
    'imgsz': 640,
    'iou': 0.3, 
    'project': '/mnt/shared-storage/yolov11L_Image_training_set_400/logs',
    'name': 'Yolo_Organic_Model',
    'patience': 100,
    'workers': 7,
    'model': '/mnt/shared-storage/yolov11L_Image_training_set_400/yolo12l.pt',
    'amp': True,
    'erasing': 0.0,
    'shear': 15,
    'scale': 0.5,
    'translate': 0.0,
    'degrees': 180,
    'data': '/mnt/shared-storage/yolov11L_Image_training_set_400/dataset.yaml',
}

#  Merge YAML args first, then override with `train_args`
final_args = {**yaml_args, **train_args}  # `train_args` has priority!
final_args['device'] = device  # Explicitly set device

#  Print final arguments to verify
print("\n Final training arguments (train_args overrides yaml_args):")
for key, value in final_args.items():
    print(f"{key}: {value}")

#  Load YOLO model with correct weights
model = YOLO(final_args['model'])  # No need for .to(device), handled internally by YOLO

#  Train the model
train_results = model.train(overrides=final_args)  # Pass all settings at once

#  Print a message after training completes
print("\n Model training complete!")

#  Automatically find best.pt from the training folder
best_model_path = os.path.join(final_args['project'], final_args['name'], 'weights', 'best.pt')

if not os.path.exists(best_model_path):
    print(f" Best model not found at {best_model_path}. Skipping validation.")
else:
    print(f" Best model located at {best_model_path}. Running validation...")
    
    # Validate the trained model
    val_results = model.val(save=True, device=device)  # Removed unnecessary args

    # Print evaluation results
    print("\n Evaluation Results:")
    print(val_results)
    print(f" Validation complete. Predictions saved in {final_args['project']}/validation_results/.")

#  Launch TensorBoard
logdir = final_args['project']
if not os.path.exists(logdir):
    os.makedirs(logdir)

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', logdir])
url = tb.launch()
print(f" TensorBoard is running at {url}")
