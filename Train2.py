from ultralytics import YOLO
import torch
from tensorboard import program
import yaml
import os

# Define the directories
#dir_1 = '/mnt/shared-storage/yolov11L_Image_training_set_400/B1_img_400_1.8k_IT1/Testing/test_predictions/diagrams_1,878k_inference'
#dir_2 = '/mnt/shared-storage/yolov11L_Image_training_set_400/training_set_400/Images'

# Get a list of files in each directory
#files_dir_1 = set(os.listdir(dir_1))
#files_dir_2 = set(os.listdir(dir_2))

# Find common files between the two directories (duplicates)
#duplicates = files_dir_1.intersection(files_dir_2)

#if duplicates:
   # print(f"Duplicate files found: {duplicates}")
#else:
    #print("No duplicates found between the two directories.")


# Check if CUDA (NVIDIA GPU) is available and set the device accordingly
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    print(f"Using device: {device} (CUDA)")
else:
    print("CUDA not available, using CPU")

# Load parameters from the args.yaml file
yaml_file_path = "/mnt/shared-storage/yolov11L_Image_training_set_400/args.yaml"
try:
    with open(yaml_file_path, 'r') as file:
        yaml_args = yaml.safe_load(file)
except FileNotFoundError:
    print(f"YAML file not found: {yaml_file_path}")
    exit(1)
except yaml.YAMLError as e:
    print(f"Error loading YAML file: {e}")
    exit(1)

# Define default training arguments (you can change values here if you want to set custom defaults)
train_args = {
    'epochs': 500,
    'batch': 32,
    'imgsz': 640, #changed
    'iou': 0.3, 
    'project': '/mnt/shared-storage/yolov11L_Image_training_set_400/logs',
    'name': 'Yolo_Organic_Model',
    'patience': 100,
    'workers': 7,
    'model': '/mnt/shared-storage/yolov11L_Image_training_set_400/BT3_img_400+1.8+1.8t_4284infer_IT3/training/logs/Yolo_Organic_Model/weights/best.pt',
    'amp': True,
    'erasing': 0.0,
    'shear': 15,
    'scale': 0.5,
    'translate': 0.0,
    'degrees': 180,
    'data': '/mnt/shared-storage/yolov11L_Image_training_set_400/dataset.yaml'
}

# Merge values: train_args takes priority
final_args = {key: train_args.get(key, yaml_args.get(key)) for key in set(train_args) | set(yaml_args)}

# Explicitly setting device
final_args['device'] = device

# Print final_args to verify correct values
print("Final training arguments:")
for key, value in final_args.items():
    print(f"{key}: {value}")

# Load a pretrained YOLO model
model = YOLO(final_args['model']).to(device)

# Train the model with the prepared arguments
results = model.train(**final_args)

# Print a message after training completes
print("Model has finished training!")

# Print training results
print("Training Results:")
print(results)

# Save the trained model
model.save("/mnt/shared-storage/yolov11L_Image_training_set_400/saved_model.pt")

# Evaluate the model's performance on the validation set
# Validate the model and save predictions
results = model.val(save=True, device=device, name="validation_results" )

# Print evaluation results
print("Evaluation Results:")
print(results)
print("Validation complete. Predictions saved to 'runs/val/exp/labels/'.")

# Launch TensorBoard
logdir = final_args['project']
if not os.path.exists(logdir):
    os.makedirs(logdir)

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', logdir])
url = tb.launch()
print(f"TensorBoard is running at {url}") 
