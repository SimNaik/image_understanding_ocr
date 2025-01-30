import os
import shutil
import cv2
import matplotlib.pyplot as plt
import csv
from ultralytics import YOLO
import torch
import numpy as np

# Check if CUDA (NVIDIA GPU) is available and set the device accordingly
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load the YOLO model with the best-trained weights
model = YOLO('/mnt/shared-storage/yolov11L_Image_training_set_400/B2_img_400+1.8kT_6244/1819kinfer_IT2/Training/logs/Yolo_Organic_Model/weights/best.pt').to(device)

# Define the directory containing the images to be tested
image_dir = '/mnt/shared-storage/yolov11L_Image_training_set_400/BT3_img_400+1.8+1.8t_4284infer_IT3/training/images/val'

# Define the directory where the predictions will be saved
output_dir = '/mnt/shared-storage/yolov11L_Image_training_set_400/test_predictions/predictions'
os.makedirs(output_dir, exist_ok=True)

# Create subdirectories for unsorted images and annotations
unsorted_images_dir = os.path.join(output_dir, "unsorted_images_val_819")
unsorted_annotations_dir = os.path.join(output_dir, "annotations_val_819")
os.makedirs(unsorted_images_dir, exist_ok=True)
os.makedirs(unsorted_annotations_dir, exist_ok=True)

# CSV file to store image names, class labels, confidence, and YOLO coordinates
csv_file = os.path.join(output_dir, 'annotations_val_819.csv')
csv_header = ['image_name', 'class_ID', 'confidence', 'x_center', 'y_center', 'width', 'height']
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)

# Function to save annotations in YOLO format, with confidence score after class ID and before coordinates
def save_annotations(result, save_path, include_confidence=True):
    with open(save_path, 'w') as f:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])  # Get class ID
                x_center, y_center, width, height = box.xywhn[0]  # Get normalized bbox coordinates
                if include_confidence:
                    confidence = box.conf[0]  # Get confidence score
                    f.write(f"{class_id} {confidence:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                else:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Get a list of image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

# Define batch size
batch_size = 32

# Open CSV file for appending annotations
with open(csv_file, 'a', newline='') as f:
    writer = csv.writer(f)
    
    # Process images in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_paths = [os.path.join(image_dir, f) for f in batch_files]

        print(f"Currently processing batch {i // batch_size + 1}/{len(image_files) // batch_size + 1}")

        # Use torch.no_grad() to disable gradient calculation during inference
        with torch.no_grad():
            # Run inference on the batch of images
            #results = model(batch_paths, device=device)
            results = model(batch_paths, device=device, iou=0.1, conf=0.3, imgsz=960)
            

            # Iterate through the results for each image in the batch
            for result, image_file in zip(results, batch_files):
                # Save all processed images to the unsorted directory
                unsorted_image_path = os.path.join(unsorted_images_dir, image_file)
                result_plotted = result.plot(line_width=2, font_size=0.5)  # Adjusting the line width and font size
                plt.imsave(unsorted_image_path, result_plotted)

                # Save annotations for the current image
                unsorted_annotation_file = os.path.join(unsorted_annotations_dir, os.path.splitext(image_file)[0] + ".txt")
                save_annotations(result, unsorted_annotation_file, include_confidence=True)

                # Write annotations to CSV
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract detection details
                        class_id = int(box.cls[0])
                        label = model.names[class_id]  # Get class label
                        confidence = box.conf[0].item()
                        x_center = box.xywhn[0][0].item()
                        y_center = box.xywhn[0][1].item()
                        width = box.xywhn[0][2].item()
                        height = box.xywhn[0][3].item()

                        # Write a new row for each detection
                        writer.writerow([image_file, class_id, confidence, x_center, y_center, width, height])

print(f"Images have been processed and saved to {unsorted_images_dir}")
print(f"Annotations have been saved in {unsorted_annotations_dir}")
print(f"CSV file with image details has been saved at {csv_file}")
