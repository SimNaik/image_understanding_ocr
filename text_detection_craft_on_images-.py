import os
import shutil
from pathlib import Path
import torch
from craft_text_detector import load_craftnet_model
from .craft_helper import process_image_and_detect_text
from src.utils import download_image
import config

# Initialize model
def initialize_model():
    try:
        craft_net = load_craftnet_model(cuda=torch.cuda.is_available())
        # Warm-up the model
        process_image_and_detect_text(image="warmup_image.jpg", model=craft_net)
        print("Warmup done")

        return craft_net
    except Exception as e:
        raise Exception(f"Failed to initialize Text Detection model: {str(e)}")


text_detection_model = initialize_model()

# Inference on a single image
def model_inference(input_data, model=text_detection_model):
    try:
        # Extract input parameters
        image_url = input_data.get("image_url")
        image = download_image(image_url)

        # Model parameters
        text_threshold = input_data.get("text_threshold", 0.3)
        low_text = input_data.get("low_text", 0.2)
        long_size = input_data.get("long_size", 720)
        link_threshold = 0.1

        # Process image using craft helper
        boxes, confs = process_image_and_detect_text(
            image=image,
            model=model,
            text_threshold=text_threshold,
            link_threshold=link_threshold,
            low_text=low_text,
            long_size=long_size,
        )

        # If there are no text regions detected
        if len(boxes) == 0:
            return False  # No text detected

        # Prepare response for text-detected images
        response = {
            "image_source": image_url,
            "text_regions": [
                {"confidence": conf, "bbox": box.tolist()}
                for box, conf in zip(boxes, confs)
            ],
        }

        return True  # Text detected

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return False  # Return False if an error occurs

# Process all images in a folder
def process_images_in_folder(input_folder, text_folder, no_text_folder):
    # Ensure destination folders exist
    os.makedirs(text_folder, exist_ok=True)
    os.makedirs(no_text_folder, exist_ok=True)

    # Loop through all images in the input folder
    for image_filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_filename)

        if os.path.isfile(image_path):
            try:
                print(f"Processing {image_filename}...")
                # Prepare input data
                input_data = {"image_url": image_path}

                # Run model inference
                has_text = model_inference(input_data)

                # Copy image to the appropriate folder based on text detection result
                if has_text:
                    shutil.copy(image_path, os.path.join(text_folder, image_filename))
                    print(f"Image {image_filename} copied to {text_folder}")
                else:
                    shutil.copy(image_path, os.path.join(no_text_folder, image_filename))
                    print(f"Image {image_filename} copied to {no_text_folder}")
            except Exception as e:
                print(f"Failed to process image {image_filename}: {str(e)}")

# Define input folder and destination folders
input_folder = "path/to/your/images"  # Folder with 100,000 images
text_folder = "path/to/text_detected"  # Folder for images with detected text
no_text_folder = "path/to/no_text_detected"  # Folder for images without detected text

# Run the image processing
process_images_in_folder(input_folder, text_folder, no_text_folder)
