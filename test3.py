#to list all images in the top_1000_images directory and then delete any matching images in the raw_images_1Lakh directory.for labelimg
#this dataset to be used for next testing so that the same batch which is corrected is not used this batch which is deleteed is used for training

import os

# Define paths
top_1000_images_path = '/mnt/shared-storage/yolov11L_Image_training_set_400/BT3_img_400+1.8+1.8t_4284infer_IT3/testing/test_predictions/predictions/top_2000_images_labelimg'
raw_images_path = '/mnt/shared-storage/yolov11L_Image_training_set_400/BT4_IMG_400+1.8+1.8+1.8+2K_2Kinfer_IT4/testing/diagrams_oc_neg_4284'

# Get list of image filenames in top_1000_images (without path)
top_1000_images = {os.path.basename(f) for f in os.listdir(top_1000_images_path) if f.endswith(('.jpg', '.png',',jpeg'))}

# Go through raw_images_1Lakh directory and delete matching images
for image in os.listdir(raw_images_path):
    image_path = os.path.join(raw_images_path, image)
    if image in top_1000_images:
        try:
            os.remove(image_path)
            print(f'Deleted: {image_path}')
        except Exception as e:
            print(f'Error deleting {image_path}: {e}')
