import os
import firebase_admin
from firebase_admin import credentials, firestore

# Path to the downloaded service account key JSON file
cred = credentials.Certificate('/mnt/shared-storage/yolov11L_Image_training_set_400/no_sql_key_db/llm-sandbox-426711-ebc75a354f86.json')  # Replace with your actual path

# Initialize the Firebase Admin SDK with the service account credentials
firebase_admin.initialize_app(cred, {
    'projectId': 'llm-sandbox-426711',  # Your Google Cloud Project ID
})

# If you're working with multiple Firestore databases, specify the database ID
db = firestore.client(database_id='image-understanding-detection')

# Function to prompt the user for the collection name (only once before processing all files)
def get_collection_name():
    collection_name = input("Please enter the name of the Firestore collection you want to use: ")
    return collection_name

# Define the label to number mapping
label_mapping = {
    "Dg Diag": 0,
    "Hw Diag": 1,
    "Annotations": 2,
    "Table Grid": 3,
    "oc": 4,
    "Table WO Grid": 5,
    "Graph": 6,
    "oc hw": 7
}

# Folder containing the .txt files
txt_folder = '/mnt/shared-storage/yolov11L_Image_training_set_400/mongodb/total_88617/yolo_txt_diagrams_88617'

# Function to prompt the user for the source_version (only once before processing all files)
def get_source_version():
    valid_versions = ['oc_positive_case', 'oc_negative_case', 'diagram_pos_case', 'dia_neg_case', 'not_touched']
    print("\nPlease choose a source version for the files:")
    print("1. oc_positive_case")
    print("2. oc_negative_case")
    print("3. diagram_pos_case")
    print("4. dia_neg_case")
    print("5. not_touched")
    
    while True:
        user_input = input("Enter the number corresponding to the source version: ")
        if user_input == '1':
            return 'oc_positive_case'
        elif user_input == '2':
            return 'oc_negative_case'
        elif user_input == '3':
            return 'diagram_pos_case'
        elif user_input == '4':
            return 'dia_neg_case'
        elif user_input == '5':
            return 'not_touched'
        else:
            print("Invalid input. Please enter a number between 1 and 5.")

# Function to parse the .txt files and upload data to Firestore
def upload_data_to_firestore():
    # Ask for the collection name and source version only once
    collection_name = get_collection_name()
    source_version = get_source_version()

    # Iterate over all the .txt files in the folder
    for txt_file in os.listdir(txt_folder):
        if txt_file.endswith(".txt"):
            image_id = txt_file.split('.')[0]  # Image ID is the file name without extension
            file_path = os.path.join(txt_folder, txt_file)

            # Initialize the data structure with False flags and empty lists
            data = {
                'image_id': image_id,
                'data_version': 'Version_1',
                'oc': False,
                'oc_coords': [],
                'diagram': False,
                'diagram_coords': [],
                'annotation': False,
                'annotation_coords': [],
                'Hw_diagram': False,
                'Hw_diagram_coords': [],
                'Table_Grid': False,
                'Table_Grid_coords': [],
                'Table_WO_Grid': False,
                'Table_WO_Grid_coords': [],
                'Graph': False,
                'Graph_coords': [],
                'oc_hw': False,
                'oc_hw_coords': [],
                'source_version': source_version  # Apply source_version here
            }

            # Read the .txt file and parse its contents
            with open(file_path, 'r') as file:
                lines = file.readlines()

                # If the file is not empty, process the contents
                if lines:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:  # Ensure it's a valid YOLO line
                            label = int(parts[0])
                            coords = list(map(float, parts[1:]))

                            # Determine which category this label belongs to and update the data
                            if label == label_mapping["Dg Diag"]:
                                data['diagram'] = True
                                data['diagram_coords'].append(coords)
                            elif label == label_mapping["Hw Diag"]:
                                data['Hw_diagram'] = True
                                data['Hw_diagram_coords'].append(coords)
                            elif label == label_mapping["Annotations"]:
                                data['annotation'] = True
                                data['annotation_coords'].append(coords)
                            elif label == label_mapping["Table Grid"]:
                                data['Table_Grid'] = True
                                data['Table_Grid_coords'].append(coords)
                            elif label == label_mapping["oc"]:
                                data['oc'] = True
                                data['oc_coords'].append(coords)
                            elif label == label_mapping["Table WO Grid"]:
                                data['Table_WO_Grid'] = True
                                data['Table_WO_Grid_coords'].append(coords)
                            elif label == label_mapping["Graph"]:
                                data['Graph'] = True
                                data['Graph_coords'].append(coords)
                            elif label == label_mapping["oc hw"]:
                                data['oc_hw'] = True
                                data['oc_hw_coords'].append(coords)

                else:
                    # If the file is empty, we ensure the structure is still created
                    print(f"File {txt_file} is empty, but still uploading the document with empty data.")

            # Upload data to Firestore under the user-defined collection name
            doc_ref = db.collection(collection_name).document(image_id)  # Use image_id as the document ID
            doc_ref.set(data)

            print(f"Data for image {image_id} has been uploaded to collection {collection_name}.")

# Call the function to upload data
upload_data_to_firestore()
