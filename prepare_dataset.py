import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

CLASS_MAP_CONFIG = {
    'car': 0,      # All 'car' labels become new class ID 0
    'biker': 0,    # All 'biker' labels become new class ID 0
    'truck': 0     # All 'truck' labels become new class ID 0
}


NEW_CLASS_NAMES = [
    'vehicle'  
]

# FILE PATH CONFIGURATION
RAW_DATA_DIR = os.path.join('dataset', 'raw_data')
IMAGE_DIR = RAW_DATA_DIR
CSV_PATH = os.path.join(RAW_DATA_DIR, '_annotations.csv')

TRAIN_DIR = os.path.join('dataset', 'train')
VAL_DIR = os.path.join('dataset', 'val')
TEST_DIR = os.path.join('dataset', 'test')

# Define our split
TRAIN_SIZE = 0.70  
VAL_SIZE = 0.15    
TEST_SIZE = 0.15   

def get_image_dimensions(image_path):

    try:
        import cv2
    except ImportError:
        print("\n---")
        print("ERROR: OpenCV is not installed.")
        print("Please install it by running: pip install opencv-python")
        print("---")
        exit()
        
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Could not read image: {image_path}")
    h, w, _ = image.shape
    return w, h

def convert_to_yolo_format(img_w, img_h, xmin, ymin, xmax, ymax, class_id):

    x_center = ((xmax + xmin) / 2) / img_w
    y_center = ((ymax + ymin) / 2) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_and_split_data():
    
    print("Starting dataset preparation with class filtering...")
    try:
        import cv2
        print("OpenCV found.")
    except ImportError:
        print("\n---")
        print("ERROR: OpenCV is not installed.")
        print("Please install it by running: pip install opencv-python")
        print("---")
        return
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"ERROR: Cannot find annotations.csv at {CSV_PATH}")
        print("Please make sure 'annotations.csv' is in 'dataset/raw_data/'")
        return

    df.columns = df.columns.str.strip()
    
    print(f"Loaded CSV. Found columns: {list(df.columns)}")

    # 3. Print our new class mapping
    print(f"\nFiltering for {len(NEW_CLASS_NAMES)} classes:")
    for i, name in enumerate(NEW_CLASS_NAMES):
        print(f"  - {name} (New Class ID: {i})")
    
    print("\nOriginal classes to merge:")
    for original_name, new_id in CLASS_MAP_CONFIG.items():
        print(f"  - '{original_name}' will be mapped to New Class ID {new_id}")
    print("All other classes will be IGNORED.")


    # 4. Group labels by image (using 'filename' column)
    image_groups = df.groupby('filename')
    all_image_names = list(image_groups.groups.keys())
    print(f"\nFound {len(all_image_names)} total unique images in the CSV.")

    # 5. Split images into train, validation, and test sets
    train_names, temp_names = train_test_split(
        all_image_names, 
        test_size=(1 - TRAIN_SIZE), 
        random_state=42
    )
    val_names, test_names = train_test_split(
        temp_names, 
        test_size=(TEST_SIZE / (TEST_SIZE + VAL_SIZE)), 
        random_state=42
    )
    
    print(f"Splitting data into:")
    print(f"  - Training:   {len(train_names)} images")
    print(f"  - Validation: {len(val_names)} images")
    print(f"  - Test:       {len(test_names)} images")

    # 6. Create all the output directories
    split_map = {
        'train': (TRAIN_DIR, train_names),
        'val': (VAL_DIR, val_names),
        'test': (TEST_DIR, test_names),
    }

    for split_name, (dir_path, _) in split_map.items():
        os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'labels'), exist_ok=True)
    
    print("\nCreated 'train', 'val', and 'test' folders with 'images' and 'labels' subfolders.")

    # 7. Process all images and labels
    print("Processing images and labels... (This may take a few minutes)")
    
    total_boxes_processed = 0
    images_with_objects = 0
    
    # Use tqdm for a progress bar
    for split_name, (dir_path, image_names) in split_map.items():
        print(f"\nProcessing {split_name} set...")
        
        for image_name in tqdm(image_names, desc=f"  {split_name} images"):
            
            src_image_path = os.path.join(IMAGE_DIR, image_name)
            if not os.path.exists(src_image_path):
                print(f"Warning: Image file {image_name} not found. Skipping.")
                continue

            dest_image_path = os.path.join(dir_path, 'images', image_name)
            label_name = os.path.splitext(image_name)[0] + '.txt'
            dest_label_path = os.path.join(dir_path, 'labels', label_name)
            
            # --- Image Handling ---
            # Copy the image (we copy all, even if they have no labels)
            shutil.copy(src_image_path, dest_image_path)
            
            # --- Label Handling ---
            try:
                img_w, img_h = get_image_dimensions(src_image_path)
                image_boxes = image_groups.get_group(image_name)
                
                yolo_labels = []
                # --- THIS IS THE NEW LOGIC ---
                for _, row in image_boxes.iterrows():
                    original_class_name = row['class']
                    
                    # Check if this class is one we want to keep
                    if original_class_name in CLASS_MAP_CONFIG:
                        # Get the *new* class ID from our config
                        new_class_id = CLASS_MAP_CONFIG[original_class_name]
                        
                        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                        
                        # Convert to YOLO format using the *new* class ID
                        yolo_line = convert_to_yolo_format(img_w, img_h, xmin, ymin, xmax, ymax, new_class_id)
                        yolo_labels.append(yolo_line)
                        total_boxes_processed += 1
                # --- END OF NEW LOGIC ---

                # Only write a .txt file if we found valid objects
                if yolo_labels:
                    with open(dest_label_path, 'w') as f:
                        f.write('\n'.join(yolo_labels))
                    images_with_objects += 1

            except Exception as e:
                print(f"\nError processing {image_name}: {e}")
                if os.path.exists(dest_image_path): os.remove(dest_image_path)

if __name__ == "__main__":
    process_and_split_data()