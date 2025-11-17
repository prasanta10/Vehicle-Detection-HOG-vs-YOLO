import cv2
import joblib
import time
import glob
import os
import json
import numpy as np
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from tqdm import tqdm
from imutils.object_detection import non_max_suppression


HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'transform_sqrt': True,
    'feature_vector': True
}
(winW, winH) = (64, 64)
WINDOW_SIZE = (winW, winH)

DOWNSCALE_FACTOR = 1.5
WINDOW_STEP = 8
CONFIDENCE_THRESHOLD = 0.5 

TEST_IMAGE_DIR = "dataset/test/images/"
MODEL_PATH = "hog_svm_model.pkl"
RESULTS_FILE = "hog_predictions.json"


try:
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"Successfully loaded '{MODEL_PATH}'")
except FileNotFoundError:
    print(f"FATAL ERROR: Model file not found at '{MODEL_PATH}'")
    exit()

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0] - windowSize[1], stepSize):
        for x in range(0, image.shape[1] - windowSize[0], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

all_image_paths = sorted(glob.glob(os.path.join(TEST_IMAGE_DIR, "*.jpg")))
if not all_image_paths:
    print(f"FATAL ERROR: No .jpg images found in {TEST_IMAGE_DIR}")
    exit()

total_time = 0
all_predictions = {} 

for image_path in tqdm(all_image_paths):
    image_name = os.path.basename(image_path)
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Warning: Could not read {image_path}, skipping.")
        continue
        
    start_time = time.time()
    detections = []
    scale = 0

    for resized in pyramid_gaussian(img_gray, downscale=DOWNSCALE_FACTOR):
        for (x, y, window) in sliding_window(resized, stepSize=WINDOW_STEP, windowSize=WINDOW_SIZE):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            
            features = hog(window, **HOG_PARAMS)
            features = features.reshape(1, -1)
          
            try:
                score = model_pipeline.decision_function(features)[0]
                if score > CONFIDENCE_THRESHOLD:
                    scale_factor = DOWNSCALE_FACTOR**scale
                    x1 = int(x * scale_factor)
                    y1 = int(y * scale_factor)
                    w = int(WINDOW_SIZE[0] * scale_factor)
                    h = int(WINDOW_SIZE[1] * scale_factor)
                    detections.append((x1, y1, x1 + w, y1 + h, score))
            except:
                pass 
            
        scale += 1

    total_time += (time.time() - start_time)
  
    boxes = np.array([[x1, y1, x2, y2] for (x1, y1, x2, y2, score) in detections])
    scores = np.array([score for (x1, y1, x2, y2, score) in detections])
    pick = non_max_suppression(boxes, probs=scores, overlapThresh=0.3)
    
    final_boxes = pick.tolist()
    all_predictions[image_name] = final_boxes

if not all_image_paths:
    print("No images were processed.")
else:
    avg_time = total_time / len(all_image_paths)
    fps = 1.0 / avg_time
    print("\n--- HOG+SVM Speed Results ---")
    print(f"Total images: {len(all_image_paths)}")
    print(f"Average Time per Image: {avg_time * 1000:.2f} ms")
    print(f"Average FPS: {fps:.2f}")

with open(RESULTS_FILE, 'w') as f:
    json.dump(all_predictions, f)

print(f"\nSaved all HOG predictions to {RESULTS_FILE}")