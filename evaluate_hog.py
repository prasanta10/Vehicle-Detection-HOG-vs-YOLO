import json
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm


PREDICTIONS_FILE = "hog_predictions.json"
GT_LABEL_DIR = "dataset/test/labels/"
IMG_DIR = "dataset/test/images/"
IOU_THRESHOLD = 0.5


def load_ground_truth_boxes(label_path, img_width, img_height):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
        
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # We assume class_id is 0 (vehicle)
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            w_pixels = width * img_width
            h_pixels = height * img_height
            xmin = int((x_center * img_width) - (w_pixels / 2))
            ymin = int((y_center * img_height) - (h_pixels / 2))
            xmax = int(xmin + w_pixels)
            ymax = int(ymin + h_pixels)
            boxes.append([xmin, ymin, xmax, ymax])
    return boxes

def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    return inter_area / union_area

print(f"Loading predictions from {PREDICTIONS_FILE}...")
with open(PREDICTIONS_FILE, 'r') as f:
    predictions = json.load(f)

total_tp = 0
total_fp = 0
total_fn = 0

print("Evaluating HOG mAP@0.5...")
for label_path in tqdm(glob.glob(os.path.join(GT_LABEL_DIR, "*.txt"))):
    
 
    label_name = os.path.basename(label_path)
    image_name = os.path.splitext(label_name)[0] + ".jpg"
    img_path = os.path.join(IMG_DIR, image_name)
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Missing image {image_name}, skipping.")
        continue
    img_h, img_w, _ = img.shape
    
    gt_boxes = load_ground_truth_boxes(label_path, img_w, img_h)
    
    pred_boxes = predictions.get(image_name, [])
    
    gt_used = [False] * len(gt_boxes)
    tp = 0
    fp = 0

    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= IOU_THRESHOLD and not gt_used[best_gt_idx]:
            tp += 1
            gt_used[best_gt_idx] = True 
        else:
            fp += 1 
            
    fn = len(gt_boxes) - sum(gt_used)
    
    total_tp += tp
    total_fp += fp
    total_fn += fn

precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

ap = precision 

print("\n--- HOG+SVM Accuracy Results ---")
print(f"Total True Positives (TP):  {total_tp}")
print(f"Total False Positives (FP): {total_fp}")
print(f"Total False Negatives (FN): {total_fn}")
print("---------------------------------")
print(f"mAP @ 0.5 (AP): {ap:.4f}")
print(f"Precision:      {precision:.4f}")
print(f"Recall:         {recall:.4f}")
print(f"F1 Score:       {f1_score:.4f}")