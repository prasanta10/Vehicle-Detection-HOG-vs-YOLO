# HOG+SVM vs. YOLO: A Comparative Study of Vehicle Detection

This repository contains the code for our Diploma in Engineering project, performing a comparative analysis of a classical computer vision algorithm and a modern deep learning model for vehicle detection.

The goal is to analyze the trade-offs between the hand-crafted feature approach of **HOG (Histogram of Oriented Gradients) + SVM** and the end-to-end deep learning approach of **YOLO (You Only Look Once)**.

## Dataset

We are using the **Udacity Self-Driving Car Dataset** for training, validation, and testing. This dataset consists of thousands of images captured from a vehicle's dashboard camera, providing a realistic scenario for evaluating our models' performance in detecting cars, trucks, and other vehicles.



---

## 🚀 The Approaches

We are implementing and evaluating two distinct pipelines:

### 1. 🚶‍♂️ Classical Approach: HOG + SVM
This method represents the "classical" computer vision pipeline, which is a multi-stage process:

1.  **Feature Extraction:** HOG features, which describe local gradient orientation, are extracted from image patches.
2.  **Training:** A linear Support Vector Machine (SVM) is trained on "positive" (vehicle) and "negative" (background) HOG features.
3.  **Detection:** A sliding window scans test images at multiple scales. Each window's HOG features are fed to the SVM, which classifies it as "vehicle" or "background."
4.  **Post-processing:** Non-Maxima Suppression (NMS) is used to clean up overlapping bounding boxes.

### 2. ⚡ Modern Approach: YOLO (You Only Look Once)
This method represents the modern "end-to-end" deep learning approach:

1.  **Unified Model:** A single deep neural network (e.g., YOLOv5) processes the entire image in one pass.
2.  **Learned Features:** The network *learns* the most effective features for detection during training, rather than having them hand-crafted.
3.  **Detection:** The model directly regresses bounding box coordinates and predicts class probabilities for all objects in the image grid simultaneously. This "single-shot" design makes it extremely fast.

---

## 📊 Evaluation Metrics

To ensure a fair and comprehensive comparison, both models will be evaluated on the same hidden **test set**. We will measure:

* **Accuracy:** **mean Average Precision (mAP)**, the standard metric for object detection accuracy (at an IoU threshold of 0.5).
* **Speed:** **Inference Time** (in milliseconds) and **Frames Per Second (FPS)** to measure how suitable each model is for real-time applications.

## 🏆 Final Results

Our final analysis and a summary table of all results will be presented here.

| Metric | HOG + SVM (Classical) | YOLOv5 (Modern) |
| :--- | :--- | :--- |
| **mAP @ 0.5 IoU** | *TBD* | *TBD* |
| **Precision** | *TBD* | *TBD* |
| **Recall** | *TBD* | *TBD* |
| **Avg. Inference Time (ms)** | *TBD* | *TBD* |
| **Speed (FPS)** | *TBD* | *TBD* |