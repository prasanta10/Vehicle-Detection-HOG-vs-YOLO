import os
import glob
import numpy as np
from tqdm import tqdm
from skimage import io, color, transform
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    roc_curve, auc, RocCurveDisplay
)
import matplotlib.pyplot as plt
import joblib  


# reproducibility
RANDOM_STATE = 42


VEH_DIR = "./datasets/vehicles/"
NONVEH_DIR = "/datasets/non-vehicles/"

# image and HOG params
TARGET_SIZE = (64, 64)  # width, height
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'transform_sqrt': True,
    'feature_vector': True
}

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-6


vehicle_paths = sorted(glob.glob(os.path.join(VEH_DIR, "*.*")))
nonveh_paths = sorted(glob.glob(os.path.join(NONVEH_DIR, "*.*")))

print(f"Found {len(vehicle_paths)} vehicle images, {len(nonveh_paths)} non-vehicle images")

paths = vehicle_paths + nonveh_paths
labels = [1] * len(vehicle_paths) + [0] * len(nonveh_paths)
paths = np.array(paths)
labels = np.array(labels)

paths_train, paths_temp, y_train, y_temp = train_test_split(
    paths, labels, train_size=TRAIN_FRAC, stratify=labels, random_state=RANDOM_STATE
)

temp_frac = VAL_FRAC + TEST_FRAC 

val_frac_of_temp = VAL_FRAC / temp_frac 

paths_val, paths_test, y_val, y_test = train_test_split(
    paths_temp, y_temp, train_size=val_frac_of_temp, stratify=y_temp, random_state=RANDOM_STATE
)

print("Splits:")
print(f"  Train: {len(paths_train)}")
print(f"  Val:   {len(paths_val)}")
print(f"  Test:  {len(paths_test)}")

def preprocess_and_hog(path, target_size=TARGET_SIZE, hog_params=HOG_PARAMS):

    img = io.imread(path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim == 3:
        gray = color.rgb2gray(img)   
    else:
        gray = img.astype('float32') / 255.0

    gray_resized = transform.resize(gray, (target_size[1], target_size[0]), anti_aliasing=True)

    hog_feat = hog(gray_resized, **hog_params)
    return hog_feat, gray_resized

def extract_features(paths_list):
    feats = []
    for p in tqdm(paths_list, desc="Extracting HOG"):
        f, _ = preprocess_and_hog(p)
        feats.append(f)
    return np.array(feats, dtype=np.float32)

X_train = extract_features(paths_train)
X_val = extract_features(paths_val)
X_test = extract_features(paths_test)

print("Feature shapes:")
print("  X_train:", X_train.shape)
print("  X_val:  ", X_val.shape)
print("  X_test: ", X_test.shape)

np.savez_compressed("hog_features_train_val_test.npz",
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_test=X_test, y_test=y_test)
print("Saved features to hog_features_train_val_test.npz")


svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', LinearSVC(max_iter=5000, dual=False, random_state=RANDOM_STATE))
])

# Fit on training set
svm_pipeline.fit(X_train, y_train)
print("Training complete")

val_preds = svm_pipeline.predict(X_val)
val_precision = precision_score(y_val, val_preds)
val_recall = recall_score(y_val, val_preds)
cm_val = confusion_matrix(y_val, val_preds)

print("Validation results:")
print(" Confusion matrix:\n", cm_val)
print(f" Precision: {val_precision:.4f}")
print(f" Recall:    {val_recall:.4f}")


y_pred = svm_pipeline.predict(X_test)

svm = svm_pipeline.named_steps['svm']
scaler = svm_pipeline.named_steps['scaler']


X_test_scaled = scaler.transform(X_test)
y_scores = svm.decision_function(X_test_scaled)  # higher -> class 1

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # true positive rate (recall)
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
fpr_array, tpr_array, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr_array, tpr_array)

print("Test set results:")
print(" Confusion matrix:")
print(f"  TN={tn}  FP={fp}")
print(f"  FN={fn}  TP={tp}")
print(f" True Positive Rate (Recall): {tpr:.4f}")
print(f" False Positive Rate:         {fpr:.4f}")
print(f" Precision:                   {precision:.4f}")
print(f" Recall:                      {recall:.4f}")
print(f" ROC AUC:                     {roc_auc:.4f}")

plt.figure(figsize=(6,6))
plt.plot(fpr_array, tpr_array, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0,1], [0,1], linestyle='--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Vehicles vs Non-Vehicles')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# Saving the trained pipeline
model_filename = 'hog_svm_model.pkl'
joblib.dump(svm_pipeline, model_filename)

print(f"\n--- Model Saved ---")
print(f"Successfully saved the trained pipeline to: {model_filename}")