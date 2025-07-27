import os
import cv2
import numpy as np
import pandas as pd
from skimage import exposure
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Configuration
CSV_PATH = "data/train.csv"
IMAGE_DIR = "data/train_images"
IMAGE_SIZE = 128
MODEL_PATH = "svm_model.pkl"

# Load CSV
df = pd.read_csv(CSV_PATH)

# Prepare features and labels
X, y = [], []
for idx, row in df.iterrows():
    filename = row["id_code"] + ".png"
    label = row["diagnosis"]
    filepath = os.path.join(IMAGE_DIR, filename)
    
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_eq = exposure.equalize_adapthist(img_resized, clip_limit=0.03)
    img_flat = img_eq.flatten()
    
    X.append(img_flat)
    y.append(label)

# Convert to NumPy
X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train SVM
model = SVC(kernel="rbf", C=1.0, gamma="scale")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# Save model
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
