import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Paths
cat_dir = "data/cat_set"
dog_dir = "data/dog_set"

# Data prep
X = []
y = []

IMG_SIZE = 64  # grayscale 64x64 = 4096 features

def load_images_from_folder(folder, label):
    for filename in tqdm(os.listdir(folder), desc=f"Loading {label}s"):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img.flatten())
            y.append(label)

# Load images
load_images_from_folder(cat_dir, 0)  # 0 for cat
load_images_from_folder(dog_dir, 1)  # 1 for dog

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = SVC(kernel='linear')  # try 'rbf' later
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# ---- PHASE 2 ADDITIONS BELOW ---- #

# (1) Save the model
with open("svm_grayscale_model.pkl", "wb") as f:
    pickle.dump(model, f)

# (2) Save confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.savefig("confusion_matrix_heatmap.png")
plt.close()

# (3) Visualization with Actual vs Predicted
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.flatten()

for i in range(10):
    img = X_test[i].reshape(IMG_SIZE, IMG_SIZE)
    pred_label = "Dog" if y_pred[i] == 1 else "Cat"
    true_label = "Dog" if y_test[i] == 1 else "Cat"
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"Pred: {pred_label}\nActual: {true_label}")
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("svm_predictions_actual_vs_pred.png")
plt.show()

# Text output
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
