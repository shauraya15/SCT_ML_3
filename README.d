# Task 3: SVM-based Cat vs Dog Image Classifier

This project implements a **Support Vector Machine (SVM)** to classify images of **cats and dogs** using grayscale features. It includes:

- Image loading from custom dataset
- Grayscale conversion and vector flattening
- SVM model training
- Evaluation (Classification report + confusion matrix)
- Visualization of actual vs predicted labels

##Dataset
- Custom dataset stored in `data/` folder (not included in repo)
- 500 cat images + 500 dog images (64x64 grayscale)

##Outputs
- Trained model: `svm_grayscale_model.pkl`
- Visuals:
  - `confusion_matrix_heatmap.png`
  - `svm_predictions_actual_vs_pred.png`

##Technologies
- Python, OpenCV, scikit-learn, matplotlib, tqdm

