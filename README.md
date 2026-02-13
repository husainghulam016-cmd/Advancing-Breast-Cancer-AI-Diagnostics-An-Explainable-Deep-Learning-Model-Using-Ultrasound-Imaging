# Breast Ultrasound 3-Class Classification using MobileNetV2

This repository contains the implementation for the manuscript:

"Advancing Breast Cancer-AI Diagnostics: An Explainable Deep Learning Model Using 2D Grayscale Ultrasound Imaging"

The project implements a lightweight and interpretable deep learning framework using MobileNetV2 for three-class classification (benign, malignant, normal) of breast ultrasound images.

------------------------------------------------------------------
1. Dataset
------------------------------------------------------------------

This study uses the publicly available Breast Ultrasound Images (BUSI) dataset.

Official source:

• Kaggle:
  https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset

Original publication:
Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A.
Dataset of Breast Ultrasound Images.
Data in Brief. 2020;28:104863.

After downloading, extract the dataset and structure it as:

BUSI_dataset/
    benign/
    malignant/
    normal/

Place the BUSI_dataset folder in the same directory as the notebooks.

------------------------------------------------------------------
2. Environment Setup
------------------------------------------------------------------

Recommended:
- Python 3.9+
- TensorFlow 2.x

Install required dependencies:

pip install -r requirements.txt

If requirements.txt is not provided, install at minimum:

tensorflow
numpy
scikit-learn
matplotlib
opencv-python
pandas

------------------------------------------------------------------
3. Repository Structure
------------------------------------------------------------------

train_holdout_validation.ipynb
    • Performs 80/20 stratified split
    • Trains MobileNetV2 model
    • Saves trained model as best_model.h5
    • Reports confusion matrix and classification metrics

five_fold_cross_validation.ipynb
    • Performs 5-fold stratified cross-validation
    • Reinitializes and retrains model for each fold
    • Reports mean accuracy, macro F1-score, and macro AUC

best_model.h5
    • Saved trained model weights (can be regenerated)

------------------------------------------------------------------
4. Experimental Configuration
------------------------------------------------------------------

Image size: 192 x 192  
Batch size: 16  
Learning rate: 5e-5  
Dropout rate: 0.5  
Maximum epochs: 150  
Early stopping patience: 15  
Random seed: 42  

Preprocessing follows MobileNetV2 normalization.

------------------------------------------------------------------
5. Reproducibility Notes
------------------------------------------------------------------

• All metrics reported in the manuscript are derived from the independent validation set.
• Cross-validation results are reported separately.
• Only 2D grayscale B-mode images were used.
• Segmentation masks included in the BUSI dataset were not used.
• Random seed is fixed for reproducibility.

------------------------------------------------------------------
6. License
------------------------------------------------------------------

This code is provided for academic and research purposes.
Please cite the associated manuscript if you use this implementation.
