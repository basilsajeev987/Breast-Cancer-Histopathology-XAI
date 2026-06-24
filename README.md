# Breast Cancer Histopathology Classification Using Explainable Transfer Learning

## Overview

This project presents a deep learning framework for automated breast cancer histopathology image classification using transfer learning and explainable artificial intelligence (XAI).

Five state-of-the-art convolutional neural networks were evaluated:

* DenseNet121
* ResNet50
* EfficientNetB0
* VGG16
* MobileNetV2

The best-performing model was further analyzed using Grad-CAM to provide visual explanations of model predictions.

---

## Objectives

* Develop an automated breast cancer classification system.
* Compare multiple transfer learning architectures.
* Evaluate model performance using comprehensive classification metrics.
* Improve interpretability using Grad-CAM visualizations.
* Ensure reproducibility through patient-level stratified splitting.

---

## Dataset

### BreakHis Dataset

The project uses the BreakHis (Breast Cancer Histopathological Image Classification) dataset.

Characteristics:

* 7,909 histopathology images
* 82 patients
* Benign and Malignant classes
* Magnifications: 40X, 100X, 200X, 400X

Dataset is not included in this repository due to licensing and size limitations.

---

## Methodology

### 1. Data Preparation

A patient-level stratified split was performed to:

* Prevent patient data leakage
* Maintain class balance across splits

Dataset partitions:

* Training Set
* Validation Set
* Test Set

### 2. Transfer Learning

Pretrained ImageNet models were used as feature extractors.

Each architecture was modified with:

* Global Average Pooling
* Dropout Layer
* Binary Classification Output Layer

### 3. Model Evaluation

The following metrics were used:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

### 4. Explainable AI

Grad-CAM was used to visualize image regions contributing most strongly to predictions.

---

## Project Structure

```text
Breast-Cancer-Histopathology-XAI/
│
├── src/
│   ├── prepare_data.py
│   ├── train_densenet121.py
│   ├── train_resnet50.py
│   ├── train_efficientnetb0.py
│   ├── train_vgg16.py
│   ├── train_mobilenetv2.py
│   ├── evaluate_models.py
│   └── gradcam.py
│
├── data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── models/
│
├── results/
│   ├── Final_Model_Comparison.csv
│   ├── Accuracy_Comparison.png
│   ├── Precision_Comparison.png
│   ├── Recall_Comparison.png
│   ├── F1_Score_Comparison.png
│   ├── ROC_AUC_Comparison.png
│   └── Confusion Matrices
│
├── gradcam/
│
├── notebooks/
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/basilsajeev987/Breast-Cancer-Histopathology-XAI.git
cd Breast-Cancer-Histopathology-XAI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1: Prepare Dataset

```bash
python src/prepare_data.py
```

### Step 2: Train Models

```bash
python src/train_densenet121.py
python src/train_resnet50.py
python src/train_efficientnetb0.py
python src/train_vgg16.py
python src/train_mobilenetv2.py
```

### Step 3: Compare Models

```bash
python src/evaluate_models.py
```

### Step 4: Generate Grad-CAM

```bash
python src/gradcam.py
```

---

## Results

The repository reports:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrices

Final comparison results are stored in:

```text
results/Final_Model_Comparison.csv
```

Generated visualizations include:

* Accuracy Comparison
* Precision Comparison
* Recall Comparison
* F1 Score Comparison
* ROC-AUC Comparison
* Grad-CAM Visualizations

---

## Reproducibility

Experiments were originally conducted using Google Colab with GPU acceleration.

The repository contains a refactored Python implementation together with notebooks documenting the experimental workflow.

A patient-level stratified split was used to ensure reliable and reproducible evaluation.
---

## Author
**Basil Sajeev**

## License

This project is intended for academic and research purposes only.
