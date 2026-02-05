![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)

# 🩺 Melanoma Image Classification with CNNs

This is a **deep learning image classification project** focused on **skin lesion risk assessment (benign vs malignant)** using **Convolutional Neural Networks (CNNs)**.

The goal of this project is to explore how **robust CNN architectures**, combined with **data augmentation** and **careful evaluation metrics**, can support **preliminary melanoma risk screening** from dermoscopic images.

> ⚠️ This project is intended for **research and educational purposes only** and **does not replace medical diagnosis**.

---

## 🧠 Project Overview

Training of a custom CNN to classify skin lesion images into:
- **Benign**
- **Malignant**

The project emphasizes:
- robustness to real-world image variability,
- class imbalance handling,
- threshold tuning based on medical-relevant metrics (recall / F1),
- transparent evaluation beyond simple accuracy.

The entire pipeline is implemented in a **Jupyter Notebook**, making it easy to reproduce, analyze, and extend.

---

## 📊 Model Performance

Evaluation was performed on a **balanced validation dataset of 1,000 images**.

**Key results:**
- **ROC-AUC:** **0.9238**
- **PR-AUC:** **0.9098**
- **Best F1-score:** **0.8645**
- **Accuracy:** **86.3%**

**Confusion Matrix (threshold = 0.43):**
- True Positives: 437  
- False Positives: 74  
- True Negatives: 426  
- False Negatives: 63  

The decision threshold was **explicitly optimized** to balance precision and recall, with additional analysis targeting **high-recall operating points** (important for medical screening contexts).

---

## 🏗️ Model Architecture

The CNN architecture is intentionally **lightweight yet expressive**, composed of:

- Input layer `(128 × 128 × 3)`
- Convolutional blocks:
  - `Conv2D → BatchNorm → MaxPooling`
  - Filters: 32 → 64 → 128
- `GlobalAveragePooling2D`
- `Dropout (0.4)` for regularization
- Output layer:
  - `Dense(1, sigmoid)` for binary classification

**Total parameters:** ~94K  
This compact design enables efficient training while limiting overfitting.

---

## 🔄 Data Pipeline & Augmentation

The training pipeline uses **TensorFlow `tf.data`** for efficiency and scalability.

**Preprocessing steps:**
- JPEG decoding
- Image resizing to `128 × 128`
- Normalization to `[0, 1]`

**Data augmentation (training only):**
- Random horizontal & vertical flips
- Random rotations
- Random zoom

This helps the model generalize to real-world variations such as:
- orientation changes,
- scale differences,
- acquisition noise.

---

## ⚖️ Handling Class Imbalance

To mitigate bias toward majority classes, the project uses:
- **balanced class weights** computed via `scikit-learn`
- weighted loss during training (`binary_crossentropy`)

This ensures fair optimization even when class distributions vary.

---

## 🧪 Training Strategy

- Optimizer: **Adam** (`lr = 1e-4`)
- Loss: **Binary Cross-Entropy**
- Callbacks:
  - `EarlyStopping` (restore best weights)
  - `ReduceLROnPlateau`
  - `ModelCheckpoint`
  - `TensorBoard` logging

Models are automatically saved to Google Drive for reproducibility.


---

## 🎯 Key Learning Outcomes

- End-to-end CNN training pipeline with TensorFlow
- Medical-oriented evaluation (ROC, PR, recall-focused thresholds)
- Data augmentation strategies for image robustness
- Class imbalance mitigation
- Clean experiment tracking and reproducibility

---

## 🚀 Future Work

- Experiment with **transfer learning** (ResNet, EfficientNet)
- Incorporate **explainability methods** (Grad-CAM)
- Extend to **multi-class skin lesion classification**
- Export model for **mobile or web inference**

---

## ⚠️ Disclaimer

This project is developed for **educational and research purposes only**.  
It is **not a certified medical tool** and must not be used for clinical decision-making.

---

## 👤 Author

Developed by **Aziz Hidri**  
Software Engineering Student — AI & Data Science  
Personal Project




