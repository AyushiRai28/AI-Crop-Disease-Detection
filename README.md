# AI-Based Crop Disease Detection

An AI-assisted crop disease classification application using **MobileNetV2** transfer learning to identify common foliar diseases in tomato (*Solanum lycopersicum*) plants.

---

## Overview

Plant diseases cause substantial yield loss in agriculture. Early and accurate detection of foliar symptoms allows farmers and gardeners to take timely cultural and management actions. This project demonstrates an end-to-end computer vision pipeline that classifies tomato leaf photographs into four health and disease categories.

### Supported Classes (4)

| Class | Scientific / Common Name | Typical Symptoms |
|-------|--------------------------|------------------|
| **Healthy** | Normal Foliage | Uniform green color, normal leaf morphology, no lesions. |
| **Early Blight** | *Alternaria solani* | Dark brown-to-black spots with distinct concentric rings ("target-board" pattern) and yellow halos. |
| **Late Blight** | *Phytophthora infestans* | Rapidly expanding water-soaked, dark lesions on leaves; severe under humid conditions. |
| **Leaf Mold** | *Passalora fulva* (*Cladosporium*) | Pale greenish-yellow spots on upper leaf surfaces with olive-green velvety mold underneath. |

---

## Dataset

- **Source**: Public [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) dataset (RGB color leaf images).
- **Subset**: 4 tomato classes, totaling **1,082 genuine, deduplicated images**.
- **Data Splits (Stratified, Random Seed 42, 0 Data Leakage)**:
  - **Train**: 702 images (65%)
  - **Validation**: 217 images (20%)
  - **Test**: 163 images (15% — held-out unseen test split)

---

## Model Architecture & Training

- **Backbone**: MobileNetV2 (pretrained on ImageNet, 155 layers).
- **Input Resolution**: 224 × 224 × 3 pixels.
- **Classification Head**:
  - GlobalAveragePooling2D
  - Dense (128 units, ReLU activation)
  - Dropout (rate = 0.3)
  - Dense (4 units, Softmax activation)
- **Two-Stage Transfer Learning**:
  - **Stage 1 (Feature Extraction)**: Base model frozen; Adam optimizer ($\text{lr} = 10^{-3}$), 5 epochs.
  - **Stage 2 (Fine-Tuning)**: Layers 100+ unfrozen; Adam optimizer ($\text{lr} = 10^{-5}$), 5 epochs.
- **Artifact**: `models/tomato_disease_mobilenetv2.keras` (11.07 MB).

---

## Official Model Evaluation Metrics

Evaluated on the **163 held-out unseen test images** (`results/evaluation_results.json`):

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | **88.96%** |
| **Macro Precision** | **89.08%** |
| **Macro Recall** | **88.50%** |
| **Macro F1-Score** | **88.38%** |
| **Mean Prediction Confidence** | **86.02%** |

### Per-Class Test Performance

| Class | Precision | Recall | F1-Score | Support (Test Images) |
|-------|-----------|--------|----------|-----------------------|
| **Healthy** | 83.33% | 100.00% | 0.9091 | 50 |
| **Early Blight** | 83.33% | 78.12% | 0.8065 | 32 |
| **Late Blight** | 100.00% | 83.02% | 0.9072 | 53 |
| **Leaf Mold** | 89.66% | 92.86% | 0.9123 | 28 |

---

## Tech Stack

- **Python**: 3.13.5
- **Deep Learning**: TensorFlow 2.21.0 / Keras 3
- **Web Frontend**: Streamlit 1.65.0
- **Image Processing**: Pillow 12.1.1, NumPy 2.3.2
- **Metrics & Data**: scikit-learn 1.9.1, pandas 2.3.2, matplotlib 3.10.6

---

## Project Structure

```text
AI-Crop-Disease-Detection/
├── data/
│   ├── raw/                   # 1,082 verified original images in 4 class folders
│   └── processed/             # train (702), validation (217), test (163) splits
├── models/
│   └── tomato_disease_mobilenetv2.keras   # Trained model artifact (11.07 MB)
├── results/
│   ├── evaluation_results.json            # Official test metrics
│   ├── training_history.json             # Epoch-by-epoch loss & accuracy
│   ├── confusion_matrix.png              # Confusion matrix visualization
│   └── training_history.png              # Training & validation curves
├── src/
│   ├── __init__.py
│   ├── config.py              # Central hyperparameters, labels, disease info
│   ├── dataset.py             # Data loader, stratified splitter, tf.data pipeline
│   ├── model.py               # MobileNetV2 architecture & callbacks
│   ├── train.py               # Two-stage training pipeline
│   ├── evaluate.py            # Evaluation & metrics generation
│   └── predict.py             # Reusable inference pipeline & CLI
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── PROJECT_STATUS.md          # Milestone & QA tracking
└── .gitignore
```

---

## Quick Start & Usage

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application

To start the interactive demonstration interface:

```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`. Upload a tomato leaf image (JPG/PNG) to view:
- Predicted disease classification
- Confidence percentage
- Top-3 probability distribution
- Disease symptom descriptions
- Actionable crop-care recommendations

### 3. Command-Line Inference

Predict a single image from the terminal:

```bash
python -m src.predict --image data/processed/test/Tomato___healthy/test_000201.jpg
```

Output structured JSON:

```bash
python -m src.predict --image data/processed/test/Tomato___Late_blight/test_000036.jpg --json
```

### 4. Run Model Evaluation

```bash
python -m src.evaluate
```

---

## Important Limitation & Scientific Disclaimer

> **Notice**: This software is an **AI-assisted foliar disease classification system** developed for educational and demonstration purposes. It does not provide a definitive laboratory or clinical diagnosis and cannot identify diseases outside its four trained classes. For critical crop management, pest control, or chemical treatment decisions, always verify symptoms with an agricultural extension officer or certified plant pathologist.
