<div align="center">

# 🎭 Deepfake Detection: Custom CNN vs. ResNet-50

**A comparative deep learning study for binary facial deepfake classification (Real vs. Fake) using both a custom convolutional neural network and transfer learning.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<!-- Se hai salvato grafici di plot.py (es. Loss/Accuracy o Matrice di Confusione), puoi caricarli e scommentare la riga sotto: -->
<!-- <img src="checkpoints/metrics_preview.png" alt="Training Metrics & Confusion Matrix" width="700"/> -->

</div>

---

## 📌 Overview

This project addresses the problem of **Facial Deepfake Detection** by formulating it as a supervised binary classification task (**REAL** vs. **FAKE**). 

The goal is to benchmark two distinct architectural approaches on the same Kaggle facial dataset:
1. **Custom CNN:** A lightweight convolutional architecture designed and trained from scratch.
2. **ResNet-50 (Transfer Learning):** A deep residual backbone with pre-trained ImageNet weights, fine-tuned for facial artifact recognition.

---

## 🧠 Model Architectures

| Feature | Custom CNN | ResNet-50 |
| :--- | :--- | :--- |
| **Approach** | Trained from scratch | Pre-trained Transfer Learning |
| **Input Resolution** | 224 × 224 px | 224 × 224 px |
| **Recommended Learning Rate** | `0.001` | `0.0001` |
| **Recommended Batch Size** | `64` | `32` (or `16` / `8` for lower VRAM) |
| **Regularization** | Early Stopping | Early Stopping & Pretrained Weights |

---

## 📁 Repository Structure

```text
├── backbones/           # Backbone model definitions (ResNet-50, etc.)
├── utils/               # Dataset loaders, metrics & helper utilities
├── FINAL_DATASET.csv    # Dataset annotations with image URLs and labels
├── download_dataset.py  # Automated script to fetch images locally
├── main.py              # CLI entry point for training and evaluation
├── net.py               # Custom CNN network architecture
├── solver.py            # Training loop, validation, and early stopping logic
├── plot.py              # Generates Loss/Accuracy curves & Confusion Matrices
├── requirements.txt     # Python package dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

It is strongly recommended to use a virtual environment to manage dependencies:

```bash
# 1. Clone the repository
git clone https://github.com/BillyErMago/deepfake-deeplearning.git
cd deepfake-deeplearning

# 2. Create and activate a virtual environment
python -m venv venv

# On Linux / macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt
```

---

## 📥 Dataset Preparation

The training and evaluation pipeline uses image references indexed in `FINAL_DATASET.csv`. 

Before starting training, download the dataset images locally:

```bash
python download_dataset.py
```

*Images will be fetched and organized into `./data/images/`.*

---

## 🚀 Model Training

The training script automatically resizes incoming images to `224x224` and includes an integrated **Early Stopping** mechanism to prevent overfitting.

### 1. Train the Custom CNN (Default)
```bash
python main.py --backbone custom --batch_size 64 --lr 0.001
```

### 2. Train the ResNet-50
```bash
python main.py --backbone resnet50 --batch_size 32 --lr 0.0001
```

> **💡 GPU / VRAM Tip:** If you encounter a CUDA `Out of Memory` (OOM) error when training ResNet-50, lower the batch size using `--batch_size 16` or `--batch_size 8`.

### Custom Training Arguments
```bash
python main.py --backbone resnet50 --epochs 100 --batch_size 32
```

---

## 🧪 Testing & Evaluation

To evaluate a trained checkpoint on the test set and output test metrics, use the `--test` flag. The script retrieves the best weights from the `./checkpoints/` directory:

```bash
# Evaluate Custom CNN
python main.py --backbone custom --test

# Evaluate ResNet-50
python main.py --backbone resnet50 --test
```

---

## 📊 Plotting Results & Confusion Matrices

Once training and evaluation are complete, generate performance curves (Loss/Accuracy) and Confusion Matrices:

```bash
python plot.py
```

*All resulting plots (`.png`) and performance summaries are exported directly to `./checkpoints/`.*

---

## 👤 Author

- **BillyErMago** - [GitHub Profile](https://github.com/BillyErMago)
