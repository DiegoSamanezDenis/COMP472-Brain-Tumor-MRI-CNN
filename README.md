# Brain Tumor MRI Classification with Convolutional Neural Networks

A PyTorch-based deep learning pipeline for classifying brain tumors from MRI scans. The project benchmarks three CNN architectures (ResNet18, MobileNetV2, VGG16) across three public datasets of increasing complexity — from binary detection to 15-class tumor-type classification — and explores the impact of transfer learning and hyperparameter tuning.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Datasets](#datasets)
4. [Requirements](#requirements)
5. [Training](#training)
6. [Hyperparameter Search](#hyperparameter-search)
7. [Running a Pre-Trained Model](#running-a-pre-trained-model)
8. [Visualizations (t-SNE & Grad-CAM)](#visualizations)
9. [Results Summary](#results-summary)

---

## Project Overview

The goal of this project is to evaluate how well standard CNN architectures can classify brain tumors from MRI images, and to understand the effect of dataset complexity, training from scratch vs. transfer learning, and hyperparameter choices on classification performance.

**Architectures evaluated:**

| Model | Parameters | Notes |
|---|---|---|
| ResNet18 | ~11M | Residual connections, lightweight |
| MobileNetV2 | ~3.4M | Depthwise separable convolutions, efficient |
| VGG16 | ~138M | Deep sequential convolutions, high capacity |

**Key findings:**

- ResNet18 and MobileNetV2 train reliably from scratch on all three datasets, reaching **98% accuracy** on binary detection and **96–97%** on 4-class classification.
- VGG16 consistently fails when trained from scratch (the loss plateaus and the model collapses to predicting a single class), but performs well with **pretrained ImageNet weights and a frozen backbone** — achieving the highest accuracy on the 15-class dataset at **80.8%**.
- Transfer learning with ResNet18 on the 4-class dataset pushes accuracy to **99.1%**.

---

## Repository Structure

```
.
├── core.py                  # Shared module: data loading, model builders,
│                            #   training loop, evaluation, plotting utilities
├── train.py                 # Main training script (single or batch runs)
├── hyperparam_search.py     # Grid search over lr, batch_size, or loss function
├── visualize.py             # t-SNE and Grad-CAM visualization generator
│
├── datasets/                # Place downloaded datasets here
│   ├── dataset_br35h/       #   Br35H (binary: yes / no)
│   ├── dataset_mri_scans/   #   Brain Tumor MRI Scans (4 classes)
│   └── dataset_44c/         #   Brain Tumor MRI Images 44c (→ merged to 15 classes)
│
└── outputs/                 # Generated automatically during training
    ├── br35h/               #   Per-dataset results
    │   ├── resnet_scratch/
    │   ├── mobilenet_scratch/
    │   └── vgg_scratch/
    ├── 4c/
    │   ├── resnet_scratch/
    │   ├── resnet_pretrained/
    │   ├── mobilenet_scratch/
    │   └── vgg_scratch/
    ├── 15c/
    │   ├── resnet_scratch/
    │   ├── mobilenet_scratch/
    │   ├── vgg_scratch/
    │   └── vgg_pretrained/
    ├── hyperparam/           #   Hyperparameter search results
    └── summary.json
```

Each run directory contains: `best_model.pth`, `history.json`, `eval_report.json`, `curves.png`, `confusion.png`, and optionally `tsne.png` and `gradcam.png`.

---

## Datasets

Three publicly available datasets are used, each representing a different level of classification difficulty.

### 1. Br35H — Binary Detection (2 classes)

Binary classification: tumor present (*yes*) vs. no tumor (*no*).

- **Source:** [Br35H :: Brain Tumor Detection 2020 — Kaggle](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)
- **Download and place in:** `datasets/dataset_br35h/`
- The folder should contain two subfolders: `yes/` and `no/`.

### 2. Brain Tumor MRI Dataset — 4 Classes

Four-class classification: glioma, meningioma, pituitary, and healthy.

- **Source:** [Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Download and place in:** `datasets/dataset_mri_scans/`
- The folder should contain four subfolders: `glioma/`, `meningioma/`, `pituitary/`, `healthy/`.

### 3. Brain Tumor MRI Images 44 Classes → Merged to 15

The original dataset has 44 folders (15 tumor types × 3 MRI sequences: T1, T1C+, T2). The pipeline automatically merges the sequences into **15 tumor-type classes** on first run.

- **Source:** [Brain Tumor MRI Images 44 Classes — Kaggle](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c)
- **Download and place in:** `datasets/dataset_44c/`
- On first training run, a merged directory `datasets/dataset_15c/` is created automatically.

---

## Requirements

**Python 3.10+** is recommended.

### Core Dependencies

| Package | Purpose |
|---|---|
| `torch` + `torchvision` | Model definition, training, inference (the only AI/ML framework used) |
| `numpy` | Array operations |
| `matplotlib` | Training curves, confusion matrices |
| `seaborn` | Heatmap styling for confusion matrices |
| `scikit-learn` | Evaluation metrics (`classification_report`, `confusion_matrix`), stratified splitting, t-SNE visualization |
| `tqdm` | Progress bars |
| `Pillow` | Image loading (required by `torchvision`) |

> **Note:** scikit-learn is used exclusively for standard evaluation metrics, data-splitting utilities, and a visualization technique (t-SNE). All model architectures, training, and inference are implemented purely in PyTorch.

### Installation

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn tqdm Pillow
```

For GPU acceleration, install the appropriate CUDA version of PyTorch from [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

---

## Training

All training is done through `train.py`. The script supports single runs, batch runs, and transfer learning.

### Train a Single Model

```bash
# ResNet18 from scratch on the binary dataset
python train.py --dataset br35h --model resnet

# MobileNetV2 from scratch on the 4-class dataset
python train.py --dataset 4c --model mobilenet

# VGG16 with pretrained ImageNet weights on the 15-class dataset
python train.py --dataset 15c --model vgg --pretrained
```

### Train All Models

```bash
# All 9 from-scratch combinations (3 datasets × 3 models)
python train.py --all-scratch

# All 11 models (9 scratch + 2 predefined transfer-learning pairs)
python train.py --all
```

The two transfer-learning pairs that `--all` adds are ResNet18 on 4-class and VGG16 on 15-class, which are the configurations that benefit most from pretrained weights.

### Override Hyperparameters

```bash
python train.py --dataset 4c --model resnet --epochs 30 --batch-size 64 --lr 0.0005
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--dataset` | — | `br35h`, `4c`, or `15c` |
| `--model` | — | `resnet`, `mobilenet`, or `vgg` |
| `--pretrained` | off | Use ImageNet pretrained weights |
| `--epochs` | 20 | Number of training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 0.001 | Initial learning rate |
| `--num-workers` | 4 (Linux) / 2 (Windows) | DataLoader worker processes |
| `--data-dir` | — | Override default dataset path |
| `--output-dir` | `./outputs` | Root output directory |

### What Happens During Training

1. The dataset is loaded via `ImageFolder` and split into **train / validation / test** sets (70% / 15% / 15%) with stratified sampling.
2. Training augmentations include random horizontal flips, rotation (±15°), color jitter, and small affine translations.
3. The optimizer is **AdamW** with weight decay of 1e-4 and a **ReduceLROnPlateau** scheduler (halves LR after 3 epochs without improvement).
4. The best model weights (by validation accuracy) are saved to `best_model.pth`.
5. After training, the model is evaluated on the held-out test set and a classification report + confusion matrix are saved.

---

## Hyperparameter Search

`hyperparam_search.py` performs a grid search over one hyperparameter at a time for a given dataset + model pair.

```bash
# Search learning rates
python hyperparam_search.py --dataset 15c --model vgg --param lr \
    --values 0.00001 0.00005 0.0001 0.0003

# Search batch sizes
python hyperparam_search.py --dataset 15c --model vgg --param batch_size \
    --values 16 32 64

# Search loss functions (CrossEntropy vs Focal Loss)
python hyperparam_search.py --dataset 15c --model vgg --param loss \
    --values cross_entropy focal
```

Each trial trains a full model, evaluates it, and saves individual curves and confusion matrices. A comparison plot and a `search_results.json` summary are saved to `outputs/hyperparam/`.

---

## Running a Pre-Trained Model

If pre-trained weights (`best_model.pth`) are already available (e.g., downloaded or from a previous training run), you can evaluate them on the test set or generate visualizations without retraining.

### Evaluate on Test Set

Use `visualize.py` to load a trained model and run inference. Although named "visualize", it rebuilds the test DataLoader and can be combined with `--tsne` / `--gradcam` for evaluation:

```bash
# Generate t-SNE and Grad-CAM for a specific model
python visualize.py --dataset br35h --model resnet \
    --weights outputs/br35h/resnet_scratch/best_model.pth \
    --tsne --gradcam
```

### Auto-Discover All Trained Models

```bash
# Finds every best_model.pth in the outputs directory and generates
# both t-SNE and Grad-CAM for each
python visualize.py --all
```

### What the Model Weights Expect

When loading weights, the script automatically:
1. Reads the dataset to determine the number of classes.
2. Builds the correct architecture (`build_model` with `pretrained=False` — the pretrained flag only controls whether to download ImageNet weights, which isn't needed when loading your own `.pth`).
3. Loads the state dict from the `.pth` file.

Make sure the dataset is available at the expected path (or use `--data-dir` to specify it), since the class names and test images are read from the dataset folder.

---

## Visualizations

`visualize.py` generates two types of post-training visualizations.

### t-SNE

Extracts features from the penultimate layer of the trained model and projects them into 2D using t-SNE. This shows how well the model's learned representation separates the classes.

```bash
python visualize.py --dataset 4c --model mobilenet \
    --weights outputs/4c/mobilenet_scratch/best_model.pth --tsne
```

### Grad-CAM

Generates Grad-CAM heatmaps overlaid on random test images, showing which regions of the MRI the model focuses on when making predictions. Correct predictions are labeled in green, incorrect ones in red.

```bash
python visualize.py --dataset 15c --model vgg --pretrained \
    --weights outputs/15c/vgg_pretrained/best_model.pth --gradcam
```

---

## Results Summary

### Binary Detection (Br35H — 2 classes)

| Model | Training | Test Accuracy |
|---|---|---|
| ResNet18 | Scratch | **98.0%** |
| MobileNetV2 | Scratch | **98.0%** |
| VGG16 | Scratch | 50.0% (failed — collapsed to single class) |

### 4-Class Classification (Glioma, Meningioma, Pituitary, Healthy)

| Model | Training | Test Accuracy |
|---|---|---|
| ResNet18 | Pretrained | **99.1%** |
| MobileNetV2 | Scratch | 96.6% |
| ResNet18 | Scratch | 96.4% |
| VGG16 | Scratch | 28.5% (failed) |

### 15-Class Classification (Tumor Types)

| Model | Training | Test Accuracy |
|---|---|---|
| VGG16 | Pretrained | **80.8%** |
| ResNet18 | Scratch | 78.7% |
| MobileNetV2 | Scratch | 76.5% |
| VGG16 | Scratch | 19.5% (failed) |

### Hyperparameter Search (15-Class, VGG16 Pretrained)

**Learning Rate:** Best at lr=5e-05 (test 88.4%), followed closely by lr=1e-04 (87.5%). Rates below 1e-05 underfit; above 3e-04, training becomes unstable.

**Batch Size:** bs=32 achieved the best validation accuracy (91.2%), marginally ahead of bs=16 (89.6%) and bs=64 (90.8%).

**Loss Function:** Focal Loss (test 87.8%) outperformed standard CrossEntropy (85.7%) on validation accuracy, likely due to better handling of class imbalance in the 15-class setting.

---

## License

This project was developed for academic purposes. The datasets are subject to their respective licenses on Kaggle.