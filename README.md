# COMP472-Brain-Tumor-MRI-CNN

A convolutional neural network (CNN) project for classifying brain tumor MRI scans, developed for COMP 472.

## Notebooks

| Notebook | Description |
|---|---|
| `notebooks/Binary_Classification_br35h.ipynb` | Binary tumor/no-tumor classification on the Br35H dataset |
| `notebooks/4c_Classification_MRI.ipynb` | 4-class MRI tumor type classification |
| `notebooks/15c_Classification_MRI.ipynb` | 15-class MRI tumor type classification |

## Commit 6dca4ad — Setup data preprocessing and stratified loaders for Br35H

This commit introduced the foundational data pipeline and training infrastructure for the Br35H binary brain-tumor detection notebook. The changes can be grouped into four areas:

### 1. Dataset download and structure verification
- Downloads the [Br35H Brain Tumor Detection dataset](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection) from Kaggle.
- Inspects the extracted directory layout and automatically descends into a single wrapper folder when present, so that `torchvision.datasets.ImageFolder` always receives the correct `<class>/<image>` structure.

### 2. Image transforms
Two separate transform pipelines are defined:

| Pipeline | Transforms applied |
|---|---|
| `train_transforms` | Resize(224×224) → RandomHorizontalFlip → RandomRotation(15°) → ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1) → RandomAffine(translate=(0.05, 0.05)) → ToTensor → Normalize |
| `eval_transforms` | Resize(224×224) → ToTensor → Normalize |

Normalization uses ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

### 3. Stratified train / val / test split
`get_stratified_split_indices(targets, val_split=0.15, test_split=0.15)` uses `sklearn.model_selection.train_test_split` twice to produce a **70 / 15 / 15** split while preserving the original class distribution in every subset (stratified sampling). The dataset is loaded twice from the same folder — once with augmentation transforms (for training) and once with evaluation transforms (for validation and test) — so each `Subset` automatically inherits the correct pipeline.

### 4. DataLoaders and training pipeline
- Three `DataLoader` objects (`train_loader`, `val_loader`, `test_loader`) with `batch_size=32`, `pin_memory=True`, and `num_workers=2`.
- `train_model()`: training loop with per-epoch LR scheduling (`ReduceLROnPlateau` on validation loss), best-weight saving, and full history logging (loss, accuracy, LR) exported as JSON.
- `evaluate_model()`: runs inference on the test set and prints a `sklearn` classification report.
- `build_experiment()`: convenience factory that wires up `CrossEntropyLoss`, `AdamW` (weight_decay=1e-4), and `ReduceLROnPlateau` for a given model.
- Three CNN architectures are instantiated (with an optional `USE_PRETRAINED` flag for transfer learning): **ResNet18**, **MobileNetV2**, and **VGG16**, each with its final classifier layer replaced to match the number of target classes.
- Visualization helpers `plot_training_curves()` and `plot_confusion_matrix()` for post-training analysis.