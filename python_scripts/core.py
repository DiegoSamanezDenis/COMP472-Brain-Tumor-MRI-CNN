"""
core.py — Shared module for brain tumor CNN classification project.

Contains: dataset preparation, model builders, training loop,
evaluation, and plotting utilities.
"""

import copy
import json
import os
import platform
import re
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Windows can't fork — multiprocessing workers re-import the whole script.
# persistent_workers=True keeps them alive between epochs, avoiding respawn.
# Default to 2 on Windows (with persistence), 4 on Linux.
DEFAULT_WORKERS = 2 if platform.system() == "Windows" else 4

MODEL_REGISTRY = {
    "resnet":    "ResNet18",
    "mobilenet": "MobileNetV2",
    "vgg":       "VGG16",
}

DATASET_CONFIGS = {
    "br35h": {
        "raw_dir": "./datasets/dataset_br35h",
        "description": "Br35H — Binary (yes / no)",
    },
    "4c": {
        "raw_dir": "./datasets/dataset_mri_scans",
        "description": "Brain Tumor MRI Scans — 4 classes",
    },
    "15c": {
        "raw_dir": "./datasets/dataset_44c",
        "description": "Brain Tumor MRI Images — 44c → 15 classes",
        "needs_merge": True,
        "merged_dir": "./datasets/dataset_15c",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Dataset Preparation
# ═══════════════════════════════════════════════════════════════════════

def _unwrap_single_subfolder(path: str) -> str:
    """If a directory contains exactly one subfolder, descend into it."""
    items = [d for d in os.listdir(path)
             if os.path.isdir(os.path.join(path, d))]
    if len(items) == 1:
        inner = os.path.join(path, items[0])
        print(f"  ⚠ Single subfolder detected — using: {inner}")
        return inner
    return path


def merge_44c_to_15c(raw_dir: str, merged_dir: str) -> str:
    """
    Merge 44-class dataset (with MRI sequences T1/T1C+/T2 as separate
    folders) into 15 tumor-type classes by stripping the sequence suffix.
    """
    if os.path.isdir(merged_dir):
        classes = sorted(d for d in os.listdir(merged_dir)
                         if os.path.isdir(os.path.join(merged_dir, d)))
        if classes:
            print(f"  ✓ Merged dataset already exists at {merged_dir} "
                  f"({len(classes)} classes)")
            return merged_dir

    raw_dir = _unwrap_single_subfolder(raw_dir)
    seq_pattern = re.compile(r'[_\s]+(T1C?\+?|T1C\+|T2)\s*$', re.IGNORECASE)

    print(f"  Merging MRI sequences from {raw_dir} → {merged_dir}")
    for folder in sorted(os.listdir(raw_dir)):
        src = os.path.join(raw_dir, folder)
        if not os.path.isdir(src):
            continue
        tumor_type = seq_pattern.sub('', folder).strip()
        dest = os.path.join(merged_dir, tumor_type)
        os.makedirs(dest, exist_ok=True)

        for img in os.listdir(src):
            img_src = os.path.join(src, img)
            if not os.path.isfile(img_src):
                continue
            img_dst = os.path.join(dest, img)
            if os.path.exists(img_dst):
                stem, ext = os.path.splitext(img)
                img_dst = os.path.join(dest,
                                       f"{stem}_{folder.split('_')[-1]}{ext}")
            shutil.copy2(img_src, img_dst)

    classes = sorted(d for d in os.listdir(merged_dir)
                     if os.path.isdir(os.path.join(merged_dir, d)))
    print(f"  ✓ Merged into {len(classes)} classes:")
    for cls in classes:
        n = len(os.listdir(os.path.join(merged_dir, cls)))
        print(f"    {cls:<40} {n:>5} images")
    return merged_dir


def prepare_dataset_path(dataset_key: str,
                         override_dir: str | None = None) -> str:
    """
    Resolve and prepare the dataset directory for a given dataset key.
    Handles unwrapping, merging (44c→15c), and binary filtering (br35h).
    Returns the path ready for ImageFolder.
    """
    cfg = DATASET_CONFIGS[dataset_key]
    raw_dir = override_dir or cfg["raw_dir"]

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {raw_dir}\n"
            f"  Download the dataset and place it there, or use "
            f"--data-dir to specify the path."
        )

    raw_dir = _unwrap_single_subfolder(raw_dir)

    # 44c needs merging
    if cfg.get("needs_merge"):
        merged_dir = cfg.get("merged_dir", raw_dir + "_15c")
        return merge_44c_to_15c(raw_dir, merged_dir)

    return raw_dir


def get_transforms():
    """Return (train_transforms, eval_transforms) tuple."""
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def get_stratified_splits(targets, val_frac=0.15, test_frac=0.15):
    """Return (train_idx, val_idx, test_idx) arrays with stratification."""
    train_idx, temp_idx = train_test_split(
        np.arange(len(targets)),
        test_size=val_frac + test_frac,
        stratify=targets, random_state=42,
    )
    temp_targets = [targets[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_frac / (val_frac + test_frac),
        stratify=temp_targets, random_state=42,
    )
    return train_idx, val_idx, test_idx


def build_dataloaders(data_dir: str, batch_size: int = 32,
                      num_workers: int | None = None):
    """
    Build train/val/test DataLoaders from an ImageFolder directory.
    Returns (train_loader, val_loader, test_loader, class_names).
    """
    if num_workers is None:
        num_workers = DEFAULT_WORKERS
    train_tf, eval_tf = get_transforms()

    ds_train = datasets.ImageFolder(data_dir, transform=train_tf)
    ds_eval = datasets.ImageFolder(data_dir, transform=eval_tf)

    train_idx, val_idx, test_idx = get_stratified_splits(ds_eval.targets)

    # persistent_workers keeps worker processes alive between epochs,
    # avoiding the Windows respawn overhead.
    persist = num_workers > 0

    train_loader = DataLoader(
        Subset(ds_train, train_idx), batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True,
        persistent_workers=persist)
    val_loader = DataLoader(
        Subset(ds_eval, val_idx), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
        persistent_workers=persist)
    test_loader = DataLoader(
        Subset(ds_eval, test_idx), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
        persistent_workers=persist)

    print(f"  Classes ({len(ds_eval.classes)}): {ds_eval.classes}")
    print(f"  Train / Val / Test: "
          f"{len(train_idx)} / {len(val_idx)} / {len(test_idx)}")

    return train_loader, val_loader, test_loader, ds_eval.classes


# ═══════════════════════════════════════════════════════════════════════
# Model Builders
# ═══════════════════════════════════════════════════════════════════════

def build_model(model_key: str, num_classes: int,
                pretrained: bool = False) -> nn.Module:
    """Build and return the specified CNN with the final layer adjusted."""
    weights = "IMAGENET1K_V1" if pretrained else None

    if model_key == "resnet":
        m = models.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif model_key == "mobilenet":
        m = models.mobilenet_v2(weights=weights)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)

    elif model_key == "vgg":
        m = models.vgg16(weights=weights)
        m.classifier[6] = nn.Linear(4096, num_classes)
        

    else:
        raise ValueError(f"Unknown model key: {model_key}")

    # Freeze backbone for pretrained VGG — without this, lr=1e-3
    # destroys the pretrained features immediately.
    if pretrained and model_key == "vgg":
        for p in m.features.parameters():
            p.requires_grad = False

    return m


def get_model_info(model: nn.Module) -> dict:
    """Return trainable/total param counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable_params": trainable, "total_params": total}


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader, device, *,
                num_epochs=20, lr=1e-3, criterion=None,
                save_dir=None):
    """
    Full training loop with validation, LR scheduling, and best-weight saving.

    Returns (model_with_best_weights, history_dict).
    """
    model = model.to(device)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "lr": []}

    dataloaders = {"train": train_loader, "val": val_loader}
    weight_path = os.path.join(save_dir, "best_model.pth") if save_dir else None

    start = time.time()
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)
        print(f"  Epoch {epoch + 1}/{num_epochs}  (lr={current_lr:.2e})")

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss, running_correct = 0.0, 0

            for inputs, labels in tqdm(dataloaders[phase],
                                       desc=f"    {phase}", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()

            n = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / n
            epoch_acc = running_correct / n
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)
            print(f"    {phase:5s}  Loss: {epoch_loss:.4f}  "
                  f"Acc: {epoch_acc:.4f}")

            if phase == "val":
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_wts = copy.deepcopy(model.state_dict())
                    if weight_path:
                        torch.save(best_wts, weight_path)
                        print(f"    ✓ Best model saved → {weight_path}")

    elapsed = time.time() - start
    print(f"  Done in {elapsed // 60:.0f}m {elapsed % 60:.0f}s — "
          f"best val acc: {best_acc:.4f}\n")

    model.load_state_dict(best_wts)

    # Save history
    if save_dir:
        hist_path = os.path.join(save_dir, "history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f)

    return model, history


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_model(model, test_loader, class_names, device):
    """
    Evaluate on test set. Returns (confusion_matrix, all_labels, all_preds).
    Also prints a classification report.
    """
    model = model.to(device).eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="  Evaluating",
                                   leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(
        all_labels, all_preds,
        target_names=class_names, zero_division=0)
    print(f"\n  Classification Report:\n{report}")

    report_dict = classification_report(
        all_labels, all_preds,
        target_names=class_names, zero_division=0, output_dict=True)

    cm = confusion_matrix(all_labels, all_preds)
    return cm, all_labels, all_preds, report_dict


# ═══════════════════════════════════════════════════════════════════════
# Plotting (saves to files)
# ═══════════════════════════════════════════════════════════════════════

def save_training_curves(history: dict, title: str, path: str):
    """Save loss, accuracy, and LR curves to a PNG file."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val")
    axes[0].set(title=f"{title} — Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "b-", label="Train")
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val")
    axes[1].set(title=f"{title} — Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs[:len(history["lr"])], history["lr"], "g-")
    axes[2].set(title=f"{title} — Learning Rate", xlabel="Epoch", ylabel="LR")
    axes[2].set_yscale("log"); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {path}")


def save_confusion_matrix(cm, class_names: list, title: str, path: str):
    """Save a confusion matrix heatmap to a PNG file."""
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(6, n * 0.5)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set(xlabel="Predicted", ylabel="True",
           title=f"{title} — Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {path}")


# ═══════════════════════════════════════════════════════════════════════
# Feature Extraction (for t-SNE / Grad-CAM)
# ═══════════════════════════════════════════════════════════════════════

def get_feature_layer(model_key: str, model: nn.Module):
    """
    Return the layer to hook into for feature extraction / Grad-CAM.
    Also returns a string name for logging.
    """
    if model_key == "resnet":
        return model.layer4[-1], "layer4[-1]"
    elif model_key == "mobilenet":
        return model.features[-1], "features[-1]"
    elif model_key == "vgg":
        return model.features[-1], "features[-1]"
    else:
        raise ValueError(f"Unknown model key: {model_key}")


def extract_features(model, model_key: str, dataloader, device):
    """
    Extract penultimate-layer features for t-SNE.
    Returns (features_array, labels_array).
    """
    model = model.to(device).eval()
    features, labels = [], []

    # Hook to capture features before the classifier
    hook_output = []

    if model_key == "resnet":
        # Hook after avgpool
        def hook_fn(module, input, output):
            hook_output.append(output.flatten(1))
        handle = model.avgpool.register_forward_hook(hook_fn)

    elif model_key == "mobilenet":
        # MobileNetV2: features → adaptive_avg_pool → classifier[0](Dropout)
        # Hook the dropout to capture the pooled features entering classifier
        handle = model.classifier[0].register_forward_hook(
            lambda m, inp, out: hook_output.append(inp[0].flatten(1)))

    elif model_key == "vgg":
        # Hook the dropout (classifier[2]) to capture features entering final Linear
        def hook_fn(module, input, output):
            hook_output.append(input[0])
        handle = model.classifier[3].register_forward_hook(hook_fn)

    with torch.no_grad():
        for inputs, lbls in tqdm(dataloader, desc="  Extracting features",
                                 leave=False):
            hook_output.clear()
            inputs = inputs.to(device)
            model(inputs)
            if hook_output:
                features.append(hook_output[0].cpu().numpy())
            labels.extend(lbls.numpy())

    handle.remove()
    return np.concatenate(features, axis=0), np.array(labels)


# ═══════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════

def print_banner(device: torch.device, extra_lines: list[str] | None = None):
    """Print a startup info banner."""
    print(f"\n{'═' * 60}")
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM   : {vram:.1f} GB")
    for line in (extra_lines or []):
        print(f"  {line}")
    print(f"{'═' * 60}\n")


def run_name(model_key: str, pretrained: bool) -> str:
    """Return a run identifier like 'resnet_scratch' or 'vgg_pretrained'."""
    tag = "pretrained" if pretrained else "scratch"
    return f"{model_key}_{tag}"