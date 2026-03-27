"""
hyperparam_search.py — Grid search over one hyperparameter for a single
dataset + model combination.

Usage:
    # Search learning rates
    python hyperparam_search.py --dataset br35h --model resnet --param lr --values 0.0001 0.0005 0.001 0.005

    # Search batch sizes
    python hyperparam_search.py --dataset 4c --model mobilenet --param batch_size --values 16 32 64

    # Search loss functions
    python hyperparam_search.py --dataset 15c --model vgg --param loss --values cross_entropy focal
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from core import (
    DATASET_CONFIGS, MODEL_REGISTRY, DEFAULT_WORKERS,
    prepare_dataset_path, build_dataloaders, build_model,
    train_model, evaluate_model,
    save_training_curves, save_confusion_matrix,
    print_banner,
)


# ═══════════════════════════════════════════════════════════════════════
# Focal Loss (alternative loss function for search)
# ═══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """Focal Loss — down-weights easy examples, focuses on hard ones."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


LOSS_FUNCTIONS = {
    "cross_entropy": lambda: nn.CrossEntropyLoss(),
    "focal":         lambda: FocalLoss(gamma=2.0),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Hyperparameter grid search",
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("--dataset", type=str, required=True,
                   choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--model", type=str, required=True,
                   choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--param", type=str, required=True,
                   choices=["lr", "batch_size", "loss"],
                   help="Which hyperparameter to search")
    p.add_argument("--values", nargs="+", required=True,
                   help="Values to try (floats for lr, ints for batch_size, "
                        "names for loss)")

    # Fixed defaults for non-searched params
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS,
                   help="DataLoader workers (0 recommended on Windows)")

    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="./outputs")

    return p.parse_args()


def save_comparison_plot(all_histories: dict, param_name: str, path: str):
    """
    Plot val accuracy and val loss curves for all searched values on one figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, history in all_histories.items():
        epochs = range(1, len(history["val_loss"]) + 1)
        axes[0].plot(epochs, history["val_loss"], label=label)
        axes[1].plot(epochs, history["val_acc"], label=label)

    axes[0].set(title=f"Val Loss vs {param_name}",
                xlabel="Epoch", ylabel="Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set(title=f"Val Accuracy vs {param_name}",
                xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Comparison plot saved → {path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nice_model = MODEL_REGISTRY[args.model]
    ds_desc = DATASET_CONFIGS[args.dataset]["description"]

    print_banner(device, [
        f"Dataset   : {ds_desc}",
        f"Model     : {nice_model}",
        f"Searching : {args.param}",
        f"Values    : {args.values}",
        f"Epochs    : {args.epochs}",
    ])

    # ── Prepare dataset ───────────────────────────────────────────────
    data_path = prepare_dataset_path(args.dataset, args.data_dir)

    # For batch_size search, we rebuild loaders per value.
    # For others, build once.
    if args.param != "batch_size":
        train_ld, val_ld, test_ld, class_names = build_dataloaders(
            data_path, args.batch_size, args.num_workers)

    # ── Search loop ───────────────────────────────────────────────────
    search_dir = os.path.join(
        args.output_dir, "hyperparam",
        f"{args.dataset}_{args.model}_{args.param}")
    os.makedirs(search_dir, exist_ok=True)

    all_histories = {}
    results = []

    for val_str in args.values:
        # Parse the value
        if args.param == "lr":
            lr = float(val_str)
            batch_size = args.batch_size
            criterion = None  # default CrossEntropyLoss
            label = f"lr={lr}"
        elif args.param == "batch_size":
            lr = args.lr
            batch_size = int(val_str)
            criterion = None
            label = f"bs={batch_size}"
        elif args.param == "loss":
            lr = args.lr
            batch_size = args.batch_size
            if val_str not in LOSS_FUNCTIONS:
                print(f"  ⚠ Unknown loss '{val_str}', "
                      f"choices: {list(LOSS_FUNCTIONS.keys())}")
                continue
            criterion = LOSS_FUNCTIONS[val_str]()
            label = f"loss={val_str}"
        else:
            raise ValueError(f"Unknown param: {args.param}")

        # Rebuild loaders if batch_size is changing
        if args.param == "batch_size":
            train_ld, val_ld, test_ld, class_names = build_dataloaders(
                data_path, batch_size, args.num_workers)

        num_classes = len(class_names)

        trial_dir = os.path.join(search_dir, val_str)
        os.makedirs(trial_dir, exist_ok=True)

        print(f"\n{'─' * 60}")
        print(f"  Trial: {label}")
        print(f"{'─' * 60}")

        model = build_model(args.model, num_classes, pretrained=False)
        model, history = train_model(
            model, train_ld, val_ld, device,
            num_epochs=args.epochs, lr=lr,
            criterion=criterion, save_dir=trial_dir)

        # Evaluate
        cm, labels, preds, report = evaluate_model(
            model, test_ld, class_names, device)

        title = f"{nice_model} — {args.dataset} — {label}"
        save_training_curves(history, title,
                             os.path.join(trial_dir, "curves.png"))
        save_confusion_matrix(cm, class_names, title,
                              os.path.join(trial_dir, "confusion.png"))

        all_histories[label] = history
        results.append({
            "value": val_str,
            "label": label,
            "best_val_acc": max(history["val_acc"]),
            "final_val_loss": history["val_loss"][-1],
            "test_acc": report["accuracy"],
        })

    # ── Comparison ────────────────────────────────────────────────────
    save_comparison_plot(
        all_histories, args.param,
        os.path.join(search_dir, "comparison.png"))

    # Summary
    print(f"\n{'═' * 60}")
    print(f"  HYPERPARAMETER SEARCH RESULTS — {args.param}")
    print(f"{'═' * 60}")
    print(f"  {'Value':<20s}  {'Val Acc':>8s}  {'Test Acc':>8s}")
    print(f"  {'─' * 20}  {'─' * 8}  {'─' * 8}")

    best = max(results, key=lambda r: r["best_val_acc"])
    for r in results:
        marker = " ★" if r["value"] == best["value"] else ""
        print(f"  {r['label']:<20s}  {r['best_val_acc']:>8.4f}  "
              f"{r['test_acc']:>8.4f}{marker}")

    print(f"\n  Best: {best['label']} "
          f"(val={best['best_val_acc']:.4f}, test={best['test_acc']:.4f})")
    print(f"  Results saved to: {os.path.abspath(search_dir)}/")
    print(f"{'═' * 60}\n")

    with open(os.path.join(search_dir, "search_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()