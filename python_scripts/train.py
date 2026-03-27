"""
train.py — Train brain tumor CNN classifiers.

Runs any combination of {dataset} × {model} × {scratch | pretrained}.

Usage:
    python train.py                                    # all 9 scratch models
    python train.py --dataset br35h --model resnet     # single run
    python train.py --dataset 4c --model resnet --pretrained  # transfer learning
    python train.py --all                              # all 11 (9 scratch + 2 pretrained)
    python train.py --epochs 30 --batch-size 64        # override hyperparams
"""

import argparse
import json
import os
import sys

import torch

from core import (
    DATASET_CONFIGS, MODEL_REGISTRY, DEFAULT_WORKERS,
    prepare_dataset_path, build_dataloaders, build_model, get_model_info,
    train_model, evaluate_model,
    save_training_curves, save_confusion_matrix,
    print_banner, run_name,
)


# ── Predefined transfer-learning pairs ────────────────────────────────
TRANSFER_LEARNING_RUNS = [
    ("4c",  "resnet"),   # ResNet18 + 4-class
    ("15c", "vgg"),      # VGG16 + 15-class
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Train brain tumor classifiers",
        formatter_class=argparse.RawTextHelpFormatter)

    # What to run
    g = p.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true",
                   help="Run all 11 models (9 scratch + 2 pretrained)")
    g.add_argument("--all-scratch", action="store_true",
                   help="Run all 9 from-scratch models")

    p.add_argument("--dataset", type=str,
                   choices=list(DATASET_CONFIGS.keys()),
                   help="Single dataset to train on")
    p.add_argument("--model", type=str,
                   choices=list(MODEL_REGISTRY.keys()),
                   help="Single model to train")
    p.add_argument("--pretrained", action="store_true",
                   help="Use ImageNet pretrained weights (transfer learning)")

    # Hyperparams
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS,
                   help="DataLoader workers (0 recommended on Windows)")

    # Paths
    p.add_argument("--data-dir", type=str, default=None,
                   help="Override dataset root directory")
    p.add_argument("--output-dir", type=str, default="./outputs",
                   help="Root output directory")

    return p.parse_args()


def build_run_list(args) -> list[dict]:
    """
    Build the list of (dataset, model, pretrained) runs to execute.
    """
    runs = []

    if args.all:
        # 9 scratch
        for ds in DATASET_CONFIGS:
            for mk in MODEL_REGISTRY:
                runs.append({"dataset": ds, "model": mk, "pretrained": False})
        # 2 transfer learning
        for ds, mk in TRANSFER_LEARNING_RUNS:
            runs.append({"dataset": ds, "model": mk, "pretrained": True})

    elif args.all_scratch:
        for ds in DATASET_CONFIGS:
            for mk in MODEL_REGISTRY:
                runs.append({"dataset": ds, "model": mk, "pretrained": False})

    elif args.dataset and args.model:
        runs.append({
            "dataset": args.dataset,
            "model": args.model,
            "pretrained": args.pretrained,
        })

    elif args.dataset:
        # All 3 models on one dataset
        for mk in MODEL_REGISTRY:
            runs.append({
                "dataset": args.dataset, "model": mk,
                "pretrained": args.pretrained,
            })

    elif args.model:
        # One model on all 3 datasets
        for ds in DATASET_CONFIGS:
            runs.append({
                "dataset": ds, "model": args.model,
                "pretrained": args.pretrained,
            })

    else:
        # Default: all 9 scratch
        for ds in DATASET_CONFIGS:
            for mk in MODEL_REGISTRY:
                runs.append({"dataset": ds, "model": mk, "pretrained": False})

    return runs


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = build_run_list(args)

    print_banner(device, [
        f"Epochs       : {args.epochs}",
        f"Batch size   : {args.batch_size}",
        f"Learning rate: {args.lr}",
        f"Runs planned : {len(runs)}",
    ])

    # ── Prepare datasets (only once per unique dataset) ───────────────
    dataset_paths = {}
    dataset_loaders = {}

    for ds_key in sorted(set(r["dataset"] for r in runs)):
        cfg = DATASET_CONFIGS[ds_key]
        print(f"[Dataset] {cfg['description']}")
        path = prepare_dataset_path(ds_key, args.data_dir)
        dataset_paths[ds_key] = path

        train_ld, val_ld, test_ld, classes = build_dataloaders(
            path, args.batch_size, args.num_workers)
        dataset_loaders[ds_key] = {
            "train": train_ld, "val": val_ld, "test": test_ld,
            "classes": classes,
        }
        print()

    # ── Execute runs ──────────────────────────────────────────────────
    summary = []

    for i, run in enumerate(runs, 1):
        ds_key = run["dataset"]
        mk = run["model"]
        pretrained = run["pretrained"]
        rn = run_name(mk, pretrained)
        nice_name = MODEL_REGISTRY[mk]
        ds_desc = DATASET_CONFIGS[ds_key]["description"]

        save_dir = os.path.join(args.output_dir, ds_key, rn)
        os.makedirs(save_dir, exist_ok=True)

        print(f"{'─' * 60}")
        print(f"  [{i}/{len(runs)}] {nice_name} on {ds_desc}")
        print(f"  Mode: {'pretrained (transfer learning)' if pretrained else 'from scratch'}")
        print(f"  Save: {save_dir}")
        print(f"{'─' * 60}")

        loaders = dataset_loaders[ds_key]
        num_classes = len(loaders["classes"])

        model = build_model(mk, num_classes, pretrained)
        info = get_model_info(model)
        print(f"  Trainable params: {info['trainable_params']:,}")

        model, history = train_model(
            model, loaders["train"], loaders["val"], device,
            num_epochs=args.epochs, lr=args.lr, save_dir=save_dir)

        # Curves
        title = f"{nice_name} — {ds_key}"
        save_training_curves(
            history, title,
            os.path.join(save_dir, "curves.png"))

        # Test evaluation
        cm, labels, preds, report = evaluate_model(
            model, loaders["test"], loaders["classes"], device)
        save_confusion_matrix(
            cm, loaders["classes"], title,
            os.path.join(save_dir, "confusion.png"))

        # Save evaluation report
        with open(os.path.join(save_dir, "eval_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        best_val = max(history["val_acc"])
        test_acc = report["accuracy"]
        summary.append({
            "run": f"{rn} / {ds_key}",
            "best_val_acc": best_val,
            "test_acc": test_acc,
        })

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  SUMMARY")
    print(f"{'═' * 60}")
    print(f"  {'Run':<35s}  {'Val Acc':>8s}  {'Test Acc':>8s}")
    print(f"  {'─' * 35}  {'─' * 8}  {'─' * 8}")
    for s in summary:
        print(f"  {s['run']:<35s}  {s['best_val_acc']:>8.4f}  "
              f"{s['test_acc']:>8.4f}")
    print(f"\n  All outputs saved to: {os.path.abspath(args.output_dir)}/")
    print(f"{'═' * 60}\n")

    # Save summary JSON
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()