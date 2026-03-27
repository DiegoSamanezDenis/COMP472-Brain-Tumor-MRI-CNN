"""
visualize.py — Generate t-SNE and Grad-CAM visualizations for trained models.

Usage:
    # t-SNE for a trained model
    python visualize.py --dataset br35h --model resnet --tsne \\
        --weights outputs/br35h/resnet_scratch/best_model.pth

    # Grad-CAM for a trained model (saves a grid of examples)
    python visualize.py --dataset 4c --model mobilenet --gradcam \\
        --weights outputs/4c/mobilenet_scratch/best_model.pth

    # Both at once
    python visualize.py --dataset 15c --model vgg --tsne --gradcam \\
        --weights outputs/15c/vgg_pretrained/best_model.pth

    # Auto-discover ALL trained models and generate both visualizations
    python visualize.py --all
    python visualize.py --all --output-dir ./outputs  # custom output root
"""

import argparse
import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torchvision import datasets, transforms

from core import (
    DATASET_CONFIGS, MODEL_REGISTRY, DEFAULT_WORKERS, IMAGENET_MEAN, IMAGENET_STD,
    prepare_dataset_path, build_dataloaders, build_model,
    extract_features, get_feature_layer, print_banner,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize trained models with t-SNE and Grad-CAM",
        formatter_class=argparse.RawTextHelpFormatter)

    # ── Auto-discover mode ────────────────────────────────────────────
    p.add_argument("--all", action="store_true",
                   help="Auto-discover all trained models in the output\n"
                        "directory and generate both t-SNE and Grad-CAM\n"
                        "for each one.")

    # ── Single-run mode (required unless --all) ───────────────────────
    p.add_argument("--dataset", type=str,
                   choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--model", type=str,
                   choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--weights", type=str,
                   help="Path to saved .pth weights")

    p.add_argument("--tsne", action="store_true", help="Generate t-SNE plot")
    p.add_argument("--gradcam", action="store_true",
                   help="Generate Grad-CAM grid")
    p.add_argument("--pretrained", action="store_true",
                   help="Model was trained with pretrained weights "
                        "(needed to build correct architecture)")

    p.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS,
                   help="DataLoader workers (0 recommended on Windows)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="./outputs",
                   help="Root output directory (used by --all to scan for\n"
                        "trained models, default: ./outputs)")
    p.add_argument("--gradcam-samples", type=int, default=8,
                   help="Number of Grad-CAM examples to generate")
    p.add_argument("--tsne-perplexity", type=float, default=30.0)

    args = p.parse_args()

    # Validation: single-run mode needs dataset, model, weights
    if not args.all:
        missing = []
        if not args.dataset:
            missing.append("--dataset")
        if not args.model:
            missing.append("--model")
        if not args.weights:
            missing.append("--weights")
        if missing:
            p.error(f"Single-run mode requires {', '.join(missing)} "
                    f"(or use --all to auto-discover trained models)")

    return args


# ═══════════════════════════════════════════════════════════════════════
# t-SNE
# ═══════════════════════════════════════════════════════════════════════

def generate_tsne(model, model_key, test_loader, class_names, device,
                  save_path, perplexity=30.0):
    """Extract features, run t-SNE, and save the scatter plot."""
    print("\n  Generating t-SNE visualization...")
    features, labels = extract_features(model, model_key, test_loader, device)

    print(f"  Feature shape: {features.shape}")
    print(f"  Running t-SNE (perplexity={perplexity})...")

    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=42)
    embeddings = tsne.fit_transform(features)

    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap("tab20" if n_classes > 10 else "tab10")

    for i, cls in enumerate(class_names):
        mask = labels == i
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   c=[cmap(i / max(n_classes - 1, 1))],
                   label=cls, s=15, alpha=0.7)

    ax.legend(fontsize=8, markerscale=2, loc="best")
    ax.set_title(f"t-SNE — {MODEL_REGISTRY[model_key]} — {len(class_names)} classes")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  t-SNE saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# Grad-CAM
# ═══════════════════════════════════════════════════════════════════════

class GradCAM:
    """Simple Grad-CAM implementation for a target convolutional layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for a single image tensor.
        Returns a 2D numpy array (H, W) in [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        # Global average pool of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear",
                            align_corners=False)

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, target_class


def denormalize(tensor):
    """Reverse ImageNet normalization for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def generate_gradcam_grid(model, model_key, test_loader, class_names,
                          device, save_path, num_samples=8):
    """Generate a grid of Grad-CAM overlays on random test images."""
    print("\n  Generating Grad-CAM visualization...")

    target_layer, layer_name = get_feature_layer(model_key, model)
    print(f"  Hooking layer: {layer_name}")

    model = model.to(device).eval()
    cam_gen = GradCAM(model, target_layer)

    # Collect some test images
    all_images, all_labels = [], []
    for inputs, labels in test_loader:
        all_images.append(inputs)
        all_labels.append(labels)
        if sum(x.size(0) for x in all_images) >= num_samples * 5:
            break
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)

    # Random sample
    indices = random.sample(range(len(all_images)),
                            min(num_samples, len(all_images)))

    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 5, rows * 2.8))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for i, idx in enumerate(indices):
        row = i // cols
        col_base = (i % cols) * 2

        img_tensor = all_images[idx].unsqueeze(0).to(device)
        true_label = all_labels[idx].item()

        cam, pred_class = cam_gen.generate(img_tensor)
        img_np = denormalize(all_images[idx])

        # Original image
        axes[row, col_base].imshow(img_np)
        axes[row, col_base].set_title(
            f"True: {class_names[true_label]}", fontsize=8)
        axes[row, col_base].axis("off")

        # Overlay
        axes[row, col_base + 1].imshow(img_np)
        axes[row, col_base + 1].imshow(cam, cmap="jet", alpha=0.4)
        color = "green" if pred_class == true_label else "red"
        axes[row, col_base + 1].set_title(
            f"Pred: {class_names[pred_class]}", fontsize=8, color=color)
        axes[row, col_base + 1].axis("off")

    # Hide unused axes
    for ax in axes.flat:
        if not ax.has_data():
            ax.set_visible(False)

    fig.suptitle(f"Grad-CAM — {MODEL_REGISTRY[model_key]}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grad-CAM saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# Auto-discovery
# ═══════════════════════════════════════════════════════════════════════

def discover_trained_models(output_dir: str) -> list[dict]:
    """
    Scan the output directory for best_model.pth files and parse the
    dataset / model / pretrained info from the path structure:
        {output_dir}/{dataset_key}/{model_key}_{scratch|pretrained}/best_model.pth
    """
    pattern = os.path.join(output_dir, "**", "best_model.pth")
    found = sorted(glob.glob(pattern, recursive=True))

    runs = []
    for weight_path in found:
        # e.g. outputs/4c/resnet_pretrained/best_model.pth
        run_dir = os.path.dirname(weight_path)         # .../resnet_pretrained
        run_folder = os.path.basename(run_dir)          # resnet_pretrained
        ds_dir = os.path.dirname(run_dir)               # .../4c
        dataset_key = os.path.basename(ds_dir)          # 4c

        # Skip folders inside hyperparam search results
        if "hyperparam" in weight_path:
            continue

        # Validate dataset key
        if dataset_key not in DATASET_CONFIGS:
            print(f"  ⚠ Skipping unknown dataset '{dataset_key}' in: "
                  f"{weight_path}")
            continue

        # Parse model key and pretrained flag from folder name
        # Expected format: {model_key}_{scratch|pretrained}
        pretrained = run_folder.endswith("_pretrained")
        if pretrained:
            model_key = run_folder.rsplit("_pretrained", 1)[0]
        elif run_folder.endswith("_scratch"):
            model_key = run_folder.rsplit("_scratch", 1)[0]
        else:
            # Non-standard folder name — try the full name as model key
            model_key = run_folder

        if model_key not in MODEL_REGISTRY:
            print(f"  ⚠ Skipping unknown model '{model_key}' in: "
                  f"{weight_path}")
            continue

        runs.append({
            "dataset": dataset_key,
            "model": model_key,
            "pretrained": pretrained,
            "weights": weight_path,
            "out_dir": run_dir,
        })

    return runs


# ═══════════════════════════════════════════════════════════════════════
# Single-model visualization runner
# ═══════════════════════════════════════════════════════════════════════

def run_single(dataset_key, model_key, weight_path, out_dir, device,
               *, pretrained=False, do_tsne=True, do_gradcam=True,
               data_dir=None, batch_size=32, num_workers=None,
               gradcam_samples=8, tsne_perplexity=30.0):
    """Load one model and generate requested visualizations."""
    nice_model = MODEL_REGISTRY[model_key]
    ds_desc = DATASET_CONFIGS[dataset_key]["description"]
    tag = "pretrained" if pretrained else "scratch"

    print(f"\n{'─' * 60}")
    print(f"  {nice_model} ({tag}) on {ds_desc}")
    print(f"  Weights : {weight_path}")
    print(f"  Output  : {out_dir}")
    print(f"{'─' * 60}")

    # Prepare dataset
    data_path = prepare_dataset_path(dataset_key, data_dir)
    _, _, test_loader, class_names = build_dataloaders(
        data_path, batch_size, num_workers)

    num_classes = len(class_names)

    # Build model and load weights
    model = build_model(model_key, num_classes, pretrained=False)
    state = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    print(f"  Loaded weights OK ({num_classes} classes)")

    os.makedirs(out_dir, exist_ok=True)

    if do_tsne:
        generate_tsne(
            model, model_key, test_loader, class_names, device,
            os.path.join(out_dir, "tsne.png"),
            perplexity=tsne_perplexity)

    if do_gradcam:
        generate_gradcam_grid(
            model, model_key, test_loader, class_names, device,
            os.path.join(out_dir, "gradcam.png"),
            num_samples=gradcam_samples)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── --all mode: discover and visualize everything ─────────────────
    if args.all:
        runs = discover_trained_models(args.output_dir)

        if not runs:
            print(f"No trained models found in: "
                  f"{os.path.abspath(args.output_dir)}/")
            print(f"  Expected structure: "
                  f"<output_dir>/<dataset>/<model>_<scratch|pretrained>/"
                  f"best_model.pth")
            return

        print_banner(device, [
            f"Mode           : Auto-discover (--all)",
            f"Output root    : {os.path.abspath(args.output_dir)}",
            f"Models found   : {len(runs)}",
            f"Visualizations : t-SNE + Grad-CAM",
        ])

        for i, run in enumerate(runs, 1):
            tag = "pretrained" if run["pretrained"] else "scratch"
            print(f"\n  [{i}/{len(runs)}] {run['model']}_{tag} / "
                  f"{run['dataset']}")

            try:
                run_single(
                    run["dataset"], run["model"], run["weights"],
                    run["out_dir"], device,
                    pretrained=run["pretrained"],
                    do_tsne=True, do_gradcam=True,
                    data_dir=args.data_dir,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    gradcam_samples=args.gradcam_samples,
                    tsne_perplexity=args.tsne_perplexity,
                )
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue

        print(f"\n{'═' * 60}")
        print(f"  ALL DONE — {len(runs)} models visualized")
        print(f"{'═' * 60}\n")
        return

    # ── Single-run mode ───────────────────────────────────────────────
    if not args.tsne and not args.gradcam:
        print("Nothing to do — specify --tsne and/or --gradcam")
        return

    out_dir = args.output_dir if args.output_dir != "./outputs" else \
        os.path.dirname(args.weights)

    run_single(
        args.dataset, args.model, args.weights, out_dir, device,
        pretrained=args.pretrained,
        do_tsne=args.tsne, do_gradcam=args.gradcam,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gradcam_samples=args.gradcam_samples,
        tsne_perplexity=args.tsne_perplexity,
    )


if __name__ == "__main__":
    main()