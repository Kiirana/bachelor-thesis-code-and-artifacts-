#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train MobileNetV3-Small texture classifier with ROBUST AUGMENTATION.

Motivation:
-----------
Baseline analysis revealed severe robustness failures under geometric transformations:
- 90°/270° rotation: 56-58% F1 (expected: ~95%+ for rotation-invariant textures)
- Perspective squeeze: 10% F1 (severe collapse)
- Heavy shear: 73% F1 (significant degradation)

Hypothesis:
-----------
We hypothesize that training with geometric augmentation (rotation ±180°, perspective
distortion, shear) will improve robustness to these transformations while maintaining
reasonable performance on clean data. This ablation isolates the effect of augmentation
by keeping all other factors constant (architecture, loss, sampling strategy).

Augmentation Modes:
-------------------
- rotation_perspective: RandomRotation(±180°) + RandomPerspective(0.3) [RECOMMENDED]
  Addresses the two most severe failure modes (90° rotation, perspective squeeze)

- full_robust: Above + RandomAffine(shear=±20°)
  Optional: Use if shear robustness is insufficient with rotation_perspective

Usage:
------
# Recommended: rotation + perspective (addresses worst failures)
python3 train_texture_mnv3_robust.py \
  --data-root texture_data_final \
  --out runs/texture_rotation_perspective \
  --epochs 20 \
  --seed 42 \
  --mode rotation_perspective

# Optional: full augmentation (if needed)
python3 train_texture_mnv3_robust.py \
  --data-root texture_data_final \
  --out runs/texture_full_robust \
  --epochs 20 \
  --seed 42 \
  --mode full_robust
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from sklearn.metrics import precision_recall_fscore_support


# ---------------------------
# Class weighting for imbalance
# ---------------------------
def compute_class_counts(train_ds: datasets.ImageFolder) -> torch.Tensor:
    """Extract class counts from ImageFolder dataset."""
    labels = [y for _, y in train_ds.samples]
    counts = torch.bincount(torch.tensor(labels), minlength=len(train_ds.classes)).float()
    return counts


def make_class_weights(counts: torch.Tensor, scheme: str = "inv", beta: float = 0.9999) -> torch.Tensor:
    """
    Compute class weights for CrossEntropyLoss.
    
    Args:
        counts: Per-class sample counts [n_0, n_1, ..., n_K]
        scheme: Weighting scheme
          - "inv": Inverse frequency weights (standard, simple)
                   w_c = N / (K * n_c)
          - "cb": Class-balanced weights using effective number of samples
                  (Cui et al., CVPR 2019)
                  w_c ∝ (1 - β) / (1 - β^n_c)
        beta: Beta parameter for "cb" scheme (typical: 0.99 to 0.9999)
    
    Returns:
        Tensor of weights [w_0, w_1, ..., w_K], normalized to mean=1
    
    References:
        - Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
        - He & Garcia, "Learning from Imbalanced Data", IEEE TKDE 2009
    """
    eps = 1e-8
    
    if scheme == "inv":
        # Inverse frequency: w_c = total / (K * n_c)
        w = counts.sum() / (len(counts) * (counts + eps))
        
    elif scheme == "cb":
        # Class-balanced: uses "effective number of samples"
        beta = float(beta)
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"beta must be in [0, 1). Got: {beta}")
        effective_num = 1.0 - torch.pow(torch.tensor(beta), counts)
        w = (1.0 - beta) / (effective_num + eps)
        
    else:
        raise ValueError(f"Unknown weight scheme: {scheme}. Choose 'inv' or 'cb'.")
    
    # Normalize weights so mean weight = 1.0 (helps training stability)
    w = w / (w.mean() + eps)
    return w


# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set seed for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value
        deterministic: If True, use deterministic CUDNN algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------
# ROBUST AUGMENTATION (addresses robustness failures)
# ---------------------------
def build_transforms(mode: str = "rotation_perspective") -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build train and eval transforms.
    
    Args:
        mode: "baseline" (no aug), 
              "rotation_perspective" (high priority: rotation + perspective only),
              "full_robust" (all: rotation + perspective + shear + photometric)
    
    Returns:
        (train_transform, eval_transform)
    
    Note:
        For thesis ablation, run "rotation_perspective" first, then "full_robust" if needed.
        This isolates the effect of each augmentation family.
    """
    weights = MobileNet_V3_Small_Weights.DEFAULT
    # Use standard ImageNet normalization (weights.meta may not contain mean/std in all versions)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    # Eval transform: official TorchVision preprocessing (deterministic)
    eval_tf = weights.transforms()
    
    # Post-augmentation preprocessing (Resize/CenterCrop/ToTensor/Normalize)
    # Extracted to avoid Compose-in-Compose issues
    post_tf = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if mode == "baseline":
        return eval_tf, eval_tf

    # Fill value: use ImageNet mean to avoid black borders after geometric transforms
    fill = tuple(int(m * 255) for m in mean)

    # Mode: rotation_perspective (HIGH PRIORITY - addresses worst failures)
    if mode == "rotation_perspective":
        train_tf = transforms.Compose([
            # GEOMETRIC AUGMENTATION (addresses catastrophic failures)
            transforms.RandomRotation(
                degrees=180,  # -180 to +180 → covers 90°/270° rotations (56% F1 → 90%+)
                interpolation=InterpolationMode.BILINEAR,
                fill=fill
            ),
            transforms.RandomPerspective(
                distortion_scale=0.3,  # Addresses perspective squeeze collapse (10% F1 → 70%+)
                p=0.5,  # Apply to 50% of samples
                interpolation=InterpolationMode.BILINEAR,
                fill=fill
            ),
            post_tf,  # Standard preprocessing (PIL → Tensor)
        ])
        return train_tf, eval_tf

    # Mode: full_robust (ALL AUGMENTATIONS - if rotation_perspective insufficient)
    if mode == "full_robust":
        train_tf = transforms.Compose([
            # GEOMETRIC AUGMENTATION
            transforms.RandomRotation(
                degrees=180,  # Covers 90°/270° rotations
                interpolation=InterpolationMode.BILINEAR,
                fill=fill
            ),
            transforms.RandomPerspective(
                distortion_scale=0.3,  # Perspective squeeze
                p=0.5,
                interpolation=InterpolationMode.BILINEAR,
                fill=fill
            ),
            transforms.RandomAffine(
                degrees=0,  # No additional rotation (already handled)
                shear=(-20, 20, -20, 20),  # Addresses heavy shear (73% F1 → 85%+)
                interpolation=InterpolationMode.BILINEAR,
                fill=fill
            ),
            
            # PHOTOMETRIC AUGMENTATION (optional, baseline already robust)
            transforms.ColorJitter(brightness=0.2, contrast=0.2),

            post_tf,  # Standard preprocessing
        ])
        return train_tf, eval_tf

    raise ValueError(f"Unknown mode: {mode}. Choose: 'baseline', 'rotation_perspective', 'full_robust'")


# ---------------------------
# Model
# ---------------------------
def build_model(num_classes: int, device: str) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model.to(device)


# ---------------------------
# Data loaders
# ---------------------------
def build_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int,
    mode: str = "rotation_perspective"
) -> Tuple[DataLoader, DataLoader, DataLoader, list, torch.Tensor]:
    """
    Build train/val/test loaders.
    
    Returns:
        (train_loader, val_loader, test_loader, classes, class_counts)
    """
    train_tf, eval_tf = build_transforms(mode=mode)

    train_ds = datasets.ImageFolder(Path(data_root) / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(Path(data_root) / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(Path(data_root) / "test", transform=eval_tf)

    class_counts = compute_class_counts(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    return train_loader, val_loader, test_loader, train_ds.classes, class_counts


# ---------------------------
# Metrics
# ---------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, num_classes: int) -> dict:
    model.eval()
    all_y = []
    all_pred = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().tolist()
        all_pred.extend(pred)
        all_y.extend(y.tolist())

    prec, rec, f1, _ = precision_recall_fscore_support(all_y, all_pred, average="macro", zero_division=0)
    per_prec, per_rec, per_f1, _ = precision_recall_fscore_support(all_y, all_pred, average=None, zero_division=0)

    true_hist = torch.bincount(torch.tensor(all_y), minlength=num_classes).tolist()
    pred_hist = torch.bincount(torch.tensor(all_pred), minlength=num_classes).tolist()

    return {
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "per_class_precision": [float(v) for v in per_prec],
        "per_class_recall": [float(v) for v in per_rec],
        "per_class_f1": [float(v) for v in per_f1],
        "true_hist": true_hist,
        "pred_hist": pred_hist,
    }


# ---------------------------
# Training loop
# ---------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str
) -> float:
    model.train()
    total_loss = 0.0
    seen = 0  # Track actual samples processed (correct with variable batch sizes)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)  # Accumulate weighted loss
        seen += x.size(0)

    return total_loss / seen


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    classes: list,
    class_counts: torch.Tensor,
    args
) -> dict:
    device = args.device
    num_classes = len(classes)

    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss function: with or without class weights
    if args.class_weighted_loss:
        scheme = args.weight_scheme
        class_weights = make_class_weights(class_counts, scheme=scheme, beta=args.cb_beta).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"  Class-weighted loss: {scheme} scheme")
        print(f"  Class counts: {class_counts.tolist()}")
        print(f"  Class weights: {[f'{x:.4f}' for x in class_weights.cpu().tolist()]}")
    else:
        class_weights = None
        criterion = nn.CrossEntropyLoss()

    # Tracking
    best_val_f1 = 0.0
    log_history = []

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Device: {device}")
    print(f"  Augmentation mode: {args.mode}")
    if args.mode == "rotation_perspective":
        print(f"    → RandomRotation(±180°) + RandomPerspective(0.3)")
    elif args.mode == "full_robust":
        print(f"    → Rotation + Perspective + Shear + ColorJitter")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Seed: {args.seed}")
    print(f"  Classes: {classes}")
    if args.class_weighted_loss:
        print(f"  Enhancement: Class-weighted loss enabled")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = evaluate(model, val_loader, device, num_classes)
        val_f1 = val_metrics["f1_macro"]

        # Log
        epoch_time = time.time() - t0
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_f1,
            "val_per_class_f1": val_metrics["per_class_f1"],
            "val_true_hist": val_metrics["true_hist"],
            "val_pred_hist": val_metrics["pred_hist"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
        }
        log_history.append(log_entry)

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Time: {epoch_time:.1f}s")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt = {
                "model": model.state_dict(),
                "classes": classes,
                "epoch": epoch,
                "val_f1": val_f1,
            }
            torch.save(ckpt, Path(args.out) / "best.pt")
            print(f"  → Saved best model (val F1: {val_f1:.4f})")

        scheduler.step()

    # Test with best model
    print(f"\n{'='*60}")
    print("Loading best model for test evaluation...")
    best_ckpt = torch.load(Path(args.out) / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    
    test_metrics = evaluate(model, test_loader, device, num_classes)
    print(f"Test Results:")
    print(f"  Macro F1: {test_metrics['f1_macro']:.4f}")
    print(f"  Per-class F1: {test_metrics['per_class_f1']}")
    print(f"  Pred histogram: {test_metrics['pred_hist']}")
    print(f"{'='*60}\n")

    # Save training log
    log = {
        "args": vars(args),
        "classes": classes,
        "train_class_counts": class_counts.tolist(),
        "augmentation": "robust",
        "train_history": log_history,
        "test": test_metrics,
        "best_val_f1": best_val_f1,
        "best_epoch": best_ckpt["epoch"],
    }
    if class_weights is not None:
        log["train_class_weights"] = [float(x) for x in class_weights.cpu().tolist()]

    log_path = Path(args.out) / "train_log.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"Saved training log: {log_path}")

    return log


def main():
    ap = argparse.ArgumentParser(description="Train robust texture classifier")
    ap.add_argument("--data-root", required=True, type=str, help="ImageFolder root with train/val/test")
    ap.add_argument("--out", required=True, type=str, help="Output directory for checkpoints and logs")
    ap.add_argument("--epochs", default=20, type=int, help="Number of training epochs (default: 20)")
    ap.add_argument("--batch-size", default=32, type=int)
    ap.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    ap.add_argument("--weight-decay", default=1e-2, type=float)
    ap.add_argument("--num-workers", default=6, type=int, help="Data loader workers (6 for speed, thesis deadline)")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu", type=str)
    ap.add_argument("--mode", default="rotation_perspective", 
                    choices=["baseline", "rotation_perspective", "full_robust"], 
                    help="Augmentation mode: 'baseline' (no aug), 'rotation_perspective' (high priority), 'full_robust' (all)")
    
    # Enhancement: Class-weighted loss for handling imbalance
    ap.add_argument("--class-weighted-loss", action="store_true",
                    help="Use class-weighted CrossEntropyLoss to handle class imbalance")
    ap.add_argument("--weight-scheme", default="inv", choices=["inv", "cb"],
                    help="Class weight scheme: 'inv' (inverse freq) or 'cb' (class-balanced, Cui et al. CVPR 2019)")
    ap.add_argument("--cb-beta", default=0.9999, type=float,
                    help="Beta for class-balanced weights (only for --weight-scheme cb). Typical: 0.99-0.9999")
    
    args = ap.parse_args()

    # Setup
    set_seed(args.seed)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, test_loader, classes, class_counts = build_loaders(
        args.data_root,
        args.batch_size,
        args.num_workers,
        mode=args.mode
    )

    # Model
    model = build_model(num_classes=len(classes), device=args.device)

    # Train
    train(model, train_loader, val_loader, test_loader, classes, class_counts, args)


if __name__ == "__main__":
    main()
