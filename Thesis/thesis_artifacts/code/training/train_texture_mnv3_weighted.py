#!/usr/bin/env python3
"""
Train a texture classifier (4 classes) using TorchVision MobileNetV3 with class-weighted loss.

This version adds class-weighted CrossEntropyLoss to handle severe class imbalance.
Two schemes available:
  - "inv": Inverse frequency weighting (simple, standard)
  - "cb": Class-balanced weights using effective number (Cui et al., CVPR 2019)

Dataset format: ImageFolder splits:
  data_root/train/<class>/*.jpg
  data_root/val/<class>/*.jpg
  data_root/test/<class>/*.jpg
"""

from __future__ import annotations

import argparse
import math
import random
from typing import Dict, Tuple, List
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def set_seed(seed: int, deterministic: bool = True):
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
        # Standard approach, treats all classes equally important
        w = counts.sum() / (len(counts) * (counts + eps))
        
    elif scheme == "cb":
        # Class-balanced: uses "effective number of samples"
        # Intuition: Re-sampling causes overlap; effective number < actual number
        # E_n = (1 - β^n) / (1 - β), weight ∝ 1/E_n = (1 - β) / (1 - β^n)
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


def build_transforms(img_size: int = 224, baseline_no_aug: bool = True):
    weights = MobileNet_V3_Small_Weights.DEFAULT

    if baseline_no_aug:
        train_tf = weights.transforms()
        eval_tf = weights.transforms()
        return train_tf, eval_tf

    mean = weights.meta.get("mean", [0.485, 0.456, 0.406])
    std = weights.meta.get("std", [0.229, 0.224, 0.225])

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_tf = weights.transforms()
    return train_tf, eval_tf


def build_loaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    use_balanced_sampling: bool,
    baseline_no_aug: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], torch.Tensor]:
    train_tf, eval_tf = build_transforms(baseline_no_aug=baseline_no_aug)

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_root / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_root / "test", transform=eval_tf)

    class_names = train_ds.classes
    class_counts = compute_class_counts(train_ds)

    sampler = None
    shuffle = True
    if use_balanced_sampling:
        labels = [y for _, y in train_ds.samples]
        counts = class_counts
        class_w = (counts.sum() / (len(class_names) * counts)).clamp(max=10.0)
        sample_w = torch.tensor([class_w[y] for y in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
        shuffle = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                              num_workers=num_workers, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=False)
    
    return train_loader, val_loader, test_loader, class_names, class_counts


def build_model(num_classes: int, device: str) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model.to(device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, num_classes: int) -> Dict:
    model.eval()
    all_y, all_pred = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy().tolist()
        all_pred.extend(pred)
        all_y.extend(y.numpy().tolist())

    true_hist = torch.bincount(torch.tensor(all_y), minlength=num_classes).tolist()
    pred_hist = torch.bincount(torch.tensor(all_pred), minlength=num_classes).tolist()

    prec, rec, f1, _ = precision_recall_fscore_support(all_y, all_pred, average="macro", zero_division=0)
    cm = confusion_matrix(all_y, all_pred).tolist()
    per_prec, per_rec, per_f1, _ = precision_recall_fscore_support(all_y, all_pred, average=None, zero_division=0)

    return {
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "per_class_precision": [float(x) for x in per_prec],
        "per_class_recall": [float(x) for x in per_rec],
        "per_class_f1": [float(x) for x in per_f1],
        "confusion_matrix": cm,
        "true_hist": true_hist,
        "pred_hist": pred_hist,
        "num_samples": len(all_y),
    }


def train(args):
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = args.device
    
    # Safety check: don't mix balanced sampling and class-weighted loss
    if args.balanced_sampling and args.class_weighted_loss:
        raise ValueError(
            "Please use either --balanced-sampling OR --class-weighted-loss (not both) "
            "for clean ablation. Mixing them confounds the experimental analysis."
        )

    train_loader, val_loader, test_loader, class_names, class_counts = build_loaders(
        Path(args.data_root), args.batch_size, args.num_workers, args.balanced_sampling,
        baseline_no_aug=not args.aug
    )

    model = build_model(num_classes=len(class_names), device=device)
    
    # Loss function: with or without class weights
    if args.class_weighted_loss:
        scheme = args.weight_scheme
        w = make_class_weights(class_counts, scheme=scheme, beta=args.cb_beta).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
        print(f"Using class-weighted loss ({scheme} scheme)")
        print(f"Class counts: {class_counts.tolist()}")
        print(f"Class weights: {[f'{x:.4f}' for x in w.cpu().tolist()]}")
    else:
        w = None
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    best_path = out / "best.pt"

    log = {
        "config": vars(args),
        "classes": class_names,
        "train_class_counts": class_counts.tolist(),
        "epochs": [],
    }
    if w is not None:
        log["train_class_weights"] = [float(x) for x in w.cpu().tolist()]

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        seen = 0  # Track actual samples processed (correct with drop_last=True)

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            seen += x.size(0)

        scheduler.step()

        train_loss = running_loss / seen
        val_metrics = evaluate(model, val_loader, device=device, num_classes=len(class_names))

        epoch_log = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val": val_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "time_s": float(time.time() - t0),
        }
        log["epochs"].append(epoch_log)

        print(f"Epoch {epoch:03d}/{args.epochs} | loss={train_loss:.4f} | val_f1={val_metrics['f1_macro']:.4f}")

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save({"model": model.state_dict(), "classes": class_names}, best_path)

    # Final test evaluation
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device=device, num_classes=len(class_names))

    log["best_val_f1"] = float(best_f1)
    log["test"] = test_metrics

    (out / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print("\nDone ✅")
    print("Best checkpoint:", best_path)
    print("Test F1 (macro):", test_metrics["f1_macro"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=str)
    ap.add_argument("--out", default="runs/texture_mnv3_weighted", type=str)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu", type=str)
    ap.add_argument("--epochs", default=20, type=int)
    ap.add_argument("--batch-size", default=32, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--weight-decay", default=1e-2, type=float)
    ap.add_argument("--num-workers", default=6, type=int, help="Data loader workers (6 for speed, thesis deadline)")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--balanced-sampling", action="store_true",
                    help="Use WeightedRandomSampler (do NOT combine with --class-weighted-loss)")
    ap.add_argument("--class-weighted-loss", action="store_true",
                    help="Use class-weighted CrossEntropyLoss to counter class imbalance")
    ap.add_argument("--weight-scheme", default="inv", choices=["inv", "cb"],
                    help="Class weight scheme: 'inv' (inverse freq) or 'cb' (class-balanced, Cui et al. CVPR 2019)")
    ap.add_argument("--cb-beta", default=0.9999, type=float,
                    help="Beta for class-balanced weights (only for --weight-scheme cb). Typical: 0.99-0.9999")

    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
