#!/usr/bin/env python3
"""
Train a texture classifier (4 classes) using TorchVision MobileNetV3-Large.

Mirrors train_texture_mnv3.py (Small baseline) exactly — only the backbone
and weight object change:  mobilenet_v3_large / MobileNet_V3_Large_Weights

Dataset format: ImageFolder splits
  data_root/train/<class>/*.jpg
  data_root/val/<class>/*.jpg
  data_root/test/<class>/*.jpg

Reference:
  https://pytorch.org/vision/stable/models/mobilenetv3.html
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.transforms import InterpolationMode


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------

def build_transforms(baseline_no_aug: bool = True):
    """Return (train_tf, eval_tf) using the official Large-weight transforms."""
    weights = MobileNet_V3_Large_Weights.DEFAULT

    if baseline_no_aug:
        # Pure baseline: same deterministic transform for train and eval
        train_tf = weights.transforms()
        eval_tf = weights.transforms()
        return train_tf, eval_tf

    # Augmented branch (when --aug is passed)
    mean = weights.meta.get("mean", [0.485, 0.456, 0.406])
    std  = weights.meta.get("std",  [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_tf = weights.transforms()
    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def build_loaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    use_balanced_sampling: bool,
    baseline_no_aug: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    train_tf, eval_tf = build_transforms(baseline_no_aug=baseline_no_aug)

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(data_root / "val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(data_root / "test",  transform=eval_tf)

    class_names = train_ds.classes

    sampler = None
    shuffle = True
    if use_balanced_sampling:
        labels   = [y for _, y in train_ds.samples]
        counts   = torch.bincount(torch.tensor(labels), minlength=len(class_names)).float()
        class_w  = (counts.sum() / (len(class_names) * counts)).clamp(max=10.0)
        sample_w = torch.tensor([class_w[y] for y in labels], dtype=torch.double)
        sampler  = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
        shuffle  = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                              num_workers=num_workers, pin_memory=False, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader, test_loader, class_names


# ---------------------------------------------------------------------------
# Model (MobileNetV3-Large)
# ---------------------------------------------------------------------------

def build_model(num_classes: int, device: str) -> nn.Module:
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model   = mobilenet_v3_large(weights=weights)

    # Replace only the final linear layer (head fine-tuning)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model.to(device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, num_classes: int) -> Dict:
    model.eval()
    all_y, all_pred = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred   = logits.argmax(dim=1).cpu().numpy().tolist()
        all_pred.extend(pred)
        all_y.extend(y.numpy().tolist())

    true_hist = torch.bincount(torch.tensor(all_y),    minlength=num_classes).tolist()
    pred_hist = torch.bincount(torch.tensor(all_pred), minlength=num_classes).tolist()

    prec, rec, f1, _ = precision_recall_fscore_support(all_y, all_pred, average="macro", zero_division=0)
    cm = confusion_matrix(all_y, all_pred).tolist()
    per_prec, per_rec, per_f1, _ = precision_recall_fscore_support(all_y, all_pred, average=None, zero_division=0)

    return {
        "precision_macro": float(prec),
        "recall_macro":    float(rec),
        "f1_macro":        float(f1),
        "per_class_precision": [float(v) for v in per_prec],
        "per_class_recall":    [float(v) for v in per_rec],
        "per_class_f1":        [float(v) for v in per_f1],
        "confusion_matrix":    cm,
        "true_hist":           true_hist,
        "pred_hist":           pred_hist,
        "num_samples":         len(all_y),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = args.device
    train_loader, val_loader, test_loader, class_names = build_loaders(
        Path(args.data_root), args.batch_size, args.num_workers, args.balanced_sampling,
        baseline_no_aug=not args.aug,
    )

    model     = build_model(num_classes=len(class_names), device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1   = -1.0
    best_path = out / "best.pt"

    log = {
        "config":  vars(args),
        "classes": class_names,
        "model":   "mobilenet_v3_large",
        "epochs":  [],
    }

    print(f"Training MobileNetV3-Large → {out}")
    print(f"Device: {device}  |  Classes: {class_names}")
    print(f"Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            seen         += x.size(0)

        scheduler.step()

        train_loss  = running_loss / seen
        val_metrics = evaluate(model, val_loader, device=device, num_classes=len(class_names))

        epoch_log = {
            "epoch":      epoch,
            "train_loss": float(train_loss),
            "val":        val_metrics,
            "lr":         float(optimizer.param_groups[0]["lr"]),
            "time_s":     float(time.time() - t0),
        }
        log["epochs"].append(epoch_log)

        print(f"Epoch {epoch:03d}/{args.epochs} | loss={train_loss:.4f} | "
              f"val_f1={val_metrics['f1_macro']:.4f} | "
              f"val_prec={val_metrics['precision_macro']:.4f} | "
              f"val_rec={val_metrics['recall_macro']:.4f}")

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save({"model": model.state_dict(), "classes": class_names}, best_path)
            print(f"  ↑ New best checkpoint saved (val F1={best_f1:.4f})")

    # --- Final test evaluation ---
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device=device, num_classes=len(class_names))

    log["best_val_f1"] = float(best_f1)
    log["test"]        = test_metrics

    (out / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("Done ✅")
    print(f"Best checkpoint : {best_path}")
    print(f"Best val F1     : {best_f1:.4f}")
    print(f"Test F1 (macro) : {test_metrics['f1_macro']:.4f}")
    print(f"Test precision  : {test_metrics['precision_macro']:.4f}")
    print(f"Test recall     : {test_metrics['recall_macro']:.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train MobileNetV3-Large texture classifier")
    ap.add_argument("--data-root",         required=True,  type=str)
    ap.add_argument("--out",               default="runs/texture_mnv3_large", type=str)
    ap.add_argument("--device",            default="mps" if torch.backends.mps.is_available() else "cpu", type=str)
    ap.add_argument("--epochs",            default=10,    type=int)
    ap.add_argument("--batch-size",        default=32,    type=int)
    ap.add_argument("--lr",                default=1e-3,  type=float)
    ap.add_argument("--weight-decay",      default=1e-2,  type=float)
    ap.add_argument("--num-workers",       default=6,     type=int)
    ap.add_argument("--seed",              default=42,    type=int)
    ap.add_argument("--aug",               action="store_true",
                    help="Enable training augmentation (off = deterministic baseline)")
    ap.add_argument("--balanced-sampling", action="store_true",
                    help="Use WeightedRandomSampler for class balance")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
