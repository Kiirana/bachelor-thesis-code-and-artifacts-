#!/usr/bin/env python3
"""
Train a texture classifier (4 classes) using TorchVision MobileNetV3.

Dataset format: ImageFolder splits:
  data_root/train/<class>/*.jpg
  data_root/val/<class>/*.jpg
  data_root/test/<class>/*.jpg
See ImageFolder docs: https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html

Model + preprocessing:
- MobileNetV3 from TorchVision with pretrained weights
- Reference: https://pytorch.org/vision/stable/models/mobilenetv3.html
- Weight objects provide recommended inference transforms
"""

from __future__ import annotations

import argparse
from typing import Dict, Tuple, List
import json
import time
import random
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


def build_transforms(img_size: int = 224, baseline_no_aug: bool = True):
    # Use TorchVision weight metadata for normalization when available.
    weights = MobileNet_V3_Small_Weights.DEFAULT

    if baseline_no_aug:
        # True baseline: deterministic transforms (no augmentation) for ALL splits
        train_tf = weights.transforms()
        eval_tf = weights.transforms()
        return train_tf, eval_tf

    # Augmented training pipeline (when baseline_no_aug=False)
    mean = weights.meta.get("mean", [0.485, 0.456, 0.406])
    std = weights.meta.get("std", [0.229, 0.224, 0.225])

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Validation/Test: use the official inference preprocessing for the weights.
    eval_tf = weights.transforms()

    return train_tf, eval_tf


def build_loaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    use_balanced_sampling: bool,
    baseline_no_aug: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    train_tf, eval_tf = build_transforms(baseline_no_aug=baseline_no_aug)

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_root / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_root / "test", transform=eval_tf)

    class_names = train_ds.classes

    sampler = None
    shuffle = True
    if use_balanced_sampling:
        # Standard approach: sample weights inverse to class frequency
        labels = [y for _, y in train_ds.samples]
        counts = torch.bincount(torch.tensor(labels), minlength=len(class_names)).float()
        class_w = (counts.sum() / (len(class_names) * counts)).clamp(max=10.0)  # cap to avoid extreme oversampling
        sample_w = torch.tensor([class_w[y] for y in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
        shuffle = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                              num_workers=num_workers, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader, test_loader, class_names


def build_model(num_classes: int, device: str) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)

    # Replace classifier head for custom classes
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

    # NEW: histograms (very useful to detect dominant-class collapse)
    # true_hist: how many samples per class in this eval split
    # pred_hist: how many predictions per class the model outputs
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
    train_loader, val_loader, test_loader, class_names = build_loaders(
        Path(args.data_root), args.batch_size, args.num_workers, args.balanced_sampling,
        baseline_no_aug=not args.aug
    )

    model = build_model(num_classes=len(class_names), device=device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # CosineAnnealingLR is a standard scheduler in PyTorch.
    # Reference: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    best_path = out / "best.pt"

    log = {
        "config": vars(args),
        "classes": class_names,
        "epochs": [],
    }

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

    # Final test evaluation with best checkpoint
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
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
    ap.add_argument("--data-root", required=True, type=str, help="Prepared ImageFolder dataset root (has train/val/test)")
    ap.add_argument("--out", default="runs/texture_mnv3", type=str)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu", type=str)
    ap.add_argument("--epochs", default=20, type=int)
    ap.add_argument("--batch-size", default=32, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--weight-decay", default=1e-2, type=float)
    ap.add_argument("--num-workers", default=6, type=int, help="Data loader workers (6 for speed, thesis deadline)")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--aug", action="store_true", 
                    help="Enable training augmentation (RandomResizedCrop + RandomHorizontalFlip). Default: off (deterministic baseline)")
    ap.add_argument("--balanced-sampling", action="store_true", 
                    help="Use WeightedRandomSampler. Note: Pure baseline = no aug + no balanced sampling")

    args = ap.parse_args()
    train(args)

if __name__ == "__main__":
    main()
