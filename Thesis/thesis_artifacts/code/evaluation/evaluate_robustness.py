#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustness Evaluation for a MobileNetV3 texture classifier (4 classes).

What this script does
---------------------
- Loads a trained MobileNetV3 checkpoint (best.pt) with class order.
- Evaluates robustness under *systematic perturbations* by applying *deterministic*
  geometric and photometric transformations to a *fixed stratified subset* of a split
  (recommended: validation split).
- Reports macro metrics and class-wise diagnostics:
  macro-F1, per-class recall/F1, confusion matrix, and prediction histogram (pred_hist).

Methodological basis (peer-reviewed / official sources)
------------------------------------------------------
The evaluation protocol follows the general idea of robustness benchmarking via
performance degradation under *defined perturbation sets* and severity levels.
This is conceptually aligned with:
- Hendrycks, D., & Dietterich, T. (2019). "Benchmarking Neural Network Robustness to
  Common Corruptions and Perturbations." ICLR 2019.
  (Official ICLR/OpenReview PDF)
- Hendrycks, D. et al. (2020). "AugMix: A Simple Data Processing Method to Improve
  Robustness and Uncertainty." ICLR 2020.
  (Official ICLR/OpenReview PDF)
Additionally, the choice of augmentations/transform families (rotation, affine,
photometric adjustments) is consistent with general augmentation literature:
- Shorten, C., & Khoshgoftaar, T. M. (2019). "A survey on Image Data Augmentation for Deep Learning."
  Journal of Big Data, 6(1), 60. https://doi.org/10.1186/s40537-019-0197-0

Implementation references (official documentation)
-------------------------------------------------
- TorchVision transforms (Rotate/Affine/Perspective/AdjustBrightness/AdjustContrast):
  https://pytorch.org/vision/stable/transforms.html
- TorchVision pretrained weights + recommended inference preprocessing:
  https://pytorch.org/vision/stable/models.html

Important note about data leakage (thesis-safe)
-----------------------------------------------
- Use split=val for robustness analysis and for deriving augmentation parameters.
- Use split=test only once for final reporting after hyperparameters are fixed.

Cross-script alignment with YOLO evaluation
---------------------------------------------
This script and ``evaluate_yolo_robustness.py`` share a common core
perturbation set so that robustness degradation is *comparable* across
model families in the thesis:

  Shared conditions (23 total):
    Rotation:    45, 90, 135, 180, 225, 270, 315 deg     (7)
    Shear:       +/-10, +/-20 deg                         (4)
    Perspective: 0.1, 0.2, 0.3                            (3)
    Brightness:  0.6, 0.8, 1.2, 1.4                       (4)
    Contrast:    0.6, 0.8, 1.2, 1.4                       (4)
    Clean:       identity                                  (1)

  Additional (classification-only, not in YOLO eval):
    Small rotations: +/-5, +/-10, +/-15 deg  (6)  -- simulate camera tilt

The *implementation library* differs by necessity:
  - MobileNet (classification): torchvision.transforms.functional
    PIL-based, deterministic, no bounding-box bookkeeping needed.
  - YOLO (detection): albumentations
    OpenCV-based, bbox-aware (Buslaev et al. 2020) -- ensures bounding
    boxes are correctly transformed alongside the image.

Fill-value policy
-----------------
Geometric transforms (rotation, affine, perspective) expose empty borders.
To avoid confounding classification with black-pixel artifacts, we fill
with the ImageNet channel mean (R=124, G=116, B=104), matching the fill
used during training augmentation (cf. train_texture_mnv3_robust.py).
The YOLO evaluation uses cv2.BORDER_REPLICATE instead because it
operates on full scenes (edge-pixel replication is natural there).

Suggested BibTeX keys (example)
-------------------------------
- hendrycks2019_common_corruptions_iclr
- hendrycks2020_augmix_iclr
- shorten2019_augmentation_survey_jbd
- buslaev2020_albumentations
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


# ---------------------------
# Constants
# ---------------------------
# ImageNet channel means as integer RGB tuple.
# Used as fill for geometric transforms to avoid black-border artifacts.
# Matches fill used in training augmentation (train_texture_mnv3_robust.py).
IMAGENET_FILL = (124, 116, 104)


# ---------------------------
# Reproducibility utilities
# ---------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# Model / checkpoint loading
# ---------------------------
def build_model(num_classes: int, device: str) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model.to(device)


def load_checkpoint(ckpt_path: Path, device: str) -> Dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" not in ckpt or "classes" not in ckpt:
        raise ValueError("Checkpoint must contain keys: 'model' and 'classes'.")
    return ckpt


# ---------------------------
# Stratified subset selection
# ---------------------------
def stratified_subset_indices(ds: datasets.ImageFolder, per_class: int, seed: int) -> List[int]:
    """
    Select exactly `per_class` indices per class from an ImageFolder dataset.
    Deterministic due to fixed seed.
    """
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {c: [] for c in range(len(ds.classes))}

    for idx, (_, y) in enumerate(ds.samples):
        by_class[y].append(idx)

    indices: List[int] = []
    for c, idxs in by_class.items():
        if len(idxs) < per_class:
            raise ValueError(f"Class '{ds.classes[c]}' has only {len(idxs)} samples, need {per_class}.")
        rng.shuffle(idxs)
        indices.extend(idxs[:per_class])

    rng.shuffle(indices)
    return indices


# ---------------------------
# Deterministic perturbations
# ---------------------------
def persp_squeeze(img, scale: float):
    """
    Deterministic perspective-like warp by moving all four corners inward
    symmetrically.  ``scale`` controls the fraction of each edge that is
    consumed; typical values: 0.1, 0.2, 0.3 (must be < 0.5).

    Unlike A.Perspective (used in the YOLO evaluation), this is fully
    deterministic: the same scale always produces the same warp.
    A.Perspective(scale=(s, s)) randomises the direction of each corner
    perturbation while keeping the magnitude fixed, so it tests average
    robustness over random perspective directions.

    Both approaches are valid for robustness evaluation; the deterministic
    variant makes per-condition comparisons easier to interpret.
    """
    w, h = img.size
    dx = int(scale * w)
    dy = int(scale * h)

    startpoints = [(0, 0), (w, 0), (w, h), (0, h)]
    endpoints = [(dx, dy), (w - dx, dy), (w - dx, h - dy), (dx, h - dy)]
    return F.perspective(
        img,
        startpoints=startpoints,
        endpoints=endpoints,
        interpolation=InterpolationMode.BILINEAR,
        fill=IMAGENET_FILL,
    )


@dataclass(frozen=True)
class Condition:
    name: str
    fn: Callable  # fn(img) -> img


def make_conditions() -> List[Condition]:
    """
    Build the full list of deterministic perturbation conditions.

    The *shared* set (rotation 45-deg steps, shear, perspective, brightness,
    contrast) is aligned 1:1 with evaluate_yolo_robustness.py so that
    robustness curves can be compared across model families in the thesis.
    The *small-rotation* conditions (+/-5, +/-10, +/-15 deg) are
    classification-specific extras that simulate realistic camera tilt.

    All geometric transforms use fill=IMAGENET_FILL to avoid confounding
    the classifier with black-border artefacts.
    """
    conds: List[Condition] = []

    # -- Clean reference ------------------------------------------------
    conds.append(Condition("clean", lambda img: img))

    # -- Rotations: 45-deg increments (aligned with YOLO evaluation) ----
    for a in [45, 90, 135, 180, 225, 270, 315]:
        conds.append(
            Condition(
                f"rot_{a}",
                lambda img, a=a: F.rotate(
                    img, a,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=IMAGENET_FILL,
                ),
            )
        )

    # -- Small rotations: camera/ROI tilt (classification-specific) -----
    for a in [-15, -10, -5, 5, 10, 15]:
        conds.append(
            Condition(
                f"rot_small_{a}",
                lambda img, a=a: F.rotate(
                    img, a,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=IMAGENET_FILL,
                ),
            )
        )

    # -- Shear (affine) -- viewing-angle changes ------------------------
    for sh in [-20, -10, 10, 20]:
        conds.append(
            Condition(
                f"shear_x_{sh:+d}",
                lambda img, sh=sh: F.affine(
                    img,
                    angle=0.0,
                    translate=[0, 0],
                    scale=1.0,
                    shear=[sh, 0.0],
                    interpolation=InterpolationMode.BILINEAR,
                    fill=IMAGENET_FILL,
                ),
            )
        )

    # -- Perspective warp (deterministic squeeze) -----------------------
    for s in [0.1, 0.2, 0.3]:
        conds.append(Condition(f"persp_squeeze_{s}", lambda img, s=s: persp_squeeze(img, s)))

    # -- Photometric: brightness ----------------------------------------
    for fct in [0.6, 0.8, 1.2, 1.4]:
        conds.append(Condition(f"brightness_{fct}", lambda img, fct=fct: F.adjust_brightness(img, fct)))

    # -- Photometric: contrast ------------------------------------------
    for fct in [0.6, 0.8, 1.2, 1.4]:
        conds.append(Condition(f"contrast_{fct}", lambda img, fct=fct: F.adjust_contrast(img, fct)))

    return conds


# ---------------------------
# Metrics / evaluation
# ---------------------------
@torch.no_grad()
def eval_loader(model: nn.Module, loader: DataLoader, device: str, num_classes: int) -> Dict:
    model.eval()
    all_y: List[int] = []
    all_pred: List[int] = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().tolist()
        all_pred.extend(pred)
        all_y.extend(y.tolist())

    prec, rec, f1, _ = precision_recall_fscore_support(all_y, all_pred, average="macro", zero_division=0)
    cm = confusion_matrix(all_y, all_pred).tolist()
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
        "confusion_matrix": cm,
        "true_hist": true_hist,
        "pred_hist": pred_hist,
        "num_samples": len(all_y),
    }


class WrappedDataset(torch.utils.data.Dataset):
    """
    Applies:
    1) a deterministic perturbation condition (PIL -> PIL)
    2) base preprocessing from TorchVision weights (PIL -> Tensor, normalize, resize/crop)
    """
    def __init__(self, base_subset, cond_fn, preprocess):
        self.base_subset = base_subset
        self.cond_fn = cond_fn
        self.preprocess = preprocess

    def __len__(self):
        return len(self.base_subset)

    def __getitem__(self, i):
        img, y = self.base_subset[i]  # img is PIL.Image
        img = self.cond_fn(img)
        img = self.preprocess(img)
        return img, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=str, help="ImageFolder root with train/val/test")
    ap.add_argument("--ckpt", required=True, type=str, help="Path to best.pt (contains model state + classes)")
    ap.add_argument("--split", default="val", choices=["val", "test"], help="Robustness split (recommend: val)")
    ap.add_argument("--per-class", default=100, type=int, help="Images per class for robust subset (e.g., 50 or 100)")
    ap.add_argument("--batch-size", default=64, type=int)
    ap.add_argument("--num-workers", default=0, type=int, help="Number of workers (0=single process, avoids pickling issues)")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu", type=str)
    ap.add_argument("--out", default="robustness_results.json", type=str)
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device

    ckpt_path = Path(args.ckpt)
    ckpt = load_checkpoint(ckpt_path, device=device)
    classes_ckpt: List[str] = ckpt["classes"]

    split_dir = Path(args.data_root) / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    # Load raw ImageFolder without transforms (we apply perturbations manually)
    raw_ds = datasets.ImageFolder(split_dir, transform=None)

    # Ensure class order matches checkpoint
    if raw_ds.classes != classes_ckpt:
        raise ValueError(
            "Class order mismatch between dataset and checkpoint.\n"
            f"Dataset classes:   {raw_ds.classes}\n"
            f"Checkpoint classes:{classes_ckpt}\n"
            "Fix: ensure consistent folder naming/order or store/compare class_to_idx mapping."
        )

    # Stratified fixed subset
    idxs = stratified_subset_indices(raw_ds, per_class=args.per_class, seed=args.seed)
    subset = Subset(raw_ds, idxs)

    # Base preprocessing from TorchVision weights (deterministic)
    weights = MobileNet_V3_Small_Weights.DEFAULT
    base_preprocess = weights.transforms()

    # Load model
    model = build_model(num_classes=len(classes_ckpt), device=device)
    model.load_state_dict(ckpt["model"], strict=True)

    conditions = make_conditions()

    results = {
        "meta": {
            "ckpt": str(ckpt_path),
            "data_root": args.data_root,
            "split": args.split,
            "per_class": args.per_class,
            "seed": args.seed,
            "device": device,
            "classes": classes_ckpt,
        },
        "conditions": [],
    }

    for cond in conditions:
        ds_cond = WrappedDataset(subset, cond.fn, base_preprocess)
        loader = DataLoader(
            ds_cond,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
        )
        metrics = eval_loader(model, loader, device=device, num_classes=len(classes_ckpt))
        results["conditions"].append({"name": cond.name, "metrics": metrics})
        print(f"{cond.name:>18s} | macroF1={metrics['f1_macro']:.4f} | pred_hist={metrics['pred_hist']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
