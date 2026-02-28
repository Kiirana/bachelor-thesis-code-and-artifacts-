#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustness Evaluation for a YOLO object detector (single-class EVCS).

What this script does
---------------------
- Loads a trained YOLO checkpoint (.pt) via Ultralytics.
- Evaluates robustness under *systematic perturbations* by applying
  deterministic geometric and photometric transformations to a dataset split
  (recommended: validation split).
- Uses Albumentations for bbox-aware transforms so that ground-truth
  bounding boxes are correctly transformed alongside the image.
- Reports official mAP50, mAP50-95, precision, recall, and F1 per condition
  via ``model.val()`` on a temporary perturbed dataset copy.

Methodological basis (peer-reviewed / official sources)
------------------------------------------------------
- Hendrycks, D., & Dietterich, T. (2019). "Benchmarking Neural Network
  Robustness to Common Corruptions and Perturbations." ICLR 2019.
  https://arxiv.org/abs/1903.12261
- Perturbations test invariance to real-world conditions (rotation,
  perspective, shear, lighting changes).

Implementation references (official documentation)
-------------------------------------------------
- Albumentations (Buslaev et al. 2020): Bbox-aware augmentations with
  mathematical correctness for object detection.
  https://albumentations.ai/ | https://arxiv.org/abs/1809.06839
- Ultralytics YOLO: Official mAP calculation via model.val().
  https://docs.ultralytics.com/modes/val/

Cross-script alignment with MobileNet evaluation
-------------------------------------------------
This script and ``evaluate_robustness.py`` (MobileNet texture classifier)
share a common core perturbation set so that robustness degradation is
*comparable* across model families in the thesis:

  Shared conditions (23 total):
    Rotation:    45, 90, 135, 180, 225, 270, 315 deg     (7)
    Shear:       +/-10, +/-20 deg                         (4)
    Perspective: 0.1, 0.2, 0.3                            (3)
    Brightness:  0.6, 0.8, 1.2, 1.4                       (4)
    Contrast:    0.6, 0.8, 1.2, 1.4                       (4)
    Clean:       identity                                  (1)

The *implementation library* differs by necessity:
  - YOLO (detection): albumentations
    OpenCV-based, bbox-aware -- bounding boxes are correctly transformed
    alongside the image (Buslaev et al. 2020).
  - MobileNet (classification): torchvision.transforms.functional
    PIL-based, deterministic, no bounding-box bookkeeping needed.

Fill/border policy
------------------
Geometric transforms expose empty borders.
  - YOLO: cv2.BORDER_REPLICATE (natural for full scenes; replicates edge pixels)
  - MobileNet: ImageNet channel mean fill (avoids black-pixel artifacts that
    could confound texture classification)

Important note about data leakage (thesis-safe)
-----------------------------------------------
- Use split=val for robustness analysis and for deriving augmentation parameters.
- Use split=test only once for final reporting after hyperparameters are fixed.

Suggested BibTeX keys (example)
-------------------------------
- hendrycks2019_common_corruptions_iclr
- buslaev2020_albumentations

Usage:
    python3 evaluate_yolo_robustness.py \\
        --model runs/yolo12m_evcs/weights/best.pt \\
        --data evcs_yolo_site_based/data.yaml \\
        --split val \\
        --out robustness_results.json \\
        --seed 42

Dependencies:
    pip install albumentations ultralytics opencv-python pyyaml tqdm
"""

import argparse
import json
import random
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import cv2
import numpy as np
import yaml
import albumentations as A
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset, img2label_paths
from tqdm import tqdm


def set_seed(seed: int):
    """Ensure reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset_paths(data_yaml: Path, split: str) -> List[Tuple[Path, Path]]:
    """
    Load image-label pairs using Ultralytics native utilities.
    Simplified version using check_det_dataset() and img2label_paths().
    
    Returns:
        List of (image_path, label_path) tuples
    """
    # Use Ultralytics to resolve all paths correctly
    data = check_det_dataset(str(data_yaml), autodownload=False)
    
    # Get image roots for this split
    img_roots = data.get(split, [])
    if isinstance(img_roots, str):
        img_roots = [img_roots]
    
    # Collect all images from all roots
    img_paths = []
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    
    for root in img_roots:
        root_path = Path(root)
        if root_path.exists():
            img_paths.extend([p for p in root_path.rglob("*") if p.suffix.lower() in valid_exts])
    
    # Use Ultralytics' built-in mapping: images/ -> labels/
    label_paths = [Path(p) for p in img2label_paths([str(p) for p in img_paths])]
    
    # Filter to only pairs where label exists
    return [(img, lbl) for img, lbl in zip(img_paths, label_paths) if lbl.exists()]


def read_yolo_labels(label_path: Path) -> Tuple[List[int], List[List[float]]]:
    """
    Read YOLO format labels: class x_center y_center width height (normalized).
    
    Returns:
        category_ids: List of class IDs
        bboxes: List of [x, y, w, h] in YOLO format (normalized 0-1)
    """
    if not label_path.exists():
        return [], []
    
    category_ids = []
    bboxes = []
    
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                category_ids.append(cls_id)
                bboxes.append([x, y, w, h])
    
    return category_ids, bboxes


def write_yolo_labels(label_path: Path, category_ids: List[int], bboxes: List[List[float]]):
    """Write YOLO format labels to file."""
    with open(label_path, 'w') as f:
        for cls_id, bbox in zip(category_ids, bboxes):
            # Clamp to [0, 1] and ensure valid boxes
            x, y, w, h = bbox
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            
            # Skip degenerate boxes
            if w > 0.001 and h > 0.001:
                f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def get_perturbations() -> Dict[str, Tuple[A.Compose, bool]]:
    """
    Define Albumentations transforms for robustness evaluation.

    The perturbation set is aligned 1:1 with ``evaluate_robustness.py``
    (MobileNet) so that cross-model robustness degradation tables can be
    compared directly in the thesis.  Condition names match exactly.

    Key design choices:
    - min_visibility=0.1: Drop boxes if <10% visible after transform
    - INTER_LANCZOS4: High-quality interpolation for perspective transforms
    - BORDER_REPLICATE: Natural edge filling for scene images

    Returns:
        Dict mapping condition names to (Albumentations Compose, needs_bbox_transform)
        needs_bbox_transform=False means only image changes (can copy labels)
    """
    bbox_params = A.BboxParams(format='yolo', label_fields=['category_ids'], min_visibility=0.1)
    transforms = {}
    
    # Clean baseline
    transforms['clean'] = (A.Compose([], bbox_params=bbox_params), False)
    
    # Rotation - critical failure mode (full circle at 45° increments)
    for angle in [45, 90, 135, 180, 225, 270, 315]:
        transforms[f'rot_{angle}'] = (
            A.Compose([A.SafeRotate(limit=(angle, angle), p=1.0, border_mode=cv2.BORDER_REPLICATE)], 
                     bbox_params=bbox_params),
            True  # Bboxes need transformation
        )
    
    # Perspective squeeze - catastrophic failure at 0.3
    for scale in [0.1, 0.2, 0.3]:
        transforms[f'persp_squeeze_{scale}'] = (
            A.Compose([A.Perspective(scale=(scale, scale), p=1.0, keep_size=True, 
                                    interpolation=cv2.INTER_LANCZOS4)], 
                     bbox_params=bbox_params),
            True  # Bboxes need transformation
        )
    
    # Shear - viewing angle changes
    for angle in [-20, -10, 10, 20]:
        transforms[f'shear_x_{angle:+d}'] = (
            A.Compose([A.Affine(shear={'x': (angle, angle)}, border_mode=cv2.BORDER_REPLICATE, p=1.0)], 
                     bbox_params=bbox_params),
            True  # Bboxes need transformation
        )
    
    # Photometric - lighting conditions (bboxes unchanged, can optimize)
    for factor in [0.6, 0.8, 1.2, 1.4]:
        brightness_limit = contrast_limit = factor - 1.0
        transforms[f'brightness_{factor}'] = (
            A.Compose([A.RandomBrightnessContrast(
                brightness_limit=(brightness_limit, brightness_limit),
                contrast_limit=(0, 0),
                p=1.0
            )], bbox_params=bbox_params),
            False  # Only image changes, labels stay valid
        )
    
    for factor in [0.6, 0.8, 1.2, 1.4]:
        contrast_limit = factor - 1.0
        transforms[f'contrast_{factor}'] = (
            A.Compose([A.RandomBrightnessContrast(
                brightness_limit=(0, 0),
                contrast_limit=(contrast_limit, contrast_limit),
                p=1.0
            )], bbox_params=bbox_params),
            False  # Only image changes, labels stay valid
        )
    
    return transforms


def evaluate_condition(
    model: YOLO,
    dataset_pairs: List[Tuple[Path, Path]],
    transform: A.Compose,
    needs_bbox_transform: bool,
    condition_name: str,
    data_yaml: Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> Dict:
    """
    Evaluate model on perturbed dataset with proper bbox transformation.
    Uses TemporaryDirectory for automatic cleanup.
    Optimizes photometric tests by copying labels directly.
    
    Returns:
        Dictionary with mAP50, mAP50-95, precision, recall, F1
    """
    print(f"  Testing: {condition_name}")
    
    # Use context manager for automatic cleanup
    with tempfile.TemporaryDirectory(prefix=f"yolo_robustness_{condition_name}_") as temp_root:
        temp_root = Path(temp_root)
        temp_images = temp_root / "images"
        temp_labels = temp_root / "labels"
        temp_images.mkdir()
        temp_labels.mkdir()
        
        # Apply perturbation and save
        for img_path, label_path in tqdm(dataset_pairs, desc=f"  {condition_name}", leave=False):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if needs_bbox_transform:
                # Geometric transform - need to transform labels
                category_ids, bboxes = read_yolo_labels(label_path)
                
                try:
                    transformed = transform(image=img, bboxes=bboxes, category_ids=category_ids)
                    img_transformed = transformed['image']
                    bboxes_transformed = transformed['bboxes']
                    category_ids_transformed = transformed['category_ids']
                except (cv2.error, ValueError, RuntimeError, IndexError) as e:
                    continue
                
                # Save transformed image and labels
                img_transformed = cv2.cvtColor(img_transformed, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(temp_images / img_path.name), img_transformed)
                write_yolo_labels(temp_labels / label_path.name, category_ids_transformed, bboxes_transformed)
            else:
                # Photometric transform - labels unchanged, can copy directly
                try:
                    transformed = transform(image=img, bboxes=[], category_ids=[])
                    img_transformed = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(temp_images / img_path.name), img_transformed)
                    shutil.copy2(label_path, temp_labels / label_path.name)  # Fast copy
                except (cv2.error, ValueError, RuntimeError) as e:
                    continue
        
        # Create temporary data.yaml
        with open(data_yaml) as f:
            original_data = yaml.safe_load(f)
        
        temp_yaml = temp_root / "data.yaml"
        with open(temp_yaml, 'w') as f:
            yaml.dump({
                'path': str(temp_root),
                'train': 'images',   # Ultralytics requires 'train' key even for val-only runs
                'val': 'images',
                'names': original_data.get('names', {}),
                'nc': original_data.get('nc', 1)
            }, f)
        
        # Run validation
        try:
            metrics = model.val(
                data=str(temp_yaml),
                split='val',
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
                plots=False
            )
            
            return {
                'condition': condition_name,
                'total_images': len(list(temp_images.glob('*'))),
                'mAP50': float(metrics.box.map50),
                'mAP50_95': float(metrics.box.map),
                'precision': float(metrics.box.mp),
                'recall': float(metrics.box.mr),
                'f1': float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-8))
            }
        except Exception as e:
            return {
                'condition': condition_name,
                'total_images': len(list(temp_images.glob('*'))),
                'error': str(e),
                'mAP50': 0.0, 'mAP50_95': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0
            }
    # Automatic cleanup via TemporaryDirectory context manager


def main():
    parser = argparse.ArgumentParser(description="YOLO Robustness Evaluation (Compact Thesis Version)")
    parser.add_argument('--model', type=Path, required=True, help='Path to trained YOLO model (.pt)')
    parser.add_argument('--data', type=Path, required=True, help='Path to data.yaml')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to use (recommended: val)')
    parser.add_argument('--out', type=Path, default='yolo_robustness_results.json',
                        help='Output JSON file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit number of images (for quick testing)')
    args = parser.parse_args()
    
    print("="*80)
    print("YOLO ROBUSTNESS EVALUATION - Compact Thesis Version")
    print("="*80)
    print(f"Model:      {args.model}")
    print(f"Data:       {args.data}")
    print(f"Split:      {args.split}")
    print(f"Seed:       {args.seed}")
    print(f"Conf:       {args.conf}")
    print(f"IoU:        {args.iou}")
    print("="*80 + "\n")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load model
    print("Loading model...")
    model = YOLO(str(args.model))
    model.conf = args.conf
    model.iou = args.iou
    
    # Load dataset
    print(f"Loading {args.split} dataset...")
    dataset_pairs = load_dataset_paths(args.data, args.split)
    
    if args.max_samples:
        dataset_pairs = random.sample(dataset_pairs, min(args.max_samples, len(dataset_pairs)))
    
    print(f"Dataset size: {len(dataset_pairs)} images\n")
    
    # Get perturbations
    perturbations = get_perturbations()
    
    # Evaluate each perturbation
    results = {}
    print(f"Testing {len(perturbations)} conditions...\n")
    
    for name, (transform, needs_bbox) in perturbations.items():
        try:
            metrics = evaluate_condition(model, dataset_pairs, transform, needs_bbox, name, 
                                        args.data, args.conf, args.iou)
            results[name] = metrics
        except Exception as e:
            print(f"  ⚠️ Error in {name}: {e}")
            results[name] = {'error': str(e)}
    
    # Save results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("ROBUSTNESS SUMMARY")
    print("="*80)
    print(f"{'Condition':<25} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-"*80)
    
    for name, metrics in results.items():
        if 'error' not in metrics:
            print(f"{name:<25} {metrics['mAP50']:>8.1%} {metrics['mAP50_95']:>9.1%} {metrics['precision']:>9.1%} {metrics['recall']:>9.1%} {metrics.get('f1', 0.0):>8.1%}")
        else:
            print(f"{name:<25} {'ERROR':<10} {metrics.get('error', 'Unknown')[:50]}")
    
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    
    # Compare critical conditions to baseline
    baseline = results.get('clean', {})
    if baseline and 'mAP50' in baseline:
        critical_conditions = ['rot_45', 'rot_90', 'rot_270', 'rot_315', 'persp_squeeze_0.3', 'shear_x_-20', 'shear_x_20']
        
        for cond in critical_conditions:
            if cond in results and 'mAP50' in results[cond]:
                baseline_map = baseline['mAP50']
                cond_map = results[cond]['mAP50']
                delta = cond_map - baseline_map
                status = "✅" if delta > -0.1 else ("⚠️" if delta > -0.3 else "❌")
                print(f"  {cond:<25}: {baseline_map:.1%} → {cond_map:.1%} ({delta:+.1%}) {status}")
    
    print(f"\n✅ Results saved to: {args.out}")
    print("="*80)


if __name__ == '__main__':
    main()
