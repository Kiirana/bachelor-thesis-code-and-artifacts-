#!/usr/bin/env python3
"""
Train YOLO12m on EVCS with Robust Augmentation for Geometric Invariance.

Hypothesis: Geometric augmentation will improve robustness against the same
failure modes identified in texture classification:
- Rotation failures (90°/270° → near-random performance)
- Perspective squeeze collapse (10% mAP)
- Shear degradation (viewing angle changes)

This script uses Ultralytics native augmentation hyperparameters:
- degrees: Rotation augmentation (±180°)
- perspective: Perspective transform (0.0-0.001, increased to 0.001 for robustness)
- shear: Shear angle (±20° → handles viewing angle changes)
- hsv_h/s/v: Photometric augmentation (lighting variations)

References:
  - Ultralytics Augmentation: https://docs.ultralytics.com/modes/train/#augmentation-settings
  - Hendrycks & Dietterich (2019): Robustness benchmarking methodology
  - Thesis: Chapter 5 - Robust Training for Geometric Invariance

Key differences from baseline training:
  1. degrees=180.0 (baseline: 0.0) → aims to address rotation failures
  2. perspective=0.001 (baseline: 0.0) → targets perspective squeeze
  3. shear=20.0 (baseline: 0.0) → aims to improve viewing angle robustness
  4. hsv_h/s/v tuned for real-world lighting variations

Research questions (informed by texture model results):
  - 90° rotation: We hypothesize significant improvement (texture: 62%→95% F1)
  - 270° rotation: We hypothesize significant improvement (texture: 58%→95% F1)
  - Shear ±20°: We hypothesize moderate improvement (texture: +2% F1)
  - Perspective 0.3: Unknown - texture model failed, YOLO may handle differently

Usage:
    # Baseline (for comparison)
    python3 train_yolo12m_evcs_robust.py \\
        --data evcs_data.yaml \\
        --mode baseline \\
        --epochs 100 \\
        --name yolo12m_evcs_baseline

    # Robust augmentation (recommended)
    python3 train_yolo12m_evcs_robust.py \\
        --data evcs_data.yaml \\
        --mode robust \\
        --epochs 100 \\
        --name yolo12m_evcs_robust
"""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


def pick_device(user_device: str | None) -> str:
    """Prefer user setting, else choose MPS on Apple Silicon if available, else CPU."""
    if user_device:
        return user_device
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_augmentation_params(mode: str) -> dict:
    """
    Get augmentation hyperparameters for different training modes.
    
    Args:
        mode: 'baseline' (minimal augmentation) or 'robust' (full geometric augmentation)
    
    Returns:
        Dictionary of augmentation hyperparameters for Ultralytics YOLO
    
    Ultralytics augmentation reference:
    https://docs.ultralytics.com/modes/train/#augmentation-settings
    
    Note: Baseline mode uses YOLO standard augmentation (mosaic, flips, minimal HSV),
    NOT zero augmentation. For thesis comparison, this establishes the "standard training"
    performance. Robust mode adds strong geometric augmentation (rotation, perspective, shear).
    """
    # Shared parameters kept identical between modes to isolate the effect
    # of geometric augmentation (clean ablation design).
    shared = {
        'hsv_h': 0.015,          # Hue variation (Ultralytics default)
        'hsv_s': 0.7,            # Saturation (Ultralytics default)
        'hsv_v': 0.4,            # Brightness (Ultralytics default)
        'mosaic': 1.0,           # YOLO standard mosaic
        'mixup': 0.0,            # No mixup in either mode
        'fliplr': 0.5,           # Horizontal flip (natural for driving scenes)
    }

    if mode == "baseline":
        # Standard YOLO augmentation - establishes "typical training" performance.
        # NOT zero augmentation: uses Ultralytics defaults for object detection.
        return {
            **shared,
            'degrees': 0.0,          # No rotation
            'perspective': 0.0,      # No perspective transform
            'shear': 0.0,            # No shear
            'flipud': 0.0,           # No vertical flip
        }

    elif mode == "robust":
        # Hypothesis: Adding geometric augmentation improves robustness against
        # camera angles, perspective distortion, and rotation.
        # Based on texture model results (90° rotation: 62%→95% F1), we expect
        # improved performance on rotated/warped EVCS in real-world deployment.
        #
        # ONLY geometric parameters differ from baseline (clean ablation).
        # copy_paste is NOT used: it requires segmentation masks (EVCS has bbox only).
        return {
            **shared,
            # GEOMETRIC AUGMENTATION (primary hypothesis)
            'degrees': 180.0,        # Full rotation ±180° (texture model: +33% F1 at 90°)
            'perspective': 0.001,    # Ultralytics max (0.0-0.001) for squeeze/warp robustness
            'shear': 20.0,           # Shear ±20° for viewing angle changes
            'flipud': 0.5,           # Vertical flip (complements rotation)
        }
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'baseline' or 'robust'")


def main():
    ap = argparse.ArgumentParser(
        description="Train YOLO12m on EVCS with configurable augmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline training (standard YOLO augmentation: mosaic, flips, minimal HSV)
  python3 train_yolo12m_evcs_robust.py --data evcs_data.yaml --mode baseline

  # Robust training (adds geometric augmentation: rotation, perspective, shear)
  python3 train_yolo12m_evcs_robust.py --data evcs_data.yaml --mode robust --epochs 150
        """
    )
    
    # Dataset and model
    ap.add_argument("--data", type=Path, required=True, help="Path to Ultralytics data.yaml")
    ap.add_argument("--model", type=str, default="yolo12m.pt", help="Pretrained model weights")
    
    # Training mode
    ap.add_argument("--mode", type=str, choices=['baseline', 'robust'], default='robust',
                    help="Training mode: 'baseline' (standard YOLO aug) or 'robust' (+ geometric aug)")
    
    # Training hyperparameters
    ap.add_argument("--imgsz", type=int, default=640, help="Input image size")
    ap.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    ap.add_argument("--batch", type=int, default=16, help="Batch size")
    ap.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Hardware and output
    ap.add_argument("--device", type=str, default=None, help="Device: 'mps', 'cpu', or '0' for CUDA")
    ap.add_argument("--workers", type=int, default=8, help="Number of worker threads for data loading")
    ap.add_argument("--project", type=str, default="runs", help="Project directory")
    ap.add_argument("--name", type=str, default=None, help="Run name (default: yolo12m_evcs_{mode})")
    ap.add_argument("--export", action="store_true", help="Export to ONNX and CoreML after training")
    
    args = ap.parse_args()
    
    # Default run name based on mode
    if args.name is None:
        args.name = f"yolo12m_evcs_{args.mode}"
    
    device = pick_device(args.device)
    aug_params = get_augmentation_params(args.mode)
    
    print("="*80)
    print(f"Training YOLO12m on EVCS Dataset - {args.mode.upper()} Mode")
    print("="*80)
    print(f"Model:        {args.model}")
    print(f"Data:         {args.data}")
    print(f"Mode:         {args.mode}")
    print(f"Device:       {device}")
    print(f"Image size:   {args.imgsz}")
    print(f"Epochs:       {args.epochs}")
    print(f"Batch:        {args.batch}")
    print(f"Patience:     {args.patience}")
    print(f"Seed:         {args.seed}")
    print(f"Workers:      {args.workers}")
    print(f"\nAugmentation Parameters ({args.mode} mode):")
    print("-"*80)
    for key, value in aug_params.items():
        print(f"  {key:<15}: {value}")
    print("="*80 + "\n")
    
    # Load model
    model = YOLO(args.model)
    
    # Train with augmentation parameters
    # Ultralytics Train API: https://docs.ultralytics.com/modes/train/
    # close_mosaic: disable mosaic in the last N epochs for fine-tuning.
    # Ultralytics default is 10 (fixed).  At 60 epochs this means the last
    # ~17 % of training runs without mosaic, which is the intended design.
    # We keep the default for reproducibility and thesis defensibility.
    # Ref: ultralytics/cfg/default.yaml  "close_mosaic: 10"
    close_mosaic_epochs = 10

    results = model.train(
        # Dataset
        data=str(args.data),
        imgsz=args.imgsz,
        single_cls=True,  # Binary EVCS detection: treat all classes as one
        
        # Training schedule
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        close_mosaic=close_mosaic_epochs,
        
        # Reproducibility
        seed=args.seed,
        deterministic=True,
        
        # Hardware
        device=device,
        workers=args.workers,
        
        # Output
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=True,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        
        # Augmentation parameters (unpacked from mode-specific dict)
        **aug_params,
        
        # Optimizer settings: Ultralytics defaults
        # optimizer='auto' → AdamW for YOLO12
        # lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=5e-4
    )
    
    # Validate best checkpoint
    save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    
    if not best.exists():
        print(f"⚠️ Warning: best.pt not found at {best}, using last.pt")
        best = save_dir / "weights" / "last.pt"
    
    best_model = YOLO(str(best))
    
    print("\n" + "="*80)
    print("Validating best model on validation set")
    print("="*80)
    metrics = best_model.val(data=str(args.data), imgsz=args.imgsz)
    
    print("\n" + "="*80)
    print("Validation Metrics")
    print("="*80)
    print(f"  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")
    print("="*80)
    
    # Save augmentation config for thesis documentation
    config_file = save_dir / "augmentation_config.txt"
    with open(config_file, 'w') as f:
        f.write(f"Training Mode: {args.mode}\n")
        f.write(f"Seed: {args.seed}\n\n")
        f.write("Augmentation Parameters:\n")
        for key, value in aug_params.items():
            f.write(f"  {key}: {value}\n")
    print(f"\n✅ Augmentation config saved to: {config_file}\n")
    
    if args.export:
        print("="*80)
        print("Exporting model to ONNX and CoreML")
        print("="*80)
        best_model.export(format="onnx", imgsz=args.imgsz)
        best_model.export(format="coreml", imgsz=args.imgsz)
        print("Export complete!\n")
    
    print("="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Run robustness evaluation:")
    print(f"   python3 evaluate_yolo_robustness.py \\")
    print(f"       --model {save_dir}/weights/best.pt \\")
    print(f"       --data {args.data} \\")
    print(f"       --split val \\")
    print(f"       --out {save_dir}/robustness_results.json")
    print("\n2. Compare with baseline (if both models trained):")
    print("   - Check rotation robustness (90°/270°)")
    print("   - Check perspective squeeze (0.3 scale)")
    print("   - Check shear robustness (±20°)")
    print("="*80)


if __name__ == "__main__":
    main()
