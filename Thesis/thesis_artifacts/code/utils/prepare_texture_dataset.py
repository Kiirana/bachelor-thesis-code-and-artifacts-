#!/usr/bin/env python3
"""
Prepare a unified texture dataset in TorchVision ImageFolder format.

Output layout:
  out/
    train/<class>/*.jpg
    val/<class>/*.jpg
    test/<class>/*.jpg
    meta.json

Why this format?
- TorchVision ImageFolder expects root/<class>/<image>...
- Reference: https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# Canonical 4-class setup used in the thesis
CANONICAL = ["asphalt", "cobblestone", "gravel", "sand"]

# Minimal, explicit mapping (extend only if you can justify it in the thesis)
LABEL_MAP = {
    # asphalt (all moisture conditions from RoadSaW + RSCD variations)
    "asphalt": "asphalt",
    "asphalt_dry": "asphalt",
    "asphalt_damp": "asphalt",
    "asphalt_wet": "asphalt",
    "asphalt_verywet": "asphalt",
    # RSCD asphalt variations
    "dry_asphalt_severe": "asphalt",
    "dry_asphalt_slight": "asphalt",
    "dry_asphalt_smooth": "asphalt",
    "wet_asphalt_severe": "asphalt",
    "wet_asphalt_slight": "asphalt",
    "wet_asphalt_smooth": "asphalt",
    "water_asphalt_severe": "asphalt",
    "water_asphalt_slight": "asphalt",
    "water_asphalt_smooth": "asphalt",
    
    # cobblestone (all moisture conditions from RoadSaW)
    "cobblestone": "cobblestone",
    "cobble_dry": "cobblestone",
    "cobble_damp": "cobblestone",
    "cobble_wet": "cobblestone",
    "cobble_verywet": "cobblestone",

    # gravel (RSCD variations)
    "gravel": "gravel",
    "pebble": "gravel",
    "stone": "gravel",
    "dry_gravel": "gravel",
    "wet_gravel": "gravel",
    "water_gravel": "gravel",

    # sand (incl. dirt/mud + RSCD mud variations)
    "sand": "sand",
    "dirt": "sand",
    "dry_mud": "sand",
    "wet_mud": "sand",
    "water_mud": "sand",
}

def iter_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix in IMAGE_EXTS]

def collect_labeled_images(dataset_root: Path) -> List[Tuple[Path, str]]:
    """
    Collect images from datasets that follow:
      dataset_root/<dataset_name>/<class_folder>/*.jpg
    or:
      dataset_root/<dataset_name>/train/<class_folder>/*.jpg etc.
    """
    samples: List[Tuple[Path, str]] = []

    for ds_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")]):
        # search either ds/train/* or ds/* as class folders
        candidates = []
        if (ds_dir / "train").exists():
            candidates.extend([ds_dir / "train", ds_dir / "test", ds_dir / "val"])
        else:
            candidates.append(ds_dir)

        for base in [c for c in candidates if c.exists()]:
            for class_dir in [p for p in base.iterdir() if p.is_dir()]:
                orig = class_dir.name.lower()
                canonical = LABEL_MAP.get(orig)
                if canonical not in CANONICAL:
                    continue
                for img in iter_images(class_dir):
                    samples.append((img, canonical))

    return samples

def stratified_split(
    samples: List[Tuple[Path, str]],
    ratios=(0.8, 0.1, 0.1),
    seed: int = 42,
) -> Dict[str, List[Tuple[Path, str]]]:
    assert abs(sum(ratios) - 1.0) < 1e-9
    rng = random.Random(seed)

    by_class: Dict[str, List[Tuple[Path, str]]] = defaultdict(list)
    for p, y in samples:
        by_class[y].append((p, y))

    splits = {"train": [], "val": [], "test": []}
    # Sort classes for deterministic RNG state
    for y in sorted(by_class.keys()):
        items = by_class[y]
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train:n_train + n_val])
        splits["test"].extend(items[n_train + n_val:])

    # final shuffle (doesn't affect stratification)
    for k in splits:
        rng.shuffle(splits[k])
    return splits

def export_imagefolder(
    splits: Dict[str, List[Tuple[Path, str]]],
    out_root: Path,
    mode: str = "copy",  # "copy" or "symlink"
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    def put(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if mode == "symlink":
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src)
        else:
            shutil.copy2(src, dst)

    for split_name, items in splits.items():
        for src, y in items:
            dst = out_root / split_name / y / src.name
            put(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    ap.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    args = ap.parse_args()

    samples = collect_labeled_images(args.dataset_root)
    if not samples:
        raise SystemExit("No samples found. Check your dataset-root and label folder names.")

    counts = Counter([y for _, y in samples])
    print("Found samples:", dict(counts))

    splits = stratified_split(samples, ratios=tuple(args.split), seed=args.seed)
    export_imagefolder(splits, args.out, mode=args.mode)

    meta = {
        "classes": CANONICAL,
        "seed": args.seed,
        "split": list(args.split),
        "counts_total": dict(counts),
        "counts_split": {
            k: dict(Counter([y for _, y in v])) for k, v in splits.items()
        },
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Done ✅ wrote:", args.out)

if __name__ == "__main__":
    main()
