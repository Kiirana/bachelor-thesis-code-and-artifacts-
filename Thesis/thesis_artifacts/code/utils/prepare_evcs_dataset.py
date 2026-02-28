#!/usr/bin/env python3
"""
EVCS (COCO) -> Ultralytics YOLO dataset builder

Creates the standard Ultralytics detection dataset layout:
  out/
    images/{train,val,test}/...
    labels/{train,val,test}/...

Label format (YOLO): one .txt per image, each line:
  class x_center y_center width height   (normalized 0..1)

References:
  - Ultralytics Dataset Format: https://docs.ultralytics.com/datasets/detect/
  - YOLO Label Format: https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Iterable, Tuple


# -----------------------------
# COCO -> YOLO conversion
# -----------------------------

@dataclass(frozen=True)
class CocoTask:
    """One COCO annotation file + the directory where image file_name paths are rooted."""
    json_path: Path
    image_root: Path


def coco_tasks_for_evcs(data_root: Path) -> List[CocoTask]:
    """
    EVCS PartA: training.json / validation.json / test.json under PartA root,
               images under PartA/{training,validation,test}/...
    EVCS PartB: additional.json under PartB root,
               images under PartB/additional/...
    """
    part_a = data_root / "EVCSDataset_VISCODA_V1.1_PartA"
    part_b = data_root / "EVCSDataset_VISCODA_V1.1_PartB"

    tasks: List[CocoTask] = []
    if part_a.exists():
        for split in ["training", "validation", "test"]:
            jp = part_a / f"{split}.json"
            ir = part_a / split
            if jp.exists() and ir.exists():
                tasks.append(CocoTask(jp, ir))

    if part_b.exists():
        jp = part_b / "additional.json"
        ir = part_b / "additional"
        if jp.exists() and ir.exists():
            tasks.append(CocoTask(jp, ir))

    return tasks


def yolo_line_from_coco_bbox(
    bbox_xywh: List[float], img_w: int, img_h: int, class_id: int = 0
) -> str | None:
    """
    COCO bbox: [x, y, w, h] in pixels (top-left origin).
    YOLO: normalized [x_center, y_center, w, h].
    
    THESIS-CRITICAL: Clamps bounding boxes to image bounds via intersection.
    COCO annotations can have boxes partially or fully outside image bounds.
    This function computes the intersection of the box with the image rectangle,
    ensuring only the valid visible portion is converted to YOLO format.
    
    Returns:
        YOLO format string if intersection exists, None if box is fully outside.
    """
    x, y, w, h = bbox_xywh
    
    # ROBUSTNESS FIX: Compute intersection with image bounds
    # Correct clamp: left edge can't be < 0, right edge can't be > img_w
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(img_w), x + w)
    y2 = min(float(img_h), y + h)
    
    # Compute clamped dimensions
    w_clamped = x2 - x1
    h_clamped = y2 - y1
    
    # Skip degenerate boxes (no intersection with image)
    if w_clamped <= 0 or h_clamped <= 0:
        return None
    
    # Compute normalized center and dimensions from clamped box
    xc = (x1 + w_clamped / 2.0) / img_w
    yc = (y1 + h_clamped / 2.0) / img_h
    wn = w_clamped / img_w
    hn = h_clamped / img_h
    
    return f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


def convert_coco_json_to_yolo_labels(task: CocoTask, class_id: int = 0) -> Dict[Path, List[str]]:
    """
    Parses COCO JSON and returns YOLO labels IN MEMORY.
    
    THESIS-CRITICAL FIX: Does NOT write to filesystem anymore.
    This keeps the original dataset clean (no .txt files created next to images).
    Labels are returned as a dictionary: {image_path: [yolo_lines]}.
    
    THESIS-CRITICAL: Validates single-category assumption before conversion.
    """
    data = json.loads(task.json_path.read_text(encoding="utf-8"))
    
    # THESIS-CRITICAL: Multi-category to single-class mapping
    # EVCS dataset contains 8 different charger types, but thesis task is binary detection:
    # "Is there an EVCS?" not "What type of EVCS?"
    # All categories are mapped to class_id=0 (EVCS present)
    cats = data.get("categories", [])
    if len(cats) == 0:
        raise ValueError(f"{task.json_path.name}: No categories found in COCO JSON")
    
    if len(cats) == 1:
        category_name = cats[0].get("name", "unknown")
        print(f"  → Single category: '{category_name}'")
    else:
        # Multi-class dataset: map all to single class for binary detection
        category_names = [c.get('name', 'unknown') for c in cats]
        print(f"  → Multi-class dataset: {len(cats)} charger types found")
        print(f"    {category_names}")
        print(f"  → Mapping ALL to single class (class_id=0) for binary EVCS detection")

    images_by_id: Dict[int, dict] = {img["id"]: img for img in data.get("images", [])}

    anns_by_image: Dict[int, List[dict]] = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    labels_dict: Dict[Path, List[str]] = {}
    for img_id, img in images_by_id.items():
        file_name = img["file_name"]              # may include subfolders (e.g., site_*/xxx.jpg)
        img_path = task.image_root / file_name

        # Skip if the image file is missing (keeps script robust)
        if not img_path.exists():
            continue

        img_w = int(img["width"])
        img_h = int(img["height"])
        lines = []

        for ann in anns_by_image.get(img_id, []):
            line = yolo_line_from_coco_bbox(ann["bbox"], img_w, img_h, class_id)
            if line is not None:  # Skip invalid/degenerate boxes after clamping
                lines.append(line)

        labels_dict[img_path] = lines

    return labels_dict


# -----------------------------
# Split + copy to Ultralytics layout
# -----------------------------

def collect_all_images(tasks: Iterable[CocoTask]) -> List[Path]:
    """
    Collect images from the source folders.
    Supports: .jpg, .jpeg, .png (common EVCS formats)
    """
    imgs: List[Path] = []
    for t in tasks:
        imgs.extend(sorted(t.image_root.rglob("*.jpg")))
        imgs.extend(sorted(t.image_root.rglob("*.jpeg")))
        imgs.extend(sorted(t.image_root.rglob("*.png")))
    return imgs


def group_by_site(images: List[Path], data_root: Path) -> Dict[str, List[Path]]:
    """
    Group images by site_* folder to enable grouped splitting (prevents data leakage).
    
    THESIS-CRITICAL: Images from the same site/location must stay in the same split
    to avoid leakage. This function extracts the site identifier from the image path.
    
    Args:
        images: List of image paths
        data_root: Dataset root (to compute relative paths)
    
    Returns:
        Dictionary mapping site identifiers to lists of image paths
    
    Example:
        PartA/training/site_01/cam1/img001.jpg → group: "site_01"
        PartB/additional/site_042/img002.jpg → group: "site_042"
    """
    groups: Dict[str, List[Path]] = {}
    
    for img in images:
        # Find site_* in the path components
        site_id = None
        try:
            rel_path = img.relative_to(data_root)
            for part in rel_path.parts:
                if part.startswith('site_'):
                    site_id = part
                    break
        except ValueError:
            pass  # img not under data_root
        
        # Fallback: if no site_* found, use parent folder name
        if site_id is None:
            # Use a combination of parent folders to create unique group
            # This handles cases where site structure is different
            if len(img.parts) >= 2:
                site_id = f"{img.parts[-3]}_{img.parts[-2]}" if len(img.parts) >= 3 else img.parts[-2]
            else:
                site_id = "unknown"
        
        groups.setdefault(site_id, []).append(img)
    
    return groups


def split_by_groups(
    groups: Dict[str, List[Path]], 
    seed: int, 
    ratios=(0.80, 0.10, 0.10)
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    THESIS-CRITICAL: Split GROUPS (sites), not individual images.
    
    This ensures all images from the same site/location stay in the same split,
    preventing data leakage between train/val/test. The actual split ratios may
    deviate from the target ratios due to varying group sizes.
    
    Args:
        groups: Dictionary mapping site IDs to lists of images
        seed: Random seed for reproducibility
        ratios: Target split ratios (train, val, test)
    
    Returns:
        (train_images, val_images, test_images)
    
    Note:
        Because we split by GROUPS (not images), the actual image counts may
        differ from exact ratio due to varying images per site. This is the
        methodologically correct approach to prevent leakage.
    """
    assert abs(sum(ratios) - 1.0) < 1e-9
    
    # Shuffle SITE IDs (not images)
    rng = random.Random(seed)
    site_ids = list(groups.keys())
    rng.shuffle(site_ids)
    
    # Split sites into train/val/test
    n_sites = len(site_ids)
    n_train = int(n_sites * ratios[0])
    n_val = int(n_sites * (ratios[0] + ratios[1]))
    
    train_sites = site_ids[:n_train]
    val_sites = site_ids[n_train:n_val]
    test_sites = site_ids[n_val:]
    
    # Collect all images from each site group
    train = [img for site in train_sites for img in groups[site]]
    val = [img for site in val_sites for img in groups[site]]
    test = [img for site in test_sites for img in groups[site]]
    
    return train, val, test


def copy_split(
    images: List[Path], 
    labels_dict: Dict[Path, List[str]], 
    out_root: Path, 
    split: str,
    data_root: Path
) -> Tuple[int, int]:
    """
    Copies images and writes YOLO labels directly to output directory.
    
    THESIS-CRITICAL FIX: Labels are written from memory (labels_dict), NOT from filesystem.
    This keeps the original dataset clean (no .txt files created next to images).
    
    THESIS-CRITICAL FIX: Preserves full relative path from data_root to prevent collisions.
    This handles arbitrary folder depths (site_*/cam_*/frame_* etc.) collision-free.
    
    Output structure:
      out_root/images/{split}/<rel_path_from_data_root>
      out_root/labels/{split}/<rel_path_from_data_root>
    """
    img_out = out_root / "images" / split
    lbl_out = out_root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    copied, empty_labels = 0, 0
    for img in images:
        # THESIS-CRITICAL: Preserve full relative path to avoid collisions
        # This supports arbitrary nesting (e.g., PartA/training/site_01/cam_1/frame_001.jpg)
        try:
            rel = img.relative_to(data_root)
        except ValueError:
            # Fallback if img is not under data_root (shouldn't happen, but be robust)
            rel = Path(img.parent.name) / img.name
        dst_img = img_out / rel
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst_img)

        # Write label from memory (NOT from filesystem)
        dst_lbl = (lbl_out / rel).with_suffix(".txt")
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)

        lines = labels_dict.get(img, [])
        if lines:
            dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
            copied += 1
        else:
            # Empty label file for background images (no annotations)
            dst_lbl.write_text("", encoding="utf-8")
            empty_labels += 1

    return copied, empty_labels


def write_data_yaml(out_root: Path, class_name: str = "EVCS") -> Path:
    """
    Ultralytics data.yaml format: path + relative image dirs + nc + names.
    Uses dict format for names ({0: EVCS}) for explicit class-index mapping.
    Reference: https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format
    """
    yaml_text = f"""path: {out_root}
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: {class_name}
"""
    yaml_path = out_root / "data.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def main():
    ap = argparse.ArgumentParser(
        description="Convert EVCS COCO dataset to Ultralytics YOLO format",
        epilog="""
THESIS-CRITICAL Split Strategy (Site-Based / Grouped Splitting):

This script implements GROUPED SPLITTING by site/location to prevent data leakage.
All images from the same site (site_* folder) are assigned to the SAME split.

Method:
  1. Parse COCO annotations for PartA + PartB
  2. Group images by site_* identifier extracted from path
  3. Randomly assign SITES (not images) to train/val/test splits
  4. Target ratio: 80/10/10 (may deviate due to varying images per site)

Why grouped splitting?
  - Images from the same location are highly correlated (same background, lighting, angle)
  - Random splitting would cause LEAKAGE: test images too similar to training
  - Grouped splitting ensures generalization to NEW locations, not just new frames

References:
  - Kapoor & Narayanan (2023): Leakage and the Reproducibility Crisis in ML
  - Ultralytics YOLO docs: https://docs.ultralytics.com/datasets/detect/
"""
    )
    ap.add_argument("--data-root", type=Path, required=True, help="Folder containing PartA/PartB directories")
    ap.add_argument("--out", type=Path, required=True, help="Output dataset root (Ultralytics layout)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducible site assignment")
    args = ap.parse_args()

    tasks = coco_tasks_for_evcs(args.data_root)
    if not tasks:
        raise SystemExit("No COCO tasks found. Check dataset paths and json names.")

    print("Step 1/3: COCO -> YOLO label conversion (in-memory)")
    all_labels: Dict[Path, List[str]] = {}
    for t in tasks:
        labels_dict = convert_coco_json_to_yolo_labels(t, class_id=0)
        all_labels.update(labels_dict)
        print(f"  - {t.json_path.name}: parsed labels for {len(labels_dict)} images")
    print(f"  Total images with labels: {len(all_labels)}")

    print("\nStep 2/3: Collect + group + split images (site-based)")
    images = collect_all_images(tasks)
    print(f"  Found {len(images)} images total")
    
    # THESIS-CRITICAL: Group by site to prevent leakage
    groups = group_by_site(images, args.data_root)
    print(f"  Grouped into {len(groups)} sites/locations")
    
    # Split SITES (not images) to keep all images from same site together
    train, val, test = split_by_groups(groups, seed=args.seed)
    print(f"  Split sizes: train={len(train)} val={len(val)} test={len(test)}")
    print(f"  Actual ratios: {len(train)/len(images):.1%} / {len(val)/len(images):.1%} / {len(test)/len(images):.1%}")
    print(f"  Note: Ratios may deviate from 80/10/10 due to grouped (site-based) splitting")

    print("\nStep 3/3: Copy into Ultralytics dataset layout")
    for split_name, split_imgs in [("train", train), ("val", val), ("test", test)]:
        copied, empty = copy_split(split_imgs, all_labels, args.out, split_name, args.data_root)
        print(f"  - {split_name}: {len(split_imgs)} images, {copied} labeled, {empty} empty-label")

    yaml_path = write_data_yaml(args.out, class_name="EVCS")
    print(f"\nDone! data.yaml written to: {yaml_path}")


if __name__ == "__main__":
    main()
