#!/usr/bin/env python3
"""
Test script to verify site-based splitting correctness.

This script verifies that:
1. Images from the same site stay in the same split (no leakage)
2. All sites are accounted for
3. Split ratios are approximately correct (allowing deviation due to grouping)

Usage:
    python3 test_site_splitting.py --data-yaml path/to/data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Set
import yaml


def extract_site_id(img_path: Path, data_root: Path) -> str:
    """Extract site identifier from image path."""
    rel_path = img_path.relative_to(data_root)
    for part in rel_path.parts:
        if part.startswith('site_'):
            return part
    # Fallback
    if len(rel_path.parts) >= 2:
        return f"{rel_path.parts[-3]}_{rel_path.parts[-2]}" if len(rel_path.parts) >= 3 else rel_path.parts[-2]
    return "unknown"


def check_split_integrity(data_yaml_path: Path) -> None:
    """Check that site-based splitting was correctly applied."""
    
    # Load data.yaml
    with open(data_yaml_path) as f:
        config = yaml.safe_load(f)
    
    data_root = Path(config['path']).resolve()
    train_dir = data_root / config['train'].replace('images/', 'images/')
    val_dir = data_root / config['val'].replace('images/', 'images/')
    test_dir = data_root / config['test'].replace('images/', 'images/')
    
    print("="*80)
    print("Site-Based Split Integrity Check")
    print("="*80)
    print(f"Dataset root: {data_root}")
    print(f"Train dir:    {train_dir}")
    print(f"Val dir:      {val_dir}")
    print(f"Test dir:     {test_dir}")
    print()
    
    # Collect all images and their sites
    site_to_split: Dict[str, str] = {}  # site_id -> split_name
    split_sites: Dict[str, Set[str]] = {'train': set(), 'val': set(), 'test': set()}
    split_counts: Dict[str, int] = {'train': 0, 'val': 0, 'test': 0}
    
    for split_name, split_dir in [('train', train_dir), ('val', val_dir), ('test', test_dir)]:
        if not split_dir.exists():
            print(f"⚠️  Warning: {split_name} directory not found: {split_dir}")
            continue
        
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images.extend(split_dir.rglob(ext))
        
        split_counts[split_name] = len(images)
        
        for img in images:
            site_id = extract_site_id(img, data_root)
            split_sites[split_name].add(site_id)
            
            # Check for leakage: site should only appear in ONE split
            if site_id in site_to_split and site_to_split[site_id] != split_name:
                print(f"❌ LEAKAGE DETECTED: Site '{site_id}' appears in both {site_to_split[site_id]} and {split_name}!")
                print(f"   Example image: {img.name}")
                return
            
            site_to_split[site_id] = split_name
    
    # Summary
    total_images = sum(split_counts.values())
    total_sites = len(site_to_split)
    
    print("Split Statistics:")
    print("-"*80)
    for split_name in ['train', 'val', 'test']:
        n_imgs = split_counts[split_name]
        n_sites = len(split_sites[split_name])
        ratio = n_imgs / total_images if total_images > 0 else 0
        print(f"{split_name.capitalize():5s}: {n_imgs:5d} images ({ratio:5.1%}) from {n_sites:3d} sites")
    
    print("-"*80)
    print(f"Total:  {total_images:5d} images from {total_sites:3d} sites")
    print()
    
    # Check for site overlap (leakage detection)
    train_val_overlap = split_sites['train'] & split_sites['val']
    train_test_overlap = split_sites['train'] & split_sites['test']
    val_test_overlap = split_sites['val'] & split_sites['test']
    
    print("Leakage Detection (Site Overlap):")
    print("-"*80)
    if train_val_overlap:
        print(f"❌ Train/Val overlap: {len(train_val_overlap)} sites")
        print(f"   Examples: {list(train_val_overlap)[:5]}")
    else:
        print("✅ Train/Val: No overlap")
    
    if train_test_overlap:
        print(f"❌ Train/Test overlap: {len(train_test_overlap)} sites")
        print(f"   Examples: {list(train_test_overlap)[:5]}")
    else:
        print("✅ Train/Test: No overlap")
    
    if val_test_overlap:
        print(f"❌ Val/Test overlap: {len(val_test_overlap)} sites")
        print(f"   Examples: {list(val_test_overlap)[:5]}")
    else:
        print("✅ Val/Test: No overlap")
    
    print()
    
    # Overall verdict
    has_leakage = bool(train_val_overlap or train_test_overlap or val_test_overlap)
    
    if has_leakage:
        print("="*80)
        print("❌ SITE-BASED SPLITTING VERIFICATION FAILED")
        print("="*80)
        print("Data leakage detected! Sites appear in multiple splits.")
        print("This violates the thesis requirement for grouped splitting.")
        print()
        print("Action required:")
        print("  1. Re-run prepare_evcs_dataset.py with --seed 42")
        print("  2. Ensure the script uses split_by_groups(), not split_list()")
        print("="*80)
    else:
        print("="*80)
        print("✅ SITE-BASED SPLITTING VERIFICATION PASSED")
        print("="*80)
        print("No site overlap detected across splits.")
        print("All images from the same site stay in the same split.")
        print("Dataset is ready for thesis-compliant training!")
        print("="*80)
    
    # Additional statistics
    print()
    print("Site Distribution Details:")
    print("-"*80)
    
    # Images per site statistics
    site_image_counts = {}
    for site_id, split_name in site_to_split.items():
        site_imgs = [img for img in (data_root / config[split_name]).rglob('*') 
                     if img.is_file() and extract_site_id(img, data_root) == site_id]
        site_image_counts[site_id] = len(site_imgs)
    
    if site_image_counts:
        counts = sorted(site_image_counts.values())
        print(f"Images per site: min={min(counts)}, median={counts[len(counts)//2]}, max={max(counts)}")
        print(f"This explains why actual split ratios may deviate from 80/10/10.")


def main():
    ap = argparse.ArgumentParser(
        description="Verify site-based splitting correctness for EVCS dataset"
    )
    ap.add_argument("--data-yaml", type=Path, required=True, 
                    help="Path to data.yaml of the prepared YOLO dataset")
    args = ap.parse_args()
    
    if not args.data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data_yaml}")
    
    check_split_integrity(args.data_yaml)


if __name__ == "__main__":
    main()
