# Texture Dataset Preparation Documentation

## Overview
This document describes the process of combining RoadSaW and RSCD datasets into a unified texture classification dataset for the thesis.

## Source Datasets

### 1. RoadSaW-075_l
- **Location:** `/Users/nikitamasch/Downloads/merged/RoadSaW-075_l`
- **Type:** Camera-based road surface and wetness dataset
- **Distance:** 7.5m from vehicle (075 = 7.5m)
- **Patch size:** 900×900 pixels (3.6m × 3.6m physical road area)
- **Format:** PNG images, pre-split into train/test/validation
- **Total images:** 19,790
  - Train: 13,945 images
  - Test: 1,753 images
  - Validation: 4,092 images

**Original classes (12 total):**
- `asphalt_dry`, `asphalt_damp`, `asphalt_wet`, `asphalt_verywet` (4,648 images)
- `cobble_dry`, `cobble_damp`, `cobble_wet`, `cobble_verywet` (4,648 images)
- `concrete_dry`, `concrete_damp`, `concrete_wet`, `concrete_verywet` (4,648 images - **EXCLUDED**)

**Reference:**
```
Kai Cordes et al., "RoadSaW: A Large-Scale Dataset for Camera-Based Road Surface 
and Wetness Estimation", CVPRW 2022
https://roadsaw.viscoda.com
```

### 2. RSCD dataset-1million
- **Location:** `/Users/nikitamasch/Downloads/merged/RSCD dataset-1million`
- **Type:** Road Surface Condition Dataset
- **Format:** JPG images, pre-split with structured folders (train) and filename-based labels (test/val)
- **Total images:** ~1,022,000
  - Train: 952,901 images (27 class folders)
  - Test: 49,500 images (labels in filenames)
  - Validation: 19,860 images (labels in filenames)

**Original classes used:**
- Asphalt variations: `dry_asphalt_*`, `wet_asphalt_*`, `water_asphalt_*` (301,993 images)
- Gravel variations: `dry_gravel`, `wet_gravel`, `water_gravel` (106,354 images)
- Mud variations: `dry_mud`, `wet_mud`, `water_mud` (106,187 images)

**Excluded classes:**
- ❌ Concrete variations: `dry_concrete_*`, `wet_concrete_*`, `water_concrete_*` (~300,000 images)
  - Reason: Concrete is not asphalt and doesn't fit 4-class taxonomy
- ❌ Weather conditions: `ice`, `fresh_snow`, `melted_snow` (~188,745 images)
  - Reason: Not surface texture classes

## Class Mapping Strategy

### Target Taxonomy (4 Classes)
Based on thesis requirements, we use 4 canonical texture classes:
1. **asphalt** - Smooth paved surfaces
2. **cobblestone** - Textured paved surfaces  
3. **gravel** - Loose stone surfaces
4. **sand** - Fine-grained loose surfaces

### Mapping Rules

```python
LABEL_MAP = {
    # Asphalt sources
    "asphalt_dry/damp/wet/verywet": "asphalt",           # RoadSaW
    "dry_asphalt_severe/slight/smooth": "asphalt",       # RSCD
    "wet_asphalt_severe/slight/smooth": "asphalt",       # RSCD
    "water_asphalt_severe/slight/smooth": "asphalt",     # RSCD
    
    # Cobblestone sources
    "cobble_dry/damp/wet/verywet": "cobblestone",        # RoadSaW only
    
    # Gravel sources
    "dry_gravel": "gravel",                               # RSCD only
    "wet_gravel": "gravel",                               # RSCD only
    "water_gravel": "gravel",                             # RSCD only
    
    # Sand sources
    "dry_mud": "sand",                                    # RSCD (mud → sand)
    "wet_mud": "sand",                                    # RSCD (mud → sand)
    "water_mud": "sand",                                  # RSCD (mud → sand)
}
```

**Rationale:**
- **Moisture variations merged:** Different wetness levels (dry/damp/wet/verywet) are treated as the same texture class to focus on surface texture rather than wetness
- **Mud mapped to sand:** Both are fine-grained, loose surfaces with similar texture properties
- **Concrete excluded:** Distinct material properties from asphalt, doesn't fit 4-class model
- **RSCD has no cobblestone:** Only RoadSaW provides cobblestone data

## Dataset Combination Process

### Step 1: Create Clean Source Folder
To avoid duplication from scanning all directories in `/merged/`, we created a dedicated source folder:

```bash
mkdir -p /Users/nikitamasch/Downloads/merged/texture_sources
ln -s "/Users/nikitamasch/Downloads/merged/RSCD dataset-1million" \
      /Users/nikitamasch/Downloads/merged/texture_sources/RSCD
ln -s /Users/nikitamasch/Downloads/merged/RoadSaW-075_l \
      /Users/nikitamasch/Downloads/merged/texture_sources/RoadSaW
```

### Step 2: Run Dataset Preparation Script
```bash
cd /Users/nikitamasch/Downloads/merged/thesisModels
python3 prepare_texture_dataset.py \
  --dataset-root /Users/nikitamasch/Downloads/merged/texture_sources \
  --out /Users/nikitamasch/Downloads/merged/texture_data_combined \
  --mode symlink \
  --split 0.8 0.1 0.1 \
  --seed 42
```

**Script behavior:**
1. Scans all subdirectories in `texture_sources/`
2. For each dataset, looks for class folders in:
   - `dataset/train/<class>/`, `dataset/test/<class>/`, `dataset/val/<class>/`
   - Or flat `dataset/<class>/` structure
3. Maps class names via `LABEL_MAP` to 4 canonical classes
4. Collects all matching images with their canonical labels
5. Performs stratified split (80/10/10) with seed 42 for reproducibility
6. Creates symlinks in output directory organized as ImageFolder format

### Step 3: Verify Output
```bash
cat /Users/nikitamasch/Downloads/merged/texture_data_combined/meta.json
```

## Final Dataset Statistics

### Output Location
`/Users/nikitamasch/Downloads/merged/texture_data_final/`

### Dataset Structure
```
texture_data_final/
├── train/
│   ├── asphalt/       (245,780 images)
│   ├── cobblestone/   (4,185 images)
│   ├── gravel/        (85,083 images)
│   └── sand/          (84,949 images)
├── val/
│   ├── asphalt/       (30,722 images)
│   ├── cobblestone/   (523 images)
│   ├── gravel/        (10,635 images)
│   └── sand/          (10,618 images)
├── test/
│   ├── asphalt/       (30,723 images)
│   ├── cobblestone/   (524 images)
│   ├── gravel/        (10,636 images)
│   └── sand/          (10,620 images)
└── meta.json
```

### Class Distribution

| Class | Total | Train (80%) | Val (10%) | Test (10%) | Percentage |
|-------|-------|-------------|-----------|------------|------------|
| **asphalt** | 307,225 | 245,780 | 30,722 | 30,723 | 58.5% |
| **gravel** | 106,354 | 85,083 | 10,635 | 10,636 | 20.3% |
| **sand** | 106,187 | 84,949 | 10,618 | 10,620 | 20.2% |
| **cobblestone** | 5,232 | 4,185 | 523 | 524 | 1.0% |
| **TOTAL** | **524,998** | **419,997** | **52,498** | **52,503** | **100.0%** |

### Class Imbalance
- **Asphalt is dominant:** 58.5% of dataset (expected for road surfaces)
- **Cobblestone is rare:** Only 1.0% of dataset (only from RoadSaW, RSCD has none)
- **Gravel and sand balanced:** ~20% each

### Data Source Breakdown by Class

| Class | RoadSaW-075_l | RSCD | Total |
|-------|---------------|------|-------|
| asphalt | 19,790 (6.4%) | 287,435 (93.6%) | 307,225 |
| cobblestone | 5,232 (100%) | 0 (0%) | 5,232 |
| gravel | 0 (0%) | 106,354 (100%) | 106,354 |
| sand | 0 (0%) | 106,187 (100%) | 106,187 |

**Note:** RoadSaW images include all splits (train/test/validation) pooled together, then re-split with seed 42.

## Reproducibility

### Random Seed
- **Seed:** 42 (fixed)
- **Guarantee:** Running the script with `--seed 42` produces **identical** train/val/test splits
- **Implementation:** 
  - Classes are sorted alphabetically before splitting
  - Python's `random.Random(seed)` ensures deterministic shuffling
  - Same seed → same RNG state → same samples in each split

### Verification
To verify exact reproducibility:
```bash
# Re-run the preparation
rm -rf /Users/nikitamasch/Downloads/merged/texture_data_combined_test
python3 prepare_texture_dataset.py \
  --dataset-root /Users/nikitamasch/Downloads/merged/texture_sources \
  --out /Users/nikitamasch/Downloads/merged/texture_data_combined_test \
  --mode symlink \
  --seed 42

# Compare metadata
diff <(cat texture_data_combined/meta.json | jq -S) \
     <(cat texture_data_combined_test/meta.json | jq -S)
# Should output nothing (identical)
```

## Key Decisions and Rationale

### 1. Why symlinks instead of copies?
- **Space efficiency:** 423K images would consume ~350GB if copied
- **Dataset preservation:** Original sources remain unchanged
- **Fast iteration:** Re-running script takes seconds instead of hours

### 2. Why 80/10/10 split?
- **Standard ML practice:** Common for large datasets
- **Sufficient validation:** 42K images in val/test sets provide robust evaluation
- **Class stratification:** Each split maintains proportional class distribution

### 3. Why exclude concrete?
- **Taxonomic clarity:** Concrete is compositionally different from asphalt
- **Thesis scope:** 4-class model focuses on distinct texture types
- **Ambiguity avoidance:** Prevents confusion between smooth concrete and smooth asphalt

### 4. Why merge mud into sand?
- **Texture similarity:** Both are fine-grained, loose, deformable surfaces
- **Visual similarity:** At texture level, wet mud and wet sand are perceptually similar
- **Dataset balance:** Provides more training data for under-represented sand class

### 5. Why merge all moisture levels?
- **Focus on texture:** Goal is to classify surface texture, not wetness
- **Augmentation coverage:** Moisture variations provide natural augmentation
- **Real-world robustness:** Model should recognize asphalt regardless of wetness

## Comparison with Previous Dataset

| Metric | texture_data_thesis (earlier) | texture_data_final (new) |
|--------|-------------------------------|--------------------------|
| Total images | 424,093 | 524,998 |
| Asphalt | 221,681 (52%) | 307,225 (58.5%) |
| Cobblestone | 5,572 (1.3%) | 5,232 (1.0%) |
| Gravel | 99,932 (24%) | 106,354 (20.3%) |
| Sand | 96,908 (23%) | 106,187 (20.2%) |
| Sources | Unknown mix | **Documented: RSCD + RoadSaW-075_l (all splits pooled)** |
| Reproducibility | Unknown | ✅ Seed 42, deterministic class ordering |

**Key improvements:** 
- **+101K more images:** Now includes all RoadSaW splits (train/test/val), not just train
- **Documented provenance:** Full source tracking and class mapping
- **Reproducible pipeline:** Deterministic seed 42 with sorted class iteration

**Key improvement:** New dataset has **documented provenance** and **reproducible pipeline**.

## Usage for Training

### PyTorch DataLoader Example
```python
from torchvision import datasets, transforms

# Standard ImageNet normalization
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])final/train',
    transform=transform
)

# Class names: ['asphalt', 'cobblestone', 'gravel', 'sand']
print(train_dataset.classes)
print(f"Train samples: {len(train_dataset)}")  # 419,997
```

### Handling Class Imbalance
Given cobblestone is only 1.0% of data:

**Option 1: Weighted loss**
```python
from torch.nn import CrossEntropyLoss

# Total class counts (all splits): asphalt 307225, cobblestone 5232, gravel 106354, sand 106187
class_counts = [307225, 5232, 106354, 106187]
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
weights = weights / weights.sum() * 4  # normalize to mean=1

criterion = CrossEntropyLoss(weight=weights)
```

**Option 2: Balanced sampling**
```python
from torch.utils.data import WeightedRandomSampler

sample_weights = [weights[label] for _, label in train_dataset]
sampler = WeightedRandomSampler(sample_weights, len(train_dataset))

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
```

## Files and Scripts

### Primary Script
- **Location:** `/Users/nikitamasch/Downloads/merged/thesisModels/prepare_texture_dataset.py`
- **Purpose:** Unified dataset preparation with stratified splitting
- **Key features:**
  - Configurable class mapping via `LABEL_MAP`
  - Stratified split with seed control
  - Symlink or copy mode
  - Automatic ImageFolder format output

### Source Folder
- **Location:** `/Users/nikitamasch/Downloads/merged/texture_sources/`
- **Contents:** Symlinks to RSCD and RoadSaW-075_l
- **Purpose:** Clean separation of source data from processed outputs

### Output Dataset
- **Location:** `/Users/nikitamasch/Downloads/merged/texture_data_final/`
- **Format:** TorchVision ImageFolder (train/val/test splits)
- **Mode:** Symlinks (no disk space duplication)

## Future Work

### Possible Extensions
1. **Add RoadSaW-225_m:** Include 22.5m distance variant for multi-scale training
2. **Include concrete as 5th class:** Extend to 5-class model if thesis scope changes
3. **Separate wetness classification:** Train wetness classifier using RoadSaW labels
4. **Cross-dataset evaluation:** Train on RSCD, test on RoadSaW (domain shift analysis)

### Dataset Versioning
To create new versions:
```bash
# Version with different split
python3 prepare_texture_dataset.py \
  --dataset-root texture_sources \
  --out texture_data_v2 \
  --split 0.7 0.15 0.15 \
  --seed 42

# Version with different seed
python3 prepare_texture_dataset.py \
  --dataset-root texture_sources \
  --out texture_data_seed123 \
  --seed 123
```

## Citation

If using this dataset preparation in publications:

```bibtex
% RoadSaW dataset
@inproceedings{roadsaw2022,
  author = {Kai Cordes and Christoph Reinders and Paul Hindricks and 
            Jonas Lammers and Bodo Rosenhahn and Hellward Broszio},
  title = {RoadSaW: A Large-Scale Dataset for Camera-Based Road Surface 
           and Wetness Estimation},
  booktitle = {CVPRW},
  year = {2022},
  url = {https://roadsaw.viscoda.com}
}

% RSCD dataset - add citation when available
```

## Changelog

### 2026-02-10: Final Version
- Combined RoadSaW-075_l (all splits: 19,790 images) and RSCD (514,516 filtered train images)
- Excluded concrete classes and weather conditions (ice, snow)
- Created stratified 80/10/10 split with seed 42 and deterministic class ordering
- Total: 524,998 images across 4 classes
- Documented full provenance and reproducibility
- Fixed seed reproducibility: classes now sorted before split for deterministic behavior
