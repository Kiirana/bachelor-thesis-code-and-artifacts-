# Thesis Implementation — Curb & EV-Charger Detection on iOS

Mobile detection pipeline combining YOLO object detection, MobileNetV3 texture classification, and Core ML on-device inference for road surface type recognition and EV charging station localization. Developed as part of a Bachelor's thesis.

---

## What Was Built

### 1. Datasets

#### EVCS (EV Charging Station Detection)
A site-based YOLO detection dataset. Sites are exclusively assigned to one split — no site appears in more than one split — preventing data leakage.

| Split | Images |
|-------|--------|
| Train | 3,512  |
| Val   | 495    |
| Test  | 397    |
| **Total** | **4,404** |

- Single class: `EVCS`
- Prepared by `prepare_evcs_dataset.py`
- Config: `evcs_yolo_site_based/data.yaml`

#### Texture Classification
Patch-based road surface classification dataset assembled from RSCD and RoadSaW. Stratified split preserves class distribution across splits.

| Split | Patches  |
|-------|----------|
| Train | 419,997  |
| Val   | 52,498   |
| Test  | 52,503   |
| **Total** | **524,998** |

- 4 classes: `asphalt`, `cobblestone`, `gravel`, `sand`
- Train class counts: Asphalt 245,780 · Gravel 85,083 · Sand 84,949 · Cobblestone 4,185
- Imbalance ratio (IR): 245,780 / 4,185 ≈ **58.73**
- Dataset directory: `texture_data_final/`

---

### 2. Models Trained

#### YOLO — EVCS Detection (YOLO12m)



Training scripts:
- `train_yolo12m_evcs_robust.py` — full training with configurable augmentation mode

Checkpoints: `models/yolo_evcs/yolo_robust/` (in artifacts).



### 4. iOS App

MVVM Swift application running the full pipeline on-device (sequentially):

```
1. yolo_car_detection         YOLO12n — full-frame car detection
2. roi_extraction             deterministic crop below car bounding box
3. mobilenet_classification   MobileNetV3-Large — texture class on ROI
4. yolo_ev_detection          YOLO12m EV charger detector — full frame
```

> **Note:** The implementation chapter (Ch. 5) describes the pipeline generically with MobileNetV3-Small;
> the final on-device evaluation (Sec. 6.7) was performed with **MobileNetV3-Large**.

Fallback when no car detected: bottom 25% of image (min 180 px height).

Key files in `ios_app/`:
- `DetectionViewModel.swift` — single-image inference + `LatencyTracker`
- `BatchEvaluationViewModel.swift` — batch evaluation, exports JSON + CSV to Documents
- `LatencyTracker.swift` — `CACurrentMediaTime()` stage stopwatch

Batch mode exports:
- `latency_stats.json` — mean/std/min/max per pipeline stage
- `latency_per_image.csv` — per-image raw timings

Core ML export for MobileNetV3:
```python
import coremltools as ct
traced = torch.jit.trace(model.eval(), torch.rand(1, 3, 224, 224))
mlmodel = ct.convert(traced,
    inputs=[ct.ImageType(shape=(1, 3, 224, 224), name="image")],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16)
mlmodel.save("MobileNetV3Texture.mlpackage")
```

---

## Project Structure

```
thesisModels/
│
├── ── Training Scripts ─────────────────────────────────────
├── train_texture_mnv3.py            # MobileNetV3-Small baseline
├── train_texture_mnv3_weighted.py   # MobileNetV3-Small with class-weighted loss
├── train_texture_mnv3_robust.py     # MobileNetV3-Small with geometric augmentation
├── train_texture_mnv3_large.py      # MobileNetV3-Large baseline
├── train_yolo12m_evcs_robust.py     # YOLO12m EVCS (baseline + robust via --mode)
│
├── ── Data Preparation ─────────────────────────────────────
├── prepare_evcs_dataset.py          # Site-based EVCS split (no site leakage)
├── prepare_texture_dataset.py       # Texture ImageFolder from RSCD + RoadSaW
├── validate_evcs_dataset.py         # EVCS dataset integrity checks
│
├── ── Evaluation Scripts ───────────────────────────────────
├── evaluate_robustness.py           # Texture robustness (29 conditions)
├── evaluate_yolo_robustness.py      # YOLO robustness (23 conditions)
├── gen_confusion_matrix.py          # Confusion matrix generation
├── gen_pr_curve.py                  # PR curve generation
├── create_pr_curve.py               # PR curve from results.csv
├── sanity_check.py                  # Cross-reference sanity checks
├── export_baseline_models.py        # CoreML export (MobileNetV3)
├── export_large_baseline.py         # CoreML export (MobileNetV3-Large)
│
├── ── Shell Launchers ──────────────────────────────────────
├── START_TRAINING.sh                # Launch all texture model training
├── START_YOLO_TRAINING.sh           # Launch YOLO training
├── train_all_models.sh              # Train all texture variants sequentially
├── train_all_yolo_models.sh         # Train all YOLO variants
├── evaluate_all_models.sh           # Run all texture evaluations
├── evaluate_all_yolo_models.sh      # Run all YOLO evaluations
│
├── ── Trained Models: Texture (MobileNetV3) ────────────────
├── ThesisRUNSFinal/
│   ├── 01_baseline/                 # best.pt · train_log.json · robustness_val.json
│   │                                  confusion_matrix_normalized.{pdf,png}
│   ├── 02_class_weighted/           # best.pt · train_log.json · robustness_val.json
│   ├── 03_robust_augmentation/      # best.pt · train_log.json · robustness_val.json
│   ├── 04_large_baseline/           # best.pt · train_log.json
│   └── *.log                        # 8 training session logs with timestamps
│
├── ── Trained Models: YOLO12m (EVCS Detection) ─────────────
├── FolderYoloTrainedModel/
│   ├── yolo_baseline/               # best.pt · args.yaml · results.csv · results.png
│   │                                  test_results.json · robustness_val.json
│   │                                  confusion_matrix{_normalized}.png
│   │                                  Box{F1,P,PR,R}_curve.png · labels.jpg
│   │                                  train_batch*.jpg · val_batch*.jpg
│   │                                  weights/ (epoch checkpoints)
│   └── yolo_robust/                 # same structure as yolo_baseline
│
├── ── iOS Application (Swift/SwiftUI) ──────────────────────
├── ios_app/
│   ├── CurbDetectorApp.swift        # App entry point
│   ├── ContentView.swift            # Main UI
│   ├── CameraView.swift             # Camera capture
│   ├── DetectionViewModel.swift     # Single-image inference + LatencyTracker
│   ├── BatchEvaluationView.swift    # Batch evaluation UI
│   ├── BatchEvaluationViewModel.swift  # Batch evaluation logic, JSON/CSV export
│   └── LatencyTracker.swift         # CACurrentMediaTime() stage stopwatch
│
├── ── Thesis Source ────────────────────────────────────────
├── alteThesisFixBedarf.tex          # Main thesis LaTeX file
├── bibtex.bib                       # Bibliography
├── preamble.tex                     # LaTeX preamble
├── content/
│   └── attachments/                 # Figures: PR curves, confusion matrices,
│                                      simulator screenshots, architecture diagrams
│
├── ── Other ────────────────────────────────────────────────
├── 100BatchrealTest.json            # GT manifest (100 images, 43/48/6/3)
├── pr_curve_data.json               # PR curve data points (1000 R/P pairs)
├── yolo12m.pt                       # Pretrained YOLO12m weights (COCO)
├── requirements_frozen.txt          # Pinned Python dependencies
└── .gitignore
```

---

## Training

### Texture — MobileNetV3-Small

```bash
# Baseline (no augmentation)
python train_texture_mnv3.py \
  --data-root texture_data_final \
  --out runs/01_baseline \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-3 \
  --weight-decay 1e-2 \
  --seed 42 \
  --device mps

# Robust augmentation (rotation + perspective)
python train_texture_mnv3_robust.py \
  --data-root texture_data_final \
  --out runs/03_robust \
  --epochs 10 \
  --seed 42 \
  --device mps

# Class-weighted loss (inverse-frequency weights)
python train_texture_mnv3_weighted.py \
  --data-root texture_data_final \
  --out runs/02_weighted \
  --epochs 10 \
  --seed 42 \
  --device mps
```

Optimizer: **AdamW** (lr=1e-3, weight_decay=1e-2) + **CosineAnnealingLR** (T_max=10).  
Pre-trained weights: `MobileNet_V3_Small_Weights.DEFAULT` (ImageNet).  
Best checkpoint saved by **val macro-F1**.

### EVCS — YOLO12m Detection

Both variants use the same script with `--mode` to switch between standard and geometric augmentation:

```bash
# Baseline (standard YOLO augmentation: mosaic, flips, minimal HSV)
python train_yolo12m_evcs_robust.py \
  --data evcs_yolo_site_based/data.yaml \
  --mode baseline \
  --epochs 100 \
  --batch 16 \
  --patience 20 \
  --seed 42 \
  --device mps \
  --name yolo12m_evcs_baseline

# Robust augmentation (adds rotation ±180°, perspective 0.001, shear ±20°)
python train_yolo12m_evcs_robust.py \
  --data evcs_yolo_site_based/data.yaml \
  --mode robust \
  --epochs 100 \
  --batch 16 \
  --patience 20 \
  --seed 42 \
  --device mps \
  --name yolo12m_evcs_robust
```

Core ML export:
```bash
yolo export model=FolderYoloTrainedModel/yolo_baseline/weights/best.pt format=coreml imgsz=640
```

---

## Robustness Evaluation

```bash
# Texture — 29 conditions (incl. clean), stratified subset (100 per class, n=400)
python evaluate_robustness.py \
  --checkpoint ThesisRUNSFinal/01_baseline/best.pt \
  --data-root texture_data_final \
  --out ThesisRUNSFinal/01_baseline/robustness_val.json

# YOLO — 23 conditions, full val split (495 images)
python evaluate_yolo_robustness.py \
  --model FolderYoloTrainedModel/yolo_baseline/weights/best.pt \
  --data evcs_yolo_site_based/data.yaml \
  --out FolderYoloTrainedModel/yolo_baseline/robustness_val.json
```

---

## Thesis File — `alteThesisFixBedarf.tex`

All metric values in the thesis are sourced directly from the training artifact JSON files on disk. No values are fabricated.

| Thesis Section | Content | Primary Artifact Source |
|----------------|---------|------------------------|
| 6.1 Experimentelles Setup | Dataset sizes, IR, hardware | filesystem counts, `meta.json` |
| 6.2 E-Ladestation-Detektion: YOLO12m | mAP50, mAP50-95, P, R | `yolo_baseline/results.csv`, `test_results.json` |
| 6.3 Bodentextur-Klassifikation: MNV3-Small | F1, P, R, confusion matrix | `01_baseline/train_log.json` |
| 6.4 Validitätseinschränkungen | Discussion of test limits | — (prose) |
| 6.5 Ablationsstudien | Large-model + class-weighting | `04_large_baseline/`, `02_class_weighted/train_log.json` |
| 6.6 Robustheitsanalyse | 29+23 condition tables, 2 models each | all `robustness_val.json` files |
| 6.7 On-Device-Test | End-to-end latency, accuracy | `100BatchrealTest.json`, iOS console output |
| 6.8 Fehleranalyse und Limitation | Qualitative + references robustness | derived from 6.2–6.6 |

---

## Requirements

```
torch>=2.1
torchvision>=0.16
ultralytics>=8.3
coremltools>=7.0
scikit-learn
numpy
```

Install: `pip install -r requirements_frozen.txt`

Device support: **MPS** (Apple Silicon), CUDA, CPU.

---

## Key Numbers at a Glance

| Item | Value |
|------|-------|
| EVCS train / val / test | 3,512 / 495 / 397 |
| Texture train / val / test | 419,997 / 52,498 / 52,503 |
| Texture imbalance ratio | 58.73 |
| YOLO baseline mAP50 (test) | 74.40% |
| YOLO robust mAP50 (test) | 65.04% |
| Texture baseline test F1 | 0.9656 |
| Texture robust test F1 | 0.9595 † |
| MNV3-Large test F1 | 0.9700 |
| YOLO input size | 640 × 640 px |
| MobileNetV3 input size | 224 × 224 px |

† Artifact value (`train_log.json`). The thesis text states 0.9612; the discrepancy
is noted — the artifact is regarded as the authoritative source.

---

**Last updated:** February 2026
