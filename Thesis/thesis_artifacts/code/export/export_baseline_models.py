"""
Export both BASELINE models to CoreML for iOS deployment.

Outputs (in /Users/nikitamasch/Downloads/BaselineModelsExport/):
  TextureClassifier_baseline.mlpackage  ← MobileNetV3-Small, 4 classes
  YOLODetector_baseline.mlpackage       ← YOLO12m EVCS detector
"""

import os, sys, json, types
from pathlib import Path

# Block tensorflow from crashing (NumPy 1.x vs 2.x incompatibility)
_tf_stub = types.ModuleType("tensorflow")
_tf_stub.__version__ = "0.0.0"
_tf_stub.__spec__ = None
for _sub in [
    "python", "python.framework", "python.framework.ops",
    "python.eager", "python.eager.context", "python.ops",
    "python.autograph", "python.autograph.core", "python.autograph.utils",
    "python.pywrap_tfe", "_api", "_api.v2", "_api.v2.__internal__",
    "keras", "lite", "io", "math", "nn", "random", "signal",
]:
    sys.modules[f"tensorflow.{_sub}"] = types.ModuleType(f"tensorflow.{_sub}")
sys.modules["tensorflow"] = _tf_stub

OUT_DIR       = Path("/Users/nikitamasch/Downloads/BaselineModelsExport")
TEXTURE_CKPT  = Path("/Users/nikitamasch/Downloads/merged/thesisModels/ThesisRUNSFinal/01_baseline/best.pt")
YOLO_CKPT     = Path("/Users/nikitamasch/Downloads/merged/thesisModels/FolderYoloTrainedModel/yolo_baseline/weights/best.pt")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  MobileNetV3-Small baseline  →  CoreML
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== [1/2] Exporting MobileNetV3-Small baseline texture classifier ===")

import torch
import torch.nn as nn
import coremltools as ct
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

ckpt = torch.load(TEXTURE_CKPT, map_location="cpu", weights_only=False)
classes = ckpt["classes"]   # ['asphalt', 'cobblestone', 'gravel', 'sand']
print(f"  Classes : {classes}")

base = mobilenet_v3_small(weights=None)
base.classifier[-1] = nn.Linear(base.classifier[-1].in_features, len(classes))
base.load_state_dict(ckpt["model"], strict=True)
base.eval()

example = torch.zeros(1, 3, 224, 224)
with torch.no_grad():
    traced = torch.jit.trace(base, example)

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(
        name="image",
        shape=(1, 3, 224, 224),
        scale=1.0 / 255.0,
        bias=[-m / s for m, s in zip(IMG_MEAN, IMG_STD)],
        color_layout=ct.colorlayout.RGB,
    )],
    outputs=[ct.TensorType(name="logits")],
    minimum_deployment_target=ct.target.iOS16,
    convert_to="mlprogram",
)

mlmodel.short_description = "MobileNetV3-Small road texture classifier (baseline). 4 classes: asphalt, cobblestone, gravel, sand."
mlmodel.author = "Thesis — 01_baseline"
mlmodel.version = "1.0"
mlmodel.user_defined_metadata["classes"]    = json.dumps(classes)
mlmodel.user_defined_metadata["input_size"] = "224x224"
mlmodel.user_defined_metadata["checkpoint"] = str(TEXTURE_CKPT)

out_texture = OUT_DIR / "TextureClassifier_baseline.mlpackage"
mlmodel.save(str(out_texture))
print(f"  Saved → {out_texture}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  YOLO12m baseline  →  CoreML
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== [2/2] Exporting YOLO baseline detector ===")

from ultralytics import YOLO
import shutil

yolo = YOLO(str(YOLO_CKPT))
export_path = yolo.export(
    format="coreml",
    imgsz=640,
    half=False,
    nms=True,
    conf=0.25,
    iou=0.45,
)

src = Path(export_path)
dst = OUT_DIR / "YOLODetector_baseline.mlpackage"
if dst.exists():
    shutil.rmtree(dst)
shutil.move(str(src), str(dst))
print(f"  Saved → {dst}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Export complete ===")
print(f"Output folder: {OUT_DIR}")
for f in sorted(OUT_DIR.iterdir()):
    size_mb = sum(p.stat().st_size for p in f.rglob("*") if p.is_file()) / 1e6
    print(f"  {f.name}  ({size_mb:.1f} MB)")
