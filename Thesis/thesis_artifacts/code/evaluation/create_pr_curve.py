#!/usr/bin/env python3
"""
Create a Precision-Recall Curve PDF for YOLO EVCS detection (Ultralytics official metrics).
- Runs Ultralytics validation (model.val) to compute PR/Confidence curves
- Plots PR curve and marks operating point at a chosen confidence threshold
- Saves a PDF (and optional PNG)

Works with Ultralytics YOLO models and their DetMetrics/Metric curve outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO


def _to_2d_array(x) -> np.ndarray:
    """
    Convert Ultralytics curve container (list of arrays or ndarray) to shape (nc, npts).
    """
    if x is None:
        raise ValueError("Expected curve data but got None.")

    if isinstance(x, np.ndarray):
        arr = x
    else:
        # usually list length nc, each element length npts
        arr = np.stack(list(x), axis=0)

    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def generate_pr_curve_pdf(
    weights: Path,
    out_pdf: Path,
    data: str | None = None,
    split: str = "val",
    op_conf: float = 0.25,
    imgsz: int | None = None,
    device: str | None = None,
    also_png: bool = True,
) -> tuple[float, float, float]:
    """
    Returns: (op_precision, op_recall, map50)
    """
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))

    # IMPORTANT: conf=0.001 is the standard validation setting to compute PR curves across thresholds.
    # plots=False to avoid writing extra images unless you want Ultralytics' default plot output.
    val_kwargs = dict(
        split=split,
        conf=0.001,
        plots=False,
        verbose=False,
        save_json=False,
    )
    if data is not None:
        val_kwargs["data"] = data
    if imgsz is not None:
        val_kwargs["imgsz"] = imgsz
    if device is not None:
        val_kwargs["device"] = device

    print(f"🔍 Running validation on {split} split...")
    metrics = model.val(**val_kwargs)

    # mAP@0.5 (mean across classes)
    map50 = float(metrics.box.map50)

    # ---- PR curve from official curves_results ----
    # curves_results[0] = [px, prec_values, "Recall", "Precision"]
    pr_x, pr_y, _, _ = metrics.box.curves_results[0]
    recall_grid = np.asarray(pr_x, dtype=float)

    # prec_values: per-class precision values along recall grid
    prec_values = _to_2d_array(pr_y)  # (nc, npts)
    precision_curve = prec_values.mean(axis=0)  # "all classes" mean curve

    # ---- Operating point from Precision-Confidence / Recall-Confidence curves ----
    # curves_results[2] = [px, p_curve, "Confidence", "Precision"]
    # curves_results[3] = [px, r_curve, "Confidence", "Recall"]
    conf_grid = np.asarray(metrics.box.px, dtype=float)
    p_curve = _to_2d_array(metrics.box.p_curve).mean(axis=0)  # mean across classes
    r_curve = _to_2d_array(metrics.box.r_curve).mean(axis=0)

    op_idx = int(np.argmin(np.abs(conf_grid - op_conf)))
    op_p = float(p_curve[op_idx])
    op_r = float(r_curve[op_idx])

    print(f"✓ mAP@0.5 = {map50:.4f}")
    print(f"✓ Operating point (conf={op_conf}): P={op_p:.4f}, R={op_r:.4f}")

    # ---- Plot (Ultralytics-like, but as vector-ready PDF) ----
    fig, ax = plt.subplots(figsize=(9, 6), tight_layout=True)

    ax.plot(
        recall_grid,
        precision_curve,
        linewidth=2.5,
        color='#3498db',
        label=f"all classes (mAP@0.5={map50:.3f})",
    )
    ax.fill_between(recall_grid, precision_curve, alpha=0.15, color='#3498db')

    ax.plot(
        op_r,
        op_p,
        "o",
        markersize=10,
        color='#e74c3c',
        markeredgecolor='darkred',
        markeredgewidth=2,
        zorder=5,
        label=f"Operating point (conf={op_conf:.2f}, P={op_p:.3f}, R={op_r:.3f})",
    )

    ax.set_xlabel("Recall", fontsize=12, fontweight='bold')
    ax.set_ylabel("Precision", fontsize=12, fontweight='bold')
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)

    _ensure_dir(out_pdf)
    fig.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight")
    print(f"✅ PR curve saved: {out_pdf}")
    
    if also_png:
        out_png = out_pdf.with_suffix(".png")
        fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight")
        print(f"✅ Preview saved: {out_png}")

    plt.close(fig)

    # Write a small sidecar file so you can paste exact values into the caption
    sidecar = out_pdf.with_suffix(".txt")
    sidecar.write_text(
        f"Operating point @ conf={op_conf:.2f}\n"
        f"Precision={op_p:.6f}\n"
        f"Recall={op_r:.6f}\n"
        f"mAP50={map50:.6f}\n",
        encoding="utf-8",
    )
    print(f"✅ Metrics saved: {sidecar}")

    return op_p, op_r, map50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path, help="Path to best.pt")
    ap.add_argument("--out", required=True, type=Path, help="Output PDF path, e.g. figures/ev_pr_curve.pdf")
    ap.add_argument("--data", default=None, help="Optional dataset YAML. If omitted, model remembers training data.")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split to validate on.")
    ap.add_argument("--op-conf", default=0.25, type=float, help="Operating confidence threshold to mark.")
    ap.add_argument("--imgsz", default=None, type=int, help="Optional image size override.")
    ap.add_argument("--device", default=None, help="Optional device, e.g. cuda:0, cpu, mps.")
    args = ap.parse_args()

    print("=" * 70)
    print("📊 Generating Ultralytics PR Curve from Validation")
    print("=" * 70)

    op_p, op_r, map50 = generate_pr_curve_pdf(
        weights=args.weights,
        out_pdf=args.out,
        data=args.data,
        split=args.split,
        op_conf=args.op_conf,
        imgsz=args.imgsz,
        device=args.device,
    )

    print("\n✅ Done! Use in LaTeX: \\includegraphics{figures/ev_pr_curve.pdf}")
    print(f"📊 Final metrics: mAP@0.5={map50:.4f}, P={op_p:.4f}, R={op_r:.4f}")


if __name__ == "__main__":
    # For direct execution with default paths
    if True:  # Set to False to use command-line args
        weights = Path("/Users/nikitamasch/Downloads/merged/runs/thesis/yolo12m_evcs_baseline/weights/best.pt")
        out_pdf = Path("/Users/nikitamasch/Downloads/merged/thesis_figures/ev_pr_curve.pdf")
        
        print("=" * 70)
        print("📊 Generating Ultralytics PR Curve from Validation")
        print("=" * 70)
        
        op_p, op_r, map50 = generate_pr_curve_pdf(
            weights=weights,
            out_pdf=out_pdf,
            split="val",
            op_conf=0.25,
            device="mps",
        )
        
        print("\n✅ Done! Use in LaTeX: \\includegraphics{figures/ev_pr_curve.pdf}")
        print(f"📊 Final metrics: mAP@0.5={map50:.4f}, P={op_p:.4f}, R={op_r:.4f}")
    else:
        main()
