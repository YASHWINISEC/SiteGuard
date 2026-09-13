"""
SiteGuard AI - Model Evaluation & Benchmark Script
Calculates Precision, Recall, F1-Score, mAP@50, and mAP@50-95 per class.
"""

import os
import json
import argparse
from pathlib import Path
from ultralytics import RTDETR, YOLO


def evaluate(
    weights_path: str = "../backend/models/best.pt",
    data_yaml: str = "../dataset/data.yaml",
    split: str = "val",
    img_size: int = 640,
    device: str = "0"
):
    root_dir = Path(__file__).resolve().parent
    weights_p = (root_dir / weights_path).resolve()
    yaml_p = (root_dir / data_yaml).resolve()

    if not weights_p.exists():
        alt_weights = root_dir / "runs" / "train" / "siteguard_rtdetr" / "weights" / "best.pt"
        if alt_weights.exists():
            weights_p = alt_weights
        else:
            print(f"[ERROR] Weights file not found at {weights_p}. Train model first via `python train.py`")
            return

    print("=" * 70)
    print("        SITEGUARD AI - COMPREHENSIVE MODEL EVALUATION")
    print("=" * 70)
    print(f"Model Checkpoint: {weights_p}")
    print(f"Dataset Config:   {yaml_p}")
    print(f"Split:            {split.upper()}")
    print(f"Resolution:       {img_size}x{img_size}")
    print("=" * 70)

    try:
        model = RTDETR(str(weights_p))
    except Exception:
        model = YOLO(str(weights_p))

    metrics = model.val(
        data=str(yaml_p),
        split=split,
        imgsz=img_size,
        device=device,
        plots=True,
        save_json=True
    )

    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map_full = float(metrics.box.map)
    f1_overall = (2 * mp * mr) / (mp + mr + 1e-6)

    print("\n" + "=" * 70)
    print("                   OVERALL PERFORMANCE METRICS")
    print("=" * 70)
    print(f"  Mean Precision (mP):     {mp * 100:.2f}%")
    print(f"  Mean Recall (mR):        {mr * 100:.2f}%")
    print(f"  Overall F1-Score:        {f1_overall * 100:.2f}%")
    print(f"  mAP@0.50:                {map50 * 100:.2f}%")
    print(f"  mAP@0.50:0.95:           {map_full * 100:.2f}%")
    print("=" * 70)

    # Class-wise Metrics Table
    class_names = model.names
    print("\n--- Per-Class Performance Breakdown ---")
    print(f"{'Class Name':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'mAP@50':<12}")
    print("-" * 68)

    class_results = {}
    if hasattr(metrics.box, 'maps') and len(metrics.box.maps) > 0:
        for idx, (cls_id, cls_name) in enumerate(class_names.items()):
            p = float(metrics.box.p[idx]) if idx < len(metrics.box.p) else 0.0
            r = float(metrics.box.r[idx]) if idx < len(metrics.box.r) else 0.0
            f1 = (2 * p * r) / (p + r + 1e-6)
            m50 = float(metrics.box.all_ap[idx][0]) if hasattr(metrics.box, 'all_ap') and len(metrics.box.all_ap) > idx else 0.0

            print(f"{cls_name:<20} {p*100:>8.1f}%   {r*100:>8.1f}%   {f1*100:>8.1f}%   {m50*100:>8.1f}%")
            class_results[cls_name] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1_score": round(f1, 4),
                "map50": round(m50, 4)
            }

    # Save to JSON
    report_file = root_dir / "evaluation_report.json"
    report_data = {
        "overall": {
            "precision": round(mp, 4),
            "recall": round(mr, 4),
            "f1_score": round(f1_overall, 4),
            "map50": round(map50, 4),
            "map50_95": round(map_full, 4)
        },
        "per_class": class_results
    }
    with open(report_file, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[INFO] Detailed evaluation report saved to: {report_file}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SiteGuard Model Performance")
    parser.add_argument("--weights", type=str, default="../backend/models/best.pt", help="Path to model weights")
    parser.add_argument("--data", type=str, default="../dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        weights_path=args.weights,
        data_yaml=args.data,
        split=args.split,
        img_size=args.imgsz,
        device=args.device
    )
