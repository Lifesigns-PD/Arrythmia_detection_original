#!/usr/bin/env python3
"""
compare_models.py — Compare v1 (signal-only) vs v2 (signal+features) models
=============================================================================

Loads both checkpoints and evaluates them on the SAME validation set,
printing side-by-side metrics so you can decide which model to deploy.

Usage:
    python scripts/compare_models.py --task rhythm
    python scripts/compare_models.py --task ectopy
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "models_training"))

from data_loader import RHYTHM_CLASS_NAMES, ECTOPY_CLASS_NAMES
from models import CNNTransformerClassifier
from models_v2 import CNNTransformerWithFeatures
from signal_processing.feature_extraction import extract_feature_vector, NUM_FEATURES

CHECKPOINTS = BASE_DIR / "models_training" / "outputs" / "checkpoints"


def evaluate_model(model, loader, device, num_classes, use_features=False):
    """Run inference and return metrics."""
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            if use_features:
                x, f, y = batch
                x, f, y = x.to(device), f.to(device), y.to(device)
                logits = model(x, f)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)

            y_true += y.cpu().tolist()
            y_pred += logits.argmax(1).cpu().tolist()

    y_arr  = np.array(y_true)
    yp_arr = np.array(y_pred)

    overall_acc = float((y_arr == yp_arr).mean())
    per_cls = {}
    for i in range(num_classes):
        mask = y_arr == i
        if mask.sum() > 0:
            per_cls[i] = float((yp_arr[mask] == i).mean())
        else:
            per_cls[i] = 0.0

    balanced_acc = float(np.mean(list(per_cls.values())))

    return {
        "overall_acc": overall_acc,
        "balanced_acc": balanced_acc,
        "per_class": per_cls,
        "total_samples": len(y_arr),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["rhythm", "ectopy"], default="rhythm")
    args = parser.parse_args()

    class_names = RHYTHM_CLASS_NAMES if args.task == "rhythm" else ECTOPY_CLASS_NAMES
    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v1_path = CHECKPOINTS / f"best_model_{args.task}.pth"
    v2_path = CHECKPOINTS / f"best_model_{args.task}_v2.pth"

    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON  |  task={args.task.upper()}")
    print(f"{'='*70}")

    # Check which checkpoints exist
    has_v1 = v1_path.exists()
    has_v2 = v2_path.exists()

    if not has_v1 and not has_v2:
        print("[ERROR] No checkpoints found. Train at least one model first.")
        sys.exit(1)

    # Print checkpoint info
    if has_v1:
        v1_state = torch.load(v1_path, map_location=device, weights_only=False)
        print(f"\n  V1 (signal-only):  {v1_path.name}")
        print(f"    Epoch:        {v1_state.get('epoch', '?')}")
        print(f"    Balanced Acc: {v1_state.get('balanced_acc', 0):.4f}")
        print(f"    Mode:         {v1_state.get('mode', '?')}")
    else:
        print(f"\n  V1: NOT FOUND at {v1_path}")

    if has_v2:
        v2_state = torch.load(v2_path, map_location=device, weights_only=False)
        print(f"\n  V2 (signal+features):  {v2_path.name}")
        print(f"    Epoch:        {v2_state.get('epoch', '?')}")
        print(f"    Balanced Acc: {v2_state.get('balanced_acc', 0):.4f}")
        print(f"    Mode:         {v2_state.get('mode', '?')}")
        print(f"    Features:     {v2_state.get('num_features', '?')} dimensions")
    else:
        print(f"\n  V2: NOT FOUND at {v2_path}")

    # Build evaluation dataset (use retrain.py's ECGEventDataset for v1, v2 for v2)
    print(f"\n  Loading evaluation data...")

    if has_v2:
        from retrain_v2 import ECGEventDatasetV2, collate_fn_v2, filename_split as fs_v2
        ds_v2 = ECGEventDatasetV2(task=args.task, source_filter="all", augment=False)
        _, val_idx_v2 = fs_v2(ds_v2.samples)
        val_ds_v2 = torch.utils.data.Subset(ds_v2, val_idx_v2)
        val_ldr_v2 = torch.utils.data.DataLoader(val_ds_v2, batch_size=32, shuffle=False, collate_fn=collate_fn_v2)

    if has_v1:
        from retrain import ECGEventDataset, collate_fn, filename_split as fs_v1
        ds_v1 = ECGEventDataset(task=args.task, source_filter="all", augment=False)
        _, val_idx_v1 = fs_v1(ds_v1.samples)
        val_ds_v1 = torch.utils.data.Subset(ds_v1, val_idx_v1)
        val_ldr_v1 = torch.utils.data.DataLoader(val_ds_v1, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Evaluate V1
    if has_v1:
        print(f"\n  Evaluating V1 (signal-only)...")
        model_v1 = CNNTransformerClassifier(num_classes=num_classes).to(device)
        model_v1.load_state_dict(v1_state["model_state"])
        metrics_v1 = evaluate_model(model_v1, val_ldr_v1, device, num_classes, use_features=False)
    else:
        metrics_v1 = None

    # Evaluate V2
    if has_v2:
        print(f"  Evaluating V2 (signal+features)...")
        model_v2 = CNNTransformerWithFeatures(
            num_classes=num_classes, num_features=v2_state.get("num_features", NUM_FEATURES)
        ).to(device)
        model_v2.load_state_dict(v2_state["model_state"])
        metrics_v2 = evaluate_model(model_v2, val_ldr_v2, device, num_classes, use_features=True)
    else:
        metrics_v2 = None

    # Print comparison
    print(f"\n{'='*70}")
    print(f"  RESULTS COMPARISON")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<30} {'V1 (signal)':>15} {'V2 (sig+feat)':>15} {'Delta':>10}")
    print(f"  {'-'*70}")

    if metrics_v1 and metrics_v2:
        v1_oa = metrics_v1["overall_acc"]
        v2_oa = metrics_v2["overall_acc"]
        v1_ba = metrics_v1["balanced_acc"]
        v2_ba = metrics_v2["balanced_acc"]

        print(f"  {'Overall Accuracy':<30} {v1_oa:>14.4f} {v2_oa:>14.4f} {v2_oa-v1_oa:>+10.4f}")
        print(f"  {'Balanced Accuracy':<30} {v1_ba:>14.4f} {v2_ba:>14.4f} {v2_ba-v1_ba:>+10.4f}")
        print(f"  {'Val Samples':<30} {metrics_v1['total_samples']:>15} {metrics_v2['total_samples']:>15}")

        print(f"\n  Per-class accuracy:")
        for i, name in enumerate(class_names):
            v1_c = metrics_v1["per_class"].get(i, 0)
            v2_c = metrics_v2["per_class"].get(i, 0)
            delta = v2_c - v1_c
            marker = " ✓" if delta > 0.01 else " ✗" if delta < -0.01 else ""
            print(f"    {i:02d} {name:<35} {v1_c:>8.3f} {v2_c:>8.3f} {delta:>+8.3f}{marker}")

        print(f"\n  VERDICT: ", end="")
        if v2_ba > v1_ba + 0.01:
            print(f"V2 is BETTER by {v2_ba-v1_ba:+.4f} balanced accuracy. Use v2!")
        elif v1_ba > v2_ba + 0.01:
            print(f"V1 is still better by {v1_ba-v2_ba:+.4f}. Keep v1, tune v2 more.")
        else:
            print(f"Models are comparable (delta={v2_ba-v1_ba:+.4f}). V2 may improve with more training.")

    elif metrics_v1:
        print(f"  {'Overall Accuracy':<30} {metrics_v1['overall_acc']:>14.4f} {'N/A':>15}")
        print(f"  {'Balanced Accuracy':<30} {metrics_v1['balanced_acc']:>14.4f} {'N/A':>15}")
        print(f"\n  Train v2 first: python models_training/retrain_v2.py --task {args.task} --mode initial")

    elif metrics_v2:
        print(f"  {'Overall Accuracy':<30} {'N/A':>15} {metrics_v2['overall_acc']:>14.4f}")
        print(f"  {'Balanced Accuracy':<30} {'N/A':>15} {metrics_v2['balanced_acc']:>14.4f}")

    print()


if __name__ == "__main__":
    main()
