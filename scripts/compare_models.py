#!/usr/bin/env python3
"""
compare_models.py — V1 vs V2 vs V3 Model Accuracy Comparison (CLI)
====================================================================
Loads available checkpoints and evaluates them against the same DB
validation set, printing a side-by-side accuracy table.

Usage:
    python scripts/compare_models.py --task rhythm
    python scripts/compare_models.py --task ectopy
    python scripts/compare_models.py --task rhythm --limit 300
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "models_training"))

from models_training.data_loader import RHYTHM_CLASS_NAMES, ECTOPY_CLASS_NAMES
from models_training.models import CNNTransformerClassifier
from models_training.models_v2 import CNNTransformerWithFeatures

CHECKPOINTS = BASE_DIR / "models_training" / "outputs" / "checkpoints"


# ─────────────────────────────────────────────────────────────────────────────
# Feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def _v2_num_features():
    from signal_processing.feature_extraction import NUM_FEATURES
    return NUM_FEATURES


def _v3_num_features():
    from signal_processing_v3.features.extraction import FEATURE_NAMES_V3
    return len(FEATURE_NAMES_V3)


def _extract_v2_features(sig, fs=125):
    from signal_processing.feature_extraction import extract_feature_vector
    return extract_feature_vector(sig, fs=fs)


def _extract_v3_features(sig, fs=125):
    from signal_processing_v3 import process_ecg_v3
    from signal_processing_v3.features.extraction import feature_dict_to_vector
    res = process_ecg_v3(sig, fs=fs, min_quality=0.0)
    return feature_dict_to_vector(res["features"])


# ─────────────────────────────────────────────────────────────────────────────
# DB loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_samples(task: str, limit: int, feature_version: str):
    import psycopg2
    from scipy.signal import resample
    from models_training.data_loader import get_rhythm_label_idx, get_ectopy_label_idx

    conn = psycopg2.connect(
        host="127.0.0.1", dbname="ecg_analysis",
        user="ecg_user", password="sais", port="5432",
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT segment_id, signal_data, arrhythmia_label, segment_fs, features_json
        FROM ecg_features_annotatable
        WHERE signal_data IS NOT NULL AND is_corrected = TRUE
        ORDER BY segment_id DESC LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    conn.close()

    extractor = _extract_v3_features if feature_version == "v3" else _extract_v2_features
    n_feat = _v3_num_features() if feature_version == "v3" else _v2_num_features()
    label_fn = get_rhythm_label_idx if task == "rhythm" else get_ectopy_label_idx

    samples, skipped = [], 0
    for _, sig_raw, label_txt, fs, feat_json in rows:
        sig = np.array(sig_raw, dtype=np.float32)
        fs  = int(fs or 125)
        if fs != 125 and len(sig) > 1:
            sig = resample(sig, int(len(sig) * 125 / fs)).astype(np.float32)
        sig = sig[:1250] if len(sig) > 1250 else np.pad(sig, (0, max(0, 1250 - len(sig))))

        y = label_fn(label_txt)
        if y is None:
            skipped += 1
            continue

        try:
            feats = extractor(sig, fs=125)
        except Exception:
            feats = np.zeros(n_feat, dtype=np.float32)

        samples.append({"signal": sig, "label": int(y), "features": feats})

    print(f"    Loaded {len(samples)} samples  (skipped {skipped} wrong-task labels)")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate one model on a sample list
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate(model, samples, device, use_features=True, batch_size=32):
    all_preds, all_labels = [], []
    for i in range(0, len(samples), batch_size):
        batch  = samples[i:i + batch_size]
        sigs   = torch.tensor(np.stack([s["signal"]   for s in batch]), dtype=torch.float32).unsqueeze(1).to(device)
        feats  = torch.tensor(np.stack([s["features"] for s in batch]), dtype=torch.float32).to(device)
        labels = [s["label"] for s in batch]

        with torch.no_grad():
            try:
                logits = model(sigs, feats) if use_features else model(sigs)
            except Exception:
                logits = model(sigs)
            preds = torch.argmax(logits, dim=1).cpu().tolist()

        all_preds.extend(preds)
        all_labels.extend(labels)

    cls_correct = defaultdict(int)
    cls_total   = defaultdict(int)
    for p, l in zip(all_preds, all_labels):
        cls_total[l] += 1
        if p == l:
            cls_correct[l] += 1

    overall  = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)
    cls_acc  = {c: cls_correct[c] / cls_total[c] for c in cls_total}
    balanced = float(np.mean(list(cls_acc.values()))) if cls_acc else 0.0

    return {
        "overall":  overall,
        "balanced": balanced,
        "per_class": cls_acc,
        "n":        len(all_labels),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="V1 / V2 / V3 model comparison")
    parser.add_argument("--task",  default="rhythm", choices=["rhythm", "ectopy"])
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = RHYTHM_CLASS_NAMES if args.task == "rhythm" else ECTOPY_CLASS_NAMES
    num_classes = len(class_names)

    print(f"\n{'='*72}")
    print(f"  MODEL COMPARISON  —  {args.task.upper()} task  |  device={device}")
    print(f"{'='*72}")

    # ── Checkpoints to evaluate ──────────────────────────────────────────────
    candidates = {
        "v1": {
            "path":         CHECKPOINTS / f"best_model_{args.task}.pth",
            "model_class":  "v1",
            "feat_version": "v2",   # V1 model doesn't use features at inference
            "use_features": False,
        },
        "v2": {
            "path":         CHECKPOINTS / f"best_model_{args.task}_v2.pth",
            "model_class":  "v2",
            "feat_version": "v2",
            "use_features": True,
        },
    }

    results = {}

    for ver, cfg in candidates.items():
        if not cfg["path"].exists():
            print(f"\n  [{ver.upper()}] checkpoint not found: {cfg['path'].name} — skip")
            continue

        state = torch.load(cfg["path"], map_location=device, weights_only=False)
        saved_epoch = state.get("epoch", "?")
        saved_acc   = state.get("balanced_acc", state.get("val_balanced_acc", 0.0))
        saved_feats = state.get("num_features", "?")

        print(f"\n  [{ver.upper()}] {cfg['path'].name}")
        print(f"    Epoch: {saved_epoch}  |  Saved balanced acc: {saved_acc:.4f}  |  Feat dims: {saved_feats}")

        n_feat = _v3_num_features() if cfg["feat_version"] == "v3" else _v2_num_features()

        if cfg["model_class"] == "v1":
            model = CNNTransformerClassifier(num_classes=num_classes).to(device)
            key = "model_state" if "model_state" in state else "model_state_dict"
        else:
            model = CNNTransformerWithFeatures(num_classes=num_classes, num_features=n_feat).to(device)
            key = "model_state_dict" if "model_state_dict" in state else "model_state"

        try:
            model.load_state_dict(state[key], strict=False)
        except Exception as e:
            print(f"    [warn] load_state_dict: {e}")

        model.eval()

        print(f"    Loading evaluation data (V{cfg['feat_version'][-1]} features, limit={args.limit})...")
        samples = _load_samples(args.task, args.limit, cfg["feat_version"])
        if not samples:
            print("    No samples — skip.")
            continue

        m = _evaluate(model, samples, device, use_features=cfg["use_features"])
        m["saved_acc"] = saved_acc
        results[ver] = m
        print(f"    → Overall acc:   {m['overall']*100:.2f}%")
        print(f"    → Balanced acc:  {m['balanced']*100:.2f}%")
        print(f"    → N evaluated:   {m['n']}")

    # ── Summary table ─────────────────────────────────────────────────────────
    if not results:
        print("\nNo models evaluated. Train a model first.")
        return

    print(f"\n{'='*72}")
    print(f"  SUMMARY  —  {args.task.upper()}")
    print(f"  {'Version':<8} {'Overall':>10} {'Balanced':>10} {'N':>6}")
    print(f"  {'-'*36}")
    for ver, r in results.items():
        print(f"  {ver:<8} {r['overall']*100:>9.2f}% {r['balanced']*100:>9.2f}% {r['n']:>6}")

    # Delta vs first available
    vers = list(results.keys())
    if len(vers) >= 2:
        base, comp = vers[0], vers[1]
        delta = results[comp]["balanced"] - results[base]["balanced"]
        better = comp if delta > 0 else base
        print(f"\n  Balanced acc delta ({comp} vs {base}): {delta:+.4f}")
        print(f"  Best: {better.upper()}")

    # ── Per-class breakdown ───────────────────────────────────────────────────
    print(f"\n  Per-class accuracy:")
    header = f"  {'Class':<32}"
    for ver in results:
        header += f"  {ver.upper():>8}"
    print(header)

    all_cls = sorted(set(c for r in results.values() for c in r["per_class"]))
    for cls in all_cls:
        name = class_names[cls] if cls < len(class_names) else str(cls)
        row  = f"  {name:<32}"
        for ver in results:
            val = results[ver]["per_class"].get(cls)
            row += f"  {val*100:>7.1f}%" if val is not None else f"  {'N/A':>7}"
        print(row)

    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    main()
