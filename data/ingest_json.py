"""
ingest_json.py — Load an ECG JSON file into the DB and export annotated PNG(s).

Usage:
    python data/ingest_json.py --file path/to/recording.json
    python data/ingest_json.py --file path/to/recording.json --output outputs/png/

Expected JSON format:
    {
        "patient_id":  "P001",          # optional
        "filename":    "recording_001", # optional (defaults to file stem)
        "fs":          500,             # required — original sampling rate (Hz)
        "label":       "Atrial Fibrillation",  # optional — ground-truth label
        "signal":      [0.12, 0.15, ...]       # required — 1-D float array (any lead)
    }

The script:
  1. Resamples signal to 125 Hz
  2. Segments into 10-second windows (1 250 samples)
  3. Detects R-peaks per window
  4. Inserts each window into ecg_features_annotatable
  5. Runs rhythm + ectopy inference on each window
  6. Saves one PNG per window with waveform, beat markers, and prediction
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import psycopg2
from scipy.signal import resample_poly, find_peaks

# ---------------------------------------------------------------------------
# Project root on sys.path so we can import internal modules
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "models_training"))

# ---------------------------------------------------------------------------
# Constants (must match retrain.py / data_loader.py)
# ---------------------------------------------------------------------------
TARGET_FS        = 125
WINDOW_SAMPLES   = 1_250    # 10 s × 125 Hz
MIN_WINDOW_SAMP  = 500      # discard trailing windows shorter than this

ECTOPY_LABELS = {
    "PVC", "PAC",
    "PVC Bigeminy", "PVC Trigeminy", "PVC Quadrigeminy",
    "PAC Bigeminy", "PAC Trigeminy",
}

INSERT_SQL = """
    INSERT INTO ecg_features_annotatable (
        dataset_source, patient_id, filename, segment_index,
        signal_data, raw_signal, features_json,
        arrhythmia_label, segment_fs, events_json
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (filename, segment_index) DO NOTHING
    RETURNING segment_id;
"""

DB_PARAMS = {
    "dbname":   "ecg_analysis",
    "user":     "ecg_user",
    "password": "sais",
    "host":     "127.0.0.1",
    "port":     "5432",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connect():
    return psycopg2.connect(**DB_PARAMS)


def _resample(signal: list[float], src_fs: int) -> np.ndarray:
    """Resample a 1-D signal from src_fs to TARGET_FS using polyphase filter."""
    sig = np.asarray(signal, dtype=np.float32)
    if src_fs == TARGET_FS:
        return sig
    from math import gcd
    g = gcd(TARGET_FS, src_fs)
    up, down = TARGET_FS // g, src_fs // g
    return resample_poly(sig, up, down).astype(np.float32)


def _segment(signal: np.ndarray) -> list[np.ndarray]:
    """Split 1-D signal into 10-second windows. Discards short trailing window."""
    windows = []
    offset = 0
    while offset + MIN_WINDOW_SAMP <= len(signal):
        chunk = signal[offset : offset + WINDOW_SAMPLES]
        if len(chunk) < WINDOW_SAMPLES:
            # Pad last window with zeros so it's exactly WINDOW_SAMPLES
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
        windows.append(chunk)
        offset += WINDOW_SAMPLES
    return windows


def _detect_r_peaks(window: np.ndarray) -> list[int]:
    """Detect R-peak indices within a single window using scipy find_peaks."""
    try:
        # Minimum distance between peaks: ~40 bpm → 125*60/40 = 187 samples
        height_threshold = np.percentile(window, 75)
        peaks, _ = find_peaks(
            window,
            distance=int(TARGET_FS * 0.4),   # min 0.4 s between peaks (150 bpm max)
            height=height_threshold,
        )
        return peaks.tolist()
    except Exception:
        return []


def _build_events_json(label: str) -> list[dict]:
    """Build events_json array matching the format used by retrain.py."""
    events = [{
        "event_type":        label,
        "event_category":    "RHYTHM",
        "start_time":        0.0,
        "end_time":          10.0,
        "annotation_source": "imported",
    }]
    if label in ECTOPY_LABELS:
        events.append({
            "event_type":        label,
            "event_category":    "ECTOPY",
            "start_time":        0.0,
            "end_time":          10.0,
            "annotation_source": "imported",
        })
    return events


# ---------------------------------------------------------------------------
# Morphology Feature Extraction
# ---------------------------------------------------------------------------

def _extract_morphology(window: np.ndarray, r_peaks: list[int]) -> dict:
    """Extract ECG morphology features (P, QRS, ST, T, QTc, RR)."""
    try:
        from signal_processing.morphology import extract_morphology
        r_arr = np.array(r_peaks, dtype=int)
        return extract_morphology(window, r_arr, TARGET_FS)
    except Exception as exc:
        warnings.warn(f"Morphology extraction failed: {exc}")
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference(window: np.ndarray):
    """
    Run rhythm + ectopy models on a single 1-D window.
    Returns (rhythm_label, rhythm_conf, ectopy_label, ectopy_conf).
    Returns None values if models cannot be loaded.
    """
    try:
        import torch
        import torch.nn.functional as F
        from xai.xai import _load_model, _init_device
        from models_training.data_loader import RHYTHM_CLASS_NAMES, ECTOPY_CLASS_NAMES

        device = _init_device()

        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        # shape: [1, 1, 1250]

        with torch.no_grad():
            # Rhythm
            m_rhythm = _load_model("rhythm")
            logits_r  = m_rhythm(x)
            probs_r   = F.softmax(logits_r, dim=-1).squeeze()
            idx_r     = int(probs_r.argmax())
            rhythm_label = RHYTHM_CLASS_NAMES[idx_r]
            rhythm_conf  = float(probs_r[idx_r])

            # Temporarily suppress BBB — not trained/validated yet
            if rhythm_label == "Bundle Branch Block":
                probs_r[idx_r] = 0.0
                idx_r = int(probs_r.argmax())
                rhythm_label = RHYTHM_CLASS_NAMES[idx_r]
                rhythm_conf = float(probs_r[idx_r])

            # Ectopy
            m_ectopy = _load_model("ectopy")
            logits_e  = m_ectopy(x)
            probs_e   = F.softmax(logits_e, dim=-1).squeeze()
            idx_e     = int(probs_e.argmax())
            # Apply ectopy threshold — require confidence >= 0.6
            if idx_e != 0 and float(probs_e[idx_e]) < 0.6:
                idx_e = 0  # Default to "None"
            ectopy_label = ECTOPY_CLASS_NAMES[idx_e]
            ectopy_conf  = float(probs_e[idx_e])

        return rhythm_label, rhythm_conf, ectopy_label, ectopy_conf

    except Exception as exc:
        warnings.warn(f"Inference failed: {exc}")
        return None, None, None, None


# ---------------------------------------------------------------------------
# PNG export
# ---------------------------------------------------------------------------

def _save_png(
    window: np.ndarray,
    r_peaks: list[int],
    filename: str,
    seg_idx: int,
    gt_label: str,
    rhythm_label: str | None,
    rhythm_conf: float | None,
    ectopy_label: str | None,
    ectopy_conf: float | None,
    output_dir: Path,
) -> Path:
    """Render waveform + R-peak markers + annotations and save as PNG."""

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{filename}_seg{seg_idx}.png"

    t = np.linspace(0, WINDOW_SAMPLES / TARGET_FS, WINDOW_SAMPLES)

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.set_facecolor("#eaf4fb")
    fig.patch.set_facecolor("#0d1117")

    # Waveform
    ax.plot(t, window, color="black", linewidth=0.8, zorder=2)

    # R-peak markers (red dashed vertical lines)
    for pk in r_peaks:
        if 0 <= pk < len(t):
            ax.axvline(t[pk], color="#e03e3e", linewidth=0.8, linestyle="--", alpha=0.8, zorder=3)

    # Axes style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555")
    ax.spines["bottom"].set_color("#555")
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.set_xlabel("Time (s)", color="#aaa", fontsize=8)
    ax.set_ylabel("Amplitude (mV)", color="#aaa", fontsize=8)
    ax.set_xlim(0, WINDOW_SAMPLES / TARGET_FS)

    # Title with prediction + ground truth
    pred_text = "N/A"
    if rhythm_label is not None:
        pred_text = f"{rhythm_label} ({rhythm_conf:.0%})"
        if ectopy_label and ectopy_label != "None":
            pred_text += f"  |  Ectopy: {ectopy_label} ({ectopy_conf:.0%})"

    title = f"Segment {seg_idx}  ·  Predicted: {pred_text}  ·  Ground truth: {gt_label}"
    ax.set_title(title, color="white", fontsize=9, pad=6)

    # Legend
    legend_items = [
        mpatches.Patch(color="#e03e3e", label="R-peak"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=7,
              facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return out_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def ingest(json_path: Path, output_dir: Path) -> None:
    print(f"[ingest] Loading {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- Validate required fields ---
    if "signal" not in data:
        sys.exit("[ERROR] JSON must contain a 'signal' key with a list of floats.")
    if "fs" not in data:
        sys.exit("[ERROR] JSON must contain an 'fs' key with the sampling rate (int).")

    src_fs     = int(data["fs"])
    raw_signal = data["signal"]
    gt_label   = str(data.get("label", "Unlabeled"))
    patient_id = str(data.get("patient_id", "unknown"))
    filename   = str(data.get("filename", json_path.stem))

    print(f"[ingest] fs={src_fs} Hz  |  label='{gt_label}'  |  samples={len(raw_signal)}")

    # --- Resample ---
    signal_125 = _resample(raw_signal, src_fs)
    print(f"[ingest] Resampled → {len(signal_125)} samples @ {TARGET_FS} Hz")

    # --- Segment ---
    windows = _segment(signal_125)
    print(f"[ingest] Segmented → {len(windows)} window(s) of {WINDOW_SAMPLES} samples")

    if not windows:
        sys.exit("[ERROR] Signal too short — no complete windows extracted.")

    # --- Normalize label ---
    try:
        from models_training.data_loader import normalize_label
        norm_label = normalize_label(gt_label)
    except Exception:
        norm_label = gt_label  # fall back to raw string

    events_template = _build_events_json(norm_label)

    # --- DB insert + inference + PNG ---
    conn = _connect()
    conn.autocommit = True
    cur  = conn.cursor()

    inserted = 0
    for idx, window in enumerate(windows):
        r_peaks = _detect_r_peaks(window)
        features = {"r_peaks": r_peaks}

        cur.execute(INSERT_SQL, (
            "json_import",
            patient_id,
            filename,
            idx,
            window.tolist(),
            json.dumps(window.tolist()),
            json.dumps(features),
            norm_label,
            TARGET_FS,
            json.dumps(events_template),
        ))

        row = cur.fetchone()
        seg_id = row[0] if row else "conflict (skipped)"
        if row:
            inserted += 1
        print(f"  [seg {idx}] DB id={seg_id}  r_peaks={len(r_peaks)}")

        # Run inference
        rhythm_label, rhythm_conf, ectopy_label, ectopy_conf = _run_inference(window)

        # Extract morphology features
        morph_data = _extract_morphology(window, r_peaks)

        # Save PNG
        png_path = _save_png(
            window, r_peaks,
            filename, idx,
            gt_label=gt_label,
            rhythm_label=rhythm_label,
            rhythm_conf=rhythm_conf,
            ectopy_label=ectopy_label,
            ectopy_conf=ectopy_conf,
            output_dir=output_dir,
        )
        print(f"  [seg {idx}] PNG → {png_path}")

        # Save morphology JSON report
        report = {
            "segment_index": idx,
            "filename": filename,
            "ground_truth": gt_label,
            "prediction": {
                "rhythm": rhythm_label,
                "rhythm_confidence": rhythm_conf,
                "ectopy": ectopy_label,
                "ectopy_confidence": ectopy_conf,
            },
            "morphology": morph_data,
        }
        report_path = output_dir / f"{filename}_seg{idx}_report.json"
        with open(report_path, "w", encoding="utf-8") as rf:
            json.dump(report, rf, indent=2, default=str)
        print(f"  [seg {idx}] Report → {report_path}")

    cur.close()
    conn.close()

    print(f"\n[ingest] Done. {inserted}/{len(windows)} segment(s) inserted into DB.")
    print(f"[ingest] PNG(s) saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Ingest ECG JSON → DB + PNG export")
    p.add_argument("--file",   required=True,  type=Path, help="Path to input JSON file")
    p.add_argument("--output", default=None,   type=Path, help="Output directory for PNGs (default: outputs/png/)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    json_path = args.file.resolve()
    if not json_path.exists():
        sys.exit(f"[ERROR] File not found: {json_path}")

    output_dir = args.output.resolve() if args.output else BASE_DIR / "outputs" / "png"

    ingest(json_path, output_dir)
