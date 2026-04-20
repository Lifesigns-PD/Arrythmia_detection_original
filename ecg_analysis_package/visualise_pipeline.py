#!/usr/bin/env python3
"""
visualise_pipeline.py — End-to-end V3 signal processing visualiser
====================================================================
Runs the full V3 pipeline on ECG data and generates visualization plots:
  Row 1 — Raw signal
  Row 2 — Preprocessed (baseline removed + denoised)
  Row 3 — Final: R-peaks + P/Q/R/S/T fiducials marked

Supports JSON file input (standalone) or database input (with psycopg2).

Usage (JSON file):
    python visualise_pipeline.py --json ecg_data.json

JSON format:
    {
      "admissionId": "...",
      "ecgData": [0.12, 0.15, ...]  // 125 Hz, mV scale
    }
"""

import sys, json, argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Requested by user snippet
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from signal_processing_v3.preprocessing.pipeline import preprocess_v3
from signal_processing_v3.detection.ensemble import detect_r_peaks_ensemble, refine_peaks_subsample
from signal_processing_v3.delineation.hybrid import delineate_v3
from signal_processing_v3.features.extraction import extract_features_v3, FEATURE_NAMES_V3

DB = dict(host="127.0.0.1", dbname="ecg_analysis", user="ecg_user", password="sais", port="5432")


# ── JSON file loader ──────────────────────────────────────────────────────────
def load_from_json(json_path, record_idx=0):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load JSON: {e}")

    fs = 125
    if isinstance(data, list):
        if not data: raise ValueError("JSON empty")
        record = data[min(record_idx, len(data)-1)]
        admission_id = record.get("admissionId", "unknown")
        device_id = record.get("facilityId", "unknown")
        timestamp = record.get("packetNo", 0)
        value = record.get("value", [[]])
        ecg_data = value[0] if isinstance(value, list) and len(value) > 0 else []
        label = record.get("arrhythmia_label", "Unknown")
        segment_id = f"{admission_id}_{record_idx}"
    else:
        admission_id = data.get("admissionId", "unknown")
        device_id = data.get("deviceId", "unknown")
        timestamp = data.get("timestamp", 0)
        ecg_data = data.get("ecgData", data.get("signal", []))
        label = data.get("arrhythmia_label", "Unknown")
        segment_id = admission_id

    if not ecg_data:
        raise ValueError("JSON missing ECG data")

    return segment_id, np.array(ecg_data, dtype=np.float32), fs, label, device_id, timestamp


# ── DB fetch ──────────────────────────────────────────────────────────────────
def fetch_segment(seg_id=None, label="Sinus Rhythm"):
    import psycopg2
    conn = psycopg2.connect(**DB)
    cur  = conn.cursor()
    if seg_id:
        cur.execute("SELECT segment_id, signal_data, segment_fs, arrhythmia_label FROM ecg_features_annotatable WHERE segment_id=%s", (seg_id,))
    else:
        cur.execute("SELECT segment_id, signal_data, segment_fs, arrhythmia_label FROM ecg_features_annotatable WHERE arrhythmia_label=%s AND signal_data IS NOT NULL AND is_corrected=TRUE ORDER BY RANDOM() LIMIT 1", (label,))
    row = cur.fetchone()
    conn.close()
    if not row: raise ValueError("No segment found")
    sig = np.array(json.loads(row[1]) if isinstance(row[1], str) else row[1], dtype=np.float32)
    return row[0], sig, int(row[2] or 125), row[3]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",  type=str, default=None)
    parser.add_argument("--record", type=int, default=0)
    parser.add_argument("--seg",   type=int, default=None)
    parser.add_argument("--label", type=str, default="Sinus Rhythm")
    args = parser.parse_args()

    if args.json:
        print(f"Loading from JSON: {args.json}")
        seg_id, raw_sig, fs, label, _, _ = load_from_json(args.json, args.record)
    else:
        print("Fetching segment from DB...")
        seg_id, raw_sig, fs, label = fetch_segment(args.seg, args.label)

    print(f"  seg_id={seg_id}  label={label}  fs={fs}  samples={len(raw_sig)}")
    t = np.arange(len(raw_sig)) / fs

    # 1. Preprocess
    print("Preprocessing...")
    prep = preprocess_v3(raw_sig, fs=fs)
    cleaned = prep["cleaned"]
    sqi = prep.get("quality_score", 1.0)
    issues = prep.get("quality_issues", [])

    # 2. Detect R-peaks
    print("Detecting R-peaks...")
    r_peaks = detect_r_peaks_ensemble(cleaned, fs=fs)
    print(f"  {len(r_peaks)} R-peaks detected")

    # 3. Delineate
    print("Delineating P/Q/R/S/T...")
    delin = delineate_v3(cleaned, r_peaks, fs=fs)
    per_beat = delin["per_beat"]

    # 4. Features
    print("Extracting features...")
    r_float = refine_peaks_subsample(cleaned, r_peaks)
    features = extract_features_v3(cleaned, r_float, delin, fs=fs)

    # Summary
    print("\n" + "="*60)
    print(f"  SEGMENT {seg_id} -- {label}")
    print("="*60)
    for k in ["mean_hr_bpm", "pr_interval_ms", "mean_qrs_duration_ms", "p_absent_fraction"]:
        v = features.get(k)
        print(f"    {k:<28} {v}")
    if issues: print(f"  Quality issues: {issues}")

    # 5. Plotting (Using exact requested logic)
    print("\nPlotting...")
    fig, axes = plt.subplots(3, 1, figsize=(18, 11), sharex=True)
    fig.suptitle(
        f"V3 Pipeline — Segment {seg_id} | Label: {label} | "
        f"HR: {features.get('mean_hr_bpm', 0):.0f} bpm | "
        f"SQI: {sqi:.2f}",
        fontsize=13, fontweight="bold"
    )

    # Row 1: Raw
    ax = axes[0]
    ax.plot(t, raw_sig, color="#555555", linewidth=0.8, label="Raw signal")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title("1. Raw Signal", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 2: Preprocessed + R-peaks
    ax = axes[1]
    ax.plot(t, cleaned, color="#1a6faf", linewidth=0.9, label="Preprocessed")
    for r in r_peaks:
        ax.axvline(r/fs, color="red", alpha=0.4, linewidth=0.7)
    ax.plot(r_peaks/fs, cleaned[r_peaks], "rv", markersize=7,
            label=f"R-peaks ({len(r_peaks)})")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title("2. Preprocessed + R-Peak Detection", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 3: Full delineation — P Q R S T
    ax = axes[2]
    ax.plot(t, cleaned, color="#1a6faf", linewidth=0.9, label="Preprocessed")

    colours = {
        "p_peak":   ("#e67e22", "P",  "^", 7),
        "q_peak":   ("#8e44ad", "Q",  "v", 6),
        "r":        ("#e74c3c", "R",  "v", 8),
        "s_peak":   ("#2ecc71", "S",  "^", 6),
        "t_peak":   ("#3498db", "T",  "^", 7),
    }
    plotted = {k: False for k in colours}

    for i, beat in enumerate(per_beat):
        r = int(r_peaks[i]) if i < len(r_peaks) else None
        if r is not None and 0 <= r < len(cleaned):
            lbl_str = "R" if not plotted["r"] else ""
            ax.plot(r/fs, cleaned[r], marker=colours["r"][2], color=colours["r"][0], markersize=colours["r"][3], label=lbl_str if lbl_str else "_")
            plotted["r"] = True

        for key in ["p_peak", "q_peak", "s_peak", "t_peak"]:
            idx = beat.get(key)
            if idx is not None and 0 <= idx < len(cleaned):
                c, letter, mk, ms = colours[key]
                lbl_str = letter if not plotted[key] else ""
                ax.plot(idx/fs, cleaned[idx], marker=mk, color=c, markersize=ms, label=lbl_str if lbl_str else "_")
                plotted[key] = True

        # Shading
        p_on, p_off = beat.get("p_onset"), beat.get("p_offset")
        if p_on is not None and p_off is not None: ax.axvspan(p_on/fs, p_off/fs, alpha=0.12, color="#e67e22")
        q_on, q_off = beat.get("qrs_onset"), beat.get("qrs_offset")
        if q_on is not None and q_off is not None: ax.axvspan(q_on/fs, q_off/fs, alpha=0.15, color="#e74c3c")
        t_on, t_off = beat.get("t_onset"), beat.get("t_offset")
        if t_on is not None and t_off is not None: ax.axvspan(t_on/fs, t_off/fs, alpha=0.10, color="#3498db")

    patches = [
        mpatches.Patch(color="#e67e22", alpha=0.3, label="P-wave region"),
        mpatches.Patch(color="#e74c3c", alpha=0.3, label="QRS region"),
        mpatches.Patch(color="#3498db", alpha=0.3, label="T-wave region"),
    ]
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles=handles + patches, loc="upper right", fontsize=8, ncol=2)
    ax.set_ylabel("Amplitude (mV)")
    ax.set_xlabel("Time (s)")
    ax.set_title("3. Full Delineation (P/Q/R/S/T)", fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()
