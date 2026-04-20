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
      "admissionId": "ADM001",
      "deviceId": "ECG-01",
      "timestamp": 1712600000,
      "ecgData": [0.12, 0.15, -0.08, ...]  // 7500 samples @ 125 Hz, mV scale
    }

Usage (Database):
    python visualise_pipeline.py                         # random Sinus Rhythm
    python visualise_pipeline.py --seg 1127             # specific segment
    python visualise_pipeline.py --label "Atrial Fibrillation"
"""

import sys, json, argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use Agg backend (works on servers without display)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from signal_processing_v3.preprocessing.pipeline import preprocess_v3
from signal_processing_v3.detection.ensemble import detect_r_peaks_ensemble, refine_peaks_subsample
from signal_processing_v3.delineation.hybrid import delineate_v3
from signal_processing_v3.features.extraction import extract_features_v3, FEATURE_NAMES_V3

DB = dict(host="127.0.0.1", dbname="ecg_analysis", user="ecg_user", password="sais", port="5432")


# ── JSON file loader ──────────────────────────────────────────────────────────
def load_from_json(json_path, record_idx=0):
    """
    Load ECG data from JSON file.
    Supports two formats:
    1. Single object: {"admissionId": "...", "ecgData": [...]}
    2. MongoDB export (list): [{"admissionId": "...", "value": [[...]]}]
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load JSON: {e}")

    fs = 125  # Standard sampling rate

    # Format 1: MongoDB export (list of records)
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError("JSON array is empty")
        if record_idx >= len(data):
            print(f"Warning: record_idx {record_idx} out of range ({len(data)} records). Using first.")
            record_idx = 0

        record = data[record_idx]
        admission_id = record.get("admissionId", "unknown")
        device_id = record.get("facilityId", "unknown")

        # Parse timestamp (MongoDB $date format)
        timestamp = 0
        ts_obj = record.get("utcTimestamp", {})
        if isinstance(ts_obj, dict) and "$date" in ts_obj:
            # MongoDB timestamp is ISO string, extract epoch
            timestamp = record.get("packetNo", 0)

        # Extract ECG data
        value = record.get("value", [[]])
        if isinstance(value, list) and len(value) > 0:
            ecg_data = value[0]  # value is [[ecg_array]]
        else:
            ecg_data = []

        label = record.get("arrhythmia_label", "Unknown")
        segment_id = f"{admission_id}_{record_idx}"

    # Format 2: Simple object with ecgData
    else:
        admission_id = data.get("admissionId", "unknown")
        device_id = data.get("deviceId", "unknown")
        timestamp = data.get("timestamp", 0)
        ecg_data = data.get("ecgData", [])
        label = data.get("arrhythmia_label", "Unknown")
        segment_id = admission_id

    if not ecg_data:
        raise ValueError("JSON missing 'ecgData' or 'value' array")

    sig = np.array(ecg_data, dtype=np.float32)

    return segment_id, sig, fs, label, device_id, timestamp


# ── DB fetch ──────────────────────────────────────────────────────────────────
def fetch_segment_from_db(seg_id=None, label="Sinus Rhythm"):
    """Fetch ECG segment from PostgreSQL database."""
    try:
        import psycopg2
    except ImportError:
        raise ImportError("psycopg2 required for DB access. Install: pip install psycopg2-binary")

    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    if seg_id:
        cur.execute(
            "SELECT segment_id, signal_data, segment_fs, arrhythmia_label "
            "FROM ecg_features_annotatable WHERE segment_id=%s", (seg_id,)
        )
    else:
        cur.execute(
            "SELECT segment_id, signal_data, segment_fs, arrhythmia_label "
            "FROM ecg_features_annotatable "
            "WHERE arrhythmia_label=%s AND signal_data IS NOT NULL "
            "AND is_corrected=TRUE ORDER BY RANDOM() LIMIT 1", (label,)
        )
    row = cur.fetchone()
    conn.close()
    if not row:
        raise ValueError(f"No segment found (seg_id={seg_id}, label={label})")
    sid, sig_raw, fs, lbl = row
    sig = np.array(json.loads(sig_raw) if isinstance(sig_raw, str) else sig_raw, dtype=np.float32)
    return str(sid), sig, int(fs or 125), lbl, None, None


# ── Signal processing pipeline ────────────────────────────────────────────────
def process_signal(raw_sig, fs):
    """Run full V3 signal processing pipeline."""
    print("Preprocessing...")
    prep = preprocess_v3(raw_sig, fs=fs)
    cleaned = prep["cleaned"]
    sqi = prep.get("quality_score", 1.0)
    issues = prep.get("quality_issues", [])

    print("Detecting R-peaks...")
    r_peaks = detect_r_peaks_ensemble(cleaned, fs=fs)
    print(f"  {len(r_peaks)} R-peaks detected")

    print("Delineating P/Q/R/S/T...")
    delin = delineate_v3(cleaned, r_peaks, fs=fs)
    per_beat = delin["per_beat"]
    summary = delin["summary"]

    print("Extracting features...")
    r_float = refine_peaks_subsample(cleaned, r_peaks)
    features = extract_features_v3(cleaned, r_float, delin, fs=fs)

    return cleaned, r_peaks, delin, features, sqi, issues


# ── NeuroKit2 comparison (optional) ───────────────────────────────────────────
def compare_with_neurokit2(raw_sig, fs):
    """Cross-check with NeuroKit2 (if available)."""
    nk_hr = nk_pr = nk_qrs = 0.0
    try:
        import neurokit2 as nk2, warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            _nk_clean = nk2.ecg_clean(raw_sig, sampling_rate=fs)
            _, _nk_peaks = nk2.ecg_peaks(_nk_clean, sampling_rate=fs)
            _nk_r = np.array(_nk_peaks["ECG_R_Peaks"], dtype=int)
            if len(_nk_r) >= 2:
                _rr = np.diff(_nk_r)
                _rr = _rr[(_rr > fs*0.2) & (_rr < fs*3)]
                if len(_rr): nk_hr = float(60 * fs / np.mean(_rr))
            _, _nk_w = nk2.ecg_delineate(_nk_clean, _nk_r, sampling_rate=fs, method="dwt", show=False)
            def _nk_idx(key, i):
                a = _nk_w.get(key, [])
                if i < len(a):
                    v = a[i]
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        return int(v)
                return None
            qrs_list, pr_list = [], []
            for _i in range(len(_nk_r)):
                on = _nk_idx("ECG_R_Onsets", _i)
                off = _nk_idx("ECG_R_Offsets", _i)
                pon = _nk_idx("ECG_P_Onsets", _i)
                if on is not None and off is not None and off > on:
                    d = (off - on) * 1000 / fs
                    if 40 <= d <= 300: qrs_list.append(d)
                if pon is not None and on is not None and on > pon:
                    p = (on - pon) * 1000 / fs
                    if 60 <= p <= 400: pr_list.append(p)
            if qrs_list: nk_qrs = float(np.median(qrs_list))
            if pr_list: nk_pr = float(np.median(pr_list))
    except Exception as _e:
        print(f"  [NeuroKit2 comparison skipped: {_e}]")

    return nk_hr, nk_pr, nk_qrs


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_pipeline(raw_sig, cleaned, r_peaks, delin, features, sqi, label, segment_id, fs, output_file=None):
    """Generate 3-row visualization of signal processing pipeline."""
    print("Plotting...")

    t = np.arange(len(raw_sig)) / fs
    per_beat = delin["per_beat"]

    fig, axes = plt.subplots(3, 1, figsize=(18, 11), sharex=True)
    fig.suptitle(
        f"V3 Pipeline — {segment_id} | Label: {label} | "
        f"HR: {features.get('mean_hr_bpm') or 0:.0f} bpm | "
        f"SQI: {sqi or 1.0:.2f}",
        fontsize=13, fontweight="bold"
    )

    # Row 1: Raw signal
    ax = axes[0]
    ax.plot(t, raw_sig, color="#555555", linewidth=0.8, label="Raw signal")
    ax.set_ylabel("Amplitude (mV)", fontsize=10)
    ax.set_title("1. Raw Signal", fontweight="bold", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Row 2: Preprocessed + R-peaks
    ax = axes[1]
    ax.plot(t, cleaned, color="#1a6faf", linewidth=0.9, label="Preprocessed")
    for r in r_peaks:
        ax.axvline(r/fs, color="red", alpha=0.4, linewidth=0.7)
    ax.plot(r_peaks/fs, cleaned[r_peaks], "rv", markersize=7, label=f"R-peaks ({len(r_peaks)})")
    ax.set_ylabel("Amplitude (mV)", fontsize=10)
    ax.set_title("2. Preprocessed + R-Peak Detection (Ensemble)", fontweight="bold", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Row 3: Full delineation
    ax = axes[2]
    ax.plot(t, cleaned, color="#1a6faf", linewidth=0.9, label="Preprocessed")

    # Mark all fiducial points
    colors = {"P": "#2ca02c", "Q": "#d62728", "R": "#ff7f0e", "S": "#9467bd", "T": "#8c564b"}
    for i, beat_data in enumerate(per_beat):
        for key in ["P_onset", "P_peak", "P_offset", "Q_peak", "R_peak", "S_peak", "T_onset", "T_peak", "T_offset"]:
            idx = beat_data.get(key)
            if idx is not None and 0 <= idx < len(cleaned):
                component = key.split("_")[0]
                color = colors.get(component, "black")
                marker = "o" if "peak" in key else "x"
                ax.plot(idx/fs, cleaned[idx], marker=marker, color=color, markersize=5, alpha=0.7)

    # Legend for components
    legend_elements = [
        mpatches.Patch(color=colors["P"], label="P-wave"),
        mpatches.Patch(color=colors["Q"], label="Q-wave"),
        mpatches.Patch(color=colors["R"], label="R-peak"),
        mpatches.Patch(color=colors["S"], label="S-wave"),
        mpatches.Patch(color=colors["T"], label="T-wave"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    ax.set_ylabel("Amplitude (mV)", fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_title("3. Waveform Delineation (P/Q/R/S/T)", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"  Saved to {output_file}")
    else:
        plt.show()

    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="V3 Signal Processing Visualiser",
        epilog="""
Examples:
  # Simple JSON (single object with ecgData):
    python visualise_pipeline.py --json sample.json

  # MongoDB export (array of records):
    python visualise_pipeline.py --json ECG_Data_Extracts/ADM1014424580.json --record 0
    python visualise_pipeline.py --json ECG_Data_Extracts/ADM1014424580.json --record 100

  # Database:
    python visualise_pipeline.py --db
    python visualise_pipeline.py --db --seg 1234
    python visualise_pipeline.py --db --label "Atrial Fibrillation"

  # Save output:
    python visualise_pipeline.py --json data.json --output report.png
        """
    )
    parser.add_argument("--json", type=str, default=None, help="Path to JSON file (simple or MongoDB export)")
    parser.add_argument("--record", type=int, default=0, help="Record index in MongoDB export array (default: 0)")
    parser.add_argument("--db", action="store_true", help="Load from database (requires psycopg2)")
    parser.add_argument("--seg", type=int, default=None, help="Specific segment ID (DB only)")
    parser.add_argument("--label", type=str, default="Sinus Rhythm", help="Arrhythmia label (DB only)")
    parser.add_argument("--output", type=str, default=None, help="Save plot to file (PNG/PDF)")
    args = parser.parse_args()

    # Load data from JSON or DB
    if args.json:
        print(f"Loading from JSON: {args.json}")
        segment_id, raw_sig, fs, label, device_id, timestamp = load_from_json(args.json, args.record)
    elif args.db:
        print("Loading from database...")
        segment_id, raw_sig, fs, label, _, _ = fetch_segment_from_db(args.seg, args.label)
    else:
        parser.print_help()
        raise ValueError("Must specify --json or --db")

    print(f"  ID={segment_id}  label={label}  fs={fs}  samples={len(raw_sig)}")

    # Run pipeline
    cleaned, r_peaks, delin, features, sqi, issues = process_signal(raw_sig, fs)

    # Optional: Compare with NeuroKit2
    nk_hr, nk_pr, nk_qrs = compare_with_neurokit2(raw_sig, fs)

    # Print summary
    print("\n" + "="*60)
    print(f"  SEGMENT {segment_id} — {label}")
    print("="*60)

    if nk_hr > 0:
        print(f"\n  {'VITAL':<10} {'V3 Pipeline':>14} {'NeuroKit2':>12}  {'Match?'}")
        print(f"  {'-'*50}")
        hr = features.get("mean_hr_bpm") or 0
        pr = features.get("pr_interval_ms") or 0
        qrs = features.get("mean_qrs_duration_ms") or 0
        def _chk(a, b, tol=15): return "✓" if abs((a or 0) - (b or 0)) < tol else "⚠"
        print(f"  {'HR (bpm)':<10} {hr:>14.1f} {nk_hr:>12.1f}  {_chk(hr, nk_hr, 10)}")
        print(f"  {'PR (ms)':<10} {pr:>14.0f} {nk_pr:>12.0f}  {_chk(pr, nk_pr, 30)}")
        print(f"  {'QRS (ms)':<10} {qrs:>14.0f} {nk_qrs:>12.0f}  {_chk(qrs, nk_qrs, 20)}")

    print("\n  KEY FEATURES:")
    key_feats = [
        "mean_hr_bpm", "pr_interval_ms", "mean_qrs_duration_ms",
        "qt_interval_ms", "sdnn_ms", "rmssd_ms",
        "p_absent_fraction", "qrs_wide_fraction", "pvc_score_mean",
        "lf_hf_ratio", "sample_entropy",
    ]
    for k in key_feats:
        v = features.get(k)
        vstr = f"{v:.3f}" if isinstance(v, float) else str(v)
        print(f"    {k:<28} {vstr}")

    none_feats = sum(1 for v in features.values() if v is None)
    print(f"\n  Total features: {len(features)}  |  None: {none_feats}  |  SQI: {sqi:.3f}")
    if issues:
        print(f"  Quality issues: {issues}")

    # Generate plot
    plot_pipeline(raw_sig, cleaned, r_peaks, delin, features, sqi, label, segment_id, fs, args.output)

    print("\n✓ Done")


if __name__ == "__main__":
    main()
