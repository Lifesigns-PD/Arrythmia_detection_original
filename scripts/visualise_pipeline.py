#!/usr/bin/env python3
"""
visualise_pipeline.py — End-to-end V3 signal processing visualiser
====================================================================
Fetches one ECG segment from DB, runs the full V3 pipeline, and plots:
  Row 1 — Raw signal
  Row 2 — Preprocessed (baseline removed + denoised)
  Row 3 — Final: R-peaks + P/Q/S/T fiducials marked

Usage:
    python scripts/visualise_pipeline.py                         # random Sinus Rhythm
    python scripts/visualise_pipeline.py --seg 1127             # specific segment
    python scripts/visualise_pipeline.py --label "Atrial Fibrillation"
    python scripts/visualise_pipeline.py --label "Bundle Branch Block"
"""

import sys, json, argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import psycopg2

from signal_processing_v3.preprocessing.pipeline  import preprocess_v3
from signal_processing_v3.detection.ensemble      import detect_r_peaks_ensemble
from signal_processing_v3.delineation.hybrid       import delineate_v3
from signal_processing_v3.features.extraction      import extract_features_v3, FEATURE_NAMES_V3

DB = dict(host="127.0.0.1", dbname="ecg_analysis", user="ecg_user", password="sais", port="5432")


# ── DB fetch ──────────────────────────────────────────────────────────────────
def fetch_segment(seg_id=None, label="Sinus Rhythm"):
    conn = psycopg2.connect(**DB)
    cur  = conn.cursor()
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
    return sid, sig, int(fs or 125), lbl


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg",   type=int,   default=None)
    parser.add_argument("--label", type=str,   default="Sinus Rhythm")
    args = parser.parse_args()

    print("Fetching segment...")
    seg_id, raw_sig, fs, label = fetch_segment(args.seg, args.label)
    print(f"  seg_id={seg_id}  label={label}  fs={fs}  samples={len(raw_sig)}")

    t = np.arange(len(raw_sig)) / fs

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    print("Preprocessing...")
    prep     = preprocess_v3(raw_sig, fs=fs)
    cleaned  = prep["cleaned"]
    sqi      = prep.get("quality_score", 1.0)
    issues   = prep.get("quality_issues", [])

    # ── Step 2: Detect R-peaks ────────────────────────────────────────────────
    print("Detecting R-peaks...")
    r_peaks = detect_r_peaks_ensemble(cleaned, fs=fs)
    print(f"  {len(r_peaks)} R-peaks detected")

    # ── Step 3: Delineate ─────────────────────────────────────────────────────
    print("Delineating P/Q/R/S/T...")
    delin   = delineate_v3(cleaned, r_peaks, fs=fs)
    per_beat = delin["per_beat"]
    summary  = delin["summary"]

    # ── Step 4: Extract features ──────────────────────────────────────────────
    print("Extracting features...")
    from signal_processing_v3.detection.ensemble import refine_peaks_subsample
    r_float  = refine_peaks_subsample(cleaned, r_peaks)
    features = extract_features_v3(cleaned, r_float, delin, fs=fs)

    # ── NeuroKit2 vitals (independent cross-check) ───────────────────────────
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
                on  = _nk_idx("ECG_R_Onsets",  _i)
                off = _nk_idx("ECG_R_Offsets", _i)
                pon = _nk_idx("ECG_P_Onsets",  _i)
                if on is not None and off is not None and off > on:
                    d = (off - on) * 1000 / fs
                    if 40 <= d <= 300: qrs_list.append(d)
                if pon is not None and on is not None and on > pon:
                    p = (on - pon) * 1000 / fs
                    if 60 <= p <= 400: pr_list.append(p)
            if qrs_list: nk_qrs = float(np.median(qrs_list))
            if pr_list:  nk_pr  = float(np.median(pr_list))
    except Exception as _e:
        print(f"  [NK2 check skipped: {_e}]")

    # ── Print key features ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  SEGMENT {seg_id} — {label}")
    print("="*60)
    print(f"\n  {'VITAL':<10} {'V3 Pipeline':>14} {'NeuroKit2':>12}  {'Match?'}")
    print(f"  {'-'*50}")
    hr  = features.get("mean_hr_bpm") or 0
    pr  = features.get("pr_interval_ms") or 0
    qrs = features.get("mean_qrs_duration_ms") or 0
    def _chk(a, b, tol=15): return "✓" if abs((a or 0) - (b or 0)) < tol else "⚠"
    print(f"  {'HR (bpm)':<10} {hr:>14.1f} {nk_hr:>12.1f}  {_chk(hr, nk_hr, 10)}")
    print(f"  {'PR (ms)':<10} {pr:>14.0f} {nk_pr:>12.0f}  {_chk(pr, nk_pr, 30)}")
    print(f"  {'QRS (ms)':<10} {qrs:>14.0f} {nk_qrs:>12.0f}  {_chk(qrs, nk_qrs, 20)}")

    print("\n  ALL FEATURES:")
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

    # ── Plot ──────────────────────────────────────────────────────────────────
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

    # Colour scheme
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

        # R
        if r is not None and 0 <= r < len(cleaned):
            lbl_str = "R" if not plotted["r"] else ""
            ax.plot(r/fs, cleaned[r], marker=colours["r"][2],
                    color=colours["r"][0], markersize=colours["r"][3],
                    label=lbl_str if lbl_str else "_")
            plotted["r"] = True

        # P Q S T
        for key in ["p_peak", "q_peak", "s_peak", "t_peak"]:
            idx = beat.get(key)
            if idx is not None and 0 <= idx < len(cleaned):
                c, letter, mk, ms = colours[key]
                lbl_str = letter if not plotted[key] else ""
                ax.plot(idx/fs, cleaned[idx], marker=mk,
                        color=c, markersize=ms,
                        label=lbl_str if lbl_str else "_")
                plotted[key] = True

        # Shade P-wave region
        p_on  = beat.get("p_onset")
        p_off = beat.get("p_offset")
        if p_on is not None and p_off is not None and p_off > p_on:
            ax.axvspan(p_on/fs, p_off/fs, alpha=0.12, color="#e67e22")

        # Shade QRS region
        q_on  = beat.get("qrs_onset")
        q_off = beat.get("qrs_offset")
        if q_on is not None and q_off is not None and q_off > q_on:
            ax.axvspan(q_on/fs, q_off/fs, alpha=0.15, color="#e74c3c")

        # Shade T-wave region
        t_on  = beat.get("t_onset")
        t_off = beat.get("t_offset")
        if t_on is not None and t_off is not None and t_off > t_on:
            ax.axvspan(t_on/fs, t_off/fs, alpha=0.10, color="#3498db")

    # Legend patches for shaded regions
    patches = [
        mpatches.Patch(color="#e67e22", alpha=0.3, label="P-wave region"),
        mpatches.Patch(color="#e74c3c", alpha=0.3, label="QRS region"),
        mpatches.Patch(color="#3498db", alpha=0.3, label="T-wave region"),
    ]
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles=handles + patches, loc="upper right", fontsize=8, ncol=2)
    ax.set_ylabel("Amplitude (mV)")
    ax.set_xlabel("Time (s)")
    ax.set_title(
        f"3. Full Delineation — P/Q/R/S/T  |  "
        f"PR: {features.get('pr_interval_ms', 0) or 0:.0f}ms  "
        f"QRS: {features.get('mean_qrs_duration_ms', 0) or 0:.0f}ms  "
        f"p_absent: {features.get('p_absent_fraction', 0) or 0:.2f}",
        fontweight="bold"
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
