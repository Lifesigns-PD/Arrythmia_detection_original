#!/usr/bin/env python3
"""
ECG Pipeline Analysis Report
=============================
Standalone script — no database required.
Upload your ECG as a JSON file and get a full visual + mathematical report.

Compares two pipelines side-by-side:
  Pipeline A — Custom (Pan-Tompkins + slope-based QRS delineation)
  Pipeline B — NeuroKit2 (nk.ecg_clean + nk.ecg_peaks + nk.ecg_delineate DWT)

Usage:
    python ecg_pipeline_report.py sample_ecg.json
    python ecg_pipeline_report.py my_ecg.json --save

JSON format (single segment):
    {"signal": [0.1, 0.15, ...], "fs": 125, "label": "Sinus Rhythm"}

JSON format (multiple segments):
    [
        {"signal": [...], "fs": 125, "label": "Sinus Rhythm"},
        {"signal": [...], "fs": 125, "label": "Atrial Fibrillation"}
    ]
"""

import sys
import json
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.signal import butter, filtfilt, find_peaks, medfilt
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE A — Custom Implementation
# ═══════════════════════════════════════════════════════════════════════════════

def custom_preprocess(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    MATH: Two-stage filtering
      1. Baseline wander removal: median filter with window = 0.6s
         baseline(t) = median(x[t - W/2 : t + W/2])
         x_clean(t)  = x(t) - baseline(t)

      2. Bandpass filter: Butterworth 4th order [0.5 Hz – 40 Hz]
         Removes: DC drift (<0.5 Hz), powerline noise (50/60 Hz),
                  high-freq EMG (>40 Hz)
    """
    sig = np.asarray(signal, dtype=np.float64)
    # Baseline removal
    window = int(0.6 * fs) | 1   # ensure odd
    baseline = medfilt(sig, kernel_size=window)
    sig = sig - baseline
    # Bandpass
    b, a = butter(4, [0.5 / (fs / 2), 40.0 / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)


def custom_detect_rpeaks(cleaned: np.ndarray, fs: int) -> np.ndarray:
    """
    MATH: Simplified Pan-Tompkins (1985)
      1. Derivative:    y[n] = (−x[n−2] − 2x[n−1] + 2x[n+1] + x[n+2]) / 8
      2. Square:        y[n] = y[n]²
      3. Moving avg:    y[n] = (1/W) Σ y[n−W+1..n]   W = 0.15s
      4. Adaptive thr:  θ(n) = 0.5 × max(y[0..n])
      5. Peak detect:   refractory = 0.2s
    """
    # Derivative
    dy = np.zeros_like(cleaned)
    dy[2:-2] = (-cleaned[:-4] - 2*cleaned[1:-3] + 2*cleaned[3:-1] + cleaned[4:]) / 8.0
    # Square
    sq = dy ** 2
    # Moving average
    w = max(int(0.15 * fs), 3)
    ma = uniform_filter1d(sq, size=w)
    # Threshold + peak detection
    threshold = 0.5 * np.max(ma)
    min_dist  = int(0.2 * fs)
    peaks, _  = find_peaks(ma, height=threshold, distance=min_dist)
    # Snap to nearest signal max within ±10 samples
    refined = []
    for p in peaks:
        lo = max(0, p - 10)
        hi = min(len(cleaned), p + 10)
        refined.append(lo + int(np.argmax(cleaned[lo:hi])))
    return np.array(refined, dtype=int)


def custom_delineate_beat(signal: np.ndarray, r: int, fs: int) -> dict:
    """
    MATH: Slope-based J-point (QRS offset) + energy-ratio P-wave detection

    QRS Onset:
      Search backward from R for first sample where slope changes sign
      (negative → positive in inverted CWT sense).
      Fallback: R − 40ms

    QRS Offset (J-point):
      Search forward from R. Stop when:
        |x[n+1] − x[n]| < 0.025 mV/sample  (flat ST plateau)
        AND x[n] < 0.15 × R_amplitude        (descended from QRS peak)
        AND n > R + 40ms                      (minimum QRS duration)

    P-wave:
      Search window: [QRS_onset − 280ms, QRS_onset − 40ms]
      Energy ratio: E_p / E_baseline ≥ 2.5
        where E = Σ(x − mean)² / N  (variance, demeaned)
      P absent if ratio < 2.5 or peak amplitude < 0.04 mV
    """
    result = {k: None for k in [
        "qrs_onset","qrs_offset","p_onset","p_peak","p_offset","p_morphology",
        "t_peak","q_peak","s_peak"
    ]}
    n = len(signal)
    r_amp = abs(float(signal[r])) if 0 <= r < n else 0.1

    # ── QRS onset ──
    lo_on = max(0, r - int(0.08 * fs))
    for i in range(r, lo_on, -1):
        if i >= 1 and signal[i] < signal[i-1] and signal[i] < 0.5 * signal[r]:
            result["qrs_onset"] = i
            break
    if result["qrs_onset"] is None:
        result["qrs_onset"] = max(0, r - int(0.040 * fs))

    # ── QRS offset (J-point) ──
    hi_off = min(n, r + int(0.200 * fs))
    flat_thresh  = 0.025
    level_thresh = r_amp * 0.15
    min_wait = r + max(3, int(0.040 * fs))
    for i in range(min_wait, hi_off - 4):
        s = [abs(float(signal[i+k+1]) - float(signal[i+k])) for k in range(4)]
        if all(v < flat_thresh for v in s) and float(signal[i]) < level_thresh:
            result["qrs_offset"] = i
            break
    if result["qrs_offset"] is None:
        result["qrs_offset"] = min(n - 1, r + int(0.050 * fs))

    # ── Q and S peaks ──
    qrs_on  = result["qrs_onset"]
    qrs_off = result["qrs_offset"]
    if qrs_on is not None and qrs_on < r:
        q_seg = signal[qrs_on:r]
        if len(q_seg): result["q_peak"] = qrs_on + int(np.argmin(q_seg))
    if qrs_off is not None and r < qrs_off:
        s_seg = signal[r:qrs_off]
        if len(s_seg): result["s_peak"] = r + int(np.argmin(s_seg))

    # ── T-peak ──
    t_start = result["qrs_offset"] or (r + int(0.050 * fs))
    t_end   = min(n, r + int(0.450 * fs))
    if t_end > t_start + 4:
        t_seg = signal[t_start:t_end]
        result["t_peak"] = t_start + int(np.argmax(np.abs(t_seg)))

    # ── P-wave (energy ratio) ──
    if qrs_on is not None:
        hi_p  = max(0, qrs_on - int(0.040 * fs))
        lo_p  = max(0, qrs_on - int(0.280 * fs))
        if hi_p > lo_p + 4:
            p_region = signal[lo_p:hi_p]
            tp_start = max(0, lo_p - int(0.06 * fs))
            baseline = signal[tp_start:lo_p]
            bw = baseline - np.mean(baseline) if len(baseline) >= 3 else baseline
            pr = p_region - np.mean(p_region)
            e_base = np.sum(bw**2) / max(len(bw), 1)
            e_p    = np.sum(pr**2) / max(len(pr), 1)
            p_pk_local = int(np.argmax(np.abs(p_region)))
            p_peak_idx = lo_p + p_pk_local
            p_amp = abs(float(signal[p_peak_idx]))

            if e_p >= 2.5 * e_base and p_amp >= 0.04:
                result["p_peak"]   = p_peak_idx
                result["p_onset"]  = max(0, p_peak_idx - int(0.040 * fs))
                result["p_offset"] = min(n-1, p_peak_idx + int(0.040 * fs))
                p_raw = float(signal[p_peak_idx])
                result["p_morphology"] = "inverted" if p_raw < -0.04 else "normal"
            else:
                result["p_morphology"] = "absent"
        else:
            result["p_morphology"] = "absent"

    return result


def custom_pipeline(signal: np.ndarray, fs: int) -> dict:
    cleaned = custom_preprocess(signal, fs)
    r_peaks = custom_detect_rpeaks(cleaned, fs)
    per_beat = [custom_delineate_beat(cleaned, int(r), fs) for r in r_peaks]
    features = compute_features(cleaned, r_peaks, per_beat, fs)
    return {"cleaned": cleaned, "r_peaks": r_peaks, "per_beat": per_beat, "features": features}


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE B — NeuroKit2
# ═══════════════════════════════════════════════════════════════════════════════

def nk2_pipeline(signal: np.ndarray, fs: int) -> dict:
    try:
        import neurokit2 as nk
    except ImportError:
        print("  [NK2 not installed: pip install neurokit2]")
        return {"cleaned": signal.copy(), "r_peaks": np.array([]), "per_beat": [], "features": {}}

    cleaned = nk.ecg_clean(signal.astype(np.float64), sampling_rate=fs, method="neurokit")
    cleaned = np.asarray(cleaned, dtype=np.float64)

    try:
        _, r_info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit")
        r_peaks   = np.array(r_info["ECG_R_Peaks"], dtype=int)
    except Exception:
        r_peaks = np.array([], dtype=int)

    per_beat = []
    if len(r_peaks) >= 2:
        try:
            _, waves = nk.ecg_delineate(cleaned, r_peaks, sampling_rate=fs,
                                         method="dwt", show=False)

            def _idx(key, i):
                a = waves.get(key, [])
                if i < len(a):
                    v = a[i]
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        idx = int(v)
                        return idx if 0 <= idx < len(cleaned) else None
                return None

            for i in range(len(r_peaks)):
                p_peak = _idx("ECG_P_Peaks",   i)
                if p_peak is not None:
                    p_amp = abs(float(cleaned[p_peak]))
                    morph = "absent" if p_amp < 0.04 else ("inverted" if float(cleaned[p_peak]) < -0.04 else "normal")
                    if morph == "absent": p_peak = None
                else:
                    morph = "absent"

                per_beat.append({
                    "qrs_onset":    _idx("ECG_R_Onsets",  i),
                    "qrs_offset":   _idx("ECG_R_Offsets", i),
                    "p_peak":       p_peak,
                    "p_onset":      _idx("ECG_P_Onsets",  i) if p_peak else None,
                    "p_offset":     _idx("ECG_P_Offsets", i) if p_peak else None,
                    "p_morphology": morph,
                    "q_peak":       _idx("ECG_Q_Peaks",   i),
                    "s_peak":       _idx("ECG_S_Peaks",   i),
                    "t_peak":       _idx("ECG_T_Peaks",   i),
                })
        except Exception:
            per_beat = []

    features = compute_features(cleaned, r_peaks, per_beat, fs)
    return {"cleaned": cleaned, "r_peaks": r_peaks, "per_beat": per_beat, "features": features}


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_features(signal: np.ndarray, r_peaks: np.ndarray, per_beat: list, fs: int) -> dict:
    """
    MATH — Key features:

    HR   = 60 × fs / mean(RR)                 [bpm]
    SDNN = std(RR)                             [ms]  — overall HRV
    RMSSD= sqrt(mean(ΔRR²))                   [ms]  — short-term HRV
    pNN50= count(|ΔRR| > 50ms) / N            [0-1]

    QRS duration = (QRS_offset − QRS_onset) × 1000/fs   [ms]
    PR interval  = (QRS_onset  − P_onset)   × 1000/fs   [ms]

    p_absent_fraction = count(p_morphology=='absent') / N_beats

    LF/HF ratio: Power spectral density of RR series
      LF band: 0.04–0.15 Hz  (sympathetic + parasympathetic)
      HF band: 0.15–0.40 Hz  (parasympathetic only)
    """
    f = {}

    if len(r_peaks) >= 2:
        rr    = np.diff(r_peaks).astype(float) / fs * 1000
        rr_v  = rr[(rr > 250) & (rr < 2500)]
        f["mean_hr_bpm"] = float(60000 / np.mean(rr_v)) if len(rr_v) else None
        f["sdnn_ms"]     = float(np.std(rr_v, ddof=1))  if len(rr_v) >= 2 else None
        drr  = np.diff(rr_v)
        f["rmssd_ms"] = float(np.sqrt(np.mean(drr**2))) if len(drr) >= 1 else None
        f["pnn50"]    = float(np.mean(np.abs(drr) > 50)) if len(drr) >= 1 else None
    else:
        f.update({"mean_hr_bpm": None, "sdnn_ms": None, "rmssd_ms": None, "pnn50": None})

    qrs_durs, pr_ms, p_absent = [], [], []
    for b in per_beat:
        on, off = b.get("qrs_onset"), b.get("qrs_offset")
        if on is not None and off is not None and off > on:
            d = (off - on) * 1000 / fs
            if 40 <= d <= 300: qrs_durs.append(d)
        pon = b.get("p_onset")
        if pon is not None and on is not None and on > pon:
            p = (on - pon) * 1000 / fs
            if 60 <= p <= 400: pr_ms.append(p)
        p_absent.append(1.0 if b.get("p_morphology") == "absent" else 0.0)

    f["mean_qrs_duration_ms"] = float(np.median(qrs_durs)) if qrs_durs else None
    f["pr_interval_ms"]       = float(np.median(pr_ms))    if pr_ms   else -1.0
    f["p_absent_fraction"]    = float(np.mean(p_absent))   if p_absent else None

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

COLOURS = {
    "raw":    "#555555",
    "custom": "#1a6faf",
    "nk2":    "#e07b39",
    "r_cust": "#e74c3c",
    "r_nk2":  "#f39c12",
    "p":      "#e67e22",
    "qrs":    "#e74c3c",
    "t":      "#3498db",
}

CLINICAL_NORMS = {
    "HR":       (60,  100, "bpm"),
    "PR":       (120, 200, "ms"),
    "QRS":      (60,  120, "ms"),
    "SDNN":     (20,  200, "ms"),
    "RMSSD":    (15,  100, "ms"),
    "p_absent": (0,   0.1, ""),
}


def _norm_tag(name, val):
    if val is None: return "N/A", "#888888"
    lo, hi, _ = CLINICAL_NORMS.get(name, (None, None, ""))
    if lo is None: return f"{val:.2f}", "#333333"
    ok = lo <= val <= hi
    return f"{val:.1f}", ("#27ae60" if ok else "#e74c3c")


def generate_report(segment: dict, save: bool = False, seg_idx: int = 0):
    raw_sig = np.array(segment["signal"], dtype=np.float64)
    fs      = int(segment.get("fs", 125))
    label   = segment.get("label", "Unknown")
    t       = np.arange(len(raw_sig)) / fs

    print(f"\n{'='*65}")
    print(f"  Segment {seg_idx+1} — {label}   ({len(raw_sig)} samples @ {fs} Hz = {len(raw_sig)/fs:.1f}s)")
    print(f"{'='*65}")

    print("  Running Custom pipeline...", end=" ", flush=True)
    cust = custom_pipeline(raw_sig, fs)
    print("done")
    print("  Running NeuroKit2 pipeline...", end=" ", flush=True)
    nk2  = nk2_pipeline(raw_sig, fs)
    print("done")

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 22), facecolor="#f8f9fa")
    fig.suptitle(
        f"ECG Pipeline Analysis  |  Segment {seg_idx+1}: {label}  |  {fs} Hz",
        fontsize=15, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(5, 2, figure=fig,
                           hspace=0.45, wspace=0.3,
                           left=0.06, right=0.97, top=0.95, bottom=0.04)

    # ── Row 0: Raw signal ────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t, raw_sig, color=COLOURS["raw"], lw=0.8, label="Raw ECG")
    ax0.set_title("① Raw Signal", fontweight="bold")
    ax0.set_ylabel("Amplitude (mV)")
    ax0.legend(fontsize=8, loc="upper right")
    ax0.grid(True, alpha=0.3)
    _add_math(ax0,
        r"$x_{raw}(t)$ — unprocessed ECG from electrode",
        0.01, 0.92)

    # ── Row 1: Preprocessing comparison ─────────────────────────────────────
    ax1a = fig.add_subplot(gs[1, 0])
    ax1a.plot(t, cust["cleaned"], color=COLOURS["custom"], lw=0.9)
    ax1a.set_title("② Custom Preprocessing", fontweight="bold", color=COLOURS["custom"])
    ax1a.set_ylabel("Amplitude (mV)")
    ax1a.grid(True, alpha=0.3)
    _add_math(ax1a,
        r"$x_1 = x - median_{0.6s}(x)$" + "\n" + r"$x_2 = Butterworth_{[0.5,40]Hz}(x_1)$",
        0.01, 0.88)

    ax1b = fig.add_subplot(gs[1, 1])
    ax1b.plot(t, nk2["cleaned"], color=COLOURS["nk2"], lw=0.9)
    ax1b.set_title("② NeuroKit2 Preprocessing", fontweight="bold", color=COLOURS["nk2"])
    ax1b.set_ylabel("Amplitude (mV)")
    ax1b.grid(True, alpha=0.3)
    _add_math(ax1b,
        r"$nk.ecg\_clean()$ — adaptive baseline + powerline filter",
        0.01, 0.92)

    # ── Row 2: R-peak detection comparison ───────────────────────────────────
    ax2a = fig.add_subplot(gs[2, 0])
    ax2a.plot(t, cust["cleaned"], color=COLOURS["custom"], lw=0.8, alpha=0.7)
    _plot_rpeaks(ax2a, cust["r_peaks"], cust["cleaned"], t, COLOURS["r_cust"])
    ax2a.set_title(f"③ Custom R-peaks  ({len(cust['r_peaks'])} detected)",
                   fontweight="bold", color=COLOURS["custom"])
    ax2a.set_ylabel("Amplitude (mV)")
    ax2a.grid(True, alpha=0.3)
    _add_math(ax2a,
        r"Pan-Tompkins: $y[n] = (\Delta x)^2 \rightarrow MA \rightarrow \theta = 0.5\cdot\max$",
        0.01, 0.92)

    ax2b = fig.add_subplot(gs[2, 1])
    ax2b.plot(t, nk2["cleaned"], color=COLOURS["nk2"], lw=0.8, alpha=0.7)
    _plot_rpeaks(ax2b, nk2["r_peaks"], nk2["cleaned"], t, COLOURS["r_nk2"])
    ax2b.set_title(f"③ NK2 R-peaks  ({len(nk2['r_peaks'])} detected)",
                   fontweight="bold", color=COLOURS["nk2"])
    ax2b.set_ylabel("Amplitude (mV)")
    ax2b.grid(True, alpha=0.3)
    _add_math(ax2b,
        r"$nk.ecg\_peaks()$ — NeuroKit2 Pan-Tompkins variant",
        0.01, 0.92)

    # ── Row 3: Delineation comparison ────────────────────────────────────────
    ax3a = fig.add_subplot(gs[3, 0])
    ax3a.plot(t, cust["cleaned"], color=COLOURS["custom"], lw=0.8, alpha=0.7)
    _plot_delineation(ax3a, cust["cleaned"], cust["r_peaks"], cust["per_beat"], t, fs)
    ax3a.set_title("④ Custom Delineation (P/Q/R/S/T)", fontweight="bold", color=COLOURS["custom"])
    ax3a.set_ylabel("Amplitude (mV)")
    ax3a.set_xlabel("Time (s)")
    ax3a.grid(True, alpha=0.3)
    _add_math(ax3a,
        r"QRS offset: $|x[n+1]-x[n]| < 0.025$ mV/sample  +  $x[n] < 0.15 \cdot R_{amp}$",
        0.01, 0.92)

    ax3b = fig.add_subplot(gs[3, 1])
    ax3b.plot(t, nk2["cleaned"], color=COLOURS["nk2"], lw=0.8, alpha=0.7)
    _plot_delineation(ax3b, nk2["cleaned"], nk2["r_peaks"], nk2["per_beat"], t, fs)
    ax3b.set_title("④ NK2 Delineation (P/Q/R/S/T)", fontweight="bold", color=COLOURS["nk2"])
    ax3b.set_ylabel("Amplitude (mV)")
    ax3b.set_xlabel("Time (s)")
    ax3b.grid(True, alpha=0.3)
    _add_math(ax3b,
        r"$nk.ecg\_delineate()$ DWT — discrete wavelet transform boundaries",
        0.01, 0.92)

    # ── Row 4: Feature comparison table ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[4, :])
    ax4.axis("off")
    _draw_feature_table(ax4, cust["features"], nk2["features"], label)

    if save:
        fname = f"ecg_report_seg{seg_idx+1}_{label.replace(' ','_')}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fname}")

    # ── Console summary ───────────────────────────────────────────────────────
    cf, nf = cust["features"], nk2["features"]
    print(f"\n  {'Feature':<25} {'Custom':>12} {'NK2':>12}  {'Clinical Norm'}")
    print(f"  {'-'*65}")
    rows = [
        ("HR (bpm)",       "HR",       cf.get("mean_hr_bpm"),   nf.get("mean_hr_bpm")),
        ("PR (ms)",        "PR",       cf.get("pr_interval_ms"), nf.get("pr_interval_ms")),
        ("QRS (ms)",       "QRS",      cf.get("mean_qrs_duration_ms"), nf.get("mean_qrs_duration_ms")),
        ("SDNN (ms)",      "SDNN",     cf.get("sdnn_ms"),        nf.get("sdnn_ms")),
        ("RMSSD (ms)",     "RMSSD",    cf.get("rmssd_ms"),       nf.get("rmssd_ms")),
        ("p_absent",       "p_absent", cf.get("p_absent_fraction"), nf.get("p_absent_fraction")),
    ]
    for name, norm_key, cv, nv in rows:
        lo, hi, unit = CLINICAL_NORMS.get(norm_key, (None, None, ""))
        c_str = f"{cv:.1f}" if cv is not None else "N/A"
        n_str = f"{nv:.1f}" if nv is not None else "N/A"
        norm  = f"{lo}–{hi} {unit}" if lo is not None else ""
        agree = ""
        if cv is not None and nv is not None:
            agree = "✓" if abs(cv - nv) < max(15, 0.15 * abs(cv)) else "⚠ differ"
        print(f"  {name:<25} {c_str:>12} {n_str:>12}  {norm:<18} {agree}")

    print(f"\n  R-peaks: Custom={len(cust['r_peaks'])}  NK2={len(nk2['r_peaks'])}", end="")
    if len(cust["r_peaks"]) and len(nk2["r_peaks"]):
        diff = abs(len(cust["r_peaks"]) - len(nk2["r_peaks"]))
        print(f"  {'✓ agree' if diff == 0 else f'⚠ differ by {diff}'}", end="")
    print()

    return fig


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _add_math(ax, text, x=0.01, y=0.95):
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=7.5, color="#444444",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fffff0",
                      edgecolor="#cccccc", alpha=0.85))


def _plot_rpeaks(ax, r_peaks, signal, t, colour):
    if len(r_peaks):
        ax.plot(r_peaks / len(t) * t[-1], signal[r_peaks],
                "v", color=colour, markersize=9, zorder=5,
                label=f"R-peaks ({len(r_peaks)})")
        for r in r_peaks:
            ax.axvline(r / len(t) * t[-1], color=colour, alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, loc="upper right")


def _plot_delineation(ax, signal, r_peaks, per_beat, t, fs):
    n = len(signal)
    def _tx(idx): return idx / fs

    colours_map = {
        "p_peak":    ("#e67e22", "P", "^", 8),
        "q_peak":    ("#8e44ad", "Q", "v", 6),
        "r":         ("#e74c3c", "R", "v", 9),
        "s_peak":    ("#27ae60", "S", "^", 6),
        "t_peak":    ("#3498db", "T", "^", 8),
    }
    plotted = {k: False for k in colours_map}

    for i, beat in enumerate(per_beat):
        r = int(r_peaks[i]) if i < len(r_peaks) else None
        if r is not None and 0 <= r < n:
            c, ltr, mk, ms = colours_map["r"]
            lbl = "R" if not plotted["r"] else "_"
            ax.plot(_tx(r), signal[r], marker=mk, color=c, markersize=ms,
                    label=lbl if lbl != "_" else None)
            plotted["r"] = True

        for key in ["p_peak", "q_peak", "s_peak", "t_peak"]:
            idx = beat.get(key)
            if idx is not None and 0 <= idx < n:
                c, ltr, mk, ms = colours_map[key]
                lbl = ltr if not plotted[key] else "_"
                ax.plot(_tx(idx), signal[idx], marker=mk, color=c, markersize=ms,
                        label=lbl if lbl != "_" else None)
                plotted[key] = True

        qon  = beat.get("qrs_onset")
        qoff = beat.get("qrs_offset")
        if qon is not None and qoff is not None and qoff > qon:
            ax.axvspan(_tx(qon), _tx(qoff), alpha=0.12, color="#e74c3c")

        pon  = beat.get("p_onset")
        poff = beat.get("p_offset")
        if pon is not None and poff is not None and poff > pon:
            ax.axvspan(_tx(pon), _tx(poff), alpha=0.12, color="#e67e22")

    handles = [h for h in ax.get_legend_handles_labels()[0]]
    labels  = [l for l in ax.get_legend_handles_labels()[1]]
    if handles:
        ax.legend(handles, labels, fontsize=7, loc="upper right", ncol=3)


def _draw_feature_table(ax, cf, nf, label):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.97, f"⑤ Feature Comparison — {label}",
            ha="center", va="top", fontsize=12, fontweight="bold",
            transform=ax.transAxes)

    headers = ["Feature", "Custom Value", "NK2 Value", "Clinical Normal", "Status"]
    col_x   = [0.0, 0.22, 0.38, 0.54, 0.72]
    col_w   = [0.20, 0.15, 0.15, 0.20, 0.25]

    for j, h in enumerate(headers):
        ax.text(col_x[j] + col_w[j]/2, 0.88, h,
                ha="center", va="center", fontsize=9, fontweight="bold",
                transform=ax.transAxes,
                color="white",
                bbox=dict(boxstyle="square,pad=0.3", facecolor="#2c3e50", edgecolor="none"))

    rows = [
        ("HR (bpm)",    "HR",       "mean_hr_bpm",           60,   100,  "bpm"),
        ("PR interval", "PR",       "pr_interval_ms",        120,  200,  "ms"),
        ("QRS duration","QRS",      "mean_qrs_duration_ms",  60,   120,  "ms"),
        ("SDNN",        "SDNN",     "sdnn_ms",               20,   200,  "ms"),
        ("RMSSD",       "RMSSD",    "rmssd_ms",              15,   100,  "ms"),
        ("pNN50",       None,       "pnn50",                 0.03, 0.50, ""),
        ("P absent",    "p_absent", "p_absent_fraction",     0,    0.1,  ""),
    ]

    for row_i, (name, norm_key, feat_key, lo, hi, unit) in enumerate(rows):
        y = 0.78 - row_i * 0.105
        bg = "#f0f4f8" if row_i % 2 == 0 else "#ffffff"
        ax.add_patch(FancyBboxPatch((0, y - 0.04), 1.0, 0.09,
                                    boxstyle="square,pad=0",
                                    facecolor=bg, edgecolor="none",
                                    transform=ax.transAxes, zorder=0))

        cv = cf.get(feat_key)
        nv = nf.get(feat_key)
        c_str = f"{cv:.1f} {unit}" if cv is not None else "N/A"
        n_str = f"{nv:.1f} {unit}" if nv is not None else "N/A"
        norm_str = f"{lo}–{hi} {unit}"

        # Status
        c_ok = (cv is not None and lo <= cv <= hi) if lo is not None else True
        n_ok = (nv is not None and lo <= nv <= hi) if lo is not None else True
        agree = (cv is not None and nv is not None and
                 abs(cv - nv) < max(15, 0.15 * abs(cv or 0)))

        if c_ok and n_ok and agree:
            status, status_col = "Both correct ✓", "#27ae60"
        elif not c_ok and not n_ok:
            status, status_col = "Both outside range", "#e74c3c"
        elif c_ok and not n_ok:
            status, status_col = "Custom ✓  NK2 ✗", "#e07b39"
        elif not c_ok and n_ok:
            status, status_col = "Custom ✗  NK2 ✓", "#3498db"
        elif not agree:
            status, status_col = "Values differ ⚠", "#e07b39"
        else:
            status, status_col = "OK", "#27ae60"

        vals = [name, c_str, n_str, norm_str, status]
        clrs = ["#333333", COLOURS["custom"], COLOURS["nk2"], "#555555", status_col]
        fws  = ["normal", "bold", "bold", "normal", "bold"]
        for j, (txt, clr, fw) in enumerate(zip(vals, clrs, fws)):
            ax.text(col_x[j] + col_w[j]/2, y + 0.01, txt,
                    ha="center", va="center", fontsize=8.5,
                    color=clr, fontweight=fw, transform=ax.transAxes)

    ax.axhline(0.83, color="#cccccc", lw=0.8)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ECG Pipeline Analysis Report")
    parser.add_argument("json_file", help="Path to JSON file with ECG segment(s)")
    parser.add_argument("--save", action="store_true", help="Save plots as PNG instead of showing")
    parser.add_argument("--max-windows", type=int, default=3,
                        help="Max 10s windows to analyse from a packet-array file (default: 3, 0=all)")
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    # ── Detect packet-array format (ECG_Data_Extracts style) ────────────────
    # Each file is a list of dicts with keys: utcTimestamp, admissionId, value, packetNo
    # value is [[sample, sample, ...]] — one sub-list of raw ECG samples per packet
    def _is_packet_array(d) -> bool:
        return (isinstance(d, list) and len(d) > 0
                and isinstance(d[0], dict)
                and "packetNo" in d[0] and "value" in d[0])

    def _packets_to_segments(packets, fs: int = 125, window_s: int = 10) -> list:
        """Assemble packets into one continuous signal, split into 10s windows."""
        admission_id = packets[0].get("admissionId", "Unknown")
        packets = sorted(packets, key=lambda p: p.get("packetNo", 0))
        full_signal = []
        for pkt in packets:
            v = pkt.get("value", [])
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                full_signal.extend(v[0])   # value = [[...samples...]]
            elif isinstance(v, list):
                full_signal.extend(v)       # value = [...samples...]
        full_signal = np.array(full_signal, dtype=np.float64)
        window_len = fs * window_s
        total_windows = len(full_signal) // window_len
        n_windows = total_windows if args.max_windows == 0 else min(total_windows, args.max_windows)
        print(f"Packet file: {len(packets)} packets → {len(full_signal)} samples "
              f"({len(full_signal)/fs:.1f}s) → showing {n_windows}/{total_windows} x {window_s}s windows")
        segs = []
        for w in range(n_windows):
            chunk = full_signal[w * window_len: (w + 1) * window_len]
            segs.append({
                "signal": chunk.tolist(),
                "fs": fs,
                "label": f"Adm {admission_id} | Window {w+1}/{total_windows}",
            })
        return segs

    if _is_packet_array(data):
        segments = _packets_to_segments(data)
    elif isinstance(data, dict):
        segments = [data]
    elif isinstance(data, list):
        segments = data
    else:
        print("Error: JSON must be a dict or list of dicts")
        sys.exit(1)

    # ── Normalise: support "signal" (internal) and "ecgData" (device JSON) ──
    def _normalise_segment(seg: dict) -> dict:
        seg = dict(seg)  # shallow copy — don't mutate original
        if "signal" not in seg and "ecgData" in seg:
            seg["signal"] = seg["ecgData"]
        if "label" not in seg:
            label_parts = []
            if seg.get("patientId"):
                label_parts.append(f"Patient {seg['patientId']}")
            if seg.get("admissionId"):
                label_parts.append(f"Adm {seg['admissionId']}")
            seg["label"] = " | ".join(label_parts) if label_parts else "Unknown"
        return seg

    segments = [_normalise_segment(s) for s in segments]

    print(f"\nECG Pipeline Analysis Report")
    print(f"Loaded {len(segments)} segment(s) from {args.json_file}")

    figs = []
    for i, seg in enumerate(segments):
        fig = generate_report(seg, save=args.save, seg_idx=i)
        figs.append(fig)

    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()
