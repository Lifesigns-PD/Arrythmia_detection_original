#!/usr/bin/env python3
"""
ecg_diagnostic_viewer.py — ECG paper-style diagnostic viewer (signal processing only, no ML).

Layout: all 10-second segments stacked vertically on a single page.
Background: standard ECG graph paper (light pink / cream, red grid).

Annotations per strip:
  - Cleaned ECG waveform (black)
  - R-peaks (red triangles)
  - P-wave peak (green dot), P onset/offset (green tick)
  - Q-point (blue down-arrow), S-point (orange down-arrow)
  - T-peak (purple diamond), T-offset (purple tick)
  - Arrhythmia label (signal processing only) on the left margin
  - HR / QRS / PR / SQI inline on each strip

Usage:
    python scripts/ecg_diagnostic_viewer.py --file ECG_Data_Extracts/ADM441825561.json
    python scripts/ecg_diagnostic_viewer.py --file data/converted_ecg/MITDB__100_seg_0000.json --segment 0
    python scripts/ecg_diagnostic_viewer.py --file ECG_Data_Extracts/ADM441825561.json --max_segs 6
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

FS          = 125          # working sample rate
WIN_SAMPLES = 1250         # 10 s × 125 Hz


# ── Colour palette (ECG paper) ────────────────────────────────────────────────
PAPER_BG    = "#fff8f0"    # warm cream
GRID_MAJOR  = "#f4a0a0"    # standard red ECG major grid
GRID_MINOR  = "#fad4d4"    # lighter minor grid
ECG_LINE    = "#000000"    # black trace
ANNOT_BOX   = "#ffffffd0"  # semi-transparent white for text boxes

LABEL_COLORS = {
    "Ventricular Tachycardia": "#cc0000",
    "Ventricular Fibrillation": "#cc0000",
    "SVT": "#cc6600",
    None: "#006600",
}


# ── Signal loading ────────────────────────────────────────────────────────────

def load_signal(path: str) -> tuple[np.ndarray, int]:
    with open(path) as f:
        data = json.load(f)

    # ADM device packet list: [{"value": [[...200 samples...]], "packetNo": N}, ...]
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "value" in data[0]:
        packets = sorted(data, key=lambda p: p.get("packetNo", 0))
        samples = []
        for pkt in packets:
            v = pkt["value"]
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                samples.extend(v[0])
            elif isinstance(v, list):
                samples.extend(v)
        return np.asarray(samples, dtype=np.float32), 500

    # Single-segment dict: {"ECG_CH_A": [...], "fs": 250}
    if isinstance(data, dict):
        fs = int(data.get("fs") or data.get("sampling_rate") or FS)
        for key in ("ECG_CH_A", "ecg_data", "signal", "ECG", "ecg", "data"):
            if key in data:
                return np.asarray(data[key], dtype=np.float32), fs
        raise ValueError(f"Cannot find ECG signal. Keys: {list(data.keys())}")

    # Root list of floats
    if isinstance(data, list):
        return np.asarray(data, dtype=np.float32), FS

    raise ValueError("Unrecognised JSON format")


def resample_to_125(sig: np.ndarray, orig_fs: int) -> np.ndarray:
    if orig_fs == FS:
        return sig
    from scipy.signal import resample as sp_resample
    return sp_resample(sig, int(len(sig) * FS / orig_fs)).astype(np.float32)


def get_segment(sig: np.ndarray, idx: int) -> np.ndarray:
    chunk = sig[idx * WIN_SAMPLES: (idx + 1) * WIN_SAMPLES]
    if len(chunk) < WIN_SAMPLES:
        chunk = np.pad(chunk, (0, WIN_SAMPLES - len(chunk)))
    return chunk


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(window: np.ndarray) -> dict:
    from signal_processing_v3 import process_ecg_v3
    return process_ecg_v3(window, fs=FS, min_quality=0.0)


def run_arrhythmia(v3: dict) -> tuple[str | None, float, str]:
    from decision_engine.lethal_detector import detect_signal_rhythm
    return detect_signal_rhythm(v3["signal"], v3["r_peaks"], v3["features"], fs=FS)


# ── ECG paper grid helper ─────────────────────────────────────────────────────

def _draw_ecg_paper(ax, x_min: float, x_max: float, y_min: float, y_max: float):
    """
    Draw standard ECG paper grid on ax.
    Standard: 1 small square = 0.04 s (5 samples at 125 Hz) × 0.1 mV
              1 large square = 0.20 s × 0.5 mV  (5 small squares)
    """
    small_t  = 0.04    # seconds per small square
    small_mv = 0.1     # mV per small square
    large_t  = 0.20
    large_mv = 0.5

    # Minor grid (small squares)
    xt = np.arange(x_min, x_max + small_t, small_t)
    for x in xt:
        ax.axvline(x, color=GRID_MINOR, linewidth=0.4, zorder=0)
    yt = np.arange(y_min, y_max + small_mv, small_mv)
    for y in yt:
        ax.axhline(y, color=GRID_MINOR, linewidth=0.4, zorder=0)

    # Major grid (large squares)
    xt5 = np.arange(x_min, x_max + large_t, large_t)
    for x in xt5:
        ax.axvline(x, color=GRID_MAJOR, linewidth=0.8, zorder=0)
    yt5 = np.arange(y_min, y_max + large_mv, large_mv)
    for y in yt5:
        ax.axhline(y, color=GRID_MAJOR, linewidth=0.8, zorder=0)


# ── Single-strip drawing ──────────────────────────────────────────────────────

def _draw_strip(ax, v3: dict, seg_idx: int, label: str | None, conf: float, reason: str):
    signal   = v3["signal"]
    r_peaks  = v3["r_peaks"]
    per_beat = v3["delineation"].get("per_beat", [])
    features = v3["features"]
    sqi      = v3["sqi"]

    t = np.arange(len(signal)) / FS

    # Dynamic y range with 20 % padding
    sig_min, sig_max = float(np.min(signal)), float(np.max(signal))
    pad = max((sig_max - sig_min) * 0.20, 0.3)
    y_min = sig_min - pad
    y_max = sig_max + pad

    ax.set_facecolor(PAPER_BG)
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(y_min, y_max)

    _draw_ecg_paper(ax, 0.0, 10.0, y_min, y_max)

    # ECG trace
    ax.plot(t, signal, color=ECG_LINE, linewidth=0.75, zorder=3)

    # ── Annotations ──────────────────────────────────────────────────────────

    def _mark(indices, color, marker, zorder=5, s=22):
        idx = np.asarray(indices, dtype=int)
        ok  = idx[(idx >= 0) & (idx < len(signal))]
        if len(ok):
            ax.scatter(ok / FS, signal[ok], color=color, marker=marker,
                       s=s, zorder=zorder, linewidths=0.5)

    _p_pk  = [b["p_peak"]   for b in per_beat if b.get("p_peak")   is not None]
    _p_on  = [b["p_onset"]  for b in per_beat if b.get("p_onset")  is not None]
    _p_off = [b["p_offset"] for b in per_beat if b.get("p_offset") is not None]
    _q     = [b["q_point"]  for b in per_beat if b.get("q_point")  is not None]
    _s     = [b["s_point"]  for b in per_beat if b.get("s_point")  is not None]
    _t_pk  = [b["t_peak"]   for b in per_beat if b.get("t_peak")   is not None]
    _t_off = [b["t_offset"] for b in per_beat if b.get("t_offset") is not None]
    _qrs_on  = [b["qrs_onset"]  for b in per_beat if b.get("qrs_onset")  is not None]
    _qrs_off = [b["qrs_offset"] for b in per_beat if b.get("qrs_offset") is not None]

    # P-wave
    _mark(_p_pk,  "#007700", "o",  s=18)
    _mark(_p_on,  "#007700", "|",  s=12)
    _mark(_p_off, "#007700", "|",  s=12)
    # QRS
    _mark(_q,     "#0055cc", "v",  s=18)
    _mark(_s,     "#cc5500", "v",  s=18)
    # T-wave
    _mark(_t_pk,  "#7700cc", "D",  s=16)
    _mark(_t_off, "#7700cc", "|",  s=12)
    # R-peaks
    if len(r_peaks) > 0:
        ok = r_peaks[(r_peaks >= 0) & (r_peaks < len(signal))]
        ax.scatter(ok / FS, signal[ok], color="#dd0000", marker="^",
                   s=30, zorder=6, linewidths=0)

    # QRS onset/offset shading
    for on, off in zip(_qrs_on, _qrs_off):
        if 0 <= on < len(signal) and 0 <= off < len(signal):
            ax.axvspan(on / FS, off / FS, color="#ffe066", alpha=0.18, zorder=1)

    # ── Left-margin rhythm label ──────────────────────────────────────────────
    lcolor = LABEL_COLORS.get(label, LABEL_COLORS[None])
    seg_label = f"Seg {seg_idx}"
    rhythm_label = label if label else "Normal (SP)"

    ax.text(
        -0.01, 0.99, seg_label,
        transform=ax.transAxes, fontsize=7, color="#444444",
        va="top", ha="right", fontweight="bold",
    )
    ax.text(
        -0.01, 0.72, rhythm_label,
        transform=ax.transAxes, fontsize=7.5, color=lcolor,
        va="top", ha="right", fontweight="bold",
        wrap=False,
    )
    short_reason = reason.split("|")[0].strip()[:28]
    if label:
        ax.text(
            -0.01, 0.50, f"{conf*100:.0f}%",
            transform=ax.transAxes, fontsize=6.5, color=lcolor,
            va="top", ha="right",
        )
        ax.text(
            -0.01, 0.34, short_reason,
            transform=ax.transAxes, fontsize=5.2, color="#666666",
            va="top", ha="right", style="italic",
        )

    # ── Right-margin feature box ──────────────────────────────────────────────
    hr     = features.get("mean_hr_bpm") or 0
    qrs_ms = features.get("qrs_duration_ms") or 0
    pr_ms  = features.get("pr_interval_ms") or 0
    rr_cv  = features.get("rr_cv") or 0

    p_absent = features.get("p_absent_fraction")
    if p_absent is None:
        ppr = features.get("p_wave_present_ratio")
        p_absent = (1.0 - float(ppr)) if ppr is not None else None

    lines = [
        f"HR {hr:.0f}",
        f"QRS {qrs_ms:.0f}ms",
        f"PR {pr_ms:.0f}ms",
        f"rr_cv {rr_cv:.2f}",
        f"P-abs {p_absent:.2f}" if p_absent is not None else "",
        f"SQI {sqi:.2f}",
        f"N={len(r_peaks)}",
    ]
    lines = [l for l in lines if l]

    ax.text(
        1.002, 0.98, "\n".join(lines),
        transform=ax.transAxes, fontsize=6.5,
        va="top", ha="left", color="#222222",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffffcc",
                  edgecolor="#cccccc", linewidth=0.5),
        family="monospace",
    )

    # ── Y-axis: mV ticks ─────────────────────────────────────────────────────
    ax.set_ylabel("mV", fontsize=6.5, color="#555555", labelpad=2)
    ax.yaxis.set_tick_params(labelsize=5.5, colors="#666666")
    ax.xaxis.set_tick_params(labelsize=5.5, colors="#666666")
    ax.set_xlabel("")
    ax.spines[["top", "right", "bottom", "left"]].set_edgecolor("#cccccc")
    ax.spines[["top", "right", "bottom", "left"]].set_linewidth(0.5)

    # Only show x-label on last strip (caller's job), keep ticks on all
    ax.set_xticks(np.arange(0, 10.5, 0.5))
    ax.xaxis.set_tick_params(labelbottom=False)  # caller enables on last


# ── Multi-strip figure ────────────────────────────────────────────────────────

def plot_all_strips(
    seg_results: list[dict],     # list of {seg_idx, v3, label, conf, reason}
    file_stem: str,
    save_path: str | None,
):
    import matplotlib
    matplotlib.use("Agg" if save_path else "TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    n = len(seg_results)
    strip_h = 2.4    # inches per strip
    fig_h   = strip_h * n + 1.2   # extra for title + legend

    fig = plt.figure(figsize=(22, fig_h), facecolor=PAPER_BG)

    # Left margin for rhythm labels, right for feature box
    left_margin  = 0.085
    right_margin = 0.075
    top_margin   = 0.06 if n <= 4 else 0.05
    bottom_margin= 0.04

    strip_height_frac = (1.0 - top_margin - bottom_margin) / n
    gap = strip_height_frac * 0.06   # thin gap between strips

    axes = []
    for i, res in enumerate(seg_results):
        bottom = 1.0 - top_margin - (i + 1) * strip_height_frac + gap / 2
        ax = fig.add_axes(
            [left_margin, bottom,
             1.0 - left_margin - right_margin, strip_height_frac - gap]
        )
        _draw_strip(ax, res["v3"], res["seg_idx"],
                    res["label"], res["conf"], res["reason"])
        axes.append(ax)

    # Enable x-tick labels only on last strip
    axes[-1].xaxis.set_tick_params(labelbottom=True)
    axes[-1].set_xlabel("Time (s)", fontsize=7, color="#555555")

    # ── Global title ─────────────────────────────────────────────────────────
    fig.text(
        0.5, 1.0 - top_margin * 0.4,
        f"ECG Diagnostic Report — {file_stem}   ({n} × 10-second segments, 125 Hz)",
        ha="center", va="top", fontsize=11, fontweight="bold", color="#222222",
    )

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color=ECG_LINE,   lw=1.2,               label="Cleaned ECG"),
        Line2D([0], [0], color="#dd0000",  lw=0, marker="^", markersize=6, label="R-peak"),
        Line2D([0], [0], color="#007700",  lw=0, marker="o", markersize=5, label="P-peak"),
        Line2D([0], [0], color="#007700",  lw=0, marker="|", markersize=5, label="P on/off"),
        Line2D([0], [0], color="#0055cc",  lw=0, marker="v", markersize=5, label="Q-point"),
        Line2D([0], [0], color="#cc5500",  lw=0, marker="v", markersize=5, label="S-point"),
        Line2D([0], [0], color="#7700cc",  lw=0, marker="D", markersize=5, label="T-peak"),
        Patch(facecolor="#ffe066", alpha=0.5, label="QRS region"),
        Patch(facecolor=GRID_MAJOR, label="Major grid (0.2 s / 0.5 mV)"),
        Patch(facecolor=GRID_MINOR, label="Minor grid (0.04 s / 0.1 mV)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=5,
        fontsize=7.5,
        framealpha=0.9,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 0.0),
        facecolor=PAPER_BG,
    )

    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight",
                    facecolor=PAPER_BG)
        print(f"Saved -> {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ECG paper-style diagnostic viewer")
    parser.add_argument("--file",     required=True,  help="Path to ECG JSON file")
    parser.add_argument("--segment",  type=int, default=None,
                        help="Single segment index (default: all)")
    parser.add_argument("--max_segs", type=int, default=None,
                        help="Cap number of segments (e.g. --max_segs 6)")
    parser.add_argument("--save",     action="store_true",
                        help="Save PNG (auto-enabled when many segments)")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.is_absolute():
        file_path = BASE_DIR / file_path

    print(f"Loading: {file_path}")
    sig_raw, fs_orig = load_signal(str(file_path))
    print(f"  {len(sig_raw)} samples @ {fs_orig} Hz  ({len(sig_raw)/fs_orig:.1f} s)")

    sig = resample_to_125(sig_raw, fs_orig)
    n_total = max(1, len(sig) // WIN_SAMPLES)
    print(f"  -> 125 Hz: {n_total} segment(s) of 10 s")

    if args.segment is not None:
        seg_indices = [args.segment]
    else:
        seg_indices = list(range(n_total))
    if args.max_segs is not None:
        seg_indices = seg_indices[:args.max_segs]

    seg_results = []
    for seg_idx in seg_indices:
        if seg_idx >= n_total:
            print(f"  Skip: segment {seg_idx} out of range")
            continue
        print(f"  Seg {seg_idx:03d} ...", end=" ", flush=True)
        window = get_segment(sig, seg_idx)
        v3     = run_pipeline(window)
        label, conf, reason = run_arrhythmia(v3)
        tag = f"{label} ({conf:.0%})" if label else "Normal (SP)"
        print(f"HR={v3['features'].get('mean_hr_bpm', 0):.0f}  SQI={v3['sqi']:.2f}  [{tag}]")
        seg_results.append({
            "seg_idx": seg_idx,
            "v3":      v3,
            "label":   label,
            "conf":    conf,
            "reason":  reason,
        })

    if not seg_results:
        print("No segments to plot.")
        return

    n = len(seg_results)
    stem = file_path.stem
    save_path = None
    if args.save or n > 1:
        save_path = str(BASE_DIR / "scripts" / f"ecg_report_{stem}.png")

    print(f"\nRendering {n} strip(s) ...")
    plot_all_strips(seg_results, stem, save_path)


if __name__ == "__main__":
    main()
