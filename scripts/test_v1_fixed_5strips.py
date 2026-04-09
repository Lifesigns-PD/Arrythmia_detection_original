"""
test_v1_fixed_5strips.py
========================
Runs 5 ECG extract files through the v1 pipeline WITH all applied fixes and
saves one diagnostic PNG per 10-second segment.

Data format notes (confirmed):
  - ECG_Data_Extracts values are already in millivolts (range ~-1.2 to +1.1 mV).
  - NO x1000 conversion needed (that was for a different device batch).
  - Each packet's "value" is a list-of-lists, 200 samples per sub-list @ 125 Hz.

Fixes active in this run:
  [1] Force-display debug bypass removed   (decision_engine/rules.py)
  [2] Ectopy threshold 0.95→0.97 + rhythm  (decision_engine/rhythm_orchestrator.py)
      trust gate (0.99 on Sinus + <3 beats)
  [3] Per-class confidence thresholds for  (decision_engine/rhythm_orchestrator.py)
      rhythm model (VF=0.90, AF=0.85, ...)
  [4] p_wave_present_ratio neutral default  (signal_processing/feature_extraction.py)
      0.5 on morphology failure (was 0.0)
  [5] LayerNorm before feature input in    (models_training/models_v2.py)
      CNNTransformerWithFeatures

Outputs saved to:  <project_root>/testing_final/v1_fixed_strips/
Filename format:   v1_fixed_<ADM_ID>_seg<N>_<rhythm>.png
"""

import sys
import json
import warnings
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "models_training"))

from ecg_processor import process, SAMPLING_RATE, WINDOW_SAMPLES

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
EXTRACTS_DIR = BASE_DIR / "ECG_Data_Extracts"
OUT_DIR      = BASE_DIR / "testing_final" / "v1_fixed_strips"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_PER_PKT  = 200          # sub-list size in each packet
TARGET_SAMPLES   = 7500         # 1 minute @ 125 Hz — what process() expects
PICKS = [
    "ADM1014424580",
    "ADM1116085313",
    "ADM1317323275",
    "ADM550416376",
    "ADM833649891",
]

# Colour map for rhythm labels on plots
RHYTHM_COLORS = {
    "Sinus Rhythm":         "#2ecc71",
    "Sinus Bradycardia":    "#27ae60",
    "Sinus Tachycardia":    "#f39c12",
    "Atrial Fibrillation":  "#e74c3c",
    "Atrial Flutter":       "#c0392b",
    "1st Degree AV Block":  "#9b59b6",
    "2nd Degree AV Block Type 1": "#8e44ad",
    "2nd Degree AV Block Type 2": "#6c3483",
    "3rd Degree AV Block":  "#4a235a",
    "Bundle Branch Block":  "#2980b9",
    "Artifact":             "#95a5a6",
    "Unknown":              "#bdc3c7",
}

# ──────────────────────────────────────────────
# Signal extraction helper
# ──────────────────────────────────────────────

def extract_signal(fpath: Path) -> tuple[np.ndarray, str]:
    """
    Extract the first TARGET_SAMPLES samples from an extract JSON.
    Data is confirmed to be in millivolts — no conversion applied.
    """
    with open(fpath) as f:
        data = json.load(f)

    samples = []
    adm_id = Path(fpath).stem

    for pkt in data:
        if len(samples) >= TARGET_SAMPLES:
            break
        val = pkt.get("value")
        if isinstance(val, str):
            val = json.loads(val)
        adm_id = pkt.get("admissionId", adm_id)
        if isinstance(val, list):
            for chunk in val:
                if isinstance(chunk, list):
                    samples.extend(float(x) for x in chunk)
                elif isinstance(chunk, (int, float)):
                    samples.extend(float(x) for x in val)
                    break
        if len(samples) >= TARGET_SAMPLES:
            break

    sig = np.array(samples[:TARGET_SAMPLES], dtype=np.float32)
    # Pad if slightly short
    if len(sig) < TARGET_SAMPLES:
        sig = np.pad(sig, (0, TARGET_SAMPLES - len(sig)))

    return sig, adm_id


# ──────────────────────────────────────────────
# Plot one segment
# ──────────────────────────────────────────────

def plot_segment(signal: np.ndarray, seg_info: dict, adm_id: str,
                 seg_idx: int, out_path: Path) -> None:
    fs = SAMPLING_RATE
    t  = np.arange(len(signal)) / fs

    rhythm     = seg_info.get("rhythm_label", "Unknown")
    r_conf     = seg_info.get("rhythm_confidence", 0.0)
    ectopy     = seg_info.get("ectopy_label", "None")
    e_conf     = seg_info.get("ectopy_confidence", 0.0)
    events     = seg_info.get("events", [])
    primary    = seg_info.get("primary_conclusion", rhythm)
    hr         = seg_info.get("morphology", {}).get("hr_bpm") or 0
    p_wave     = seg_info.get("morphology", {}).get("p_wave_present_ratio")
    qrs_ms     = seg_info.get("morphology", {}).get("qrs_duration_ms") or 0
    pr_ms      = seg_info.get("morphology", {}).get("pr_interval_ms") or 0

    color = RHYTHM_COLORS.get(rhythm, "#3498db")

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    ax.plot(t, signal, color="#39ff14", linewidth=0.7, alpha=0.92)

    # Colour band at top showing rhythm
    ax.axhspan(signal.max() * 1.02, signal.max() * 1.12,
               color=color, alpha=0.4, clip_on=False)

    # Title
    ectopy_str = f"  |  Ectopy: {ectopy} ({e_conf:.2f})" if ectopy not in ("None", None, "") else ""
    events_str = "  Events: " + ", ".join(events) if events else ""
    p_str      = f"{p_wave:.2f}" if p_wave is not None else "N/A"

    title = (
        f"{adm_id}  ·  Segment {seg_idx}  ·  [FIX-V1]\n"
        f"Rhythm: {rhythm} ({r_conf:.2f}){ectopy_str}   "
        f"HR: {hr:.0f} bpm   QRS: {qrs_ms:.0f} ms   PR: {pr_ms:.0f} ms   "
        f"P-wave ratio: {p_str}{events_str}"
    )
    ax.set_title(title, color="white", fontsize=9, loc="left", pad=6)

    ax.set_xlabel("Time (s)", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("Amplitude (mV)", color="#aaaaaa", fontsize=8)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    # Grid
    ax.grid(True, linestyle="--", alpha=0.2, color="#444444")
    ax.set_xlim(0, t[-1])

    # Legend patch
    patches = [mpatches.Patch(color=color, label=f"{rhythm} ({r_conf:.0%})")]
    if ectopy not in ("None", None, ""):
        patches.append(mpatches.Patch(color="#e74c3c", label=f"{ectopy} ({e_conf:.0%})"))
    ax.legend(handles=patches, loc="upper right",
              facecolor="#1a1a2e", edgecolor="#555", labelcolor="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# ──────────────────────────────────────────────
# Summary table builder
# ──────────────────────────────────────────────

def print_summary(all_results: list[dict]) -> None:
    print("\n" + "="*90)
    print(f"  V1-FIXED  |  5 Strips  |  Segment-level results")
    print("="*90)
    print(f"  {'Admission':<20} {'Seg':>4}  {'Rhythm':<32} {'Conf':>5}  {'Ectopy':<12} {'Events'}")
    print("-"*90)
    for r in all_results:
        adm   = r["adm_id"]
        idx   = r["seg_idx"]
        rhy   = r["rhythm"][:31]
        conf  = r["conf"]
        ect   = r["ectopy"] or "—"
        evts  = ", ".join(r["events"]) if r["events"] else "—"
        print(f"  {adm:<20} {idx:>4}  {rhy:<32} {conf:>5.2f}  {ect:<12} {evts}")
    print("="*90)

    # Aggregate
    rhythms = [r["rhythm"] for r in all_results]
    from collections import Counter
    dist = Counter(rhythms)
    print("\n  Rhythm distribution across all segments:")
    for rhy, cnt in dist.most_common():
        pct = cnt / len(rhythms) * 100
        print(f"    {rhy:<35}  {cnt:>3}  ({pct:.0f}%)")
    print()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    all_results = []
    saved_files = []

    print(f"\n[v1-fixed] Output dir: {OUT_DIR}")
    print(f"[v1-fixed] Running {len(PICKS)} admissions...\n")

    for adm_id in PICKS:
        fpath = EXTRACTS_DIR / f"{adm_id}.json"
        if not fpath.exists():
            print(f"  [SKIP] {adm_id} — file not found")
            continue

        print(f"  Processing {adm_id}...")

        try:
            signal, resolved_adm = extract_signal(fpath)
            amp_range = float(signal.max() - signal.min())
            print(f"    Signal: {len(signal)} samples, range={amp_range:.3f} mV "
                  f"(confirmed mV — no conversion applied)")
        except Exception as exc:
            print(f"    [ERROR] signal extraction: {exc}")
            continue

        try:
            result = process(
                ecg_data=signal.tolist(),
                admission_id=adm_id,
                device_id="extract_test",
                timestamp=0,
            )
        except Exception as exc:
            print(f"    [ERROR] process() failed: {exc}")
            import traceback; traceback.print_exc()
            continue

        segs = result.get("analysis", {}).get("segments", [])
        print(f"    -> {len(segs)} segments | dominant: "
              f"{result['analysis']['background_rhythm']} | "
              f"HR: {result['analysis']['heart_rate_bpm']} bpm")

        for seg in segs:
            seg_idx = seg.get("segment_index", 0)
            rhythm  = seg.get("rhythm_label", "Unknown")

            # Slice signal for this segment
            s_start = seg_idx * WINDOW_SAMPLES
            s_end   = s_start + WINDOW_SAMPLES
            win     = signal[s_start:s_end]
            if len(win) < WINDOW_SAMPLES:
                win = np.pad(win, (0, WINDOW_SAMPLES - len(win)))

            # Build safe filename
            safe_rhythm = rhythm.replace(" ", "_").replace("/", "-")
            fname = f"v1_fixed_{adm_id}_seg{seg_idx:02d}_{safe_rhythm}.png"
            out_path = OUT_DIR / fname

            plot_segment(win, seg, adm_id, seg_idx, out_path)
            saved_files.append(str(out_path))

            all_results.append({
                "adm_id":  adm_id,
                "seg_idx": seg_idx,
                "rhythm":  rhythm,
                "conf":    seg.get("rhythm_confidence", 0.0),
                "ectopy":  seg.get("ectopy_label", "None") if seg.get("ectopy_label") not in ("None", None) else None,
                "events":  seg.get("events", []),
                "primary": seg.get("primary_conclusion", rhythm),
            })

        print(f"    Saved {len(segs)} PNGs to {OUT_DIR.name}/")

    print_summary(all_results)

    print(f"\n[v1-fixed] Total files saved: {len(saved_files)}")
    for f in saved_files:
        print(f"  {Path(f).name}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
