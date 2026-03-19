"""
ingest_ecg_extracts.py — Standalone ECG Analyzer for MongoDB-exported patient data.

Reads ECG_Data_Extracts JSON files (packet-based format), runs the full
ML + rules pipeline, and outputs annotated PNGs + JSON reports.
No database involved.

Usage:
    python data/ingest_ecg_extracts.py --file ECG_Data_Extracts/ADM441825561.json
    python data/ingest_ecg_extracts.py --folder ECG_Data_Extracts/
    python data/ingest_ecg_extracts.py --file ECG_Data_Extracts/ADM441825561.json \
        --start "2026-03-07T17:15:00" --end "2026-03-07T17:20:00"
    python data/ingest_ecg_extracts.py --folder ECG_Data_Extracts/ --output outputs/patient_ecg/
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "models_training"))

from data.ingest_json import (
    _resample,
    _segment,
    _detect_r_peaks,
    _run_inference,
    _save_png,
    _extract_morphology,
    TARGET_FS,
    WINDOW_SAMPLES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GAP_THRESHOLD_S = 5.0   # seconds — gaps larger than this split into separate runs


# ---------------------------------------------------------------------------
# MongoDB JSON parsing
# ---------------------------------------------------------------------------

def _parse_timestamp(ts_obj: dict) -> datetime:
    """Parse MongoDB {'$date': '...'} timestamp to datetime."""
    iso = ts_obj["$date"].replace("Z", "+00:00")
    return datetime.fromisoformat(iso)


def _parse_extract_json(
    json_path: Path,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Tuple[str, List[Tuple[np.ndarray, int, str, str]]]:
    """
    Parse a MongoDB-exported ECG JSON file.

    Returns:
        (admission_id, runs)
        where each run is (signal_1d, estimated_fs, time_start_iso, time_end_iso)
        Runs are split on gaps > GAP_THRESHOLD_S.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        packets = json.load(f)

    if not packets:
        return "unknown", []

    # Sort by timestamp
    packets.sort(key=lambda p: p["utcTimestamp"]["$date"])

    # Time filter
    if start or end:
        filtered = []
        for p in packets:
            ts = _parse_timestamp(p["utcTimestamp"])
            ts_naive = ts.replace(tzinfo=None) if ts.tzinfo else ts
            if start and ts_naive < start:
                continue
            if end and ts_naive > end:
                continue
            filtered.append(p)
        packets = filtered

    if not packets:
        return "unknown", []

    admission_id = packets[0].get("admissionId", json_path.stem)

    # Extract timestamps and signals
    timestamps = []
    signals = []
    for p in packets:
        ts = _parse_timestamp(p["utcTimestamp"])
        timestamps.append(ts)
        # value is [[floats...]] — take inner list
        val = p["value"]
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
            signals.append(val[0])
        elif isinstance(val, list):
            signals.append(val)
        else:
            signals.append([])

    samples_per_packet = len(signals[0]) if signals else 0

    # Estimate fs from median inter-packet timing
    if len(timestamps) >= 2:
        deltas = []
        for i in range(1, len(timestamps)):
            dt = (timestamps[i] - timestamps[i - 1]).total_seconds()
            if 0.01 < dt < GAP_THRESHOLD_S:  # ignore gaps and zero-deltas
                deltas.append(dt)
        if deltas:
            median_dt = float(np.median(deltas))
            estimated_fs = int(round(samples_per_packet / median_dt))
        else:
            estimated_fs = 125  # fallback
    else:
        estimated_fs = 125

    # Split into continuous runs based on gaps
    runs = []
    run_signals = [signals[0]]
    run_start = timestamps[0]

    for i in range(1, len(timestamps)):
        gap = (timestamps[i] - timestamps[i - 1]).total_seconds()
        if gap > GAP_THRESHOLD_S:
            # Close current run
            concat = np.concatenate([np.array(s, dtype=np.float32) for s in run_signals])
            t_start_iso = run_start.strftime("%Y-%m-%dT%H:%M:%S")
            t_end_iso = timestamps[i - 1].strftime("%Y-%m-%dT%H:%M:%S")
            runs.append((concat, estimated_fs, t_start_iso, t_end_iso))
            # Start new run
            run_signals = [signals[i]]
            run_start = timestamps[i]
        else:
            run_signals.append(signals[i])

    # Close final run
    if run_signals:
        concat = np.concatenate([np.array(s, dtype=np.float32) for s in run_signals])
        t_start_iso = run_start.strftime("%Y-%m-%dT%H:%M:%S")
        t_end_iso = timestamps[-1].strftime("%Y-%m-%dT%H:%M:%S")
        runs.append((concat, estimated_fs, t_start_iso, t_end_iso))

    total_samples = sum(len(r[0]) for r in runs)
    total_dur = (timestamps[-1] - timestamps[0]).total_seconds()
    print(f"[extract] {json_path.name} -- {len(packets)} packets, spp={samples_per_packet}, "
          f"fs~{estimated_fs}Hz, {total_dur/60:.1f} min, {len(runs)} run(s)")

    return admission_id, runs


# ---------------------------------------------------------------------------
# Rules engine wrapper
# ---------------------------------------------------------------------------

def _run_rules_engine(
    window: np.ndarray,
    r_peaks: list,
    rhythm_label: Optional[str],
    rhythm_conf: Optional[float],
    ectopy_label: Optional[str],
    ectopy_conf: Optional[float],
    fs: int = 125,
) -> dict:
    """
    Run the full RhythmOrchestrator pipeline on a single window.
    Returns dict with background_rhythm, final_events, primary_conclusion.
    """
    try:
        from decision_engine.rhythm_orchestrator import RhythmOrchestrator
        from xai.xai import explain_segment

        signal_arr = np.asarray(window, dtype=np.float32)
        features = {"r_peaks": r_peaks, "fs": fs}

        # Get full ML evidence (rhythm + per-beat ectopy)
        ml_prediction = explain_segment(signal_arr, features)

        # Build clinical features
        r_arr = np.array(r_peaks, dtype=int)
        rr_intervals_ms = []
        mean_hr = 0.0
        if len(r_arr) >= 2:
            rr_intervals_ms = (np.diff(r_arr) * 1000.0 / fs).tolist()
            mean_rr = np.mean(rr_intervals_ms)
            mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 0.0

        clinical_features = {
            "mean_hr": mean_hr,
            "pr_interval": 0.0,
            "rr_intervals_ms": rr_intervals_ms,
            "qrs_durations_ms": [],
            "fs": fs,
        }

        sqi_result = {"is_acceptable": True, "overall_sqi": 1.0}

        orchestrator = RhythmOrchestrator()
        decision = orchestrator.decide(ml_prediction, clinical_features, sqi_result)

        final_events = [e.event_type for e in decision.final_display_events]
        primary = final_events[0] if final_events else decision.background_rhythm

        return {
            "background_rhythm": decision.background_rhythm,
            "final_events": final_events,
            "primary_conclusion": primary,
        }

    except Exception as exc:
        warnings.warn(f"Rules engine failed: {exc}")
        return {
            "background_rhythm": "Unknown",
            "final_events": [],
            "primary_conclusion": rhythm_label or "Unknown",
        }


# ---------------------------------------------------------------------------
# Process a single file
# ---------------------------------------------------------------------------

def process_file(
    json_path: Path,
    output_dir: Path,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> int:
    """Process one ECG_Data_Extracts JSON file. Returns number of segments processed."""

    admission_id, runs = _parse_extract_json(json_path, start, end)

    if not runs:
        print(f"[extract] No data after filtering. Skipping.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = json_path.stem
    total_segments = 0

    for run_idx, (signal, src_fs, t_start, t_end) in enumerate(runs):
        run_tag = f"_run{run_idx}" if len(runs) > 1 else ""
        print(f"[extract] Run {run_idx}: {len(signal)} samples, "
              f"{t_start} to {t_end}")

        # Resample to 125 Hz
        signal_125 = _resample(signal.tolist(), src_fs)
        print(f"[extract] Resampled -> {len(signal_125)} samples @ {TARGET_FS} Hz")

        # Segment into 10s windows
        windows = _segment(signal_125)
        print(f"[extract] Segmented -> {len(windows)} windows")

        for idx, window in enumerate(windows):
            seg_name = f"{filename}{run_tag}"
            r_peaks = _detect_r_peaks(window)

            # ML inference
            rhythm_label, rhythm_conf, ectopy_label, ectopy_conf = _run_inference(window)

            # Rules engine
            rules_result = _run_rules_engine(
                window, r_peaks,
                rhythm_label, rhythm_conf,
                ectopy_label, ectopy_conf,
                fs=TARGET_FS,
            )

            # Morphology
            morph_data = _extract_morphology(window, r_peaks)

            # Determine primary conclusion for display
            primary = rules_result.get("primary_conclusion", rhythm_label or "Unknown")
            events_str = ", ".join(rules_result.get("final_events", [])) or "None"

            # Print summary
            r_text = f"{rhythm_label} ({rhythm_conf:.0%})" if rhythm_label else "N/A"
            e_text = f"{ectopy_label} ({ectopy_conf:.0%})" if ectopy_label and ectopy_label != "None" else "-"
            print(f"  [seg {idx}] Rhythm: {r_text} | Ectopy: {e_text} | "
                  f"Rules: {primary} | Events: {events_str}")

            # Save PNG
            png_path = _save_png(
                window, r_peaks,
                seg_name, idx,
                gt_label=f"Rules: {primary}",
                rhythm_label=rhythm_label,
                rhythm_conf=rhythm_conf,
                ectopy_label=ectopy_label,
                ectopy_conf=ectopy_conf,
                output_dir=output_dir,
            )
            print(f"  [seg {idx}] PNG -> {png_path}")

            # Calculate segment timestamp offset
            seg_offset_s = idx * (WINDOW_SAMPLES / TARGET_FS)

            # Save JSON report
            report = {
                "segment_index": idx,
                "admission_id": admission_id,
                "filename": seg_name,
                "timestamp_start": t_start,
                "segment_offset_seconds": seg_offset_s,
                "prediction": {
                    "rhythm_label": rhythm_label,
                    "rhythm_confidence": rhythm_conf,
                    "ectopy_label": ectopy_label,
                    "ectopy_confidence": ectopy_conf,
                },
                "rules_engine": rules_result,
                "morphology": morph_data,
            }
            report_path = output_dir / f"{seg_name}_seg{idx}_report.json"
            with open(report_path, "w", encoding="utf-8") as rf:
                json.dump(report, rf, indent=2, default=str)
            print(f"  [seg {idx}] Report -> {report_path}")

            total_segments += 1

    print(f"[extract] Done. {total_segments} segments processed from {json_path.name}")
    return total_segments


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Standalone ECG analyzer for ECG_Data_Extracts (MongoDB JSON format)"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path, help="Single JSON file to process")
    group.add_argument("--folder", type=Path, help="Folder of JSON files to batch-process")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output directory for PNGs + reports (default: outputs/patient_ecg/)")
    p.add_argument("--start", default=None,
                   help="Start time filter (ISO format, e.g. '2026-03-07T17:15:00')")
    p.add_argument("--end", default=None,
                   help="End time filter (ISO format, e.g. '2026-03-07T17:20:00')")
    args = p.parse_args()

    output_dir = args.output.resolve() if args.output else BASE_DIR / "outputs" / "patient_ecg"

    start_dt = datetime.fromisoformat(args.start) if args.start else None
    end_dt = datetime.fromisoformat(args.end) if args.end else None

    if args.file:
        fpath = args.file.resolve()
        if not fpath.exists():
            sys.exit(f"[ERROR] File not found: {fpath}")
        process_file(fpath, output_dir, start_dt, end_dt)

    elif args.folder:
        folder = args.folder.resolve()
        if not folder.is_dir():
            sys.exit(f"[ERROR] Not a directory: {folder}")
        json_files = sorted(folder.glob("*.json"))
        if not json_files:
            sys.exit(f"[ERROR] No JSON files found in {folder}")

        print(f"[extract] Found {len(json_files)} JSON files in {folder}")
        grand_total = 0
        for fpath in json_files:
            print(f"\n{'='*60}")
            count = process_file(fpath, output_dir / fpath.stem, start_dt, end_dt)
            grand_total += count

        print(f"\n{'='*60}")
        print(f"[extract] BATCH COMPLETE. {grand_total} total segments from {len(json_files)} files.")
        print(f"[extract] Output: {output_dir}")


if __name__ == "__main__":
    main()
