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
import math
import sys
import warnings
import statistics
from datetime import datetime, timedelta, timezone
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
from signal_processing.cleaning import clean_signal

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
            # Snap to standard ECG frequencies to avoid artificial resampling due to packet jitter
            for std_fs in [125, 200, 250, 500]:
                if abs(estimated_fs - std_fs) <= 5:
                    estimated_fs = std_fs
                    break
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
# SP sinus rhythm detection — multi-feature (RR regularity + P-wave + PR)
# ---------------------------------------------------------------------------
# RR regularity thresholds
_SINUS_FILTER_LOW   = 0.85   # remove RR < 0.85 × median (premature beat)
_SINUS_FILTER_HIGH  = 1.15   # remove RR > 1.15 × median (compensatory pause)
_SINUS_MIN_BEATS    = 3
_SINUS_MIN_FRACTION = 0.40

# Sinus score thresholds
_SINUS_SCORE_PREMODEL  = 0.75  # strict: skip rhythm model (needs RR + P-wave)
_SINUS_SCORE_POSTMODEL = 0.60  # moderate: override bad model labels

# Post-model override fires on these model outputs
_SINUS_CANDIDATES = {
    "Atrial Fibrillation", "Atrial Flutter", "Unknown",
    "Junctional Rhythm", "Idioventricular Rhythm",
    "Ventricular Fibrillation",
}


def _detect_sinus_cv(r_peaks: list, cv_threshold: float = 0.10) -> tuple[bool, float | None]:
    """
    Filter ectopic beats from RR intervals and check regularity.
    Returns (is_regular, hr_bpm).
    """
    if len(r_peaks) < 4:
        return False, None
    rr = np.diff(np.array(r_peaks, dtype=np.float64)) * (1000.0 / TARGET_FS)
    if len(rr) < 3:
        return False, None
    median_rr = float(np.median(rr))
    if median_rr <= 0:
        return False, None
    sinus_rr = rr[(rr >= _SINUS_FILTER_LOW * median_rr) & (rr <= _SINUS_FILTER_HIGH * median_rr)]
    if len(sinus_rr) < _SINUS_MIN_BEATS or len(sinus_rr) / len(rr) < _SINUS_MIN_FRACTION:
        return False, None
    mean_s = float(np.mean(sinus_rr))
    cv = float(np.std(sinus_rr)) / mean_s if mean_s > 0 else 1.0
    if cv < cv_threshold:
        return True, 60_000.0 / mean_s if mean_s > 0 else None
    return False, None


def _compute_sinus_score(r_peaks: list, morph_data: dict) -> tuple[float, str]:
    """
    Multi-feature sinus confidence score combining:
      1. Filtered RR regularity   (0.40 weight) — regular underlying rhythm
      2. P-wave presence ratio    (0.35 weight) — P-waves confirm atrial sinus origin
      3. Normal PR interval       (0.25 weight) — normal AV conduction = sinus pathway

    Returns (score 0.0-1.0, human-readable reason string).
    """
    score = 0.0
    reasons = []
    summary = morph_data.get("summary", {})

    # Feature 1: Filtered RR regularity (CV < 0.10 relaxed)
    is_regular, hr = _detect_sinus_cv(r_peaks, 0.10)
    if is_regular:
        score += 0.40
        hr_str = f"{hr:.0f}" if hr else "?"
        reasons.append(f"regular_RR(HR={hr_str})")

    # Feature 2: P-wave presence ratio (>= 0.5 = majority of beats have P-waves)
    p_ratio = summary.get("p_wave_present_ratio", 0.0) or 0.0
    if p_ratio >= 0.5:
        score += 0.35
        reasons.append(f"P_waves({p_ratio:.0%})")

    # Feature 3: Normal PR interval (80-220ms = sinus AV conduction)
    pr = summary.get("pr_interval_ms")
    if pr and 80 <= pr <= 220:
        score += 0.25
        reasons.append(f"PR={pr:.0f}ms")

    return score, " + ".join(reasons) if reasons else "no sinus features"


def _sinus_label_from_hr(hr: float | None) -> str:
    if hr is not None and hr < 60:
        return "Sinus Bradycardia"
    if hr is not None and hr > 100:
        return "Sinus Tachycardia"
    return "Sinus Rhythm"


def _sp_sinus_premodel(r_peaks: list, morph_data: dict) -> tuple[bool, str | None, float | None]:
    """
    Pre-model sinus gate (strict score >= 0.75).

    Requires morphology to be extracted first (for P-wave + PR features).
    If SP confirms sinus → skip rhythm model, set label from SP.

    Returns (is_sinus, rhythm_label, hr_bpm).
    """
    score, reasons = _compute_sinus_score(r_peaks, morph_data)
    if score >= _SINUS_SCORE_PREMODEL:
        _, hr = _detect_sinus_cv(r_peaks, 0.10)
        label = _sinus_label_from_hr(hr)
        print(f"    [SINUS_GATE] pre-model: score={score:.2f} ({reasons}) -> {label} (rhythm model skipped)")
        return True, label, hr
    return False, None, None


def _sp_sinus_postmodel(rhythm_label: str, r_peaks: list, morph_data: dict) -> str:
    """
    Post-model sinus gate (moderate score >= 0.60).

    Only fires if rhythm model output is in _SINUS_CANDIDATES.
    Overrides to sinus variant when SP detects sinus features.
    """
    if rhythm_label not in _SINUS_CANDIDATES:
        return rhythm_label
    score, reasons = _compute_sinus_score(r_peaks, morph_data)
    if score >= _SINUS_SCORE_POSTMODEL:
        _, hr = _detect_sinus_cv(r_peaks, 0.10)
        new_label = _sinus_label_from_hr(hr)
        print(f"    [SINUS_GATE] post-model: score={score:.2f} ({reasons}) -> '{rhythm_label}' overridden to '{new_label}'")
        return new_label
    return rhythm_label


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
    morph_data: Optional[dict] = None,
) -> dict:
    """
    Run the full RhythmOrchestrator pipeline on a single window.
    Returns dict with background_rhythm, final_events, primary_conclusion.
    """
    try:
        from decision_engine.rhythm_orchestrator import RhythmOrchestrator
        from decision_engine.models import EventCategory
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

        # Populate clinical features from morphology if available
        morph_summary = (morph_data or {}).get("summary", {})
        pr_val = morph_summary.get("pr_interval_ms", 0.0) or 0.0
        qrs_list = [b.get("qrs_duration_ms") for b in (morph_data or {}).get("per_beat", [])
                     if b.get("qrs_duration_ms") is not None]
        per_beat_pr = [b.get("pr_interval_ms") for b in (morph_data or {}).get("per_beat", [])
                       if b.get("pr_interval_ms") is not None]
        p_wave_ratio = morph_summary.get("p_wave_present_ratio")

        clinical_features = {
            "mean_hr": mean_hr,
            "pr_interval": pr_val,
            "rr_intervals_ms": rr_intervals_ms,
            "qrs_durations_ms": qrs_list,
            "per_beat_pr_ms": per_beat_pr,
            "p_wave_present_ratio": p_wave_ratio,
            "fs": fs,
        }

        sqi_result = {"is_acceptable": True, "overall_sqi": 1.0}

        orchestrator = RhythmOrchestrator()
        decision = orchestrator.decide(ml_prediction, clinical_features, sqi_result)

        final_events = [e.event_type for e in decision.final_display_events]
        primary = final_events[0] if final_events else decision.background_rhythm

        # Extract per-beat ectopy markers from fully arbitrated decision events
        beat_markers = []
        for e in decision.final_display_events:
            if hasattr(e, "event_category") and e.event_category == EventCategory.ECTOPY and e.beat_indices:
                # Get the peak_sample from the ml_prediction for this beat_idx
                # (since Event stores start/end time, original peak_sample is safest to pull from ml_prediction)
                beat_idx = e.beat_indices[0]
                peak_sample = None
                for b in ml_prediction.get("ectopy", {}).get("beat_events", []):
                    if b.get("beat_idx") == beat_idx:
                        peak_sample = b.get("peak_sample")
                        conf = b.get("conf", 0.0)
                        break
                
                if peak_sample is not None:
                    # Prefer pattern_label (e.g., Bigeminy, Couplet) over raw event_type if available
                    label = getattr(e, "pattern_label", None) or e.event_type
                    beat_markers.append({
                        "peak_sample": peak_sample,
                        "label": label,
                        "conf": conf,
                    })

        # Full event detail with start/end times for PDF annotation
        events_detail = [
            {
                "event_type": e.event_type,
                "event_category": e.event_category.value,
                "start_time": e.start_time,
                "end_time": e.end_time,
                "beat_indices": e.beat_indices,
                "priority": e.priority,
            }
            for e in decision.final_display_events
            if hasattr(e, "display_state") and e.display_state.value == "DISPLAYED"
        ]

        return {
            "background_rhythm": decision.background_rhythm,
            "final_events": final_events,
            "primary_conclusion": primary,
            "beat_markers": beat_markers,
            "ml_prediction": ml_prediction,
            "rr_intervals_ms": rr_intervals_ms,
            "mean_hr": mean_hr,
            "events_detail": events_detail,
        }

    except Exception as exc:
        warnings.warn(f"Rules engine failed: {exc}")
        return {
            "background_rhythm": "Unknown",
            "final_events": [],
            "primary_conclusion": rhythm_label or "Unknown",
            "beat_markers": [],
            "ml_prediction": {},
            "rr_intervals_ms": [],
            "mean_hr": 0.0,
            "events_detail": [],
        }


# ---------------------------------------------------------------------------
# gRPC-aligned JSON report builder
# ---------------------------------------------------------------------------

def _build_grpc_report(
    admission_id: str,
    seg_name: str,
    seg_idx: int,
    seg_offset_s: float,
    t_start: str,
    rhythm_label: Optional[str],
    rhythm_conf: Optional[float],
    ectopy_label: Optional[str],
    ectopy_conf: Optional[float],
    rules_result: dict,
    morph_data: dict,
    detail: bool = False,
) -> dict:
    """Build a JSON report matching the grpc_output_example.json schema."""

    primary = rules_result.get("primary_conclusion", rhythm_label or "Unknown")
    ml_pred = rules_result.get("ml_prediction", {})
    morph_summary = morph_data.get("summary", {})

    # --- arrhythmia_type: rhythm, ectopy, or both ---
    rhythm_abnormal = primary not in (
        "Sinus Rhythm", "Normal Sinus Rhythm", "Unknown", "None", None,
    )
    ectopy_abnormal = ectopy_label not in ("None", None, "Unknown")

    if rhythm_abnormal and ectopy_abnormal:
        arrhythmia_type = f"{primary} + {ectopy_label}"
    elif rhythm_abnormal:
        arrhythmia_type = primary
    elif ectopy_abnormal:
        arrhythmia_type = ectopy_label
    else:
        arrhythmia_type = "Normal Sinus Rhythm"

    # --- confidence: use the higher of rhythm/ectopy ---
    confidence = max(
        rhythm_conf if rhythm_conf is not None else 0.0,
        ectopy_conf if ectopy_conf is not None else 0.0,
    )

    # --- timestamp (Unix ms) ---
    timestamp_ms = 0
    try:
        from datetime import timezone as _tz
        dt = datetime.fromisoformat(str(t_start))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_tz.utc)
        timestamp_ms = int(dt.timestamp() * 1000)
    except Exception:
        pass

    # --- averaged morphology features (top-level) ---
    mean_hr = rules_result.get("mean_hr", morph_summary.get("heart_rate_bpm", 0.0))
    rr_intervals_ms = rules_result.get("rr_intervals_ms", [])

    # --- message ---
    hr_str = f"{mean_hr:.0f}" if mean_hr else "N/A"
    message = f"[{admission_id}] {arrhythmia_type} detected (HR: {hr_str} bpm)"

    # --- detection_window with per-beat start_pos/end_pos ---
    beat_events_raw = ml_pred.get("ectopy", {}).get("beat_events", [])

    # events[0] = rhythm span (full window)
    rhythm_span_label = primary if rhythm_abnormal else (rhythm_label or "Sinus Rhythm")
    rhythm_span_conf = round(rhythm_conf if rhythm_conf is not None else 0.0, 2)
    detection_events = [{
        "label": rhythm_span_label,
        "start_pos": 0,
        "end_pos": WINDOW_SAMPLES,
        "confidence": rhythm_span_conf,
    }]

    # events[1:] = only arrhythmia beats (PVC/PAC) — skip label "None"
    for br in beat_events_raw:
        lbl = br.get("label", "None")
        if lbl in ("None", None):
            continue
        ps = br.get("peak_sample", 0)
        detection_events.append({
            "beat_idx": br.get("beat_idx", 0),
            "peak_sample": ps,
            "start_pos": max(0, ps - 37),
            "end_pos": min(WINDOW_SAMPLES, ps + 37),
            "label": lbl,
            "confidence": round(br.get("conf", 0.0), 4),
        })

    if rhythm_abnormal and not ectopy_abnormal:
        det_desc = "Rhythm-level arrhythmia - spans entire 10-second analysis window"
    elif ectopy_abnormal and not rhythm_abnormal:
        det_desc = "Ectopy-level arrhythmia - localized to specific beats"
    elif rhythm_abnormal and ectopy_abnormal:
        det_desc = "Rhythm + Ectopy arrhythmia - combined findings"
    else:
        det_desc = "No arrhythmia detected"

    detection_window = {
        "description": det_desc,
        "beat_events": {
            "events": detection_events,
        },
    }

    # --- xai_explanation ---
    rhythm_data = ml_pred.get("rhythm", {})
    ectopy_data = ml_pred.get("ectopy", {})
    saliency_data = ml_pred.get("saliency", [])

    # Build class_probabilities dict from probs array
    class_probs = {}
    rhythm_probs = rhythm_data.get("probs", [])
    if rhythm_probs:
        try:
            from models_training.data_loader import RHYTHM_CLASS_NAMES
            for i, name in enumerate(RHYTHM_CLASS_NAMES):
                if i < len(rhythm_probs):
                    class_probs[name] = round(rhythm_probs[i], 4)
        except ImportError:
            class_probs = {"raw_probs": rhythm_probs}

    # Clinical reasoning via generate_detailed_ledger
    clinical_reasoning = ""
    try:
        from xai.xai import generate_detailed_ledger
        bg = rules_result.get("background_rhythm", "Unknown")
        events_for_ledger = []
        for bm in rules_result.get("beat_markers", []):
            events_for_ledger.append({
                "event_type": bm["label"],
                "start_time": bm["peak_sample"] / TARGET_FS,
            })
        clinical_reasoning = generate_detailed_ledger(bg, events_for_ledger)
    except Exception:
        clinical_reasoning = f"Primary: {arrhythmia_type} (confidence {confidence:.0%})"

    xai_explanation = {
        "rhythm": {
            "label": rhythm_data.get("label", rhythm_label or "Unknown"),
            "class_probabilities": class_probs,
            "transformer_attention": rhythm_data.get("attention", "N/A"),
        },
        "ectopy": {
            "label": ectopy_data.get("label", ectopy_label or "None"),
            "confidence": round(ectopy_data.get("confidence", ectopy_conf or 0.0), 4),
            "beat_events": detection_events,
        },
        "saliency": {
            "description": "Gradient saliency map - normalized 0.0 to 1.0 per sample. "
                           "High values = signal regions that most influenced the model decision.",
            "values": saliency_data if isinstance(saliency_data, list) else [],
        },
        "clinical_reasoning": clinical_reasoning,
    }

    # --- Assemble two-part report ---
    report = {
        # ── Main output (gRPC wire fields) ──
        "arrhythmia_type": arrhythmia_type,
        "confidence": round(confidence, 4),
        "message": message,
        "patient_id": admission_id,
        "timestamp": timestamp_ms,

        # Averaged morphology features
        "mean_hr": round(mean_hr, 2) if mean_hr else 0.0,
        "p_wave_duration_ms": round(morph_summary.get("p_wave_duration_ms", 0.0), 1),
        "p_wave_amplitude_mv": round(morph_summary.get("p_wave_amplitude_mv", 0.0), 4),
        "p_wave_present_ratio": round(morph_summary.get("p_wave_present_ratio", 0.0), 2),
        "pr_interval_ms": round(morph_summary.get("pr_interval_ms", 0.0), 1),
        "pr_segment_ms": round(morph_summary.get("pr_segment_ms", 0.0), 1),
        "qrs_duration_ms": round(morph_summary.get("qrs_duration_ms", 0.0), 1),
        "qrs_amplitude_mv": round(morph_summary.get("qrs_amplitude_mv", 0.0), 4),
        "st_segment_ms": round(morph_summary.get("st_segment_ms", 0.0), 1),
        "st_deviation_mv": round(morph_summary.get("st_deviation_mv", 0.0), 4),
        "t_wave_duration_ms": round(morph_summary.get("t_wave_duration_ms", 0.0), 1),
        "t_wave_amplitude_mv": round(morph_summary.get("t_wave_amplitude_mv", 0.0), 4),
        "qt_interval_ms": round(morph_summary.get("qtc_bazett_ms", 0.0) / 1.0, 1),  # QTc as proxy
        "qtc_bazett_ms": round(morph_summary.get("qtc_bazett_ms", 0.0), 1),
        "rr_interval_ms": round(morph_summary.get("rr_interval_ms", 0.0), 1),
        "sdnn_ms": round(morph_summary.get("sdnn_ms", 0.0), 2),
        "rmssd_ms": round(morph_summary.get("rmssd_ms", 0.0), 2),
        "rr_intervals_ms": [round(v, 1) for v in rr_intervals_ms],

        "detection_window": detection_window,

        # AI Ledger (always included)
        "clinical_reasoning": clinical_reasoning,
    }

    # ── BACKEND (only when --detail flag is set) ──
    if detail:
        report["BACKEND"] = {
            "xai_explanation": xai_explanation,
            "_detailed": {
                "segment_index": seg_idx,
                "filename": seg_name,
                "segment_offset_seconds": seg_offset_s,
                "timestamp_start": t_start,
                "prediction": {
                    "rhythm_label": rhythm_label,
                    "rhythm_confidence": rhythm_conf,
                    "ectopy_label": ectopy_label,
                    "ectopy_confidence": ectopy_conf,
                },
                "rules_engine": {
                    "background_rhythm": rules_result.get("background_rhythm", "Unknown"),
                    "final_events": rules_result.get("final_events", []),
                    "primary_conclusion": primary,
                    "beat_markers": rules_result.get("beat_markers", []),
                },
                "morphology": morph_data,
            },
        }

    return report


# ---------------------------------------------------------------------------
# PDF Report Generation (LEPU-style)
# ---------------------------------------------------------------------------

# Event highlight color palette (matching LEPU style)
PDF_EVENT_PALETTE = [
    {"fill": (220, 252, 231), "line": (34, 197, 94)},     # light green
    {"fill": (219, 234, 254), "line": (59, 130, 246)},    # light blue
    {"fill": (236, 253, 245), "line": (16, 185, 129)},    # light mint
    {"fill": (254, 249, 195), "line": (202, 138, 4)},     # light yellow
    {"fill": (237, 233, 254), "line": (139, 92, 246)},    # light lavender
    {"fill": (207, 250, 254), "line": (6, 182, 212)},     # light cyan
    {"fill": (228, 247, 239), "line": (5, 150, 105)},     # pastel green
    {"fill": (228, 240, 255), "line": (30, 64, 175)},     # pastel blue
    {"fill": (255, 249, 235), "line": (180, 83, 9)},      # pastel amber
    {"fill": (245, 240, 255), "line": (126, 34, 206)},    # pastel purple
]


def _generate_pdf_report(
    admission_id: str,
    filename: str,
    runs: List[Tuple[np.ndarray, int, str, str]],
    all_segments: list,
    output_dir: Path,
) -> Path:
    """
    Generate a LEPU-style PDF report with:
      - Portrait summary page (patient info, diagnoses, events, HR, morphology)
      - Landscape ECG strip pages (6 strips/page, pink grid, event highlights, waveform, labels)

    Parameters
    ----------
    admission_id : str
    filename : str
    runs : list of (signal_125hz, src_fs, t_start, t_end)
        The resampled continuous runs at 125 Hz.
    all_segments : list of dicts
        Each dict has keys: window, r_peaks, rhythm_label, rhythm_conf,
        ectopy_label, ectopy_conf, rules_result, morph_data, seg_idx, run_idx
    output_dir : Path
    """
    try:
        from fpdf import FPDF
    except ImportError:
        warnings.warn("fpdf2 not installed — skipping PDF generation. Install with: pip install fpdf2")
        return None

    pdf_path = output_dir / f"{filename}_report.pdf"

    # Collect all resampled samples into one continuous array for ECG strips
    all_samples = np.concatenate([seg["window"] for seg in all_segments])
    total_samples = len(all_samples)

    # Build event spans and beat markers from rules engine results
    event_spans = _build_event_spans(all_segments)
    beat_markers = _build_beat_markers(all_segments)

    # Collect unique diagnoses
    diagnoses = []
    seen_diag = set()
    for seg in all_segments:
        primary = seg["rules_result"].get("primary_conclusion", "Unknown")
        if primary and primary not in seen_diag:
            seen_diag.add(primary)
            diagnoses.append(primary)
        for ev in seg["rules_result"].get("final_events", []):
            if ev and ev not in seen_diag:
                seen_diag.add(ev)
                diagnoses.append(ev)

    # Compute average HR from morphology
    hr_values = []
    for seg in all_segments:
        morph = seg.get("morph_data", {})
        summary = morph.get("summary", {})
        hr = summary.get("heart_rate_bpm", 0)
        if hr and hr > 0:
            hr_values.append(hr)
    avg_hr = f"{np.mean(hr_values):.0f} bpm" if hr_values else "N/A"
    min_hr = f"{min(hr_values):.0f} bpm" if hr_values else "N/A"
    max_hr = f"{max(hr_values):.0f} bpm" if hr_values else "N/A"

    # Time range
    t_start_all = all_segments[0]["t_start"] if all_segments else "N/A"
    t_end_all = all_segments[-1]["t_end"] if all_segments else "N/A"

    # ── PORTRAIT SUMMARY PAGE ──
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 17)
    pdf.cell(0, 8, "AI-ECG Clinical Review Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(75, 85, 99)
    pdf.cell(0, 6, f"Admission: {admission_id}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    # Summary table
    summary_rows = [
        ("Admission ID", admission_id),
        ("Source File", filename),
        ("Window Start", str(t_start_all)),
        ("Window End", str(t_end_all)),
        ("Total Segments", str(len(all_segments))),
        ("Total ECG Samples", str(total_samples)),
        ("Duration", f"{total_samples / TARGET_FS / 60:.1f} min"),
        ("Average HR", avg_hr),
        ("Min HR", min_hr),
        ("Max HR", max_hr),
    ]
    _pdf_section_table(pdf, "Summary", summary_rows)

    # Diagnoses
    if not diagnoses:
        diagnoses = ["No abnormal diagnosis reported"]
    _pdf_bullet_section(pdf, "Diagnoses", diagnoses)

    # Per-segment results table — two-row layout per segment
    _pdf_section_header(pdf, "Segment Analysis")
    usable = pdf.w - pdf.l_margin - pdf.r_margin
    lh = 5.4
    for seg in all_segments:
        if pdf.get_y() > 255:
            pdf.add_page()
        idx = seg["seg_idx"]
        r_lbl = seg.get("rhythm_label") or "N/A"
        r_conf = seg.get("rhythm_conf")
        e_lbl = seg.get("ectopy_label") or "-"
        e_conf = seg.get("ectopy_conf")
        rules = seg.get("rules_result", {})
        primary = rules.get("primary_conclusion", "Unknown")
        bg_rhythm = rules.get("background_rhythm", "")

        r_text = f"{r_lbl} ({r_conf:.0%})" if r_conf is not None else r_lbl
        e_text = f"{e_lbl} ({e_conf:.0%})" if e_conf is not None and e_lbl not in (None, "None", "-") else (e_lbl if e_lbl not in (None, "-") else "None")

        # Compute segment clock window
        try:
            seg_t = seg.get("t_start") or t_start_all
            dt_s = datetime.fromisoformat(str(seg_t))
            dt_e = dt_s + timedelta(seconds=10)
            clock_range = f"{dt_s.strftime('%H:%M:%S')} - {dt_e.strftime('%H:%M:%S')}"
        except Exception:
            clock_range = f"Seg {idx}"

        # Per-seg morphology
        ms = seg.get("morph_data", {}).get("summary", {})
        hr_v = ms.get("heart_rate_bpm", 0)
        pr_v = ms.get("pr_interval_ms", 0)
        qrs_v = ms.get("qrs_duration_ms", 0)
        qtc_v = ms.get("qtc_bazett_ms", 0)
        morph_str = ""
        if ms and ms.get("num_beats", 0) > 0:
            morph_str = f"  HR:{hr_v:.0f}bpm  PR:{pr_v:.0f}ms  QRS:{qrs_v:.0f}ms  QTc:{qtc_v:.0f}ms"

        # Row A — bold, Primary Conclusion
        row_a = f"Seg {idx}  [{clock_range}]   PRIMARY: {primary}"
        if bg_rhythm:
            row_a += f"   BG: {bg_rhythm}"

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(248, 250, 252)
        pdf.multi_cell(usable, lh, row_a, border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

        # Row B — lighter, ML outputs + morphology
        row_b = f"  ML Rhythm: {r_text}   |   ML Ectopy: {e_text}{morph_str}"
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(60, 70, 80)
        pdf.multi_cell(usable, lh, row_b, border=1, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(0.5)
    pdf.ln(2)

    # HR Info
    hr_rows = [("Average Heart Rate", avg_hr), ("Min Heart Rate", min_hr), ("Max Heart Rate", max_hr)]
    _pdf_section_table(pdf, "HR Info", hr_rows)

    # Aggregate morphology summary — median across all valid segments
    _agg: dict[str, list] = {
        "heart_rate_bpm": [], "pr_interval_ms": [], "qrs_duration_ms": [],
        "qtc_bazett_ms": [], "st_deviation_mv": [], "rr_interval_ms": [],
        "sdnn_ms": [], "rmssd_ms": [], "p_wave_duration_ms": [],
        "p_wave_present_ratio": [], "num_beats": [],
    }
    _agg_flags: dict[str, list] = {
        "pr_interval_flag": [], "qrs_flag": [], "qtc_flag": [],
        "st_flag": [], "rr_flag": [], "p_wave_flag": [],
    }
    for seg in all_segments:
        ms = seg.get("morph_data", {}).get("summary", {})
        if not ms or ms.get("num_beats", 0) == 0:
            continue
        for k in _agg:
            v = ms.get(k)
            if v is not None and v > 0:
                _agg[k].append(v)
        for k in _agg_flags:
            v = ms.get(k)
            if v:
                _agg_flags[k].append(v)

    def _median(lst):
        return statistics.median(lst) if lst else 0.0

    def _majority_flag(lst):
        if not lst:
            return "N/A"
        counts: dict[str, int] = {}
        for f in lst:
            counts[f] = counts.get(f, 0) + 1
        return max(counts, key=lambda x: counts[x])

    if any(_agg["heart_rate_bpm"]):
        morph_rows = [
            ("Heart Rate (median)", f"{_median(_agg['heart_rate_bpm']):.1f} bpm"),
            ("PR Interval (median)", f"{_median(_agg['pr_interval_ms']):.1f} ms  ({_majority_flag(_agg_flags['pr_interval_flag'])})"),
            ("QRS Duration (median)", f"{_median(_agg['qrs_duration_ms']):.1f} ms  ({_majority_flag(_agg_flags['qrs_flag'])})"),
            ("QTc Bazett (median)", f"{_median(_agg['qtc_bazett_ms']):.1f} ms  ({_majority_flag(_agg_flags['qtc_flag'])})"),
            ("ST Deviation (median)", f"{_median(_agg['st_deviation_mv']):.3f} mV  ({_majority_flag(_agg_flags['st_flag'])})"),
            ("RR Interval (median)", f"{_median(_agg['rr_interval_ms']):.1f} ms  ({_majority_flag(_agg_flags['rr_flag'])})"),
            ("P Wave Duration (median)", f"{_median(_agg['p_wave_duration_ms']):.1f} ms  ({_majority_flag(_agg_flags['p_wave_flag'])})"),
            ("SDNN (mean)", f"{_median(_agg['sdnn_ms']):.1f} ms"),
            ("RMSSD (mean)", f"{_median(_agg['rmssd_ms']):.1f} ms"),
            ("Segments with data", str(len(_agg["heart_rate_bpm"]))),
        ]
        _pdf_section_table(pdf, "Morphology Summary (Aggregate - All Segments)", morph_rows)

    # ── LANDSCAPE ECG STRIP PAGES ──
    _pdf_ecg_strips(pdf, all_samples, event_spans, beat_markers, diagnoses, t_start_all, all_segments)

    pdf.output(str(pdf_path))
    return pdf_path


def _pdf_section_header(pdf, title: str):
    """Draw a styled section header."""
    if pdf.get_y() > 250:
        pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.cell(0, 8, title, border=1, new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.set_font("Helvetica", "", 10)


def _pdf_section_table(pdf, title: str, rows: list):
    """Draw a two-column key-value table section."""
    _pdf_section_header(pdf, title)
    usable = pdf.w - pdf.l_margin - pdf.r_margin
    left_w = usable * 0.34
    right_w = usable - left_w
    line_h = 5.4

    for key, value in rows:
        if not key.strip():
            continue
        value = value.strip() if value else "-"
        if not value:
            value = "-"

        y_before = pdf.get_y()
        x_before = pdf.get_x()

        # Left cell (key)
        pdf.set_font("Helvetica", "B", 9)
        pdf.multi_cell(left_w, line_h, key, border=1, new_x="RIGHT", new_y="TOP")
        left_h = pdf.get_y() - y_before

        # Right cell (value)
        pdf.set_xy(x_before + left_w, y_before)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(right_w, line_h, value, border=1)
        right_h = pdf.get_y() - y_before

        row_h = max(left_h, right_h)
        pdf.set_xy(x_before, y_before + row_h)

        if pdf.get_y() > 260:
            pdf.add_page()
    pdf.ln(2)


def _pdf_bullet_section(pdf, title: str, items: list):
    """Draw a bulleted list section."""
    _pdf_section_header(pdf, title)
    pdf.set_font("Helvetica", "", 10)
    for item in items:
        if pdf.get_y() > 262:
            pdf.add_page()
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 6, "- " + item, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


_SKIP_EVENT_LABELS = {
    "Sinus Rhythm", "Sinus Bradycardia", "Sinus Tachycardia",
    "Artifact", "Unknown", "None", "Normal", "Normal Sinus Rhythm",
}


def _compute_clock(seg_t_start: str, offset_s: float) -> str:
    """Convert a segment start timestamp + offset seconds to HH:MM:SS string."""
    try:
        dt = datetime.fromisoformat(str(seg_t_start)) + timedelta(seconds=float(offset_s))
        return dt.strftime("%H:%M:%S")
    except Exception:
        return f"{offset_s:.1f}s"


def _build_event_spans(all_segments: list) -> list:
    """
    Build event spans from segment rules results for ECG strip overlay.
    Uses actual event start_time/end_time (from events_detail) so ectopy
    clusters are highlighted only over their true beat range, not the full strip.
    Normal sinus variants are skipped entirely — no highlight on clean strips.

    Each span: {start_idx, end_idx, label, color_idx, clock_start, clock_end, category}
    """
    spans = []
    label_color = {}
    color_counter = 0

    for seg in all_segments:
        seg_idx = seg["seg_idx"]
        seg_start_sample = seg_idx * WINDOW_SAMPLES
        seg_t_start = seg.get("t_start", "")

        events_detail = seg["rules_result"].get("events_detail", [])

        # Fall back to final_events strings if events_detail not populated
        if not events_detail:
            primary = seg["rules_result"].get("primary_conclusion", "")
            fallback = list(seg["rules_result"].get("final_events", []))
            if primary and primary not in fallback:
                fallback.insert(0, primary)
            for ev_label in fallback:
                if not ev_label or ev_label in _SKIP_EVENT_LABELS:
                    continue
                if ev_label not in label_color:
                    label_color[ev_label] = color_counter
                    color_counter += 1
                spans.append({
                    "start_idx": seg_start_sample,
                    "end_idx": seg_start_sample + WINDOW_SAMPLES - 1,
                    "label": ev_label,
                    "color_idx": label_color[ev_label],
                    "clock_start": _compute_clock(seg_t_start, 0),
                    "clock_end": _compute_clock(seg_t_start, 10),
                    "category": "RHYTHM",
                })
            continue

        for ev in events_detail:
            ev_label = ev.get("event_type", "")
            if not ev_label or ev_label in _SKIP_EVENT_LABELS:
                continue

            start_s = float(ev.get("start_time", 0.0))
            end_s = float(ev.get("end_time", 10.0))

            # For rhythm events spanning the whole segment, use full strip width
            # For ectopy patterns, use the actual cluster start/end
            start_sample = seg_start_sample + int(start_s * TARGET_FS)
            end_sample = seg_start_sample + int(end_s * TARGET_FS)
            end_sample = min(end_sample, seg_start_sample + WINDOW_SAMPLES - 1)

            if ev_label not in label_color:
                label_color[ev_label] = color_counter
                color_counter += 1

            spans.append({
                "start_idx": start_sample,
                "end_idx": end_sample,
                "label": ev_label,
                "color_idx": label_color[ev_label],
                "clock_start": _compute_clock(seg_t_start, start_s),
                "clock_end": _compute_clock(seg_t_start, end_s),
                "category": ev.get("event_category", "RHYTHM"),
            })

    return spans


def _build_beat_markers(all_segments: list) -> list:
    """
    Build global beat marker list from per-segment beat_markers.
    Each marker: {global_sample, label, conf}
    """
    markers = []
    for seg in all_segments:
        seg_idx = seg["seg_idx"]
        offset = seg_idx * WINDOW_SAMPLES
        for bm in seg["rules_result"].get("beat_markers", []):
            markers.append({
                "global_sample": offset + bm["peak_sample"],
                "label": bm["label"],
                "conf": bm.get("conf", 0.0),
            })
    return markers


def _pdf_ecg_strips(
    pdf,
    all_samples: np.ndarray,
    event_spans: list,
    beat_markers: list,
    diagnoses: list,
    t_start: str,
    all_segments: list = None,
):
    """
    Generate landscape ECG strip pages matching LEPU format.
    Draw order: event fills -> pink grid -> waveform -> beat markers -> annotation labels.
    """
    samples_per_row = WINDOW_SAMPLES  # 1250
    rows_per_page = 6
    total_rows = math.ceil(len(all_samples) / samples_per_row)
    if total_rows == 0:
        return

    # Pad samples to fill last row
    padded_len = total_rows * samples_per_row
    if len(all_samples) < padded_len:
        all_samples = np.pad(all_samples, (0, padded_len - len(all_samples)))

    # Page dimensions (A4 landscape)
    page_w = 297.0
    page_h = 210.0
    margin_l = 8.0
    margin_t = 8.0
    margin_b = 8.0
    strip_x = margin_l
    strip_w = page_w - margin_l - 8.0  # 281 mm

    # Parse start time for line labels
    try:
        t_start_dt = datetime.fromisoformat(str(t_start))
    except Exception:
        t_start_dt = None

    for page_start_row in range(0, total_rows, rows_per_page):
        # Add landscape page
        pdf.add_page(orientation="L")

        # Page header
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 6, f"ECG Paper ({total_rows} lines x 10s per strip) - {t_start}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(80, 80, 80)
        diag_text = "Diagnoses: " + (" | ".join(diagnoses) if diagnoses else "No abnormal diagnosis reported")
        pdf.multi_cell(0, 4.5, diag_text)
        pdf.ln(1)

        header_bottom = pdf.get_y()
        usable = page_h - margin_t - margin_b - (header_bottom - margin_t)
        row_gap = 2.0
        strip_h = (usable - (rows_per_page - 1) * row_gap) / rows_per_page

        for row in range(page_start_row, min(page_start_row + rows_per_page, total_rows)):
            rel_row = row - page_start_row
            strip_y = header_bottom + rel_row * (strip_h + row_gap)
            center_y = strip_y + strip_h / 2.0

            start_idx = row * samples_per_row
            end_idx = min(start_idx + samples_per_row, len(all_samples))
            has_data = start_idx < end_idx

            # Line label with timestamps
            seg_data = all_segments[row] if all_segments and row < len(all_segments) else None
            
            if seg_data and seg_data.get("t_start"):
                try:
                    t_start_run = datetime.fromisoformat(str(seg_data["t_start"]))
                    offset_s = seg_data.get("seg_idx_in_run", 0) * 10
                    ls = t_start_run + timedelta(seconds=offset_s)
                    le = ls + timedelta(seconds=10)
                    line_label = f"Line {row + 1} (Run {seg_data.get('run_idx', 0) + 1})  {ls.strftime('%H:%M:%S')} - {le.strftime('%H:%M:%S')}"
                except Exception:
                    line_label = f"Line {row + 1}  ({row * 10}s - {(row + 1) * 10}s)"
            elif t_start_dt:
                ls = t_start_dt + timedelta(seconds=row * 10)
                le = t_start_dt + timedelta(seconds=(row + 1) * 10)
                line_label = f"Line {row + 1}  {ls.strftime('%H:%M:%S')} - {le.strftime('%H:%M:%S')}"
            else:
                line_label = f"Line {row + 1}  ({row * 10}s - {(row + 1) * 10}s)"

            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(60, 60, 60)
            pdf.text(strip_x, strip_y - 0.6, line_label)

            # LAYER 1: Event highlight fills (behind everything)
            if has_data:
                for ev in event_spans:
                    if ev["end_idx"] < start_idx or ev["start_idx"] > end_idx - 1:
                        continue
                    seg_start = max(ev["start_idx"], start_idx)
                    seg_end = min(ev["end_idx"], end_idx - 1)
                    if seg_end < seg_start:
                        continue
                    local_start = seg_start - start_idx
                    local_end = seg_end - start_idx
                    xs = strip_x + (local_start / (samples_per_row - 1)) * strip_w
                    xe = strip_x + (local_end / (samples_per_row - 1)) * strip_w
                    if xe < xs:
                        xs, xe = xe, xs
                    w = max(1.0, xe - xs)
                    c = PDF_EVENT_PALETTE[ev["color_idx"] % len(PDF_EVENT_PALETTE)]
                    pdf.set_fill_color(*c["fill"])
                    pdf.rect(xs, strip_y, w, strip_h, style="F")

            # LAYER 2: ECG grid + border (pink/red grid lines)
            small_x = strip_w / 250.0
            small_y = strip_h / 26.0

            # Small grid lines (pink)
            pdf.set_draw_color(236, 72, 153)
            pdf.set_line_width(0.08)
            gx = strip_x
            while gx <= strip_x + strip_w + 0.01:
                pdf.line(gx, strip_y, gx, strip_y + strip_h)
                gx += small_x
            gy = strip_y
            while gy <= strip_y + strip_h + 0.01:
                pdf.line(strip_x, gy, strip_x + strip_w, gy)
                gy += small_y

            # Large grid lines (red, every 5th)
            pdf.set_draw_color(220, 38, 38)
            pdf.set_line_width(0.14)
            gx = strip_x
            while gx <= strip_x + strip_w + 0.01:
                pdf.line(gx, strip_y, gx, strip_y + strip_h)
                gx += small_x * 5
            gy = strip_y
            while gy <= strip_y + strip_h + 0.01:
                pdf.line(strip_x, gy, strip_x + strip_w, gy)
                gy += small_y * 5

            # Border
            pdf.set_draw_color(70, 70, 70)
            pdf.set_line_width(0.3)
            pdf.rect(strip_x, strip_y, strip_w, strip_h)

            if not has_data:
                pdf.set_font("Helvetica", "", 7)
                pdf.set_text_color(150, 150, 150)
                pdf.text(strip_x + 4, center_y + 2, "No ECG samples available")
                continue

            # LAYER 3: ECG waveform
            segment = all_samples[start_idx:end_idx].astype(np.float64)
            median_val = np.median(segment)
            centered = segment - median_val
            # Use peak-to-peak range (1st-99th percentile) for robust scaling
            if len(centered) > 10:
                p01 = np.percentile(centered, 1)
                p99 = np.percentile(centered, 99)
                pp_range = max(p99 - p01, 0.01)
            else:
                pp_range = 1.0
            # Fill 80% of strip height with the signal range
            target_height = strip_h * 0.80
            scale = target_height / pp_range

            denom = float(samples_per_row - 1)
            pdf.set_draw_color(15, 15, 15)
            pdf.set_line_width(0.3)

            for i in range(1, len(centered)):
                xp = strip_x + ((i - 1) / denom) * strip_w
                xc = strip_x + (i / denom) * strip_w
                yp = max(strip_y + 0.3, min(center_y - centered[i - 1] * scale, strip_y + strip_h - 0.3))
                yc = max(strip_y + 0.3, min(center_y - centered[i] * scale, strip_y + strip_h - 0.3))
                pdf.line(xp, yp, xc, yc)

            # LAYER 3.5: Beat markers (PAC/PVC triangles on waveform)
            for bm in beat_markers:
                gs = bm["global_sample"]
                if gs < start_idx or gs >= end_idx:
                    continue
                local_idx = gs - start_idx
                bx = strip_x + (local_idx / denom) * strip_w

                # Y position: place marker at the waveform point
                if local_idx < len(centered):
                    by_wave = center_y - centered[local_idx] * scale
                    by_wave = max(strip_y + 0.3, min(by_wave, strip_y + strip_h - 0.3))
                else:
                    by_wave = center_y

                lbl = bm["label"]
                # PVC = red, PAC = blue, Run/pattern = orange
                if lbl == "PVC":
                    r, g, b = 220, 38, 38     # red
                elif lbl == "PAC":
                    r, g, b = 37, 99, 235     # blue
                else:
                    r, g, b = 234, 140, 0     # orange (Run/pattern)

                # Draw downward-pointing triangle above the peak
                tri_size = 1.8  # mm
                ty = by_wave - tri_size - 0.5  # above the waveform
                # Clamp within strip
                if ty < strip_y + 0.3:
                    ty = by_wave + 0.5  # place below if no room above
                pdf.set_fill_color(r, g, b)
                pdf.set_draw_color(r, g, b)
                pdf.set_line_width(0.2)
                # Triangle: three lines forming filled triangle
                x1, y1 = bx, ty + tri_size          # bottom point
                x2, y2 = bx - tri_size * 0.7, ty    # top-left
                x3, y3 = bx + tri_size * 0.7, ty    # top-right
                # Use polygon via lines + fill
                pdf.polygon([(x2, y2), (x3, y3), (x1, y1)], style="F")

                # Small label next to triangle
                pdf.set_font("Helvetica", "B", 4.5)
                pdf.set_text_color(r, g, b)
                lbl_x = bx + tri_size * 0.8
                lbl_y = ty + tri_size * 0.6
                # Clamp label within strip
                if lbl_x + 5 > strip_x + strip_w:
                    lbl_x = bx - tri_size * 0.8 - pdf.get_string_width(lbl)
                pdf.text(lbl_x, lbl_y, lbl)

            # LAYER 4: Annotation labels + event boundary lines
            seen_label = set()
            slot = 0
            for ev in event_spans:
                if ev["end_idx"] < start_idx or ev["start_idx"] > end_idx - 1:
                    continue
                key = f"{ev['label']}|{row}"
                if key in seen_label:
                    continue
                seen_label.add(key)

                lbl = ev["label"]
                if len(lbl) > 30:
                    lbl = lbl[:29] + "."

                # Compute exact x positions for event start/end within this strip
                ev_local_start = max(ev["start_idx"], start_idx) - start_idx
                ev_local_end = min(ev["end_idx"], end_idx - 1) - start_idx
                xs = strip_x + (ev_local_start / (samples_per_row - 1)) * strip_w
                xe = strip_x + (ev_local_end / (samples_per_row - 1)) * strip_w

                # Dashed vertical lines at event boundaries
                pdf.set_draw_color(*PDF_EVENT_PALETTE[ev["color_idx"] % len(PDF_EVENT_PALETTE)]["line"])
                pdf.set_line_width(0.4)
                dash_step = 1.2
                yy = strip_y
                while yy < strip_y + strip_h:
                    pdf.line(xs, yy, xs, min(yy + 0.7, strip_y + strip_h))
                    yy += dash_step
                if xe > xs + 2:
                    yy = strip_y
                    while yy < strip_y + strip_h:
                        pdf.line(xe, yy, xe, min(yy + 0.7, strip_y + strip_h))
                        yy += dash_step

                # Label box
                pdf.set_font("Helvetica", "B", 5.5)
                tw = pdf.get_string_width(lbl) + 2.4
                lx = xs + 0.5
                if lx + tw > strip_x + strip_w - 1:
                    lx = strip_x + strip_w - tw - 1

                # Stack: name box (row A) + time range (row B)
                box_h = 4.0
                ly = strip_y + 1.0 + slot * 9.0
                if ly + box_h + 4.5 > strip_y + strip_h - 0.5:
                    break

                # Row A — event name box
                pdf.set_fill_color(255, 255, 255)
                pdf.set_draw_color(0, 0, 0)
                pdf.set_line_width(0.25)
                pdf.rect(lx, ly, tw, box_h, style="FD")
                pdf.set_text_color(0, 0, 0)
                pdf.text(lx + 1.2, ly + 3.1, lbl)

                # Row B — clock time range in gray
                clock_start = ev.get("clock_start", "")
                clock_end = ev.get("clock_end", "")
                if clock_start and clock_end:
                    time_str = f"{clock_start} -> {clock_end}"
                    pdf.set_font("Helvetica", "", 4.5)
                    pdf.set_text_color(90, 90, 90)
                    pdf.text(lx, ly + box_h + 3.2, time_str)

                slot += 1

            # LAYER 5: Per-strip morphology line (below strip border)
            morph_line_y = strip_y + strip_h + 1.8
            if seg_data and morph_line_y < strip_y + strip_h + row_gap - 0.5:
                ms = seg_data.get("morph_data", {}).get("summary", {}) if seg_data else {}
                if ms and ms.get("num_beats", 0) > 0:
                    hr_v = ms.get("heart_rate_bpm")
                    pr_v = ms.get("pr_interval_ms")
                    qrs_v = ms.get("qrs_duration_ms")
                    qtc_v = ms.get("qtc_bazett_ms")
                    rr_v = ms.get("rr_interval_ms")

                    pr_flag = ms.get("pr_interval_flag", "normal")
                    qrs_flag = ms.get("qrs_flag", "normal")
                    qtc_flag = ms.get("qtc_flag", "normal")

                    parts = []
                    if hr_v:  parts.append(f"HR:{hr_v:.0f}bpm")
                    if pr_v:  parts.append(f"PR:{pr_v:.0f}ms{'*' if pr_flag != 'normal' else ''}")
                    if qrs_v: parts.append(f"QRS:{qrs_v:.0f}ms{'*' if qrs_flag != 'normal' else ''}")
                    if qtc_v: parts.append(f"QTc:{qtc_v:.0f}ms{'*' if qtc_flag != 'normal' else ''}")
                    if rr_v:  parts.append(f"RR:{rr_v:.0f}ms")

                    morph_text = "  |  ".join(parts)
                    has_flag = any(f != "normal" for f in [pr_flag, qrs_flag, qtc_flag] if f)
                    pdf.set_font("Helvetica", "", 5.0)
                    pdf.set_text_color(180, 30, 30) if has_flag else pdf.set_text_color(70, 70, 70)
                    pdf.text(strip_x + 1.0, morph_line_y, morph_text)
                    pdf.set_text_color(0, 0, 0)


# ---------------------------------------------------------------------------
# Process a single file
# ---------------------------------------------------------------------------

def process_file(
    json_path: Path,
    output_dir: Path,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    detail: bool = False,
    pdf_only: bool = False,
) -> int:
    """Process one ECG_Data_Extracts JSON file. Returns number of segments processed."""

    admission_id, runs = _parse_extract_json(json_path, start, end)

    if not runs:
        print(f"[extract] No data after filtering. Skipping.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = json_path.stem
    total_segments = 0
    all_pdf_segments = []  # Collect for PDF report

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
            window = np.ascontiguousarray(clean_signal(window, TARGET_FS), dtype=np.float32)
            r_peaks = _detect_r_peaks(window)

            # Morphology FIRST — needed for multi-feature sinus gate (P-wave + PR)
            morph_data = _extract_morphology(window, r_peaks)

            # SP sinus pre-model gate (score >= 0.75: RR regularity + P-waves + PR)
            # If SP confirms sinus → skip rhythm model, run only ectopy model.
            is_sinus_pre, sinus_label, _ = _sp_sinus_premodel(r_peaks, morph_data)
            if is_sinus_pre:
                rhythm_label = sinus_label
                rhythm_conf = 1.0
                _, _, ectopy_label, ectopy_conf = _run_inference(window)
            else:
                rhythm_label, rhythm_conf, ectopy_label, ectopy_conf = _run_inference(window)

            # SP sinus post-model gate (score >= 0.60) — catches remaining mislabels
            rhythm_label = _sp_sinus_postmodel(rhythm_label, r_peaks, morph_data)

            # Rules engine (receives morph_data for P-wave ratio in AF detection)
            rules_result = _run_rules_engine(
                window, r_peaks,
                rhythm_label, rhythm_conf,
                ectopy_label, ectopy_conf,
                fs=TARGET_FS,
                morph_data=morph_data,
            )

            # Determine primary conclusion for display
            primary = rules_result.get("primary_conclusion", rhythm_label or "Unknown")
            events_str = ", ".join(rules_result.get("final_events", [])) or "None"

            # Print summary
            r_text = f"{rhythm_label} ({rhythm_conf:.0%})" if rhythm_label else "N/A"
            e_text = f"{ectopy_label} ({ectopy_conf:.0%})" if ectopy_label and ectopy_label != "None" else "-"
            print(f"  [seg {idx}] Rhythm: {r_text} | Ectopy: {e_text} | "
                  f"Rules: {primary} | Events: {events_str}")

            # Save PNG
            if not pdf_only:
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

            # Build gRPC-aligned JSON report
            if not pdf_only:
                report = _build_grpc_report(
                    admission_id=admission_id,
                    seg_name=seg_name,
                    seg_idx=idx,
                    seg_offset_s=seg_offset_s,
                    t_start=t_start,
                    rhythm_label=rhythm_label,
                    rhythm_conf=rhythm_conf,
                    ectopy_label=ectopy_label,
                    ectopy_conf=ectopy_conf,
                    rules_result=rules_result,
                    morph_data=morph_data,
                    detail=detail,
                )
                report_path = output_dir / f"{seg_name}_seg{idx}_report.json"
                with open(report_path, "w", encoding="utf-8") as rf:
                    json.dump(report, rf, indent=2, default=str)
                print(f"  [seg {idx}] Report -> {report_path}")

            # Collect segment data for PDF
            all_pdf_segments.append({
                "window": window,
                "r_peaks": r_peaks,
                "rhythm_label": rhythm_label,
                "rhythm_conf": rhythm_conf,
                "ectopy_label": ectopy_label,
                "ectopy_conf": ectopy_conf,
                "rules_result": rules_result,
                "morph_data": morph_data,
                "seg_idx": total_segments,
                "seg_idx_in_run": idx,
                "run_idx": run_idx,
                "t_start": t_start,
                "t_end": t_end,
            })

            total_segments += 1

    # Generate LEPU-style PDF report
    if all_pdf_segments:
        print(f"[extract] Generating PDF report...")
        pdf_path = _generate_pdf_report(
            admission_id=admission_id,
            filename=filename,
            runs=runs,
            all_segments=all_pdf_segments,
            output_dir=output_dir,
        )
        if pdf_path:
            print(f"[extract] PDF -> {pdf_path}")

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
    p.add_argument("--detail", action="store_true", default=False,
                   help="Include BACKEND section (xai_explanation, _detailed, morphology per-beat) in JSON output")
    p.add_argument("--pdf-only", action="store_true", default=False,
                   help="Only generate the PDF report, skip PNGs and JSONs")
    args = p.parse_args()

    output_dir = args.output.resolve() if args.output else BASE_DIR / "outputs" / "patient_ecg"

    start_dt = datetime.fromisoformat(args.start) if args.start else None
    end_dt = datetime.fromisoformat(args.end) if args.end else None

    if args.file:
        fpath = args.file.resolve()
        if not fpath.exists():
            sys.exit(f"[ERROR] File not found: {fpath}")
        process_file(fpath, output_dir, start_dt, end_dt, detail=args.detail)

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
            target_out = output_dir if args.pdf_only else output_dir / fpath.stem
            count = process_file(fpath, target_out, start_dt, end_dt, detail=args.detail, pdf_only=args.pdf_only)
            grand_total += count

        print(f"\n{'='*60}")
        print(f"[extract] BATCH COMPLETE. {grand_total} total segments from {len(json_files)} files.")
        print(f"[extract] Output: {output_dir}")


if __name__ == "__main__":
    main()
