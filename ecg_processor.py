"""
ecg_processor.py — Wrap the full ECG arrhythmia detection pipeline.

Input:  raw ecg_data (list[float], 7500 samples, 125 Hz, mV values)
Output: structured analysis dict matching MongoDB schema
"""
from __future__ import annotations

import sys
import warnings
import time
from pathlib import Path
from collections import Counter

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "models_training"))

from config import SAMPLING_RATE, WINDOW_SAMPLES
from signal_processing.cleaning import clean_signal


def _segment(signal: np.ndarray) -> list[np.ndarray]:
    """Split 1-D signal into 10-second windows (1250 samples each)."""
    windows, offset = [], 0
    min_samp = 500
    while offset + min_samp <= len(signal):
        chunk = signal[offset: offset + WINDOW_SAMPLES]
        if len(chunk) < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
        windows.append(chunk)
        offset += WINDOW_SAMPLES
    return windows


def _detect_r_peaks(window: np.ndarray) -> list[int]:
    from scipy.signal import find_peaks
    try:
        peaks, _ = find_peaks(
            window,
            distance=int(SAMPLING_RATE * 0.4),
            height=np.percentile(window, 75),
        )
        return peaks.tolist()
    except Exception:
        return []


def _extract_morphology(window: np.ndarray, r_peaks: list[int]) -> dict:
    try:
        from signal_processing.morphology import extract_morphology
        return extract_morphology(window, np.array(r_peaks, dtype=int), SAMPLING_RATE)
    except Exception as exc:
        return {"error": str(exc)}


def _run_orchestrator(window: np.ndarray, r_peaks: list[int], morph_data: dict) -> dict:
    """
    Full pipeline: V2 inference (signal + features) → RhythmOrchestrator.decide().
    Returns a dict with primary_conclusion, background_rhythm, events, and raw
    rhythm/ectopy labels for the per-segment output record.
    """
    try:
        from xai.xai import explain_segment
        from signal_processing.feature_extraction import extract_feature_vector, FEATURE_NAMES
        from signal_processing.sqi import calculate_sqi_score
        from decision_engine.rhythm_orchestrator import RhythmOrchestrator

        summary  = morph_data.get("summary", {})
        per_beat = morph_data.get("per_beat", [])

        # Build clinical features dict for the orchestrator
        rr_intervals = [b.get("rr_interval_ms") for b in per_beat
                        if b.get("rr_interval_ms") is not None]
        clinical_features = {
            "mean_hr":              summary.get("heart_rate_bpm", 0),
            "pr_interval":          summary.get("pr_interval_ms", 0),
            "rr_intervals_ms":      rr_intervals,
            "qrs_durations_ms":     [b.get("qrs_duration_ms") for b in per_beat
                                     if b.get("qrs_duration_ms") is not None],
            "p_wave_present_ratio": summary.get("p_wave_present_ratio", 1.0),
            "r_peaks":              r_peaks,
            "fs":                   SAMPLING_RATE,
        }
        # Merge scalar feature values so explain_segment can build the features tensor
        feat_vec = extract_feature_vector(window, fs=SAMPLING_RATE,
                                          r_peaks=np.array(r_peaks, dtype=int) if r_peaks else None)
        for i, name in enumerate(FEATURE_NAMES):
            clinical_features[name] = float(feat_vec[i])

        # SQI
        sqi_score = calculate_sqi_score(window, SAMPLING_RATE)
        sqi_result = {"score": sqi_score, "is_acceptable": sqi_score >= 0.3}

        # ML inference (V2: signal + features, per-beat ectopy)
        ml_prediction = explain_segment(window, clinical_features)
        if "error" in ml_prediction:
            raise RuntimeError(ml_prediction["error"])

        # Full orchestrator: rules + pattern detection + display arbitration
        orchestrator = RhythmOrchestrator()
        decision = orchestrator.decide(ml_prediction, clinical_features, sqi_result)

        # Extract event type strings for the output record
        event_types = [e.event_type for e in decision.final_display_events]

        rhythm_block = ml_prediction.get("rhythm", {})
        ectopy_block  = ml_prediction.get("ectopy", {})

        return {
            "primary_conclusion": decision.background_rhythm if not event_types
                                  else event_types[0],
            "background_rhythm":  decision.background_rhythm,
            "events":             event_types,
            "rhythm_label":       rhythm_block.get("label", "Unknown"),
            "rhythm_conf":        rhythm_block.get("confidence", 0.0),
            "ectopy_label":       ectopy_block.get("label", "None"),
            "ectopy_conf":        ectopy_block.get("confidence", 0.0),
            "sinus_gate_fired":   False,
        }

    except Exception as exc:
        warnings.warn(f"Orchestrator failed: {exc}")
        return {
            "primary_conclusion": "Unknown",
            "background_rhythm":  "Unknown",
            "events":             [],
            "rhythm_label":       "Unknown",
            "rhythm_conf":        0.0,
            "ectopy_label":       "None",
            "ectopy_conf":        0.0,
            "sinus_gate_fired":   False,
        }


def process(
    ecg_data: list[float],
    admission_id: str,
    device_id: str,
    timestamp: int,
    patient_id: str = "unknown",
    facility_id: str = "unknown",
) -> dict:
    """
    Process 1-minute ECG data through the full arrhythmia detection pipeline.

    Steps:
    1. Convert to numpy, clean signal
    2. Segment into 6 x 10-second windows
    3. Per window: R-peaks → morphology → inference → rules
    4. Aggregate: dominant rhythm, HR, events
    5. Return structured result matching MongoDB schema
    """
    t_start = time.time()
    signal = np.ascontiguousarray(ecg_data, dtype=np.float32)

    # Clean: baseline wander removal + powerline noise removal
    signal = np.ascontiguousarray(clean_signal(signal, SAMPLING_RATE), dtype=np.float32)

    windows = _segment(signal)
    segments_out = []
    all_rhythms, all_hrs, all_events = [], [], []

    for idx, window in enumerate(windows):
        window = np.ascontiguousarray(window, dtype=np.float32)
        r_peaks = _detect_r_peaks(window)
        morph   = _extract_morphology(window, r_peaks)

        result  = _run_orchestrator(window, r_peaks, morph)

        summary = morph.get("summary", {})
        hr = summary.get("heart_rate_bpm")
        if hr:
            all_hrs.append(hr)

        primary       = result["primary_conclusion"]
        events        = result["events"]
        rhythm_label  = result["rhythm_label"]
        rhythm_conf   = result["rhythm_conf"]
        ectopy_label  = result["ectopy_label"]
        ectopy_conf   = result["ectopy_conf"]

        all_rhythms.append(primary)
        all_events.extend(events)

        segments_out.append({
            "segment_index":        idx,
            "start_time_s":         round(idx * 10.0, 1),
            "end_time_s":           round((idx + 1) * 10.0, 1),
            "rhythm_label":         rhythm_label,
            "rhythm_confidence":    round(float(rhythm_conf), 4),
            "ectopy_label":         ectopy_label,
            "ectopy_confidence":    round(float(ectopy_conf), 4),
            "events":               events,
            "primary_conclusion":   primary,
            "background_rhythm":    result["background_rhythm"],
            "morphology": {
                "hr_bpm":               hr,
                "pr_interval_ms":       summary.get("pr_interval_ms"),
                "qrs_duration_ms":      summary.get("qrs_duration_ms"),
                "qtc_ms":               summary.get("qtc_bazett_ms"),
                "p_wave_present_ratio": summary.get("p_wave_present_ratio"),
            },
            "sinus_gate_fired":     result["sinus_gate_fired"],
        })

    # Aggregate
    dominant_rhythm = Counter(all_rhythms).most_common(1)[0][0] if all_rhythms else "Unknown"
    avg_hr = round(float(np.mean(all_hrs))) if all_hrs else None
    unique_events = sorted(set(all_events))
    arrhythmia_detected = any(
        e not in {"Sinus Rhythm", "Sinus Bradycardia", "Sinus Tachycardia", "None", "Unknown"}
        for e in unique_events
    )

    elapsed = round(time.time() - t_start, 2)

    return {
        "admissionId":   admission_id,
        "deviceId":      device_id,
        "patientId":     patient_id,
        "facilityId":    facility_id,
        "timestamp":     timestamp,
        "ecgData":       ecg_data,          # full raw array (mV)
        "analysis": {
            "background_rhythm":  dominant_rhythm,
            "heart_rate_bpm":     avg_hr,
            "segments":           segments_out,
            "summary": {
                "total_segments":      len(segments_out),
                "dominant_rhythm":     dominant_rhythm,
                "arrhythmia_detected": arrhythmia_detected,
                "events_found":        unique_events,
                "signal_quality":      "acceptable",
            },
        },
        "processingStatus": None,
        "processedAt":      None,
        "processedBy":      None,
        "_processing_time_s": elapsed,
    }
