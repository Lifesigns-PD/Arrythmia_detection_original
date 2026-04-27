"""
ecg_processor.py — Full ECG arrhythmia detection pipeline (V3)
===============================================================
Input:  raw ecg_data (list[float], 7500 samples, 125 Hz, mV values)
Output: structured analysis dict matching MongoDB schema

Signal chain (all V3):
  preprocess_v3 → detect_r_peaks_ensemble → delineate_v3 → extract_features_v3
  → RhythmOrchestrator (rules + ML)
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
from signal_processing_v3 import process_ecg_v3


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


def _run_orchestrator(window: np.ndarray, v3_result: dict) -> dict:
    """
    Run ML inference + rules on a V3-processed window.
    v3_result: output of process_ecg_v3()
    """
    try:
        from xai.xai import explain_segment
        from decision_engine.rhythm_orchestrator import RhythmOrchestrator

        r_peaks  = v3_result["r_peaks"].tolist()
        features = v3_result["features"]
        summary  = v3_result["delineation"].get("summary", {})
        per_beat = v3_result["delineation"].get("per_beat", [])
        sqi      = v3_result["sqi"]

        rr_intervals = []
        if len(r_peaks) > 1:
            rr_intervals = (np.diff(r_peaks) / SAMPLING_RATE * 1000).tolist()

        clinical_features = {
            "mean_hr":              features.get("mean_hr_bpm", 0) or 0,
            "mean_hr_bpm":          features.get("mean_hr_bpm", 0) or 0,
            "pr_interval":          features.get("pr_interval_ms", 0) or 0,
            "rr_intervals_ms":      rr_intervals,
            "qrs_duration_ms":      features.get("qrs_duration_ms", 0) or 0,
            "qrs_durations_ms":     [b.get("qrs_duration_ms") for b in per_beat
                                     if b.get("qrs_duration_ms") is not None],
            "p_wave_present_ratio": summary.get("p_wave_present_ratio", 1.0),
            "r_peaks":              r_peaks,
            "fs":                   SAMPLING_RATE,
            "_signal":              window.tolist(),
            "_signal_clean":        v3_result.get("signal", window).tolist(),
            "_per_beat_delineation": per_beat,
        }
        # Merge all V3 features into clinical_features for orchestrator/explain_segment
        for k, v in features.items():
            clinical_features[k] = float(v) if v is not None else 0.0

        sqi_result = {"score": sqi, "is_acceptable": sqi >= 0.3}

        ml_prediction = explain_segment(window, clinical_features)
        if "error" in ml_prediction:
            raise RuntimeError(ml_prediction["error"])

        orchestrator = RhythmOrchestrator()
        decision = orchestrator.decide(ml_prediction, clinical_features, sqi_result)

        event_types    = [e.event_type for e in decision.final_display_events]
        rhythm_block   = ml_prediction.get("rhythm", {})
        ectopy_block   = ml_prediction.get("ectopy", {})

        return {
            "primary_conclusion": decision.background_rhythm if not event_types else event_types[0],
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
    Process 1-minute ECG data through the full V3 arrhythmia detection pipeline.

    Steps:
    1. Convert to numpy, segment into 6 x 10-second windows
    2. Per window: V3 full pipeline (preprocess → R-peaks → delineate → features)
    3. Run ML inference + rule-based orchestrator
    4. Aggregate: dominant rhythm, HR, events
    5. Return structured result matching MongoDB schema
    """
    t_start = time.time()
    signal  = np.ascontiguousarray(ecg_data, dtype=np.float32)
    windows = _segment(signal)

    segments_out = []
    all_rhythms, all_hrs, all_events = [], [], []

    for idx, window in enumerate(windows):
        window   = np.ascontiguousarray(window, dtype=np.float32)
        v3       = process_ecg_v3(window, fs=SAMPLING_RATE, min_quality=0.2)
        result   = _run_orchestrator(window, v3)

        summary  = v3["delineation"].get("summary", {})
        hr       = summary.get("mean_hr_bpm") or v3["features"].get("mean_hr_bpm")
        if hr:
            all_hrs.append(hr)

        primary      = result["primary_conclusion"]
        events       = result["events"]
        rhythm_label = result["rhythm_label"]
        rhythm_conf  = result["rhythm_conf"]
        ectopy_label = result["ectopy_label"]
        ectopy_conf  = result["ectopy_conf"]

        all_rhythms.append(primary)
        all_events.extend(events)
        if ectopy_label and ectopy_label not in ("None", "none", ""):
            all_events.append(ectopy_label)

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
                "qtc_ms":               summary.get("qtc_ms"),
                "p_wave_present_ratio": summary.get("p_wave_present_ratio"),
            },
            "signal_quality":       v3["sqi"],
            "sinus_gate_fired":     result["sinus_gate_fired"],
        })

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
        "ecgData":       ecg_data,
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
