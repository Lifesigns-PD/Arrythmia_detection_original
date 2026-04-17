"""
signal_processing_v3 — Unified V3 ECG Processing API
======================================================
Single entry point: process_ecg_v3()

Active pipeline: custom wavelet (fixed)
  1. Preprocess   — adaptive baseline + denoising
  2. Quality gate — SQI v3
  3. Detect       — ensemble R-peak (Pan-Tompkins + Hilbert + Wavelet)
  4. Delineate    — CWT wavelet + template matching
  5. Extract      — 60 features: HRV time/freq, nonlinear, morphology, beat discriminators
  6. SQI final    — recompute with detected peaks

Backup pipeline: pipeline_nk2.py (NK2 DWT)
  NK2 DWT gives better p_absent for AF but incorrect QRS at 125 Hz (coarse
  wavelet scales → boundaries land in wrong place for narrow-complex rhythms).
  To activate: comment out the body below and uncomment the NK2 line.
"""

import numpy as np
from typing import Dict, Any

from .preprocessing.pipeline import preprocess_v3
from .detection.ensemble     import detect_r_peaks_ensemble, refine_peaks_subsample
from .delineation.hybrid     import delineate_v3
from .quality.signal_quality import compute_sqi_v3
from .features.extraction    import extract_features_v3, feature_dict_to_vector, FEATURE_NAMES_V3

# Backup NK2 pipeline — uncomment + comment body of process_ecg_v3 to activate
# from .pipeline_nk2 import process_ecg_nk2 as process_ecg_v3  # noqa: F401


def process_ecg_v3(
    raw_signal: np.ndarray,
    fs: int = 125,
    min_quality: float = 0.3,
) -> Dict[str, Any]:
    """
    Full V3 ECG processing pipeline.

    Parameters
    ----------
    raw_signal  : 1-D raw ECG (mV)
    fs          : sampling rate in Hz
    min_quality : SQI threshold below which processing is aborted

    Returns
    -------
    dict with keys:
      signal, r_peaks, delineation, features, feature_vector,
      sqi, sqi_issues, method, skipped
    """
    raw_sqi, raw_issues = compute_sqi_v3(raw_signal, fs=fs)

    prep    = preprocess_v3(raw_signal, fs=fs, skip_if_unusable=(min_quality > 0))
    cleaned = prep["cleaned"]

    if prep.get("was_skipped"):
        return _skipped(raw_signal, raw_sqi, raw_issues)

    r_peaks       = detect_r_peaks_ensemble(cleaned, fs=fs)
    r_peaks_float = refine_peaks_subsample(cleaned, r_peaks)

    sqi, issues = compute_sqi_v3(cleaned, r_peaks=r_peaks, fs=fs)
    if sqi < min_quality:
        return _skipped(cleaned, sqi, issues)

    delineation    = delineate_v3(cleaned, r_peaks, fs=fs)
    features       = extract_features_v3(cleaned, r_peaks_float, delineation, fs=fs)
    feature_vector = feature_dict_to_vector(features)

    return {
        "signal":         cleaned,
        "r_peaks":        r_peaks,
        "delineation":    delineation,
        "features":       features,
        "feature_vector": feature_vector,
        "sqi":            sqi,
        "sqi_issues":     issues,
        "method":         delineation.get("method", "unknown"),
        "skipped":        False,
    }


def _skipped(signal, sqi, issues):
    n = len(FEATURE_NAMES_V3)
    return {
        "signal":         np.asarray(signal, dtype=float),
        "r_peaks":        np.array([], dtype=int),
        "delineation":    {"per_beat": [], "summary": {}, "method": "none"},
        "features":       {k: None for k in FEATURE_NAMES_V3},
        "feature_vector": np.zeros(n, dtype=np.float32),
        "sqi":            sqi,
        "sqi_issues":     issues,
        "method":         "none",
        "skipped":        True,
    }
