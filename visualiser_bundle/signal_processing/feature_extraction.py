"""
feature_extraction.py — Signal-Only ECG Feature Extraction for Training
========================================================================

Extracts numeric features from an ECG signal (in millivolts) that can be
used as auxiliary inputs alongside the raw waveform.

ALL features are derived ONLY from the ECG signal itself:
  - No patient metadata (age, sex, BMI) is used or required.
  - Units: signal in mV, durations in ms, amplitudes in mV, rates in bpm.

Feature Vector (FEATURE_NAMES defines the canonical order):
  0  mean_hr_bpm          — Average heart rate from R-R intervals
  1  hr_std_bpm           — Heart-rate variability (std of beat-to-beat HR)
  2  sdnn_ms              — SDNN: standard deviation of R-R intervals
  3  rmssd_ms             — RMSSD: root-mean-square of successive RR diffs
  4  qrs_duration_ms      — Median QRS complex width
  5  pr_interval_ms       — Median PR interval (P-onset to QRS-onset)
  6  qtc_ms               — Median corrected QT interval (Bazett)
  7  p_wave_amplitude_mv  — Median P-wave amplitude
  8  t_wave_amplitude_mv  — Median T-wave amplitude
  9  st_deviation_mv      — Median ST-segment deviation from baseline
 10  qrs_amplitude_mv     — Median R-peak amplitude
 11  p_wave_present_ratio — Fraction of beats with detectable P-wave
 12  sqi_score            — Signal Quality Index (0–1)

Usage:
    from signal_processing.feature_extraction import extract_feature_vector, FEATURE_NAMES

    vec = extract_feature_vector(signal_1d, fs=125)
    # vec is np.ndarray of shape (13,), dtype float32
    # FEATURE_NAMES[i] tells you what vec[i] means
"""

import numpy as np
from typing import Optional

# Canonical feature order — MUST match the model's expected input dimension.
FEATURE_NAMES = [
    "mean_hr_bpm",
    "hr_std_bpm",
    "sdnn_ms",
    "rmssd_ms",
    "pnn50",        # % successive RR diffs >50ms — strong AF indicator
    "rr_cv",        # RR coefficient of variation — separates AFib from Sinus Arrhythmia
    "qrs_duration_ms",
    "pr_interval_ms",
    "qtc_ms",
    "p_wave_amplitude_mv",
    "t_wave_amplitude_mv",
    "st_deviation_mv",
    "qrs_amplitude_mv",
    "p_wave_present_ratio",
    "sqi_score",
]

NUM_FEATURES = len(FEATURE_NAMES)


def extract_feature_vector(
    signal: np.ndarray,
    fs: int = 125,
    r_peaks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract a fixed-length numeric feature vector from an ECG signal.

    Parameters
    ----------
    signal : np.ndarray
        1-D ECG signal in millivolts.
    fs : int
        Sampling rate in Hz (default 125).
    r_peaks : np.ndarray or None
        Pre-computed R-peak indices. If None, detected automatically.

    Returns
    -------
    np.ndarray
        Shape (NUM_FEATURES,), dtype float32.  Values default to 0.0
        when a feature cannot be computed (e.g. too few R-peaks).
    """
    vec = np.zeros(NUM_FEATURES, dtype=np.float32)

    if signal is None or len(signal) < fs:
        return vec

    # ------------------------------------------------------------------
    # 1. Detect R-peaks if not provided
    # ------------------------------------------------------------------
    if r_peaks is None:
        r_peaks = _detect_r_peaks(signal, fs)

    if r_peaks is None or len(r_peaks) < 2:
        # Can still compute SQI
        vec[_idx("sqi_score")] = _compute_sqi(signal, fs)
        return vec

    r_peaks = np.asarray(r_peaks, dtype=int)

    # ------------------------------------------------------------------
    # 2. RR intervals → HR features
    # ------------------------------------------------------------------
    rr_ms = np.diff(r_peaks) * (1000.0 / fs)
    rr_ms = rr_ms[rr_ms > 200]  # discard physiologically impossible (<200ms = >300bpm)
    rr_ms = rr_ms[rr_ms < 2500]  # discard pauses >2.5s

    if len(rr_ms) >= 2:
        hr_bpm = 60000.0 / rr_ms
        vec[_idx("mean_hr_bpm")]  = float(np.mean(hr_bpm))
        vec[_idx("hr_std_bpm")]   = float(np.std(hr_bpm))
        vec[_idx("sdnn_ms")]      = float(np.std(rr_ms))
        vec[_idx("rmssd_ms")]     = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
        # pNN50: fraction of successive RR differences >50ms (strong AF indicator)
        rr_diffs = np.abs(np.diff(rr_ms))
        vec[_idx("pnn50")]  = float(np.mean(rr_diffs > 50)) if len(rr_diffs) > 0 else 0.0
        # RR coefficient of variation: std/mean — separates AFib from Sinus Arrhythmia
        vec[_idx("rr_cv")]  = float(np.std(rr_ms) / np.mean(rr_ms)) if np.mean(rr_ms) > 0 else 0.0
    elif len(rr_ms) == 1:
        vec[_idx("mean_hr_bpm")] = float(60000.0 / rr_ms[0])

    # ------------------------------------------------------------------
    # 3. Morphology features (via existing morphology module)
    # ------------------------------------------------------------------
    try:
        from signal_processing.morphology import extract_morphology
        morph = extract_morphology(signal, r_peaks, fs)
        summary = morph.get("summary", {})

        vec[_idx("qrs_duration_ms")]     = float(summary.get("qrs_duration_ms", 0.0))
        vec[_idx("pr_interval_ms")]      = float(summary.get("pr_interval_ms", 0.0))
        vec[_idx("qtc_ms")]              = float(summary.get("qtc_ms") or summary.get("qtc_bazett_ms", 0.0))
        vec[_idx("p_wave_amplitude_mv")] = float(summary.get("p_wave_amplitude_mv", 0.0))
        vec[_idx("t_wave_amplitude_mv")] = float(summary.get("t_wave_amplitude_mv", 0.0))
        vec[_idx("st_deviation_mv")]     = float(summary.get("st_deviation_mv", 0.0))
        vec[_idx("qrs_amplitude_mv")]    = float(summary.get("qrs_amplitude_mv", 0.0))
        vec[_idx("p_wave_present_ratio")] = float(summary.get("p_wave_present_ratio", 0.0))
    except Exception:
        # Morphology extraction failed.
        # Use neutral/uncertain defaults so the model does not misread failure
        # as a pathological finding:
        #   p_wave_present_ratio = 0.5  (uncertain, NOT 0.0 which looks like
        #                                "no P-waves" and biases toward PVC/PAC)
        #   pr_interval_ms       = 0.0  (kept — 0 is filtered by rules anyway)
        #   qrs_duration_ms      = 0.0  (kept — model learned 0 = unknown)
        vec[_idx("p_wave_present_ratio")] = 0.5
        # R-peak amplitude as the one reliable fallback
        valid_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < len(signal))]
        if len(valid_peaks) > 0:
            vec[_idx("qrs_amplitude_mv")] = float(np.median(signal[valid_peaks]))

    # ------------------------------------------------------------------
    # 4. Signal Quality Index
    # ------------------------------------------------------------------
    vec[_idx("sqi_score")] = _compute_sqi(signal, fs)

    # Replace any NaN/Inf with 0.0
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    return vec


def extract_feature_dict(
    signal: np.ndarray,
    fs: int = 125,
    r_peaks: Optional[np.ndarray] = None,
) -> dict:
    """
    Same as extract_feature_vector but returns a dict keyed by FEATURE_NAMES.
    Useful for storing in features_json in the database.
    """
    vec = extract_feature_vector(signal, fs, r_peaks)
    return {name: float(vec[i]) for i, name in enumerate(FEATURE_NAMES)}


# ======================================================================
# Internal helpers
# ======================================================================

# Build index lookup once
_FEATURE_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _idx(name: str) -> int:
    return _FEATURE_IDX[name]


def _detect_r_peaks(signal: np.ndarray, fs: int) -> Optional[np.ndarray]:
    """Detect R-peaks using NeuroKit2, fallback to simple peak detection."""
    try:
        import neurokit2 as nk
        _, info = nk.ecg_peaks(signal, sampling_rate=fs)
        peaks = info.get("ECG_R_Peaks", np.array([]))
        if hasattr(peaks, "tolist"):
            peaks = np.array(peaks, dtype=int)
        return peaks if len(peaks) >= 2 else None
    except Exception:
        pass

    # Fallback: simple threshold-based detection
    try:
        from scipy.signal import find_peaks
        # Use amplitude threshold at 60% of max
        threshold = 0.6 * np.max(np.abs(signal))
        min_dist = int(0.3 * fs)  # minimum 300ms between beats
        peaks, _ = find_peaks(signal, height=threshold, distance=min_dist)
        return peaks if len(peaks) >= 2 else None
    except Exception:
        return None


def _compute_sqi(signal: np.ndarray, fs: int) -> float:
    """Compute Signal Quality Index using existing SQI module."""
    try:
        from signal_processing.sqi import calculate_sqi_score
        return calculate_sqi_score(signal, fs)
    except Exception:
        return 0.0
