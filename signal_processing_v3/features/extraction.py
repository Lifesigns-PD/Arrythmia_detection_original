"""
extraction.py — Master V3 Feature Extractor
============================================
Aggregates all 60 features from 5 domains:
  1. HRV Time Domain   (11 features)
  2. HRV Frequency     ( 8 features)
  3. Nonlinear HRV     ( 8 features)
  4. Morphology        (13 features)
  5. Beat Discriminators (20 features) ← PVC/PAC/polarity/delta/T-inversion

Total: 60 named features. All values are float or None.
Missing values are filled with 0 for model use.
"""

import numpy as np
from typing import Dict, List, Optional, Any

from .hrv_time_domain     import compute_hrv_time_domain
from .hrv_frequency       import compute_hrv_frequency
from .nonlinear           import compute_nonlinear_features
from .morphology_features import compute_morphology_features
from .beat_morphology     import compute_beat_discriminators, BEAT_DISC_FEATURES


# Canonical feature name list (order is stable — do NOT reorder)
FEATURE_NAMES_V3 = [
    # ── HRV time domain (11) ──────────────────────────────────────────────────
    "mean_rr_ms", "sdnn_ms", "rmssd_ms", "pnn50", "pnn20",
    "rr_range_ms", "rr_cv", "rr_skewness", "rr_kurtosis",
    "triangular_index", "mean_hr_bpm",
    # ── HRV frequency (8) ────────────────────────────────────────────────────
    "vlf_power", "lf_power", "hf_power", "lf_hf_ratio",
    "lf_norm", "hf_norm", "total_power", "spectral_entropy_hrv",
    # ── Nonlinear (8) ────────────────────────────────────────────────────────
    "sample_entropy", "approx_entropy", "permutation_entropy",
    "hurst_exponent", "dfa_alpha1", "sd1", "sd2", "sd1_sd2_ratio",
    # ── Morphology (13) ──────────────────────────────────────────────────────
    "qrs_duration_ms", "qrs_area", "pr_interval_ms", "qt_interval_ms",
    "qtc_bazett_ms", "st_elevation_mv", "st_slope", "t_wave_asymmetry",
    "r_s_ratio", "p_wave_duration_ms", "t_wave_amplitude_mv",
    "r_amplitude_mv", "qrs_amplitude_ms_product",
    # ── Beat discriminators (20) — PVC/PAC/polarity/delta/T-inversion ────────
] + BEAT_DISC_FEATURES


def extract_features_v3(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    delineation: Dict[str, Any],
    fs: int = 125,
) -> Dict[str, Optional[float]]:
    """
    Extract full V3 feature dict (60 features).

    Parameters
    ----------
    signal       : preprocessed 1-D ECG
    r_peaks      : R-peak indices
    delineation  : output of delineate_v3() — must contain "per_beat" key
    fs           : sampling rate

    Returns
    -------
    dict with all FEATURE_NAMES_V3 keys; values are float or None
    """
    per_beat = delineation.get("per_beat", []) if delineation else []

    hrv_t  = compute_hrv_time_domain(r_peaks, fs)
    hrv_f  = compute_hrv_frequency(r_peaks, fs)
    nonlin = compute_nonlinear_features(r_peaks, fs)
    morph  = compute_morphology_features(signal, per_beat, r_peaks, fs)
    beat_d = compute_beat_discriminators(signal, per_beat, r_peaks, fs)

    combined = {}
    combined.update(hrv_t)
    combined.update(hrv_f)
    combined.update(nonlin)
    combined.update(morph)
    combined.update(beat_d)

    return {k: combined.get(k) for k in FEATURE_NAMES_V3}


def feature_dict_to_vector(
    feature_dict: Dict[str, Optional[float]],
    fill_none: float = 0.0,
) -> np.ndarray:
    """
    Convert feature dict → fixed-length float32 numpy vector.
    None values are replaced with fill_none (default 0).
    """
    vec = [
        float(feature_dict.get(k) or fill_none)
        for k in FEATURE_NAMES_V3
    ]
    return np.array(vec, dtype=np.float32)
