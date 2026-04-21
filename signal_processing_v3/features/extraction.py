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

# ── RHYTHM model: 36 features (no beat discriminators; nonlinear trimmed to 4) ──
# Rationale: rhythm model classifies background rhythms (AF, BBB, AV blocks) —
# it does not need per-beat ectopy discriminators. Noisy nonlinear features
# (Hurst, DFA, approx/permutation entropy) need minutes of data; on 10s windows
# they add noise, not signal.
FEATURE_NAMES_RHYTHM = [
    # HRV Time Domain — all 11 (RR regularity is critical for AF, AV blocks)
    "mean_rr_ms", "sdnn_ms", "rmssd_ms", "pnn50", "pnn20",
    "rr_range_ms", "rr_cv", "rr_skewness", "rr_kurtosis",
    "triangular_index", "mean_hr_bpm",
    # HRV Frequency — all 8 (spectral entropy separates AF from sinus)
    "vlf_power", "lf_power", "hf_power", "lf_hf_ratio",
    "lf_norm", "hf_norm", "total_power", "spectral_entropy_hrv",
    # Nonlinear — 4 meaningful on 10s windows (drop hurst, approx/permutation entropy)
    "sample_entropy", "dfa_alpha1", "sd1", "sd2",
    # Morphology — all 13 (QRS width=BBB, PR=AV block, QTc=risk, P wave=AF)
    "qrs_duration_ms", "qrs_area", "pr_interval_ms", "qt_interval_ms",
    "qtc_bazett_ms", "st_elevation_mv", "st_slope", "t_wave_asymmetry",
    "r_s_ratio", "p_wave_duration_ms", "t_wave_amplitude_mv",
    "r_amplitude_mv", "qrs_amplitude_ms_product",
]  # 36 features total

# ── ECTOPY model: 47 features (all beat discriminators; HRV/nonlinear trimmed) ──
# Rationale: ectopy model classifies individual beats (PVC vs PAC vs None) —
# beat discriminators were specifically engineered for this task and are GOLD.
# Redundant HRV features (pnn20≈pnn50, sd1_sd2_ratio≈sd1/sd2) and noisy nonlinear
# features are dropped to reduce noise without losing clinical information.
FEATURE_NAMES_ECTOPY = [
    # HRV Time Domain — 7 (drop pnn20, rr_skewness, rr_kurtosis, triangular_index)
    "mean_rr_ms", "sdnn_ms", "rmssd_ms", "pnn50",
    "rr_range_ms", "rr_cv", "mean_hr_bpm",
    # HRV Frequency — 4 key discriminators (drop vlf/lf/hf raw powers, keep ratios)
    "lf_hf_ratio", "spectral_entropy_hrv", "hf_norm", "total_power",
    # Nonlinear — 3 (sd1/sd2 capture Poincaré scatter; sample_entropy = beat regularity)
    "sd1", "sd2", "sample_entropy",
    # Morphology — all 13 (QRS duration=PVC wide, PR=PAC, P wave=PAC precursor)
    "qrs_duration_ms", "qrs_area", "pr_interval_ms", "qt_interval_ms",
    "qtc_bazett_ms", "st_elevation_mv", "st_slope", "t_wave_asymmetry",
    "r_s_ratio", "p_wave_duration_ms", "t_wave_amplitude_mv",
    "r_amplitude_mv", "qrs_amplitude_ms_product",
    # Beat Discriminators — all 20 (purpose-built for PVC/PAC separation)
    "qrs_wide_fraction", "mean_qrs_duration_ms", "qrs_duration_std_ms",
    "p_absent_fraction", "p_inverted_fraction", "p_biphasic_fraction",
    "mean_coupling_ratio", "short_coupling_fraction",
    "compensatory_pause_fraction",
    "t_discordant_fraction", "t_inverted_fraction",
    "qrs_negative_fraction",
    "mean_rs_ratio", "rs_ratio_std",
    "mean_q_depth", "pathological_q_fraction",
    "mean_s_depth",
    "delta_wave_fraction",
    "pvc_score_mean", "pac_score_mean",
]  # 47 features total


def _get_feature_names_for_task(task: str) -> List[str]:
    """Return the canonical feature name list for the given task."""
    if task == "rhythm":
        return FEATURE_NAMES_RHYTHM
    elif task == "ectopy":
        return FEATURE_NAMES_ECTOPY
    return FEATURE_NAMES_V3  # fallback: full 60


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

    # Suppress numpy divide/invalid warnings from feature computations —
    # all divisions are already guarded; warnings arise only on edge-case
    # segments (< 3 R-peaks, flat signal) and are safely returned as None.
    with np.errstate(divide='ignore', invalid='ignore'):
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
    Convert feature dict → fixed-length float32 numpy vector (full 60 features).
    None values are replaced with fill_none (default 0).
    """
    vec = [
        float(feature_dict.get(k) or fill_none)
        for k in FEATURE_NAMES_V3
    ]
    return np.array(vec, dtype=np.float32)


def feature_dict_to_vector_task(
    feature_dict: Dict[str, Optional[float]],
    task: str = "rhythm",
    fill_none: float = 0.0,
) -> np.ndarray:
    """
    Convert feature dict → task-specific float32 vector.
      task="rhythm" → 36 features
      task="ectopy" → 47 features
    None values replaced with fill_none (default 0).
    """
    names = _get_feature_names_for_task(task)
    vec = [float(feature_dict.get(k) or fill_none) for k in names]
    return np.array(vec, dtype=np.float32)
