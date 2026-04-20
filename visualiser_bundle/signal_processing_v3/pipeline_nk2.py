"""
pipeline_nk2.py — NeuroKit2-based ECG Processing Pipeline
==========================================================
Replaces the custom wavelet delineation with NeuroKit2 (DWT method).
Feature extraction (60 features) and output format are UNCHANGED —
backfill_features.py and retrain_v2.py work without modification.

Old custom pipeline files are kept in their original locations as backup.
To revert: change __init__.py to import process_ecg_v3 from the old modules.
"""

import warnings
import numpy as np
import neurokit2 as nk
from typing import Dict, Any, List, Optional

from .detection.ensemble    import refine_peaks_subsample
from .features.extraction   import extract_features_v3, feature_dict_to_vector, FEATURE_NAMES_V3
from .quality.signal_quality import compute_sqi_v3


# ── Public entry point ────────────────────────────────────────────────────────

def process_ecg_nk2(
    raw_signal: np.ndarray,
    fs: int = 125,
    min_quality: float = 0.3,
) -> Dict[str, Any]:
    """
    Full ECG processing pipeline using NeuroKit2 for cleaning, R-peak
    detection and P/Q/R/S/T delineation.

    Output format is identical to the old process_ecg_v3() so all
    downstream code (backfill, training, inference) is unchanged.

    Parameters
    ----------
    raw_signal  : 1-D raw ECG (mV)
    fs          : sampling rate in Hz
    min_quality : SQI threshold; segments below this are skipped

    Returns
    -------
    dict with keys:
      signal, r_peaks, delineation, features, feature_vector,
      sqi, sqi_issues, method, skipped
    """
    raw_signal = np.asarray(raw_signal, dtype=np.float64)

    # ── Step 1: Clean signal ──────────────────────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cleaned = nk.ecg_clean(raw_signal, sampling_rate=fs, method="neurokit")
    except Exception:
        cleaned = raw_signal.copy()

    cleaned = np.asarray(cleaned, dtype=np.float64)

    # ── Step 2: Initial quality check ────────────────────────────────────────
    raw_sqi, raw_issues = compute_sqi_v3(cleaned, fs=fs)
    if raw_sqi < min_quality:
        return _skipped(cleaned, raw_sqi, raw_issues)

    # ── Step 3: R-peak detection ──────────────────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, r_info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit")
        r_peaks = np.array(r_info["ECG_R_Peaks"], dtype=int)
    except Exception:
        r_peaks = np.array([], dtype=int)

    if len(r_peaks) < 2:
        return _skipped(cleaned, raw_sqi, raw_issues)

    # ── Step 4: SQI with detected peaks ──────────────────────────────────────
    sqi, issues = compute_sqi_v3(cleaned, r_peaks=r_peaks, fs=fs)
    if sqi < min_quality:
        return _skipped(cleaned, sqi, issues)

    # ── Step 5: Sub-sample R-peak refinement (HRV precision) ─────────────────
    # NK2 returns integer peaks. Parabolic interpolation gives float positions
    # with sub-ms precision — HRV metrics (RMSSD, LF/HF) are more accurate.
    # Integer peaks are kept for all sample-index operations.
    r_peaks_float = refine_peaks_subsample(cleaned, r_peaks)

    # ── Step 6: Delineate P/Q/R/S/T ──────────────────────────────────────────
    per_beat   = _delineate_nk2(cleaned, r_peaks, fs)
    delineation = {"per_beat": per_beat, "summary": {}, "method": "nk2_dwt"}

    # ── Step 7: Extract 60 features (same code as before) ────────────────────
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
        "method":         "nk2_dwt",
        "skipped":        False,
    }


# ── NK2 delineation → per_beat converter ─────────────────────────────────────

def _delineate_nk2(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: int,
) -> List[Dict]:
    """
    Run NK2 DWT delineation and convert output to the per_beat list format
    expected by extract_features_v3() / morphology_features.py.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, waves = nk.ecg_delineate(
                signal,
                r_peaks,
                sampling_rate=fs,
                method="dwt",
                show=False,
            )
    except Exception:
        return _empty_beats(len(r_peaks))

    return _nk2_waves_to_per_beat(waves, r_peaks, signal)


def _nk2_waves_to_per_beat(
    waves: Dict,
    r_peaks: np.ndarray,
    signal: np.ndarray,
) -> List[Dict]:
    """
    Convert NK2 waves dict (per-beat indexed arrays) to the per_beat list
    format used by morphology_features.py and beat_morphology.py.

    NK2 key mapping:
      ECG_P_Peaks / ECG_P_Onsets / ECG_P_Offsets  → p_peak / p_onset / p_offset
      ECG_Q_Peaks                                  → q_peak
      ECG_S_Peaks                                  → s_peak
      ECG_T_Peaks / ECG_T_Onsets / ECG_T_Offsets  → t_peak / t_onset / t_offset
      ECG_R_Onsets / ECG_R_Offsets                 → qrs_onset / qrs_offset
    """

    def _idx(key: str, i: int) -> Optional[int]:
        """Safely get beat i's index from a NK2 waves array. Returns None if absent."""
        arr = waves.get(key)
        if arr is None or i >= len(arr):
            return None
        val = arr[i]
        if val is None:
            return None
        try:
            f = float(val)
            if np.isnan(f):
                return None
            idx = int(f)
            return idx if 0 <= idx < len(signal) else None
        except (TypeError, ValueError):
            return None

    per_beat = []

    for i in range(len(r_peaks)):
        r = int(r_peaks[i])

        p_peak     = _idx("ECG_P_Peaks",   i)
        p_onset    = _idx("ECG_P_Onsets",  i)
        p_offset   = _idx("ECG_P_Offsets", i)
        q_peak     = _idx("ECG_Q_Peaks",   i)
        s_peak     = _idx("ECG_S_Peaks",   i)
        t_peak     = _idx("ECG_T_Peaks",   i)
        t_onset    = _idx("ECG_T_Onsets",  i)
        t_offset   = _idx("ECG_T_Offsets", i)
        qrs_onset  = _idx("ECG_R_Onsets",  i)
        qrs_offset = _idx("ECG_R_Offsets", i)

        # ── P morphology ──────────────────────────────────────────────────────
        # NK2 DWT correctly sets P_Peaks to NaN in AF (no true P-wave).
        # We add an amplitude floor (0.04 mV) to reject residual noise peaks.
        if p_peak is None:
            p_morphology = "absent"
        else:
            p_amp = float(signal[p_peak])
            if abs(p_amp) < 0.04:
                p_morphology = "absent"
                p_peak = p_onset = p_offset = None
            elif p_amp < -0.04:
                p_morphology = "inverted"
            else:
                p_morphology = "normal"

        # ── QRS polarity ──────────────────────────────────────────────────────
        r_amp = float(signal[r]) if 0 <= r < len(signal) else 0.0
        qrs_polarity = "negative" if r_amp < -0.05 else "positive"

        # ── T-wave inversion ─────────────────────────────────────────────────
        t_inverted = False
        if t_peak is not None:
            t_inverted = bool(float(signal[t_peak]) < -0.05)

        # ── Q / S depths ─────────────────────────────────────────────────────
        q_depth = float(signal[q_peak]) if q_peak is not None else None
        s_depth = float(signal[s_peak]) if s_peak is not None else None

        per_beat.append({
            "p_onset":      p_onset,
            "p_peak":       p_peak,
            "p_offset":     p_offset,
            "p_morphology": p_morphology,
            "qrs_onset":    qrs_onset,
            "qrs_offset":   qrs_offset,
            "q_peak":       q_peak,
            "s_peak":       s_peak,
            "t_onset":      t_onset,
            "t_peak":       t_peak,
            "t_offset":     t_offset,
            "qrs_polarity": qrs_polarity,
            "t_inverted":   t_inverted,
            "delta_wave":   False,   # NK2 DWT does not detect delta waves
            "q_depth":      q_depth,
            "s_depth":      s_depth,
        })

    return per_beat


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty_beats(n: int) -> List[Dict]:
    """Return n beats with all fiducials absent — used on delineation failure."""
    return [
        {
            "p_onset": None, "p_peak": None, "p_offset": None,
            "p_morphology": "absent",
            "qrs_onset": None, "qrs_offset": None,
            "q_peak": None, "s_peak": None,
            "t_onset": None, "t_peak": None, "t_offset": None,
            "qrs_polarity": "positive",
            "t_inverted": False, "delta_wave": False,
            "q_depth": None, "s_depth": None,
        }
        for _ in range(n)
    ]


def _skipped(signal, sqi, issues):
    n = len(FEATURE_NAMES_V3)
    return {
        "signal":         np.asarray(signal, dtype=float),
        "r_peaks":        np.array([], dtype=int),
        "delineation":    {"per_beat": [], "summary": {}, "method": "none"},
        "features":       {k: None for k in FEATURE_NAMES_V3},
        "feature_vector": np.zeros(n, dtype=np.float32),
        "sqi":            0.0,
        "sqi_issues":     issues,
        "method":         "none",
        "skipped":        True,
    }
