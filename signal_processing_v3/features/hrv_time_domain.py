"""
hrv_time_domain.py — HRV Time-Domain Features
===============================================
Computes classical and nonlinear time-domain HRV metrics from RR intervals.
"""

import numpy as np
from typing import Dict, Optional


def compute_hrv_time_domain(r_peaks: np.ndarray, fs: int = 125) -> Dict[str, Optional[float]]:
    """
    Compute time-domain HRV features from R-peak indices.

    Parameters
    ----------
    r_peaks : array of R-peak sample indices
    fs      : sampling rate (Hz)

    Returns
    -------
    dict with keys:
      mean_rr_ms, sdnn_ms, rmssd_ms, pnn50, pnn20,
      rr_range_ms, rr_cv, rr_skewness, rr_kurtosis,
      triangular_index, mean_hr_bpm
    """
    keys = ["mean_rr_ms", "sdnn_ms", "rmssd_ms", "pnn50", "pnn20",
            "rr_range_ms", "rr_cv", "rr_skewness", "rr_kurtosis",
            "triangular_index", "mean_hr_bpm"]
    empty = {k: None for k in keys}

    if r_peaks is None or len(r_peaks) < 3:
        return empty

    rr = np.diff(r_peaks).astype(float) / fs * 1000  # ms
    # Filter physiological range
    rr = rr[(rr > 250) & (rr < 2500)]
    if len(rr) < 2:
        return empty

    mean_rr = float(np.mean(rr))
    sdnn    = float(np.std(rr, ddof=1)) if len(rr) > 1 else None
    diff_rr = np.diff(rr)
    rmssd   = float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) > 0 else None
    pnn50   = float(np.mean(np.abs(diff_rr) > 50)) if len(diff_rr) > 0 else None
    pnn20   = float(np.mean(np.abs(diff_rr) > 20)) if len(diff_rr) > 0 else None
    rr_range = float(np.max(rr) - np.min(rr))
    rr_cv   = float(sdnn / mean_rr) if sdnn is not None and mean_rr > 0 else None

    from scipy.stats import skew, kurtosis
    rr_skew = float(skew(rr)) if len(rr) >= 4 else None
    rr_kurt = float(kurtosis(rr)) if len(rr) >= 4 else None

    # Triangular index: total count / height of histogram
    tri_idx = None
    if len(rr) >= 8:
        hist, _ = np.histogram(rr, bins=max(4, len(rr) // 4))
        if hist.max() > 0:
            tri_idx = float(len(rr) / hist.max())

    mean_hr = float(60000 / mean_rr) if mean_rr > 0 else None

    return {
        "mean_rr_ms":        mean_rr,
        "sdnn_ms":           sdnn,
        "rmssd_ms":          rmssd,
        "pnn50":             pnn50,
        "pnn20":             pnn20,
        "rr_range_ms":       rr_range,
        "rr_cv":             rr_cv,
        "rr_skewness":       rr_skew,
        "rr_kurtosis":       rr_kurt,
        "triangular_index":  tri_idx,
        "mean_hr_bpm":       mean_hr,
    }
