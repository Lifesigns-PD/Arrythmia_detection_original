"""
hrv_frequency.py — HRV Frequency-Domain Features
==================================================
Uses Welch PSD on uniformly resampled RR series to compute
VLF / LF / HF band powers and their ratios.

Standard HRV frequency bands (Task Force, 1996):
  VLF : 0.003 – 0.04 Hz
  LF  : 0.04  – 0.15 Hz
  HF  : 0.15  – 0.40 Hz
"""

import numpy as np
from scipy.integrate import trapezoid
from typing import Dict, Optional


_VLF = (0.003, 0.04)
_LF  = (0.04,  0.15)
_HF  = (0.15,  0.40)
_RESAMP_HZ = 4.0   # Interpolation rate for RR series


def compute_hrv_frequency(r_peaks: np.ndarray, fs: int = 125) -> Dict[str, Optional[float]]:
    """
    Compute frequency-domain HRV features.

    Returns
    -------
    dict with keys:
      vlf_power, lf_power, hf_power, lf_hf_ratio,
      lf_norm, hf_norm, total_power, spectral_entropy_hrv
    """
    keys = ["vlf_power", "lf_power", "hf_power", "lf_hf_ratio",
            "lf_norm", "hf_norm", "total_power", "spectral_entropy_hrv"]
    empty = {k: None for k in keys}

    if r_peaks is None or len(r_peaks) < 5:
        return empty

    rr_s = np.diff(r_peaks).astype(float) / fs          # RR in seconds
    rr_times = np.cumsum(rr_s)                            # Cumulative time axis

    # Filter physiological RR
    valid = (rr_s > 0.25) & (rr_s < 2.5)
    if valid.sum() < 4:
        return empty
    rr_s = rr_s[valid]
    rr_times = rr_times[valid]

    # Uniformly resample
    try:
        from scipy.interpolate import interp1d
        from scipy.signal import welch
    except ImportError:
        return empty

    t_uniform = np.arange(rr_times[0], rr_times[-1], 1.0 / _RESAMP_HZ)
    if len(t_uniform) < 16:
        return empty

    interp = interp1d(rr_times, rr_s, kind="cubic", bounds_error=False,
                      fill_value=(rr_s[0], rr_s[-1]))
    rr_uniform = interp(t_uniform)

    # Welch PSD
    nperseg = min(len(rr_uniform), 256)
    freqs, psd = welch(rr_uniform, fs=_RESAMP_HZ, nperseg=nperseg, scaling="density")

    def band_power(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        if mask.sum() == 0:
            return 0.0
        return float(trapezoid(psd[mask], freqs[mask]))

    vlf   = band_power(*_VLF)
    lf    = band_power(*_LF)
    hf    = band_power(*_HF)
    total = vlf + lf + hf

    lf_hf   = float(lf / hf)     if hf > 0 else None
    lf_norm = float(lf / (lf+hf)) if (lf+hf) > 0 else None
    hf_norm = float(hf / (lf+hf)) if (lf+hf) > 0 else None

    # Spectral entropy of full PSD
    mask_all = (freqs >= _VLF[0]) & (freqs <= _HF[1])
    psd_norm = psd[mask_all]
    if psd_norm.sum() > 0:
        p = psd_norm / psd_norm.sum()
        p = p[p > 0]
        spec_entropy = float(-np.sum(p * np.log2(p)) / np.log2(len(p))) if len(p) > 1 else None
    else:
        spec_entropy = None

    return {
        "vlf_power":           vlf,
        "lf_power":            lf,
        "hf_power":            hf,
        "lf_hf_ratio":         lf_hf,
        "lf_norm":             lf_norm,
        "hf_norm":             hf_norm,
        "total_power":         total,
        "spectral_entropy_hrv": spec_entropy,
    }
