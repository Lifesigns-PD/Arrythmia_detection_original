"""
quality_check.py — Pre-flight ECG Signal Quality Assessment
============================================================
Runs BEFORE any processing. Identifies signals too degraded to use.

Checks:
  1. Flatline (no activity)
  2. Signal saturation / clipping
  3. Excessive high-frequency noise
  4. NaN / Inf values
  5. Insufficient length
  6. Extreme amplitude (disconnected lead)

Returns a quality_score (0–1) and a list of issue strings.
"""

import numpy as np
from scipy.signal import welch
from typing import Tuple, List


# ─────────────────────────────────────────────
# Thresholds (can be overridden per call)
# ─────────────────────────────────────────────
FLATLINE_STD_THRESHOLD   = 0.005   # mV — std below this = flatline
SATURATION_THRESHOLD_MV  = 5.0     # mV — |amplitude| above this likely saturated
CLIP_RATIO_THRESHOLD     = 0.05    # fraction of samples at exact min/max = clipping
HF_NOISE_RATIO_THRESHOLD = 0.60    # fraction of power above 40 Hz = noisy
MIN_SIGNAL_LENGTH_SEC    = 3.0     # seconds — shorter segments unreliable
DISCONNECT_THRESHOLD_MV  = 8.0     # mV — peak-to-peak above this = lead-off


def assess_signal_quality(
    signal: np.ndarray,
    fs: int = 125,
) -> Tuple[float, List[str]]:
    """
    Pre-flight quality assessment.

    Parameters
    ----------
    signal : np.ndarray  — 1-D ECG in mV
    fs     : int         — sampling rate

    Returns
    -------
    quality_score : float  — 0.0 (unusable) to 1.0 (excellent)
    issues        : list   — human-readable issue strings
    """
    issues: List[str] = []

    # ── 1. NaN / Inf ──
    if np.any(~np.isfinite(signal)):
        issues.append("nan_or_inf_values")
        # Cannot do further checks meaningfully
        return 0.0, issues

    # ── 2. Length check ──
    min_samples = int(MIN_SIGNAL_LENGTH_SEC * fs)
    if len(signal) < min_samples:
        issues.append(f"signal_too_short ({len(signal)} < {min_samples} samples)")
        return 0.0, issues

    # ── 3. Flatline ──
    if np.std(signal) < FLATLINE_STD_THRESHOLD:
        issues.append("flatline")

    # ── 4. Saturation / disconnected lead ──
    if np.max(np.abs(signal)) > SATURATION_THRESHOLD_MV:
        issues.append("signal_saturated")

    if (np.max(signal) - np.min(signal)) > DISCONNECT_THRESHOLD_MV:
        issues.append("lead_disconnection")

    # ── 5. Clipping detection ──
    sig_min, sig_max = signal.min(), signal.max()
    clip_low  = np.mean(signal == sig_min)
    clip_high = np.mean(signal == sig_max)
    if (clip_low + clip_high) > CLIP_RATIO_THRESHOLD:
        issues.append("signal_clipped")

    # ── 6. High-frequency noise ratio ──
    try:
        f, pxx = welch(signal, fs=fs, nperseg=min(256, len(signal) // 4))
        total_power = np.sum(pxx) + 1e-12
        hf_power    = np.sum(pxx[f > 40])
        hf_ratio    = hf_power / total_power
        if hf_ratio > HF_NOISE_RATIO_THRESHOLD:
            issues.append(f"high_frequency_noise (hf_ratio={hf_ratio:.2f})")
    except Exception:
        pass

    # ── Score ──
    # Each issue costs 0.2; minimum 0.0
    score = max(0.0, 1.0 - 0.2 * len(issues))
    return round(score, 3), issues


def is_usable(signal: np.ndarray, fs: int = 125, min_score: float = 0.4) -> bool:
    """Returns True if the signal quality is above min_score."""
    score, _ = assess_signal_quality(signal, fs)
    return score >= min_score
