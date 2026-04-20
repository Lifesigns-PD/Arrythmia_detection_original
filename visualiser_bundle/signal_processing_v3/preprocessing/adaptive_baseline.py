"""
adaptive_baseline.py — Adaptive Baseline Wander Removal
=========================================================
V2 used a fixed Butterworth HP at 0.5 Hz.  That works but leaves slow
drift when baseline wander is non-sinusoidal (e.g. patient breathing).

V3 uses a cascade of three methods, choosing the best based on drift
characteristics:

  Method A — Butterworth HP (fast, good for sinusoidal drift)
  Method B — Savitzky-Golay low-order smoothing + subtraction
             (better for polynomial / asymmetric drift)
  Method C — Morphological opening (best for sharp spikes + slow drift)

Selection rule:
  - If drift power < threshold → Method A (simple, sufficient)
  - If drift is smooth / low-frequency → Method B
  - If drift has spikes              → Method C
  - Final output is weighted average (A×0.3 + best×0.7)
"""

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.ndimage import grey_opening


# ── Threshold: drift is "significant" if low-freq power exceeds this fraction ──
DRIFT_POWER_THRESHOLD = 0.08   # 8 % of total signal power


def remove_baseline_adaptive(signal: np.ndarray, fs: int = 125) -> np.ndarray:
    """
    Remove baseline wander adaptively.

    BBB guard: estimates the current heart rate from the raw signal via
    autocorrelation-based peak detection and caps the SG/morphological
    filter windows at 80% of the median RR interval.  This prevents a
    filter window that is longer than one RR from treating a wide BBB
    QRS complex (120–180 ms) as part of the "drifting baseline" and
    subtracting it.

    Parameters
    ----------
    signal : np.ndarray   1-D ECG (mV)
    fs     : int          Sampling rate

    Returns
    -------
    np.ndarray  — baseline-corrected signal, same length
    """
    if len(signal) < 4:
        return signal.copy()

    signal = signal.astype(np.float64)

    # ── Estimate median RR for dynamic window sizing (BBB guard) ──────────────
    # This is a quick rough estimate using simple peak detection; it does NOT
    # need to be accurate — it only prevents the window from exceeding one RR.
    median_rr_samples = _estimate_median_rr(signal, fs)

    # Estimate drift power (energy below 0.5 Hz)
    drift_ratio = _low_freq_power_ratio(signal, fs, cutoff=0.5)

    if drift_ratio < DRIFT_POWER_THRESHOLD:
        # Minimal drift — simple HP filter is enough
        return _butterworth_hp(signal, fs, cutoff=0.5)

    # Significant drift — choose based on signal characteristic
    smoothness = _estimate_drift_smoothness(signal, fs)

    baseline_sg    = _savgol_baseline(signal, fs, median_rr_samples)
    baseline_morph = _morphological_baseline(signal, fs, median_rr_samples)

    # Smooth drift → SG is more accurate; spiky drift → morphological is better
    w_sg    = smoothness          # 0.0–1.0 (1.0 = very smooth)
    w_morph = 1.0 - smoothness

    combined_baseline = w_sg * baseline_sg + w_morph * baseline_morph
    corrected = signal - combined_baseline

    # Final HP to catch any residual
    corrected = _butterworth_hp(corrected, fs, cutoff=0.15)

    return corrected.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _butterworth_hp(signal: np.ndarray, fs: int, cutoff: float = 0.5) -> np.ndarray:
    nyq = 0.5 * fs
    norm = max(cutoff / nyq, 0.001)
    norm = min(norm, 0.999)
    b, a = butter(3, norm, btype="high")
    return filtfilt(b, a, signal)


def _estimate_median_rr(signal: np.ndarray, fs: int) -> int:
    """
    Quick rough estimate of median RR interval (in samples) via simple peak
    detection.  Used only to cap the baseline filter window so it does not
    exceed one cardiac cycle — prevents wide BBB QRS from being mistaken for
    slow baseline drift.  Returns None when estimation fails.
    """
    try:
        from scipy.signal import find_peaks
        # Rough threshold: 50% of max absolute amplitude
        threshold = 0.5 * float(np.max(np.abs(signal)))
        if threshold < 1e-6:
            return None
        peaks, _ = find_peaks(signal, height=threshold,
                               distance=max(int(0.25 * fs), 1))
        if len(peaks) < 3:
            return None
        rr = np.diff(peaks)
        rr_valid = rr[(rr > int(0.2 * fs)) & (rr < int(3.0 * fs))]
        if len(rr_valid) < 2:
            return None
        return int(np.median(rr_valid))
    except Exception:
        return None


def _savgol_baseline(
    signal: np.ndarray,
    fs: int,
    median_rr_samples: int = None,
) -> np.ndarray:
    """
    Estimate baseline using a wide Savitzky-Golay smoother.

    Window is normally 2 s but is capped at 80% of the median RR interval
    (minimum 1 s) so that a wide BBB QRS (120–180 ms) is always much shorter
    than the filter window and is never treated as slow baseline drift.
    """
    # Base window = 2 s
    window = int(fs * 2.0)
    # BBB guard: cap at 80% of median RR, but keep at least 1 s
    if median_rr_samples is not None:
        max_win = max(int(median_rr_samples * 0.8), int(fs * 1.0))
        window  = min(window, max_win)
    if window % 2 == 0:
        window += 1
    window = max(window, 5)
    # Clamp to signal length
    window = min(window, len(signal) - 2 if len(signal) % 2 == 0 else len(signal) - 1)
    if window < 5:
        return np.zeros_like(signal)
    try:
        return savgol_filter(signal, window_length=window, polyorder=2)
    except Exception:
        return np.zeros_like(signal)


def _morphological_baseline(
    signal: np.ndarray,
    fs: int,
    median_rr_samples: int = None,
) -> np.ndarray:
    """
    Estimate baseline via morphological opening (erosion + dilation).

    Structuring element is normally 600 ms but is capped at 50% of the
    median RR so the SE never spans an entire cardiac cycle (which would
    cause it to include the QRS in the morphological erosion and distort
    the baseline estimate for wide BBB complexes).
    """
    # Base SE = 600 ms
    se_size = max(int(fs * 0.6), 3)
    # BBB guard: cap at 50% of median RR, but keep at least 100 ms
    if median_rr_samples is not None:
        max_se = max(int(median_rr_samples * 0.5), int(fs * 0.1))
        se_size = min(se_size, max_se)
    try:
        opened = grey_opening(signal, size=se_size)
        # Smooth the opened signal
        win = min(int(fs * 0.4), len(opened) - 1)
        if win % 2 == 0:
            win += 1
        win = max(win, 5)
        return savgol_filter(opened, window_length=win, polyorder=2)
    except Exception:
        return np.zeros_like(signal)


def _low_freq_power_ratio(signal: np.ndarray, fs: int, cutoff: float) -> float:
    """Fraction of signal power below `cutoff` Hz."""
    try:
        from scipy.signal import welch
        f, pxx = welch(signal, fs=fs, nperseg=min(512, len(signal) // 2))
        total = np.sum(pxx) + 1e-12
        low   = np.sum(pxx[f < cutoff])
        return float(low / total)
    except Exception:
        return 0.0


def _estimate_drift_smoothness(signal: np.ndarray, fs: int) -> float:
    """
    Returns 0–1 where 1 = very smooth drift, 0 = spiky drift.
    Based on ratio of power in 0.05–0.5 Hz vs 0.5–2 Hz.
    """
    try:
        from scipy.signal import welch
        f, pxx = welch(signal, fs=fs, nperseg=min(512, len(signal) // 2))
        slow  = np.sum(pxx[(f >= 0.05) & (f < 0.5)]) + 1e-12
        fast  = np.sum(pxx[(f >= 0.5)  & (f < 2.0)]) + 1e-12
        ratio = slow / (slow + fast)
        return float(np.clip(ratio, 0.0, 1.0))
    except Exception:
        return 0.5
