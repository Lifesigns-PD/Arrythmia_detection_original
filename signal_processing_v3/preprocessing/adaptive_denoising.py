"""
adaptive_denoising.py — Adaptive Powerline + High-Freq Noise Removal
=====================================================================
V2 blindly applied notch filters at both 50 Hz and 60 Hz, which can
over-filter and distort the T-wave at low sampling rates.

V3 improvements:
  1. Auto-detect dominant powerline frequency (50 vs 60 Hz) from PSD
  2. Apply notch only to the detected frequency + harmonics (if present)
  3. Final Butterworth LP at 45 Hz (steeper rolloff than V2's order-2)
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch


def remove_noise_adaptive(signal: np.ndarray, fs: int = 125) -> np.ndarray:
    """
    Adaptive powerline noise removal + anti-alias LP filter.

    Parameters
    ----------
    signal : np.ndarray  1-D ECG (mV)
    fs     : int         Sampling rate

    Returns
    -------
    np.ndarray  — denoised signal
    """
    if len(signal) < 4:
        return signal.copy()

    sig = signal.astype(np.float64)

    # 1. Detect dominant powerline frequency
    pl_freq = _detect_powerline_frequency(sig, fs)

    # 2. Notch at detected frequency (and harmonics if present)
    if pl_freq is not None:
        sig = _apply_notch(sig, pl_freq, fs, Q=35)
        # 2nd harmonic (100 or 120 Hz) — only applies if below Nyquist
        harmonic = pl_freq * 2
        if harmonic < 0.45 * fs:
            sig = _apply_notch(sig, harmonic, fs, Q=35)

    # 3. Final LP at 45 Hz (order 4 — steeper than V2's order 2)
    sig = _lowpass(sig, fs, cutoff=45.0, order=4)

    return sig.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_powerline_frequency(signal: np.ndarray, fs: int) -> float | None:
    """
    Returns 50.0, 60.0, or None (no significant powerline interference).
    Decision based on PSD peak power in narrow bands around 50 and 60 Hz.
    """
    if fs < 110:          # can't see 50/60 Hz if Nyquist < 55 Hz
        return None
    try:
        f, pxx = welch(signal, fs=fs, nperseg=min(1024, len(signal) // 2))
        def band_power(center, half_width=2.0):
            mask = (f >= center - half_width) & (f <= center + half_width)
            return float(np.sum(pxx[mask]))

        p50 = band_power(50.0)
        p60 = band_power(60.0)
        threshold = 0.01 * np.sum(pxx)  # must be at least 1% of total power

        if max(p50, p60) < threshold:
            return None          # no significant interference
        return 50.0 if p50 >= p60 else 60.0
    except Exception:
        return None


def _apply_notch(signal: np.ndarray, f0: float, fs: int, Q: float = 35) -> np.ndarray:
    """Apply IIR notch filter at f0 Hz with quality factor Q."""
    if f0 >= 0.5 * fs:
        return signal
    try:
        b, a = iirnotch(f0, Q, fs=fs)
        return filtfilt(b, a, signal)
    except Exception:
        return signal


def _lowpass(signal: np.ndarray, fs: int, cutoff: float = 45.0, order: int = 4) -> np.ndarray:
    """Butterworth LP filter."""
    nyq = 0.5 * fs
    if cutoff >= nyq:
        return signal
    norm = cutoff / nyq
    norm = min(norm, 0.999)
    b, a = butter(order, norm, btype="low")
    return filtfilt(b, a, signal)
