"""
wavelet_detector.py — Wavelet Modulus Maxima QRS Detector
===========================================================
Uses the maxima of the continuous wavelet transform (CWT) at the scale
corresponding to QRS width (~10–50 ms) to locate R-peaks.

Particularly robust for:
  - Irregular rhythms (AF, PVCs)
  - Low-amplitude signals
  - Multiple morphologies in same recording
"""

import numpy as np
from scipy.signal import find_peaks, butter, filtfilt


def detect_r_peaks_wavelet(signal: np.ndarray, fs: int = 125) -> np.ndarray:
    """
    Wavelet-based R-peak detection.

    Parameters
    ----------
    signal : np.ndarray  1-D cleaned ECG (mV)
    fs     : int         sampling rate

    Returns
    -------
    np.ndarray  — integer indices of R-peaks
    """
    if len(signal) < fs:
        return np.array([], dtype=int)

    # 1. Apply Mexican Hat wavelet at multiple scales to capture normal QRS (25 ms),
    #    wide QRS (60–100 ms), and PVC morphologies (100–150 ms).
    #    Energy is normalized by sqrt(scale) to prevent larger scales from dominating
    #    (raw CWT energy increases with scale width — without normalization, the 150 ms
    #    scale always wins and the detector fires on T-waves instead of QRS complexes).
    scales_sec = [0.025, 0.06, 0.10, 0.15]   # 25 / 60 / 100 / 150 ms
    energy = np.zeros(len(signal))
    for sc in scales_sec:
        s = max(int(sc * fs), 3)
        raw_energy  = _mexican_hat_energy(signal, s)
        norm_energy = raw_energy / np.sqrt(s)   # scale-normalised energy
        energy = np.maximum(energy, norm_energy)

    # 2. Threshold: keep only positive modulus maxima
    #    (QRS upstroke produces positive response)
    energy = np.maximum(energy, 0)

    # 3. Smooth briefly
    win = max(int(0.04 * fs), 3)   # 40 ms
    energy = np.convolve(energy, np.ones(win) / win, mode="same")

    # 4. Detect peaks in energy with adaptive threshold
    threshold = np.percentile(energy[energy > 0], 60) if energy.any() else 0.0
    min_dist  = int(0.25 * fs)
    peaks, _  = find_peaks(energy, height=threshold, distance=min_dist)

    if len(peaks) == 0:
        return np.array([], dtype=int)

    # 5. Refine to signal max in ±40 ms window
    peaks = _refine_to_signal_max(signal, peaks, fs, window_ms=40)

    return peaks.astype(int)


# ─────────────────────────────────────────────────────────────────────────────

def _mexican_hat_energy(signal: np.ndarray, scale: int) -> np.ndarray:
    """
    Approximate Mexican Hat CWT at a single scale via convolution.
    MH(t) = (1 - t²) * exp(-t²/2)   (normalised)
    """
    half = max(scale * 3, 5)
    t    = np.linspace(-3, 3, 2 * half + 1)
    mh   = (1 - t ** 2) * np.exp(-0.5 * t ** 2)
    mh  /= (np.sum(mh ** 2) ** 0.5 + 1e-9)
    return np.convolve(signal, mh, mode="same")


def _refine_to_signal_max(
    signal: np.ndarray, peaks: np.ndarray, fs: int, window_ms: int = 40
) -> np.ndarray:
    """Snap to positive maximum — do NOT use abs().
    The ensemble passes polarity-corrected work_sig (QRS always positive).
    abs() snaps to inverted T-waves when |T| > |R|, causing mis-detection.
    """
    half    = int(window_ms * fs / 1000)
    refined = []
    for p in peaks:
        lo  = max(0, p - half)
        hi  = min(len(signal), p + half)
        idx = int(np.argmax(signal[lo:hi])) + lo
        refined.append(idx)
    return np.array(refined, dtype=int)
