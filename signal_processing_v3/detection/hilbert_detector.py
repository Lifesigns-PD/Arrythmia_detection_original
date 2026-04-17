"""
hilbert_detector.py — Hilbert Envelope QRS Detector
=====================================================
Computes the analytic signal envelope via Hilbert transform.
QRS complexes appear as prominent peaks in the envelope.

Advantages over Pan-Tompkins:
  - No bandpass needed; works on cleaned ECG directly
  - Robust to tall T-waves (envelope is broader, easier to threshold)
  - Simpler adaptive thresholding
"""

import numpy as np
from scipy.signal import hilbert, butter, filtfilt, find_peaks


def detect_r_peaks_hilbert(signal: np.ndarray, fs: int = 125) -> np.ndarray:
    """
    Detect R-peaks using the Hilbert envelope method.

    Parameters
    ----------
    signal : np.ndarray  1-D cleaned ECG (mV)
    fs     : int         sampling rate

    Returns
    -------
    np.ndarray  — integer indices of detected R-peaks
    """
    if len(signal) < fs:
        return np.array([], dtype=int)

    # 1. Bandpass 5–25 Hz to isolate QRS band
    sig_bp = _bandpass(signal, fs, low=5.0, high=25.0)

    # 2. Analytic signal → envelope
    envelope = np.abs(hilbert(sig_bp))

    # 3. Smooth envelope
    win = max(int(0.08 * fs), 3)   # 80 ms smoothing
    envelope = np.convolve(envelope, np.ones(win) / win, mode="same")

    # 4. Adaptive threshold (70% of running max, 1-s window)
    threshold = _adaptive_threshold(envelope, fs, window_sec=1.0, fraction=0.40)

    # 5. Find peaks above threshold
    min_dist = int(0.25 * fs)   # minimum 250 ms between peaks
    peaks, props = find_peaks(envelope, height=threshold, distance=min_dist)

    if len(peaks) == 0:
        return np.array([], dtype=int)

    # 6. Refine: snap each peak to the actual signal maximum in ±40 ms window
    peaks = _refine_to_signal_max(signal, peaks, fs, window_ms=40)

    return peaks.astype(int)


# ─────────────────────────────────────────────────────────────────────────────

def _bandpass(signal: np.ndarray, fs: int, low: float, high: float) -> np.ndarray:
    nyq  = 0.5 * fs
    lo   = max(low  / nyq, 0.001)
    hi   = min(high / nyq, 0.999)
    b, a = butter(2, [lo, hi], btype="band")
    return filtfilt(b, a, signal)


def _adaptive_threshold(
    envelope: np.ndarray, fs: int, window_sec: float = 1.0, fraction: float = 0.40
) -> np.ndarray:
    """Per-sample adaptive threshold = fraction × running max."""
    win    = max(int(window_sec * fs), 1)
    n      = len(envelope)
    thresh = np.zeros(n)
    for i in range(n):
        lo = max(0, i - win // 2)
        hi = min(n, i + win // 2)
        thresh[i] = fraction * np.max(envelope[lo:hi])
    return thresh


def _refine_to_signal_max(
    signal: np.ndarray, peaks: np.ndarray, fs: int, window_ms: int = 40
) -> np.ndarray:
    """Snap detected peak to signal maximum within ±window_ms.

    Uses argmax (not argmax-of-abs) because the ensemble always passes
    a polarity-corrected signal (work_sig) where QRS is positive.
    Using abs() would incorrectly snap to inverted T-waves whose
    absolute amplitude exceeds the R-peak, causing T-wave mis-detection.
    """
    half = int(window_ms * fs / 1000)
    refined = []
    for p in peaks:
        lo  = max(0, p - half)
        hi  = min(len(signal), p + half)
        idx = int(np.argmax(signal[lo:hi])) + lo
        refined.append(idx)
    return np.array(refined, dtype=int)
