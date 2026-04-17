"""
artifact_removal.py — Muscle Artifact + Spike Removal
=======================================================
Muscle tremor produces high-frequency bursts (20–300 Hz) that cause
false QRS detections.  Motion artifacts cause sudden amplitude jumps.

V3 approach:
  1. Detect artifact segments using high-frequency energy + amplitude jump detection
  2. Suppress artifact using wavelet soft-thresholding (pywavelets)
     Fallback: linear interpolation over artifact segments if pywt not available
  3. Clip extreme single-sample spikes (±5 σ)
"""

import numpy as np
from scipy.signal import butter, filtfilt
from typing import List, Tuple


SPIKE_SIGMA_THRESHOLD    = 5.0     # clip samples beyond ±5σ
ARTIFACT_WINDOW_MS       = 200     # ms — analysis window for artifact detection
HF_ARTIFACT_THRESHOLD    = 3.0     # HF energy ratio above this = artifact window
JUMP_THRESHOLD_MV        = 2.0     # mV — inter-sample jump flagged as artifact


def remove_artifacts(signal: np.ndarray, fs: int = 125) -> np.ndarray:
    """
    Remove muscle artifacts and amplitude spikes from ECG signal.

    Parameters
    ----------
    signal : np.ndarray  1-D ECG (mV)
    fs     : int         Sampling rate

    Returns
    -------
    np.ndarray  — artifact-suppressed signal
    """
    if len(signal) < 10:
        return signal.copy()

    sig = signal.astype(np.float64)

    # Stage 1: Remove single-sample spikes (amplitude outliers)
    sig = _clip_spikes(sig)

    # Stage 2: Detect artifact windows
    artifact_mask = _detect_artifact_windows(sig, fs)

    # Stage 3: Suppress artifacts
    if artifact_mask.any():
        sig = _suppress_artifacts(sig, artifact_mask, fs)

    return sig.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clip_spikes(signal: np.ndarray) -> np.ndarray:
    """Replace samples beyond ±5σ with clipped values."""
    mu  = np.median(signal)
    sig_std = np.std(signal)
    if sig_std < 1e-9:
        return signal
    lo = mu - SPIKE_SIGMA_THRESHOLD * sig_std
    hi = mu + SPIKE_SIGMA_THRESHOLD * sig_std
    return np.clip(signal, lo, hi)


def _detect_artifact_windows(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Returns a boolean mask (same length as signal) where True = artifact.
    Uses two criteria:
      A. High HF energy (muscle tremor)
      B. Large inter-sample jumps (motion artifact)

    VF bypass: if the window has dominant low-frequency chaotic energy (2–10 Hz)
    AND amplitude is within the physiological VF range (0.15–3.0 mV), skip
    artifact flagging entirely.  VF is low-frequency chaos, not HF muscle noise.
    Motion artifacts (jogging etc.) exceed 3.0 mV; ADC static noise is < 0.15 mV.
    """
    n       = len(signal)
    mask    = np.zeros(n, dtype=bool)
    win_len = max(int(ARTIFACT_WINDOW_MS * fs / 1000), 10)

    # ── VF bypass (whole-window guard before per-window analysis) ──
    try:
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        psd   = np.abs(np.fft.rfft(signal)) ** 2
        total_power   = float(np.sum(psd[psd > 0]))
        vf_band_power = float(np.sum(psd[(freqs >= 2.0) & (freqs <= 10.0)]))
        max_amp = float(np.max(np.abs(signal)))
        if (total_power > 0
                and (vf_band_power / total_power) > 0.6
                and 0.15 < max_amp < 3.0):
            return mask   # dominant 2–10 Hz energy in physiological range → likely VF
    except Exception:
        pass

    # ── A. HF energy ──
    try:
        # High-pass above 35 Hz to isolate muscle band
        nyq  = 0.5 * fs
        if nyq > 36:
            b, a = butter(2, 35.0 / nyq, btype="high")
            hf = filtfilt(b, a, signal)
        else:
            hf = np.zeros_like(signal)

        # Compute RMS in sliding windows
        total_rms = np.sqrt(np.mean(signal ** 2)) + 1e-9
        for start in range(0, n - win_len, win_len // 2):
            end    = min(start + win_len, n)
            w_rms  = np.sqrt(np.mean(hf[start:end] ** 2))
            if w_rms / total_rms > HF_ARTIFACT_THRESHOLD:
                mask[start:end] = True
    except Exception:
        pass

    # ── B. Inter-sample jumps ──
    jumps = np.abs(np.diff(signal))
    jump_indices = np.where(jumps > JUMP_THRESHOLD_MV)[0]
    for idx in jump_indices:
        lo = max(0, idx - win_len // 2)
        hi = min(n, idx + win_len // 2)
        mask[lo:hi] = True

    return mask


def _suppress_artifacts(
    signal: np.ndarray, mask: np.ndarray, fs: int
) -> np.ndarray:
    """
    Try wavelet soft-thresholding on artifact windows.
    Falls back to linear interpolation if pywt is unavailable.
    """
    try:
        import pywt
        return _wavelet_denoise(signal, mask, level=3, wavelet="db4")
    except ImportError:
        return _interpolate_artifacts(signal, mask)


def _wavelet_denoise(
    signal: np.ndarray, mask: np.ndarray, level: int = 3, wavelet: str = "db4"
) -> np.ndarray:
    """Wavelet soft-thresholding limited to artifact windows."""
    import pywt
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Threshold only the detail coefficients covering artifact windows
    threshold = np.std(signal) * 0.5  # moderate threshold
    coeffs_t  = [coeffs[0]] + [
        pywt.threshold(c, value=threshold, mode="soft")
        for c in coeffs[1:]
    ]
    denoised = pywt.waverec(coeffs_t, wavelet)[: len(signal)]
    # Blend: use denoised only where artifact mask is True
    result = signal.copy()
    result[mask] = denoised[mask]
    return result


def _interpolate_artifacts(signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Linear interpolation over artifact windows."""
    result  = signal.copy()
    indices = np.arange(len(signal))
    good    = ~mask
    if good.sum() < 2:
        return result
    result[mask] = np.interp(indices[mask], indices[good], signal[good])
    return result
