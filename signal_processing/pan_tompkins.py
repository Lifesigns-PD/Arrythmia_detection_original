"""
Pan-Tompkins QRS Detection Algorithm
=====================================

Full implementation of the Pan-Tompkins real-time QRS detection algorithm.

Reference:
    Pan, J., & Tompkins, W. J. (1985).
    "A Real-Time QRS Detection Algorithm."
    IEEE Transactions on Biomedical Engineering, BME-32(3), 230-236.

Pipeline:
    1. Bandpass Filter  (5-15 Hz)  — isolates QRS energy band
    2. Derivative        (5-point)  — emphasises steep QRS slopes
    3. Squaring                     — makes all values positive, amplifies large slopes
    4. Moving-Window Integration    — smooths into a single energy pulse per QRS
    5. Adaptive Thresholding        — dual thresholds (signal / noise) with search-back
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
#  Stage 1 — Bandpass Filter (5–15 Hz)
# ---------------------------------------------------------------------------

def _bandpass_filter(signal: np.ndarray, fs: int,
                     lowcut: float = 5.0, highcut: float = 15.0,
                     order: int = 2) -> np.ndarray:
    """
    Butterworth bandpass filter to isolate the QRS frequency band.

    The original Pan-Tompkins paper uses cascaded integer-coefficient LP/HP
    filters designed for 200 Hz.  Here we use a standard Butterworth filter
    so the algorithm works at ANY sampling rate.

    Parameters
    ----------
    signal : np.ndarray
        Raw or pre-cleaned ECG signal.
    fs : int
        Sampling frequency in Hz.
    lowcut : float
        Low cutoff frequency (default 5 Hz).
    highcut : float
        High cutoff frequency (default 15 Hz).
    order : int
        Filter order (default 2 — matches the original paper's passband).

    Returns
    -------
    np.ndarray
        Bandpass-filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Clamp to valid range (0, 1) exclusive
    low = max(low, 0.001)
    high = min(high, 0.999)

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


# ---------------------------------------------------------------------------
#  Stage 2 — Five-Point Derivative
# ---------------------------------------------------------------------------

def _derivative(signal: np.ndarray) -> np.ndarray:
    """
    Five-point derivative as specified in the original Pan-Tompkins paper.

        y(n) = (1/8T) * [ -x(n-2) - 2x(n-1) + 2x(n+1) + x(n+2) ]

    This provides a good approximation of the first derivative while
    suppressing high-frequency noise better than a simple np.diff().

    Parameters
    ----------
    signal : np.ndarray
        Bandpass-filtered ECG signal.

    Returns
    -------
    np.ndarray
        Derivative signal (same length as input, edges zero-padded).
    """
    result = np.zeros_like(signal)
    # Coefficients: [-1, -2, 0, 2, 1] scaled by 1/8
    for i in range(2, len(signal) - 2):
        result[i] = (-signal[i - 2] - 2 * signal[i - 1]
                      + 2 * signal[i + 1] + signal[i + 2]) / 8.0
    return result


# ---------------------------------------------------------------------------
#  Stage 3 — Squaring
# ---------------------------------------------------------------------------

def _squaring(signal: np.ndarray) -> np.ndarray:
    """
    Point-by-point squaring.

    Makes all values positive and non-linearly amplifies large slopes
    (QRS complexes) relative to smaller slopes (P/T waves).

    Parameters
    ----------
    signal : np.ndarray
        Derivative signal.

    Returns
    -------
    np.ndarray
        Squared signal.
    """
    return signal ** 2


# ---------------------------------------------------------------------------
#  Stage 4 — Moving-Window Integration
# ---------------------------------------------------------------------------

def _moving_window_integration(signal: np.ndarray, fs: int,
                                window_sec: float = 0.150) -> np.ndarray:
    """
    Moving-window integrator.

    Smooths the squared signal into a single broad pulse per QRS complex.
    The original paper recommends a window width of ~150 ms.

    - Too narrow  → multiple peaks per QRS (split detection)
    - Too wide    → merges adjacent QRS complexes (missed beats)

    Parameters
    ----------
    signal : np.ndarray
        Squared signal.
    fs : int
        Sampling frequency in Hz.
    window_sec : float
        Integration window length in seconds (default 0.150 s).

    Returns
    -------
    np.ndarray
        Integrated signal.
    """
    window_size = max(1, int(window_sec * fs))
    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode='same')


# ---------------------------------------------------------------------------
#  Stage 5 — Adaptive Thresholding with Search-Back
# ---------------------------------------------------------------------------

def _adaptive_thresholding(integrated: np.ndarray,
                           original_filtered: np.ndarray,
                           fs: int) -> np.ndarray:
    """
    Dual adaptive thresholding on both the integrated signal and the
    bandpass-filtered signal, with a search-back mechanism for missed beats.

    The algorithm maintains two running estimates:
        - SPKI / NPKI  : signal / noise peak levels on the integrated waveform
        - SPKF / NPKF  : signal / noise peak levels on the filtered waveform

    Decision rules (from the original paper):
        1. A candidate peak must exceed THRESHOLD_I1 on the integrated signal
           AND THRESHOLD_F1 on the filtered signal to be classified as a QRS.
        2. If no QRS is found within 166% of the average RR interval,
           the algorithm searches back with lower thresholds
           (THRESHOLD_I2, THRESHOLD_F2) to rescue a possibly missed beat.
        3. After each classification the running estimates are updated.

    Parameters
    ----------
    integrated : np.ndarray
        Moving-window integrated signal.
    original_filtered : np.ndarray
        Bandpass-filtered signal (used for secondary threshold check).
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Array of sample indices where R-peaks (QRS complexes) were detected.
    """
    # -- Initial peak finding on the integrated signal ----------------------
    min_distance = int(0.2 * fs)   # 200 ms refractory period
    peaks, _ = find_peaks(integrated, distance=min_distance)

    if len(peaks) == 0:
        return np.array([], dtype=int)

    # -- Initialise adaptive thresholds (training on first 2 seconds) -------
    training_end = min(int(2.0 * fs), len(integrated))
    training_peaks = peaks[peaks < training_end]

    if len(training_peaks) > 0:
        spki = np.max(integrated[training_peaks])      # signal peak (integrated)
        spkf = np.max(np.abs(original_filtered[training_peaks]))  # signal peak (filtered)
    else:
        spki = np.max(integrated[:training_end])
        spkf = np.max(np.abs(original_filtered[:training_end]))

    npki = 0.0     # noise peak (integrated)
    npkf = 0.0     # noise peak (filtered)

    threshold_i1 = npki + 0.25 * (spki - npki)   # Primary threshold (integrated)
    threshold_i2 = 0.5 * threshold_i1             # Search-back threshold (integrated)
    threshold_f1 = npkf + 0.25 * (spkf - npkf)   # Primary threshold (filtered)
    threshold_f2 = 0.5 * threshold_f1             # Search-back threshold (filtered)

    # -- RR interval tracking -----------------------------------------------
    rr_average = fs  # Start with 1-second assumption (~60 bpm)
    rr_history: List[int] = []
    RR_LOW_LIMIT = 0.92
    RR_HIGH_LIMIT = 1.16
    RR_MISSED_LIMIT = 1.66

    # -- Classification loop ------------------------------------------------
    qrs_peaks: List[int] = []
    noise_peaks: List[int] = []

    for i, peak_idx in enumerate(peaks):
        peak_val_i = integrated[peak_idx]
        peak_val_f = np.abs(original_filtered[peak_idx])

        # --- Primary threshold test ----------------------------------------
        is_qrs = False

        if peak_val_i > threshold_i1 and peak_val_f > threshold_f1:
            # Passed both thresholds → QRS candidate
            # Check refractory period (200 ms after last QRS)
            if len(qrs_peaks) > 0:
                time_since_last = peak_idx - qrs_peaks[-1]
                if time_since_last < int(0.2 * fs):
                    # Inside refractory — treat as T-wave / noise
                    is_qrs = False
                elif time_since_last < int(0.36 * fs):
                    # Between 200-360 ms: possible T-wave check
                    # MUCH stricter: if peak is within 150-360 ms of last QRS AND
                    # current peak is > 30% of previous QRS → very likely a T-wave
                    # Only accept if it's significantly larger than previous (rare arrhythmia)
                    if peak_val_i > 0.30 * integrated[qrs_peaks[-1]]:
                        # This is likely a tall T-wave or artifact, not a real QRS
                        is_qrs = False
                    else:
                        is_qrs = True
                else:
                    is_qrs = True
            else:
                is_qrs = True

        if is_qrs:
            qrs_peaks.append(peak_idx)

            # Update signal peak estimates
            spki = 0.125 * peak_val_i + 0.875 * spki
            spkf = 0.125 * peak_val_f + 0.875 * spkf

            # Update RR history
            if len(qrs_peaks) >= 2:
                rr = qrs_peaks[-1] - qrs_peaks[-2]
                rr_history.append(rr)
                if len(rr_history) > 8:
                    rr_history = rr_history[-8:]
                rr_average = int(np.mean(rr_history))

        else:
            noise_peaks.append(peak_idx)
            # Update noise peak estimates
            npki = 0.125 * peak_val_i + 0.875 * npki
            npkf = 0.125 * peak_val_f + 0.875 * npkf

        # --- Recalculate thresholds ----------------------------------------
        threshold_i1 = npki + 0.25 * (spki - npki)
        threshold_i2 = 0.5 * threshold_i1
        threshold_f1 = npkf + 0.25 * (spkf - npkf)
        threshold_f2 = 0.5 * threshold_f1

        # --- Search-back for missed beats ----------------------------------
        if len(qrs_peaks) >= 2:
            rr_current = qrs_peaks[-1] - qrs_peaks[-2]

            if rr_current > RR_MISSED_LIMIT * rr_average:
                # A QRS may have been missed — search back with lower thresholds
                search_start = qrs_peaks[-2] + int(0.2 * fs)
                search_end = qrs_peaks[-1] - int(0.2 * fs)

                # Find all candidate peaks in the gap
                candidates = peaks[(peaks > search_start) & (peaks < search_end)]

                for cand in candidates:
                    cand_val_i = integrated[cand]
                    cand_val_f = np.abs(original_filtered[cand])

                    if cand_val_i > threshold_i2 and cand_val_f > threshold_f2:
                        # Rescue this beat
                        qrs_peaks.append(cand)

                        # Update signal peak (search-back uses half weight)
                        spki = 0.25 * cand_val_i + 0.75 * spki
                        spkf = 0.25 * cand_val_f + 0.75 * spkf
                        break  # Only rescue one beat per gap

                # Re-sort after insertion
                qrs_peaks.sort()

    # --- Final sanity check: reject peaks that break RR rhythm ---------------
    # If any RR interval is <65% of median RR, it's likely a false positive
    # (typically a T-wave detected as R-peak)
    if len(qrs_peaks) >= 2:
        rr_intervals = np.diff(qrs_peaks)
        median_rr = np.median(rr_intervals)

        # Keep only peaks that maintain proper RR timing
        valid_qrs = [qrs_peaks[0]]
        for i in range(1, len(qrs_peaks)):
            rr = qrs_peaks[i] - valid_qrs[-1]
            # Allow 65% of median (accounts for ectopy/PVCs) up to 150% (skipped beat)
            if median_rr * 0.65 <= rr <= median_rr * 1.50:
                valid_qrs.append(qrs_peaks[i])
        qrs_peaks = valid_qrs

    return np.array(qrs_peaks, dtype=int)


# ---------------------------------------------------------------------------
#  Stage 6 — Peak Refinement (find true R-peak in original signal)
# ---------------------------------------------------------------------------

def _refine_peaks(detected_peaks: np.ndarray,
                  original_signal: np.ndarray,
                  fs: int,
                  search_window_sec: float = 0.050) -> np.ndarray:
    """
    Refine detected QRS locations back to the TRUE R-peak in the original
    (unfiltered or cleaned) ECG signal.

    The integrated signal introduces a group delay and smoothing, so the
    detected peak position may be slightly offset from the actual R-peak.
    This step searches a small window around each detection and picks the
    maximum amplitude in the original signal.

    Parameters
    ----------
    detected_peaks : np.ndarray
        QRS peak indices from the adaptive thresholding stage.
    original_signal : np.ndarray
        The original (or minimally cleaned) ECG signal.
    fs : int
        Sampling frequency in Hz.
    search_window_sec : float
        Half-width of search window in seconds (default ±50 ms).

    Returns
    -------
    np.ndarray
        Refined R-peak indices aligned to the original signal.
    """
    if len(detected_peaks) == 0:
        return np.array([], dtype=int)

    search_samples = int(search_window_sec * fs)
    refined = []

    for peak in detected_peaks:
        start = max(0, peak - search_samples)
        end = min(len(original_signal), peak + search_samples + 1)
        local_max = start + np.argmax(original_signal[start:end])
        refined.append(local_max)

    # Remove duplicates that may arise from overlapping search windows
    refined = sorted(set(refined))
    return np.array(refined, dtype=int)


# ===========================================================================
#  PUBLIC API — Main Entry Point
# ===========================================================================

def pan_tompkins_detect(signal: np.ndarray,
                        fs: int,
                        bandpass_low: float = 5.0,
                        bandpass_high: float = 15.0,
                        integration_window_sec: float = 0.150,
                        refine: bool = True) -> dict:
    """
    Full Pan-Tompkins QRS detection pipeline.

    Parameters
    ----------
    signal : np.ndarray
        ECG signal (1-D array). Can be raw or pre-cleaned.
        If the signal has already been bandpass filtered externally,
        the internal bandpass will still run but on a narrower QRS band.
    fs : int
        Sampling frequency in Hz.
    bandpass_low : float
        Low cutoff for bandpass filter (default 5 Hz).
    bandpass_high : float
        High cutoff for bandpass filter (default 15 Hz).
    integration_window_sec : float
        Moving-window integration width in seconds (default 0.150 s).
    refine : bool
        If True, refine detected peaks back to the original signal's
        local maxima (recommended). Default True.

    Returns
    -------
    dict
        {
            "r_peaks"       : np.ndarray — final R-peak sample indices,
            "filtered"      : np.ndarray — bandpass-filtered signal,
            "derivative"    : np.ndarray — derivative signal,
            "squared"       : np.ndarray — squared signal,
            "integrated"    : np.ndarray — integrated signal,
            "num_beats"     : int        — number of detected beats,
            "mean_hr_bpm"   : float      — estimated mean heart rate (bpm),
        }

    Example
    -------
    >>> from signal_processing.pan_tompkins import pan_tompkins_detect
    >>> results = pan_tompkins_detect(ecg_signal, fs=125)
    >>> r_peaks = results["r_peaks"]
    >>> print(f"Detected {results['num_beats']} beats, HR ≈ {results['mean_hr_bpm']:.0f} bpm")
    """

    # --- Stage 1: Bandpass Filter ---
    filtered = _bandpass_filter(signal, fs, bandpass_low, bandpass_high)

    # --- Stage 2: Five-Point Derivative ---
    derivative = _derivative(filtered)

    # --- Stage 3: Squaring ---
    squared = _squaring(derivative)

    # --- Stage 4: Moving-Window Integration ---
    integrated = _moving_window_integration(squared, fs, integration_window_sec)

    # --- Stage 5: Adaptive Thresholding with Search-Back ---
    qrs_indices = _adaptive_thresholding(integrated, filtered, fs)

    # --- Stage 6: Refine to true R-peak locations ---
    if refine and len(qrs_indices) > 0:
        r_peaks = _refine_peaks(qrs_indices, signal, fs)
    else:
        r_peaks = qrs_indices

    # --- Compute summary metrics ---
    num_beats = len(r_peaks)
    mean_hr = 0.0
    if num_beats >= 2:
        rr_intervals_s = np.diff(r_peaks) / fs
        mean_hr = 60.0 / np.mean(rr_intervals_s)

    return {
        "r_peaks":      r_peaks,
        "filtered":     filtered,
        "derivative":   derivative,
        "squared":      squared,
        "integrated":   integrated,
        "num_beats":    num_beats,
        "mean_hr_bpm":  mean_hr,
    }


# ===========================================================================
#  Convenience wrapper (drop-in replacement for ecgprocessor._r_peak_detection)
# ===========================================================================

def detect_r_peaks(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Simplified wrapper that returns ONLY the R-peak indices.

    Drop-in replacement for ``ECGProcessor._r_peak_detection()``
    in ``utils/ecgprocessor.py``.

    Parameters
    ----------
    signal : np.ndarray
        Pre-processed ECG signal.
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Array of R-peak sample indices.
    """
    result = pan_tompkins_detect(signal, fs)
    return result["r_peaks"]


# ===========================================================================
#  Quick self-test
# ===========================================================================

if __name__ == "__main__":
    # Generate a synthetic ECG-like signal for quick validation
    print("=" * 60)
    print("  Pan-Tompkins QRS Detection — Self-Test")
    print("=" * 60)

    fs = 125
    duration = 10  # seconds
    t = np.arange(0, duration, 1 / fs)

    # Simulate ~75 bpm (RR ≈ 0.8 s) with Gaussian QRS-like spikes
    rr_interval = 0.8
    num_beats = int(duration / rr_interval)
    synthetic_ecg = np.zeros_like(t)

    true_peaks = []
    for beat in range(num_beats):
        center = beat * rr_interval + 0.1  # slight offset
        if center >= duration:
            break
        idx = int(center * fs)
        true_peaks.append(idx)
        # Create a narrow Gaussian spike (QRS-like)
        qrs_width = 0.04  # 40 ms
        for i in range(len(t)):
            dt = t[i] - center
            synthetic_ecg[i] += 1.0 * np.exp(-(dt ** 2) / (2 * (qrs_width / 3) ** 2))

    # Add some noise
    np.random.seed(42)
    synthetic_ecg += 0.05 * np.random.randn(len(synthetic_ecg))

    # Run Pan-Tompkins
    results = pan_tompkins_detect(synthetic_ecg, fs)

    print(f"\n  Sampling Rate     : {fs} Hz")
    print(f"  Signal Duration   : {duration} s")
    print(f"  True Beats        : {len(true_peaks)}")
    print(f"  Detected Beats    : {results['num_beats']}")
    print(f"  Mean HR           : {results['mean_hr_bpm']:.1f} bpm")
    print(f"  R-peak Indices    : {results['r_peaks'][:10]}{'...' if len(results['r_peaks']) > 10 else ''}")
    print(f"\n  ✓ Self-test complete.")
    print("=" * 60)
