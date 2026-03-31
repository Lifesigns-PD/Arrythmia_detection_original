"""
morphology.py — ECG Morphology Feature Extraction
===================================================

Computes per-beat and segment-level morphology features from an ECG signal
using NeuroKit2 DWT delineation + fallback heuristics.

Features extracted (per-beat and averaged):
    1. P wave      — duration (ms), amplitude (mV), presence
    2. PR interval — P-onset to QRS-onset (ms)
    3. PR segment  — P-end to QRS-onset (ms)
    4. QRS complex — duration (ms), amplitude (mV)
    5. QTc interval— QRS-onset to T-end, corrected by Bazett (ms)
    6. ST segment  — QRS-end to T-onset (ms), deviation (mV)
    7. T wave      — duration (ms), amplitude (mV)
    8. RR interval — R-to-R (ms), heart rate (bpm)

Normal reference ranges (from clinical literature):
    P wave:      60-80 ms duration,    <0.25 mV amplitude
    PR interval: 120-200 ms
    PR segment:  50-120 ms
    QRS complex: 80-120 ms duration
    QTc:         360-440 ms (Bazett)
    ST segment:  100-120 ms,           deviation < ±0.1 mV
    T wave:      120-160 ms
    RR interval: 600-1200 ms (50-100 bpm)
"""

import numpy as np
from scipy.signal import butter, filtfilt
from typing import Dict, Any, List, Optional

# Normal reference ranges
NORMAL_RANGES = {
    "p_wave_duration_ms":    (60, 80),
    "pr_interval_ms":        (120, 200),
    "pr_segment_ms":         (50, 120),
    "qrs_duration_ms":       (80, 120),
    "qtc_ms":                (360, 440),
    "st_segment_ms":         (100, 120),
    "st_deviation_mv":       (-0.1, 0.1),
    "t_wave_duration_ms":    (120, 160),
    "rr_interval_ms":        (600, 1200),
}


def _safe_median(values: list) -> float:
    """Return median of non-NaN values, or 0.0 if empty."""
    clean = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.median(clean)) if clean else 0.0


def _safe_mean(values: list) -> float:
    clean = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(clean)) if clean else 0.0


def _lowpass_40hz(signal: np.ndarray, fs: int) -> np.ndarray:
    """40 Hz lowpass to smooth noise before delineation."""
    nyq = 0.5 * fs
    if nyq <= 40:
        return signal
    b, a = butter(2, 40.0 / nyq, btype='low')
    return filtfilt(b, a, signal.astype(np.float64))


def _flag(value: float, key: str) -> str:
    """Return 'normal', 'low', 'high', or 'unavailable'."""
    if value == 0.0:
        return "unavailable"
    lo, hi = NORMAL_RANGES.get(key, (None, None))
    if lo is None:
        return "unavailable"
    if value < lo:
        return "low"
    elif value > hi:
        return "high"
    return "normal"


def extract_morphology(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: int = 125,
) -> Dict[str, Any]:
    """
    Extract all ECG morphology features from a single segment.

    Parameters
    ----------
    signal : np.ndarray
        1-D ECG signal (e.g. 1250 samples at 125 Hz = 10 seconds).
    r_peaks : np.ndarray
        Array of R-peak sample indices within `signal`.
    fs : int
        Sampling frequency in Hz (default 125).

    Returns
    -------
    dict
        {
            "per_beat": [ {...features per beat...}, ... ],
            "summary": { ...averaged features + flags... },
            "normal_ranges": { ...reference ranges... },
        }
    """
    result = {
        "per_beat": [],
        "summary": {},
        "normal_ranges": NORMAL_RANGES,
    }

    if r_peaks is None or len(r_peaks) < 2:
        result["summary"] = _empty_summary()
        return result

    r_peaks = np.asarray(r_peaks, dtype=int)
    smooth = _lowpass_40hz(signal, fs)

    # Try NeuroKit2 delineation
    waves = _delineate(smooth, r_peaks, fs)

    # Extract per-beat features
    per_beat = []
    n_beats = len(r_peaks)

    for i in range(n_beats):
        beat = _extract_single_beat(signal, smooth, r_peaks, i, n_beats, waves, fs)
        per_beat.append(beat)

    result["per_beat"] = per_beat

    # Compute summary (median across beats)
    result["summary"] = _summarize(per_beat, r_peaks, fs)

    return result


def _delineate(smooth_signal: np.ndarray, r_peaks: np.ndarray, fs: int) -> Optional[dict]:
    """Run NeuroKit2 DWT delineation. Returns waves dict or None."""
    try:
        import neurokit2 as nk
        import pandas as pd
        _, waves = nk.ecg_delineate(
            smooth_signal, r_peaks,
            sampling_rate=fs, method="dwt", show=False
        )
        return waves
    except Exception:
        return None


def _get_wave_idx(waves: Optional[dict], key: str, beat_idx: int) -> Optional[int]:
    """Safely get a delineation index for a specific beat."""
    if waves is None:
        return None
    arr = waves.get(key, [])
    if beat_idx >= len(arr):
        return None
    val = arr[beat_idx]
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return int(val)


def _extract_single_beat(
    raw: np.ndarray,
    smooth: np.ndarray,
    r_peaks: np.ndarray,
    beat_idx: int,
    n_beats: int,
    waves: Optional[dict],
    fs: int,
) -> Dict[str, Any]:
    """Extract morphology features for a single beat."""
    r_sample = int(r_peaks[beat_idx])

    # Delineation indices
    p_onset  = _get_wave_idx(waves, "ECG_P_Onsets", beat_idx)
    p_peak   = _get_wave_idx(waves, "ECG_P_Peaks", beat_idx)
    p_offset = _get_wave_idx(waves, "ECG_P_Offsets", beat_idx)
    r_onset  = _get_wave_idx(waves, "ECG_R_Onsets", beat_idx)
    r_offset = _get_wave_idx(waves, "ECG_R_Offsets", beat_idx)
    t_onset  = _get_wave_idx(waves, "ECG_T_Onsets", beat_idx)
    t_peak   = _get_wave_idx(waves, "ECG_T_Peaks", beat_idx)
    t_offset = _get_wave_idx(waves, "ECG_T_Offsets", beat_idx)

    beat = {"beat_index": beat_idx, "r_peak_sample": r_sample}

    # --- P wave ---
    if p_onset is not None and p_offset is not None and p_offset > p_onset:
        beat["p_wave_duration_ms"] = (p_offset - p_onset) * 1000.0 / fs
        beat["p_wave_present"] = True
    else:
        beat["p_wave_duration_ms"] = None
        beat["p_wave_present"] = False

    if p_peak is not None and 0 <= p_peak < len(raw):
        beat["p_wave_amplitude_mv"] = float(raw[p_peak])
    else:
        beat["p_wave_amplitude_mv"] = None

    # --- PR interval (P-onset to R-onset) ---
    if p_onset is not None and r_onset is not None and r_onset > p_onset:
        beat["pr_interval_ms"] = (r_onset - p_onset) * 1000.0 / fs
    else:
        beat["pr_interval_ms"] = None

    # --- PR segment (P-offset to R-onset) ---
    if p_offset is not None and r_onset is not None and r_onset > p_offset:
        beat["pr_segment_ms"] = (r_onset - p_offset) * 1000.0 / fs
    else:
        beat["pr_segment_ms"] = None

    # --- QRS complex ---
    if r_onset is not None and r_offset is not None and r_offset > r_onset:
        beat["qrs_duration_ms"] = (r_offset - r_onset) * 1000.0 / fs
    else:
        beat["qrs_duration_ms"] = None

    beat["qrs_amplitude_mv"] = float(raw[r_sample]) if 0 <= r_sample < len(raw) else None

    # --- ST segment (R-offset to T-onset) ---
    if r_offset is not None and t_onset is not None and t_onset > r_offset:
        beat["st_segment_ms"] = (t_onset - r_offset) * 1000.0 / fs
        # ST deviation: mean amplitude in ST segment relative to baseline
        st_region = raw[r_offset:t_onset]
        if len(st_region) > 0:
            # Baseline = mean of PR segment (isoelectric)
            if p_offset is not None and r_onset is not None and r_onset > p_offset:
                baseline = float(np.mean(raw[p_offset:r_onset]))
            else:
                baseline = 0.0
            beat["st_deviation_mv"] = float(np.mean(st_region)) - baseline
        else:
            beat["st_deviation_mv"] = None
    else:
        beat["st_segment_ms"] = None
        beat["st_deviation_mv"] = None

    # --- T wave ---
    if t_onset is not None and t_offset is not None and t_offset > t_onset:
        beat["t_wave_duration_ms"] = (t_offset - t_onset) * 1000.0 / fs
    else:
        beat["t_wave_duration_ms"] = None

    if t_peak is not None and 0 <= t_peak < len(raw):
        beat["t_wave_amplitude_mv"] = float(raw[t_peak])
    else:
        beat["t_wave_amplitude_mv"] = None

    # --- QT interval (R-onset to T-offset) ---
    if r_onset is not None and t_offset is not None and t_offset > r_onset:
        qt_ms = (t_offset - r_onset) * 1000.0 / fs
        beat["qt_interval_ms"] = qt_ms

        # QTc (Bazett): QTc = QT / sqrt(RR in seconds)
        if beat_idx < n_beats - 1:
            rr_s = (r_peaks[beat_idx + 1] - r_peaks[beat_idx]) / fs
        elif beat_idx > 0:
            rr_s = (r_peaks[beat_idx] - r_peaks[beat_idx - 1]) / fs
        else:
            rr_s = 0.0

        if rr_s > 0:
            beat["qtc_bazett_ms"] = qt_ms / np.sqrt(rr_s)
        else:
            beat["qtc_bazett_ms"] = None
    else:
        beat["qt_interval_ms"] = None
        beat["qtc_bazett_ms"] = None

    # --- RR interval ---
    if beat_idx < n_beats - 1:
        rr_ms = (r_peaks[beat_idx + 1] - r_peaks[beat_idx]) * 1000.0 / fs
        beat["rr_interval_ms"] = rr_ms
        beat["heart_rate_bpm"] = 60000.0 / rr_ms if rr_ms > 0 else None
    else:
        beat["rr_interval_ms"] = None
        beat["heart_rate_bpm"] = None

    return beat


def _summarize(per_beat: List[dict], r_peaks: np.ndarray, fs: int) -> Dict[str, Any]:
    """Compute median/mean summary across all beats + flags."""
    summary = {}

    def med(key):
        return _safe_median([b.get(key) for b in per_beat])

    def avg(key):
        return _safe_mean([b.get(key) for b in per_beat])

    # P wave
    p_present_count = sum(1 for b in per_beat if b.get("p_wave_present"))
    summary["p_wave_present_ratio"] = p_present_count / len(per_beat) if per_beat else 0.0
    summary["p_wave_duration_ms"] = med("p_wave_duration_ms")
    summary["p_wave_amplitude_mv"] = med("p_wave_amplitude_mv")
    summary["p_wave_flag"] = _flag(summary["p_wave_duration_ms"], "p_wave_duration_ms")

    # PR interval
    summary["pr_interval_ms"] = med("pr_interval_ms")
    summary["pr_interval_flag"] = _flag(summary["pr_interval_ms"], "pr_interval_ms")

    # PR segment
    summary["pr_segment_ms"] = med("pr_segment_ms")
    summary["pr_segment_flag"] = _flag(summary["pr_segment_ms"], "pr_segment_ms")

    # QRS
    summary["qrs_duration_ms"] = med("qrs_duration_ms")
    summary["qrs_amplitude_mv"] = med("qrs_amplitude_mv")
    summary["qrs_flag"] = _flag(summary["qrs_duration_ms"], "qrs_duration_ms")

    # ST segment
    summary["st_segment_ms"] = med("st_segment_ms")
    summary["st_deviation_mv"] = med("st_deviation_mv")
    summary["st_flag"] = _flag(summary["st_deviation_mv"], "st_deviation_mv")

    # T wave
    summary["t_wave_duration_ms"] = med("t_wave_duration_ms")
    summary["t_wave_amplitude_mv"] = med("t_wave_amplitude_mv")
    summary["t_wave_flag"] = _flag(summary["t_wave_duration_ms"], "t_wave_duration_ms")

    # QTc
    summary["qtc_bazett_ms"] = med("qtc_bazett_ms")
    summary["qtc_flag"] = _flag(summary["qtc_bazett_ms"], "qtc_ms")

    # RR / HR
    rr_vals = [b.get("rr_interval_ms") for b in per_beat if b.get("rr_interval_ms") is not None]
    summary["rr_interval_ms"] = _safe_median(rr_vals)
    summary["rr_flag"] = _flag(summary["rr_interval_ms"], "rr_interval_ms")

    hr_vals = [b.get("heart_rate_bpm") for b in per_beat if b.get("heart_rate_bpm") is not None]
    summary["heart_rate_bpm"] = _safe_mean(hr_vals)

    # SDNN / RMSSD
    if len(rr_vals) >= 2:
        rr_arr = np.array(rr_vals)
        summary["sdnn_ms"] = float(np.std(rr_arr))
        summary["rmssd_ms"] = float(np.sqrt(np.mean(np.diff(rr_arr) ** 2)))
    else:
        summary["sdnn_ms"] = 0.0
        summary["rmssd_ms"] = 0.0

    # Total beats
    summary["num_beats"] = len(per_beat)

    return summary


def _empty_summary() -> Dict[str, Any]:
    """Return a summary dict with all zeros when no features available."""
    return {
        "p_wave_present_ratio": 0.0,
        "p_wave_duration_ms": 0.0,
        "p_wave_amplitude_mv": 0.0,
        "p_wave_flag": "unavailable",
        "pr_interval_ms": 0.0,
        "pr_interval_flag": "unavailable",
        "pr_segment_ms": 0.0,
        "pr_segment_flag": "unavailable",
        "qrs_duration_ms": 0.0,
        "qrs_amplitude_mv": 0.0,
        "qrs_flag": "unavailable",
        "st_segment_ms": 0.0,
        "st_deviation_mv": 0.0,
        "st_flag": "unavailable",
        "t_wave_duration_ms": 0.0,
        "t_wave_amplitude_mv": 0.0,
        "t_wave_flag": "unavailable",
        "qtc_bazett_ms": 0.0,
        "qtc_flag": "unavailable",
        "rr_interval_ms": 0.0,
        "rr_flag": "unavailable",
        "heart_rate_bpm": 0.0,
        "sdnn_ms": 0.0,
        "rmssd_ms": 0.0,
        "num_beats": 0,
    }
