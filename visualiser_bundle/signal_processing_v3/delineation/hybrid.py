"""
hybrid.py — Hybrid Delineator (Wavelet + Template Matching)
============================================================
Primary: Wavelet CWT
Refinement: Template matching (patient-specific)
Fallback: NeuroKit2 DWT (V2 logic)
Final fallback: Fixed heuristic windows

Returns per-beat list AND segment-level summary dict.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from .wavelet_delineation import delineate_beats_wavelet
from .template_matching   import refine_delineation_template


def delineate_v3(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: int = 125,
    use_neurokit_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Full V3 delineation pipeline.

    Returns
    -------
    dict with keys:
      "per_beat"  : list of per-beat dicts
      "summary"   : segment-level medians + flags
      "method"    : str — "wavelet+template", "neurokit", or "heuristic"
    """
    if r_peaks is None or len(r_peaks) < 1:
        return {"per_beat": [], "summary": _empty_summary(), "method": "none"}

    r_peaks = np.asarray(r_peaks, dtype=int)

    # ── Step 1: Wavelet delineation ──
    try:
        wav_beats = delineate_beats_wavelet(signal, r_peaks, fs)
    except Exception:
        wav_beats = [{k: None for k in _KEYS} for _ in r_peaks]

    # ── Step 2: Template refinement ──
    try:
        final_beats = refine_delineation_template(signal, r_peaks, wav_beats, fs)
    except Exception:
        final_beats = wav_beats

    # ── Step 3: NeuroKit2 fallback for beats with too many None values ──
    if use_neurokit_fallback:
        final_beats = _fill_with_neurokit(signal, r_peaks, final_beats, fs)

    # ── Step 4: Compute summary ──
    summary = _summarize(final_beats, r_peaks, signal, fs)

    # ── Count method quality ──
    none_count = sum(
        1 for beat in final_beats
        for key in ["qrs_onset", "p_onset", "t_offset"]
        if beat.get(key) is None
    )
    total_possible = len(final_beats) * 3
    if none_count == 0:
        method = "wavelet+template"
    elif none_count < total_possible * 0.3:
        method = "wavelet+template+nk"
    else:
        method = "heuristic_heavy"

    return {
        "per_beat": final_beats,
        "summary":  summary,
        "method":   method,
    }


# ─────────────────────────────────────────────────────────────────────────────

_KEYS = ["p_onset", "p_peak", "p_offset", "qrs_onset", "qrs_offset",
         "t_onset", "t_peak", "t_offset"]


def _fill_with_neurokit(signal, r_peaks, beats, fs):
    """For any beat missing key fiducial points, try NeuroKit2."""
    try:
        import neurokit2 as nk
        from scipy.signal import butter, filtfilt
        b, a = butter(2, [1.0/(0.5*fs), 40.0/(0.5*fs)], btype="band")
        smooth = filtfilt(b, a, signal.astype(float))
        _, waves = nk.ecg_delineate(smooth, r_peaks, sampling_rate=fs,
                                     method="dwt", show=False)
    except Exception:
        return beats

    nk_key_map = {
        "p_onset":    "ECG_P_Onsets",
        "p_peak":     "ECG_P_Peaks",
        "p_offset":   "ECG_P_Offsets",
        "qrs_onset":  "ECG_R_Onsets",
        "qrs_offset": "ECG_R_Offsets",
        "t_onset":    "ECG_T_Onsets",
        "t_peak":     "ECG_T_Peaks",
        "t_offset":   "ECG_T_Offsets",
    }

    filled = []
    for i, beat in enumerate(beats):
        new_beat = dict(beat)
        for our_key, nk_key in nk_key_map.items():
            if new_beat.get(our_key) is None:
                arr = waves.get(nk_key, [])
                if i < len(arr):
                    val = arr[i]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        new_beat[our_key] = int(val)
        filled.append(new_beat)
    return filled


def _summarize(beats: List[Dict], r_peaks: np.ndarray,
               signal: np.ndarray, fs: int) -> Dict:
    def safe_ms(vals):
        clean = [v for v in vals if v is not None]
        return float(np.median(clean) * 1000 / fs) if clean else None

    def safe_amp(indices):
        vals = []
        for idx in indices:
            if idx is not None and 0 <= idx < len(signal):
                vals.append(float(signal[idx]))
        return float(np.median(vals)) if vals else None

    rr_ms  = np.diff(r_peaks) / fs * 1000 if len(r_peaks) > 1 else np.array([])
    rr_ms  = rr_ms[(rr_ms > 200) & (rr_ms < 3000)]
    mean_hr = float(60000 / np.mean(rr_ms)) if len(rr_ms) > 0 else None

    # QRS duration
    qrs_durations = [
        (b.get("qrs_offset", None) or 0) - (b.get("qrs_onset", None) or 0)
        for b in beats
        if b.get("qrs_onset") is not None and b.get("qrs_offset") is not None
    ]

    # PR interval
    pr_ms = [
        ((b.get("qrs_onset") or 0) - (b.get("p_onset") or 0)) / fs * 1000
        for b in beats
        if b.get("p_onset") is not None and b.get("qrs_onset") is not None
        and b.get("qrs_onset", 0) > b.get("p_onset", 0)
    ]
    pr_ms = [v for v in pr_ms if 60 <= v <= 400]

    # QT interval
    qt_ms = [
        ((b.get("t_offset") or 0) - (b.get("qrs_onset") or 0)) / fs * 1000
        for b in beats
        if b.get("qrs_onset") is not None and b.get("t_offset") is not None
        and b.get("t_offset", 0) > b.get("qrs_onset", 0)
    ]

    # QTc Bazett
    qtc_list = []
    for qt, rr in zip(qt_ms, rr_ms[:len(qt_ms)]):
        if rr > 0:
            qtc_list.append(qt / np.sqrt(rr / 1000))

    # ST deviation
    st_devs = []
    for b in beats:
        off = b.get("qrs_offset")
        ton = b.get("t_onset")
        if off is not None and ton is not None and ton > off:
            mid = (off + ton) // 2
            if 0 <= mid < len(signal):
                # Baseline = signal at P-onset (or 0 if missing)
                p_on = b.get("p_onset")
                baseline = float(signal[p_on]) if (p_on is not None and 0 <= p_on < len(signal)) else 0.0
                st_devs.append(float(signal[mid]) - baseline)

    p_present = sum(1 for b in beats if b.get("p_onset") is not None) / max(len(beats), 1)

    return {
        "mean_hr_bpm":        mean_hr,
        "qrs_duration_ms":    float(np.median(qrs_durations) * 1000 / fs) if qrs_durations else None,
        "pr_interval_ms":     float(np.median(pr_ms)) if pr_ms else None,
        "qt_interval_ms":     float(np.median(qt_ms)) if qt_ms else None,
        "qtc_ms":             float(np.median(qtc_list)) if qtc_list else None,
        "st_deviation_mv":    float(np.median(st_devs)) if st_devs else None,
        "p_wave_amplitude_mv": safe_amp([b.get("p_peak") for b in beats]),
        "t_wave_amplitude_mv": safe_amp([b.get("t_peak") for b in beats]),
        "qrs_amplitude_mv":   float(np.median([float(signal[int(r)]) for r in r_peaks
                                                if 0 <= int(r) < len(signal)])) if len(r_peaks) else None,
        "p_wave_present_ratio": p_present,
        "n_beats":            len(beats),
    }


def _empty_summary() -> Dict:
    keys = ["mean_hr_bpm", "qrs_duration_ms", "pr_interval_ms", "qt_interval_ms",
            "qtc_ms", "st_deviation_mv", "p_wave_amplitude_mv", "t_wave_amplitude_mv",
            "qrs_amplitude_mv", "p_wave_present_ratio", "n_beats"]
    return {k: None for k in keys}
