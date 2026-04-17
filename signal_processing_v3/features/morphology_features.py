"""
morphology_features.py — Morphological ECG Features
=====================================================
Extracts morphology-level features from per-beat delineation results
(output of delineate_v3) and raw signal.
"""

import numpy as np
from scipy.integrate import trapezoid
from typing import Dict, List, Optional


def compute_morphology_features(
    signal: np.ndarray,
    per_beat: List[Dict],
    r_peaks: np.ndarray,
    fs: int = 125,
) -> Dict[str, Optional[float]]:
    """
    Parameters
    ----------
    signal   : preprocessed ECG signal
    per_beat : list of dicts from delineate_v3()["per_beat"]
    r_peaks  : R-peak sample indices
    fs       : sampling rate

    Returns
    -------
    dict with keys:
      qrs_duration_ms, qrs_area, pr_interval_ms, qt_interval_ms,
      qtc_bazett_ms, st_elevation_mv, st_slope, t_wave_asymmetry,
      r_s_ratio, p_wave_duration_ms, t_wave_amplitude_mv,
      r_amplitude_mv, qrs_amplitude_ms_product
    """
    keys = [
        "qrs_duration_ms", "qrs_area", "pr_interval_ms", "qt_interval_ms",
        "qtc_bazett_ms", "st_elevation_mv", "st_slope", "t_wave_asymmetry",
        "r_s_ratio", "p_wave_duration_ms", "t_wave_amplitude_mv",
        "r_amplitude_mv", "qrs_amplitude_ms_product",
    ]
    empty = {k: None for k in keys}
    if not per_beat or len(r_peaks) == 0:
        return empty

    def safe_median(vals):
        clean = [v for v in vals if v is not None and not np.isnan(v)]
        return float(np.median(clean)) if clean else None

    # ── QRS duration ──
    qrs_dur = [
        (b.get("qrs_offset", 0) - b.get("qrs_onset", 0)) * 1000 / fs
        for b in per_beat
        if b.get("qrs_onset") is not None and b.get("qrs_offset") is not None
        and b["qrs_offset"] > b["qrs_onset"]
    ]
    qrs_dur = [v for v in qrs_dur if 40 <= v <= 200]

    # ── QRS area (integral of |signal| over QRS) ──
    qrs_areas = []
    for b in per_beat:
        on, off = b.get("qrs_onset"), b.get("qrs_offset")
        if on is not None and off is not None and off > on and off < len(signal):
            qrs_areas.append(float(trapezoid(np.abs(signal[on:off+1])) / fs))

    # ── PR interval ──
    pr_ms = [
        (b.get("qrs_onset", 0) - b.get("p_onset", 0)) * 1000 / fs
        for b in per_beat
        if b.get("p_onset") is not None and b.get("qrs_onset") is not None
        and b["qrs_onset"] > b["p_onset"]
    ]
    pr_ms = [v for v in pr_ms if 60 <= v <= 400]

    # ── QT interval ──
    qt_ms = [
        (b.get("t_offset", 0) - b.get("qrs_onset", 0)) * 1000 / fs
        for b in per_beat
        if b.get("qrs_onset") is not None and b.get("t_offset") is not None
        and b["t_offset"] > b["qrs_onset"]
    ]
    qt_ms = [v for v in qt_ms if 200 <= v <= 600]

    # ── QTc Bazett ──
    rr_ms = np.diff(r_peaks).astype(float) / fs * 1000
    rr_ms = rr_ms[(rr_ms > 250) & (rr_ms < 2500)]
    qtc_list = []
    for qt, rr in zip(qt_ms, rr_ms[:len(qt_ms)]):
        if rr > 0:
            qtc_list.append(qt / np.sqrt(rr / 1000))

    # ── ST elevation ──
    st_elevs = []
    for b in per_beat:
        off = b.get("qrs_offset")
        ton = b.get("t_onset")
        if off is not None and ton is not None and ton > off + 2:
            mid = (off + ton) // 2
            if 0 <= mid < len(signal):
                p_on = b.get("p_onset")
                baseline = float(signal[p_on]) if (p_on is not None and 0 <= p_on < len(signal)) else 0.0
                st_elevs.append(float(signal[mid]) - baseline)

    # ── ST slope ──
    st_slopes = []
    for b in per_beat:
        off = b.get("qrs_offset")
        ton = b.get("t_onset")
        if off is not None and ton is not None and ton > off + 4:
            seg = signal[off:ton]
            x   = np.arange(len(seg))
            if len(x) > 2:
                slope = float(np.polyfit(x, seg, 1)[0] * fs / 1000)  # mV/s
                st_slopes.append(slope)

    # ── T-wave asymmetry ──
    t_asym = []
    for b in per_beat:
        ton, tpk, toff = b.get("t_onset"), b.get("t_peak"), b.get("t_offset")
        if all(x is not None for x in [ton, tpk, toff]) and tpk > ton and toff > tpk:
            rise = tpk - ton
            fall = toff - tpk
            if rise + fall > 0:
                t_asym.append(float((fall - rise) / (fall + rise)))

    # ── R/S ratio ──
    rs_ratios = []
    for r in r_peaks:
        r = int(r)
        lo = max(0, r - int(0.05 * fs))
        hi = min(len(signal), r + int(0.05 * fs))
        seg = signal[lo:hi]
        R_amp = float(signal[r]) if 0 <= r < len(signal) else 0.0
        S_amp = float(np.min(seg)) if len(seg) > 0 else 0.0
        denom = abs(S_amp)
        if denom > 0:
            rs_ratios.append(R_amp / denom)

    # ── P-wave duration ──
    p_dur = [
        (b.get("p_offset", 0) - b.get("p_onset", 0)) * 1000 / fs
        for b in per_beat
        if b.get("p_onset") is not None and b.get("p_offset") is not None
        and b["p_offset"] > b["p_onset"]
    ]
    p_dur = [v for v in p_dur if 20 <= v <= 200]

    # ── T-wave amplitude ──
    t_amps = [
        float(signal[b["t_peak"]]) for b in per_beat
        if b.get("t_peak") is not None and 0 <= b["t_peak"] < len(signal)
    ]

    # ── R amplitude ──
    r_amps = [float(signal[int(r)]) for r in r_peaks if 0 <= int(r) < len(signal)]

    qrs_dur_med = safe_median(qrs_dur)
    r_amp_med   = safe_median(r_amps)
    qrs_amp_ms  = (qrs_dur_med * r_amp_med) if (qrs_dur_med and r_amp_med) else None

    # ── PR interval sentinel ──
    # When no beats have a detectable P-wave (e.g. AF, high-degree AV block),
    # pr_ms will be empty and safe_median returns None.  Using 0.0 as the
    # fill (which extraction.py would apply) is clinically wrong: a PR of 0
    # looks like WPW pre-excitation.  -1.0 is the agreed sentinel for
    # "PR unmeasurable due to absent P-wave" — the model is trained to treat
    # this sentinel as a distinct signal, not a near-zero interval.
    pr_med = safe_median(pr_ms)
    if pr_med is None:
        pr_med = -1.0

    return {
        "qrs_duration_ms":          qrs_dur_med,
        "qrs_area":                 safe_median(qrs_areas),
        "pr_interval_ms":           pr_med,
        "qt_interval_ms":           safe_median(qt_ms),
        "qtc_bazett_ms":            safe_median(qtc_list),
        "st_elevation_mv":          safe_median(st_elevs),
        "st_slope":                 safe_median(st_slopes),
        "t_wave_asymmetry":         safe_median(t_asym),
        "r_s_ratio":                safe_median(rs_ratios),
        "p_wave_duration_ms":       safe_median(p_dur),
        "t_wave_amplitude_mv":      safe_median(t_amps),
        "r_amplitude_mv":           r_amp_med,
        "qrs_amplitude_ms_product": qrs_amp_ms,
    }
