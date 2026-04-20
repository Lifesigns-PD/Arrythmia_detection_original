"""
beat_morphology.py — Per-Beat Discriminative Features for PVC/PAC/Normal
=========================================================================
Extracts features that directly separate:
  - PVC  (wide QRS, no preceding P, compensatory pause, abnormal T axis)
  - PAC  (narrow QRS, early/ectopic P, short coupling interval, incomplete comp. pause)
  - Normal sinus beat

Also captures:
  - Inverted R peaks (negative QRS polarity)
  - T-wave inversion
  - Delta waves (WPW pre-excitation)
  - Compensatory vs non-compensatory pause

Electrophysiology basis:
  PVC:
    - QRS duration > 120 ms (wide complex)
    - No preceding P wave (or dissociated P)
    - T wave in opposite direction to QRS (discordant)
    - Compensatory pause: post-PVC RR ≈ 2× normal RR
    - R/S ratio < 1 in V1-type leads (LBBB pattern) or > 1 (RBBB pattern)
    - QRS onset sudden (no delta), steep descent
  PAC:
    - QRS narrow (< 120 ms) — conducted normally through bundle
    - Preceding P wave with different morphology (ectopic / inverted / biphasic)
    - Short coupling interval (early compared to normal sinus RR)
    - Non-compensatory pause (sinus node reset)
    - Normal R/S ratio in QRS
"""

import numpy as np
from typing import Dict, List, Optional


def compute_beat_discriminators(
    signal: np.ndarray,
    per_beat: List[Dict],
    r_peaks: np.ndarray,
    fs: int = 125,
) -> Dict[str, Optional[float]]:
    """
    Compute segment-level summary of beat discriminator features.

    Parameters
    ----------
    signal   : preprocessed ECG
    per_beat : delineation output per beat (from delineate_v3)
    r_peaks  : R-peak indices
    fs       : sampling rate

    Returns
    -------
    dict with keys listed in BEAT_DISC_FEATURES
    """
    if len(r_peaks) < 2 or not per_beat:
        return {k: None for k in BEAT_DISC_FEATURES}

    rr_ms = np.diff(r_peaks).astype(float) / fs * 1000
    rr_ms = rr_ms[(rr_ms > 200) & (rr_ms < 3000)]
    median_rr = float(np.median(rr_ms)) if len(rr_ms) > 0 else 800.0

    per_beat_feats = [
        _single_beat_features(signal, per_beat, r_peaks, i, median_rr, fs)
        for i in range(len(per_beat))
    ]

    def _frac(key, condition_fn):
        vals = [b[key] for b in per_beat_feats if b[key] is not None]
        if not vals:
            return None
        return float(np.mean([1.0 if condition_fn(v) else 0.0 for v in vals]))

    def _med(key):
        vals = [b[key] for b in per_beat_feats if b[key] is not None]
        return float(np.median(vals)) if vals else None

    return {
        # QRS width
        "qrs_wide_fraction":        _frac("qrs_duration_ms", lambda v: v > 120),
        "mean_qrs_duration_ms":     _med("qrs_duration_ms"),
        "qrs_duration_std_ms":      _std([b["qrs_duration_ms"] for b in per_beat_feats]),

        # P-wave presence and morphology
        "p_absent_fraction":        _frac("p_morphology", lambda v: v == "absent"),
        "p_inverted_fraction":      _frac("p_morphology", lambda v: v == "inverted"),
        "p_biphasic_fraction":      _frac("p_morphology", lambda v: v == "biphasic"),

        # Coupling interval (early beats = PAC/PVC)
        "mean_coupling_ratio":      _med("coupling_ratio"),
        "short_coupling_fraction":  _frac("coupling_ratio", lambda v: v < 0.85),

        # Compensatory pause (PVC marker)
        "compensatory_pause_fraction": _frac("compensatory_pause", lambda v: v is True),

        # T-wave discordance (T opposite to QRS — PVC marker)
        "t_discordant_fraction":    _frac("t_discordant", lambda v: v is True),
        "t_inverted_fraction":      _frac("t_inverted", lambda v: v is True),

        # QRS polarity
        "qrs_negative_fraction":    _frac("qrs_polarity", lambda v: v == "negative"),

        # R/S ratio (QRS morphology)
        "mean_rs_ratio":            _med("rs_ratio"),
        "rs_ratio_std":             _std([b["rs_ratio"] for b in per_beat_feats]),

        # Q wave depth
        "mean_q_depth":             _med("q_depth"),
        "pathological_q_fraction":  _frac("q_depth", lambda v: v < -0.1),  # > 25% R amp

        # S wave depth
        "mean_s_depth":             _med("s_depth"),

        # Delta wave (WPW)
        "delta_wave_fraction":      _frac("delta_wave", lambda v: v is True),

        # Beat classification confidence scores
        "pvc_score_mean":           _med("pvc_score"),
        "pac_score_mean":           _med("pac_score"),
    }


BEAT_DISC_FEATURES = [
    "qrs_wide_fraction", "mean_qrs_duration_ms", "qrs_duration_std_ms",
    "p_absent_fraction", "p_inverted_fraction", "p_biphasic_fraction",
    "mean_coupling_ratio", "short_coupling_fraction",
    "compensatory_pause_fraction",
    "t_discordant_fraction", "t_inverted_fraction",
    "qrs_negative_fraction",
    "mean_rs_ratio", "rs_ratio_std",
    "mean_q_depth", "pathological_q_fraction",
    "mean_s_depth",
    "delta_wave_fraction",
    "pvc_score_mean", "pac_score_mean",
]


# ─────────────────────────────────────────────────────────────────────────────

def _single_beat_features(
    signal: np.ndarray,
    per_beat: List[Dict],
    r_peaks: np.ndarray,
    i: int,
    median_rr_ms: float,
    fs: int,
) -> Dict:
    """Compute discriminative features for beat i."""
    b  = per_beat[i]
    r  = int(r_peaks[i])
    out: Dict = {k: None for k in [
        "qrs_duration_ms", "p_morphology", "coupling_ratio",
        "compensatory_pause", "t_discordant", "t_inverted",
        "qrs_polarity", "rs_ratio", "q_depth", "s_depth",
        "delta_wave", "pvc_score", "pac_score",
    ]}

    # ── QRS duration ──────────────────────────────────────────────────────────
    qrs_on  = b.get("qrs_onset")
    qrs_off = b.get("qrs_offset")
    if qrs_on is not None and qrs_off is not None and qrs_off > qrs_on:
        out["qrs_duration_ms"] = (qrs_off - qrs_on) * 1000 / fs

    # ── P morphology (already classified in delineation) ─────────────────────
    out["p_morphology"] = b.get("p_morphology", "absent")

    # ── QRS polarity ─────────────────────────────────────────────────────────
    out["qrs_polarity"] = b.get("qrs_polarity", "positive")

    # ── Coupling interval ratio ───────────────────────────────────────────────
    # = (RR before this beat) / median_rr; < 0.85 = early (ectopic)
    if i > 0:
        prev_r = int(r_peaks[i - 1])
        coupling_ms = (r - prev_r) * 1000 / fs
        out["coupling_ratio"] = coupling_ms / median_rr_ms if median_rr_ms > 0 else None

    # ── Compensatory pause ────────────────────────────────────────────────────
    # PVC: post-ectopic RR ≈ 2× normal (compensatory)
    # PAC: post-ectopic RR < 2× normal (non-compensatory, sinus reset)
    if i > 0 and i < len(r_peaks) - 1:
        prev_r = int(r_peaks[i - 1])
        next_r = int(r_peaks[i + 1])
        pre_rr  = (r - prev_r) * 1000 / fs
        post_rr = (next_r - r) * 1000 / fs
        # Compensatory: pre + post ≈ 2× median_rr (±15%)
        sum_rr = pre_rr + post_rr
        out["compensatory_pause"] = bool(
            abs(sum_rr - 2 * median_rr_ms) < 0.15 * 2 * median_rr_ms
        )

    # ── R/S ratio ─────────────────────────────────────────────────────────────
    r_amp = float(signal[r]) if 0 <= r < len(signal) else 0.0
    s_pk  = b.get("s_peak")
    if s_pk is not None and 0 <= s_pk < len(signal):
        s_amp = float(signal[s_pk])
        denom = abs(s_amp)
        out["rs_ratio"] = r_amp / denom if denom > 1e-4 else None

    # ── Q depth (from delineation) ────────────────────────────────────────────
    q_pk = b.get("q_peak")
    if q_pk is not None and 0 <= q_pk < len(signal):
        out["q_depth"] = b.get("q_depth", float(signal[q_pk]))

    # ── S depth ───────────────────────────────────────────────────────────────
    if s_pk is not None and 0 <= s_pk < len(signal):
        out["s_depth"] = b.get("s_depth", float(signal[s_pk]))

    # ── T-wave inversion flag ─────────────────────────────────────────────────
    out["t_inverted"] = b.get("t_inverted", False)

    # ── T-wave discordance (T polarity opposite to QRS — classic PVC sign) ───
    t_pk = b.get("t_peak")
    if t_pk is not None and 0 <= t_pk < len(signal):
        t_amp  = float(signal[t_pk])
        # Discordant: T and R have opposite signs
        out["t_discordant"] = bool((r_amp > 0 and t_amp < -0.05) or
                                    (r_amp < 0 and t_amp > 0.05))

    # ── Delta wave ────────────────────────────────────────────────────────────
    out["delta_wave"] = b.get("delta_wave", False)

    # ── Rule-based PVC / PAC scores (0–1) ─────────────────────────────────────
    out["pvc_score"] = _pvc_score(out, median_rr_ms)
    out["pac_score"] = _pac_score(out, median_rr_ms)

    return out


def _pvc_score(feat: Dict, median_rr_ms: float) -> float:
    """
    Heuristic PVC likelihood score [0–1].
    High score → PVC-like morphology.
    """
    score = 0.0
    weights = 0.0

    def _add(condition, w):
        nonlocal score, weights
        weights += w
        if condition:
            score += w

    qrs_ms = feat.get("qrs_duration_ms")
    _add(qrs_ms is not None and qrs_ms > 120, 3.0)    # wide QRS: strong PVC marker
    _add(qrs_ms is not None and qrs_ms > 140, 2.0)    # very wide: extra weight
    _add(feat.get("p_morphology") in ("absent", None), 2.5)  # no P wave
    _add(feat.get("compensatory_pause") is True, 2.0)
    _add(feat.get("t_discordant") is True, 2.0)
    _add(feat.get("coupling_ratio") is not None and feat["coupling_ratio"] < 0.85, 1.0)
    _add(feat.get("qrs_polarity") == "negative", 1.0)  # LBBB pattern
    q_d = feat.get("q_depth")
    _add(q_d is not None and q_d < -0.1, 0.5)

    return float(score / weights) if weights > 0 else 0.0


def _pac_score(feat: Dict, median_rr_ms: float) -> float:
    """
    Heuristic PAC likelihood score [0–1].
    High score → PAC-like morphology.
    """
    score = 0.0
    weights = 0.0

    def _add(condition, w):
        nonlocal score, weights
        weights += w
        if condition:
            score += w

    qrs_ms = feat.get("qrs_duration_ms")
    _add(qrs_ms is not None and qrs_ms < 120, 2.5)    # narrow QRS: PAC key criterion
    _add(feat.get("p_morphology") in ("inverted", "biphasic"), 3.0)  # ectopic P
    _add(feat.get("p_morphology") == "absent", 0.5)   # absent P also possible in PAC
    _add(feat.get("compensatory_pause") is False, 1.5) # non-compensatory
    _add(feat.get("coupling_ratio") is not None and feat["coupling_ratio"] < 0.85, 1.5)
    _add(feat.get("t_discordant") is False and feat.get("t_discordant") is not None, 1.0)

    return float(score / weights) if weights > 0 else 0.0


def _std(vals: list) -> Optional[float]:
    clean = [v for v in vals if v is not None]
    return float(np.std(clean, ddof=1)) if len(clean) >= 2 else None
