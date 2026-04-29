"""
beat_classifier.py — Post-ML PVC/PAC refinement via template correlation.

Ported from BATCH_PROCESS/lifesigns_engine.py classify_beats() (lines 672-835).

Runs AFTER the ectopy ML model in rhythm_orchestrator.py Step 4C.
Overrides low-confidence ML ectopy calls using:
  - Pearson template correlation against the segment's own sinus beats
  - Ashman phenomenon (long-short RR → aberrant PAC mimicking PVC)
  - Preceding T-wave deformation (retrograde P-wave buried in T)
  - Multi-criteria scoring (PVC and PAC thresholds ≥ 3.0)

Only overrides when:
  1. ML ectopy confidence is below 0.99 (high-confidence ML is trusted)
  2. Beat coupling ratio < 0.92 (premature — analysis is applicable)
  3. ≥ 3 normal beats exist in the window to build a reliable sinus template
"""
from __future__ import annotations

import numpy as np
from typing import Any


# ── Public entry point ────────────────────────────────────────────────────────

def refine_beat_labels(
    beat_events: list[dict],
    signal: np.ndarray,
    r_peaks: list[int] | np.ndarray,
    per_beat_delineation: list[dict],
    fs: int = 125,
    ml_conf_trust_threshold: float = 0.99,
) -> list[dict]:
    """
    Refine per-beat ectopy labels using template correlation.

    Parameters
    ----------
    beat_events : list of dicts from ml_prediction["ectopy"]["beat_events"]
                  Each dict has keys: beat_idx, peak_sample, label, conf
    signal      : preprocessed (cleaned) 10s ECG window
    r_peaks     : R-peak sample indices from V3 ensemble detector
    per_beat_delineation : list of per-beat dicts from V3 DWT delineation,
                           each with keys: P, QRS, T (onset/peak/offset/morphology tuples)
    fs          : sampling rate in Hz
    ml_conf_trust_threshold : ML calls at or above this confidence are kept as-is

    Returns
    -------
    Refined beat_events list (same structure, labels may be updated)
    """
    if not beat_events or len(r_peaks) < 3 or not per_beat_delineation:
        return beat_events

    r_peaks = np.asarray(r_peaks, dtype=int)
    signal  = np.asarray(signal, dtype=np.float32)

    rr_ints   = np.diff(r_peaks).astype(float)
    median_rr = float(np.median(rr_ints))
    if median_rr <= 0:
        return beat_events

    # Build sinus template from beats the ML model called Normal/None
    normal_indices = {b["beat_idx"] for b in beat_events
                      if b.get("label", "None") in ("None", "Normal", "Sinus Rhythm")}
    # Also treat beats not in beat_events as normal (ML only emits ectopic beats)
    all_indices = set(range(len(r_peaks)))
    normal_indices |= (all_indices - {b["beat_idx"] for b in beat_events})

    template = _build_template(signal, r_peaks, normal_indices, fs)
    if template is None:
        return beat_events

    # Reference T-wave amplitude from delineation
    t_amps = []
    for b in per_beat_delineation:
        t_pk = b.get("T", (None, None, None, None))[2]
        if t_pk is not None and 0 <= t_pk < len(signal):
            t_amps.append(abs(float(signal[t_pk])))
    ref_t_amp = float(np.median(t_amps)) if t_amps else 0.0

    # Build a lookup: beat_idx → beat_event dict (for mutation)
    event_by_idx = {b["beat_idx"]: b for b in beat_events}
    total_beats  = len(r_peaks)

    refined = []
    for beat_event in beat_events:
        i    = beat_event["beat_idx"]
        conf = float(beat_event.get("conf", 0.0))

        # Boundary guard — first and last beat never reliable
        if i == 0 or i == total_beats - 1:
            refined.append(beat_event)
            continue

        # Trust high-confidence ML calls
        if conf >= ml_conf_trust_threshold:
            refined.append(beat_event)
            continue

        # Only refine premature beats
        if i >= len(r_peaks):
            refined.append(beat_event)
            continue

        rr_prev = float(r_peaks[i] - r_peaks[i - 1]) if i > 0 else median_rr
        rr_next = float(r_peaks[i + 1] - r_peaks[i]) if i < len(r_peaks) - 1 else median_rr
        coupling_ratio  = rr_prev / (median_rr + 1e-8)
        pause_sum_ratio = (rr_prev + rr_next) / (median_rr + 1e-8)

        if coupling_ratio >= 0.92:
            # Not sufficiently premature — keep ML label
            refined.append(beat_event)
            continue

        # Get delineation for this beat
        delin = per_beat_delineation[i] if i < len(per_beat_delineation) else {}
        qrs   = delin.get("QRS", (None, None))
        t_inf = delin.get("T",   (None, None, None, None))
        p_inf = delin.get("P",   (None, None, None, None))

        q_on, q_off = qrs[0], qrs[1]
        qrs_dur_ms  = ((q_off - q_on) / fs * 1000.0) if (q_on is not None and q_off is not None) else 0.0

        p_morph    = p_inf[3] if len(p_inf) > 3 else None
        t_morph    = t_inf[3] if len(t_inf) > 3 else None
        p_absent   = p_morph in ("Absent", "Explicitly Absent", None)
        p_inverted = p_morph in ("Inverted", "Explicitly Inverted")

        r_amp        = delin.get("R_amp", 1.0) or 1.0
        qrs_inverted = float(r_amp) < 0
        t_discordant = ((t_morph == "Inverted" and not qrs_inverted) or
                        (t_morph == "Normal"   and     qrs_inverted))

        # Template correlation
        corr = _template_correlation(signal, r_peaks[i], template, fs)

        # Ashman phenomenon
        ashman = False
        if i >= 2:
            rr_pre_prev = float(r_peaks[i - 1] - r_peaks[i - 2])
            ashman = (rr_pre_prev > 1.30 * median_rr) and (rr_prev < 0.92 * median_rr)

        # Preceding T-wave deformation
        prev_t_deformed = False
        if ref_t_amp > 0.01 and i > 0:
            prev_delin = per_beat_delineation[i - 1] if i - 1 < len(per_beat_delineation) else {}
            prev_is_normal = (i - 1) in normal_indices
            if prev_is_normal:
                prev_t_pk = (prev_delin.get("T", (None, None, None, None)) or (None,)*4)[2]
                if prev_t_pk is not None and 0 <= prev_t_pk < len(signal):
                    prev_t_amp      = abs(float(signal[prev_t_pk]))
                    prev_t_deformed = prev_t_amp > 1.30 * ref_t_amp

        # PVC scoring
        pvc = 0.0
        if qrs_dur_ms > 120:                                          pvc += 2.0
        if t_discordant:                                              pvc += 1.5
        if p_absent:                                                  pvc += 1.5
        if coupling_ratio < 0.88:                                     pvc += 1.0
        if coupling_ratio < 0.92 and pause_sum_ratio > 1.85:         pvc += 1.0
        if corr is not None and corr < 0.60:                         pvc += 2.5

        # PAC scoring
        pac = 0.0
        if qrs_dur_ms < 110:                                          pac += 1.0
        if p_inverted:                                                pac += 1.5
        if not p_absent:                                              pac += 0.5
        if coupling_ratio < 0.88:                                     pac += 1.0
        if coupling_ratio < 0.92 and pause_sum_ratio < 1.90:         pac += 1.0
        if corr is not None and corr >= 0.85:                        pac += 2.5
        if ashman:                                                    pac += 1.5
        if prev_t_deformed:                                           pac += 1.0

        # Soft correlation adjustment
        if corr is not None and corr >= 0.85:
            pvc -= 1.0
            pac += 0.5

        # Decision
        new_event = dict(beat_event)
        if pvc >= 3.0 and pvc > pac:
            new_event["label"] = "PVC"
            new_event["_refined_by"] = "template_correlation"
        elif pac >= 3.0 and coupling_ratio < 0.92:
            new_event["label"] = "PAC"
            new_event["_refined_by"] = "template_correlation"

        refined.append(new_event)

    return refined


# ── Signal-processing-only entry point (no ML needed) ────────────────────────

def classify_beats_sp(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    per_beat_delineation: list[dict],
    fs: int = 125,
) -> list[dict]:
    """
    Classify every beat as Normal / PVC / PAC using signal processing only.
    No ML model is used. Suitable for the diagnostic viewer.

    Returns a list of dicts (one per beat):
      {
        "beat_idx":      int,
        "peak_sample":   int,
        "label":         "Normal" | "PVC" | "PAC",
        "pvc_score":     float,
        "pac_score":     float,
        "corr":          float | None,   # Pearson r vs sinus template
        "coupling_ratio": float,
        "qrs_dur_ms":    float,
      }
    """
    r_peaks = np.asarray(r_peaks, dtype=int)
    signal  = np.asarray(signal,  dtype=np.float32)
    n       = len(r_peaks)
    if n < 2:
        return []

    rr_ints   = np.diff(r_peaks).astype(float)
    median_rr = float(np.median(rr_ints))
    if median_rr <= 0:
        return []

    # Step 1: identify normal beats by coupling ratio alone (first pass)
    coupling = np.ones(n)
    for i in range(1, n):
        coupling[i] = (r_peaks[i] - r_peaks[i - 1]) / (median_rr + 1e-8)

    normal_indices = {i for i in range(n) if coupling[i] >= 0.92}

    # Step 2: build sinus template from normal beats
    template = _build_template(signal, r_peaks, normal_indices, fs)

    # Step 3: reference T-wave amplitude
    t_amps = []
    for b in per_beat_delineation:
        t_pk = b.get("t_peak")
        if t_pk is not None and 0 <= t_pk < len(signal):
            t_amps.append(abs(float(signal[t_pk])))
    ref_t_amp = float(np.median(t_amps)) if t_amps else 0.0

    # Step 4: score every beat
    results = []
    for i in range(n):
        peak_sample = int(r_peaks[i])
        rr_prev = float(r_peaks[i] - r_peaks[i - 1]) if i > 0 else median_rr
        rr_next = float(r_peaks[i + 1] - r_peaks[i]) if i < n - 1 else median_rr

        coupling_ratio  = rr_prev / (median_rr + 1e-8)
        pause_sum_ratio = (rr_prev + rr_next) / (median_rr + 1e-8)

        # Delineation for this beat — use flat keys (viewer style)
        delin    = per_beat_delineation[i] if i < len(per_beat_delineation) else {}
        q_on     = delin.get("qrs_onset")
        q_off    = delin.get("qrs_offset")
        qrs_dur_ms = ((q_off - q_on) / fs * 1000.0) if (q_on is not None and q_off is not None) else 0.0

        p_morph    = delin.get("p_morphology")
        t_inverted = delin.get("t_inverted", False)
        qrs_pol    = delin.get("qrs_polarity", "positive")

        p_absent   = p_morph in ("Absent", "Explicitly Absent", None, "absent")
        p_inverted = p_morph in ("Inverted", "Explicitly Inverted", "inverted")
        qrs_inv    = qrs_pol in ("negative", "inverted")
        t_discordant = (t_inverted and not qrs_inv) or (not t_inverted and qrs_inv)

        # Template correlation
        corr = _template_correlation(signal, peak_sample, template, fs)

        # Ashman phenomenon
        ashman = False
        if i >= 2:
            rr_pre_prev = float(r_peaks[i - 1] - r_peaks[i - 2])
            ashman = (rr_pre_prev > 1.30 * median_rr) and (rr_prev < 0.92 * median_rr)

        # Preceding T-wave deformation
        prev_t_deformed = False
        if ref_t_amp > 0.01 and i > 0:
            prev_delin = per_beat_delineation[i - 1] if i - 1 < len(per_beat_delineation) else {}
            prev_t_pk  = prev_delin.get("t_peak")
            if prev_t_pk is not None and 0 <= prev_t_pk < len(signal):
                prev_t_amp      = abs(float(signal[prev_t_pk]))
                prev_t_deformed = prev_t_amp > 1.30 * ref_t_amp

        # PVC scoring
        pvc = 0.0
        if qrs_dur_ms > 120:                                          pvc += 2.0
        if t_discordant:                                              pvc += 1.5
        if p_absent:                                                  pvc += 1.5
        if coupling_ratio < 0.88:                                     pvc += 1.0
        if coupling_ratio < 0.92 and pause_sum_ratio > 1.85:         pvc += 1.0
        if corr is not None and corr < 0.60:                         pvc += 2.5

        # PAC scoring
        pac = 0.0
        if qrs_dur_ms < 110:                                          pac += 1.0
        if p_inverted:                                                pac += 1.5
        if not p_absent:                                              pac += 0.5
        if coupling_ratio < 0.88:                                     pac += 1.0
        if coupling_ratio < 0.92 and pause_sum_ratio < 1.90:         pac += 1.0
        if corr is not None and corr >= 0.85:                        pac += 2.5
        if ashman:                                                    pac += 1.5
        if prev_t_deformed:                                           pac += 1.0
        if corr is not None and corr >= 0.85:
            pvc -= 1.0
            pac += 0.5

        # Boundary beats never flagged as ectopic
        if i == 0 or i == n - 1:
            label = "Normal"
        elif pvc >= 3.0 and pvc > pac and coupling_ratio < 0.92:
            label = "PVC"
        elif pac >= 3.0 and coupling_ratio < 0.92:
            label = "PAC"
        else:
            label = "Normal"

        results.append({
            "beat_idx":      i,
            "peak_sample":   peak_sample,
            "label":         label,
            "pvc_score":     round(pvc, 2),
            "pac_score":     round(pac, 2),
            "corr":          round(corr, 3) if corr is not None else None,
            "coupling_ratio": round(coupling_ratio, 3),
            "qrs_dur_ms":    round(qrs_dur_ms, 1),
        })

    return results


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_template(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    normal_indices: set[int],
    fs: int,
    half_win_ms: int = 200,
) -> np.ndarray | None:
    """
    Build an averaged sinus-beat template from beats in normal_indices.
    Returns a 1-D array of length (2 * half_win_samples + 1), or None if
    fewer than 3 normal beats exist.
    """
    half = int(half_win_ms / 1000.0 * fs)
    beats = []
    for idx in normal_indices:
        if idx >= len(r_peaks):
            continue
        center = int(r_peaks[idx])
        lo, hi = center - half, center + half + 1
        if lo < 0 or hi > len(signal):
            continue
        segment = signal[lo:hi].astype(np.float64)
        beats.append(segment)

    if len(beats) < 3:
        return None

    lengths = [len(b) for b in beats]
    min_len = min(lengths)
    beats   = [b[:min_len] for b in beats]
    return np.mean(beats, axis=0)


def _template_correlation(
    signal: np.ndarray,
    r_peak: int,
    template: np.ndarray | None,
    fs: int,
    half_win_ms: int = 200,
) -> float | None:
    """Pearson correlation between a beat window and the sinus template."""
    if template is None:
        return None
    half   = int(half_win_ms / 1000.0 * fs)
    lo, hi = r_peak - half, r_peak + half + 1
    if lo < 0 or hi > len(signal):
        return None
    beat = signal[lo:hi].astype(np.float64)
    n    = min(len(beat), len(template))
    if n < 10:
        return None
    a, b = beat[:n], template[:n]
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])
