"""
lethal_detector.py — Signal-processing-based detection of lethal and non-ML rhythms.

Detects VTach, VFib, and SVT purely from the ECG signal and V3 features.
No ML model is used. Called from rhythm_orchestrator.py BEFORE the ML rhythm step.

Detection order (first match wins):
  1. Spectral lethal pre-check  — Welch PSD SPI >0.75 → VTach or VFib  (conf 0.92)
  2. Kinetic VTach              — wide QRS + AV dissociation              (conf 0.85)
  3. Polymorphic VTach          — fast + mildly irregular + no P          (conf 0.82)
  4. Coarse VFib                — chaotic RR + fast + no P                (conf 0.80)
  5. SVT                        — narrow + rapid + regular + no P         (conf 0.80)
  None → no signal-processing rhythm detected, proceed to ML.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch


def detect_signal_rhythm(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    features: dict,
    fs: int = 125,
) -> tuple[str | None, float, str]:
    """
    Returns (label, confidence, reason).
    label is one of:
      "Ventricular Tachycardia" | "Ventricular Fibrillation" | "SVT" | None
    None = nothing detected by signal processing — proceed to ML.
    """
    # ── Spectral check (most authoritative) ──────────────────────────────────
    is_lethal, spec_label, _ = _spectral_lethal_precheck(signal, r_peaks, fs)
    if is_lethal:
        label = ("Ventricular Tachycardia" if "TACHYCARDIA" in spec_label
                 else "Ventricular Fibrillation")
        return label, 0.92, f"Spectral SPI >0.75 — {spec_label}"

    # ── Feature-based checks (use V3 60-feature dict) ────────────────────────
    # Key mapping — V3 features use these exact names:
    #   qrs_wide_fraction   (BEAT_DISC_FEATURES, not "wide_qrs_fraction")
    #   p_absent_fraction   (BEAT_DISC_FEATURES, not "p_wave_present_ratio")
    #   rr_cv, qrs_duration_ms, mean_hr_bpm — exact match
    hr        = float(features.get("mean_hr_bpm") or features.get("mean_hr") or 0)
    rr_cv     = float(features.get("rr_cv") or 0)
    qrs_ms    = float(features.get("qrs_duration_ms") or 0)
    wide_frac = float(features.get("qrs_wide_fraction")   # V3 canonical name
                      or features.get("wide_qrs_fraction")  # fallback alias
                      or 0)
    # p_absent: prefer direct p_absent_fraction, else invert p_wave_present_ratio
    _p_absent_direct  = features.get("p_absent_fraction")
    _p_present_stored = features.get("p_wave_present_ratio")
    if _p_absent_direct is not None:
        p_absent = float(_p_absent_direct)
    elif _p_present_stored is not None:
        p_absent = 1.0 - float(_p_present_stored)
    else:
        p_absent = 0.0

    if _kinetic_vtach(hr, qrs_ms, wide_frac, p_absent):
        return ("Ventricular Tachycardia", 0.85,
                f"Kinetic VTach: HR={hr:.0f} QRS={qrs_ms:.0f}ms wide_frac={wide_frac:.2f} P_absent={p_absent:.2f}")

    if _polymorphic_vtach(hr, rr_cv, p_absent):
        return ("Ventricular Tachycardia", 0.82,
                f"Polymorphic VTach/Torsades: HR={hr:.0f} rr_cv={rr_cv:.2f} P_absent={p_absent:.2f}")

    if _coarse_vfib(hr, rr_cv, p_absent):
        return ("Ventricular Fibrillation", 0.80,
                f"Coarse VFib: rr_cv={rr_cv:.2f} HR={hr:.0f} P_absent={p_absent:.2f}")

    if _svt(hr, qrs_ms, rr_cv, p_absent):
        return ("SVT", 0.80,
                f"SVT: HR={hr:.0f} QRS={qrs_ms:.0f}ms rr_cv={rr_cv:.2f} P_absent={p_absent:.2f}")

    return None, 0.0, "No signal-processing rhythm detected"


# keep old name as alias so any existing references don't break
detect_lethal_rhythm = detect_signal_rhythm


# ── Internal algorithms ───────────────────────────────────────────────────────

def _spectral_lethal_precheck(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: int,
) -> tuple[bool, str, float]:
    """
    Two-stage lethal pre-check ported from BATCH_PROCESS/lifesigns_engine.py
    spectral_lethal_precheck() lines 260-343.

    Stage 1 — Organization gate (false-positive prevention):
      - Bigeminy fingerprint: even/odd RR alternation >12%, each group CV <20% → SURVIVABLE
      - Very regular + slow: rr_cv <0.20 AND rate <130 BPM AND ≥6 beats → SURVIVABLE
      - Flutter/SVT guard (V3 addition): rr_cv <0.08 → SURVIVABLE
      - All else falls through to spectral stage

    Stage 2 — Welch PSD:
      SPI = power(1.5–7 Hz) / total(0.5–40 Hz)
      SPI >0.75 + concentration >0.35 → VENTRICULAR TACHYCARDIA
      SPI >0.75 + concentration ≤0.35 → VENTRICULAR FIBRILLATION

    Returns (is_lethal, label_str, calc_rate_bpm)
    """
    # Stage 1 — Organization gate
    if r_peaks is not None and len(r_peaks) >= 4:
        rr        = np.diff(np.asarray(r_peaks, dtype=float))
        mean_rr   = float(np.mean(rr))
        median_rr = float(np.median(rr))
        rr_cv     = float(np.std(rr)) / (mean_rr + 1e-8)
        rate_bpm  = 60.0 * fs / mean_rr if mean_rr > 0 else 0.0

        # Flutter / SVT guard — very regular rhythms are never VTach/VFib
        if rr_cv < 0.08:
            return False, "SURVIVABLE", rate_bpm

        # Bigeminy fingerprint
        bigeminy = False
        if len(rr) >= 4:
            even_rr   = rr[0::2]
            odd_rr    = rr[1::2]
            even_mean = float(np.mean(even_rr))
            odd_mean  = float(np.mean(odd_rr))
            alt_ratio = abs(even_mean - odd_mean) / (median_rr + 1e-8)
            if alt_ratio > 0.12:
                cv_even  = float(np.std(even_rr)) / (even_mean + 1e-8)
                cv_odd   = float(np.std(odd_rr))  / (odd_mean  + 1e-8)
                bigeminy = (cv_even < 0.20) and (cv_odd < 0.20)

        if bigeminy:
            return False, "SURVIVABLE", rate_bpm

        # Very regular + slow (sinus tachycardia guard)
        if len(r_peaks) >= 6 and rr_cv < 0.20 and rate_bpm < 130:
            return False, "SURVIVABLE", rate_bpm

    # Stage 2 — Welch PSD spectral power ratio
    nperseg = min(int(2.0 * fs), len(signal))
    freqs, psd = welch(signal, fs, nperseg=nperseg)

    lethal_mask = (freqs >= 1.5) & (freqs <= 7.0)
    if not np.any(lethal_mask):
        return False, "SURVIVABLE", 0.0

    band_lethal = float(np.sum(psd[lethal_mask]))
    band_total  = float(np.sum(psd[(freqs >= 0.5) & (freqs <= 40.0)])) + 1e-8
    spi         = band_lethal / band_total

    if spi > 0.75:
        dom_idx       = int(np.argmax(psd[lethal_mask]))
        dom_freq      = float(freqs[lethal_mask][dom_idx])
        calc_rate     = dom_freq * 60.0
        concentration = float(psd[lethal_mask][dom_idx]) / (band_lethal + 1e-8)
        if concentration > 0.35:
            return True, "VENTRICULAR TACHYCARDIA", calc_rate
        else:
            return True, "VENTRICULAR FIBRILLATION", calc_rate

    return False, "SURVIVABLE", 0.0


def _kinetic_vtach(hr: float, qrs_ms: float, wide_frac: float, p_absent: float) -> bool:
    """
    Wide complex + AV dissociation criteria.
    Ported from BATCH_PROCESS calculate_metrics() lines 951-972.
    At HR >150 the QRS width threshold is relaxed (measurement noise at high rates).
    """
    if hr <= 100:
        return False
    av_dissociation = p_absent > 0.70
    if not av_dissociation:
        return False
    if hr > 150:
        return qrs_ms > 100 and wide_frac > 0.60
    return qrs_ms > 120 and wide_frac > 0.75


def _polymorphic_vtach(hr: float, rr_cv: float, p_absent: float) -> bool:
    """
    Polymorphic VTach / Torsades de Pointes.
    Ported from BATCH_PROCESS calculate_metrics() lines 905-915.
    """
    return hr > 150 and 0.08 < rr_cv < 0.55 and p_absent > 0.60


def _coarse_vfib(hr: float, rr_cv: float, p_absent: float) -> bool:
    """
    Coarse VFib: completely chaotic RR at fast rate.
    Ported from BATCH_PROCESS calculate_metrics() lines 896-903.
    """
    return rr_cv > 0.50 and hr > 80 and p_absent > 0.60


def _svt(hr: float, qrs_ms: float, rr_cv: float, p_absent: float) -> bool:
    """
    Supraventricular Tachycardia: narrow + rapid + very regular + no visible P-waves.
    Ported from BATCH_PROCESS calculate_metrics() svt_flag block lines 974-982.
    SVT is not lethal but caught here because the rhythm ML model has only 27 training
    samples (0 corrected) — ML cannot learn it reliably.
    """
    return (hr > 100
            and qrs_ms < 120
            and rr_cv < 0.10
            and p_absent > 0.60)
