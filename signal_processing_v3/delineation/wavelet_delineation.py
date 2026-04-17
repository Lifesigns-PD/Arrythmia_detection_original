"""
wavelet_delineation.py — CWT-Based P/Q/R/S/T Boundary Detection
================================================================
Detects all 5 waveform components of each beat:
  P  : P-onset, P-peak, P-offset
  Q  : Q-wave (negative deflection before R)
  R  : R-peak (provided externally)
  S  : S-wave (negative deflection after R)
  T  : T-onset, T-peak, T-offset

Also detects:
  - QRS polarity (positive / negative / biphasic) — handles inverted leads
  - Inverted T-wave flag per beat
  - Delta wave (slurred upstroke, pre-excitation / WPW)
  - P-wave morphology: normal / biphasic / inverted / absent

Algorithm per beat:
  1. Detect lead polarity from R-peak amplitude
  2. Apply Mexican Hat CWT at QRS/P/T scales
  3. Zero-crossing and local extrema detection
  4. Physiological validation
  5. Heuristic fallbacks
"""

import numpy as np
from typing import Optional, Dict, List


# Physiological bounds (ms)
_BOUNDS = {
    "p_onset_before_r":   (60,  280),
    "p_duration":         (40,  160),
    "qrs_onset_before_r": (20,  80),
    "qrs_offset_after_r": (20,  120),
    "t_onset_after_r":    (60,  300),
    "t_offset_after_r":   (150, 500),
    "q_wave_before_r":    (5,   60),    # Q begins 5–60 ms before R
    "s_wave_after_r":     (5,   80),    # S ends 5–80 ms after R
}


def delineate_beats_wavelet(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: int = 125,
) -> List[Dict]:
    """
    Wavelet-based delineation of all beats.

    Returns
    -------
    list of dicts, one per beat, with keys:
      p_onset, p_peak, p_offset, p_morphology,
      q_peak, q_depth,
      qrs_onset, qrs_offset, qrs_polarity,
      s_peak, s_depth,
      t_onset, t_peak, t_offset, t_inverted,
      delta_wave
    """
    if len(r_peaks) < 1:
        return []

    results = []
    for i, r in enumerate(r_peaks):
        beat = _delineate_single_beat(signal, r_peaks, i, fs)
        results.append(beat)
    return results


# ─────────────────────────────────────────────────────────────────────────────

def _mexican_hat(scale: int, n_points: int = None) -> np.ndarray:
    if n_points is None:
        n_points = scale * 6 + 1
    t   = np.linspace(-3, 3, n_points)
    mh  = (1 - t ** 2) * np.exp(-0.5 * t ** 2)
    mh /= (np.sum(mh ** 2) ** 0.5 + 1e-9)
    return mh


def _cwt_at_scale(signal: np.ndarray, scale_samples: int) -> np.ndarray:
    kernel = _mexican_hat(scale_samples)
    return np.convolve(signal, kernel, mode="same")


def _find_zero_crossings(arr: np.ndarray, direction: str = "pos_to_neg") -> np.ndarray:
    signs = np.sign(arr)
    if direction == "pos_to_neg":
        return np.where((signs[:-1] > 0) & (signs[1:] <= 0))[0]
    else:
        return np.where((signs[:-1] <= 0) & (signs[1:] > 0))[0]


def _delineate_single_beat(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    beat_idx: int,
    fs: int,
) -> Dict:
    r = int(r_peaks[beat_idx])
    result = {k: None for k in [
        "p_onset", "p_peak", "p_offset", "p_morphology",
        "q_peak", "q_depth",
        "qrs_onset", "qrs_offset", "qrs_polarity",
        "s_peak", "s_depth",
        "t_onset", "t_peak", "t_offset", "t_inverted",
        "delta_wave",
    ]}

    r_amp = float(signal[r]) if 0 <= r < len(signal) else 0.0

    # ── Detect QRS polarity ──────────────────────────────────────────────────
    # Negative R = inverted lead (e.g. aVR) or LBBB/RV pattern
    if r_amp < -0.1:
        result["qrs_polarity"] = "negative"
        # For inverted QRS, work on flipped signal for delineation
        work_sig = -signal
        work_r   = r   # R is now a positive peak on the flipped signal
    elif r_amp > 0.1:
        result["qrs_polarity"] = "positive"
        work_sig = signal
        work_r   = r
    else:
        result["qrs_polarity"] = "isoelectric"
        work_sig = signal
        work_r   = r

    # ── CWT responses ────────────────────────────────────────────────────────
    # QRS scale: use 40ms (5 samples at 125 Hz) so that MH zero-crossings land
    # at ±scale×√2 ≈ ±7 samples = ±56 ms from the R-peak — well within the
    # physiological QRS onset/offset window of 20–80 ms.
    # Using 20 ms (scale=2) placed crossings at ±22 ms, which is inside the
    # QRS complex itself and caused both onset and offset to fall through to
    # hardcoded fallbacks on every beat (producing constant 88 ms QRS).
    qrs_scale = max(int(0.040 * fs), 5)
    p_scale   = max(int(0.040 * fs), 2)
    t_scale   = max(int(0.060 * fs), 3)

    cwt_qrs = _cwt_at_scale(work_sig, qrs_scale)
    cwt_p   = _cwt_at_scale(work_sig, p_scale)
    cwt_t   = _cwt_at_scale(work_sig, t_scale)

    # ── QRS onset ────────────────────────────────────────────────────────────
    # Use neg_to_pos: the CWT transitions from near-zero/negative (isoelectric
    # or PR segment) to positive (QRS upstroke).  The previous pos_to_neg
    # direction only worked when a P-wave was present (positive CWT) and failed
    # for PVC / BBB beats where no P-wave creates prior positive CWT activity.
    lo_qrs = max(0, work_r - int(0.120 * fs))
    zc_before = _find_zero_crossings(cwt_qrs[lo_qrs:work_r], "neg_to_pos")
    if len(zc_before) > 0:
        qrs_onset = lo_qrs + zc_before[-1]
        result["qrs_onset"] = _validate_sample(
            qrs_onset,
            work_r - int(_BOUNDS["qrs_onset_before_r"][1] * fs / 1000),
            work_r - int(_BOUNDS["qrs_onset_before_r"][0] * fs / 1000),
            len(signal),
        )
    if result["qrs_onset"] is None:
        result["qrs_onset"] = max(0, work_r - int(0.040 * fs))

    # ── QRS offset ───────────────────────────────────────────────────────────
    # Extend search window to 200 ms to catch wide QRS (PVC, BBB up to 180 ms)
    hi_off = min(len(signal), work_r + int(0.200 * fs))

    # Primary: slope-flatness J-point detection.
    # Works for slurred wide QRS (PVC/BBB without a clear S-wave trough) where
    # the CWT neg_to_pos crossing may not occur within the search window.
    # Algorithm: find the first run of 2+ consecutive samples where |slope| < 0.05
    # AND the signal has already descended below 5% of R-amplitude — that plateau
    # marks the QRS-ST junction (J-point).
    slope_offset = _detect_qrs_offset_slope(work_sig, work_r, hi_off,
                                             abs(float(work_sig[work_r])), fs)
    if slope_offset is not None:
        result["qrs_offset"] = _validate_sample(
            slope_offset,
            work_r + int(_BOUNDS["qrs_offset_after_r"][0] * fs / 1000),
            work_r + int(_BOUNDS["qrs_offset_after_r"][1] * fs / 1000),
            len(signal),
        )

    # Fallback: CWT neg_to_pos crossing (works well when clear S-wave is present)
    if result["qrs_offset"] is None:
        zc_after = _find_zero_crossings(cwt_qrs[work_r:hi_off], "neg_to_pos")
        if len(zc_after) > 0:
            qrs_offset = work_r + zc_after[0]
            result["qrs_offset"] = _validate_sample(
                qrs_offset,
                work_r + int(_BOUNDS["qrs_offset_after_r"][0] * fs / 1000),
                work_r + int(_BOUNDS["qrs_offset_after_r"][1] * fs / 1000),
                len(signal),
            )

    # Last-resort heuristic
    if result["qrs_offset"] is None:
        result["qrs_offset"] = min(len(signal) - 1, work_r + int(0.050 * fs))

    qrs_on  = result["qrs_onset"]
    qrs_off = result["qrs_offset"]

    # ── Q wave (negative deflection just before R) ───────────────────────────
    q_lo = qrs_on if qrs_on is not None else max(0, work_r - int(0.060 * fs))
    q_hi = work_r
    if q_hi > q_lo + 2:
        q_region = work_sig[q_lo:q_hi]
        q_local  = int(np.argmin(q_region))
        q_abs    = q_lo + q_local
        q_depth  = float(work_sig[q_abs])
        # Q wave is significant if negative (depth < 0 on work_sig = < 0 in original direction)
        result["q_peak"]  = q_abs
        result["q_depth"] = q_depth * (1 if result["qrs_polarity"] != "negative" else -1)

    # ── S wave (negative deflection just after R) ─────────────────────────────
    s_lo = work_r
    s_hi = qrs_off if qrs_off is not None else min(len(signal), work_r + int(0.080 * fs))
    if s_hi > s_lo + 2:
        s_region = work_sig[s_lo:s_hi]
        s_local  = int(np.argmin(s_region))
        s_abs    = s_lo + s_local
        s_depth  = float(work_sig[s_abs])
        result["s_peak"]  = s_abs
        result["s_depth"] = s_depth * (1 if result["qrs_polarity"] != "negative" else -1)

    # ── Delta wave (slurred QRS upstroke — WPW marker) ───────────────────────
    # Delta = short PR + initial slow upstroke before main R deflection
    if qrs_on is not None:
        upstroke_len = work_r - qrs_on
        if upstroke_len > 2:
            upstroke = work_sig[qrs_on:work_r]
            # Measure slope in first 1/3 vs last 1/3
            third = max(1, upstroke_len // 3)
            slope_early = float(np.mean(np.diff(upstroke[:third])))
            slope_late  = float(np.mean(np.diff(upstroke[-third:])))
            # Delta wave: early slope is >30% of peak slope (slurred, not sharp onset)
            if slope_late > 0 and slope_early / (slope_late + 1e-9) > 0.3:
                result["delta_wave"] = True
            else:
                result["delta_wave"] = False
        else:
            result["delta_wave"] = False

    # ── T-wave ───────────────────────────────────────────────────────────────
    if qrs_off is not None:
        lo_t = min(len(signal) - 1, qrs_off + int(0.060 * fs))
        hi_t = min(len(signal),     work_r   + int(0.500 * fs))
        if hi_t > lo_t + 2:
            t_region   = work_sig[lo_t:hi_t]
            # Use abs for peak position (works for both + and − T)
            t_pk_local = int(np.argmax(np.abs(t_region)))
            t_peak     = lo_t + t_pk_local
            result["t_peak"]    = t_peak
            t_amp = float(work_sig[t_peak])
            # T-wave inversion: T-peak amplitude negative relative to QRS polarity
            # (on work_sig which is already polarity-corrected, T should be positive in normal)
            result["t_inverted"] = bool(t_amp < -0.05)

            # T-onset
            cwt_t_region = cwt_t[lo_t:t_peak] if t_peak > lo_t else np.array([])
            if len(cwt_t_region) > 0:
                t_onset_local = int(np.argmin(cwt_t_region))
                t_onset = lo_t + t_onset_local
                result["t_onset"] = _validate_sample(
                    t_onset,
                    work_r + int(_BOUNDS["t_onset_after_r"][0] * fs / 1000),
                    work_r + int(_BOUNDS["t_onset_after_r"][1] * fs / 1000),
                    len(signal),
                )
            if result["t_onset"] is None:
                result["t_onset"] = min(len(signal) - 1, qrs_off + int(0.060 * fs))

            # T-offset
            hi_toff = min(len(signal), t_peak + int(0.250 * fs))
            if hi_toff > t_peak + 2:
                zc_t = _find_zero_crossings(cwt_t[t_peak:hi_toff], "pos_to_neg")
                if len(zc_t) > 0:
                    t_off = t_peak + zc_t[0]
                    result["t_offset"] = _validate_sample(
                        t_off,
                        work_r + int(_BOUNDS["t_offset_after_r"][0] * fs / 1000),
                        work_r + int(_BOUNDS["t_offset_after_r"][1] * fs / 1000),
                        len(signal),
                    )
            if result["t_offset"] is None and result["t_peak"] is not None:
                result["t_offset"] = min(len(signal) - 1, result["t_peak"] + int(0.100 * fs))

    # ── P-wave ───────────────────────────────────────────────────────────────
    if qrs_on is not None:
        lo_p = max(0, qrs_on - int(0.280 * fs))
        hi_p = max(0, qrs_on - int(0.040 * fs))
        if hi_p > lo_p + 2:
            p_region = work_sig[lo_p:hi_p]

            # ── Fiducial confidence: compare P-window energy to TP baseline ──
            # The TP segment (isoelectric line 60 ms before the P-window) is used
            # as baseline — NOT signal[0], which may land on an R-peak or T-wave
            # apex in streaming/chunk architectures and produce a huge false energy.
            # AF f-waves (0.1–0.3 mV) can exceed a simple amplitude threshold of
            # 0.05 mV. An energy ratio against the local TP baseline catches this:
            # if the P-window is not clearly louder than the surrounding isoelectric
            # line, there is no detectable P-wave → output None for all P fiducials.
            tp_end        = lo_p
            tp_start      = max(0, tp_end - int(0.06 * fs))
            baseline_win  = work_sig[tp_start:tp_end]
            # Demean before energy: measure AC variance, not DC+AC power.
            bw_dm = baseline_win - np.mean(baseline_win) if len(baseline_win) >= 3 else baseline_win
            pr_dm = p_region    - np.mean(p_region)
            baseline_energy = (np.sum(bw_dm ** 2) / max(len(bw_dm), 1))
            p_region_energy = (np.sum(pr_dm ** 2) / max(len(pr_dm), 1))

            # Detect the candidate P-peak position first (needed for amplitude check)
            p_pk_local = int(np.argmax(np.abs(p_region)))
            p_peak     = lo_p + p_pk_local
            p_pk_amp   = abs(float(work_sig[p_peak]))

            # Dual gate:
            # 1. Energy ratio ≥ 2.5× baseline — real P-waves sit clearly above
            #    the isoelectric line; AF f-waves have similar energy in both windows
            #    (ratio ≈ 1.0–2.0) so they fail here.
            # 2. Absolute peak ≥ 0.04 mV — rejects pure noise peaks on flat segments.
            #    Set low so small-amplitude PTBXL P-waves (~0.05 mV) still pass.
            p_is_real = (
                len(baseline_win) >= 3
                and p_region_energy >= 2.5 * baseline_energy
                and p_pk_amp >= 0.04
            )

            if not p_is_real:
                # P-wave energy indistinguishable from baseline → absent (AF, noise)
                result["p_onset"]     = None
                result["p_peak"]      = None
                result["p_offset"]    = None
                result["p_morphology"] = "absent"
            else:
                result["p_peak"] = p_peak

                p_amp = float(work_sig[p_peak])

                # Detect P-wave morphology
                result["p_morphology"] = _classify_p_morphology(p_region, p_amp)

                # P-onset
                lo_p_on = max(0, p_peak - int(0.100 * fs))
                cwt_before_p = cwt_p[lo_p_on:p_peak]
                if len(cwt_before_p) > 1:
                    zc_p = _find_zero_crossings(cwt_before_p[::-1], "pos_to_neg")
                    if len(zc_p) > 0:
                        p_on = p_peak - zc_p[0]
                        result["p_onset"] = _validate_sample(p_on, lo_p, p_peak, len(signal))
                if result["p_onset"] is None:
                    result["p_onset"] = max(0, p_peak - int(0.040 * fs))

                # P-offset
                hi_p_off = min(qrs_on, p_peak + int(0.100 * fs))
                if hi_p_off > p_peak:
                    cwt_after_p = cwt_p[p_peak:hi_p_off]
                    zc_p2 = _find_zero_crossings(cwt_after_p, "pos_to_neg")
                    if len(zc_p2) > 0:
                        p_off = p_peak + zc_p2[0]
                        result["p_offset"] = _validate_sample(p_off, p_peak, qrs_on, len(signal))
                if result["p_offset"] is None:
                    result["p_offset"] = min(len(signal) - 1, p_peak + int(0.040 * fs))

    return result


# ─────────────────────────────────────────────────────────────────────────────

def _classify_p_morphology(p_region: np.ndarray, p_amp: float) -> str:
    """
    Classify P-wave morphology for arrhythmia context:
      normal   — single upright deflection
      inverted — predominantly negative
      biphasic — positive then negative (or vice versa)
      absent   — amplitude too small
    """
    if len(p_region) < 4:
        return "absent"
    if abs(p_amp) < 0.05:
        return "absent"
    if p_amp < -0.05:
        return "inverted"
    # Biphasic: both positive and negative portions are substantial
    pos_energy = float(np.sum(p_region[p_region > 0.03]))
    neg_energy = float(np.abs(np.sum(p_region[p_region < -0.03])))
    if neg_energy > 0.3 * pos_energy and pos_energy > 0:
        return "biphasic"
    return "normal"


def _detect_qrs_offset_slope(
    signal: np.ndarray,
    r: int,
    hi_off: int,
    r_amp: float,
    fs: int = 125,
) -> Optional[int]:
    """
    Detect QRS offset (J-point) by finding where the signal has plateaued
    after the R-peak.

    Criterion: 3 consecutive samples with |slope| < flat_thresh AND the
    signal has already descended below level_thresh (i.e. the main QRS
    deflection is over).  Requires waiting at least 40 ms after R.

    Both thresholds are amplitude-relative so the detector works equally
    well for low-amplitude PTBXL signals and high-amplitude MITDB signals:
      flat_thresh  = max(0.04, r_amp * 0.08)   — 8% of R per sample
      level_thresh = max(0.10, r_amp * 0.15)   — 15% of R amplitude

    The 3-sample window (vs 2) prevents early triggers on brief noise-
    induced flat spots inside a still-descending QRS.

    Returns the sample index of the J-point, or None if not found.
    """
    # flat_thresh: fixed mV/sample — identifies the ST-segment plateau regardless
    #   of signal amplitude.  QRS slopes during descent are always >> 0.025 mV/sample;
    #   the ST-segment isoelectric floor is always < 0.025 mV/sample at 125 Hz.
    # level_thresh: scales with R-peak amplitude so it works equally for large MITDB
    #   signals (~2 mV) and weak PTBXL signals (~0.3–0.8 mV).
    flat_thresh  = 0.025                      # mV per sample — ST flatness criterion
    level_thresh = r_amp * 0.15              # signal must descend below 15% of R-peak (purely relative — works for both low-amplitude PTBXL ~0.3mV and high-amplitude MITDB ~2mV)

    min_wait = max(3, int(0.040 * fs))  # never trigger within first 40 ms of R

    # Require 4 consecutive flat samples to avoid triggering on brief noise-induced
    # flat spots still inside a descending wide QRS.
    for i in range(r + min_wait, hi_off - 4):
        s0 = abs(float(signal[i + 1]) - float(signal[i]))
        s1 = abs(float(signal[i + 2]) - float(signal[i + 1]))
        s2 = abs(float(signal[i + 3]) - float(signal[i + 2]))
        s3 = abs(float(signal[i + 4]) - float(signal[i + 3]))
        if (s0 < flat_thresh and s1 < flat_thresh
                and s2 < flat_thresh and s3 < flat_thresh
                and float(signal[i]) < level_thresh):
            return i
    return None


def _validate_sample(val: int, lo: int, hi: int, sig_len: int) -> Optional[int]:
    if val is None:
        return None
    if lo <= val <= hi and 0 <= val < sig_len:
        return int(val)
    return None
