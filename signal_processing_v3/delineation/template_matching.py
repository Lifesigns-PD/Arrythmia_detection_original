"""
template_matching.py — Patient-Specific Template Delineation
=============================================================
Builds a median beat template from the first N beats in a recording,
then uses cross-correlation to refine fiducial point positions for
each subsequent beat.

This dramatically improves consistency when NeuroKit2 / wavelets
give noisy results on individual beats — the template acts as a
"prior" for where P/QRS/T should be.
"""

import numpy as np
from typing import Dict, List, Optional


TEMPLATE_N_BEATS   = 8      # Use first 8 beats to build template
BEAT_HALF_WIN_MS   = 400    # Extract ±400 ms around each R-peak
XCORR_SEARCH_MS    = 60     # ±60 ms search window for cross-corr refinement
TACHYCARDIA_BPM    = 150    # HR threshold above which T-P overlap is expected


def refine_delineation_template(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    wavelet_results: List[Dict],
    fs: int = 125,
) -> List[Dict]:
    """
    Refine wavelet delineation using template matching.

    T-P merge guard (HR > 150 bpm):
    At high heart rates, the T-wave of beat N overlaps the P-wave of beat
    N+1.  The wavelet zero-crossing detector sees one merged hump and
    falsely assigns it as either a wide T-wave or a P-wave, corrupting
    both features.  When mean HR > 150 bpm, this function subtracts the
    median T-wave tail (built from low-HR template beats) from the
    P-search window of each subsequent beat before running delineation,
    leaving the residual P-wave visible.

    Parameters
    ----------
    signal           : 1-D ECG
    r_peaks          : R-peak indices
    wavelet_results  : output from delineate_beats_wavelet()
    fs               : sampling rate

    Returns
    -------
    List of refined delineation dicts (same format as wavelet output)
    """
    if len(r_peaks) < 2 or len(wavelet_results) == 0:
        return wavelet_results

    half_win = int(BEAT_HALF_WIN_MS * fs / 1000)
    n_templ  = min(TEMPLATE_N_BEATS, len(r_peaks))

    # ── Build template from first n_templ beats ──
    template, template_r_idx = _build_template(signal, r_peaks[:n_templ], half_win)
    if template is None:
        return wavelet_results

    # ── Get fiducial points from template using wavelet result on first beat ──
    templ_fiducials = _get_template_fiducials(wavelet_results[:n_templ], r_peaks[:n_templ], half_win)

    # ── Detect tachycardia: check if T-P overlap correction is needed ────────
    rr_samples = np.diff(r_peaks)
    rr_valid   = rr_samples[(rr_samples > int(0.2 * fs)) & (rr_samples < int(3.0 * fs))]
    mean_hr    = float(fs * 60 / np.mean(rr_valid)) if len(rr_valid) >= 2 else 0.0
    use_tp_subtraction = mean_hr > TACHYCARDIA_BPM

    # Build T-wave tail template (used only when use_tp_subtraction is True)
    t_tail_template = None
    if use_tp_subtraction:
        t_tail_template = _build_t_tail_template(
            signal, r_peaks[:n_templ], wavelet_results[:n_templ], fs
        )

    # ── Refine each beat using template ──
    refined = []
    xcorr_win = int(XCORR_SEARCH_MS * fs / 1000)

    for i, (r, wav_beat) in enumerate(zip(r_peaks, wavelet_results)):
        lo   = max(0, int(r) - half_win)
        hi   = min(len(signal), int(r) + half_win)
        beat = signal[lo:hi]

        if len(beat) < half_win:
            refined.append(wav_beat)
            continue

        # Cross-correlate beat with template to find alignment offset
        offset = _xcorr_offset(beat, template, xcorr_win)

        # ── T-P merge correction ──────────────────────────────────────────────
        # If HR > 150 bpm AND we have a T-tail template AND there is a previous
        # beat, subtract the T-wave tail of beat (i-1) from the P-search window
        # of beat i.  The subtraction is done on a scratch copy used only to
        # re-detect P-wave boundaries — the original signal is never modified.
        if use_tp_subtraction and t_tail_template is not None and i > 0:
            prev_r = int(r_peaks[i - 1])
            wav_beat = _apply_t_subtraction(
                signal, wav_beat, r, prev_r, t_tail_template, fs
            )

        # Shift fiducial points by offset.
        # Start from the wavelet beat so that non-fiducial keys (p_morphology,
        # qrs_polarity, delta_wave, q_depth, s_depth, t_inverted) are preserved.
        new_beat = dict(wav_beat)
        for key, templ_val in templ_fiducials.items():
            if templ_val is None:
                new_beat[key] = wav_beat.get(key)
                continue
            # Absolute position = lo + (template position + offset)
            abs_pos = lo + templ_val + offset
            # Blend: 60% template + 40% wavelet (if available)
            wav_val = wav_beat.get(key)
            if wav_val is not None:
                blended = int(round(0.6 * abs_pos + 0.4 * wav_val))
            else:
                blended = int(abs_pos)
            blended = int(np.clip(blended, 0, len(signal) - 1))
            new_beat[key] = blended

        refined.append(new_beat)

    return refined


# ─────────────────────────────────────────────────────────────────────────────

def _build_template(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    half_win: int,
) -> tuple:
    beats = []
    for r in r_peaks:
        lo = max(0, int(r) - half_win)
        hi = min(len(signal), int(r) + half_win)
        beat = signal[lo:hi]
        # Only include beats with expected length (avoid edge beats)
        if len(beat) >= half_win:
            # Pad to 2×half_win if needed
            if len(beat) < 2 * half_win:
                beat = np.pad(beat, (0, 2 * half_win - len(beat)))
            beats.append(beat[:2 * half_win])

    if len(beats) < 2:
        return None, half_win

    stack    = np.stack(beats, axis=0)
    template = np.median(stack, axis=0)
    r_in_template = half_win   # R-peak is at the centre
    return template, r_in_template


def _get_template_fiducials(
    wavelet_beats: List[Dict],
    r_peaks: np.ndarray,
    half_win: int,
) -> Dict:
    """
    Average fiducial offsets (relative to R-peak) from first N beats.
    Returns offsets relative to the template centre (= half_win).
    """
    offsets = {k: [] for k in [
        "p_onset", "p_peak", "p_offset",
        "qrs_onset", "qrs_offset",
        "t_onset", "t_peak", "t_offset",
    ]}

    for i, (r, beat) in enumerate(zip(r_peaks, wavelet_beats)):
        r = int(r)
        lo = max(0, r - half_win)
        for key in offsets:
            val = beat.get(key)
            if val is not None:
                # Offset relative to template centre
                offsets[key].append(val - lo)

    result = {}
    for key, vals in offsets.items():
        result[key] = int(np.median(vals)) if len(vals) >= 2 else None
    return result


def _xcorr_offset(beat: np.ndarray, template: np.ndarray, max_shift: int) -> int:
    """
    Returns the integer offset (in samples) that best aligns beat to template.
    Negative = beat is shifted left; positive = shifted right.
    """
    min_len = min(len(beat), len(template))
    b = beat[:min_len]
    t = template[:min_len]

    # Compute cross-correlation in the search window
    best_corr = -np.inf
    best_shift = 0
    for shift in range(-max_shift, max_shift + 1):
        b_shifted = np.roll(b, shift)
        corr = float(np.dot(b_shifted, t))
        if corr > best_corr:
            best_corr = corr
            best_shift = shift
    return best_shift


# ─────────────────────────────────────────────────────────────────────────────
# T-P merge helpers (tachycardia HR > 150 bpm)
# ─────────────────────────────────────────────────────────────────────────────

def _build_t_tail_template(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    wavelet_beats: List[Dict],
    fs: int,
) -> Optional[np.ndarray]:
    """
    Build a median T-wave tail template from the first N template beats.

    The T-tail is the portion of the signal from QRS offset to the end of
    the T-wave (or up to 350 ms after R-peak if T-offset is unavailable).
    This segment is what bleeds into the next beat's P-window at HR > 150 bpm.

    Returns a 1-D array of length = T_TAIL_SAMPLES, or None if insufficient data.
    """
    T_TAIL_SAMPLES = int(0.350 * fs)   # 350 ms after R-peak = covers T-wave tail
    tails = []
    for r, beat in zip(r_peaks, wavelet_beats):
        r = int(r)
        # Start from QRS offset (or 60 ms after R if unavailable)
        qrs_off = beat.get("qrs_offset")
        start   = int(qrs_off) if qrs_off is not None else r + int(0.060 * fs)
        end     = r + T_TAIL_SAMPLES
        if start >= end or end > len(signal):
            continue
        tail = signal[start:end]
        # Zero-mean each tail so subtraction doesn't shift the isoelectric line
        tail = tail - float(np.mean(tail[:max(1, int(0.020 * fs))]))
        # Pad or trim to standard length
        if len(tail) < T_TAIL_SAMPLES:
            tail = np.pad(tail, (0, T_TAIL_SAMPLES - len(tail)))
        tails.append(tail[:T_TAIL_SAMPLES])

    if len(tails) < 2:
        return None
    return np.median(np.stack(tails, axis=0), axis=0).astype(np.float64)


def _apply_t_subtraction(
    signal: np.ndarray,
    wav_beat: Dict,
    r: int,
    prev_r: int,
    t_tail_template: np.ndarray,
    fs: int,
) -> Dict:
    """
    Subtract the T-wave tail of the previous beat from this beat's P-search
    window, then re-detect P-wave onset/peak/offset on the residual.

    The original wavelet delineation for QRS / T / S / Q / delta is kept
    unchanged — only P-wave fiducials are updated from the residual.

    Returns the (potentially updated) beat dict.
    """
    try:
        T_TAIL_SAMPLES = len(t_tail_template)

        # Where does the previous beat's T-tail start?
        prev_qrs_off = int(prev_r + int(0.060 * fs))  # conservative fallback
        tail_start   = prev_qrs_off
        tail_end     = prev_r + T_TAIL_SAMPLES
        if tail_start >= tail_end or tail_end > len(signal):
            return wav_beat

        # Build a residual signal: subtract T-tail from the P-search window
        # The P-search window is [beat_lo, qrs_onset - 40ms]
        qrs_on = wav_beat.get("qrs_onset")
        if qrs_on is None:
            qrs_on = int(r) - int(0.040 * fs)
        p_search_end   = max(0, int(qrs_on) - int(0.020 * fs))
        p_search_start = max(0, p_search_end - int(0.280 * fs))
        if p_search_end <= p_search_start + 4:
            return wav_beat

        # How much of the T-tail overlaps with the P-window?
        overlap_start = max(tail_start, p_search_start)
        overlap_end   = min(tail_end,   p_search_end)
        if overlap_end <= overlap_start:
            return wav_beat

        # Build residual segment
        residual = signal[p_search_start:p_search_end].copy().astype(np.float64)

        # Align T-tail template to the overlap region
        tail_offset_in_template = overlap_start - tail_start
        overlap_len = overlap_end - overlap_start
        res_offset  = overlap_start - p_search_start
        if (tail_offset_in_template >= 0
                and tail_offset_in_template + overlap_len <= T_TAIL_SAMPLES
                and res_offset + overlap_len <= len(residual)):
            tail_seg = t_tail_template[tail_offset_in_template:
                                       tail_offset_in_template + overlap_len]
            # Scale subtraction: don't over-subtract if T-tail is larger than signal
            scale = min(1.0, float(np.std(residual[res_offset:res_offset + overlap_len]) + 1e-9)
                              / float(np.std(tail_seg) + 1e-9))
            residual[res_offset:res_offset + overlap_len] -= scale * tail_seg

        # Re-detect P-wave peak on the residual
        p_amp_residual = np.abs(residual)
        p_pk_local     = int(np.argmax(p_amp_residual))
        p_peak_abs     = p_search_start + p_pk_local

        # TP-segment baseline check on residual (same logic as wavelet_delineation)
        tp_end    = p_search_start
        tp_start  = max(0, tp_end - int(0.060 * fs))
        tp_win    = signal[tp_start:tp_end].astype(np.float64)
        baseline_energy = np.sum(tp_win ** 2) / max(len(tp_win), 1)
        p_energy        = np.sum(residual ** 2) / max(len(residual), 1)

        if len(tp_win) < 3 or p_energy < 1.5 * baseline_energy:
            # Still no clear P-wave after subtraction → mark absent
            wav_beat = dict(wav_beat)
            wav_beat["p_onset"]    = None
            wav_beat["p_peak"]     = None
            wav_beat["p_offset"]   = None
            wav_beat["p_morphology"] = "absent"
            return wav_beat

        # Accept the new P-peak; set onset/offset ±40 ms heuristic
        wav_beat = dict(wav_beat)
        wav_beat["p_peak"]   = int(np.clip(p_peak_abs, 0, len(signal) - 1))
        wav_beat["p_onset"]  = int(np.clip(p_peak_abs - int(0.040 * fs), 0, len(signal) - 1))
        wav_beat["p_offset"] = int(np.clip(p_peak_abs + int(0.040 * fs), 0, int(qrs_on)))
        wav_beat["p_morphology"] = "normal"   # conservative; wavelet will refine

    except Exception:
        pass  # any failure → return original wav_beat unchanged

    return wav_beat
