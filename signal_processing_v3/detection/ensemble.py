"""
ensemble.py — Multi-Algorithm R-Peak Detection Ensemble
=========================================================
Runs 3 independent detectors and combines results by voting.

Logic:
  1. Run Pan-Tompkins, Hilbert, Wavelet detectors
  2. For each detected peak, check if ≥ 2 detectors agree within ±50 ms
  3. Agreed peaks are "confirmed"; use the median position across agreeing detectors
  4. Validate final peaks: remove RR interval outliers (ectopic or false)
  5. Fall back to best single detector if ensemble is too sparse
"""

import sys
import numpy as np
from pathlib import Path
from typing import List

# Make sure parent signal_processing_v3 is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from signal_processing_v3.detection.hilbert_detector import detect_r_peaks_hilbert
from signal_processing_v3.detection.wavelet_detector import detect_r_peaks_wavelet

# Re-use V2 Pan-Tompkins (already solid)
_V2_SP = Path(__file__).resolve().parents[2] / "signal_processing"
sys.path.insert(0, str(_V2_SP.parent))
try:
    from signal_processing.pan_tompkins import detect_r_peaks as detect_r_peaks_pt
except ImportError:
    from scipy.signal import find_peaks as _fp
    def detect_r_peaks_pt(signal, fs):
        h = 0.5 * np.max(np.abs(signal))
        peaks, _ = _fp(signal, height=h, distance=int(0.25*fs))
        return peaks


AGREE_WINDOW_SAMPLES = None   # Set dynamically: 50 ms


def detect_r_peaks_ensemble(signal: np.ndarray, fs: int = 125) -> np.ndarray:
    """
    Ensemble R-peak detection with automatic polarity handling.

    Handles inverted QRS complexes (e.g. aVR lead, LBBB patterns):
    if the dominant deflection is negative, flips the signal before
    running all detectors, then maps peaks back to original indices.

    Returns
    -------
    np.ndarray  — validated R-peak indices (int), refined to true signal max/min
    """
    agree_win = max(int(0.050 * fs), 3)   # 50 ms default

    # ── Polarity check: if dominant peaks are negative, work on flipped signal ──
    polarity, work_sig = _detect_polarity(signal, fs)

    # ── Run all 3 detectors on work_sig ──
    peaks_pt  = _safe_detect(detect_r_peaks_pt,      work_sig, fs)
    peaks_hil = _safe_detect(detect_r_peaks_hilbert, work_sig, fs)
    peaks_wav = _safe_detect(detect_r_peaks_wavelet, work_sig, fs)

    all_peaks = [peaks_pt, peaks_hil, peaks_wav]
    valid     = [p for p in all_peaks if len(p) >= 2]

    if len(valid) == 0:
        return np.array([], dtype=int)

    # Exclude catastrophically-failed detectors from voting.
    # If a detector found < 25% of the median count it almost certainly
    # failed (e.g. Pan-Tompkins at very fast rates or low-amplitude signals).
    # Keeping it in the vote would require 2-of-3 agreement that is impossible
    # to satisfy for any beat the failed detector missed.
    counts   = [len(p) for p in valid]
    med_cnt  = float(np.median(counts))
    reliable = [p for p in valid if len(p) >= max(2, 0.25 * med_cnt)]
    if len(reliable) >= 2:
        # Use only reliable detectors; enough to do majority voting
        vote_lists = reliable
    elif len(reliable) == 1:
        # Only one detector is reliable — trust it directly, skip voting
        agreed = _validate_rr(reliable[0], fs)
        refine_win = max(int(0.020 * fs), 2)
        refined = []
        for p in agreed:
            lo = max(0, int(p) - refine_win)
            hi = min(len(signal), int(p) + refine_win + 1)
            seg = signal[lo:hi]
            local = int(np.argmin(seg)) if polarity == "negative" else int(np.argmax(seg))
            refined.append(lo + local)
        return np.array(refined, dtype=int) if refined else np.array([], dtype=int)
    else:
        vote_lists = valid

    # ── Adaptive voting window for irregular rhythms / wide PVC QRS ──────────
    # PVCs produce wide QRS complexes (120–180 ms).  Pan-Tompkins fires on the
    # steep upstroke while Hilbert / Wavelet fire on the amplitude peak — these
    # can be separated by 30–70 ms on a single beat.  When RR is irregular
    # (CV > 0.15, typical of AF or frequent PVCs), widen the agreement window
    # to 80 ms so a valid wide-QRS beat is not rejected as a "non-agreement".
    if len(vote_lists) >= 1:
        best_det = max(vote_lists, key=len)
        if len(best_det) >= 3:
            rr = np.diff(np.sort(best_det))
            rr_valid = rr[(rr > int(0.2 * fs)) & (rr < int(3.0 * fs))]
            if len(rr_valid) >= 2:
                rr_cv = float(np.std(rr_valid) / (np.mean(rr_valid) + 1e-9))
                if rr_cv > 0.15:
                    # Irregular rhythm (AF, PVC-laden) — widen to 80 ms
                    agree_win = max(int(0.080 * fs), agree_win)

    if len(vote_lists) == 1:
        agreed = vote_lists[0]
    else:
        agreed = _vote(vote_lists, agree_win)
        if len(agreed) < 2:
            agreed = max(vote_lists, key=len)

    # Remove any residual duplicates within 220 ms (false doubles from union)
    min_gap = max(int(0.220 * fs), 2)
    agreed  = _dedup(agreed, min_gap)

    validated = _validate_rr(agreed, fs)

    # ── Refine each peak to the true extreme in original signal ──
    # (max for positive QRS, min for inverted QRS)
    refine_win = max(int(0.020 * fs), 2)
    refined = []
    for p in validated:
        lo = max(0, int(p) - refine_win)
        hi = min(len(signal), int(p) + refine_win + 1)
        seg = signal[lo:hi]
        if polarity == "negative":
            local = int(np.argmin(seg))
        else:
            local = int(np.argmax(seg))
        refined.append(lo + local)

    int_peaks = np.array(refined, dtype=int) if refined else np.array([], dtype=int)
    return int_peaks


def refine_peaks_subsample(signal: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """
    Sub-sample R-peak refinement via 3-point parabolic interpolation.

    At 125 Hz each sample = 8 ms, giving ±4 ms of hardware quantization
    jitter per peak.  This corrupts RMSSD (which measures beat-to-beat ms
    differences) and HF/LF power.  Parabolic interpolation around the
    integer peak position reduces jitter to sub-millisecond accuracy.

    Parameters
    ----------
    signal : preprocessed ECG signal
    peaks  : integer R-peak indices (output of detect_r_peaks_ensemble)

    Returns
    -------
    np.ndarray  float64 — sub-sample refined peak positions.
                          Use these for RR-interval / HRV computation only.
                          For sample indexing (delineation, beat windows) cast
                          back to int or use the original integer peaks.
    """
    refined = peaks.astype(float)
    for i, p in enumerate(peaks):
        p = int(p)
        if 1 <= p < len(signal) - 1:
            y0, y1, y2 = float(signal[p - 1]), float(signal[p]), float(signal[p + 1])
            denom = (2.0 * y1 - y0 - y2)
            if abs(denom) > 1e-9:
                refined[i] = p + 0.5 * (y0 - y2) / denom
    return refined


# ─────────────────────────────────────────────────────────────────────────────

def _detect_polarity(signal: np.ndarray, fs: int):
    """
    Determine QRS polarity by comparing the strongest positive vs negative peaks.
    Returns ("positive"|"negative", work_signal).
    Inverted QRS (e.g. aVR, LBBB) returns flipped signal so detectors always
    see an upward deflection.
    """
    from scipy.signal import find_peaks
    min_dist = max(int(0.25 * fs), 1)
    pos_peaks, _ = find_peaks( signal, distance=min_dist, height=0)
    neg_peaks, _ = find_peaks(-signal, distance=min_dist, height=0)

    pos_max = float(np.max(signal[pos_peaks])) if len(pos_peaks) > 0 else 0.0
    neg_max = float(np.max(-signal[neg_peaks])) if len(neg_peaks) > 0 else 0.0

    if neg_max > pos_max * 1.3:          # dominant deflection is negative
        return "negative", -signal
    return "positive", signal


def _safe_detect(fn, signal, fs):
    try:
        result = fn(signal, fs)
        if result is None:
            return np.array([], dtype=int)
        return np.asarray(result, dtype=int)
    except Exception:
        return np.array([], dtype=int)


def _vote(peak_lists: List[np.ndarray], agree_win: int) -> np.ndarray:
    """
    Return peaks confirmed by ≥ 2 detectors within agree_win samples.
    Position = median of agreeing detector positions.

    Special case — exactly 2 reliable detectors:
    Take the union of both lists (deduped within agree_win) rather than
    requiring 2-of-2 agreement.  Both Hilbert and Wavelet are trusted
    detectors; their disagreements at fast rates are timing offsets on
    the same real beat, not false positives.  Requiring exact agreement
    causes systematic beat dropout at HR > 130 bpm.
    """
    if len(peak_lists) == 2:
        return _union_two(peak_lists[0], peak_lists[1], agree_win)

    # ≥ 3 detectors: standard majority-vote (≥ 2 must agree)
    reference = max(peak_lists, key=len)
    if len(reference) == 0:
        return np.array([], dtype=int)

    confirmed = []
    for ref_peak in reference:
        supporters = [ref_peak]
        for other_list in peak_lists:
            if other_list is reference:
                continue
            if len(other_list) == 0:
                continue
            diffs = np.abs(other_list - ref_peak)
            nearest_idx = int(np.argmin(diffs))
            if diffs[nearest_idx] <= agree_win:
                supporters.append(other_list[nearest_idx])

        if len(supporters) >= 2:
            confirmed.append(int(np.median(supporters)))

    if len(confirmed) == 0:
        return np.array([], dtype=int)

    confirmed = np.unique(np.array(confirmed, dtype=int))
    return confirmed


def _union_two(a: np.ndarray, b: np.ndarray, dedup_win: int) -> np.ndarray:
    """
    Union of two peak lists.  Where both detectors agree within dedup_win
    samples, use the midpoint.  Peaks unique to one detector are included as-is.
    """
    if len(a) == 0 and len(b) == 0:
        return np.array([], dtype=int)
    if len(a) == 0:
        return np.array(b, dtype=int)
    if len(b) == 0:
        return np.array(a, dtype=int)

    merged = []
    used_b = set()
    for pa in np.sort(a):
        diffs = np.abs(b - pa)
        idx   = int(np.argmin(diffs))
        if diffs[idx] <= dedup_win and idx not in used_b:
            merged.append(int(round((pa + b[idx]) / 2.0)))
            used_b.add(idx)
        else:
            merged.append(int(pa))
    for j, pb in enumerate(np.sort(b)):
        if j not in used_b:
            merged.append(int(pb))

    return np.unique(np.array(merged, dtype=int))


def _dedup(peaks: np.ndarray, min_gap: int) -> np.ndarray:
    """
    Remove duplicate peaks closer than min_gap samples.
    When two peaks are within min_gap, keep the one with the higher amplitude
    is not available here, so keep the second (later) one which is typically
    the true peak when the first is a rising-edge false trigger.
    Simple greedy: scan sorted, drop any peak that is < min_gap from the
    last kept peak.
    """
    if len(peaks) < 2:
        return peaks
    peaks  = np.sort(peaks)
    kept   = [peaks[0]]
    for p in peaks[1:]:
        if p - kept[-1] >= min_gap:
            kept.append(p)
    return np.array(kept, dtype=int)


def _validate_rr(peaks: np.ndarray, fs: int) -> np.ndarray:
    """
    Remove peaks whose RR interval is physiologically impossible
    (< 200 ms = > 300 bpm, or > 3000 ms = < 20 bpm)
    OR is an extreme outlier (> 5× median RR).

    Note: multiplier is 5× not 3× to handle bigeminy / alternating-RR rhythms
    where the "long" interval can be ~3× the short coupling interval without
    being a missed beat.
    """
    if len(peaks) < 2:
        return peaks

    peaks = np.sort(peaks)
    rr    = np.diff(peaks)
    med   = np.median(rr)

    keep  = [0]   # always keep first peak
    for i, interval in enumerate(rr):
        samples_200ms  = int(0.200 * fs)
        samples_3000ms = int(3.000 * fs)
        # Keep if physiologically plausible AND not extreme outlier
        if samples_200ms <= interval <= samples_3000ms and interval < 5.0 * med:
            keep.append(i + 1)

    return peaks[keep].astype(int)
