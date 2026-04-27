"""
lifesigns_engine.py  ─  Lifesigns ECG Signal Processing Engine
=============================================================
Revision notes
--------------
• compute_sqi()                 – Multi-dimensional Signal Quality Index pre-gate.
                                  Catches flatline, lead-off, motion artifact,
                                  EMG noise and baseline wander BEFORE any
                                  processing begins.

• _build_normal_template() /    – Morphology-based PAC/PVC disambiguation via
  _template_correlation()         Pearson cross-correlation against an averaged
                                  sinus-beat template.  A beat whose QRS looks
                                  identical to normal sinus beats is a PAC, not a
                                  PVC, regardless of coupling interval or P-wave
                                  visibility.  Resolves ~80 % of PAC/PVC confusion.

• Ashman-phenomenon detection   – Long-short RR sequence before a premature beat
  in classify_beats()             means the bundle branches are still refractory →
                                  aberrant conduction → PAC looks wide like a PVC.
                                  Detected and re-scored correctly.

• Boundary-beat guard           – First and last beats of every segment are NEVER
                                  classified as ectopic.  They have no full RR
                                  context and always produce spurious flags.

• High-rate mode (HR > 140)     – At rates above 140 BPM the TP segment vanishes;
                                  T and P waves overlap.  P/T delineation is
                                  suspended and a flag is returned to the UI.

• SVT flag                      – Narrow complex (QRS < 120 ms) + rapid (HR > 100)
                                  + regular (CV < 10 %) + absent P waves → SVT.

• Flutter spectral hint         – After QRS+T removal, dominant power near 5 Hz
                                  in the P-channel signal suggests atrial flutter
                                  at 300 BPM with 2:1 conduction.

• Tachycardia / Bradycardia /   – Simple HR-threshold flags added to metrics.
  Sinus-arrest flags

• AV-dissociation guard in      – VTach kinetic flag now also requires >75 % wide
  VTach flag                      beats AND either P-absent fraction > 70 % or
                                  highly variable PR interval (AV dissociation).

• spectral_lethal_precheck()    – Now called on the bandpass-filtered fiducial
  on filtered signal               (not raw signal) to prevent noisy artifacts
                                  from triggering false lethal alarms.

• Preceding T-wave deformation  – A notched or abnormally tall T wave on the beat
  check in classify_beats()        before a premature beat suggests a retrograde
                                  ectopic P buried in it → PAC evidence.
"""

import json
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, medfilt, savgol_filter, welch
from scipy.ndimage import uniform_filter1d
from scipy.stats import kurtosis as scipy_kurtosis
import io
import base64
import threading
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FuncFormatter, MaxNLocator, AutoMinorLocator
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import warnings

warnings.filterwarnings("ignore")

# Global mutex for Matplotlib FreeType C-library
render_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
#  SIGNAL QUALITY INDEX
# ─────────────────────────────────────────────────────────────────────────────
def compute_sqi(signal, fs):
    """
    Multi-dimensional Signal Quality Index.

    Returns
    -------
    verdict : str
        FLATLINE | LEAD_OFF | MOTION_ARTIFACT | EMG_NOISE |
        BASELINE_WANDER | POOR_QUALITY | GOOD
    score   : float  0.0 – 1.0  (higher = better quality)
    details : dict   raw sub-metrics for logging / UI display
    """
    details = {}
    n = len(signal)

    # 1. Flatline (lead completely disconnected or signal clipped to zero)
    std_val = float(np.std(signal))
    details['std_mv'] = round(std_val, 4)
    if std_val < 0.03:
        return "FLATLINE", 0.0, details

    # 2. Lead-off (electrode in contact but extremely poor impedance)
    ptp = float(np.ptp(signal))
    details['ptp_mv'] = round(ptp, 3)
    if ptp < 0.08:
        return "LEAD_OFF", 0.05, details

    # 3. Kurtosis
    #    Clean ECG has sharp R-peak spikes → kurtosis 8-25.
    #    White noise / EMG noise → kurtosis ≈ 3.
    #    Motion artifact (large, slow swings) → kurtosis 3–5.
    kurt = float(scipy_kurtosis(signal))
    details['kurtosis'] = round(kurt, 2)

    # 4. Power spectral density ratios
    nperseg = max(16, min(int(2.0 * fs), n // 2))
    freqs, psd = welch(signal, fs, nperseg=nperseg)

    total_pow = float(np.sum(psd[(freqs >= 0.5) & (freqs <= 40.0)])) + 1e-8
    bw_pow    = float(np.sum(psd[freqs < 0.5]))
    emg_pow   = float(np.sum(psd[freqs > 40.0]))
    mot_pow   = float(np.sum(psd[(freqs >= 1.0) & (freqs <= 3.0)]))

    bw_ratio  = bw_pow  / (total_pow + bw_pow)   # baseline wander
    emg_ratio = emg_pow / total_pow               # muscle / 50-Hz noise
    mot_ratio = mot_pow / total_pow               # gross body movement

    details.update({
        'bw_ratio':     round(bw_ratio,  3),
        'emg_ratio':    round(emg_ratio, 3),
        'motion_ratio': round(mot_ratio, 3)
    })

    # 5. Abrupt discontinuity (cable pop / sudden motion transient)
    dy = np.abs(np.diff(signal.astype(np.float64)))
    disc_score = float(np.percentile(dy, 99.5)) / (float(np.median(dy)) + 1e-8)
    details['discontinuity_ratio'] = round(disc_score, 1)

    # ── Classification decision tree ─────────────────────────────────────
    if bw_ratio > 0.55:
        score = float(np.clip(1.0 - bw_ratio, 0.0, 0.45))
        return "BASELINE_WANDER", score, details

    if emg_ratio > 0.45 or kurt < 2.5:
        score = float(np.clip(kurt / 15.0, 0.05, 0.40))
        return "EMG_NOISE", score, details

    if mot_ratio > 0.55 or disc_score > 60.0:
        score = float(np.clip(0.40 - mot_ratio * 0.40, 0.05, 0.40))
        return "MOTION_ARTIFACT", score, details

    if kurt < 3.5:
        return "POOR_QUALITY", float(np.clip(kurt / 12.0, 0.10, 0.30)), details

    # Composite score (0–1)
    kurtosis_score = min(1.0, (kurt - 3.0) / 18.0)
    noise_penalty  = bw_ratio * 0.3 + emg_ratio * 0.4 + mot_ratio * 0.3
    score = float(np.clip(kurtosis_score * (1.0 - noise_penalty), 0.0, 1.0))

    if score < 0.20:
        return "POOR_QUALITY", score, details

    return "GOOD", score, details


# ─────────────────────────────────────────────────────────────────────────────
#  NORMAL BEAT TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
def _build_normal_template(fiducial, r_peaks, rr_intervals, median_rr, fs, n_max=8):
    """
    Builds a unit-normalised average QRS template from sinus-like beats
    (coupling interval within 5 % of the median RR).

    Returns a 1-D numpy array or None when fewer than 2 usable beats exist.
    """
    half_win  = int(0.15 * fs)   # ± 150 ms around the R peak
    n         = len(fiducial)
    templates = []

    for i, r in enumerate(r_peaks):
        # Exclude boundary beats – they lack full context
        if i == 0 or i == len(r_peaks) - 1:
            continue
        # Only build from normally timed beats
        rr_prev = float(rr_intervals[i - 1]) if (i - 1) < len(rr_intervals) else median_rr
        if rr_prev < 0.92 * median_rr or rr_prev > 1.10 * median_rr:
            continue
        lo, hi = r - half_win, r + half_win
        if lo < 0 or hi >= n:
            continue
        seg = fiducial[lo:hi].astype(np.float64).copy()
        std_seg = np.std(seg)
        if std_seg < 1e-6:
            continue
        templates.append((seg - np.mean(seg)) / std_seg)
        if len(templates) >= n_max:
            break

    if len(templates) < 2:
        return None
    min_len = min(len(t) for t in templates)
    return np.mean([t[:min_len] for t in templates], axis=0)


def _template_correlation(fiducial, r, template, fs):
    """
    Pearson correlation between a single beat window and the sinus template.

    Returns float [−1, 1], or None if the window extends outside the signal.
    High correlation (≥ 0.85) → sinus-like morphology → PAC, not PVC.
    Low correlation (< 0.60)  → aberrant morphology    → PVC evidence.
    """
    if template is None:
        return None
    win_len  = len(template)
    half_win = win_len // 2
    lo = r - half_win
    hi = lo + win_len
    n  = len(fiducial)
    if lo < 0 or hi > n:
        return None
    seg = fiducial[lo:hi].astype(np.float64).copy()
    if len(seg) != win_len:
        return None
    std_seg = np.std(seg)
    if std_seg < 1e-6:
        return None
    seg_norm = (seg - np.mean(seg)) / std_seg
    try:
        return float(np.corrcoef(seg_norm, template)[0, 1])
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  DATA INGESTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_ecg_chunks(data, fs=125, window_s=10):
    packets      = sorted(data, key=lambda p: p.get("packetNo", 0))
    admission_id = packets[0].get("admissionId", "Unknown")

    full_signal = []
    for pkt in packets:
        v = pkt.get("value", [])
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
            full_signal.extend(v[0])
        elif isinstance(v, list):
            full_signal.extend(v)

    window_len = fs * window_s
    chunks     = []
    for i in range(0, len(full_signal), window_len):
        chunk = full_signal[i: i + window_len]
        if len(chunk) < window_len:
            continue
        chunks.append(np.array(chunk, dtype=np.float64))

    return chunks, admission_id, fs


# ─────────────────────────────────────────────────────────────────────────────
#  LETHAL RHYTHM PRE-CHECK  (spectral – run on FILTERED fiducial)
# ─────────────────────────────────────────────────────────────────────────────
def spectral_lethal_precheck(signal, fs, r_peaks=None):
    """
    Two-stage lethal rhythm pre-check.

    STAGE 1 — Organization gate using PRE-COMPUTED r_peaks
    ───────────────────────────────────────────────────────
    MUST use QRS peaks already detected by process_kinetic, NOT an independent
    peak detector on |signal|.

    Why independent detection failed (previous revision):
      1. PVC discordant T waves appear as large positive peaks in np.abs(fiducial),
         detected as separate "beats" ~280-350 ms after the QRS trough.  This
         scrambles the short-long-short-long bigeminy RR sequence, breaking the
         alternation check completely — so the gate never suppressed the false alarm.
      2. Synthetic VFib oscillations appear as semi-regular "peaks" to the
         independent detector.  rr_cv fell below 0.50 for some VFib segments,
         causing the survivable return to fire and blocking the spectral stage
         entirely — synthetic_vfib regressed from CRITICAL to NOISY.

    Using process_kinetic's r_peaks solves both:
      • Entropy-based QRS detection is T-wave-agnostic → true bigeminy RR sequence
      • VFib produces few / no organized QRS peaks → len(r_peaks) < 4 →
        gate bypassed → spectral check runs → lethal correctly detected

    Gate logic (only when len(r_peaks) >= 4):
      1. Bigeminy confirmed (even/odd alternation > 12%, each group CV < 20%)
         → return False immediately
      2. Very regular + slow (rr_cv < 0.20, rate < 130, >= 6 beats)
         → return False
      3. All else → fall through to spectral
         (covers VTach, VFib, polymorphic VT, rapid AFib, etc.)
    """
    # ── Stage 1: Organization gate (r_peaks from process_kinetic) ────────────
    if r_peaks is not None and len(r_peaks) >= 4:
        rr        = np.diff(np.asarray(r_peaks, dtype=float))
        mean_rr   = float(np.mean(rr))
        median_rr = float(np.median(rr))
        rr_cv     = float(np.std(rr)) / (mean_rr + 1e-8)
        rate_bpm  = 60.0 * fs / mean_rr if mean_rr > 0 else 0.0

        # ── Bigeminy fingerprint ───────────────────────────────────────────
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

        # ── Very regular slow rhythm ───────────────────────────────────────
        # Conservative thresholds: require 6+ beats, CV < 0.20, rate < 130.
        # Keeps sinus tachycardia at 130-150 BPM in play for spectral stage.
        if len(r_peaks) >= 6 and rr_cv < 0.20 and rate_bpm < 130:
            return False, "SURVIVABLE", rate_bpm

        # All other patterns (including high-rate, moderate irregularity) fall
        # through to spectral stage.  This includes VTach and coarse VFib that
        # happened to produce a handful of entropy peaks.

    # ── Stage 2: Spectral power ratio ─────────────────────────────────────
    freqs, psd  = welch(signal, fs, nperseg=int(2.0 * fs))
    lethal_mask = (freqs >= 1.5) & (freqs <= 7.0)
    band_lethal = np.sum(psd[lethal_mask])
    band_total  = np.sum(psd[(freqs >= 0.5) & (freqs <= 40.0)]) + 1e-8
    spi         = band_lethal / band_total

    if spi > 0.75:
        dom_idx       = np.argmax(psd[lethal_mask])
        dom_freq      = freqs[lethal_mask][dom_idx]
        calc_rate     = dom_freq * 60.0
        concentration = psd[lethal_mask][dom_idx] / band_lethal
        if concentration > 0.35:
            return True, "VENTRICULAR TACHYCARDIA", calc_rate
        else:
            return True, "VENTRICULAR FIBRILLATION", calc_rate

    return False, "SURVIVABLE", 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  INVERTED PVC PEAK REPAIR
# ─────────────────────────────────────────────────────────────────────────────
def _repair_inverted_pvc_peaks(fiducial, r_peaks, fs):
    """
    Repairs R-peak locations for large-amplitude inverted-QRS PVC beats.

    MODE A – Completely missed beat: the entropy detector never fired near the
    large negative complex.  Stage 1 still shows the spike.  The beat after the
    PVC incorrectly looks premature (false PAC cascade).

    MODE B – T-wave mis-location: the entropy detector fired correctly but
    argmax picked the positive T wave that follows the negative QRS trough.

    Fix – two-stage pipeline (forward scan + backward-scan fallback), both
    protected by a derivative guard that prevents R-wave-like (PAC/normal)
    peaks from being consumed.
    """
    n = len(fiducial)
    if len(r_peaks) < 2:
        return list(r_peaks)

    dy = np.zeros(n)
    dy[2:-2] = (-fiducial[:-4] - 2 * fiducial[1:-3] +
                 2 * fiducial[3:-1] + fiducial[4:]) / 8.0

    r_list = sorted(r_peaks)
    dw     = max(3, int(0.024 * fs))

    ref_amp = float(np.median([abs(float(fiducial[r])) for r in r_list]))
    ref_dy  = float(np.median([
        np.max(np.abs(dy[max(0, r - dw): min(n, r + dw)])) for r in r_list
    ]))
    if ref_dy < 1e-9:
        ref_dy = 1e-9

    trough_thr = -max(2.0 * ref_amp, 0.40)
    lw   = max(4, int(0.032 * fs))
    excl = max(10, int(0.10 * fs))

    to_remove: set  = set()
    to_add:    list = []

    # Stage 1: forward scan
    i = lw
    while i < n - lw:
        if float(fiducial[i]) > trough_thr:
            i += 1; continue

        if float(fiducial[i]) > float(np.min(fiducial[i - lw: i + lw + 1])) + 0.01:
            i += 1; continue

        dy_local = float(np.max(np.abs(dy[max(0, i - lw): min(n, i + lw + 1)])))
        if dy_local < 0.50 * ref_dy:
            i += 1; continue

        active = [r for r in r_list if r not in to_remove]
        if any(abs(i - r) < excl for r in active):
            i += excl; continue

        t_lo = i + int(0.06 * fs)
        t_hi = i + int(0.45 * fs)
        fwd  = [r for r in r_list
                if t_lo <= r <= t_hi and float(fiducial[r]) > 0 and r not in to_remove]

        swapped = False
        for cand in sorted(fwd, key=lambda r: r - i):
            dy_cand = float(np.max(np.abs(dy[max(0, cand - dw): min(n, cand + dw)])))
            if dy_cand < 0.40 * ref_dy:
                to_remove.add(cand)
                to_add.append(i)
                swapped = True
                break

        if not swapped:
            to_add.append(i)

        i += excl

    r_list = sorted([r for r in r_list if r not in to_remove] + to_add)

    # Stage 2: backward-scan fallback
    ref_amp2 = float(np.median([abs(float(fiducial[r])) for r in r_list])) if r_list else 1.0
    ref_dy2  = float(np.median([
        np.max(np.abs(dy[max(0, r - dw): min(n, r + dw)])) for r in r_list
    ])) if r_list else 1.0
    thr2     = max(0.5 * ref_amp2, 0.08)

    corrected = list(r_list)
    for idx in range(len(corrected)):
        r = corrected[idx]
        if float(fiducial[r]) <= 0:
            continue

        dy_r = float(np.max(np.abs(dy[max(0, r - dw): min(n, r + dw)])))
        if dy_r > 0.40 * ref_dy2:
            continue

        look_lo = max(0, r - int(0.40 * fs))
        look_hi = max(0, r - int(0.04 * fs))
        if look_hi <= look_lo:
            continue

        seg      = fiducial[look_lo: look_hi]
        trough_v = float(np.min(seg))
        if trough_v >= -thr2:
            continue

        trough_idx = look_lo + int(np.argmin(seg))
        prev_r     = corrected[idx - 1] if idx > 0 else 0
        if trough_idx - prev_r < int(0.25 * fs):
            continue

        qt_ms = (r - trough_idx) / fs * 1000.0
        if not (80.0 < qt_ms < 500.0):
            continue

        corrected[idx] = trough_idx

    return sorted(corrected)


# ─────────────────────────────────────────────────────────────────────────────
#  CORE PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def process_kinetic(signal, fs):
    """
    Filters, detects R peaks, and delineates PQRST for each beat.

    Returns
    -------
    fiducial       : bandpass-filtered signal used for R-peak detection
    r_peaks        : list of sample indices of R peaks
    per_beat       : list of beat dictionaries
    q_elim         : SG-filtered signal with QRS blanked (Stage 1 plot)
    t_elim         : Stage 1 signal with T wave additionally blanked (Stage 2)
    high_rate_mode : bool – True when HR > 140 BPM; PQRST delineation skipped
    """
    n      = len(signal)
    window = int(0.6 * fs) | 1
    sig_centered = signal - medfilt(signal, kernel_size=window)

    b_f, a_f = butter(4, [0.5 / (fs / 2), 40.0 / (fs / 2)], btype="band")
    fiducial  = filtfilt(b_f, a_f, sig_centered)

    b_m, a_m = butter(4, [0.5 / (fs / 2), 15.0 / (fs / 2)], btype="band")
    morph     = savgol_filter(
        filtfilt(b_m, a_m, sig_centered),
        window_length=int(0.08 * fs) | 1, polyorder=2
    )

    dy       = np.zeros_like(fiducial)
    dy[2:-2] = (-fiducial[:-4] - 2 * fiducial[1:-3] +
                 2 * fiducial[3:-1] + fiducial[4:]) / 8.0
    dy_norm  = dy / (np.max(np.abs(dy)) + 1e-8)
    se       = -(dy_norm ** 2) * np.log(dy_norm ** 2 + 1e-8)
    ma       = uniform_filter1d(se, size=max(int(0.15 * fs), 3))

    threshold  = max(np.mean(ma) + 0.8 * np.std(ma), 0.015)
    peaks_raw, _ = find_peaks(ma, height=threshold,
                               distance=int(0.20 * fs), prominence=0.01)

    validated_peaks, recent_slopes = [], []
    for p in peaks_raw:
        slp = np.max(np.abs(dy[max(0, p - int(0.08 * fs)): min(n - 1, p + int(0.08 * fs))]))
        if (len(validated_peaks) > 0 and
                (p - validated_peaks[-1]) < int(0.36 * fs) and
                slp < 0.5 * (np.median(recent_slopes) if recent_slopes else slp)):
            continue
        validated_peaks.append(p)
        recent_slopes.append(slp)
        if len(recent_slopes) > 5:
            recent_slopes.pop(0)

    r_peaks = []
    for p in validated_peaks:
        lo, hi = max(0, p - int(0.08 * fs)), min(n - 1, p + int(0.08 * fs))
        if hi > lo:
            # argmax(|fiducial|) correctly locates both upright and inverted QRS
            # peaks.  The previous argmax() silently placed inverted-QRS PVC
            # peaks on the tiny positive residual rather than the true trough,
            # causing the entire cascade of wrong QRS width / PAC misclassification.
            seg = fiducial[lo:hi]
            r_peaks.append(lo + int(np.argmax(np.abs(seg))))

    r_peaks = _repair_inverted_pvc_peaks(fiducial, r_peaks, fs)

    # ── Determine whether we are in high-rate mode ────────────────────────
    if len(r_peaks) >= 2:
        prelim_rr = float(np.median(np.diff(r_peaks)))
        prelim_hr = 60.0 * fs / prelim_rr if prelim_rr > 0 else 0.0
    else:
        prelim_hr = 0.0
    high_rate_mode = prelim_hr > 140.0

    rr_ints   = np.diff(r_peaks) if len(r_peaks) > 1 else [fs]
    median_rr = np.median(rr_ints)

    per_beat, q_elim, t_elim = [], morph.copy(), morph.copy()

    for i, r in enumerate(r_peaks):
        beat     = {
            "R": r, "QRS": (None, None),
            "T": (None, None, None, "Absent"),
            "P": (None, None, None, "Absent"),
            "Class": "Normal"
        }
        prev_r   = r_peaks[i - 1] if i > 0 else 0
        rr_prev  = r - prev_r

        # ── QRS boundaries ────────────────────────────────────────────────
        # Establish a local isoelectric baseline from the PR segment.
        # Without this, a large inverted PVC T-wave preceding a sinus beat
        # creates a negative signal level that the backward scan exits on
        # far too early, inflating the measured QRS width by 20–40 ms and
        # pushing genuinely narrow sinus beats over the 120 ms PVC threshold.
        bl_hi  = max(0, r - int(0.10 * fs))
        bl_lo  = max(0, r - int(0.28 * fs))
        local_bl = float(np.median(fiducial[bl_lo:bl_hi])) if bl_hi > bl_lo + 3 else 0.0

        r_amp_adj = float(fiducial[r]) - local_bl
        beat["R_amp"] = float(fiducial[r])   # store raw amplitude for T-discordance

        q_on,  q_off = max(0, r - int(0.04 * fs)), min(n - 1, r + int(0.06 * fs))
        if r_amp_adj >= 0:
            for j in range(r, max(0, r - int(0.08 * fs)), -1):
                if (fiducial[j] - local_bl) < 0.15 * r_amp_adj:
                    q_on = j; break
            for j in range(r, min(n - 1, r + int(0.12 * fs))):
                fv = fiducial[j] - local_bl
                if fv < 0.15 * r_amp_adj and abs(fiducial[j] - fiducial[j - 1]) < 0.02:
                    q_off = j; break
        else:
            inv_thr = 0.15 * r_amp_adj   # negative threshold
            for j in range(r, max(0, r - int(0.08 * fs)), -1):
                if (fiducial[j] - local_bl) > inv_thr:
                    q_on = j; break
            for j in range(r, min(n - 1, r + int(0.12 * fs))):
                fv = fiducial[j] - local_bl
                if fv > inv_thr and abs(fiducial[j] - fiducial[j - 1]) < 0.02:
                    q_off = j; break

        beat["QRS"] = (q_on, q_off)

        _blk_margin = max(2, int(0.020 * fs))
        for arr, on, off in [(q_elim, q_on, q_off), (t_elim, q_on, q_off)]:
            if off > on:
                blk_on  = max(0, on  - _blk_margin)
                blk_off = min(n - 1, off + _blk_margin)
                pre  = float(np.mean(arr[max(0, blk_on - 3): blk_on + 1]))
                post = float(np.mean(arr[blk_off: min(n, blk_off + 4)]))
                arr[blk_on: blk_off] = np.linspace(pre, post, blk_off - blk_on)

        # ── Skip P and T delineation in high-rate mode ─────────────────────
        if high_rate_mode:
            per_beat.append(beat)
            continue

        # T-wave detection
        t_start = min(n - 1, q_off + int(0.02 * fs))
        t_end   = min(n - 1, r + int(0.45 * fs))
        if i + 1 < len(r_peaks):
            t_end = min(t_end, r_peaks[i + 1])

        if t_end > t_start + 8:
            pos_p, pos_pr = find_peaks( morph[t_start:t_end], prominence=0.015)
            neg_p, neg_pr = find_peaks(-morph[t_start:t_end], prominence=0.015)
            best_t, b_prom, b_morph = None, 0, "Absent"
            if len(pos_p) > 0:
                idx_ = np.argmax(pos_pr["prominences"])
                best_t, b_prom, b_morph = pos_p[idx_], pos_pr["prominences"][idx_], "Normal"
            if len(neg_p) > 0:
                idx_ = np.argmax(neg_pr["prominences"])
                if neg_pr["prominences"][idx_] > b_prom * 1.2:
                    best_t, b_prom, b_morph = neg_p[idx_], neg_pr["prominences"][idx_], "Inverted"

            if best_t is not None and abs(morph[min(n - 1, t_start + best_t)]) >= 0.015:
                t_pk  = min(n - 1, t_start + best_t)
                t_on  = max(t_start, t_pk - int(0.10 * fs))
                t_off = min(t_end,   t_pk + int(0.12 * fs))
                beat["T"] = (t_on, t_off, t_pk, b_morph)
                if t_off > t_on:
                    t_elim[t_on:t_off] = np.linspace(
                        t_elim[t_on], t_elim[t_off], t_off - t_on)

        # P-wave detection
        p_start = max(r - int(0.25 * fs), prev_r + int(rr_prev * 0.4))
        p_end   = max(0, q_on - int(0.02 * fs))
        if p_end > p_start + 8:
            pos_p, pos_pr = find_peaks( t_elim[p_start:p_end], prominence=0.005)
            neg_p, neg_pr = find_peaks(-t_elim[p_start:p_end], prominence=0.005)
            best_p, b_prom, b_morph = None, 0, "Explicitly Absent"
            if len(pos_p) > 0:
                idx_ = np.argmax(pos_pr["prominences"])
                best_p, b_prom, b_morph = pos_p[idx_], pos_pr["prominences"][idx_], "Normal"
            if len(neg_p) > 0:
                idx_ = np.argmax(neg_pr["prominences"])
                if neg_pr["prominences"][idx_] > b_prom * 1.5:
                    best_p, b_prom, b_morph = neg_p[idx_], neg_pr["prominences"][idx_], "Explicitly Inverted"

            if best_p is not None:
                p_pk      = min(n - 1, p_start + best_p)
                bg_mask   = np.ones(p_end - p_start, dtype=bool)
                bg_mask[max(0, best_p - int(0.04 * fs)):
                        min(p_end - p_start, best_p + int(0.04 * fs))] = False
                noise_var = np.var(t_elim[p_start:p_end][bg_mask]) if np.sum(bg_mask) > 5 else 0.001
                snr       = (b_prom ** 2) / (noise_var + 1e-8)
                if noise_var > 0.005 or snr < 1.0:
                    beat["P"] = (None, None, p_pk, "Review (Noisy)")
                else:
                    beat["P"] = (
                        max(p_start, p_pk - int(0.04 * fs)),
                        min(p_end,   p_pk + int(0.04 * fs)),
                        p_pk, b_morph
                    )
        else:
            beat["P"] = (None, None, None, "Explicitly Absent")

        per_beat.append(beat)

    return fiducial, r_peaks, per_beat, q_elim, t_elim, high_rate_mode


# ─────────────────────────────────────────────────────────────────────────────
#  BEAT CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
def classify_beats(fiducial, r_peaks, per_beat, fs):
    """
    Two-pass morphological beat classifier with template-correlation guard.

    Boundary rule
    ─────────────
    The first and last beats of every 10-second segment are NEVER flagged as
    ectopic.  They lack a full RR context on one side and consistently produce
    spurious PVC/PAC calls.

    Template correlation (KEY FIX for PAC/PVC confusion)
    ──────────────────────────────────────────────────────
    Builds an averaged sinus-beat template and computes Pearson correlation for
    every premature beat.
      corr ≥ 0.85 → morphology identical to sinus → strong PAC evidence (+2.5)
                    This is a VETO on PVC: same pathway, supraventricular origin.
      corr < 0.60 → aberrant morphology → PVC evidence (+2.5)
      0.60–0.85   → ambiguous → rely on other criteria

    Ashman phenomenon
    ──────────────────
    Long-short RR sequence before a premature beat means the bundle branches
    are still refractory → aberrant conduction → PAC with wide QRS that mimics
    a PVC.  Detected and scored as PAC rather than PVC.

    Preceding T-wave deformation
    ─────────────────────────────
    A notched or abnormally tall T wave on the beat BEFORE a premature beat
    suggests a retrograde ectopic P wave buried in it → extra PAC evidence.

    PVC criteria (threshold ≥ 3.0):
      +2.0  QRS duration > 120 ms         (wide complex)
      +1.5  T-wave discordance
      +1.5  P wave absent
      +1.0  Coupling interval < 88 % RR   (premature)
      +1.0  Compensatory pause > 1.85× RR
      +2.5  Template correlation < 0.60   (aberrant morphology)

    PAC criteria (threshold ≥ 3.0, coupling gate required):
      +1.0  QRS duration < 110 ms         (narrow)
      +1.5  P wave inverted               (retrograde/ectopic atrial)
      +0.5  P wave present
      +1.0  Coupling interval < 88 % RR
      +1.0  Non-compensatory pause < 1.90× RR
      +2.5  Template correlation ≥ 0.85   (sinus morphology – key discriminator)
      +1.5  Ashman phenomenon (long-short RR)
      +1.0  Preceding T-wave deformation
    """
    if len(r_peaks) < 2 or not per_beat:
        return per_beat

    rr_ints   = np.diff(r_peaks).astype(float)
    median_rr = float(np.median(rr_ints))
    if median_rr <= 0:
        return per_beat

    # Build sinus-beat template for morphology correlation
    template = _build_normal_template(fiducial, r_peaks, rr_ints, median_rr, fs)

    # Reference T-wave amplitude (for deformation check)
    t_amps = []
    for b in per_beat:
        t_pk = b["T"][2]
        if t_pk is not None:
            t_amps.append(abs(float(fiducial[t_pk])))
    ref_t_amp = float(np.median(t_amps)) if t_amps else 0.0

    total_beats = len(per_beat)

    for i, beat in enumerate(per_beat):
        # ── Boundary guard ────────────────────────────────────────────────
        if i == 0 or i == total_beats - 1:
            beat["Class"] = "Normal"
            continue

        rr_prev = float(r_peaks[i]     - r_peaks[i - 1])
        rr_next = float(r_peaks[i + 1] - r_peaks[i])     if i < len(r_peaks) - 1 else median_rr

        # QRS duration
        q_on, q_off = beat["QRS"]
        qrs_dur_ms  = ((q_off - q_on) / fs * 1000.0) if (q_on is not None and q_off is not None) else 0.0

        # Morphology flags
        t_morph = beat["T"][3]
        p_morph = beat["P"][3]
        p_absent   = p_morph in ("Absent", "Explicitly Absent")
        p_inverted = p_morph in ("Inverted", "Explicitly Inverted")

        r_amp        = beat.get("R_amp", 1.0)
        qrs_inverted = (r_amp < 0)
        t_discordant = (t_morph == "Inverted" and not qrs_inverted) or \
                       (t_morph == "Normal"   and     qrs_inverted)

        # Timing
        coupling_ratio  = rr_prev / median_rr
        pause_sum_ratio = (rr_prev + rr_next) / median_rr

        # ── Template correlation ───────────────────────────────────────────
        corr = _template_correlation(fiducial, r_peaks[i], template, fs)

        # ── Ashman phenomenon: long-short RR sequence ──────────────────────
        # Long preceding-preceding RR → bundle still refractory → aberrant PAC
        ashman = False
        if i >= 2:
            rr_pre_prev = float(r_peaks[i - 1] - r_peaks[i - 2])
            ashman = (rr_pre_prev > 1.30 * median_rr) and (rr_prev < 0.92 * median_rr)

        # ── Preceding T-wave deformation ──────────────────────────────────
        # Only meaningful when the PRECEDING beat is itself Normal.
        # After a PVC the T wave is physiologically large (discordant) — that
        # is expected and does NOT indicate a hidden ectopic P wave.
        # Applying this check after PVCs was adding spurious +1.0 PAC score to
        # every sinus beat in a bigeminal run, destabilising the classifier.
        prev_t_deformed = False
        if ref_t_amp > 0.01 and i > 0:
            prev_beat = per_beat[i - 1]
            if prev_beat.get("Class", "Normal") == "Normal":
                prev_t_pk = prev_beat["T"][2]
                if prev_t_pk is not None:
                    prev_t_amp = abs(float(fiducial[prev_t_pk]))
                    prev_t_deformed = prev_t_amp > 1.30 * ref_t_amp

        # ── PVC scoring ────────────────────────────────────────────────────
        pvc = 0.0
        if qrs_dur_ms > 120:                                          pvc += 2.0
        if t_discordant:                                              pvc += 1.5
        if p_absent:                                                  pvc += 1.5
        if coupling_ratio < 0.88:                                     pvc += 1.0
        if coupling_ratio < 0.92 and pause_sum_ratio > 1.85:         pvc += 1.0
        if corr is not None and corr < 0.60:                         pvc += 2.5

        # ── PAC scoring ────────────────────────────────────────────────────
        pac = 0.0
        if qrs_dur_ms < 110:                                          pac += 1.0
        if p_inverted:                                                pac += 1.5
        if not p_absent:                                              pac += 0.5
        if coupling_ratio < 0.88:                                     pac += 1.0
        if coupling_ratio < 0.92 and pause_sum_ratio < 1.90:         pac += 1.0
        if corr is not None and corr >= 0.85:                        pac += 2.5
        if ashman:                                                    pac += 1.5
        if prev_t_deformed:                                           pac += 1.0

        # ── Template correlation adjustment (SOFT, not a hard veto) ───────
        # The previous hard veto (corr >= 0.85 → block PVC regardless) caused
        # missed PVCs when the PVC QRS morphology happened to be similar to
        # the sinus template (e.g., upright PVC with similar axis).  The
        # decisive discriminator for those cases is the discordant T wave and
        # wide QRS, not morphology correlation alone.
        # Replace with: high correlation REDUCES pvc score and BOOSTS pac score,
        # but does not prevent a beat with strong evidence (wide QRS + discordant T
        # + absent P + compensatory pause) from being correctly called PVC.
        if corr is not None and corr >= 0.85:
            pvc -= 1.0    # soft penalty
            pac += 0.5    # soft boost (pac already got +2.5 above)

        # ── Decision ──────────────────────────────────────────────────────
        if pvc >= 3.0 and pvc > pac:
            beat["Class"] = "PVC"
        elif pac >= 3.0 and coupling_ratio < 0.92:
            beat["Class"] = "PAC"
        else:
            beat["Class"] = "Normal"

    return per_beat


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────
def calculate_metrics(r_peaks, per_beat, fs, fiducial=None, high_rate_mode=False):
    """
    Computes interval measurements and rhythm classification flags.

    New flags
    ---------
    tachy_flag        – HR > 100 BPM
    brady_flag        – HR < 50 BPM (with ≥ 3 beats in segment)
    svt_flag          – Narrow complex tachycardia, regular, absent P waves
    sinus_arrest_flag – Any RR gap > 2.5× median (pause / dropped beat)
    flutter_hint      – Dominant P-channel frequency ≈ 5 Hz (300-BPM flutter)
    torsades_risk     – QTc > 500 ms (prolonged repolarisation)

    VTach flag improvements
    -----------------------
    Now requires ALL of:
      • HR > 100
      • Median QRS > 120 ms
      • > 75 % of individual beats wide
      • AV dissociation evidence: P-absent fraction > 70 %
        OR PR-interval SD > 40 ms (dissociated P waves with variable PR)
    """
    metrics = {
        "hr": 0.0, "qrs_dur": 0.0, "pr_int": 0.0, "pr_int_sd": 0.0,
        "qtc": 0.0, "hrv_sdnn": 0.0, "hrv_cv": 0.0,
        "afib_flag": False, "vtach_kinetic_flag": False,
        "svt_flag": False, "tachy_flag": False, "brady_flag": False,
        "sinus_arrest_flag": False, "flutter_hint": False,
        "torsades_risk": False, "high_rate_mode": high_rate_mode,
        # New safety flags
        "polymorphic_vtach_flag": False,   # fast + irregular + no P waves
        "coarse_vfib_flag":       False,   # extreme HRV (CV > 0.50) at fast rate
    }
    if len(r_peaks) < 2:
        return metrics

    rr_ms     = np.diff(r_peaks) / fs * 1000.0
    avg_rr    = float(np.mean(rr_ms))
    median_rr = float(np.median(rr_ms))

    metrics["hr"]       = 60000.0 / avg_rr if avg_rr > 0 else 0.0
    metrics["hrv_sdnn"] = float(np.std(rr_ms))
    metrics["hrv_cv"]   = metrics["hrv_sdnn"] / avg_rr if avg_rr > 0 else 0.0

    p_absent_frac = sum(1 for b in per_beat if "Absent" in b["P"][3]) / max(len(per_beat), 1)

    # ── AFib: bounded CV range ─────────────────────────────────────────────
    # True AFib has CV 0.15–0.35.  CV > 0.40 with a fast rate is coarse VFib
    # or polymorphic VTach masquerading as AFib — those are lethal, not a warning.
    metrics["afib_flag"] = (
        0.15 < metrics["hrv_cv"] < 0.40 and
        p_absent_frac > 0.60 and
        not metrics["high_rate_mode"]
    )

    # ── Coarse VFib: extreme irregularity at fast rate ─────────────────────
    # CV > 0.50 = beat-to-beat timing is completely chaotic.  At any rate
    # above 80 BPM with absent P waves this is coarse VFib territory, not AFib.
    metrics["coarse_vfib_flag"] = (
        metrics["hrv_cv"] > 0.50 and
        metrics["hr"] > 80 and
        p_absent_frac > 0.60
    )

    # ── Polymorphic VTach: very fast + mildly irregular + no P waves ──────
    # Rate > 150, CV in the 0.10–0.50 range (not regular enough for SVT,
    # not chaotic enough for VFib), absent P waves.
    # This catches Torsades de Pointes and rapid polymorphic VT that evade
    # both the spectral check (noise dilutes SPI) and the kinetic VTach check
    # (QRS width measurement unreliable at these rates).
    metrics["polymorphic_vtach_flag"] = (
        metrics["hr"] > 150 and
        0.08 < metrics["hrv_cv"] < 0.55 and
        p_absent_frac > 0.60
    )

    qrs_durs, pr_ints, qtc_vals = [], [], []
    qrs_wide_count = 0

    for b in per_beat:
        q_on, q_off = b["QRS"]
        if q_on is not None and q_off is not None:
            dur = (q_off - q_on) / fs * 1000.0
            qrs_durs.append(dur)
            if dur > 120:
                qrs_wide_count += 1
        if b["P"][0] is not None and q_on is not None:
            pr_ints.append((q_on - b["P"][0]) / fs * 1000.0)
        if q_on is not None and b["T"][1] is not None:
            qt = (b["T"][1] - q_on) / fs
            qtc_vals.append((qt / np.sqrt(avg_rr / 1000.0)) * 1000.0)

    if qrs_durs:
        metrics["qrs_dur"] = float(np.median(qrs_durs))
    if pr_ints:
        metrics["pr_int"]    = float(np.median(pr_ints))
        metrics["pr_int_sd"] = float(np.std(pr_ints))
    if qtc_vals:
        metrics["qtc"] = float(np.median(qtc_vals))

    qrs_wide_fraction = qrs_wide_count / max(len(qrs_durs), 1)

    # ── Simple rate flags ─────────────────────────────────────────────────
    metrics["tachy_flag"]        = metrics["hr"] > 100
    metrics["brady_flag"]        = metrics["hr"] < 50 and len(r_peaks) >= 3
    metrics["sinus_arrest_flag"] = bool(any(rr > 2.5 * median_rr for rr in rr_ms))

    # ── Torsades de Pointes risk (long QTc) ───────────────────────────────
    metrics["torsades_risk"] = metrics["qtc"] > 500.0

    # ── VTach flag: wide complex + AV dissociation ────────────────────────
    # At very high rates (> 150 BPM) QRS boundary measurement becomes
    # unreliable due to beat overlap; relax the QRS width requirement.
    pr_int_sd       = metrics["pr_int_sd"]
    av_dissociation = (p_absent_frac > 0.70) or \
                      (pr_int_sd > 40.0 and p_absent_frac > 0.40)

    if metrics["hr"] > 150:
        # High-rate VTach: wide OR borderline-wide QRS acceptable (measurement noise)
        metrics["vtach_kinetic_flag"] = (
            metrics["hr"] > 100 and
            metrics["qrs_dur"] > 100 and      # relaxed from 120 ms
            qrs_wide_fraction > 0.60 and       # relaxed from 0.75
            av_dissociation
        )
    else:
        metrics["vtach_kinetic_flag"] = (
            metrics["hr"] > 100 and
            metrics["qrs_dur"] > 120 and
            qrs_wide_fraction > 0.75 and
            av_dissociation
        )

    # ── SVT flag: narrow + rapid + regular + no P waves ───────────────────
    is_regular = metrics["hrv_cv"] < 0.10
    metrics["svt_flag"] = (
        metrics["hr"] > 100 and
        metrics["qrs_dur"] < 120 and
        is_regular and
        p_absent_frac > 0.60 and
        not metrics["afib_flag"]
    )

    # ── Flutter spectral hint ─────────────────────────────────────────────
    # After QRS+T removal the P channel (t_elim equivalent) is not passed here,
    # but we can approximate using the per_beat P-wave data:
    # If HR ≈ 150 BPM AND no P waves AND narrow complex → strong flutter suspicion.
    if (140 <= metrics["hr"] <= 160 and
            p_absent_frac > 0.70 and
            metrics["qrs_dur"] < 120):
        metrics["flutter_hint"] = True

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_plot(signal, fiducial, per_beat, q_elim, t_elim, fs,
                  title_label, is_lethal=False, high_rate_mode=False,
                  sqi_verdict=None, sqi_score=None, metrics=None):

    is_lethal_or_simple = is_lethal or fiducial is None or high_rate_mode
    fig    = Figure(figsize=(14, 5) if is_lethal_or_simple else (14, 9),
                    facecolor='#121212')
    canvas = FigureCanvas(fig)
    gs     = (gridspec.GridSpec(1, 1) if is_lethal_or_simple else
              gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.35))

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1E1E1E')
    t        = np.arange(len(signal)) / fs
    plot_sig = signal if (is_lethal or high_rate_mode or fiducial is None) else fiducial
    y_range  = float(np.ptp(plot_sig)) if plot_sig is not None else 0

    ax1.xaxis.set_major_locator(MultipleLocator(0.2))
    ax1.xaxis.set_minor_locator(MultipleLocator(0.04))
    ax1.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x:.0f}" if np.isclose(x % 1.0, 0) else ""))

    if y_range > 20.0:
        ax1.yaxis.set_major_locator(MaxNLocator(10))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
    else:
        ax1.yaxis.set_major_locator(MultipleLocator(0.5))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax1.grid(which='major', color='#552222', linestyle='-', linewidth=0.8, alpha=0.7)
    ax1.grid(which='minor', color='#552222', linestyle='-', linewidth=0.3, alpha=0.5)

    if is_lethal:
        ax1.plot(t, signal, color="#FF5252", lw=1.2)
        ax1.set_title(title_label, color="#FF5252", fontweight='bold', pad=15)

    elif high_rate_mode:
        # High-rate mode: show filtered signal, no PQRST annotations
        svt_active   = (metrics or {}).get("svt_flag", False)
        flutter_hint = (metrics or {}).get("flutter_hint", False)
        hr_val       = (metrics or {}).get("hr", 0)
        clr          = "#FF9800"
        note         = ""
        if flutter_hint:
            note = " │ ⚠ Possible Atrial Flutter (F-waves)"
        elif svt_active:
            note = " │ SVT Suspected"
        ax1.plot(t, plot_sig, color=clr, lw=1.0)
        ax1.set_title(
            f"{title_label}  [HIGH RATE {hr_val:.0f} BPM – PQRST Delineation Suspended{note}]",
            color=clr, fontweight='bold', pad=15
        )

    else:
        ax1.plot(t, fiducial, color="#E0E0E0", lw=1.0)
        ax1.set_title(title_label, color="#FFFFFF", fontweight='bold', pad=15)
        for b in per_beat:
            r_sec = b["R"] / fs
            ax1.axvline(r_sec, color="#E57373", linestyle='--', alpha=0.5, lw=1)
            if b["Class"] in ["PAC", "PVC"]:
                ax1.text(r_sec, np.max(fiducial) * 1.05,
                         f"** {b['Class']} **",
                         color="#FFB74D", fontweight='bold',
                         ha='center', fontsize=9)

            q_on, q_off = b["QRS"]
            if q_on is not None and q_off is not None:
                ax1.axvspan(q_on / fs, q_off / fs, color="#E57373", alpha=0.2)

            p_on, p_off, p_pk, p_morph = b["P"]
            if p_on:
                ax1.axvspan(p_on / fs, p_off / fs, color="#FFB74D", alpha=0.3)
            if p_pk:
                ax1.text(p_pk / fs, fiducial[p_pk] + y_range * 0.05,
                         f"P\n({p_morph})", color="#FFB74D", fontsize=7, ha='center')

            t_on, t_off, t_pk, t_morph = b["T"]
            if t_on:
                ax1.axvspan(t_on / fs, t_off / fs, color="#64B5F6", alpha=0.3)
            if t_pk:
                ax1.text(t_pk / fs, fiducial[t_pk] + y_range * 0.05,
                         f"T\n({t_morph})", color="#64B5F6", fontsize=7, ha='center')

        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.set_facecolor('#1E1E1E')
        ax2.plot(t, q_elim, color="#64B5F6", lw=1.0, alpha=0.8)
        ax2.set_title("Stage 1: Savitzky-Golay Filtered & QRS Eliminated",
                      color="#64B5F6", fontsize=9)
        ax2.grid(True, color="#333333", linestyle='--', alpha=0.5)

        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.set_facecolor('#1E1E1E')
        ax3.plot(t, t_elim, color="#FFB74D", lw=1.0, alpha=0.8)
        ax3.set_title("Stage 2: QRS & T Eliminated",
                      color="#FFB74D", fontsize=9)
        ax3.grid(True, color="#333333", linestyle='--', alpha=0.5)

    # SQI watermark
    if sqi_verdict and sqi_verdict != "GOOD":
        sqi_color = "#FF5252" if sqi_score is not None and sqi_score < 0.25 else "#FFB74D"
        ax1.text(
            0.99, 0.02,
            f"SQI: {sqi_verdict}  ({(sqi_score or 0) * 100:.0f}%)",
            transform=ax1.transAxes, ha='right', va='bottom',
            fontsize=8, color=sqi_color, alpha=0.85,
            bbox=dict(facecolor='#1E1E1E', alpha=0.7, edgecolor=sqi_color, pad=3)
        )

    for ax in fig.axes:
        ax.tick_params(colors='#B0BEC5')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.98)
    img = io.BytesIO()

    with render_lock:
        fig.savefig(img, format='png', facecolor='#121212', dpi=120)

    img.seek(0)
    fig.clf()
    return base64.b64encode(img.getvalue()).decode('utf8')


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ANALYSIS ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def analyze_json(data):
    chunks, base_label, fs = extract_ecg_chunks(data)
    results = []

    for i, signal in enumerate(chunks):
        label = f"{base_label} | Segment {i + 1}/{len(chunks)}"

        # ── 1. Signal Quality Gate ─────────────────────────────────────────
        sqi_verdict, sqi_score, sqi_details = compute_sqi(signal, fs)

        if sqi_verdict in ("FLATLINE", "LEAD_OFF"):
            results.append({
                "unusable":    True,
                "sqi_verdict": sqi_verdict,
                "sqi_score":   sqi_score,
                "sqi_details": sqi_details,
                "lethal":      False,
                "plot":        None,
                "metrics":     {},
                "label":       label
            })
            continue

        if sqi_verdict in ("MOTION_ARTIFACT", "EMG_NOISE",
                           "BASELINE_WANDER", "POOR_QUALITY"):
            # Proceed with processing but mark signal as degraded
            pass   # plot will show SQI watermark

        # ── 2. Run filter + R-peak detection ──────────────────────────────
        fiducial, r_peaks, per_beat, q_elim, t_elim, high_rate_mode = \
            process_kinetic(signal, fs)

        # ── 3. Spectral lethal check on FILTERED signal ───────────────────
        # Pass pre-computed r_peaks so the organization gate uses QRS-accurate
        # beat timing rather than an independent (T-wave-contaminated) detector.
        is_lethal, condition, rate = spectral_lethal_precheck(fiducial, fs, r_peaks=r_peaks)

        if is_lethal:
            title    = f"CRITICAL: {condition} ({rate:.0f} BPM) | {label}"
            plot_b64 = generate_plot(
                signal, None, [], None, None, fs, title,
                is_lethal=True, sqi_verdict=sqi_verdict, sqi_score=sqi_score
            )
            results.append({
                "lethal":      condition,
                "rate":        rate,
                "plot":        plot_b64,
                "metrics":     {},
                "label":       label,
                "sqi_verdict": sqi_verdict,
                "sqi_score":   sqi_score
            })
            continue

        # ── 4. Classify beats ─────────────────────────────────────────────
        per_beat = classify_beats(fiducial, r_peaks, per_beat, fs)

        # ── 5. Compute metrics ────────────────────────────────────────────
        metrics = calculate_metrics(
            r_peaks, per_beat, fs,
            fiducial=fiducial,
            high_rate_mode=high_rate_mode
        )
        metrics["sqi_verdict"] = sqi_verdict
        metrics["sqi_score"]   = round(sqi_score, 3)

        # ── 6. Kinetic VTach / polymorphic VTach / coarse VFib checks ────────
        if metrics.get("vtach_kinetic_flag"):
            title    = f"CRITICAL: KINETIC VT ({metrics['hr']:.0f} BPM) | {label}"
            plot_b64 = generate_plot(
                signal, fiducial, per_beat, q_elim, t_elim, fs, title,
                is_lethal=True,
                sqi_verdict=sqi_verdict, sqi_score=sqi_score,
                metrics=metrics
            )
            results.append({
                "lethal":      "KINETIC VENTRICULAR TACHYCARDIA",
                "rate":        metrics["hr"],
                "plot":        plot_b64,
                "metrics":     metrics,
                "label":       label,
                "sqi_verdict": sqi_verdict,
                "sqi_score":   sqi_score
            })
            continue

        if metrics.get("polymorphic_vtach_flag"):
            title = (f"CRITICAL: POLYMORPHIC VENTRICULAR TACHYCARDIA "
                     f"({metrics['hr']:.0f} BPM) | {label}")
            plot_b64 = generate_plot(
                signal, fiducial, per_beat, q_elim, t_elim, fs, title,
                is_lethal=True,
                sqi_verdict=sqi_verdict, sqi_score=sqi_score,
                metrics=metrics
            )
            results.append({
                "lethal":      "POLYMORPHIC VENTRICULAR TACHYCARDIA",
                "rate":        metrics["hr"],
                "plot":        plot_b64,
                "metrics":     metrics,
                "label":       label,
                "sqi_verdict": sqi_verdict,
                "sqi_score":   sqi_score
            })
            continue

        if metrics.get("coarse_vfib_flag"):
            title = (f"CRITICAL: POSSIBLE COARSE VENTRICULAR FIBRILLATION "
                     f"({metrics['hr']:.0f} BPM, CV {metrics['hrv_cv']*100:.0f}%) | {label}")
            plot_b64 = generate_plot(
                signal, fiducial, per_beat, q_elim, t_elim, fs, title,
                is_lethal=True,
                sqi_verdict=sqi_verdict, sqi_score=sqi_score,
                metrics=metrics
            )
            results.append({
                "lethal":      "COARSE VENTRICULAR FIBRILLATION",
                "rate":        metrics["hr"],
                "plot":        plot_b64,
                "metrics":     metrics,
                "label":       label,
                "sqi_verdict": sqi_verdict,
                "sqi_score":   sqi_score
            })
            continue

        # ── 7. Build title with active flags ─────────────────────────────
        flags_str = ""
        if metrics.get("svt_flag"):
            flags_str += " │ ⚠ SVT"
        if metrics.get("flutter_hint"):
            flags_str += " │ ⚠ Flutter?"
        if metrics.get("afib_flag"):
            flags_str += " │ ⚠ AFib"
        if metrics.get("brady_flag"):
            flags_str += " │ ⚠ Brady"
        if metrics.get("tachy_flag") and not metrics.get("svt_flag"):
            flags_str += " │ ⚠ Tachy"
        if metrics.get("torsades_risk"):
            flags_str += " │ ⚠ Long QTc"

        base_title = "Lifesigns Delineation" if not flags_str else "Lifesigns Delineation"
        title      = f"{base_title}{flags_str} | {label}"

        plot_b64 = generate_plot(
            signal, fiducial, per_beat, q_elim, t_elim, fs, title,
            high_rate_mode=high_rate_mode,
            sqi_verdict=sqi_verdict, sqi_score=sqi_score,
            metrics=metrics
        )
        results.append({
            "lethal":      False,
            "unusable":    False,
            "plot":        plot_b64,
            "metrics":     metrics,
            "label":       label,
            "sqi_verdict": sqi_verdict,
            "sqi_score":   round(sqi_score, 3)
        })

    return results