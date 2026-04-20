"""
signal_quality.py — V3 Signal Quality Index (SQI)
===================================================
10-criteria quality score returning a float in [0, 1].
Criteria:
  1. No NaN / Inf
  2. Sufficient length (≥ 2 s)
  3. Non-flatline (std > 0.01 mV)
  4. No saturation (max |x| < 5 mV)
  5. Clipping ratio < 5 %
  6. SNR (signal-to-noise via bandpass residual) > 10 dB
  7. HF noise ratio < 40 % (power above 40 Hz / total)
  8. Baseline wander < 30 % (power below 0.5 Hz / total)
  9. R-peak regularity (CV of RR intervals < 0.35)
  10. Sufficient beats detected (≥ 2 for short, ≥ 3 for > 5 s)
"""

import numpy as np
from scipy.integrate import trapezoid
from typing import Dict, List, Tuple, Optional


def compute_sqi_v3(
    signal: np.ndarray,
    r_peaks: Optional[np.ndarray] = None,
    fs: int = 125,
) -> Tuple[float, List[str]]:
    """
    Returns
    -------
    (score, issues)
      score  : float in [0, 1]; 1.0 = perfect quality
      issues : list of human-readable strings describing failed criteria
    """
    issues: List[str] = []
    passed = 0
    total  = 10

    sig = np.asarray(signal, dtype=float)

    # 1. NaN / Inf check
    if np.all(np.isfinite(sig)):
        passed += 1
    else:
        issues.append("signal contains NaN or Inf")
        sig = np.nan_to_num(sig)

    # 2. Length ≥ 2 s
    if len(sig) >= int(2 * fs):
        passed += 1
    else:
        issues.append(f"signal too short ({len(sig)/fs:.1f} s < 2 s)")

    # 3. Not flatline
    if np.std(sig) > 0.01:
        passed += 1
    else:
        issues.append("signal is flatline (std < 0.01 mV)")

    # 4. No saturation
    if np.max(np.abs(sig)) < 5.0:
        passed += 1
    else:
        issues.append(f"amplitude too large ({np.max(np.abs(sig)):.2f} mV > 5 mV)")

    # 5. Clipping ratio < 5 %
    max_val = np.max(np.abs(sig))
    clip_ratio = np.mean(np.abs(sig) > 0.98 * max_val) if max_val > 0 else 0.0
    if clip_ratio < 0.05:
        passed += 1
    else:
        issues.append(f"clipping detected ({clip_ratio*100:.1f}% of samples)")

    # 6. SNR > 10 dB
    try:
        from scipy.signal import butter, filtfilt
        b, a = butter(4, [1.0 / (0.5 * fs), 40.0 / (0.5 * fs)], btype="band")
        clean = filtfilt(b, a, sig)
        noise = sig - clean
        sig_power   = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power > 0:
            snr_db = 10 * np.log10(sig_power / noise_power)
        else:
            snr_db = 60.0
        if snr_db > 10:
            passed += 1
        else:
            issues.append(f"low SNR ({snr_db:.1f} dB < 10 dB)")
    except Exception:
        passed += 1  # Can't compute — don't penalise

    # 7. HF noise ratio < 40 %
    try:
        from scipy.signal import periodogram
        f, psd = periodogram(sig, fs=fs)
        total_pow = trapezoid(psd, f) + 1e-12
        hf_pow    = trapezoid(psd[f > 40], f[f > 40]) if np.any(f > 40) else 0.0
        hf_ratio  = hf_pow / total_pow
        if hf_ratio < 0.40:
            passed += 1
        else:
            issues.append(f"excessive HF noise ({hf_ratio*100:.0f}% > 40%)")
    except Exception:
        passed += 1

    # 8. Baseline wander < 30 %
    try:
        bw_pow = trapezoid(psd[f < 0.5], f[f < 0.5]) if np.any(f < 0.5) else 0.0
        bw_ratio = bw_pow / total_pow
        if bw_ratio < 0.30:
            passed += 1
        else:
            issues.append(f"excessive baseline wander ({bw_ratio*100:.0f}% > 30%)")
    except Exception:
        passed += 1

    # 9. R-peak RR regularity (CV < 0.35)
    if r_peaks is not None and len(r_peaks) >= 3:
        rr = np.diff(r_peaks).astype(float)
        rr = rr[(rr > int(0.25 * fs)) & (rr < int(2.5 * fs))]
        if len(rr) >= 2:
            cv = np.std(rr, ddof=1) / np.mean(rr) if np.mean(rr) > 0 else 1.0
            if cv < 0.35:
                passed += 1
            else:
                issues.append(f"irregular RR intervals (CV={cv:.2f} > 0.35)")
        else:
            issues.append("insufficient valid RR intervals")
    else:
        # No peaks provided — neutral
        passed += 1

    # 10. Sufficient beats
    min_beats = 3 if len(sig) / fs > 5 else 2
    n_beats = len(r_peaks) if r_peaks is not None else 0
    if n_beats >= min_beats:
        passed += 1
    else:
        issues.append(f"too few beats detected ({n_beats} < {min_beats})")

    score = passed / total
    return float(score), issues
