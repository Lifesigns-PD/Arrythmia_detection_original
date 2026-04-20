"""
nonlinear.py — Nonlinear HRV / Complexity Features
====================================================
Computes entropy and fractal measures from RR intervals.
All are O(N) or O(N log N); safe on short (~30s) recordings.
"""

import numpy as np
import math
from typing import Dict, Optional


def compute_nonlinear_features(r_peaks: np.ndarray, fs: int = 125) -> Dict[str, Optional[float]]:
    """
    Returns
    -------
    dict with keys:
      sample_entropy, approx_entropy, permutation_entropy,
      hurst_exponent, dfa_alpha1, sd1, sd2, sd1_sd2_ratio
    """
    keys = ["sample_entropy", "approx_entropy", "permutation_entropy",
            "hurst_exponent", "dfa_alpha1", "sd1", "sd2", "sd1_sd2_ratio"]
    empty = {k: None for k in keys}

    if r_peaks is None or len(r_peaks) < 6:
        return empty

    rr = np.diff(r_peaks).astype(float) / fs * 1000
    rr = rr[(rr > 250) & (rr < 2500)]
    if len(rr) < 4:
        return empty

    result = {}

    # ── Sample entropy ──────────────────────────────────────────────────────
    result["sample_entropy"] = _sample_entropy(rr, m=2, r_tol=0.2)

    # ── Approximate entropy ──────────────────────────────────────────────────
    result["approx_entropy"] = _approx_entropy(rr, m=2, r_tol=0.2)

    # ── Permutation entropy ──────────────────────────────────────────────────
    result["permutation_entropy"] = _permutation_entropy(rr, order=3)

    # ── Hurst exponent (R/S analysis) ────────────────────────────────────────
    result["hurst_exponent"] = _hurst_rs(rr) if len(rr) >= 8 else None

    # ── DFA α1 (short-term scaling, lags 4–16) ───────────────────────────────
    result["dfa_alpha1"] = _dfa(rr, lags=range(4, min(17, len(rr)//2))) if len(rr) >= 16 else None

    # ── Poincaré plot SD1 / SD2 ──────────────────────────────────────────────
    if len(rr) >= 3:
        sd1, sd2 = _poincare(rr)
        result["sd1"] = sd1
        result["sd2"] = sd2
        result["sd1_sd2_ratio"] = float(sd1 / sd2) if sd2 > 0 else None
    else:
        result["sd1"] = result["sd2"] = result["sd1_sd2_ratio"] = None

    return result


# ─────────────────────────────────────────────────────────────────────────────

def _sample_entropy(rr: np.ndarray, m: int = 2, r_tol: float = 0.2) -> Optional[float]:
    n   = len(rr)
    r   = r_tol * float(np.std(rr, ddof=1))
    if r == 0 or n < m + 2:
        return None

    def _count_matches(m_):
        count = 0
        for i in range(n - m_):
            template = rr[i:i + m_]
            for j in range(i + 1, n - m_):
                if np.max(np.abs(rr[j:j + m_] - template)) < r:
                    count += 1
        return count

    A = _count_matches(m + 1)
    B = _count_matches(m)
    if B == 0:
        return None
    return float(-np.log(A / B)) if A > 0 else None


def _approx_entropy(rr: np.ndarray, m: int = 2, r_tol: float = 0.2) -> Optional[float]:
    n = len(rr)
    r = r_tol * float(np.std(rr, ddof=1))
    if r == 0 or n < m + 1:
        return None

    def _phi(m_):
        count = np.zeros(n - m_)
        for i in range(n - m_):
            template = rr[i:i + m_]
            for j in range(n - m_):
                if np.max(np.abs(rr[j:j + m_] - template)) <= r:
                    count[i] += 1
        c = count / (n - m_)
        c = c[c > 0]
        return float(np.mean(np.log(c))) if len(c) > 0 else None

    phi_m  = _phi(m)
    phi_m1 = _phi(m + 1)
    if phi_m is None or phi_m1 is None:
        return None
    return float(phi_m - phi_m1)


def _permutation_entropy(rr: np.ndarray, order: int = 3) -> Optional[float]:
    n = len(rr)
    if n < order + 1:
        return None
    perms = {}
    for i in range(n - order + 1):
        p = tuple(np.argsort(rr[i:i + order]))
        perms[p] = perms.get(p, 0) + 1
    total = sum(perms.values())
    probs = np.array(list(perms.values())) / total
    probs = probs[probs > 0]
    max_entropy = np.log2(math.factorial(order))
    raw = float(-np.sum(probs * np.log2(probs)))
    return float(raw / max_entropy) if max_entropy > 0 else None


def _hurst_rs(rr: np.ndarray) -> Optional[float]:
    n = len(rr)
    lags = [int(n / k) for k in [8, 4, 2] if int(n / k) >= 4]
    if len(lags) < 2:
        return None
    rs_vals = []
    for lag in lags:
        sub = rr[:lag]
        mean_s = np.mean(sub)
        deviation = np.cumsum(sub - mean_s)
        R = np.max(deviation) - np.min(deviation)
        S = np.std(sub, ddof=1)
        if S > 0:
            rs_vals.append(R / S)
    if len(rs_vals) < 2:
        return None
    log_lags = np.log(lags[:len(rs_vals)])
    log_rs   = np.log(rs_vals)
    coeffs   = np.polyfit(log_lags, log_rs, 1)
    return float(coeffs[0])


def _dfa(rr: np.ndarray, lags) -> Optional[float]:
    y = np.cumsum(rr - np.mean(rr))
    f_vals = []
    used_lags = []
    for lag in lags:
        if lag < 4 or lag > len(rr) // 2:
            continue
        n_seg = len(y) // lag
        if n_seg < 1:
            continue
        rms_list = []
        for k in range(n_seg):
            seg  = y[k * lag: (k + 1) * lag]
            t    = np.arange(lag)
            coeffs = np.polyfit(t, seg, 1)
            trend  = np.polyval(coeffs, t)
            rms_list.append(np.sqrt(np.mean((seg - trend) ** 2)))
        if rms_list:
            f_vals.append(np.mean(rms_list))
            used_lags.append(lag)
    if len(f_vals) < 2:
        return None
    coeffs = np.polyfit(np.log(used_lags), np.log(f_vals), 1)
    return float(coeffs[0])


def _poincare(rr: np.ndarray):
    rr1 = rr[:-1]
    rr2 = rr[1:]
    sd1 = float(np.std((rr2 - rr1) / np.sqrt(2), ddof=1))
    sd2 = float(np.std((rr2 + rr1) / np.sqrt(2), ddof=1))
    return sd1, sd2
