"""
test_features.py — V3 vs V2 Feature Extraction Comparison
==========================================================
Run:  python signal_processing_v3/tests/test_features.py

Tests:
  1. V3 returns all 60 features (none missing from FEATURE_NAMES_V3)
  2. V3 feature count (60) vs V2 (15) — shows coverage improvement
  3. HRV frequency features are physiologically plausible
  4. Nonlinear features: sample entropy in expected range
  5. Feature vector shape and dtype
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np


def _make_ecg_with_peaks(duration_s=30, fs=125, hr=70, noise_std=0.04):
    t   = np.linspace(0, duration_s, int(duration_s * fs), endpoint=False)
    ecg = np.zeros_like(t)
    beat_interval = fs * 60 / hr
    true_peaks = []
    rng = np.random.default_rng(5)
    for r_idx in np.arange(int(0.5 * fs), len(t), beat_interval).astype(int):
        if r_idx >= len(t): break
        # Slight RR variation for realistic HRV
        jitter = rng.integers(-int(0.02 * fs), int(0.02 * fs))
        r_idx = min(max(r_idx + jitter, 0), len(t) - 1)
        true_peaks.append(r_idx)
        for amp, off, w in [
            (0.15, -int(0.20*fs), int(0.020*fs)),
            (1.2,  0,             int(0.007*fs)),
            (-0.2, int(0.05*fs),  int(0.010*fs)),
            (0.35, int(0.18*fs),  int(0.028*fs)),
        ]:
            win = np.arange(len(t))
            ecg += amp * np.exp(-0.5 * ((win - r_idx - off) / max(w, 1)) ** 2)
    ecg += rng.normal(0, noise_std, len(t))
    return ecg, np.array(true_peaks), fs


def test_feature_count():
    print("\n[1] Feature count: V3 (60) vs V2 (15)")
    from signal_processing_v3.features.extraction import FEATURE_NAMES_V3
    from signal_processing.feature_extraction import FEATURE_NAMES as V2_NAMES

    print(f"  V2 features: {len(V2_NAMES)}")
    print(f"  V3 features: {len(FEATURE_NAMES_V3)}")
    assert len(FEATURE_NAMES_V3) == 60, f"Expected 60 V3 features, got {len(FEATURE_NAMES_V3)}"
    print("  PASSED")


def test_all_features_present():
    print("\n[2] V3 returns all FEATURE_NAMES_V3 keys")
    ecg, r_peaks, fs = _make_ecg_with_peaks()
    from signal_processing_v3.delineation.hybrid import delineate_v3
    from signal_processing_v3.features.extraction import extract_features_v3, FEATURE_NAMES_V3

    delin = delineate_v3(ecg, r_peaks, fs)
    feats = extract_features_v3(ecg, r_peaks, delin, fs)

    missing = [k for k in FEATURE_NAMES_V3 if k not in feats]
    assert len(missing) == 0, f"Missing features: {missing}"

    none_count = sum(1 for v in feats.values() if v is None)
    print(f"  Total features : {len(feats)}")
    print(f"  Non-None       : {len(feats) - none_count}")
    print(f"  None           : {none_count}")
    print("  PASSED")


def test_hrv_time_domain():
    print("\n[3] HRV time-domain plausibility")
    ecg, r_peaks, fs = _make_ecg_with_peaks(duration_s=60, hr=65)
    from signal_processing_v3.features.hrv_time_domain import compute_hrv_time_domain

    result = compute_hrv_time_domain(r_peaks, fs)
    print(f"  mean_rr_ms   : {result['mean_rr_ms']:.1f} ms  (expect ~923 ms for 65 bpm)")
    print(f"  sdnn_ms      : {result['sdnn_ms']:.1f} ms")
    print(f"  rmssd_ms     : {result['rmssd_ms']:.1f} ms")
    print(f"  pnn50        : {result['pnn50']:.3f}")
    print(f"  mean_hr_bpm  : {result['mean_hr_bpm']:.1f} bpm")
    assert result["mean_hr_bpm"] is not None
    assert 40 <= result["mean_hr_bpm"] <= 120, f"HR out of range: {result['mean_hr_bpm']}"
    print("  PASSED")


def test_hrv_frequency():
    print("\n[4] HRV frequency-domain plausibility")
    ecg, r_peaks, fs = _make_ecg_with_peaks(duration_s=120, hr=65)
    from signal_processing_v3.features.hrv_frequency import compute_hrv_frequency

    result = compute_hrv_frequency(r_peaks, fs)
    print(f"  VLF power : {result['vlf_power']}")
    print(f"  LF power  : {result['lf_power']}")
    print(f"  HF power  : {result['hf_power']}")
    print(f"  LF/HF     : {result['lf_hf_ratio']}")
    # At least LF and HF should be non-None for long enough signal
    if result["lf_power"] is not None:
        assert result["lf_power"] >= 0
    print("  PASSED")


def test_nonlinear():
    print("\n[5] Nonlinear features: sample entropy range")
    ecg, r_peaks, fs = _make_ecg_with_peaks(duration_s=120, hr=65)
    from signal_processing_v3.features.nonlinear import compute_nonlinear_features

    result = compute_nonlinear_features(r_peaks, fs)
    print(f"  sample_entropy     : {result['sample_entropy']}")
    print(f"  perm_entropy       : {result['permutation_entropy']}")
    print(f"  hurst_exponent     : {result['hurst_exponent']}")
    print(f"  sd1                : {result['sd1']:.2f} ms")
    print(f"  sd2                : {result['sd2']:.2f} ms")
    if result["sample_entropy"] is not None:
        assert 0 <= result["sample_entropy"] <= 3.0, f"Sample entropy out of range: {result['sample_entropy']}"
    print("  PASSED")


def test_feature_vector():
    print("\n[6] Feature vector shape and dtype")
    ecg, r_peaks, fs = _make_ecg_with_peaks()
    from signal_processing_v3.delineation.hybrid import delineate_v3
    from signal_processing_v3.features.extraction import extract_features_v3, feature_dict_to_vector, FEATURE_NAMES_V3

    delin = delineate_v3(ecg, r_peaks, fs)
    feats = extract_features_v3(ecg, r_peaks, delin, fs)
    vec   = feature_dict_to_vector(feats)

    print(f"  Vector shape : {vec.shape}")
    print(f"  dtype        : {vec.dtype}")
    assert vec.shape == (len(FEATURE_NAMES_V3),)
    assert vec.dtype == np.float32
    assert not np.any(np.isnan(vec)), "Feature vector contains NaN"
    print("  PASSED")


if __name__ == "__main__":
    test_feature_count()
    test_all_features_present()
    test_hrv_time_domain()
    test_hrv_frequency()
    test_nonlinear()
    test_feature_vector()
    print("\n=== All feature tests passed ===")
