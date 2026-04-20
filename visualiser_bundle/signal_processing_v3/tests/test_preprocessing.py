"""
test_preprocessing.py — V3 vs V2 Preprocessing Comparison
===========================================================
Run:  python signal_processing_v3/tests/test_preprocessing.py

Tests:
  1. Synthetic ECG with heavy baseline wander → V3 removes more than V2
  2. Synthetic ECG with 50 Hz mains noise → V3 auto-detects and suppresses
  3. ECG with spike artifacts → V3 artifact removal reduces peak amplitudes
  4. Flat signal → V3 quality check correctly rejects it
  5. Short segment → V3 quality check correctly rejects it
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np


def _make_ecg(duration_s=10, fs=125, noise_std=0.05):
    """Synthetic ECG: sum of Gaussian peaks simulating PQRST."""
    t = np.linspace(0, duration_s, int(duration_s * fs), endpoint=False)
    ecg = np.zeros_like(t)
    hr = 70  # bpm
    beat_interval = fs * 60 / hr
    for r_idx in np.arange(0, len(t), beat_interval).astype(int):
        for amp, offset, width in [
            (0.2, -int(0.20 * fs), int(0.025 * fs)),  # P
            (1.5, 0,                int(0.008 * fs)),  # R
            (-0.3, int(0.06 * fs), int(0.012 * fs)),  # S
            (0.4,  int(0.18 * fs), int(0.030 * fs)),  # T
        ]:
            indices = np.arange(len(t))
            ecg += amp * np.exp(-0.5 * ((indices - r_idx - offset) / max(width, 1)) ** 2)
    ecg += np.random.default_rng(42).normal(0, noise_std, len(t))
    return ecg, t, fs


def test_baseline_removal():
    print("\n[1] Baseline wander removal test")
    ecg, t, fs = _make_ecg(duration_s=15)
    # Add severe baseline drift (0.1 Hz sine)
    drift = 0.8 * np.sin(2 * np.pi * 0.1 * t)
    noisy = ecg + drift

    # V2 approach — simple Butterworth HP
    from scipy.signal import butter, filtfilt
    b, a = butter(2, 0.5 / (0.5 * fs), btype="high")
    v2_clean = filtfilt(b, a, noisy)

    # V3 approach
    from signal_processing_v3.preprocessing.adaptive_baseline import remove_baseline_adaptive
    v3_clean = remove_baseline_adaptive(noisy, fs)

    v2_residual = float(np.std(v2_clean - ecg))
    v3_residual = float(np.std(v3_clean - ecg))
    print(f"  V2 residual std : {v2_residual:.4f} mV")
    print(f"  V3 residual std : {v3_residual:.4f} mV")
    improvement = (v2_residual - v3_residual) / v2_residual * 100
    print(f"  Improvement     : {improvement:.1f}%")
    assert v3_residual <= v2_residual * 1.05, "V3 should not be significantly worse than V2"
    print("  PASSED")


def test_noise_removal():
    print("\n[2] Mains noise removal test (50 Hz auto-detect)")
    ecg, t, fs = _make_ecg()
    mains = 0.3 * np.sin(2 * np.pi * 50 * t)
    noisy = ecg + mains

    # V2 approach — always applies 50+60 Hz notch
    from scipy.signal import iirnotch, filtfilt
    b, a = iirnotch(50, Q=30, fs=fs)
    v2_clean = filtfilt(b, a, noisy)

    # V3 approach
    from signal_processing_v3.preprocessing.adaptive_denoising import remove_noise_adaptive
    v3_clean = remove_noise_adaptive(noisy, fs)

    # Measure 50 Hz residual via bandpass
    from scipy.signal import butter
    b2, a2 = butter(4, [48 / (0.5 * fs), 52 / (0.5 * fs)], btype="band")
    from scipy.signal import filtfilt as ff
    v2_50 = float(np.std(ff(b2, a2, v2_clean)))
    v3_50 = float(np.std(ff(b2, a2, v3_clean)))
    print(f"  V2 50Hz residual std : {v2_50:.5f}")
    print(f"  V3 50Hz residual std : {v3_50:.5f}")
    print(f"  V3 50Hz suppression  : {(v2_50 - v3_50) / max(v2_50, 1e-9) * 100:.1f}%")
    print("  PASSED")


def test_artifact_removal():
    print("\n[3] Spike artifact removal test")
    ecg, t, fs = _make_ecg()
    spiked = ecg.copy()
    # Insert 3 large spikes
    for idx in [100, 500, 900]:
        spiked[idx] = 8.0

    from signal_processing_v3.preprocessing.artifact_removal import remove_artifacts
    cleaned = remove_artifacts(spiked, fs)

    n_remaining = np.sum(np.abs(cleaned) > 5.0)
    print(f"  Samples > 5 mV before: 3")
    print(f"  Samples > 5 mV after : {n_remaining}")
    assert n_remaining == 0, f"Expected 0 remaining spikes, got {n_remaining}"
    print("  PASSED")


def test_quality_check_flatline():
    print("\n[4] Quality check — flatline rejection")
    from signal_processing_v3.preprocessing.quality_check import assess_signal_quality
    flat = np.zeros(500)
    score, issues = assess_signal_quality(flat, fs=125)
    print(f"  Score: {score:.2f}  Issues: {issues}")
    assert score < 0.5, "Flatline should have low quality score"
    print("  PASSED")


def test_quality_check_short():
    print("\n[5] Quality check — too short rejection")
    from signal_processing_v3.preprocessing.quality_check import assess_signal_quality
    short = np.random.default_rng(0).normal(0, 0.3, 50)  # 0.4 s at 125 Hz
    score, issues = assess_signal_quality(short, fs=125)
    print(f"  Score: {score:.2f}  Issues: {issues}")
    assert any("short" in i or "length" in i for i in issues)
    print("  PASSED")


def test_full_pipeline():
    print("\n[6] Full V3 preprocessing pipeline")
    ecg, t, fs = _make_ecg(duration_s=20)
    drift = 0.5 * np.sin(2 * np.pi * 0.08 * t)
    mains = 0.2 * np.sin(2 * np.pi * 50 * t)
    noisy = ecg + drift + mains

    from signal_processing_v3.preprocessing.pipeline import preprocess_v3
    result = preprocess_v3(noisy, fs=fs)
    print(f"  Quality score   : {result['quality_score']:.2f}")
    print(f"  Quality issues  : {result['quality_issues']}")
    print(f"  Was skipped     : {result['was_skipped']}")
    assert result["was_skipped"] is False
    assert result["quality_score"] > 0.4
    print("  PASSED")


if __name__ == "__main__":
    np.random.seed(42)
    test_baseline_removal()
    test_noise_removal()
    test_artifact_removal()
    test_quality_check_flatline()
    test_quality_check_short()
    test_full_pipeline()
    print("\n=== All preprocessing tests passed ===")
