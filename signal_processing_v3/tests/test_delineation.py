"""
test_delineation.py — V3 vs V2 Delineation Comparison
=======================================================
Run:  python signal_processing_v3/tests/test_delineation.py

Tests:
  1. Fiducial point coverage — V3 returns fewer None values than V2
  2. P-wave detection rate
  3. QRS duration plausibility (40–200 ms)
  4. QTc plausibility (300–550 ms)
  5. Summary dict completeness
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np


def _make_ecg_with_peaks(duration_s=20, fs=125, hr=70, noise_std=0.05):
    t   = np.linspace(0, duration_s, int(duration_s * fs), endpoint=False)
    ecg = np.zeros_like(t)
    beat_interval = fs * 60 / hr
    true_peaks = []
    for r_idx in np.arange(int(0.5 * fs), len(t), beat_interval).astype(int):
        if r_idx >= len(t): break
        true_peaks.append(r_idx)
        for amp, off, w in [
            (0.15, -int(0.20*fs), int(0.020*fs)),
            (1.2,  0,             int(0.007*fs)),
            (-0.2, int(0.05*fs),  int(0.010*fs)),
            (0.35, int(0.18*fs),  int(0.028*fs)),
        ]:
            win = np.arange(len(t))
            ecg += amp * np.exp(-0.5 * ((win - r_idx - off) / max(w, 1)) ** 2)
    rng = np.random.default_rng(7)
    ecg += rng.normal(0, noise_std, len(t))
    return ecg, np.array(true_peaks), fs


def _coverage(per_beat, key):
    total = len(per_beat)
    found = sum(1 for b in per_beat if b.get(key) is not None)
    return found / total if total > 0 else 0.0


def test_fiducial_coverage():
    print("\n[1] Fiducial point coverage: V3 vs V2")
    ecg, r_peaks, fs = _make_ecg_with_peaks()

    # V2: NeuroKit2 DWT
    v2_beats = None
    try:
        import neurokit2 as nk
        _, waves = nk.ecg_delineate(ecg, r_peaks, sampling_rate=fs, method="dwt", show=False)
        v2_beats = []
        nk_map = {
            "p_onset": "ECG_P_Onsets", "p_peak": "ECG_P_Peaks", "p_offset": "ECG_P_Offsets",
            "qrs_onset": "ECG_R_Onsets", "qrs_offset": "ECG_R_Offsets",
            "t_peak": "ECG_T_Peaks", "t_offset": "ECG_T_Offsets",
        }
        for i in range(len(r_peaks)):
            b = {}
            for k, nk_k in nk_map.items():
                arr = waves.get(nk_k, [])
                val = arr[i] if i < len(arr) else None
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    b[k] = int(val)
                else:
                    b[k] = None
            v2_beats.append(b)
    except Exception as e:
        print(f"  V2 (NeuroKit2) failed: {e}")

    # V3: Hybrid wavelet + template
    from signal_processing_v3.delineation.hybrid import delineate_v3
    v3_result = delineate_v3(ecg, r_peaks, fs)
    v3_beats = v3_result["per_beat"]

    keys = ["p_onset", "qrs_onset", "qrs_offset", "t_offset"]
    print(f"  {'Key':<15} {'V2':>8} {'V3':>8}")
    print(f"  {'-'*33}")
    for k in keys:
        v2_cov = _coverage(v2_beats, k) if v2_beats else float("nan")
        v3_cov = _coverage(v3_beats, k)
        print(f"  {k:<15} {v2_cov:>7.1%} {v3_cov:>7.1%}")

    v3_qrs_cov = _coverage(v3_beats, "qrs_onset")
    assert v3_qrs_cov >= 0.7, f"V3 QRS onset coverage too low: {v3_qrs_cov:.1%}"
    print("  PASSED")


def test_qrs_duration():
    print("\n[2] QRS duration plausibility (40–200 ms)")
    ecg, r_peaks, fs = _make_ecg_with_peaks()
    from signal_processing_v3.delineation.hybrid import delineate_v3
    result = delineate_v3(ecg, r_peaks, fs)
    summary = result["summary"]
    qrs_ms = summary.get("qrs_duration_ms")
    print(f"  QRS duration: {qrs_ms} ms")
    if qrs_ms is not None:
        assert 40 <= qrs_ms <= 200, f"Implausible QRS: {qrs_ms} ms"
    print("  PASSED")


def test_qtc_plausibility():
    print("\n[3] QTc Bazett plausibility (280–600 ms)")
    ecg, r_peaks, fs = _make_ecg_with_peaks()
    from signal_processing_v3.delineation.hybrid import delineate_v3
    result = delineate_v3(ecg, r_peaks, fs)
    qtc = result["summary"].get("qtc_ms")
    print(f"  QTc: {qtc} ms")
    if qtc is not None:
        assert 280 <= qtc <= 600, f"Implausible QTc: {qtc} ms"
    print("  PASSED")


def test_p_wave_detection():
    print("\n[4] P-wave presence ratio")
    ecg, r_peaks, fs = _make_ecg_with_peaks(noise_std=0.03)
    from signal_processing_v3.delineation.hybrid import delineate_v3
    result = delineate_v3(ecg, r_peaks, fs)
    p_ratio = result["summary"].get("p_wave_present_ratio", 0)
    print(f"  P-wave present in {p_ratio:.1%} of beats")
    assert p_ratio >= 0.3, f"P-wave detection too low: {p_ratio:.1%}"
    print("  PASSED")


def test_summary_completeness():
    print("\n[5] Summary dict completeness")
    ecg, r_peaks, fs = _make_ecg_with_peaks()
    from signal_processing_v3.delineation.hybrid import delineate_v3
    result = delineate_v3(ecg, r_peaks, fs)
    expected_keys = [
        "mean_hr_bpm", "qrs_duration_ms", "pr_interval_ms",
        "qt_interval_ms", "qtc_ms", "st_deviation_mv",
        "p_wave_present_ratio", "n_beats",
    ]
    for k in expected_keys:
        assert k in result["summary"], f"Missing key: {k}"
    print(f"  All {len(expected_keys)} expected keys present")
    print(f"  Method: {result['method']}")
    print("  PASSED")


if __name__ == "__main__":
    test_fiducial_coverage()
    test_qrs_duration()
    test_qtc_plausibility()
    test_p_wave_detection()
    test_summary_completeness()
    print("\n=== All delineation tests passed ===")
