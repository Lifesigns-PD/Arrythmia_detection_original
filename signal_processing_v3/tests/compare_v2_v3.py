"""
compare_v2_v3.py — Side-by-Side V2 vs V3 Pipeline Comparison
=============================================================
Run:  python signal_processing_v3/tests/compare_v2_v3.py [--db] [--synthetic]

Modes:
  --synthetic  (default) : generates synthetic ECG test cases
  --db                   : pulls real segments from PostgreSQL

Output: printed table + optional CSV
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_cases(n=10, fs=125):
    """Return list of (name, signal, r_peaks_true) tuples."""
    cases = []
    rng = np.random.default_rng(0)
    configs = [
        ("clean_70bpm",  70,  0.04, 20),
        ("noisy_70bpm",  70,  0.25, 20),
        ("bradycardia",  45,  0.06, 30),
        ("tachycardia", 130,  0.06, 15),
        ("low_snr",      70,  0.50, 20),
        ("baseline_wander", 70, 0.06, 20),
        ("mains_50hz",   70,  0.06, 20),
        ("short_5s",     70,  0.05,  5),
        ("long_60s",     70,  0.04, 60),
        ("mixed_noise",  70,  0.15, 20),
    ]
    for name, hr, noise, dur in configs[:n]:
        t   = np.linspace(0, dur, int(dur * fs), endpoint=False)
        ecg = np.zeros_like(t)
        beat_interval = fs * 60 / hr
        true_peaks = []
        for r_idx in np.arange(int(0.5*fs), len(t), beat_interval).astype(int):
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
        ecg += rng.normal(0, noise, len(t))
        if "baseline" in name:
            ecg += 0.8 * np.sin(2 * np.pi * 0.08 * t)
        if "mains" in name:
            ecg += 0.3 * np.sin(2 * np.pi * 50 * t)
        cases.append((name, ecg, np.array(true_peaks), fs))
    return cases


def _evaluate_peaks(detected, true_peaks, fs, tol_ms=75):
    tol = int(tol_ms * fs / 1000)
    tp = fp = fn = 0
    matched = set()
    for d in detected:
        dists = np.abs(true_peaks - d)
        idx   = int(np.argmin(dists))
        if dists[idx] <= tol and idx not in matched:
            tp += 1; matched.add(idx)
        else:
            fp += 1
    fn = len(true_peaks) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
    return prec, rec, f1


# ─────────────────────────────────────────────────────────────────────────────
# V2 pipeline (minimal)
# ─────────────────────────────────────────────────────────────────────────────

def _run_v2(signal, fs):
    t0 = time.time()
    try:
        from scipy.signal import butter, filtfilt, iirnotch
        # Preprocessing
        b, a = butter(2, 0.5/(0.5*fs), btype="high")
        clean = filtfilt(b, a, signal.astype(float))
        for fn_hz in [50, 60]:
            if fn_hz < fs/2:
                b2, a2 = iirnotch(fn_hz, Q=30, fs=fs)
                clean = filtfilt(b2, a2, clean)

        # R-peak detection
        from signal_processing.pan_tompkins import detect_r_peaks
        r_peaks = detect_r_peaks(clean, fs)

        # Delineation
        import neurokit2 as nk
        _, waves = nk.ecg_delineate(clean, r_peaks, sampling_rate=fs, method="dwt", show=False)
        n_beats = len(r_peaks)
        n_none_qrs = sum(
            1 for i in range(n_beats)
            if (waves.get("ECG_R_Onsets") or [None]*n_beats)[i] is None
        )

        # Features (V2)
        from signal_processing.feature_extraction import extract_feature_vector
        feat_vec = extract_feature_vector(clean, r_peaks, fs)
        n_features = len(feat_vec) if feat_vec is not None else 0

        elapsed = time.time() - t0
        return {
            "r_peaks": r_peaks,
            "n_beats": n_beats,
            "n_none_qrs": n_none_qrs,
            "n_features": n_features,
            "elapsed_s": elapsed,
            "error": None,
        }
    except Exception as e:
        return {"r_peaks": [], "n_beats": 0, "n_none_qrs": 0, "n_features": 0,
                "elapsed_s": time.time()-t0, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# V3 pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _run_v3(signal, fs):
    t0 = time.time()
    try:
        from signal_processing_v3 import process_ecg_v3
        result = process_ecg_v3(signal, fs=fs)
        n_none_qrs = sum(
            1 for b in result["delineation"]["per_beat"]
            if b.get("qrs_onset") is None
        )
        elapsed = time.time() - t0
        return {
            "r_peaks": result["r_peaks"],
            "n_beats": len(result["r_peaks"]),
            "n_none_qrs": n_none_qrs,
            "n_features": sum(1 for v in result["features"].values() if v is not None),
            "sqi": result["sqi"],
            "method": result["method"],
            "elapsed_s": elapsed,
            "error": None,
        }
    except Exception as e:
        return {"r_peaks": [], "n_beats": 0, "n_none_qrs": 0, "n_features": 0,
                "sqi": 0, "method": "error", "elapsed_s": time.time()-t0, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(cases):
    print("\n" + "="*90)
    print(f"  {'Case':<22} {'True':>5} | "
          f"{'V2 det':>6} {'V2 F1':>6} {'V2 miss%':>8} {'V2 feat':>7} | "
          f"{'V3 det':>6} {'V3 F1':>6} {'V3 miss%':>8} {'V3 feat':>7} {'SQI':>5}")
    print("="*90)

    totals = {"v2_f1": [], "v3_f1": [], "v2_feat": [], "v3_feat": []}

    for name, signal, true_peaks, fs in cases:
        r2 = _run_v2(signal, fs)
        r3 = _run_v3(signal, fs)

        _, _, f1_v2 = _evaluate_peaks(r2["r_peaks"], true_peaks, fs) if len(r2["r_peaks"]) else (0,0,0)
        _, _, f1_v3 = _evaluate_peaks(r3["r_peaks"], true_peaks, fs) if len(r3["r_peaks"]) else (0,0,0)

        miss_v2 = r2["n_none_qrs"] / max(r2["n_beats"], 1) * 100
        miss_v3 = r3["n_none_qrs"] / max(r3["n_beats"], 1) * 100

        totals["v2_f1"].append(f1_v2)
        totals["v3_f1"].append(f1_v3)
        totals["v2_feat"].append(r2["n_features"])
        totals["v3_feat"].append(r3["n_features"])

        flag = "✓" if f1_v3 >= f1_v2 - 0.01 else "↓"
        sqi  = f"{r3.get('sqi', 0):.2f}"

        print(f"  {name:<22} {len(true_peaks):>5} | "
              f"{r2['n_beats']:>6} {f1_v2:>6.3f} {miss_v2:>7.1f}% {r2['n_features']:>7} | "
              f"{r3['n_beats']:>6} {f1_v3:>6.3f} {miss_v3:>7.1f}% {r3['n_features']:>7} {sqi:>5}  {flag}")

        if r2["error"]:
            print(f"    [V2 ERROR] {r2['error']}")
        if r3["error"]:
            print(f"    [V3 ERROR] {r3['error']}")

    print("="*90)
    avg = lambda lst: sum(lst)/len(lst) if lst else 0
    print(f"\n  Summary:")
    print(f"    Average V2 F1  : {avg(totals['v2_f1']):.3f}")
    print(f"    Average V3 F1  : {avg(totals['v3_f1']):.3f}")
    print(f"    F1 improvement : {avg(totals['v3_f1']) - avg(totals['v2_f1']):+.3f}")
    print(f"    Avg V2 features: {avg(totals['v2_feat']):.0f}")
    print(f"    Avg V3 features: {avg(totals['v3_feat']):.0f}")
    print(f"    Feature gain   : {avg(totals['v3_feat']) - avg(totals['v2_feat']):+.0f}")


def main():
    parser = argparse.ArgumentParser(description="V2 vs V3 ECG pipeline comparison")
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--db",        action="store_true")
    parser.add_argument("--n",         type=int, default=10, help="Number of synthetic cases")
    args = parser.parse_args()

    cases = []

    if args.synthetic or not args.db:
        print(f"\nGenerating {args.n} synthetic test cases...")
        cases.extend(_make_synthetic_cases(n=args.n))

    if args.db:
        print("\nFetching real segments from database...")
        try:
            import sys; sys.path.insert(0, ".")
            from database.db_service import get_db_connection
            conn = get_db_connection()
            cur  = conn.cursor()
            cur.execute("""
                SELECT id, signal_data, sampling_rate
                FROM ecg_features_annotatable
                WHERE signal_data IS NOT NULL
                  AND is_corrected = TRUE
                LIMIT 20
            """)
            rows = cur.fetchall()
            conn.close()
            for row in rows:
                seg_id, sig_data, fs = row
                sig = np.array(sig_data, dtype=float)
                # No ground truth peaks for real data — use V3 as reference
                from signal_processing_v3.detection.ensemble import detect_r_peaks_ensemble
                ref_peaks = detect_r_peaks_ensemble(sig, fs or 125)
                cases.append((f"db_{seg_id}", sig, ref_peaks, fs or 125))
            print(f"  Loaded {len(rows)} real segments")
        except Exception as e:
            print(f"  DB fetch failed: {e}")

    if not cases:
        print("No test cases — run with --synthetic or --db")
        return

    run_comparison(cases)


if __name__ == "__main__":
    main()
