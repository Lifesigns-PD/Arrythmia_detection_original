#!/usr/bin/env python3
"""
generate_sample.py — Create realistic synthetic ECG JSON samples
================================================================
Run this once to generate sample_ecg.json for testing the report script.
Uses NeuroKit2's synthetic ECG generator.

Usage:
    python generate_sample.py
    → creates sample_ecg.json
"""

import json
import numpy as np

try:
    import neurokit2 as nk

    segments = []

    configs = [
        ("Sinus Rhythm",         dict(heart_rate=75,  duration=10)),
        ("Sinus Bradycardia",    dict(heart_rate=48,  duration=10)),
        ("Sinus Tachycardia",    dict(heart_rate=115, duration=10)),
        ("Atrial Fibrillation",  dict(heart_rate=85,  duration=10)),
    ]

    for label, kwargs in configs:
        ecg = nk.ecg_simulate(sampling_rate=125, noise=0.05, **kwargs)
        segments.append({
            "signal": ecg.tolist(),
            "fs": 125,
            "label": label
        })
        print(f"Generated: {label}")

    with open("sample_ecg.json", "w") as f:
        json.dump(segments, f)
    print("\nSaved: sample_ecg.json")
    print("Run:   python ecg_pipeline_report.py sample_ecg.json")

except ImportError:
    # Fallback: pure numpy synthetic ECG (simplified)
    print("neurokit2 not found — generating simplified synthetic ECG")

    def make_ecg(hr=75, fs=125, duration=10, noise=0.05):
        n = fs * duration
        t = np.arange(n) / fs
        period = fs * 60 / hr
        sig = np.zeros(n)
        for beat_start in np.arange(0, n, period):
            bs = int(beat_start)
            # P wave
            for j in range(max(0, bs-20), min(n, bs-5)):
                sig[j] += 0.15 * np.exp(-((j-bs+12)**2) / 10)
            # QRS
            for j in range(max(0, bs-5), min(n, bs+5)):
                sig[j] += 1.0 * np.exp(-((j-bs)**2) / 3) - 0.15 * np.exp(-((j-bs+3)**2) / 2)
            # T wave
            for j in range(max(0, bs+8), min(n, bs+25)):
                sig[j] += 0.25 * np.exp(-((j-bs-16)**2) / 20)
        sig += np.random.randn(n) * noise
        return sig.tolist()

    segments = [
        {"signal": make_ecg(hr=75),  "fs": 125, "label": "Sinus Rhythm"},
        {"signal": make_ecg(hr=48),  "fs": 125, "label": "Sinus Bradycardia"},
        {"signal": make_ecg(hr=115), "fs": 125, "label": "Sinus Tachycardia"},
    ]
    with open("sample_ecg.json", "w") as f:
        json.dump(segments, f)
    print("Saved: sample_ecg.json")
