# Signal Processing V3

Comprehensive ECG signal processing pipeline — complete redesign over V2.

## What Changed From V2

| Stage | V2 | V3 |
|-------|----|----|
| Preprocessing | HP→Notch→LP | Adaptive quality check + baseline + denoising + artifact removal |
| R-peak detection | Pan-Tompkins + fallback | Ensemble: Pan-Tompkins + Hilbert + Wavelet, voted |
| Delineation | NeuroKit2 DWT (fails ~40%) + heuristic | CWT-based P/Q/R/S/T detection with inverted-lead handling |
| Features | 15 basic features | **60 features**: HRV time/freq, nonlinear, morphology, beat discriminators |
| Beat typing | None | Per-beat PVC/PAC scoring with 7+ electrophysiology criteria each |
| Quality | Single SQI score | Multi-criteria quality framework (10 checks) |

## Directory Layout

```
signal_processing_v3/
├── README.md                    ← This file
├── __init__.py                  ← Top-level API: process_ecg_v3()
│
├── preprocessing/
│   ├── quality_check.py         ← Pre-flight: saturation, flatline, noise check
│   ├── adaptive_baseline.py     ← Morphological + Savitzky-Golay baseline removal
│   ├── adaptive_denoising.py    ← Adaptive notch (auto-detect 50 vs 60 Hz)
│   ├── artifact_removal.py      ← Wavelet-based muscle artifact suppression
│   └── pipeline.py              ← Orchestrates all preprocessing stages: preprocess_v3()
│
├── detection/
│   ├── pan_tompkins.py          ← Classic Pan-Tompkins
│   ├── hilbert_detector.py      ← Hilbert envelope-based QRS detection
│   ├── wavelet_detector.py      ← Wavelet modulus maxima detection
│   └── ensemble.py              ← Vote across 3 detectors + polarity detection + validate RR
│
├── delineation/
│   ├── wavelet_delineation.py   ← CWT-based P/Q/R/S/T per-beat boundaries (inverted-lead aware)
│   ├── template_matching.py     ← Patient-specific template matching
│   ├── hybrid.py                ← Combines wavelet + template, falls back gracefully
│   └── __init__.py              ← Exports delineate_v3()
│
├── features/
│   ├── hrv_time_domain.py       ← SDNN, RMSSD, pNN50, skewness, kurtosis etc. (11 features)
│   ├── hrv_frequency.py         ← LF/HF power, LF/HF ratio, spectral entropy (8 features)
│   ├── nonlinear.py             ← Sample entropy, Hurst, DFA, SD1/SD2, Poincaré (8 features)
│   ├── morphology_features.py   ← QRS area, T-wave asymmetry, ST slope, QTc (13 features)
│   ├── beat_morphology.py       ← Per-beat PVC/PAC discriminators, delta wave, T-inversion (20 features)
│   ├── extraction.py            ← Master extractor: FEATURE_NAMES_V3, extract_features_v3(), feature_dict_to_vector()
│   └── __init__.py
│
├── quality/
│   └── signal_quality.py        ← SQI v3: 10-criteria quality scoring + issue flags
│
└── tests/
    ├── test_preprocessing.py    ← Unit + visual tests for preprocessing
    ├── test_detection.py        ← R-peak detection accuracy tests
    ├── test_delineation.py      ← Delineation quality tests
    ├── test_features.py         ← Feature extraction sanity tests
    └── compare_v2_v3.py         ← Side-by-side V2 vs V3 comparison on real DB data
```

## Quick Start

```python
from signal_processing_v3 import process_ecg_v3

result = process_ecg_v3(signal, fs=125)

print(result["r_peaks"])         # R-peak sample indices (numpy array)
print(result["features"])        # Dict of 60 named features
print(result["feature_vector"])  # float32 ndarray shape (60,) — ready for model
print(result["sqi"])             # 0.0–1.0 quality score
print(result["sqi_issues"])      # e.g. ["baseline_wander", "high_noise"]
print(result["delineation"])     # Per-beat P/Q/R/S/T boundaries
print(result["skipped"])         # True if quality too low to process
```

## Feature Vector (60 features)

```
HRV Time Domain (11):
  mean_rr_ms, sdnn_ms, rmssd_ms, pnn50, pnn20,
  rr_range_ms, rr_cv, rr_skewness, rr_kurtosis,
  triangular_index, mean_hr_bpm

HRV Frequency (8):
  vlf_power, lf_power, hf_power, lf_hf_ratio,
  lf_norm, hf_norm, total_power, spectral_entropy_hrv

Nonlinear HRV (8):
  sample_entropy, approx_entropy, permutation_entropy,
  hurst_exponent, dfa_alpha1, sd1, sd2, sd1_sd2_ratio

Morphology (13):
  qrs_duration_ms, qrs_area, pr_interval_ms, qt_interval_ms,
  qtc_bazett_ms, st_elevation_mv, st_slope, t_wave_asymmetry,
  r_s_ratio, p_wave_duration_ms, t_wave_amplitude_mv,
  r_amplitude_mv, qrs_amplitude_ms_product

Beat Discriminators (20):
  qrs_wide_fraction, mean_qrs_duration_ms, qrs_duration_std_ms,
  p_absent_fraction, p_inverted_fraction, p_biphasic_fraction,
  mean_coupling_ratio, short_coupling_fraction,
  compensatory_pause_fraction,
  t_discordant_fraction, t_inverted_fraction,
  qrs_negative_fraction,
  mean_rs_ratio, rs_ratio_std,
  mean_q_depth, pathological_q_fraction,
  mean_s_depth,
  delta_wave_fraction,
  pvc_score_mean, pac_score_mean
```

## Waveform Detection (per beat)

V3 delineates all 5 waveform components per beat:

| Component | Detected | Key Outputs |
|-----------|----------|-------------|
| P wave    | Yes | p_onset, p_peak, p_offset, p_amplitude, p_morphology (normal/inverted/biphasic/absent) |
| Q wave    | Yes | q_peak, q_depth |
| R wave    | Yes | r_peak, r_amplitude, qrs_polarity (positive/negative/isoelectric) |
| S wave    | Yes | s_peak, s_depth |
| T wave    | Yes | t_onset, t_peak, t_offset, t_amplitude, t_inverted |

Inverted-lead handling: when QRS polarity is "negative" (e.g. aVR), detection runs on the
flipped signal internally; all indices map back to the original signal coordinates.

Delta wave (WPW pre-excitation) is detected as slurring of the early QRS upstroke.

## PVC vs PAC Discrimination

Each beat receives a `pvc_score` and `pac_score` [0–1]:

**PVC criteria** (wide complex, ventricular origin):
- QRS duration > 120 ms (+3.0)
- No preceding P wave (+2.5)
- Compensatory pause: post-beat RR ≈ 2× normal RR (+2.0)
- T wave discordant (opposite to QRS) (+2.0)
- QRS > 150 ms (additional weight +2.0)
- Short coupling interval (+1.0)
- Negative QRS polarity (+1.0)
- Deep Q wave (+0.5)

**PAC criteria** (narrow complex, atrial origin):
- Narrow QRS < 110 ms (+2.5)
- Inverted or biphasic P wave (+3.0)
- Non-compensatory pause (+1.5)
- Short coupling interval (+1.5)
- Concordant T wave (+1.0)

The `pvc_score_mean` and `pac_score_mean` in the feature vector are the median scores
across all beats in the segment, used by the decision engine rules layer.

## Running Tests

```bash
# Individual module tests
python signal_processing_v3/tests/test_preprocessing.py
python signal_processing_v3/tests/test_detection.py
python signal_processing_v3/tests/test_delineation.py
python signal_processing_v3/tests/test_features.py

# Compare V2 vs V3 on synthetic or real database data
python signal_processing_v3/tests/compare_v2_v3.py --synthetic
python signal_processing_v3/tests/compare_v2_v3.py --db --n 100
```

## Integration Points

- **`ecg_processor.py`** — calls `process_ecg_v3()` per 10-second window; passes 60 features into `clinical_features` dict for the decision engine
- **`xai/xai.py`** — uses `FEATURE_NAMES_V3` for SHAP explanations in real-time output
- **`models_training/data_loader.py`** — loads V3 features from DB for Rhythm + Ectopy model training
- **`models_training/retrain_v2.py`** — trains CNN+Transformer on 60 V3 features; 9 rhythm classes + 3 ectopy classes
- **`scripts/backfill_features.py`** — one-time script to backfill V3 features for all existing DB segments

## Retraining Workflow

After backfilling features, run in order:

```bash
# 1. Backfill V3 features for existing DB segments
python scripts/backfill_features.py

# 2. Retrain rhythm model (15 classes, 60 features)
python models_training/retrain_v2.py --task rhythm --mode initial

# 3. Retrain ectopy model (3 classes: None/PVC/PAC, 60 features)
python models_training/retrain_v2.py --task ectopy --mode initial

# 4. Evaluate against previous checkpoint
python scripts/compare_models.py --task rhythm
```

Only `is_corrected = TRUE` segments are used for training (cardiologist-verified only).
