# Signal Processing Methods — ECG Pipeline

This document describes the signal processing methodology used in this ECG analysis tool.
It is intended for technical reviewers and clinical validators.

---

## Overview

The pipeline processes a raw ECG signal through four sequential stages and produces
clinical features comparable against an independent NeuroKit2 reference implementation.

```
Raw ECG Signal
      │
      ▼
1. Preprocessing        — noise removal, baseline correction
      │
      ▼
2. R-Peak Detection     — locate each heartbeat
      │
      ▼
3. Waveform Delineation — mark P, Q, R, S, T boundaries per beat
      │
      ▼
4. Feature Extraction   — compute clinical intervals and HRV metrics
```

---

## Stage 1 — Preprocessing

**Goal:** Remove noise sources that obscure clinical features without distorting waveform morphology.

### Baseline Wander Removal
- **Method:** Morphological baseline estimation (rolling median over 0.6-second window)
- **Removes:** Respiratory drift (0.15–0.4 Hz), electrode impedance fluctuations
- **Why median, not high-pass filter:** Preserves DC level without phase distortion that shifts P-wave timing

### Bandpass Filtering
- **Method:** 4th-order Butterworth bandpass, 0.5–40 Hz, zero-phase (`filtfilt`)
- **Removes:** Sub-0.5 Hz baseline residual + powerline harmonics + EMG > 40 Hz
- **Clinical range preserved:** QRS (5–40 Hz), P and T waves (0.5–10 Hz)

---

## Stage 2 — R-Peak Detection

**Goal:** Accurately locate the R-peak (tallest point of each QRS complex) to measure
heart rate and anchor all subsequent waveform measurements.

### Ensemble Approach
Three independent detectors run in parallel. A peak is confirmed only if **≥ 2 of 3
detectors agree** within a 50 ms tolerance window:

| Detector | Principle |
|----------|-----------|
| Pan-Tompkins | Derivative → square → moving average; threshold-based |
| Hilbert Envelope | Analytic signal envelope of bandpass-filtered QRS band |
| Wavelet | Multi-scale energy at QRS-matched scales |

### Polarity Handling
Some ECG leads record a downward (negative) QRS deflection (e.g. aVR, certain ambulatory
lead placements). The pipeline automatically detects inverted polarity and processes on the
corrected signal, returning peak indices in the original signal coordinates.

### Post-Detection Validation
- Physiological refractory period enforced: minimum 200 ms between peaks (> 300 bpm impossible)
- Maximum 3000 ms gap enforced (< 20 bpm is bradycardia floor)
- RR outliers > 3× median RR removed (ectopic or detection artefact)
- Sub-sample parabolic interpolation applied for HRV accuracy (±4 ms quantisation at 125 Hz)

---

## Stage 3 — Waveform Delineation

**Goal:** Mark the onset, peak, and offset of each P, Q, R, S, T waveform component
per beat to compute PR interval, QRS duration, QT interval, and P-wave morphology.

### Method
- **QRS boundaries:** Slope and amplitude criteria relative to R-peak amplitude
  - Onset: signal rising steeply toward R (slope threshold)
  - Offset (J-point): signal descends below 15% of R amplitude AND slope < 0.025 mV/sample
  - Minimum 40 ms post-R before searching for J-point
- **T-wave:** Search window 60–400 ms after J-point; peak by amplitude; boundaries by slope return
- **P-wave:** Search window 50–250 ms before QRS onset; validated by energy ratio gate
  - P-wave accepted if: segment energy ≥ 2.5× local TP baseline energy AND peak amplitude ≥ 0.04 mV
  - This gate rejects AF f-waves and noise masquerading as P-waves

### P-Wave Morphology Classification
Each detected P-wave is classified:
- **Normal:** positive deflection, energy > gate threshold
- **Inverted:** negative deflection (retrograde atrial activation)
- **Biphasic:** both positive and negative components
- **Absent:** energy below gate threshold (AF, junctional rhythm, noise)

---

## Stage 4 — Feature Extraction

### Interval Features
| Feature | Definition | Clinical Use |
|---------|-----------|--------------|
| HR (bpm) | 60,000 / mean(RR intervals in ms) | Rate classification |
| PR interval (ms) | QRS onset − P-wave onset | AV conduction delay |
| QRS duration (ms) | QRS offset − QRS onset | Conduction width (BBB if >120 ms) |
| QT interval (ms) | T-wave offset − QRS onset | Repolarisation |
| QTc (Bazett) | QT / √(RR/1000) | Rate-corrected QT |

### HRV Features
| Feature | Formula | Clinical Use |
|---------|---------|--------------|
| SDNN (ms) | std(RR intervals) | Overall HRV |
| RMSSD (ms) | √mean(ΔRR²) | Vagal tone / AF irregularity |
| pNN50 | count(\|ΔRR\| > 50ms) / N | Parasympathetic activity |
| LF/HF ratio | LF power / HF power (Welch PSD) | Autonomic balance |

### Beat Discriminators
| Feature | Purpose |
|---------|---------|
| p_absent_fraction | Fraction of beats with no detectable P-wave (high in AF) |
| qrs_wide_fraction | Fraction of beats with QRS > 120 ms (BBB, PVC detection) |
| compensatory_pause_fraction | Post-ectopic RR ≈ 2× normal (PVC indicator) |
| short_coupling_fraction | Early ectopic beats (PVC/PAC indicator) |
| t_discordant_fraction | T-wave opposite to QRS polarity (ventricular origin) |

---

## Comparison Against NeuroKit2

The report tool runs both pipelines on the same signal and compares outputs side-by-side.

**NeuroKit2** (reference): open-source, well-validated, uses DWT delineation.

**Key difference at 125 Hz:** NeuroKit2 DWT delineation was designed and tuned primarily for
500 Hz recordings. At 125 Hz the wavelet scales are coarser, which can cause QRS boundary
offsets of 30–50 ms for some morphologies. The custom pipeline uses amplitude-relative
thresholds that are sampling-rate independent.

The `⚠` indicator in the feature table flags disagreements > 15% between pipelines.

---

## Clinical Normal Ranges (Reference)

| Metric | Normal Range | Abnormal |
|--------|-------------|----------|
| Heart Rate | 60–100 bpm | Brady < 60, Tachy > 100 |
| PR interval | 120–200 ms | 1st Degree AVB > 200 ms |
| QRS duration | 80–120 ms | BBB / PVC > 120 ms |
| QTc (Bazett) | 350–440 ms (M), 350–460 ms (F) | Long QT > 470 ms |
| SDNN | > 50 ms | Reduced HRV (cardiac risk) |
| p_absent_fraction | < 0.15 (Sinus) | > 0.70 suggests AF |

---

## Dependencies

```
numpy >= 1.21
scipy >= 1.7
matplotlib >= 3.4
neurokit2 >= 0.2
```
