# Signal Processing Design — V3 Pipeline
## Why This Method, Pros/Cons, Design Rationale

---

## 1. Overview: What the Signal Processing Stage Does

Every incoming ECG segment passes through V3 before any model sees it.
The pipeline answers three questions:

| Question | Output |
|----------|--------|
| Is this signal usable? | SQI score [0–1] + issue list |
| Where are the heartbeats? | R-peak sample indices |
| What are the exact waveform boundaries? | Per-beat P/Q/R/S/T locations |

From those answers, **60 numerical features** are computed that describe
the whole segment's heart rate variability, morphology, and beat-level
ectopy characteristics. These 60 numbers are the language the CNN+Transformer
model speaks.

---

## 2. Stage 1 — Preprocessing

### What It Does
Four sequential operations on the raw ADC signal:

| Step | Module | What It Removes |
|------|--------|-----------------|
| Pre-flight quality check | `quality_check.py` | Flat-line, saturation, NaN/Inf — abort early |
| Adaptive baseline removal | `adaptive_baseline.py` | Respiration-induced wandering (0.05–0.5 Hz drift) |
| Adaptive denoising | `adaptive_denoising.py` | Powerline interference (auto-detects 50 vs 60 Hz) |
| Wavelet artifact removal | `artifact_removal.py` | Muscle/motion artifact bursts |

### Why This Design (vs V2 Fixed Filter Chain)

**V2 approach**: Fixed high-pass (0.5 Hz) → Notch (50 Hz) → Low-pass (40 Hz)

**Problem with V2**:
- A fixed 0.5 Hz high-pass distorts QRS morphology when baseline wander is slow
- Notch filter at exactly 50 Hz misses 50.3 Hz or 49.8 Hz powerline on cheap devices
- No artifact detection — muscle bursts pass through and confuse the delineation stage
- No early-abort: a flat-line segment wastes compute through the full pipeline

**V3 approach**:
- Baseline removal uses **morphological opening** (math-morphology median filter),
  which is waveform-shape-aware — it follows the baseline contour without touching QRS
- Notch frequency is **estimated from the signal spectrum** (dominant peak between 45–65 Hz),
  so it tracks the actual powerline on any device
- Artifact removal uses **discrete wavelet transform** — replaces contaminated wavelet
  coefficients in HF bands without touching the low-frequency morphology
- Early abort: if `quality_score < 0.4`, preprocessing returns immediately, saving ~80 ms

### Pros / Cons

| | Pros | Cons |
|-|------|------|
| Adaptive baseline | Preserves QRS shape during heavy wander | Slower than fixed IIR (adds ~5 ms per 10s window) |
| Auto-detect notch freq | Works on both 50 Hz and 60 Hz mains without config | Fails on very short (<2s) noisy clips where spectrum is unreliable |
| Wavelet artifact removal | Removes burst noise without phase distortion | Moderate complexity; threshold tuning matters |
| Early quality abort | Prevents garbage features from entering model | Short poor-quality bursts (<0.4 SQI) silently skipped — need monitoring |

---

## 3. Stage 2 — R-Peak Detection (Ensemble)

### What It Does
Three independent algorithms vote on where the QRS complexes are:

| Detector | Algorithm Basis | Strength |
|----------|----------------|----------|
| Pan-Tompkins | Differentiation + squaring + moving average | Fast; robust on normal sinus |
| Hilbert | Analytic signal envelope | Handles irregular amplitude (AF, BBB) |
| Wavelet | Multi-scale modulus maxima | Handles low-amplitude QRS (bradycardic, paediatric) |

A peak at sample `k` is accepted if **≥ 2 of 3 detectors agree** within ±50 ms.

**Polarity detection**: before running the detectors, the signal polarity is checked.
If the dominant QRS deflection is negative (as in lead aVR), the signal is internally
flipped for detection, then all peak indices are mapped back to the original.

### Why Ensemble (vs Single Detector)

Single Pan-Tompkins failure modes:
- Atrial flutter waves can fool the thresholding into counting flutter peaks as QRS
- Very wide QRS (BBB, >160 ms) spreads the energy over time — moving average misses the peak
- Low SNR or low amplitude → threshold set too high → beats missed

Each detector has orthogonal failure modes. Requiring 2/3 agreement:
- Eliminates isolated false positives (Hilbert picks up T-waves: Wavelet and P-T don't)
- Reduces missed beats (if Pan-Tompkins fails on wide QRS, Hilbert still fires)

### Pros / Cons

| | Pros | Cons |
|-|------|------|
| 3-detector vote | ~4–8% fewer false peaks vs single detector on AF/BBB | 3× compute vs single detector (~15 ms overhead per 10s window) |
| Polarity detection | Handles aVR and other inverted leads without config | Isoelectric QRS (rare) classified as positive by default |
| Post-vote RR validation | Removes physiologically impossible peaks (<200 ms or >3000 ms RR) | Aggressive RR gate can miss genuine extreme bradycardia (HR<20) |

---

## 4. Stage 3 — Waveform Delineation (P/Q/R/S/T)

### What It Does
For each detected R-peak, find the precise location of all 5 waveform components:

| Component | Method | Output |
|-----------|--------|--------|
| R-peak | Ensemble vote → refined to exact max/min in ±40 ms window | `r_peak`, `r_amplitude`, `qrs_polarity` |
| Q-wave | argmin in [QRS onset → R] window | `q_peak`, `q_depth` |
| S-wave | argmin in [R → QRS offset] window | `s_peak`, `s_depth` |
| P-wave | CWT ridge tracking in [−300 ms → −50 ms] pre-R window | `p_onset`, `p_peak`, `p_offset`, `p_amplitude`, `p_morphology` |
| T-wave | CWT ridge tracking in [+50 ms → +400 ms] post-R window | `t_onset`, `t_peak`, `t_offset`, `t_amplitude`, `t_inverted` |

P-wave morphology is classified per beat:
- **Normal**: positive P with amplitude > 0.05 mV
- **Inverted**: P amplitude < −0.05 mV (common in junctional, retrograde conduction)
- **Biphasic**: P waveform crosses zero (common in left atrial enlargement, AVNRT)
- **Absent**: no P-wave detected (AF, junctional escape, complete AV block)

### Why CWT (Continuous Wavelet Transform)

The alternative (NeuroKit2's DWT-based delineation) fails on ~40% of segments in our
clinical data because:
- It expects a template of a "normal" beat to work well
- Arrhythmias (especially AF, with chaotic baseline) confuse the DWT boundaries
- No handling of inverted leads — classifies aVR R-peaks as T-waves

CWT uses a Mexican Hat wavelet to find local ridge maxima at the scale corresponding to
each waveform's expected duration. This is scale-selective:
- QRS scale: ~10–50 ms
- T-wave scale: ~100–250 ms
- P-wave scale: ~60–120 ms

Scale-selectivity means AF fibrillation waves (which have very short cycle length ~180 ms)
do not get confused with T-waves (cycle length 250–400 ms).

### Template Matching Hybrid

For long recordings (>30 beats), a representative beat template is extracted (median of
well-isolated normal beats). Template correlation provides a second estimate of delineation
that handles noisy beats. The hybrid picks CWT or template result per beat based on a
local SNR estimate.

### Delta Wave Detection (WPW)
The early QRS upstroke slope ratio is computed:
- `early_slope` = average slope of first 20% of QRS duration
- `late_slope` = average slope of middle 60% of QRS duration
- `delta_wave = True` if `early_slope / late_slope < 0.3` (slurred upstroke characteristic of pre-excitation)

### Pros / Cons

| | Pros | Cons |
|-|------|------|
| CWT delineation | Works on AF, BBB, irregular rhythms | More complex than DWT; ~20 ms overhead |
| All 5 waveforms | Enables precise PVC/PAC/WPW discrimination | Q/S detection unreliable in very noisy or very short QRS (<60ms) |
| Inverted-lead handling | No lead-selection needed from caller | Auto-detection can misclassify isoelectric QRS (~2% of cases) |
| P-wave morphology | Directly encodes retrograde/ectopic atrial activation | Biphasic classification threshold (zero-crossing) is heuristic |

---

## 5. Stage 4 — Feature Extraction (60 Features)

### Why 60 Features (vs 15 in V2)

V2 had 15 features: 5 HRV time-domain + 5 morphology + 5 basic interval measures.
These 15 numbers are insufficient to distinguish many clinically similar rhythms:

| Confusion pair | V2 can distinguish? | V3 fix |
|----------------|---------------------|--------|
| AF vs Junctional brady | Barely (only SDNN) | HF power collapses in AF; P-absent fraction = 1.0 in both; LF/HF ratio separates |
| PVC vs PAC | No — no per-beat typing | `pvc_score_mean` vs `pac_score_mean` from 7-criteria scoring |
| VT vs Fast AF | No — similar HR, similar SDNN | QRS width fraction, compensatory pause, negative polarity |
| WPW pre-excitation | No | `delta_wave_fraction` |
| T-wave inversion (ischemia) | No | `t_inverted_fraction` |

### Feature Domains

```
Domain               Count  Key Features
─────────────────────────────────────────────────────────────────
HRV Time Domain        11   SDNN, RMSSD, pNN50, RR skewness/kurtosis, triangular index
HRV Frequency           8   VLF/LF/HF power, LF/HF ratio, spectral entropy
Nonlinear HRV           8   Sample entropy, Hurst exponent, DFA α1, SD1/SD2 (Poincaré)
Morphology             13   QRS duration/area, PR interval, QTc Bazett, ST elevation/slope,
                            T-wave asymmetry, R/S ratio, P-wave duration
Beat Discriminators    20   QRS width fraction, P-absent fraction, compensatory pause fraction,
                            T-discordant fraction, delta wave fraction, pvc_score_mean, pac_score_mean
─────────────────────────────────────────────────────────────────
Total                  60
```

### Why These Specific Features

**HRV time domain**: Captures beat-to-beat regularity.
- SDNN is the global spread of RR intervals — high in AF (chaotic), low in sinus
- RMSSD emphasizes short-term HRV — parasympathetic tone marker
- RR skewness/kurtosis: normal sinus is near-symmetric; AF is right-skewed

**HRV frequency**: Decomposes RR interval variability by frequency band.
- HF band (0.15–0.40 Hz) = respiratory sinus arrhythmia
- LF/HF ratio separates sympathetic vs parasympathetic dominance
- LF power collapses in AF (no organised atrial activity → no LF modulation)

**Nonlinear**: Captures complexity that linear HRV misses.
- Sample entropy: high in AF (maximum irregularity), low in VT (very regular)
- DFA α1: self-similarity exponent — AF has α1 ~0.5 (white noise), normal sinus ~0.8–1.1
- SD1/SD2 from Poincaré plot: SD1 = short-term variability, SD2 = long-term

**Morphology**: Interval and amplitude measurements averaged across all beats.
- QTc Bazett: corrected QT interval — prolonged in BBB, drug toxicity, hypokalaemia
- ST elevation: direct ischemia/pericarditis marker
- T-wave asymmetry: asymmetric T common in ischemia, symmetric in normal

**Beat discriminators**: The most novel addition — per-beat electrophysiology scores.
- `qrs_wide_fraction`: fraction of beats with QRS > 120 ms (high in BBB, PVC)
- `compensatory_pause_fraction`: fraction of beats followed by compensatory pause
- `pvc_score_mean`: aggregate evidence for ventricular origin per beat
- `pac_score_mean`: aggregate evidence for atrial ectopic origin per beat

### Pros / Cons of Feature-Augmented Model

| | Pros | Cons |
|-|------|------|
| 60 explicit features | Clinical interpretability; SHAP explanations meaningful | Feature extraction takes ~30–50 ms; adds latency |
| Separate from CNN | Features handle rare classes where signal alone is ambiguous | Feature quality depends on R-peak accuracy — bad R-peaks = bad features |
| Beat discriminators | Direct PVC/PAC separation without explicit per-beat labeling needed at rhythm model level | Requires reliable delineation (P/T waves) — less reliable in <3 beats or high noise |

---

## 6. Stage 5 — Signal Quality Index (SQI)

### 10 Quality Criteria

| # | Criterion | Threshold | Rationale |
|---|-----------|-----------|-----------|
| 1 | No NaN/Inf | Any → fail | Corrupt data check |
| 2 | Minimum length | < 2s → fail | Can't compute HRV on < 3 beats |
| 3 | Not flat-line | Variance < 1e-8 → fail | Disconnected lead |
| 4 | No saturation | Clipping > 2% → deduct | ADC saturation distorts QRS peaks |
| 5 | SNR | < 10 dB → deduct | Excess noise masks P/T waves |
| 6 | HF noise | > 40% total power in 40+ Hz → deduct | Muscle artifact |
| 7 | Baseline wander | > 30% power below 0.5 Hz → deduct | Heavy baseline drift |
| 8 | RR regularity | RR CV > 0.35 → flag (not deduct) | Extreme irregularity (AF) should pass |
| 9 | Sufficient beats | < 3 in 10s window → deduct | Can't compute HRV |
| 10 | Clipping ratio | > 5% → deduct heavily | Extreme saturation |

SQI = (criteria_passed / total_applicable) × 1.0

Segments with SQI < 0.3 are marked `skipped=True` and not sent to the model.
This prevents garbage features from influencing model predictions.

---

## 7. V2 vs V3 Summary

| Dimension | V2 | V3 | Improvement |
|-----------|----|----|-------------|
| Features | 15 | 60 | 4× more information per segment |
| Waveforms detected | R only | P, Q, R, S, T | Full electrophysiology capture |
| R-peak detectors | 1 | 3 (ensemble) | Fewer missed beats on AF/BBB |
| Delineation success rate | ~60% (NK2 fails on AF) | ~95% (CWT handles AF) | Critical for morphology features |
| Inverted lead handling | No | Yes | aVR/augmented leads work correctly |
| Artifact detection | No | Yes (wavelet) | Prevents muscle artifact from entering model |
| Quality gating | Single threshold | 10 criteria | Targeted issue reporting |
| PVC/PAC discrimination | None | 7-criteria score each | Direct clinical basis for beat typing |
