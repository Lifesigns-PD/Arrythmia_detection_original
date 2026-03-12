# ECG Signal Processing Pipeline — Technical Reference

This document describes every stage of signal processing performed by this system, from raw ADC samples to final rhythm/ectopy classification. Each stage is explained at the signal-processing level with its biological justification.

---

## 1. Signal Acquisition & Resampling

### What happens
Raw ECG waveforms are loaded from JSON database records. Different source datasets have different native sampling rates:

| Source | Native Fs | Target Fs | Method |
|--------|-----------|-----------|--------|
| PTB-XL | 500 Hz | 125 Hz | `scipy.signal.resample` (polyphase) |
| MIT-BIH | 360 Hz | 125 Hz | same |
| AFDB | 250 Hz | 125 Hz | same |

**Target: 125 Hz, 10-second segments = 1 250 samples** (`signal_processing/config.yaml`)

### Why 125 Hz?
The Nyquist criterion requires `Fs ≥ 2 × f_max`. The diagnostically important ECG content lies:
- QRS complex: 5–40 Hz (main energy <30 Hz)
- P and T waves: 0.5–10 Hz
- HF noise / artifact: >40 Hz

125 Hz > 2 × 40 Hz = 80 Hz → Nyquist satisfied with comfortable margin. Higher rates (500 Hz) offer negligible clinical benefit for rhythm analysis while quadrupling memory and compute cost.

### Resampling artefact guard
`scipy.signal.resample` uses an FFT-based method that is exact for band-limited signals. A mild anti-alias LP filter at 40 Hz (see §3) is applied after resampling to remove any spectral aliasing introduced during downsampling.

---

## 2. Baseline Wander Removal

**Code:** `signal_processing/cleaning.py → remove_baseline_wander()`
**Config:** `baseline.method = butterworth_highpass`, `cutoff = 0.5 Hz`, `order = 3`

### What is baseline wander?
A slow sinusoidal drift of the ECG baseline, typically 0.05–0.5 Hz. Caused by:
- **Respiration** — chest wall movement shifts electrode position (~0.15–0.4 Hz)
- **Patient movement** — body sway, postural shifts
- **Electrode impedance changes** — skin–electrode contact fluctuations

Visually it looks like the entire waveform riding up and down on a slow wave. Clinically it can falsely elevate or depress the ST segment and hide P waves.

### The filter: 3rd-order Butterworth high-pass at 0.5 Hz
```python
b, a = scipy.signal.butter(order=3, Wn=0.5/(fs/2), btype='high')
cleaned = scipy.signal.filtfilt(b, a, signal)
```

- **Butterworth**: maximally flat passband — no ripple that could distort QRS morphology
- **3rd order**: adequate roll-off (-60 dB/decade above cutoff) without excessive phase distortion
- **`filtfilt`**: zero-phase filtering (applies filter forwards then backwards) — eliminates group delay, preserving exact timing of wave peaks
- **0.5 Hz cutoff**: safely above respiratory band, safely below the P wave's lowest frequency (~0.5 Hz lower edge), so the P wave is preserved

> **Biological note**: The lowest frequency component of the P wave is approximately 0.5 Hz. Setting the cutoff exactly at 0.5 Hz means the very lowest edge of the P wave energy is attenuated by −3 dB (Butterworth −3 dB point). This is an accepted clinical trade-off because ST-segment accuracy (requires removing drift) is more diagnostically important than absolute P-wave amplitude fidelity.

---

## 3. Powerline Noise Removal (Notch Filter)

**Code:** `signal_processing/cleaning.py → remove_powerline_noise()`
**Config:** `powerline.frequencies = [50, 60]`, `quality_factor = 30`

### What is powerline noise?
Mains electricity induces a narrowband sinusoidal interference at exactly 50 Hz (Europe/Asia) or 60 Hz (Americas) into ECG electrodes via capacitive/inductive coupling. It appears as a uniform high-frequency buzz overlaid on the ECG trace.

### The filter: IIR notch (band-stop) at 50 Hz and 60 Hz
```python
for freq in [50, 60]:
    b, a = scipy.signal.iirnotch(w0=freq, Q=30, fs=fs)
    signal = scipy.signal.filtfilt(b, a, signal)
```

- **`iirnotch`**: designs a 2nd-order IIR band-rejection filter centred at `w0`
- **Quality factor Q = 30**: bandwidth = f₀/Q = 50/30 = 1.67 Hz. Very narrow rejection band, so frequencies either side of 50 Hz are unaffected (no distortion of nearby HF QRS energy)
- **`filtfilt`**: zero-phase again, no peak timing distortion
- Both 50 and 60 Hz are filtered so the same codebase handles all geographies

---

## 4. Anti-Alias / Noise Reduction (Low-Pass Filter)

**Code:** `utils/ecgprocessor.py → _preprocess()` — applies LP at 40 Hz after the notch filters

### Purpose
Removes high-frequency noise above the ECG signal band:
- Electromyographic (EMG) artifact from muscle tremor: 20–500 Hz
- Electronic noise from ADC quantisation: white noise across all frequencies
- Any residual content above 40 Hz that survives resampling

### Filter design
A Butterworth low-pass at 40 Hz. At 125 Hz this is `Wn = 40/62.5 = 0.64` (normalised).

> **Biological note**: The QRS complex's primary energy is below 30 Hz; the Einthoven criterion for ECG diagnostic quality requires the system to pass 0.05–150 Hz, but for rhythm analysis (not fine morphology) cutting off at 40 Hz has no clinical consequence and substantially improves SNR.

---

## 5. Artifact & Signal Quality Detection

**Code:** `signal_processing/artifact_detection.py → check_signal_quality()`
**Code:** `signal_processing/sqi.py → calculate_sqi_score()`

### Checks performed

#### 5.1 Flatline (lead-off / disconnection)
```python
if np.std(segment) < 0.001:   # threshold from config.yaml
    issues.append("flatline")
```
Standard deviation of a normal ECG is 0.1–1.0 mV. A flat line (std < 0.001 mV) indicates the electrode has been removed, the lead wire is broken, or the patient's heart has stopped. The segment is rejected from training.

#### 5.2 Sudden Amplitude Jumps (motion artifact / EMG burst)
```python
derivative = np.abs(np.diff(segment))
if np.any(derivative > 4.0):   # mV/sample
    issues.append("amplitude_jump")
```
Between consecutive 125 Hz samples (8 ms), a real ECG cannot change by more than ~4 mV (even the steepest QRS upstroke is ~2 mV in 8 ms). Jumps above 4 mV are electrode pop, cable movement, or defibrillation artifact.

#### 5.3 Saturation / Clipping
```python
frac_at_min = np.mean(segment == segment.min())
frac_at_max = np.mean(segment == segment.max())
if frac_at_min > 0.05 or frac_at_max > 0.05:
    issues.append("saturation")
```
If more than 5% of samples are at the ADC rail (min or max), the amplifier is saturating. Clipped waveforms cannot be classified reliably — QRS peaks are truncated.

#### 5.4 Signal Quality Index (SQI, 0–100)
`sqi.py` computes a composite score:

| Component | Rationale |
|-----------|-----------|
| Amplitude range 0.05–10 mV | Normal ECG amplitude window |
| Kurtosis > 5 | ECG has sharp QRS spikes → high kurtosis; noise is Gaussian → kurtosis ≈ 3 |
| Power at mains (50/60 Hz) | Residual notch leakage indicates poor contact |
| Spectral tilt | Baseline drift indicator (excess low-frequency power) |

Segments with `is_acceptable = False` (any hard failure) or SQI < threshold are excluded from model training but are still stored and displayed.

---

## 6. R-Peak Detection (QRS Detection)

**Code:** `utils/ecgprocessor.py → _r_peak_detection()`

The system uses a Pan-Tompkins-inspired algorithm — the gold standard for real-time QRS detection, published by Pan & Tompkins (1985) and validated extensively at 100–200 Hz sampling rates.

### Step-by-step

#### Step 1 — Differentiation
```python
diff_signal = np.diff(cleaned_signal)
```
Computes the discrete first derivative: `d[n] = x[n] - x[n-1]`. This emphasises rapid voltage changes (QRS slopes) and suppresses slowly varying signals (P/T waves). A normal QRS rises ~1.5 mV in ~40 ms = 37.5 mV/s; T waves rise ~0.5 mV in ~100 ms = 5 mV/s. The derivative amplifies QRS by 7× relative to T.

#### Step 2 — Squaring
```python
squared = diff_signal ** 2
```
All values become positive (rectification). Squaring further amplifies large deflections non-linearly: a QRS derivative of 5 → 25; a T-wave derivative of 1 → 1. This provides ~25:1 QRS-to-T discrimination versus the original ~7:1.

#### Step 3 — Moving-Window Integration
```python
window = int(0.15 * fs)   # 150 ms window
integrated = np.convolve(squared, np.ones(window)/window, mode='same')
```
Smooths the squared signal over a 150 ms window (≈ QRS duration + margin). The squared signal has multiple spikes per QRS (each derivative peak gives one); integration merges them into a single smooth bump per beat. This is critical for correct beat counting.

#### Step 4 — Peak Finding
```python
peaks, _ = scipy.signal.find_peaks(
    integrated,
    height=np.mean(integrated) * 0.5,   # adaptive threshold
    distance=int(0.3 * fs)              # 300 ms minimum RR interval (max 200 bpm)
)
```
- **Adaptive height threshold**: 50% of mean integrated amplitude adapts to gain changes and signal amplitude variation
- **Minimum distance 300 ms**: Physiologically, the minimum human RR interval is ~250 ms (240 bpm). Using 300 ms prevents double-detection while allowing tachycardia detection up to ~200 bpm

Output: array of sample indices, each pointing to the centre of the R peak.

> **Biological note**: R-peak detection at 125 Hz has a worst-case timing error of ±4 ms (half a sample). This is well within the ±10 ms needed for accurate HRV computation (which requires ~1 ms accuracy for research-grade HRV but 10 ms is sufficient for clinical rhythm analysis).

---

## 7. ECG Wave Anatomy and Clinical Context

The system uses R-peak timing as the reference point for all other feature extraction. Here is what each wave represents and how this system uses it:

### P Wave
- **Biology**: Atrial depolarisation (right then left atrium contract)
- **Timing**: Begins ~120 ms before the R peak; duration 80–120 ms; amplitude 0.1–0.25 mV
- **Clinical use**: Absence of P waves → AFib; saw-tooth P waves → flutter; shortened PR → pre-excitation (WPW)
- **This system**: P wave is not explicitly delineated; the rhythm model learns AFib patterns from the absence of organised atrial activity in the signal texture

### QRS Complex
- **Biology**: Ventricular depolarisation (both ventricles contract)
- **Timing**: Duration 60–100 ms (normal); 100–120 ms (BBB); >120 ms (wide QRS = aberrant)
- **Q wave**: First negative deflection before R; pathological Q (>0.04s, >25% of R) = old MI
- **R wave**: Main upward deflection; this is what R-peak detection finds
- **S wave**: Negative deflection after R; deep S in V1 with tall R in V5/V6 = LVH
- **This system**: QRS energy is extracted in a ±100 ms (±12.5 samples) window around each R peak (`_calculate_morphology_features()`). This captures Q, R, S and the QRS onset/offset.

### T Wave
- **Biology**: Ventricular repolarisation (ventricles relax)
- **Timing**: 160–440 ms after R peak; rounded, usually upright in most leads
- **Clinical use**: T-wave inversion → ischaemia; peaked T → hyperkalaemia; long QT → torsades risk
- **This system**: T wave is within the 10-second segment the model analyses; CNN attention layers implicitly learn T-wave features without explicit delineation

### Key Intervals

| Interval | Normal | Measured From/To | Clinical Meaning |
|----------|--------|------------------|-----------------|
| PR | 120–200 ms | P onset → R onset | AV conduction time; long PR = 1st degree block |
| QRS | 60–100 ms | QRS onset → QRS offset | Ventricular conduction; wide = BBB or VT |
| QT | 350–450 ms | QRS onset → T offset | Total ventricular electrical activity; long QT = arrhythmia risk |
| RR | 600–1000 ms | R → next R | Heart rate; irregularity = AFib, ectopy |

---

## 8. Heart Rate Variability (HRV) Analysis

**Code:** `utils/ecgprocessor.py → _calculate_frequency_hrv()`, `_calculate_nonlinear_hrv()`

HRV measures the variation in time between consecutive heartbeats (RR intervals). It is a non-invasive window into autonomic nervous system function.

### 8.1 RR Interval Extraction
```python
rr_intervals_ms = np.diff(r_peaks) / fs * 1000  # convert samples → milliseconds
```

### 8.2 Time-Domain HRV

| Feature | Formula | Meaning |
|---------|---------|---------|
| Mean RR | mean(RR) | Average heart rate |
| SDNN | std(RR) | Overall HRV; low SDNN = autonomic dysfunction |
| RMSSD | √(mean(ΔRR²)) | Short-term HRV; reflects parasympathetic (vagal) tone |
| pNN50 | % of consecutive pairs where \|ΔRR\| > 50 ms | Vagal activity marker |

### 8.3 Frequency-Domain HRV (Welch's Method)

To compute the power spectral density of RR intervals, the irregularly-spaced RR series must first be interpolated to a uniform time axis at 4 Hz, then Welch's method (overlapping windowed FFT) is applied.

```python
t_uniform = np.arange(0, total_time, 1/4)
rr_interp = np.interp(t_uniform, rr_times, rr_intervals)
freqs, psd = scipy.signal.welch(rr_interp, fs=4, nperseg=256)
```

| Band | Frequency Range | Physiological Meaning |
|------|----------------|----------------------|
| VLF (Very Low Freq) | 0.003–0.04 Hz | Thermoregulation, renin-angiotensin system, slow sympathetic |
| LF (Low Freq) | 0.04–0.15 Hz | Mix of sympathetic and parasympathetic; baroreceptor reflex |
| HF (High Freq) | 0.15–0.40 Hz | Purely parasympathetic (vagal); respiratory sinus arrhythmia |
| LF/HF ratio | — | Autonomic balance; elevated in stress, reduced in parasympathetic dominance |

> **Biological note**: 10-second segments are very short for frequency-domain HRV. The LF band requires at least 2 minutes for stable estimates (Task Force 1996). The system stores these features but they should be interpreted cautiously on 10s windows — they are more useful as inputs to the ML model than as standalone clinical readings.

### 8.4 Non-Linear HRV

#### Poincaré Plot (SD1 / SD2)
Each RR interval is plotted against the next: `(RR[n], RR[n+1])`. The cloud of points forms an ellipse:
- **SD1** = width perpendicular to the identity line = short-term (beat-to-beat) variability = parasympathetic index
- **SD2** = width along the identity line = long-term variability = overall autonomic modulation

```python
rr1 = rr_intervals[:-1]
rr2 = rr_intervals[1:]
sd1 = np.std((rr2 - rr1) / np.sqrt(2))   # = RMSSD / √2
sd2 = np.std((rr2 + rr1) / np.sqrt(2))
```

#### Sample Entropy (SampEn)
Measures the regularity/complexity of the RR series. A perfectly regular signal (pacemaker rhythm) has SampEn ≈ 0. A healthy heart has moderate complexity. Reduced complexity indicates disease or autonomic impairment.

---

## 9. Rhythm Classification

**Code:** `decision_engine/rhythm_orchestrator.py`, `models_training/models.py`

### Model Architecture
The classifier is a **CNN-Transformer hybrid**:
1. **1D CNN layers**: Extract local morphological features from the raw 1 250-sample signal — QRS shape, amplitude, duration
2. **Transformer encoder**: Capture long-range temporal patterns — RR interval sequences, beat-to-beat variability, rhythm regularity

Input: normalised 1 250-sample signal (10 s at 125 Hz)
Output: probability distribution over rhythm/ectopy classes

### Two Parallel Tasks

| Task | Classes (count) | What it detects |
|------|----------------|-----------------|
| RHYTHM | 17 | SinusRhythm, AFib, AFlutter, VTach, VFib, SVT, BradycardiaAbs, etc. |
| ECTOPY | 4 | None, PVC, PAC, Run (≥3 consecutive PVCs) |

Tasks are kept separate because they require different label semantics (a segment can be "SinusRhythm with PVCs" — both labels valid simultaneously).

### Rules Engine (Post-Model)
After ML inference, `rhythm_orchestrator.py` applies deterministic rules:
- **NSVT detection**: ≥7 consecutive PVC beats within a 10 s window → NSVT event generated
- **Event suppression**: A NSVT event suppresses individual PVC events in the same time span
- **Confidence thresholding**: Events below the configured confidence threshold are flagged as low-confidence

---

## 10. Biological Correctness Assessment

| Stage | Standard | This System | Assessment |
|-------|----------|-------------|-----------|
| Sampling rate | ≥200 Hz for morphology; ≥100 Hz for rhythm | 125 Hz | ✅ Adequate for rhythm; borderline for fine morphology |
| Baseline filter | HP ≥ 0.05 Hz (AHA); ≤ 0.67 Hz | HP 0.5 Hz | ✅ Within clinical standard |
| Notch filter | At mains frequency | 50 + 60 Hz, Q=30 | ✅ Narrow notch, minimal distortion |
| QRS detection | Pan-Tompkins or equivalent | Pan-Tompkins derivative-square-integrate | ✅ Gold standard algorithm |
| QRS window for features | ±50–100 ms around R | ±100 ms | ✅ Captures full QRS |
| HRV frequency bands | Task Force (1996) standard | VLF/LF/HF as per standard | ✅ Correct band definitions |
| Minimum segment for HRV | 5 min (reliable); 1 min (acceptable) | 10 s | ⚠️ Too short for clinical HRV; acceptable as ML features |
| R-peak timing error | < ±10 ms for rhythm | ±4 ms (half-sample at 125 Hz) | ✅ Within tolerance |
| Artifact rejection | Lead-off, saturation, EMG | Flatline + jumps + clipping | ✅ Covers main failure modes |

### Known Limitations
1. **Single-lead analysis**: This system processes one ECG channel at a time. Multi-lead analysis (12-lead ECG) would improve specificity, especially for localising ischaemia and differentiating VT from SVT with aberrancy.
2. **No explicit P/T wave delineation**: The system relies on the CNN to implicitly learn P/T wave features. Explicit delineation (e.g., wavelet-based) would improve AF detection sensitivity and QT measurement.
3. **10s HRV windows**: Frequency-domain HRV is unreliable at 10 seconds. LF/HF ratios on 10s windows should be treated as supplementary ML features, not standalone clinical metrics.
4. **Fixed-threshold Pan-Tompkins**: The adaptive threshold uses 50% of mean integrated amplitude. In severe bradycardia or very-low-amplitude ECGs, missed beats can occur. A fully adaptive two-threshold approach (refractory period + dynamic threshold) would improve sensitivity.

---

## 11. Data Flow Summary

```
Raw ECG JSON
    │
    ▼ Resample to 125 Hz (scipy.signal.resample)
    │
    ▼ Butterworth HP 0.5 Hz → remove baseline wander
    │
    ▼ Notch 50/60 Hz Q=30 → remove powerline noise
    │
    ▼ Butterworth LP 40 Hz → remove EMG / high-freq noise
    │
    ▼ Artifact check: flatline / amplitude jumps / clipping
    │  └─ rejected? → store with is_acceptable=False, skip training
    │
    ▼ Pan-Tompkins R-peak detection
    │  └─ RR intervals → HRV features (SDNN, RMSSD, LF/HF, SD1/SD2)
    │  └─ QRS energy windows → morphology features
    │
    ▼ CNN-Transformer inference (1250 samples)
    │  ├─ RHYTHM task → 17-class probabilities
    │  └─ ECTOPY task → 4-class probabilities (per beat context)
    │
    ▼ Rules engine
    │  └─ NSVT from ≥7 consecutive PVCs
    │  └─ Event suppression / confidence filtering
    │
    ▼ Events stored to DB (events_json JSONB)
    │
    ▼ Dashboard display + cardiologist annotation loop
    │
    ▼ Finetune: python models_training/retrain.py --mode finetune
```

---

*Document generated from codebase analysis. Source files: `utils/ecgprocessor.py`, `signal_processing/cleaning.py`, `signal_processing/artifact_detection.py`, `signal_processing/sqi.py`, `signal_processing/config.yaml`, `models_training/data_loader.py`, `decision_engine/rhythm_orchestrator.py`.*
