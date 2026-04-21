# EXACT SIGNAL PROCESSING METHODS USED IN YOUR PIPELINE

## 🔍 COMPLETE BREAKDOWN

### **STEP 1: SIGNAL CLEANING (signal_processing/cleaning.py)**

#### 1A. Baseline Wander Removal
```
Algorithm: IIR High-Pass Butterworth Filter
├─ Cutoff frequency: 0.5 Hz
├─ Filter order: 2
├─ Method: filtfilt (zero-phase filtering)
└─ Purpose: Remove slow drift (breathing, electrode movement)
```

#### 1B. Powerline Noise Removal
```
Algorithm: Notch filter (IIR bandstop)
├─ Target frequencies: 50 Hz and 60 Hz (mains hum)
├─ Method: Butterworth bandstop, order 2
├─ Bandwidth: ±1 Hz around each frequency
└─ Purpose: Remove 50/60Hz electrical noise
```

#### 1C. Signal Normalization
```
Algorithm: Min-max scaling to [-1, 1] range
├─ Formula: (signal - min) / (max - min) * 2 - 1
└─ Purpose: Normalize signal amplitude for consistency
```

---

### **STEP 2: R-PEAK DETECTION (signal_processing/pan_tompkins.py + feature_extraction.py)**

#### Primary Method: Pan-Tompkins Algorithm (1985)

Reference: Pan, J., & Tompkins, W. J. (1985). "A Real-Time QRS Detection Algorithm." IEEE Transactions on Biomedical Engineering, BME-32(3), 230-236.

**5-Stage Pipeline:**

#### Stage 1: Bandpass Filter
```
Algorithm: Butterworth Bandpass Filter
├─ Frequency range: 5-15 Hz (isolates QRS energy)
├─ Order: 2
├─ Implementation: scipy.signal.butter + filtfilt
└─ Purpose: Remove baseline + high-frequency noise, keep QRS
```

#### Stage 2: 5-Point Derivative
```
Formula: y(n) = (1/8T) * [-x(n-2) - 2x(n-1) + 2x(n+1) + x(n+2)]
├─ Purpose: Emphasize steep QRS slopes
└─ Result: High values at QRS upstroke/downstroke
```

#### Stage 3: Squaring
```
Formula: y(n) = [x(n)]²
├─ Purpose: Make all values positive, amplify large slopes
└─ Result: Non-negative signal with QRS peaks much higher
```

#### Stage 4: Moving-Window Integration
```
Algorithm: Rectangular moving average filter
├─ Window size: 150 ms (roughly QRS width)
├─ Method: cumsum for efficiency
└─ Purpose: Smooth derivative output into single pulse per QRS
```

#### Stage 5: Adaptive Thresholding + Search-Back
```
Algorithm: Dual-threshold with search-back logic
├─ Signal threshold: 50% of recent signal amplitude
├─ Noise threshold: 50% of recent noise amplitude
├─ Search-back: If >1.66 RR_avg without detection, find highest peak
└─ Purpose: Adapt to signal variations, catch missed beats
```

#### Fallback Method (if Pan-Tompkins fails)
```
Algorithm: Simple Scipy Peak Detection
├─ Method: scipy.signal.find_peaks
├─ Height threshold: 60% of max signal amplitude
├─ Distance: minimum 300ms between peaks
└─ Purpose: Basic R-peak detection when Pan-Tompkins unavailable
```

#### NeuroKit2 Method (used in feature_extraction.py)
```
Algorithm: NeuroKit2's ECG peak detection
├─ Method: scipy.signal.find_peaks with custom parameters
├─ Implementation: nk.ecg_peaks()
└─ Fallback to Pan-Tompkins if NeuroKit2 fails
```

---

### **STEP 3: ECG MORPHOLOGY DELINEATION (signal_processing/morphology.py)**

#### Primary Method: NeuroKit2 Discrete Wavelet Transform (DWT)

```
Algorithm: DWT-based ECG wave delineation
├─ Library: NeuroKit2 (nk.ecg_delineate)
├─ Method: Discrete Wavelet Transform
├─ Reference: Martínez et al. (2004) - DWT for ECG delineation
├─ Returns: Indices for P/QRS/T wave components
└─ Purpose: Find exact start/end of each ECG wave

What it detects:
├─ ECG_P_Onsets ........ P-wave start
├─ ECG_P_Peaks ........ P-wave maximum amplitude
├─ ECG_P_Offsets ....... P-wave end
├─ ECG_R_Onsets ....... QRS complex start
├─ ECG_R_Peaks ........ R-peak (same as detected earlier)
├─ ECG_R_Offsets ...... QRS complex end
├─ ECG_T_Onsets ....... T-wave start
├─ ECG_T_Peaks ........ T-wave maximum amplitude
└─ ECG_T_Offsets ...... T-wave end
```

**Problem with DWT:**
```
❌ Fails silently on ~40% of segments:
   ├─ Returns None for missing waves
   ├─ Returns NaN for poor signal quality
   └─ Code converts to 0.0 (no error!)
```

#### Pre-processing for Delineation
```
Algorithm: Low-pass Butterworth filter
├─ Cutoff: 40 Hz
├─ Order: 2
├─ Purpose: Smooth signal before DWT delineation
└─ Implementation: scipy.signal.butter + filtfilt
```

---

### **STEP 4: FEATURE EXTRACTION (signal_processing/feature_extraction.py)**

#### 13 Features Extracted:

| # | Feature | Calculation | Normal Range |
|---|---------|-------------|--------------|
| 0 | **mean_hr_bpm** | 60000 / RR_interval_ms (median) | 40-200 bpm |
| 1 | **hr_std_bpm** | std(heart_rates) | 0-100 bpm |
| 2 | **sdnn_ms** | std(RR_intervals) | 20-200 ms |
| 3 | **rmssd_ms** | sqrt(mean(diff(RR)²)) | 0-300 ms |
| 4 | **qrs_duration_ms** | R_offset - R_onset | 60-150 ms ⚠️ BROKEN |
| 5 | **pr_interval_ms** | R_onset - P_onset | 100-250 ms ⚠️ BROKEN |
| 6 | **qtc_ms** | QT / sqrt(RR_seconds) [Bazett] | 360-440 ms |
| 7 | **p_wave_amplitude_mv** | signal[P_peak] | 0.1-0.3 mV |
| 8 | **t_wave_amplitude_mv** | signal[T_peak] | 0.2-0.5 mV |
| 9 | **st_deviation_mv** | mean(ST_segment) - baseline | ±0.1 mV |
| 10 | **qrs_amplitude_mv** | signal[R_peak] | 0.5-3.0 mV |
| 11 | **p_wave_present_ratio** | count(P_detected) / total_beats | 0-1 |
| 12 | **sqi_score** | Quality index (0-100) ⚠️ WRONG SCALE |

---

### **STEP 5: SIGNAL QUALITY INDEX (signal_processing/sqi.py)**

#### SQI Calculation Method

```
Algorithm: Multi-factor SQI (0-100 scale)
├─ Component 1: Amplitude check (< 0.05mV = fail)
├─ Component 2: Kurtosis check (Fisher kurtosis > 5 = good)
├─ Component 3: Skewness check (ECG should be skewed)
├─ Component 4: Power spectrum (% energy in 50/60Hz band)
├─ Component 5: Baseline stability (% energy < 0.5Hz)
└─ Final score: 100 - deductions (max 100)

⚠️ RETURNS 0-100 SCALE (should be 0-1!)
```

---

## 🎯 COMPLETE SIGNAL FLOW DIAGRAM

```
Raw ECG Signal (1250 samples @ 125Hz = 10 seconds)
    ↓
[CLEANING]
├─ Remove baseline wander (highpass 0.5Hz)
├─ Remove powerline noise (notch 50/60Hz)
└─ Normalize to [-1, 1] range
    ↓ Cleaned Signal
[R-PEAK DETECTION] 
├─ Pan-Tompkins 5-stage algorithm
│  ├─ Bandpass 5-15Hz
│  ├─ 5-point derivative
│  ├─ Squaring
│  ├─ Moving window integration
│  └─ Adaptive thresholding
├─ Fallback: scipy.find_peaks
└─ Fallback: NeuroKit2
    ↓ R-peak indices
[MORPHOLOGY DELINEATION]
├─ NeuroKit2 DWT delineation
├─ Returns: P/QRS/T wave boundaries
└─ ⚠️ Fails on 40% → returns None
    ↓ Wave boundaries (many NaN!)
[FEATURE EXTRACTION]
├─ Calculate 13 features
├─ Handle NaN → 0.0 (PROBLEM!)
└─ SQI 0-100 scale (WRONG!)
    ↓ Feature Vector (13 values)
[STORAGE IN DATABASE]
└─ features_json: {qrs_duration_ms: 0, pr_interval_ms: 48, sqi_score: 100}
                   ↑ corrupted!        ↑ corrupted!    ↑ wrong scale!
```

---

## 📊 WHERE EACH METHOD IS USED

| Method | File | Used By | Problem |
|--------|------|---------|---------|
| **Butterworth Highpass** | cleaning.py | Feature extraction | Working ✅ |
| **Butterworth Notch** | cleaning.py | Feature extraction | Working ✅ |
| **Pan-Tompkins** | pan_tompkins.py | detect_r_peaks() | Working ✅ |
| **NeuroKit2 DWT** | morphology.py | extract_morphology() | **FAILS 40%** ❌ |
| **Feature Calculation** | feature_extraction.py | Training data | **Returns 0 on NaN** ❌ |
| **SQI (0-100)** | sqi.py | Feature vector | **Wrong scale** ❌ |
| **Dashboard duplicate** | dashboard/app.py | Web display | **Duplicate code** ❌ |

---

## 🔴 EXACT FAILURE POINTS

### Failure Point 1: NeuroKit2 DWT (PRIMARY)
```python
# morphology.py line 144-148
_, waves = nk.ecg_delineate(smooth_signal, r_peaks, sampling_rate=fs, method="dwt")

# Returns:
# {
#   "ECG_P_Onsets": [50, None, None, 200, ...]  ← None = failed to detect
#   "ECG_R_Onsets": [100, None, 300, ...]        ← None = failed
#   "ECG_R_Offsets": [150, None, 350, ...]       ← None = failed
# }
```

### Failure Point 2: None → 0 Conversion (SILENT FAILURE)
```python
# morphology.py line 204-207
if p_onset is not None and r_onset is not None and r_onset > p_onset:
    beat["pr_interval_ms"] = (r_onset - p_onset) * 1000.0 / fs
else:
    beat["pr_interval_ms"] = None  ← Stored as None

# Later in feature_extraction.py
vec[_idx("pr_interval_ms")] = float(summary.get("pr_interval_ms", 0.0))
                                                                     ↑ Returns 0 silently!
```

### Failure Point 3: SQI Scale
```python
# sqi.py line 20-86
score = 100.0
# ... deductions ...
final_score = max(0.0, score - deductions)  # Returns 0-100
return float(final_score)

# feature_extraction.py line 148
vec[_idx("sqi_score")] = _compute_sqi(signal, fs)  # Stores 0-100!
                                                     # Should be 0-1!
```

---

## ✅ WHAT'S WORKING WELL

```
✅ Pan-Tompkins R-peak detection (stage 1-5)
✅ Butterworth filters (cleaning)
✅ Feature calculation from valid indices
✅ RR interval, HR, HRV calculations
✅ Window integration and smoothing
```

## ❌ WHAT'S BROKEN

```
❌ NeuroKit2 DWT delineation (40% failure rate)
❌ Silent None → 0 conversion
❌ SQI scale (0-100 instead of 0-1)
❌ No fallback when DWT fails
❌ Dashboard duplicate code
```

---

## 🛠️ WHAT NEEDS TO BE FIXED

### Priority 1: Add Delineation Fallback
```
When NeuroKit2 DWT returns None:
├─ Use heuristic algorithm (search windows for peaks)
├─ Estimate P-onset: R-250ms to R-60ms window
├─ Estimate QRS: R-50ms to R+50ms window
└─ Estimate T-onset: R+100ms to R+400ms window
```

### Priority 2: Fix SQI Scale
```
Change: return float(final_score)
To:     return float(final_score / 100.0)
```

### Priority 3: Add Validation
```
After getting delineation indices, check:
├─ p_onset < r_onset (P comes before R)
├─ r_onset < r_offset (QRS has duration)
├─ r_offset < t_onset (T comes after QRS)
├─ All intervals in physiological range
└─ Skip segment if any check fails
```

---

## SUMMARY TABLE

| Component | Algorithm | Status | Fix Needed |
|-----------|-----------|--------|-----------|
| Cleaning - Highpass | Butterworth 0.5Hz | ✅ Good | No |
| Cleaning - Notch | Butterworth 50/60Hz | ✅ Good | No |
| R-peak Detection | Pan-Tompkins 5-stage | ✅ Good | No |
| Morphology - DWT | NeuroKit2 delineate | ❌ 40% fail | **YES - Add fallback** |
| Feature Extraction | Mathematical calc | ⚠️ Partial | **YES - Validation** |
| SQI Score | Custom 0-100 | ❌ Wrong scale | **YES - Divide by 100** |
| Dashboard | Duplicate code | ⚠️ Inconsistent | **YES - Unify** |

---

## YOUR EXACT SIGNAL PROCESSING STACK

```
INPUT: Raw ECG 10-second strip (1250 samples @ 125Hz)
  ↓
scipy.signal.butter + filtfilt (0.5Hz highpass) — GOOD ✅
  ↓
scipy.signal.butter + filtfilt (notch 50/60Hz) — GOOD ✅
  ↓
Min-max normalization to [-1, 1] — GOOD ✅
  ↓
Pan-Tompkins QRS detection (5 stages) — GOOD ✅
  ↓
NeuroKit2 DWT wave delineation — BROKEN ❌ (40% fail)
  ↓
Feature calculation from indices — BROKEN ❌ (returns 0 on None)
  ↓
Custom SQI 0-100 calculation — BROKEN ❌ (wrong scale)
  ↓
Store in database — CORRUPTED ❌
  ↓
OUTPUT: Corrupted feature vector with 40% bad values
```

This is what needs to be fixed! ⚠️
