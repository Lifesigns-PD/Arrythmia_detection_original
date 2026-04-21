# ECG Arrhythmia Detection System

Real-time arrhythmia detection for continuous patient monitoring.
Processes raw 1-minute ECG recordings end-to-end: from a Kafka message to a
structured clinical result written to MongoDB — fully automated, no human in the loop.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Signal Flow — End to End](#signal-flow--end-to-end)
3. [Kafka Consumer](#kafka-consumer)
4. [ECG Processor](#ecg-processor)
5. [Signal Processing Pipeline (V3)](#signal-processing-pipeline-v3)
6. [ML Inference](#ml-inference)
7. [Decision Engine](#decision-engine)
8. [Output — MongoDB Document](#output--mongodb-document)
9. [Detectable Arrhythmias](#detectable-arrhythmias)
10. [Database (PostgreSQL)](#database-postgresql)
11. [Interactive Dashboard](#interactive-dashboard)
12. [Setup & Deployment](#setup--deployment)
13. [Training the Models](#training-the-models)
14. [File Structure](#file-structure)

---

## System Overview

```
ECG Device (125 Hz)
        │
        │  JSON over Kafka
        ▼
┌───────────────────┐
│   kafka_consumer  │  ← 5 parallel worker threads
│   (confluent-     │
│    kafka)         │
└────────┬──────────┘
         │  ecg_data: list[float] (7500 samples)
         ▼
┌───────────────────┐
│  ecg_processor    │  ← segments into 6 × 10s windows
│  .process()       │
└────────┬──────────┘
         │  per 10-second window (1250 samples)
         ▼
┌───────────────────┐
│  signal_          │  preprocess → R-peaks → delineation → 60 features
│  processing_v3    │
└────────┬──────────┘
         │  cleaned signal + r_peaks + delineation + features
         ▼
┌───────────────────┐
│  xai/xai.py       │  ← two CNN+Transformer models
│  explain_segment  │    rhythm (9 classes) + ectopy per beat
└────────┬──────────┘
         │  ml_prediction: rhythm label + beat-level ectopy
         ▼
┌───────────────────┐
│  decision_engine  │  sinus gate → ML → rules → pattern detection
│  orchestrator     │
└────────┬──────────┘
         │  SegmentDecision: background_rhythm + events
         ▼
┌───────────────────┐
│  mongo_writer     │  insert_one with retry + backoff
│  write_result()   │
└───────────────────┘
         │
         ▼
   MongoDB Collection
   arrhythmia_results
```

---

## Signal Flow — End to End

### Step 1 — ECG Device Sends Data

The bedside ECG device (or test producer) publishes a JSON message to a Kafka topic every 60 seconds:

```json
{
  "admissionId":  "ADM1014424580",
  "deviceId":     "ECG-DEVICE-01",
  "patientId":    "PAT789",
  "facilityId":   "CF1315821527",
  "timestamp":    1712600000,
  "ecgData":      [0.12, 0.15, 0.09, ...]
}
```

| Field | Type | Description |
|---|---|---|
| `admissionId` | string | Unique admission identifier |
| `deviceId` | string | Device that sent the recording |
| `patientId` | string | Patient identifier |
| `facilityId` | string | Facility/ward identifier |
| `timestamp` | int (Unix) | Recording start time |
| `ecgData` | float[] | **7500 samples** — 1 minute at 125 Hz, millivolt scale |

> The ECG_Data_Extracts folder contains 659-packet hospital data files where each packet
> is `{"packetNo": N, "admissionId": "ADM...", "value": [[samples...]]}`.
> These are concatenated to reconstruct the same 7500-sample flat array before processing.

---

### Step 2 — Kafka Consumer Receives and Dispatches

**File:** `kafka_consumer.py`

- Subscribes to `KAFKA_TOPIC` (default: `ecg-arrhythmia-topic`)
- Uses `enable.auto.commit = False` — offsets committed only after successful MongoDB write
- Runs a `ThreadPoolExecutor` with `CONSUMER_THREADS` workers (default: 5)
- SIGTERM / SIGINT handled gracefully — waits for in-flight messages before shutdown

Message validation before processing:
- Required fields: `deviceId`, `admissionId`, `timestamp`, `ecgData`
- Expected length: `SAMPLES_PER_MINUTE = 7500` (warns but continues on mismatch)

---

### Step 3 — Segmentation

**File:** `ecg_processor.py → _segment()`

The 7500-sample (60s) signal is sliced into non-overlapping **10-second windows**:

```
[  0 – 1249]  Segment 0   (0s –  10s)
[1250 – 2499]  Segment 1   (10s – 20s)
[2500 – 3749]  Segment 2   (20s – 30s)
[3750 – 4999]  Segment 3   (30s – 40s)
[5000 – 6249]  Segment 4   (40s – 50s)
[6250 – 7499]  Segment 5   (50s – 60s)
```

Each window: 1250 samples, 125 Hz = 10 seconds.
Shorter final windows are zero-padded to exactly 1250 samples.

Every window is processed **independently** through the full pipeline below.

---

## Signal Processing Pipeline (V3)

**Package:** `signal_processing_v3/`
**Entry function:** `process_ecg_v3(signal, fs=125)` in `signal_processing_v3/__init__.py`

### Stage 1 — Preprocessing

**File:** `signal_processing_v3/preprocessing/pipeline.py → preprocess_v3()`

Four sequential steps:

| Step | Method | Purpose |
|---|---|---|
| Artifact clipping | `artifact_removal.py` | Hard-clips extreme spikes before filtering |
| Baseline wander removal | `adaptive_baseline.py` | Butterworth HP 0.15 Hz + Savitzky-Golay (window=251) + morphological opening with 200ms structuring element |
| Powerline notch | `adaptive_denoising.py` | Auto-detects 50 Hz vs 60 Hz using FFT peak, applies IIR notch (Q=30) |
| Lowpass filter | `adaptive_denoising.py` | Butterworth LP 45 Hz, order 4 — removes high-frequency muscle noise |

Returns: `cleaned` signal array + `quality_score` (0–1) + `quality_issues` list.

---

### Stage 2 — Ensemble R-Peak Detection

**File:** `signal_processing_v3/detection/ensemble.py`
**Functions:** `detect_r_peaks_ensemble()` then `refine_peaks_subsample()`

Three independent detectors run in parallel, then vote:

#### Detector A — Pan-Tompkins (`pan_tompkins.py`)

The classical algorithm adapted for 125 Hz:

```
cleaned signal
    │
    ├─ Bandpass filter 5–15 Hz (Butterworth order 3)
    │    isolates QRS frequency band
    │
    ├─ Derivative: emphasises steep QRS slopes
    │    d[n] = (-x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2]) / 8
    │
    ├─ Squaring: makes all values positive, enhances large deflections
    │    s[n] = d[n]²
    │
    ├─ Moving-window integration (window = 150ms = 19 samples at 125Hz)
    │    smooths the energy envelope
    │
    └─ Adaptive dual threshold:
         · Signal threshold: 0.25 × current peak estimate
         · Noise threshold:  0.5  × noise level estimate
         · Both updated after each detection (online learning)
         Refractory period: 200ms (prevents double-detection)
```

#### Detector B — Hilbert Envelope (`hilbert_detector.py`)

```
cleaned signal
    │
    ├─ Bandpass 8–20 Hz
    ├─ Hilbert transform → analytic signal
    ├─ Envelope = |analytic signal|
    └─ Peak picking above 0.3 × max envelope, min spacing 200ms
```

#### Detector C — CWT Mexican Hat (`wavelet_detector.py`)

```
cleaned signal
    │
    ├─ Continuous Wavelet Transform, Mexican Hat (scale 4–8 for 125 Hz QRS)
    ├─ Sum absolute CWT coefficients across scales → energy signal
    └─ Peak picking above 0.4 × max energy, min spacing 200ms
```

#### Ensemble Vote

A candidate R-peak is **accepted** only if at least 2 of the 3 detectors agree
within a ±50ms tolerance window.

#### Sub-sample Refinement (`refine_peaks_subsample()`)

Parabolic interpolation around each accepted peak:

```
p(t) = a·t² + b·t + c
peak_refined = -b / (2a)    [vertex of fitted parabola]
```

This gives sub-sample timing accuracy, which matters for HRV frequency domain features.

---

### Stage 3 — Signal Quality Index

**File:** `signal_processing_v3/quality/signal_quality.py → compute_sqi_v3()`

Composite score from three components:

| Component | What it measures | Weight |
|---|---|---|
| SNR | Signal-to-noise ratio (dB) | 40% |
| Kurtosis | Peakedness — QRS spikes have kurtosis > 5 | 30% |
| Flatline fraction | Consecutive identical samples | 30% |

SQI < 0.3 → segment marked `UNRELIABLE` → output shows "Artifact" event.
SQI ≥ 0.5 → eligible for the None-class pool in training.

---

### Stage 4 — Waveform Delineation (P/Q/R/S/T)

**File:** `signal_processing_v3/delineation/hybrid.py → delineate_v3()`
Combines two sub-methods per beat:

#### CWT Delineation (`wavelet_delineation.py`)

```
Per beat window (centred on R-peak):
    │
    ├─ CWT Mexican Hat at multiple scales
    ├─ Zero-crossings in CWT coefficients locate wave boundaries:
    │    · QRS onset  = last zero-crossing before R descending slope
    │    · QRS offset = first zero-crossing after S-wave
    │    · P-wave     = search [-300ms, -100ms] from R; energy gate for AF rejection
    │    · T-wave     = search [+100ms, +400ms] from R
    └─ P-wave morphology classification: normal / inverted / biphasic / absent
```

#### Template Matching (`template_matching.py`)

```
Build median template from 8 beats:
    · Align on R-peak
    · Take median at each sample → noise-robust template

Per-beat refinement:
    · Cross-correlation with template in ±60ms window
    · Shift peak to maximum correlation position
    · Final peak = 60% template + 40% wavelet blend
```

Output per beat: `p_onset`, `p_peak`, `p_offset`, `qrs_onset`, `q_peak`, `r_peak`,
`s_peak`, `qrs_offset`, `t_onset`, `t_peak`, `t_offset`, `qrs_duration_ms`,
`pr_interval_ms`, `qt_ms`, `p_morphology`.

---

### Stage 5 — Feature Extraction (60 Features)

**File:** `signal_processing_v3/features/extraction.py → extract_features_v3()`

60 features in 5 domains:

#### HRV Time Domain (11 features) — `hrv_time_domain.py`

`mean_hr_bpm`, `sdnn`, `rmssd`, `pnn50`, `mean_rr_ms`, `median_rr_ms`,
`rr_cv`, `rr_std`, `nn50`, `triangular_index`, `hr_range`

#### HRV Frequency Domain (8 features) — `hrv_frequency.py`

`vlf_power`, `lf_power`, `hf_power`, `lf_hf_ratio`, `total_power`,
`lf_norm`, `hf_norm`, `peak_hf_hz`

Welch PSD on RR intervals: VLF 0.003–0.04 Hz, LF 0.04–0.15 Hz, HF 0.15–0.4 Hz.

#### Nonlinear HRV (8 features) — `nonlinear.py`

`sd1`, `sd2`, `sd1_sd2_ratio` (Poincaré plot axes),
`sampen` (Sample Entropy, m=2, r=0.2×std),
`dfa_alpha1` (Detrended Fluctuation Analysis short-range scaling exponent),
`apen`, `rr_entropy`, `correlation_dimension`

#### Morphology (13 features) — `beat_morphology.py` + `morphology_features.py`

`mean_qrs_duration_ms`, `qrs_duration_std`, `mean_pr_interval_ms`,
`pr_interval_ms`, `mean_qt_ms`, `mean_qtc_ms` (Bazett's formula),
`p_wave_amplitude_mean`, `t_wave_amplitude_mean`, `r_wave_amplitude_mean`,
`st_level_mean`, `p_wave_present_ratio`, `p_absent_fraction`, `qrs_wide_fraction`

#### Beat Discriminators (20 features)

`pvc_score_mean`, `pvc_score_max`, `pac_score_mean`, `pac_score_max`,
`compensatory_pause_ratio`, `prematurity_index_mean`, `t_discordance_mean`,
`beat_rr_ratio_cv`, `consecutive_wide_beats`, `narrow_beat_fraction`,
`wide_beat_fraction`, `early_beat_fraction`, `beat_coupling_interval_mean`,
`beat_coupling_cv`, `ectopy_beat_count`, `atrial_ectopy_count`,
`ventricular_ectopy_count`, `f_wave_power_ratio`, `flutter_spectral_ratio`,
`spectral_irregularity`

These 60 features are scaled with a `StandardScaler` (fitted on training data,
saved as `outputs/checkpoints/feature_scaler_*.joblib`) before being fed to the ML model.

---

## ML Inference

**File:** `xai/xai.py → explain_segment()`
**Model definition:** `models_training/models_v2.py → CNNTransformerWithFeatures`

### Model Architecture

Two separate models share the same architecture but different output heads:

```
Input Signal: (B, 1, 1250)          ← 10-second window, single lead
    │
    ├─ Conv1d(1→32,  k=7) → BatchNorm → ReLU → MaxPool1d(2)  → (B,  32, 625)
    ├─ Conv1d(32→64, k=7) → BatchNorm → ReLU → MaxPool1d(2)  → (B,  64, 312)
    ├─ Conv1d(64→128,k=7) → BatchNorm → ReLU → MaxPool1d(2)  → (B, 128, 156)
    ├─ Conv1d(128→128, k=1) [channel projection]              → (B, 128, 156)
    │
    ├─ TransformerEncoder (2 layers, 8 heads, d_ff=256)
    │    · MultiHeadAttention: each of 156 time-steps attends to all others
    │    · Captures long-range rhythm patterns (e.g. AF irregularity across beats)
    │
    └─ Global Average Pooling over time                        → z_sig (B, 128)

Input Features: (B, 60)             ← clinical feature vector
    │
    ├─ LayerNorm(60)
    ├─ Linear(60→64) → BatchNorm → ReLU → Dropout(0.1)
    └─ Linear(64→64) → ReLU                                    → z_feat (B, 64)

Fusion:
    concat([z_sig, z_feat])                                    → (B, 192)
    LayerNorm → Linear(192→64) → ReLU → Dropout(0.2) → Linear(64→N)
    Softmax                                                    → class probabilities
```

### Rhythm Model

| Item | Value |
|---|---|
| Checkpoint | `outputs/checkpoints/best_model_rhythm_v2.pth` |
| Output classes | 9 |
| Feature scaler | `feature_scaler_rhythm.joblib` (36 features) |

**9 Rhythm Classes:**
Sinus Rhythm, Sinus Bradycardia, Atrial Fibrillation, Atrial Flutter,
1st Degree AV Block, 2nd Degree AV Block Type 2, 3rd Degree AV Block,
Bundle Branch Block, Artifact

### Ectopy Model

| Item | Value |
|---|---|
| Checkpoint | `outputs/checkpoints/best_model_ectopy_v2.pth` |
| Output classes | 3 (None / PVC / PAC) |
| Feature scaler | `feature_scaler_ectopy.joblib` (47 features) |
| Beat window | 2-second window centred on each R-peak |
| Confidence threshold | ≥ 0.97 to accept a beat as ectopic |

Per-beat ectopy detection: the model runs once per R-peak using a 250-sample window
(±125 samples around the peak). Results are collected as `beat_events`:
`[{beat_idx, peak_sample, label, conf}, ...]`.

---

## Decision Engine

**File:** `decision_engine/rhythm_orchestrator.py → RhythmOrchestrator.decide()`

The orchestrator applies a **strict priority hierarchy** — signal processing rules
fire before the ML model, not after.

### Step 1 — SQI Gate

If `sqi < 0.3` → mark segment `UNRELIABLE`, attach "Artifact" event,
still derive background rhythm from HR + P-wave ratio for context.

### Step 2 — Sinus Detector (Signal Processing First)

**File:** `decision_engine/sinus_detector.py → detect_sinus_and_rhythm()`

9 criteria must ALL pass for Sinus classification:

| Criterion | Threshold | Clinical meaning |
|---|---|---|
| P-waves present | `p_absent_fraction ≤ 0.20` | AF removes P-waves in >20% of beats |
| QRS narrow | `mean_qrs_duration_ms < 120 ms` | BBB / PVC widens QRS |
| PR interval | `100–250 ms` | Short = WPW/junctional; Long >200ms = 1st degree block |
| RR regularity | `rr_cv ≤ 0.15` | AF has chaotic RR intervals (CV > 0.15) |
| Low wide-beat fraction | `qrs_wide_fraction ≤ 0.10` | < 10% of beats are wide |
| Low PVC score | `pvc_score_mean ≤ 2.0` | Composite: QRS width + compensatory pause + T-discordance |
| Low PAC score | `pac_score_mean ≤ 2.0` | Early P + short PR + narrow QRS pattern score |
| LF/HF ratio | `≥ 0.5` | AF collapses LF/HF spectral ratio toward 0 |
| HR in range | `40–150 bpm` | Escape rhythms or extreme tachycardia excluded |

If Sinus detected → classify variant by HR:
- HR < 60 bpm → **Sinus Bradycardia**
- HR 60–100 bpm → **Sinus Rhythm**
- HR > 100 bpm → **Sinus Tachycardia**

### Step 3 — ML Veto (Dangerous Rhythm Override)

Even after Sinus is declared, the ML model can override if it is confident
about a **dangerous rhythm** (threshold: ≥ 0.88 confidence):

Rhythms that trigger a veto:
`Atrial Fibrillation`, `Atrial Flutter`, `3rd Degree AV Block`,
`2nd Degree AV Block Type 2`, `Ventricular Fibrillation`, `Ventricular Tachycardia`

### Step 4 — ML Classification (Non-Sinus Only)

If the Sinus detector says "not sinus", the rhythm model result is accepted
only if confidence exceeds the per-class threshold:

| Rhythm | Min Confidence |
|---|---|
| Ventricular Fibrillation | 0.90 |
| VT / Ventricular Tachycardia | 0.88 |
| NSVT | 0.85 |
| Atrial Fibrillation / Flutter | 0.85 |
| 3rd / 2nd Degree AV Block | 0.85 |
| 2nd Degree AV Block Type 1 | 0.82 |
| 1st Degree AV Block / BBB | 0.80 |

### Step 5 — Rules Engine

**File:** `decision_engine/rules.py → derive_rule_events()`

Three direct rules fire on the signal and features:

| Rule | Trigger | Output |
|---|---|---|
| Pause | Any RR interval > 2000 ms | "Pause" event |
| AF Safety Net | `rr_std > 160 ms AND p_absent_fraction > 0.4` AND not Sinus | "Atrial Fibrillation" event |
| Atrial Flutter | HR 130–175 bpm AND FFT peak between 4–6 Hz in raw signal | "Atrial Flutter" event |

### Step 6 — Ectopy Pattern Detection

**File:** `decision_engine/rules.py → apply_ectopy_patterns()`

Beat-level PVC/PAC events are analysed for higher-level patterns:

| Pattern | Detection Logic |
|---|---|
| PVC / PAC Bigeminy | Every other beat is ectopic (alternating beat indices) |
| PVC / PAC Trigeminy | Every third beat is ectopic |
| PVC Couplet | 2 consecutive PVCs |
| Ventricular Run | 3–5 consecutive PVCs |
| NSVT | ≥ 3 consecutive PVCs at rate > 100 bpm, duration < 30 s |
| Ventricular Tachycardia | ≥ 3 consecutive PVCs at rate > 100 bpm, duration ≥ 30 s |
| Atrial Run | 3+ consecutive PACs |
| SVT / PSVT | HR > 150 + narrow QRS + sudden-onset PAC cluster |

Hallucination suppression gates applied before pattern matching:
- Base threshold: ectopy confidence ≥ 0.97
- Rhythm trust gate: if rhythm model is confident Sinus and < 3 ectopic beats detected, threshold raised to 0.99
- Density gate: if > 60% of all beats are flagged as ectopic on a Sinus background (and no consecutive run), all ectopy suppressed as hallucination

### Step 7 — Display Rules

**File:** `decision_engine/rules.py → apply_display_rules()`

Filters `decision.events` down to `final_display_events` based on background rhythm.
Example: individual PVC events are suppressed when VT is already the background rhythm.

---

## Output — MongoDB Document

**File:** `mongo_writer.py → write_result()`

Written to collection `arrhythmia_results` in database `ecg_analysis`.

```json
{
  "uuid": "3f2c1a09-...",
  "admissionId": "ADM1014424580",
  "deviceId": "ECG-DEVICE-01",
  "patientId": "PAT789",
  "facilityId": "CF1315821527",
  "timestamp": 1712600000,
  "ecgData": [0.12, 0.15, "..."],
  "analysis": {
    "background_rhythm": "Sinus Rhythm",
    "heart_rate_bpm": 72,
    "segments": [
      {
        "segment_index": 0,
        "start_time_s": 0.0,
        "end_time_s": 10.0,
        "rhythm_label": "Sinus Rhythm",
        "rhythm_confidence": 0.9231,
        "ectopy_label": "PVC",
        "ectopy_confidence": 0.9812,
        "events": ["PVC Bigeminy"],
        "primary_conclusion": "PVC Bigeminy",
        "background_rhythm": "Sinus Rhythm",
        "morphology": {
          "hr_bpm": 72,
          "pr_interval_ms": 148,
          "qrs_duration_ms": 98,
          "qtc_ms": 412,
          "p_wave_present_ratio": 0.9
        },
        "signal_quality": 0.84,
        "sinus_gate_fired": false
      }
    ],
    "summary": {
      "total_segments": 6,
      "dominant_rhythm": "Sinus Rhythm",
      "arrhythmia_detected": true,
      "events_found": ["PVC Bigeminy"],
      "signal_quality": "acceptable"
    }
  },
  "_processing_time_s": 2.14
}
```

MongoDB write uses retry logic: 3 attempts with exponential backoff (2s, 4s).
Kafka offset is committed only after a successful write.

---

## Detectable Arrhythmias

| Category | Events |
|---|---|
| Normal | Sinus Rhythm, Sinus Bradycardia, Sinus Tachycardia |
| Atrial | Atrial Fibrillation, Atrial Flutter, PAC, Atrial Couplet, Atrial Run, SVT, PSVT, PAC Bigeminy, PAC Trigeminy |
| Ventricular | PVC, PVC Couplet, PVC Bigeminy, PVC Trigeminy, Ventricular Run, NSVT, Ventricular Tachycardia |
| Conduction | 1st Degree AV Block, 2nd Degree AV Block Type 1, 2nd Degree AV Block Type 2, 3rd Degree AV Block, Bundle Branch Block |
| Other | Pause (RR > 2000ms), Junctional Rhythm, Idioventricular Rhythm, Artifact |

---

## Database (PostgreSQL)

Training data and cardiologist annotations are stored in PostgreSQL (separate from MongoDB).

**Table:** `ecg_features_annotatable`
**DB:** `ecg_analysis` (host: 127.0.0.1:5432)

Key columns:

| Column | Type | Purpose |
|---|---|---|
| `signal_data` | REAL[] | 1250-sample float array for model training |
| `features_json` | JSONB | 60-feature dict used for model input |
| `arrhythmia_label` | VARCHAR(50) | Ground-truth label (cardiologist-assigned) |
| `events_json` | JSONB | Beat-level PVC/PAC annotations |
| `sqi_score` | FLOAT | Signal quality — min 0.5 for None-class pool |
| `is_corrected` | BOOLEAN | Cardiologist verified = TRUE |
| `used_for_training` | BOOLEAN | Included in current training run |

Schema initialisation: `database/init_db.sql`
CRUD operations: `database/db_service.py`

---

## Interactive Dashboard

**File:** `streamlit_dashboard.py`

A local web app that visualises every step of the pipeline interactively.

```bash
streamlit run streamlit_dashboard.py
```

Opens at `http://localhost:8501`.

**7 tabs:**

| Tab | Contents |
|---|---|
| Signal Pipeline | 6-stage walkthrough with Plotly charts per stage |
| R-Peak Detection | Pan-Tompkins 5-step visualisation, ensemble vote detail |
| Delineation | Full P/Q/R/S/T annotations, single-beat zoom, CWT algorithm detail |
| ML Inference | Model architecture with tensor shapes, live inference button, rhythm + beat results |
| Decision Engine | 9 sinus criteria table (PASS/FAIL per row), rules engine output |
| Database | Full schema table, SQL queries per use case, live DB stats |
| Architecture | Complete system call chain as annotated code |

Accepts the sample ECG (`ecg_analysis_package/sample_ecg.json`) or any ECG_Data_Extracts
JSON file directly via file upload in the sidebar.

---

## Setup & Deployment

### 1. Clone and install

```bash
git clone <repo-url>
cd Project_Submission_Clean

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
pip install streamlit plotly    # for dashboard only
```

### 2. Configure environment

```bash
cp .env.template .env
```

Edit `.env`:

```env
# Kafka
KAFKA_BOOTSTRAP_SERVERS=your-kafka-broker:9092
KAFKA_TOPIC=ecg-arrhythmia-topic
KAFKA_GROUP_ID=ecg-arrhythmia-consumer-group

# MongoDB (results output)
MONGO_URI=mongodb://user:password@your-mongo-host:27017
MONGO_DB=ecg_analysis
MONGO_COLLECTION=arrhythmia_results

# PostgreSQL (training data)
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=ecg_analysis
DB_USER=postgres
DB_PASSWORD=yourpassword

# Optional
CONSUMER_THREADS=5
LOG_LEVEL=INFO
FACILITY_ID=CF1315821527
```

### 3. Run with Docker

```bash
docker build -t ecg-processor .
docker run --env-file .env ecg-processor
```

Or with docker-compose (includes Kafka + MongoDB):

```bash
docker-compose up
```

### 4. Run locally (no Docker)

```bash
python kafka_consumer.py
```

### 5. Dashboard only

```bash
streamlit run streamlit_dashboard.py
```

---

## Training the Models

Training pulls labelled segments from PostgreSQL, not from files.

```bash
# Step 1 — backfill V3 features for all annotated DB segments
python scripts/backfill_features.py --force --corrected

# Step 2 — initial training
python models_training/retrain_v2.py --task rhythm --mode initial --epochs 50
python models_training/retrain_v2.py --task ectopy --mode initial --epochs 50

# Step 3 — fine-tune on recent corrections
python models_training/retrain_v2.py --task rhythm --mode finetune --epochs 20 --lr 0.00001
python models_training/retrain_v2.py --task ectopy --mode finetune --epochs 20 --lr 0.00001

# Step 4 — verify checkpoints
python scripts/check_models.py --task rhythm
python scripts/check_models.py --task ectopy
```

**Training details:**
- Only rows with `is_corrected = TRUE` are used (cardiologist-verified labels)
- Class imbalance handled by `WeightedRandomSampler` + `FocalLoss(gamma=2)`
- Optimiser: `AdamW`, `weight_decay=1e-4`
- Best checkpoint saved by balanced accuracy across all classes
- NaN feature rows are skipped automatically before appending to dataset

---

## File Structure

```
Project_Submission_Clean/
│
├── ecg_processor.py              Entry point — segments + runs full pipeline per window
├── kafka_consumer.py             Kafka consumer — 5 threads, graceful SIGTERM shutdown
├── mongo_writer.py               MongoDB write with 3-attempt retry + exponential backoff
├── config.py                     All config from env vars (Kafka, Mongo, constants)
├── streamlit_dashboard.py        Interactive local dashboard (7 tabs)
├── requirements.txt              Python dependencies
├── .env.template                 Environment variable template
├── Dockerfile                    Container image definition
├── docker-compose.yml            Multi-service orchestration
│
├── signal_processing_v3/         V3 signal processing package
│   ├── __init__.py               Exports process_ecg_v3()
│   ├── preprocessing/            Baseline wander, notch filter, lowpass, artifact clips
│   ├── detection/                Ensemble R-peak: Pan-Tompkins + Hilbert + CWT
│   ├── delineation/              CWT + template matching → P/Q/R/S/T per beat
│   ├── features/                 60-feature extraction (HRV time/freq/nonlinear + morphology)
│   └── quality/                  SQI: SNR + kurtosis + flatline fraction
│
├── decision_engine/
│   ├── rhythm_orchestrator.py    Main logic — sinus gate → ML → rules → pattern detection
│   ├── sinus_detector.py         9-criteria rule-based sinus detection (runs before ML)
│   ├── rules.py                  Pause / AF / AFL rules + ectopy pattern recognition
│   └── models.py                 SegmentDecision, Event, EventCategory dataclasses
│
├── models_training/
│   ├── models_v2.py              CNNTransformerWithFeatures (SmallCNN + Transformer + MLP)
│   ├── retrain_v2.py             Training script (ECGEventDatasetV2, FocalLoss, AdamW)
│   ├── data_loader.py            RHYTHM_CLASS_NAMES, ECTOPY_CLASS_NAMES, label mappings
│   └── outputs/checkpoints/
│       ├── best_model_rhythm_v2.pth       Active rhythm model (9 classes)
│       ├── best_model_ectopy_v2.pth       Active ectopy model (3 classes)
│       ├── feature_scaler_rhythm.joblib   StandardScaler for rhythm features
│       └── feature_scaler_ectopy.joblib   StandardScaler for ectopy features
│
├── xai/
│   └── xai.py                    explain_segment() — loads both models, runs inference
│
├── database/
│   ├── db_service.py             get_segment_new(), save_model_prediction(), update_segment_status()
│   ├── init_db.sql               Table + index creation for ecg_features_annotatable
│   └── import_to_sql.py          Import labelled JSON segments into PostgreSQL
│
├── ecg_analysis_package/
│   ├── ecg_pipeline_report.py    CLI — runs pipeline on a JSON file, outputs PDF/PNG report
│   └── sample_ecg.json           Demo ECG (1250 samples, 125 Hz) for dashboard default
│
├── ECG_Data_Extracts/            Real patient recordings (659-packet JSON arrays per file)
│
└── scripts/                      Data utilities: import, backfill features, verify DB, reset flags
```
