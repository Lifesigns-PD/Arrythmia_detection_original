# ECG Arrhythmia Detection System — Project Overview

## Executive Summary

This is a **real-time ECG arrhythmia detection system** designed for continuous patient monitoring in clinical settings. The system processes raw ECG signals through an AI-powered pipeline and identifies 25+ types of cardiac arrhythmias, ranging from benign (Sinus Bradycardia) to life-threatening (Ventricular Fibrillation).

**Input**: Kafka stream of 1-minute ECG recordings (7500 samples @ 125 Hz)  
**Output**: MongoDB document with detected rhythms, ectopic beats, and clinical intervals per 10-second segment  
**Deployment**: Docker (Kafka + ECG processor) + AWS ECS/Lambda for model inference

---

## System Architecture

```
┌─────────────────┐
│   ECG Device    │  (125 Hz, 1-minute recording)
└────────┬────────┘
         │ Kafka Topic (ecg-raw)
         ▼
┌─────────────────────────────┐
│  Kafka Consumer (5 workers)  │  Consumes JSON messages
└────────┬────────────────────┘
         │ 7500 samples
         ▼
┌─────────────────────────────────────┐
│  ECG Processor (ecg_processor.py)    │
│  ├─ Split: 6 × 10-second segments   │
│  ├─ Signal Processing (V3)           │
│  ├─ ML Inference (Rhythm + Ectopy)   │
│  └─ Rules Engine                    │
└────────┬────────────────────────────┘
         │ Structured result
         ▼
┌─────────────────┐
│    MongoDB      │  Store events, intervals, confidence scores
└─────────────────┘
```

---

## Signal Processing Pipeline (V3)

### Stage 1: Preprocessing
- **Baseline Wander Removal**: Morphological median (0.6s window) removes respiratory drift
- **Bandpass Filter**: 0.5–40 Hz (4th-order Butterworth, zero-phase `filtfilt`)
- **Adaptive Notch**: Auto-detects 50 vs 60 Hz powerline interference
- **Artifact Suppression**: Wavelet-based EMG muscle artifact removal
- **Quality Index**: 10-criteria SQI check (saturation, flatline, noise, VF chaos, etc.)

### Stage 2: R-Peak Detection (Ensemble)
Three independent detectors run in parallel; a peak is confirmed only if **≥2 of 3 agree** within 50 ms:

| Detector | Method |
|----------|--------|
| Pan-Tompkins | Derivative → square → moving average; threshold-based |
| Hilbert Envelope | Analytic signal envelope at QRS band (5–40 Hz) |
| Wavelet | Multi-scale CWT energy (25, 60, 100, 150 ms scales) |

**Post-detection validation**:
- Physiological refractory period (min 200 ms between peaks)
- Sub-sample parabolic interpolation (±4 ms quantization correction)
- RR outlier removal (>3× median RR = artefact)

### Stage 3: Waveform Delineation (Per-Beat)
Locates 5 components per beat using CWT and amplitude thresholds:

| Component | Detected | Key Metrics |
|-----------|----------|-------------|
| P wave | Yes | onset, peak, offset, morphology (normal/inverted/biphasic/absent) |
| Q wave | Yes | depth, peak |
| R wave | Yes | amplitude, polarity (positive/negative/isoelectric) |
| S wave | Yes | depth, peak |
| T wave | Yes | onset, peak, offset, inverted flag |

**Inverted-lead handling**: Automatically detects negative QRS (e.g., aVR), processes internally on flipped signal, maps results back to original coordinates.

### Stage 4: Feature Extraction (60 Features)

| Category | Count | Examples |
|----------|-------|----------|
| HRV Time Domain | 11 | SDNN, RMSSD, pNN50, mean HR, triangular index |
| HRV Frequency | 8 | LF power, HF power, LF/HF ratio, spectral entropy |
| Nonlinear HRV | 8 | Sample entropy, Hurst exponent, Poincaré SD1/SD2, DFA |
| Morphology | 13 | PR interval, QRS duration, QT interval, QTc, ST elevation, T-wave asymmetry |
| Beat Discriminators | 20 | PVC score, PAC score, p_absent_fraction, qrs_wide_fraction, short_coupling_fraction |

---

## Machine Learning Models

### Architecture: CNN + Transformer + Feature Fusion

```
Input: 1250 signal samples + 60 clinical features
         │
    ┌────┴────────────────────┐
    │                         │
 CNN (local)         Feature Projection
(32→64→128)         (60 → 64)
    │                         │
    └────┬────────────────────┘
    Transformer (global)
         │
    Classification Head
    (192 → 256 → num_classes)
         │
    Softmax → Probabilities
```

**Why this architecture?**
- **CNN**: Efficiently extracts local waveform shapes (QRS morphology, T-wave) — reduces 1250 samples to 78 features
- **Transformer**: Processes all beats globally at once (no sequential bottleneck) — ideal for rhythm patterns (AF, blocks)
- **Feature Fusion**: 60 V3 features inject clinical domain knowledge — compensates for rare classes with small training sets
- **Separate models**: Rhythm model (10-second segment) and Ectopy model (2-second per-beat window) avoid task confusion

### Training Configuration
- **Loss**: Focal Loss (γ=3.0) — down-weights easy examples, forces learning from rare classes
- **Sampler**: WeightedRandomSampler — ensures rare classes appear proportionally per batch
- **Feature Normalization**: StandardScaler fit on training split, saved via joblib, applied at inference
- **Validation**: Balanced accuracy metric (accounts for class imbalance)

---

## Detectable Arrhythmias

### Normal Rhythms (3)
| Event | Definition |
|-------|-----------|
| **Sinus Rhythm** | Normal heart rate (60–100 bpm), regular PR intervals, normal QRS |
| **Sinus Bradycardia** | Slow sinus rhythm (HR < 60 bpm) |
| **Sinus Tachycardia** | Fast sinus rhythm (HR > 100 bpm) |

### Atrial Arrhythmias (7)
| Event | Definition | ML Class | Confidence Threshold |
|-------|-----------|----------|----------------------|
| **Atrial Fibrillation (AF)** | Chaotic atrial activity, no P-waves, irregular RR | Rhythm (class 1) | 0.85 |
| **Atrial Flutter** | Regular atrial rate 250–350 bpm, sawtooth baseline | Rhythm (class 2) | 0.85 |
| **PAC** (Premature Atrial Contraction) | Early beat from atrium, narrow QRS, inverted P | Ectopy (class 2) | 0.97 |
| **PAC Bigeminy** | PAC alternating with normal beats | Rules-derived | — |
| **PAC Trigeminy** | PAC every 3rd beat | Rules-derived | — |
| **Atrial Couplet** | 2 consecutive PACs | Rules-derived | — |
| **Atrial Run** | 3–10 consecutive PACs (slow VT rate, narrow QRS) | Rules-derived | — |

### Ventricular Arrhythmias (8)
| Event | Definition | ML Class | Confidence Threshold |
|-------|-----------|----------|----------------------|
| **PVC** (Premature Ventricular Contraction) | Early beat from ventricle, wide QRS (>120ms), no P-wave | Ectopy (class 1) | 0.97 |
| **PVC Bigeminy** | PVC alternating with normal beats | Rules-derived | — |
| **PVC Trigeminy** | PVC every 3rd beat | Rules-derived | — |
| **PVC Couplet** | 2 consecutive PVCs | Rules-derived | — |
| **Ventricular Run** | 3–10 consecutive PVCs (HR 100–200 bpm) | Rules-derived | — |
| **NSVT** (Non-Sustained VT) | 3–30 consecutive PVCs (HR > 120 bpm, <30s duration) | Rules-derived | — |
| **VT** (Ventricular Tachycardia) | >120 bpm, wide QRS, no P-waves. Includes sustained VT + folded NSVT | Rhythm (class TBD) | 0.88 |
| **VF** (Ventricular Fibrillation) | Chaotic uncoordinated activity, no recognizable QRS, disorganized baseline | Rhythm (class TBD) | 0.90 |

### Conduction Disturbances (5)
| Event | Definition | ML Class | Confidence Threshold |
|-------|-----------|----------|----------------------|
| **1st Degree AV Block** | PR interval > 200 ms (delayed conduction, all beats conduct) | Rhythm (class 3) | 0.80 |
| **2nd Degree AV Block Type 1** (Wenckebach) | Progressive PR lengthening → dropped beat | Rhythm (class 8) | 0.82 |
| **2nd Degree AV Block Type 2** (Mobitz II) | Sudden dropped beats without PR prolongation | Rhythm (class 9) | 0.85 |
| **3rd Degree AV Block** (Complete Heart Block) | No AV conduction, atria and ventricles beat independently | Rhythm (class 4) | 0.85 |
| **Bundle Branch Block** (BBB, LBBB, RBBB) | QRS duration > 120 ms (intraventricular conduction delay) | Rhythm (class 5) | 0.80 |

### Escape Rhythms & Other (5)
| Event | Definition | Background Only |
|-------|-----------|-----------------|
| **Junctional Rhythm** | HR 40–60, no P-waves, narrow QRS (AV node escape) | Yes |
| **Idioventricular Rhythm** | HR < 40, no P-waves, wide QRS (ventricular escape) | Yes |
| **Pause** | RR interval > 2.0 seconds (sinus pause or arrest) | Yes |
| **Artifact** | Unreadable signal, saturation, flatline, high EMG | Yes |
| **Unknown** | Insufficient data or ambiguous pattern | Yes |

---

## Decision Engine (Rules Layer)

After ML inference, the **Rules Engine** detects higher-level patterns:

### Beat-Level Rules
- **PVC/PAC Scoring**: 8–9 electrophysiology criteria per beat (QRS width, P-wave presence, T-wave discordance, coupling interval, etc.)

### Segment-Level Pattern Detection
- **Bigeminy**: Alternating normal + ectopic beat
- **Trigeminy**: Ectopic every 3rd beat
- **Couplet**: 2 consecutive ectopic beats
- **Run**: 3+ consecutive ectopic beats (rate determines VT vs Atrial Run)
- **Compensatory Pause**: Post-PVC RR ≈ 2× normal (VT indicator)

### Confidence Gates
Not every ML output triggers an event. Per-class thresholds prevent false alarms:
- **Critical** (VF, VT): 0.88–0.90 — high bar, immediate alarm if passed
- **High risk** (AF, 3rd Degree AV Block): 0.85 — careful balance
- **Moderate** (1st Degree AV Block): 0.80 — informational
- **Low** (Sinus Bradycardia): 0.75 — common, lower bar

### Priority & Display
High-priority events **promote to background rhythm**:
1. VF > 2. VT > 3. AF/AFL > 4. NSVT > 5. Everything else

---

## Data Flow & Integration Points

### 1. Kafka → ECG Processor
```json
{
  "admissionId": "ADM123456",
  "patientId": "PAT789",
  "deviceId": "ECG-DEVICE-01",
  "facilityId": "FACILITY-A",
  "timestamp": 1712600000,
  "ecgData": [0.12, 0.15, ...]  // 7500 floats, 125 Hz, mV scale
}
```

### 2. ECG Processor → MongoDB
```json
{
  "admissionId": "ADM123456",
  "patientId": "PAT789",
  "timestamp": 1712600000,
  "analysis": {
    "background_rhythm": "Sinus Rhythm",
    "heart_rate_bpm": 72,
    "segments": [
      {
        "segment_index": 0,
        "start_time_s": 0.0,
        "end_time_s": 10.0,
        "background_rhythm": "Sinus Rhythm",
        "primary_conclusion": "PVC Bigeminy",
        "rhythm_confidence": 0.923,
        "ectopy_label": "PVC",
        "ectopy_confidence": 0.981,
        "events": [
          {
            "event_type": "PVC Bigeminy",
            "event_category": "ECTOPY",
            "start_time": 2.3,
            "end_time": 8.7,
            "confidence": 0.981,
            "xai_notes": ["qrs_wide_fraction=1.0", "short_coupling_fraction=0.9"]
          }
        ],
        "morphology": {
          "hr_bpm": 72,
          "pr_interval_ms": 148,
          "qrs_duration_ms": 98,
          "qtc_ms": 412,
          "p_wave_present_ratio": 0.9,
          "sqi_score": 0.94,
          "sqi_issues": []
        }
      }
    ],
    "summary": {
      "dominant_rhythm": "Sinus Rhythm",
      "arrhythmia_detected": true,
      "events_found": ["PVC Bigeminy"],
      "total_segments": 6,
      "signal_quality": "acceptable"
    }
  }
}
```

### 3. Dashboard Display
Real-time web UI (Flask) shows:
- 10-second ECG trace (waveform plot)
- Detected R-peaks, P/Q/R/S/T markers
- Clinical intervals (HR, PR, QRS, QT, QTc)
- Events with confidence scores
- XAI explanation (top 5 contributing features)
- Signal quality indicator

---

## Clinical Validation & Reference Ranges

### Normal Intervals (Reference)
| Metric | Normal Range | Abnormal | Clinical Significance |
|--------|-------------|----------|----------------------|
| **Heart Rate** | 60–100 bpm | Brady < 60, Tachy > 100 | Rate classification, workload |
| **PR Interval** | 120–200 ms | > 200 ms = 1st Degree AV Block | AV conduction delay |
| **QRS Duration** | 80–120 ms | > 120 ms = BBB or PVC | Intraventricular conduction |
| **QT Interval** | 360–440 ms | Long QT > 470 ms | Repolarization prolongation (arrhythmia risk) |
| **QTc (Bazett)** | 350–440 ms (M), 350–460 ms (F) | > 460 ms = Long QT syndrome | Rate-corrected repolarization |
| **SDNN (HRV)** | > 50 ms | < 50 ms = reduced HRV | Autonomic nervous system balance (cardiac risk predictor) |

### Arrhythmia Risk Stratification
| Level | Events | Clinical Action |
|-------|--------|-----------------|
| **CRITICAL** | VF, sustained VT, 3rd Degree AV Block | Immediate alarm, consider defibrillation, pacing |
| **HIGH** | AF, 2nd Degree Type 2, runs of VT | Physician review, medication/intervention |
| **MODERATE** | PVC, PAC, 1st/2nd Degree Type 1, AFl | Monitoring, may require treatment |
| **LOW** | Bigeminy, Trigeminy, Sinus Brady/Tachy | Informational, routine follow-up |
| **NORMAL** | Sinus Rhythm | No action needed |

---

## Comparison to NeuroKit2 (Reference Standard)

The custom V3 pipeline is **independent** of NeuroKit2 but uses it as a **display reference**:

| Aspect | V3 Custom | NeuroKit2 |
|--------|-----------|----------|
| **Preprocessing** | Morphological baseline + Butterworth | High-pass filter + bandpass |
| **R-Peak Detection** | Ensemble voting (3 detectors) | Single Neurokit method |
| **Delineation** | CWT + amplitude thresholds (rate-independent) | Discrete Wavelet Transform (DWT) |
| **Sampling Rate Tuning** | Optimized for 125 Hz | Designed for 500 Hz (scales with coarser resolution at 125 Hz) |
| **Features Returned** | 60 clinical features | Basic HR/intervals only |

**Why separate?**
- V3 is optimized for 125 Hz sampling (typical ICU/portable devices)
- NeuroKit2 HR/QRS/PR are computed **live on dashboard** (independent of stored features)
- V3 features feed **ML models** (trained on 60-feature vectors)
- Fusion considered but rejected — V3 already outperforms NK2 on 125 Hz at delineation

---

## File Structure

```
├── README.md                           Entry point for developers
├── PROJECT_OVERVIEW.md                 This file
│
├── signal_processing_v3/               V3 pipeline (preprocessing → delineation → features)
│   ├── preprocessing/                  Baseline, denoising, artifact suppression
│   ├── detection/                      Ensemble R-peak detection (Pan-T + Hilbert + Wavelet)
│   ├── delineation/                    CWT P/Q/R/S/T waveform detection
│   ├── features/                       60-feature extraction (HRV, morphology, beat discriminators)
│   └── quality/                        Signal quality index (10-criteria)
│
├── models_training/                    Model training & checkpoints
│   ├── models_v2.py                    CNN+Transformer architecture definition
│   ├── data_loader.py                  Dataset class, label mapping (9 rhythm, 3 ectopy)
│   ├── retrain_v2.py                   Training script with FocalLoss + WeightedRandomSampler
│   └── outputs/checkpoints/            Saved .pth weights + feature scalers (.joblib)
│
├── xai/                                Inference & explainability
│   └── xai.py                          Model inference, per-beat ectopy, SHAP explanations
│
├── decision_engine/                    Rules + orchestration
│   ├── rhythm_orchestrator.py          ML + rules → final decision
│   └── rules.py                        Pattern detection (bigeminy, trigeminy, runs)
│
├── kafka_consumer.py                   Kafka consumer loop (5-worker thread pool)
├── ecg_processor.py                    Full pipeline orchestration (6 segments per message)
├── mongo_writer.py                     MongoDB write logic
│
├── dashboard/                          Flask web UI
│   ├── app.py                          Routes, API endpoints
│   └── templates/index.html            Real-time waveform + events display
│
├── scripts/                            Utility scripts
│   ├── backfill_features.py            Compute V3 features for all DB segments
│   ├── compare_models.py               Evaluate checkpoints (balanced accuracy, confusion matrix)
│   └── visualise_pipeline.py           Debug pipeline per segment (matplotlib plots)
│
├── docker-compose.yml                  Kafka 4.0 KRaft + Kafbat UI
├── Dockerfile                          Python 3.13 + model checkpoints
│
└── docs/                               Technical documentation
    ├── MODEL_ARCHITECTURE.md           Model design, FocalLoss, feature fusion rationale
    ├── SIGNAL_PROCESSING_METHODS.md    Signal processing methodology (for reviewers)
    └── arrhythmia_rules_documentation.md  Rules engine reference
```

---

## Training & Deployment

### Pre-Training Verification (Complete ✓)
- [x] All 6,082 cardiologist-verified segments backfilled with V3 features
- [x] 2nd Degree AV Block Type 2 restored as class 8 (was missing, 68 segments recovered)
- [x] All T-wave snap bugs fixed (hilbert, wavelet detectors)
- [x] R-peak sub-sample parabolic interpolation verified
- [x] Feature normalization (StandardScaler + joblib) configured
- [x] All documentation updated
- [x] Dashboard verified
- [x] Data ingestion format confirmed (7500 samples @ 125 Hz, mV scale)

### Training (Next Steps)
```bash
# 1. Rhythm model (9 classes)
python models_training/retrain_v2.py --task rhythm --mode initial

# 2. Ectopy model (3 classes: None/PVC/PAC)
python models_training/retrain_v2.py --task ectopy --mode initial

# 3. Evaluate
python scripts/compare_models.py --task rhythm
python scripts/compare_models.py --task ectopy
```

**Target Metrics**: Balanced accuracy > 0.75 (acceptable), > 0.85 (good)

### Deployment
```bash
# Build Docker image with trained checkpoints
docker build -t ecg-processor:latest .

# Run consumer (5 workers, auto-scales)
docker run --env-file .env --name ecg-consumer ecg-processor:latest
```

---

## Known Limitations & Future Enhancements

### Current Scope
- ✓ Single-lead ECG (can be extended to multi-lead)
- ✓ 125 Hz sampling (optimized for portable/ICU devices)
- ✓ 10-second windows (can be adjusted to 5/15/30 seconds)
- ✓ Real-time processing (latency < 500 ms per segment)

### Future Enhancements
- [ ] Multi-lead fusion (12-lead ECG diagnostic refinement)
- [ ] Temporal context across segments (segment N-1 influences N)
- [ ] Fine-tuning for specific patient cohorts (post-MI, cardiac surgery, etc.)
- [ ] Streaming model updates (online learning from cardiologist corrections)
- [ ] Confidence calibration (uncertainty quantification per prediction)

---

## Contact & Support

**Questions about**: Signal processing? Model training? Data format? Rules engine?

See:
- [signal_processing_v3/README.md](signal_processing_v3/README.md) — V3 pipeline details
- [docs/MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md) — Architecture & training rationale
- [decision_engine/rules.py](decision_engine/rules.py) — Rules implementation

---

**Last Updated**: 2026-04-17  
**Status**: Ready for training and deployment
