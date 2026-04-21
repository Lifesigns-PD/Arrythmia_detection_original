# ECG Arrhythmia Detection System

Real-time ECG arrhythmia detection for continuous patient monitoring.
Processes raw ECG signals through a multi-stage AI pipeline and produces
structured clinical event output per 10-second segment.

---

## What This System Does

```
ECG Device → Kafka Topic → ECG Processor → MongoDB
```

For every 1-minute ECG recording:
1. Splits into 6 × 10-second segments
2. Runs full V3 signal processing pipeline (preprocessing → R-peak detection → delineation → feature extraction)
3. Feeds signal + 60 clinical features into two AI models
4. Applies clinical rules engine for pattern detection
5. Writes structured arrhythmia events to MongoDB

**→ [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** — Comprehensive system overview, all 25+ arrhythmia types, clinical validation, deployment guide

---

## AI Models

| Model | Architecture | Classes | Input |
|-------|-------------|---------|-------|
| Rhythm V2 | CNN + Transformer + Feature Fusion | 9 rhythm classes | Signal (1250 samples) + 60 features |
| Ectopy V2 | CNN + Transformer + Feature Fusion | None / PVC / PAC | Signal (1250 samples) + 60 features |

**Rhythm classes:** Sinus Rhythm, Atrial Fibrillation, Atrial Flutter, 1st Degree AV Block,
3rd Degree AV Block, 2nd Degree AV Block Type 2, Bundle Branch Block, Artifact, Sinus Bradycardia

Checkpoints: `models_training/outputs/checkpoints/`

---

## Signal Processing Pipeline (V3)

```
Raw ECG (125 Hz, 10s = 1250 samples)
    │
    ├─ 1. Preprocessing
    │      Adaptive baseline removal + Butterworth bandpass (0.5–40 Hz)
    │      Adaptive notch filter (50/60 Hz auto-detect)
    │      Artifact suppression + signal quality index (SQI)
    │
    ├─ 2. R-Peak Detection (Ensemble)
    │      3 independent detectors voted by ≥2/3 agreement
    │      Polarity-corrected (handles inverted leads)
    │      Sub-sample parabolic interpolation for HRV accuracy
    │
    ├─ 3. Waveform Delineation
    │      CWT-based P/Q/R/S/T boundaries per beat
    │      Inverted-lead aware (aVR, LBBB patterns)
    │      P-wave morphology classification (normal/inverted/biphasic/absent)
    │
    └─ 4. Feature Extraction (60 features)
           HRV time domain (11) + HRV frequency domain (8)
           Nonlinear HRV (8) + Morphology (13) + Beat discriminators (20)
```

Full technical reference: `docs/ECG_Signal_Processing_Pipeline.md`
Signal processing module reference: `signal_processing_v3/README.md`

---

## Rules Engine

Pattern detection runs after the ML models and derives higher-level events:

| Pattern | Method |
|---------|--------|
| PVC Bigeminy / Trigeminy | Beat alternation counting |
| PVC Couplet / Run / NSVT | Consecutive PVC detection |
| Ventricular Tachycardia | ≥3 consecutive PVCs + rate > 100 bpm |
| SVT / PSVT | HR > 150 + narrow QRS + sudden onset |
| AV Block patterns | PR + dropped beat analysis |

Rules reference: `decision_engine/rules.py`, `docs/arrhythmia_rules_documentation.md`

---

## Setup

### 1. Environment Variables

```bash
cp .env.template .env
```

```
KAFKA_BOOTSTRAP_SERVERS=your-kafka-broker:9092
KAFKA_TOPIC=ecg-raw
KAFKA_GROUP_ID=ecg-processor-group
MONGO_URI=mongodb://user:password@your-mongo-host:27017
MONGO_DB=ecg_db
MONGO_COLLECTION=ecg_results
```

### 2. Docker

```bash
docker build -t ecg-processor .
docker run --env-file .env ecg-processor
```

---

## Input Format (Kafka)

```json
{
  "admissionId":  "ADM123456",
  "deviceId":     "ECG-DEVICE-01",
  "patientId":    "PAT789",
  "facilityId":   "FACILITY-A",
  "timestamp":    1712600000,
  "ecgData":      [0.12, 0.15, ...]
}
```

`ecgData`: 7500 float values (1 minute at 125 Hz, mV scale)

---

## Output Format (MongoDB)

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
        "primary_conclusion": "PVC Bigeminy",
        "background_rhythm": "Sinus Rhythm",
        "rhythm_confidence": 0.923,
        "ectopy_label": "PVC",
        "ectopy_confidence": 0.981,
        "events": ["PVC Bigeminy"],
        "morphology": {
          "hr_bpm": 72,
          "pr_interval_ms": 148,
          "qrs_duration_ms": 98,
          "qtc_ms": 412,
          "p_wave_present_ratio": 0.9
        }
      }
    ],
    "summary": {
      "dominant_rhythm": "Sinus Rhythm",
      "arrhythmia_detected": true,
      "events_found": ["PVC Bigeminy"],
      "signal_quality": "acceptable"
    }
  }
}
```

---

## Detectable Events

**Normal:** Sinus Rhythm, Sinus Bradycardia, Sinus Tachycardia

**Atrial:** Atrial Fibrillation, Atrial Flutter, PAC, Atrial Couplet, Atrial Run, SVT, PSVT,
PAC Bigeminy, PAC Trigeminy

**Ventricular:** PVC, PVC Couplet, PVC Bigeminy, PVC Trigeminy, Ventricular Run, NSVT, VT

**Conduction:** 1st Degree AV Block, 2nd Degree AV Block Type 1/2, 3rd Degree AV Block,
Bundle Branch Block

**Other:** Junctional Rhythm, Idioventricular Rhythm, Pause, Artifact

---

## File Structure

```
├── ecg_processor.py               Entry point — full pipeline per segment
├── kafka_consumer.py              Kafka consumer loop
├── mongo_writer.py                MongoDB write logic
│
├── signal_processing_v3/          V3 signal processing pipeline
│   ├── preprocessing/             Baseline, denoising, artifact removal
│   ├── detection/                 Ensemble R-peak detection (3 detectors)
│   ├── delineation/               CWT P/Q/R/S/T waveform delineation
│   ├── features/                  60-feature extraction
│   └── quality/                   Signal quality index
│
├── models_training/
│   ├── models_v2.py               CNN+Transformer+Features architecture
│   ├── data_loader.py             Class definitions (9 rhythm, 3 ectopy)
│   ├── retrain_v2.py              Training script
│   └── outputs/checkpoints/       Saved model weights
│
├── xai/
│   └── xai.py                     Model inference + per-beat ectopy detection
│
├── decision_engine/
│   ├── rhythm_orchestrator.py     ML + rules → final decision
│   └── rules.py                   Pattern detection
│
├── scripts/
│   ├── backfill_features.py       Populate features_json for all DB segments
│   ├── visualise_pipeline.py      Visual pipeline debugger (per segment)
│   └── compare_models.py          Evaluate model checkpoints
│
└── docs/
    ├── ECG_Signal_Processing_Pipeline.md   Full signal processing technical reference
    ├── MODEL_ARCHITECTURE.md               Model design and rationale
    └── arrhythmia_rules_documentation.md   Rules engine documentation
```

---

## Training

```bash
# 1. Backfill V3 features for all annotated segments
python scripts/backfill_features.py --force --corrected

# 2. Train
python models_training/retrain_v2.py --task rhythm --mode initial
python models_training/retrain_v2.py --task ectopy --mode initial

# 3. Evaluate
python scripts/compare_models.py --task rhythm
python scripts/compare_models.py --task ectopy
```

Training uses only `is_corrected = TRUE` segments (cardiologist-verified).
Class imbalance handled by `WeightedRandomSampler` + Focal Loss.
