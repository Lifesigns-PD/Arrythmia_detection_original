# ECG Arrhythmia Detection Pipeline

Real-time ECG arrhythmia detection system for continuous patient monitoring.
Consumes raw ECG data from Kafka, runs AI analysis, writes results to MongoDB.

---

## What This System Does

```
ECG Device → Kafka Topic → [This Container] → MongoDB
```

For every 1-minute ECG recording received:
1. Splits into 6 × 10-second segments
2. Cleans signal (baseline wander + powerline noise removal)
3. Detects R-peaks (Pan-Tompkins algorithm)
4. Extracts 13 clinical features (HR, HRV, PR interval, QRS duration, etc.)
5. Runs two AI models:
   - **Rhythm model** — classifies background rhythm (Sinus, AF, AV Block, BBB, etc.)
   - **Ectopy model** — detects PVC/PAC beats per-beat
6. Applies clinical rules engine — derives patterns (Bigeminy, NSVT, VT, PSVT, etc.)
7. Writes structured result to MongoDB

---

## AI Models

| Model | Architecture | Classes | Accuracy |
|-------|-------------|---------|----------|
| Rhythm V2 | CNN + Transformer + Features | 13 rhythm classes | Balanced acc (see training logs) |
| Ectopy V2 | CNN + Transformer + Features | None / PVC / PAC | 0.77 balanced acc |

Both models use signal + 13 clinical features as input (V2 architecture).
Checkpoints are included in `models_training/outputs/checkpoints/`.

---

## Setup

### 1. Environment Variables

Copy `.env.template` to `.env` and fill in your values:

```bash
cp .env.template .env
```

Required variables:

```
KAFKA_BOOTSTRAP_SERVERS=your-kafka-broker:9092
KAFKA_TOPIC=ecg-raw
KAFKA_GROUP_ID=ecg-processor-group
MONGO_URI=mongodb://user:password@your-mongo-host:27017
MONGO_DB=ecg_db
MONGO_COLLECTION=ecg_results
```

### 2. Build Docker Image

```bash
docker build -t ecg-processor .
```

Build takes 3–5 minutes (installs scipy, torch, neurokit2).

### 3. Run

```bash
docker run --env-file .env ecg-processor
```

Or with Docker Compose / Kubernetes — use environment variables from ConfigMap/Secret.

---

## Kafka Message Format (Input)

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

- `ecgData`: 7500 float values (1 minute at 125 Hz, mV scale)
- `timestamp`: Unix timestamp (seconds)

---

## MongoDB Document Format (Output)

```json
{
  "admissionId": "ADM123456",
  "deviceId": "ECG-DEVICE-01",
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
        "rhythm_label": "Sinus Rhythm",
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
      "total_segments": 6,
      "dominant_rhythm": "Sinus Rhythm",
      "arrhythmia_detected": true,
      "events_found": ["PVC Bigeminy"],
      "signal_quality": "acceptable"
    }
  }
}
```

---

## Possible Output Values

### `primary_conclusion` / `events_found`

**Normal:**
- `Sinus Rhythm`, `Sinus Bradycardia`, `Sinus Tachycardia`

**Atrial:**
- `Atrial Fibrillation`, `Atrial Flutter`
- `PAC`, `Atrial Couplet`, `Atrial Run`, `PSVT`, `SVT`
- `PAC Bigeminy`, `PAC Trigeminy`, `PAC Quadrigeminy`

**Ventricular:**
- `PVC`, `PVC Couplet`, `Ventricular Run`, `NSVT`, `VT`
- `PVC Bigeminy`, `PVC Trigeminy`, `PVC Quadrigeminy`

**Conduction:**
- `1st Degree AV Block`, `2nd Degree AV Block Type 1/2`, `3rd Degree AV Block`
- `Bundle Branch Block`

**Other:**
- `Junctional Rhythm`, `Idioventricular Rhythm`, `Pause`, `Artifact`

---

## File Structure

```
├── kafka_consumer.py          Entry point — Kafka consumer loop
├── ecg_processor.py           Full pipeline: clean → detect → infer → rules
├── mongo_writer.py            MongoDB write logic
├── config.py                  All config (reads from env vars)
├── Dockerfile
├── requirements.txt
│
├── signal_processing/
│   ├── cleaning.py            Bandpass + notch filters
│   ├── pan_tompkins.py        R-peak detection
│   ├── morphology.py          PR/QRS/QT interval extraction
│   ├── feature_extraction.py  13-feature vector builder
│   └── sqi.py                 Signal Quality Index (0–1)
│
├── models_training/
│   ├── models.py              V1 CNN+Transformer (signal only)
│   ├── models_v2.py           V2 CNN+Transformer+Features
│   ├── data_loader.py         Class name definitions
│   └── outputs/checkpoints/
│       ├── best_model_rhythm_v2.pth    Active rhythm model
│       ├── best_model_ectopy_v2.pth    Active ectopy model
│       ├── best_model_rhythm.pth       V1 fallback
│       └── best_model_ectopy.pth       V1 fallback
│
├── xai/
│   └── xai.py                 Model inference + per-beat ectopy detection
│
└── decision_engine/
    ├── rhythm_orchestrator.py  Combines ML + rules into final decision
    ├── rules.py                Pattern detection (Bigeminy, NSVT, VT, etc.)
    └── models.py               Data classes (Event, SegmentDecision, etc.)
```

---

## Requirements

- Docker (no other local dependencies needed)
- Kafka broker accessible from container
- MongoDB instance accessible from container
- Python 3.13 (handled by Docker)

Key Python packages (see `requirements.txt`):
- `torch` — AI model inference
- `neurokit2` — ECG waveform delineation
- `scipy`, `numpy` — signal processing
- `confluent-kafka` — Kafka consumer
- `pymongo` — MongoDB writes
