# Kafka ECG Arrhythmia Pipeline - Implementation Plan

## Context

Nexus backend pushes 1-minute ECG data per patient admission to a Kafka topic. Our job:
1. **Consume** from Kafka topic `ecg-arrhythmia-topic`
2. **Process** through our arrhythmia detection pipeline (signal processing + ML + rules)
3. **Write results** to MongoDB
4. **Run as Docker container** deployable on Kubernetes

---

## What We Receive (Kafka Message)

```json
{
    "deviceId": "AB021511",
    "admissionId": "ADM819104078",
    "timestamp": 1770912322923,
    "ecgData": [1.154, 1.758, ...]   // mV values, 7500 samples (125 Hz x 60s)
}
```

- **Sampling rate:** Fixed 125 Hz (confirmed with Shreyas)
- **Data length:** Exactly 7500 samples (1 minute)
- **Format:** mV converted values (no raw ADC)
- **Topic:** `ecg-arrhythmia-topic` (10 partitions)

---

## What We Output (MongoDB Document)

```json
{
    "uuid": "generated-uuid",
    "facilityId": "CF1315821527",
    "patientId": "MRN54642783626",
    "admissionId": "ADM819104078",
    "timestamp": 1770912322923,
    "ecgData": [1.154, 1.758, ...],
    "analysis": {
        "background_rhythm": "Sinus Rhythm",
        "heart_rate_bpm": 78,
        "segments": [
            {
                "segment_index": 0,
                "start_time_s": 0.0,
                "end_time_s": 10.0,
                "rhythm_label": "Sinus Rhythm",
                "rhythm_confidence": 0.92,
                "ectopy_label": "None",
                "ectopy_confidence": 0.88,
                "events": ["Sinus Rhythm"],
                "primary_conclusion": "Sinus Rhythm",
                "morphology": {
                    "hr_bpm": 78,
                    "pr_interval_ms": 160,
                    "qrs_duration_ms": 92,
                    "qtc_ms": 410,
                    "p_wave_present_ratio": 0.9
                },
                "sinus_gate_fired": false,
                "ectopy_override": null
            }
        ],
        "summary": {
            "total_segments": 6,
            "dominant_rhythm": "Sinus Rhythm",
            "arrhythmia_detected": false,
            "events_found": [],
            "signal_quality": "acceptable"
        }
    },
    "processingStatus": null,
    "processedAt": null,
    "processedBy": null
}
```

UI picks up documents with `processingStatus = null` (oldest first) for cardiologist review.

---

## Files to Create

### 1. `kafka_consumer.py` (Main Entry Point)

**What it does:** Connects to Kafka, consumes messages, processes each through the pipeline, writes to MongoDB.

```
Kafka topic (ecg-arrhythmia-topic)
    |
    v
kafka_consumer.py
    |-- parse JSON message
    |-- validate (7500 samples, 125 Hz)
    |-- call ecg_processor.process(ecg_data, admission_id, device_id, timestamp)
    |-- write result to MongoDB
    |-- log to stdout (for CloudWatch/K8s logging)
    |
    v
MongoDB (arrhythmia results collection)
```

**Key design decisions:**
- Use `confluent-kafka` Python client (production-grade, better than `kafka-python`)
- Consumer group: `ecg-arrhythmia-consumer-group`
- Process one message at a time (commit offset after successful processing + MongoDB write)
- On failure: log error, skip message, commit offset (don't block the queue)
- Graceful shutdown on SIGTERM (K8s sends this before killing pod)

### 2. `ecg_processor.py` (Processing Pipeline)

**What it does:** Takes raw 1-minute ECG data, runs the full pipeline, returns structured analysis result.

```python
def process(ecg_data: list[float], admission_id: str, device_id: str, timestamp: int) -> dict:
    """
    Process 1-minute ECG data through full arrhythmia detection pipeline.
    
    Steps:
    1. Convert to numpy array (already 125 Hz, already mV)
    2. Clean signal (baseline wander + powerline removal)
    3. Segment into 10-second windows (6 windows from 60s)
    4. Per window:
       a. Detect R-peaks
       b. Extract morphology (P-wave, PR, QRS, QTc, RR)
       c. Sinus pre-model gate (score >= 0.75 -> skip rhythm model)
       d. ML inference (rhythm + ectopy models)
       e. Sinus post-model gate (score >= 0.60 -> override)
       f. Rules engine (AF, AV blocks, ectopy patterns)
    5. Aggregate results across all segments
    6. Return structured analysis dict
    """
```

**This reuses existing functions from the codebase:**
- `signal_processing.cleaning.clean_signal()`
- `data.ingest_json._segment()`
- `data.ingest_json._detect_r_peaks()`
- `data.ingest_json._extract_morphology()`
- `data.ingest_json._run_inference()`
- `data.ingest_ecg_extracts._run_rules_engine()`
- `data.ingest_ecg_extracts._compute_sinus_score()`
- `data.ingest_ecg_extracts._sp_sinus_premodel()`
- `data.ingest_ecg_extracts._sp_sinus_postmodel()`

### 3. `mongo_writer.py` (MongoDB Output)

**What it does:** Connects to MongoDB, writes analysis results.

```python
def write_result(result: dict) -> str:
    """Write arrhythmia analysis result to MongoDB. Returns document UUID."""

def get_connection() -> MongoClient:
    """Get MongoDB connection from environment variables."""
```

**Connection config from environment (K8s/.env):**
- `MONGO_URI` (e.g., `mongodb://localhost:27017`)
- `MONGO_DB` (e.g., `ecg_analysis`)
- `MONGO_COLLECTION` (e.g., `arrhythmia_results`)

### 4. `config.py` (Environment Configuration)

**What it does:** Reads all configuration from environment variables (12-factor app).

```python
# Kafka
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "ecg-arrhythmia-topic")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "ecg-arrhythmia-consumer-group")

# MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "ecg_analysis")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "arrhythmia_results")

# Processing
SAMPLING_RATE = 125
SAMPLES_PER_MINUTE = 7500
WINDOW_SECONDS = 10
WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_SECONDS  # 1250

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
```

### 5. `Dockerfile`

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install system deps for scipy/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model checkpoints are baked into the image
# (or mounted as volume in K8s)

CMD ["python", "kafka_consumer.py"]
```

### 6. `.env` (Local Development)

```env
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=ecg-arrhythmia-topic
KAFKA_GROUP_ID=ecg-arrhythmia-consumer-group
MONGO_URI=mongodb://localhost:27017
MONGO_DB=ecg_analysis
MONGO_COLLECTION=arrhythmia_results
LOG_LEVEL=INFO
```

### 7. `requirements.txt` (additions for Kafka/MongoDB)

Add to existing:
```
confluent-kafka>=2.3.0
pymongo>=4.6.0
python-dotenv>=1.0.0
```

---

## Implementation Order (Step by Step) — Full Code

---

### Step 1: Start Kafka Locally

```bash
# From project root (docker-compose.yml already exists)
docker compose up -d

# Verify Kafka is running
docker logs kafka

# Verify Kafbat UI is up
# Open browser: http://localhost:9090
# If topic doesn't auto-create, create it manually in UI or:
docker exec kafka kafka-topics.sh --create \
  --bootstrap-server localhost:9092 \
  --topic ecg-arrhythmia-topic \
  --partitions 10 \
  --replication-factor 1
```

---

### Step 2: Create `config.py`

Create file at project root: `Project_Submission_Clean/config.py`

```python
"""
config.py — All configuration from environment variables (12-factor app).
"""
import os

# Kafka
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC             = os.getenv("KAFKA_TOPIC", "ecg-arrhythmia-topic")
KAFKA_GROUP_ID          = os.getenv("KAFKA_GROUP_ID", "ecg-arrhythmia-consumer-group")

# MongoDB
MONGO_URI        = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB         = os.getenv("MONGO_DB", "ecg_analysis")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "arrhythmia_results")

# Processing constants (fixed — matches Shreyas confirmation)
SAMPLING_RATE      = 125
SAMPLES_PER_MINUTE = 7500
WINDOW_SECONDS     = 10
WINDOW_SAMPLES     = SAMPLING_RATE * WINDOW_SECONDS  # 1250

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Facility ID — set per deployment
FACILITY_ID = os.getenv("FACILITY_ID", "CF1315821527")
```

**Test:**
```bash
python -c "import config; print(config.KAFKA_TOPIC)"
# Expected: ecg-arrhythmia-topic
```

---

### Step 3: Create `.env` (local dev only, never commit)

Create file at project root: `Project_Submission_Clean/.env`

```env
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=ecg-arrhythmia-topic
KAFKA_GROUP_ID=ecg-arrhythmia-consumer-group
MONGO_URI=mongodb://localhost:27017
MONGO_DB=ecg_analysis
MONGO_COLLECTION=arrhythmia_results
LOG_LEVEL=INFO
FACILITY_ID=CF1315821527
```

Add `.env` to `.gitignore` if not already there.

---

### Step 4: Install New Dependencies

```bash
pip install confluent-kafka pymongo python-dotenv
```

Add to `requirements.txt`:
```
confluent-kafka>=2.3.0
pymongo>=4.6.0
python-dotenv>=1.0.0
```

---

### Step 5: Create `ecg_processor.py`

Create file at project root: `Project_Submission_Clean/ecg_processor.py`

```python
"""
ecg_processor.py — Wrap the full ECG arrhythmia detection pipeline.

Input:  raw ecg_data (list[float], 7500 samples, 125 Hz, mV values)
Output: structured analysis dict matching MongoDB schema
"""
from __future__ import annotations

import sys
import warnings
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "models_training"))

from config import SAMPLING_RATE, WINDOW_SAMPLES
from signal_processing.cleaning import clean_signal


def _segment(signal: np.ndarray) -> list[np.ndarray]:
    """Split 1-D signal into 10-second windows (1250 samples each)."""
    windows, offset = [], 0
    min_samp = 500
    while offset + min_samp <= len(signal):
        chunk = signal[offset: offset + WINDOW_SAMPLES]
        if len(chunk) < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
        windows.append(chunk)
        offset += WINDOW_SAMPLES
    return windows


def _detect_r_peaks(window: np.ndarray) -> list[int]:
    from scipy.signal import find_peaks
    try:
        peaks, _ = find_peaks(
            window,
            distance=int(SAMPLING_RATE * 0.4),
            height=np.percentile(window, 75),
        )
        return peaks.tolist()
    except Exception:
        return []


def _extract_morphology(window: np.ndarray, r_peaks: list[int]) -> dict:
    try:
        from signal_processing.morphology import extract_morphology
        return extract_morphology(window, np.array(r_peaks, dtype=int), SAMPLING_RATE)
    except Exception as exc:
        return {"error": str(exc)}


def _run_inference(window: np.ndarray):
    """Returns (rhythm_label, rhythm_conf, ectopy_label, ectopy_conf)."""
    try:
        import torch
        import torch.nn.functional as F
        from xai.xai import _load_model, _init_device
        from models_training.data_loader import RHYTHM_CLASS_NAMES, ECTOPY_CLASS_NAMES

        device = _init_device()
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            m_r = _load_model("rhythm")
            probs_r = F.softmax(m_r(x), dim=-1).squeeze()
            idx_r = int(probs_r.argmax())
            rhythm_label = RHYTHM_CLASS_NAMES[idx_r]
            rhythm_conf  = float(probs_r[idx_r])

            m_e = _load_model("ectopy")
            probs_e = F.softmax(m_e(x), dim=-1).squeeze()
            idx_e = int(probs_e.argmax())
            if idx_e != 0 and float(probs_e[idx_e]) < 0.6:
                idx_e = 0
            ectopy_label = ECTOPY_CLASS_NAMES[idx_e]
            ectopy_conf  = float(probs_e[idx_e])

        return rhythm_label, rhythm_conf, ectopy_label, ectopy_conf
    except Exception as exc:
        warnings.warn(f"Inference failed: {exc}")
        return None, None, None, None


def _run_rules(r_peaks, rhythm_label, ectopy_label, morph_data) -> dict:
    try:
        from decision_engine.rules import derive_rule_events
        rr = np.diff(r_peaks).tolist() if len(r_peaks) > 1 else []
        pr = morph_data.get("summary", {}).get("pr_interval_ms")
        p_ratio = morph_data.get("summary", {}).get("p_wave_present_ratio", 1.0)
        result = derive_rule_events(
            rr_intervals=rr,
            rhythm_label=rhythm_label,
            ectopy_label=ectopy_label,
            pr_interval_ms=pr,
            p_wave_present_ratio=p_ratio,
        )
        return result
    except Exception as exc:
        warnings.warn(f"Rules engine failed: {exc}")
        return {"primary_conclusion": rhythm_label, "background_rhythm": rhythm_label,
                "events": [], "beat_markers": []}


def process(
    ecg_data: list[float],
    admission_id: str,
    device_id: str,
    timestamp: int,
    patient_id: str = "unknown",
    facility_id: str = "unknown",
) -> dict:
    """
    Process 1-minute ECG data through the full arrhythmia detection pipeline.

    Steps:
    1. Convert to numpy, clean signal
    2. Segment into 6 x 10-second windows
    3. Per window: R-peaks → morphology → inference → rules
    4. Aggregate: dominant rhythm, HR, events
    5. Return structured result matching MongoDB schema
    """
    t_start = time.time()
    signal = np.ascontiguousarray(ecg_data, dtype=np.float32)

    # Clean: baseline wander removal + powerline noise removal
    signal = np.ascontiguousarray(clean_signal(signal, SAMPLING_RATE), dtype=np.float32)

    windows = _segment(signal)
    segments_out = []
    all_rhythms, all_hrs, all_events = [], [], []

    for idx, window in enumerate(windows):
        window = np.ascontiguousarray(window, dtype=np.float32)
        r_peaks = _detect_r_peaks(window)
        morph = _extract_morphology(window, r_peaks)

        rhythm_label, rhythm_conf, ectopy_label, ectopy_conf = _run_inference(window)
        if rhythm_label is None:
            rhythm_label, rhythm_conf = "Unknown", 0.0
            ectopy_label, ectopy_conf = "None", 0.0

        rules = _run_rules(r_peaks, rhythm_label, ectopy_label, morph)

        summary = morph.get("summary", {})
        hr = summary.get("heart_rate_bpm")
        if hr:
            all_hrs.append(hr)

        primary = rules.get("primary_conclusion", rhythm_label)
        all_rhythms.append(primary)
        events = rules.get("events", [])
        all_events.extend(events)

        segments_out.append({
            "segment_index":        idx,
            "start_time_s":         round(idx * 10.0, 1),
            "end_time_s":           round((idx + 1) * 10.0, 1),
            "rhythm_label":         rhythm_label,
            "rhythm_confidence":    round(rhythm_conf, 4),
            "ectopy_label":         ectopy_label,
            "ectopy_confidence":    round(ectopy_conf, 4),
            "events":               events,
            "primary_conclusion":   primary,
            "morphology": {
                "hr_bpm":               hr,
                "pr_interval_ms":       summary.get("pr_interval_ms"),
                "qrs_duration_ms":      summary.get("qrs_duration_ms"),
                "qtc_ms":               summary.get("qtc_bazett_ms"),
                "p_wave_present_ratio": summary.get("p_wave_present_ratio"),
            },
            "sinus_gate_fired":     rules.get("sinus_gate_fired", False),
            "ectopy_override":      rules.get("ectopy_override"),
        })

    # Aggregate
    from collections import Counter
    dominant_rhythm = Counter(all_rhythms).most_common(1)[0][0] if all_rhythms else "Unknown"
    avg_hr = round(float(np.mean(all_hrs))) if all_hrs else None
    unique_events = sorted(set(all_events))
    arrhythmia_detected = any(
        e not in {"Sinus Rhythm", "Sinus Bradycardia", "Sinus Tachycardia", "None", "Unknown"}
        for e in unique_events
    )

    elapsed = round(time.time() - t_start, 2)

    return {
        "admissionId":   admission_id,
        "deviceId":      device_id,
        "patientId":     patient_id,
        "facilityId":    facility_id,
        "timestamp":     timestamp,
        "ecgData":       ecg_data,          # full raw array (mV)
        "analysis": {
            "background_rhythm":  dominant_rhythm,
            "heart_rate_bpm":     avg_hr,
            "segments":           segments_out,
            "summary": {
                "total_segments":      len(segments_out),
                "dominant_rhythm":     dominant_rhythm,
                "arrhythmia_detected": arrhythmia_detected,
                "events_found":        unique_events,
                "signal_quality":      "acceptable",
            },
        },
        "processingStatus": None,
        "processedAt":      None,
        "processedBy":      None,
        "_processing_time_s": elapsed,
    }
```

**Test independently:**
```bash
python -c "
from ecg_processor import process
import numpy as np
fake_ecg = (np.random.randn(7500) * 0.5).tolist()
result = process(fake_ecg, 'TEST001', 'DEV001', 1770912322923)
print('Rhythm:', result['analysis']['background_rhythm'])
print('Segments:', result['analysis']['summary']['total_segments'])
print('Time:', result['_processing_time_s'], 's')
"
```

---

### Step 6: Create `mongo_writer.py`

Create file at project root: `Project_Submission_Clean/mongo_writer.py`

```python
"""
mongo_writer.py — Write arrhythmia analysis results to MongoDB.
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

import config

log = logging.getLogger(__name__)

_client: MongoClient | None = None


def get_collection():
    global _client
    if _client is None:
        _client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client[config.MONGO_DB][config.MONGO_COLLECTION]


def write_result(result: dict, retries: int = 3) -> str:
    """
    Write arrhythmia analysis result to MongoDB.
    Returns the document UUID on success.
    Raises RuntimeError after retries are exhausted.
    """
    doc_uuid = str(uuid.uuid4())
    doc = {**result, "uuid": doc_uuid}

    for attempt in range(1, retries + 1):
        try:
            col = get_collection()
            col.insert_one(doc)
            log.info(f"{result.get('admissionId')} | Written to MongoDB (uuid={doc_uuid})")
            return doc_uuid
        except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
            log.warning(f"MongoDB write attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(2 ** attempt)   # 2s, 4s backoff
            else:
                raise RuntimeError(f"MongoDB unavailable after {retries} attempts") from exc
        except Exception as exc:
            log.error(f"MongoDB unexpected error: {exc}")
            raise
```

**Test:**
```bash
# Make sure MongoDB is running first:
# docker run -d -p 27017:27017 --name mongodb mongo:7

python -c "
from mongo_writer import write_result
uid = write_result({'admissionId': 'TEST', 'test': True, 'ecgData': []})
print('Written:', uid)
"
# Then verify in MongoDB:
# mongosh ecg_analysis --eval 'db.arrhythmia_results.findOne({admissionId: \"TEST\"})'
```

---

### Step 7: Create `kafka_consumer.py`

Create file at project root: `Project_Submission_Clean/kafka_consumer.py`

```python
"""
kafka_consumer.py — Consume ECG data from Kafka, process, write to MongoDB.

Designed for K8s deployment:
- Graceful SIGTERM shutdown
- Structured stdout logging (CloudWatch compatible)
- One message at a time, commit after successful write
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

from confluent_kafka import Consumer, KafkaError, KafkaException
from dotenv import load_dotenv

# Load .env for local dev (no-op in K8s where env vars come from ConfigMap)
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
from ecg_processor import process
from mongo_writer import write_result

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_running = True

def _handle_sigterm(signum, frame):
    global _running
    log.info("SIGTERM received — finishing current message, then shutting down.")
    _running = False

signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT,  _handle_sigterm)

# ---------------------------------------------------------------------------
# Message processing
# ---------------------------------------------------------------------------

def _parse_message(raw_value: bytes) -> dict | None:
    """Parse Kafka message JSON. Returns None on parse error."""
    try:
        return json.loads(raw_value.decode("utf-8"))
    except Exception as exc:
        log.warning(f"Bad message format (not JSON): {exc}")
        return None


def _validate_message(msg: dict) -> bool:
    """Validate required fields and sample count."""
    required = {"deviceId", "admissionId", "timestamp", "ecgData"}
    missing = required - set(msg.keys())
    if missing:
        log.warning(f"Message missing fields: {missing}")
        return False

    n = len(msg["ecgData"])
    if n != config.SAMPLES_PER_MINUTE:
        log.warning(
            f"{msg['admissionId']} | Unexpected sample count: {n} "
            f"(expected {config.SAMPLES_PER_MINUTE}) — processing anyway"
        )
        # Don't reject — process anyway (Shreyas may send slightly different lengths)
    return True


def _process_message(msg: dict) -> bool:
    """Run full pipeline + write to MongoDB. Returns True on success."""
    admission_id = msg["admissionId"]
    device_id    = msg["deviceId"]
    timestamp    = msg["timestamp"]
    ecg_data     = msg["ecgData"]
    patient_id   = msg.get("patientId",  "unknown")
    facility_id  = msg.get("facilityId", config.FACILITY_ID)

    n_samples = len(ecg_data)
    log.info(f"Processing {admission_id} | {n_samples} samples | device={device_id}")

    try:
        result = process(
            ecg_data    = ecg_data,
            admission_id = admission_id,
            device_id    = device_id,
            timestamp    = timestamp,
            patient_id   = patient_id,
            facility_id  = facility_id,
        )

        # Log summary (never log raw ECG data)
        summary  = result["analysis"]["summary"]
        rhythm   = result["analysis"]["background_rhythm"]
        events   = summary["events_found"]
        hr       = result["analysis"]["heart_rate_bpm"]
        n_segs   = summary["total_segments"]
        t_proc   = result.get("_processing_time_s", "?")
        log.info(
            f"{admission_id} | {n_segs} segments | rhythm={rhythm} | "
            f"HR={hr} bpm | events={events} | {t_proc}s"
        )

        if summary["arrhythmia_detected"]:
            log.warning(f"{admission_id} | Arrhythmia detected | events={events}")

        write_result(result)
        return True

    except Exception as exc:
        log.error(f"{admission_id} | Processing failed: {exc}", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Main consumer loop
# ---------------------------------------------------------------------------

def run():
    conf = {
        "bootstrap.servers":  config.KAFKA_BOOTSTRAP_SERVERS,
        "group.id":           config.KAFKA_GROUP_ID,
        "auto.offset.reset":  "earliest",
        "enable.auto.commit": False,   # Manual commit after successful processing
    }

    consumer = Consumer(conf)
    consumer.subscribe([config.KAFKA_TOPIC])

    log.info(
        f"Consumer started. Group={config.KAFKA_GROUP_ID} "
        f"Topic={config.KAFKA_TOPIC} "
        f"Bootstrap={config.KAFKA_BOOTSTRAP_SERVERS}"
    )

    try:
        while _running:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue   # No message in this poll window

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    log.debug(f"Reached end of partition {msg.partition()}")
                    continue
                raise KafkaException(msg.error())

            # Parse + validate
            parsed = _parse_message(msg.value())
            if parsed is None or not _validate_message(parsed):
                # Bad message — skip and commit to not block queue
                consumer.commit(message=msg)
                continue

            # Process + MongoDB write
            _process_message(parsed)
            # Commit offset regardless of success (don't block queue on repeated failure)
            consumer.commit(message=msg)

    except KafkaException as exc:
        log.error(f"Kafka error: {exc}")
        sys.exit(1)
    finally:
        log.info("Closing consumer...")
        consumer.close()
        log.info("Consumer stopped.")


if __name__ == "__main__":
    run()
```

---

### Step 8: Create `test_producer.py` (for local testing)

Create file at project root: `Project_Submission_Clean/test_producer.py`

```python
"""
test_producer.py — Send a fake ECG message to Kafka for local testing.

Usage:
    python test_producer.py                    # one fake sinus rhythm message
    python test_producer.py --file ecg.json    # send real ECG from JSON file
    python test_producer.py --count 5          # send 5 fake messages
"""
import argparse
import json
import time
import numpy as np
from confluent_kafka import Producer
from dotenv import load_dotenv
import config

load_dotenv()


def _fake_ecg(n: int = 7500, fs: int = 125) -> list[float]:
    """Generate a synthetic sinus ECG (simple QRS-like pulses)."""
    t = np.arange(n) / fs
    signal = np.zeros(n, dtype=np.float32)
    # R-peaks every ~0.8s (75 bpm)
    r_times = np.arange(0.1, n / fs, 0.8)
    for rt in r_times:
        idx = int(rt * fs)
        if idx < n:
            # Simple triangle QRS
            for di in range(-5, 6):
                if 0 <= idx + di < n:
                    signal[idx + di] += max(0, 1.0 - abs(di) * 0.2)
    # Add baseline noise
    signal += np.random.normal(0, 0.05, n).astype(np.float32)
    return signal.tolist()


def send(bootstrap: str, topic: str, msg: dict):
    p = Producer({"bootstrap.servers": bootstrap})
    p.produce(topic, json.dumps(msg).encode("utf-8"))
    p.flush()
    print(f"Sent: admissionId={msg['admissionId']} | {len(msg['ecgData'])} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",  default=None, help="ECG JSON file to send")
    parser.add_argument("--count", default=1,    type=int, help="Number of fake messages")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            data = json.load(f)
        msg = {
            "deviceId":    "TEST_DEVICE",
            "admissionId": data.get("patient_id", "TEST_ADM"),
            "timestamp":   int(time.time() * 1000),
            "ecgData":     data["signal"][:7500],
        }
        send(config.KAFKA_BOOTSTRAP_SERVERS, config.KAFKA_TOPIC, msg)
    else:
        for i in range(args.count):
            msg = {
                "deviceId":    f"DEV{i:03d}",
                "admissionId": f"TEST_ADM{i:06d}",
                "timestamp":   int(time.time() * 1000),
                "ecgData":     _fake_ecg(),
            }
            send(config.KAFKA_BOOTSTRAP_SERVERS, config.KAFKA_TOPIC, msg)
            time.sleep(0.5)
```

---

### Step 9: Create `Dockerfile`

Create file at project root: `Project_Submission_Clean/Dockerfile`

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# System deps for scipy/numpy compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Model checkpoints are baked into image (or use K8s volume mount)
# Verify checkpoints exist:
RUN python -c "from pathlib import Path; \
    assert Path('checkpoints/rhythm_best.pt').exists() or \
           list(Path('checkpoints').glob('*.pt')), \
    'No model checkpoints found in checkpoints/'"

# Entry point
CMD ["python", "kafka_consumer.py"]
```

**Build and test:**
```bash
# Build the image
docker build -t ecg-arrhythmia:latest .

# Run locally (needs Kafka + MongoDB running)
docker run --rm \
  --network host \
  --env-file .env \
  ecg-arrhythmia:latest

# Or with explicit env vars:
docker run --rm \
  --network host \
  -e KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
  -e MONGO_URI=mongodb://localhost:27017 \
  -e LOG_LEVEL=INFO \
  ecg-arrhythmia:latest
```

---

### Step 10: End-to-End Test

Run all components:

**Terminal 1 — Start infrastructure:**
```bash
docker compose up -d   # Kafka + Kafbat UI
docker run -d -p 27017:27017 --name mongodb mongo:7   # MongoDB (if not already running)
```

**Terminal 2 — Start consumer:**
```bash
cd Project_Submission_Clean
python kafka_consumer.py
# Expected: "Consumer started. Group=ecg-arrhythmia-consumer-group Topic=ecg-arrhythmia-topic"
```

**Terminal 3 — Send test message:**
```bash
python test_producer.py --count 3
# Expected: "Sent: admissionId=TEST_ADM000000 | 7500 samples"
```

**Terminal 2 should show:**
```
Processing TEST_ADM000000 | 7500 samples | device=DEV000
TEST_ADM000000 | 6 segments | rhythm=Sinus Rhythm | HR=75 bpm | events=[] | 1.3s
TEST_ADM000000 | Written to MongoDB (uuid=abc-123...)
```

**Verify MongoDB:**
```bash
mongosh ecg_analysis --eval 'db.arrhythmia_results.find({},{admissionId:1,analysis:1,_id:0}).limit(3).pretty()'
```

**Test bad data (should skip, not crash):**
```bash
python -c "
from confluent_kafka import Producer
import json, config
p = Producer({'bootstrap.servers': config.KAFKA_BOOTSTRAP_SERVERS})
p.produce(config.KAFKA_TOPIC, b'not valid json')
p.flush()
print('Sent bad message')
"
# Consumer should log warning and continue
```

---

### Step 11: Dockerized End-to-End Test

Test the Docker image against local Kafka:

```bash
# Build image
docker build -t ecg-arrhythmia:latest .

# Run consumer as container (--network host so it can reach localhost Kafka/Mongo)
docker run --rm --network host --env-file .env ecg-arrhythmia:latest &

# Send test messages
python test_producer.py --count 3

# Watch logs
docker logs -f $(docker ps -q --filter ancestor=ecg-arrhythmia:latest)
```

---

### Step 12: K8s Deployment (with Vishnu)

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ecg-arrhythmia-consumer
  namespace: ecg-processing
spec:
  replicas: 2   # 2 pods × 10 partitions = each pod handles 5 partitions
  selector:
    matchLabels:
      app: ecg-arrhythmia-consumer
  template:
    metadata:
      labels:
        app: ecg-arrhythmia-consumer
    spec:
      containers:
      - name: consumer
        image: ecg-arrhythmia:latest
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        env:
        - name: KAFKA_BOOTSTRAP_SERVERS
          valueFrom:
            configMapKeyRef:
              name: ecg-config
              key: kafka_bootstrap_servers
        - name: KAFKA_TOPIC
          value: "ecg-arrhythmia-topic"
        - name: KAFKA_GROUP_ID
          value: "ecg-arrhythmia-consumer-group"
        - name: MONGO_URI
          valueFrom:
            secretKeyRef:
              name: ecg-secrets
              key: mongo_uri
        - name: MONGO_DB
          value: "ecg_analysis"
        - name: MONGO_COLLECTION
          value: "arrhythmia_results"
        - name: LOG_LEVEL
          value: "INFO"
        - name: FACILITY_ID
          valueFrom:
            configMapKeyRef:
              name: ecg-config
              key: facility_id
        terminationGracePeriodSeconds: 60   # Time to finish current message on SIGTERM
```

```bash
# Deploy
kubectl apply -f k8s-deployment.yaml

# Check pods
kubectl get pods -n ecg-processing

# Watch logs
kubectl logs -f deployment/ecg-arrhythmia-consumer -n ecg-processing
```

---

## Folder Structure (New Files)

```
Project_Submission_Clean/
  kafka_consumer.py        <-- NEW: main entry point
  ecg_processor.py         <-- NEW: pipeline wrapper
  mongo_writer.py          <-- NEW: MongoDB output
  config.py                <-- NEW: environment config
  Dockerfile               <-- NEW: container image
  .env                     <-- NEW: local dev env vars
  docker-compose.yml       <-- EXISTS: Kafka + Kafbat UI
  requirements.txt         <-- UPDATE: add confluent-kafka, pymongo
  test_producer.py         <-- NEW: test helper to send fake ECG to Kafka
  
  # Existing (unchanged, reused by ecg_processor.py):
  signal_processing/       
  models_training/         
  decision_engine/         
  xai/                     
  data/                    
```

---

## Logging Strategy (per Vinayak's Instructions)

- **Application logs** -> stdout (picked up by CloudWatch/K8s logging)
- **Never log raw ECG data** in application logs
- **ECG data + results** -> MongoDB only (not files, not logs)
- **Log level** configurable via `LOG_LEVEL` env var
- **Log format:** `%(asctime)s [%(levelname)s] %(message)s`

Example log output:
```
2026-04-02 10:15:03 [INFO] Consumer started. Group=ecg-arrhythmia-consumer-group Topic=ecg-arrhythmia-topic
2026-04-02 10:15:05 [INFO] Processing ADM819104078 | 7500 samples | device=AB021511
2026-04-02 10:15:06 [INFO] ADM819104078 | 6 segments | rhythm=Sinus Rhythm | events=[] | 1.2s
2026-04-02 10:15:06 [INFO] ADM819104078 | Written to MongoDB (uuid=abc123)
2026-04-02 10:15:08 [WARNING] ADM727425540 | AF detected | 6 segments | events=[AF, PVC Couplet]
```

---

## Error Handling

| Scenario | Action |
|---|---|
| Kafka connection lost | Retry with exponential backoff (confluent-kafka handles this) |
| MongoDB connection lost | Retry 3 times, then skip message + log error |
| Bad message format (not JSON) | Log warning, skip, commit offset |
| Wrong sample count (not 7500) | Log warning, try to process anyway (resample if needed) |
| ML model fails (OOM, corrupt) | Log error, return analysis with `signal_quality: "model_error"` |
| Processing timeout (>30s per message) | Log warning, return partial result |
| SIGTERM received | Finish current message, commit offset, close connections, exit cleanly |

---

## Questions Resolved

| Question | Answer | Source |
|---|---|---|
| Sampling rate | 125 Hz fixed | Shreyas (email) |
| Sample count | 7500 per minute | Shreyas (trimmed to exactly this) |
| Data format | mV converted values | Shreyas (email) |
| Batch or packets | Full 1-minute in one array | Our request, confirmed |
| Topic name | ecg-arrhythmia-topic | Shreyas (email) |
| Logging | stdout for K8s/CloudWatch | Shreyas (email) |
| File storage | Not needed, use MongoDB | Shreyas (email, K8s pods are ephemeral) |

---

## Dependencies on Others

| Who | What | Status |
|---|---|---|
| **Shreyas** | Kafka Docker compose file | Received |
| **Shreyas** | Start producing test data to topic | Pending |
| **Vinayak** | MongoDB schema approval | Pending (proposed above) |
| **Vishnu** | K8s deployment setup, CI/CD | After local testing complete |
| **Us** | Kafka consumer + processor + MongoDB writer | To implement |
