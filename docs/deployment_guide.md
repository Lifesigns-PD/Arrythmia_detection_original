# ECG Arrhythmia Service — Deployment & Run Guide

## Prerequisites
- Docker Desktop installed and running
- Python 3.11+ (for test producer only)

---

## Quick Start

### Step 1: Start Kafka
```bash
docker compose up -d
```
This starts:
- **Kafka** (port 9092) — KRaft mode, no ZooKeeper
- **Kafbat UI** (port 9090) — web UI to manage topics at http://localhost:9090

### Step 2: Start MongoDB
```bash
docker run -d --name mongo -p 27017:27017 mongo:7
```

### Step 3: Build the Consumer Image
```bash
docker build -t ecg-consumer .
```

### Step 4: Run the Consumer (5 threads)
```bash
docker run -d --name ecg-consumer \
  --network host \
  -e KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
  -e MONGO_URI=mongodb://localhost:27017 \
  ecg-consumer
```

### Step 5: Send Test Messages
```bash
pip install confluent-kafka python-dotenv
python test_producer.py --count 5
```

### Step 6: Verify Output
Check consumer logs:
```bash
docker logs ecg-consumer
```

Expected output:
```
[INFO] [MainThread] Consumer started. Group=ecg-arrhythmia-consumer-group Topic=ecg-arrhythmia-topic Threads=5
[INFO] [ecg-worker_0] Processing TEST_ADM000000 | 7500 samples | device=DEV000
[INFO] [ecg-worker_0] TEST_ADM000000 | 6 segments | rhythm=AF | HR=91 bpm | events=['AF'] | 8.55s
[INFO] [ecg-worker_0] TEST_ADM000000 | Written to MongoDB (uuid=...)
```

Check MongoDB:
```bash
python -c "from pymongo import MongoClient; c=MongoClient(); print(c.ecg_analysis.arrhythmia_results.count_documents({}))"
```

### Step 7: Stop Everything
```bash
docker stop ecg-consumer mongo
docker compose down
```

---

## Configuration

All settings via environment variables (see `.env.template`):

| Variable | Default | Description |
|----------|---------|-------------|
| KAFKA_BOOTSTRAP_SERVERS | localhost:9092 | Kafka broker address |
| KAFKA_TOPIC | ecg-arrhythmia-topic | Kafka topic to consume |
| KAFKA_GROUP_ID | ecg-arrhythmia-consumer-group | Consumer group ID |
| MONGO_URI | mongodb://localhost:27017 | MongoDB connection string |
| MONGO_DB | ecg_analysis | MongoDB database name |
| MONGO_COLLECTION | arrhythmia_results | MongoDB collection name |
| CONSUMER_THREADS | 5 | Number of parallel processing threads |
| LOG_LEVEL | INFO | Logging level |
| FACILITY_ID | CF1315821527 | Default facility ID |

---

## Kafka Topics

### Current Topic
- `ecg-arrhythmia-topic` (10 partitions, auto-created)

### Adding New Topics
Kafka has `auto.create.topics.enable=true`, so topics are created automatically when a producer/consumer references them.

**Option 1 — Via Kafbat UI:**
1. Open http://localhost:9090
2. Go to Topics > Create Topic
3. Set name, partitions, retention

**Option 2 — Via environment variable:**
```bash
docker run -d --name ecg-consumer \
  -e KAFKA_TOPIC=my-new-topic \
  ecg-consumer
```

**Option 3 — Via Kafka CLI inside container:**
```bash
docker exec -it kafka /opt/kafka/bin/kafka-topics.sh \
  --create --topic my-new-topic \
  --partitions 10 \
  --replication-factor 1 \
  --bootstrap-server localhost:9092
```

---

## MongoDB Output Schema

Each processed ECG message is stored as:
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
    "heart_rate_bpm": 72,
    "segments": [
      {
        "segment_index": 0,
        "start_time_s": 0.0,
        "end_time_s": 10.0,
        "rhythm_label": "Sinus Rhythm",
        "rhythm_confidence": 0.95,
        "ectopy_label": "None",
        "ectopy_confidence": 0.98,
        "events": [],
        "primary_conclusion": "Sinus Rhythm",
        "morphology": {
          "hr_bpm": 72,
          "pr_interval_ms": 160,
          "qrs_duration_ms": 90,
          "qtc_ms": 410,
          "p_wave_present_ratio": 1.0
        }
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

UI picks up documents where `processingStatus` is `null` (oldest timestamp first).

---

## Architecture

```
Kafka (ecg-arrhythmia-topic)
    |
    v
kafka_consumer.py (5 threads, ThreadPoolExecutor)
    |
    v
ecg_processor.py
    |-- signal_processing/cleaning.py  (baseline wander + powerline removal)
    |-- _segment()                     (6 x 10-second windows)
    |-- _detect_r_peaks()              (scipy find_peaks)
    |-- signal_processing/morphology.py (PR, QRS, QTc, P-wave)
    |-- xai/xai.py                     (rhythm + ectopy ML models)
    |-- decision_engine/rules.py       (rules engine)
    |
    v
mongo_writer.py --> MongoDB (ecg_analysis.arrhythmia_results)
```

---

## Logging

- All logs go to **stdout** (CloudWatch / K8s logging picks them up)
- No local log files (per Shreyas: pods are ephemeral)
- Thread name included in logs: `[ecg-worker_0]`, `[ecg-worker_1]`, etc.
- Arrhythmia detections logged as WARNING level

---

## Kubernetes Deployment Notes

- Consumer runs as a pod with env vars from ConfigMap
- Models are baked into the Docker image (or mount via K8s volume)
- Graceful shutdown via SIGTERM signal handler
- MongoDB connection via K8s service DNS
- Scale horizontally by running multiple pod replicas (Kafka rebalances partitions)
