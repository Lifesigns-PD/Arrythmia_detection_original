"""
kafka_consumer.py — Consume ECG data from Kafka, process, write to MongoDB.

Designed for K8s deployment:
- Multi-threaded: N worker threads (default 5) for parallel processing
- Graceful SIGTERM shutdown
- Structured stdout logging (CloudWatch compatible)
- Manual offset commit after successful write
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue, Empty

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
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
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
    log.info("SIGTERM received — finishing in-flight messages, then shutting down.")
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
# Main consumer loop with thread pool
# ---------------------------------------------------------------------------

def run():
    num_threads = config.CONSUMER_THREADS
    conf = {
        "bootstrap.servers":  config.KAFKA_BOOTSTRAP_SERVERS,
        "group.id":           config.KAFKA_GROUP_ID,
        "auto.offset.reset":  "earliest",
        "enable.auto.commit": False,
    }

    consumer = Consumer(conf)
    consumer.subscribe([config.KAFKA_TOPIC])

    log.info(
        f"Consumer started. Group={config.KAFKA_GROUP_ID} "
        f"Topic={config.KAFKA_TOPIC} "
        f"Bootstrap={config.KAFKA_BOOTSTRAP_SERVERS} "
        f"Threads={num_threads}"
    )

    # Lock for consumer.commit() — confluent-kafka Consumer is not thread-safe
    commit_lock = threading.Lock()

    def _worker(parsed: dict, kafka_msg):
        """Process one message and commit its offset."""
        _process_message(parsed)
        with commit_lock:
            consumer.commit(message=kafka_msg)

    try:
        with ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix="ecg-worker") as pool:
            futures = {}

            while _running:
                # Don't fetch more if pool is saturated
                if len(futures) >= num_threads:
                    # Wait for at least one to finish
                    done = next(as_completed(futures), None)
                    if done:
                        exc = done.exception()
                        if exc:
                            log.error(f"Worker thread error: {exc}", exc_info=exc)
                        futures.pop(done, None)

                msg = consumer.poll(timeout=1.0)

                if msg is None:
                    # Drain completed futures while idle
                    _drain_futures(futures)
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        log.debug(f"Reached end of partition {msg.partition()}")
                        continue
                    raise KafkaException(msg.error())

                # Parse + validate
                parsed = _parse_message(msg.value())
                if parsed is None or not _validate_message(parsed):
                    consumer.commit(message=msg)
                    continue

                # Submit to thread pool
                future = pool.submit(_worker, parsed, msg)
                futures[future] = parsed.get("admissionId", "unknown")

                # Clean up completed futures
                _drain_futures(futures)

            # Shutdown: wait for in-flight messages
            log.info(f"Shutting down — waiting for {len(futures)} in-flight messages...")
            for future in as_completed(futures):
                exc = future.exception()
                if exc:
                    log.error(f"Worker thread error during shutdown: {exc}", exc_info=exc)

    except KafkaException as exc:
        log.error(f"Kafka error: {exc}")
        sys.exit(1)
    finally:
        log.info("Closing consumer...")
        consumer.close()
        log.info("Consumer stopped.")


def _drain_futures(futures: dict):
    """Remove completed futures from the tracking dict."""
    done_keys = [f for f in futures if f.done()]
    for f in done_keys:
        exc = f.exception()
        if exc:
            log.error(f"Worker thread error: {exc}", exc_info=exc)
        futures.pop(f)


if __name__ == "__main__":
    run()
