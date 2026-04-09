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
