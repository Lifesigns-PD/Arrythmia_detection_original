"""
gRPC Test Client - Simulates an ECG device streaming data.
===========================================================
Generates synthetic ECG-like waveforms and streams them to the gRPC server.
Prints any ArrhythmiaAlert responses received.

Usage:
    python scripts/grpc_test_client.py
    python scripts/grpc_test_client.py --port 50052 --device DEV_002
"""

import sys
import os
import time
import math
import random
import argparse
import threading
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import grpc
import numpy as np
from grpc_gen import ecg_pb2, ecg_pb2_grpc


# -- Synthetic ECG Generator --
def generate_synthetic_ecg(duration_s: float = 10.0, fs: int = 125,
                            heart_rate: float = 75.0, add_pvcs: bool = False):
    """
    Generate a synthetic ECG-like signal with QRS-like spikes.
    Optionally injects PVC-like wide complexes.
    """
    n_samples = int(duration_s * fs)
    t = np.linspace(0, duration_s, n_samples)
    signal = np.zeros(n_samples)

    beat_interval = 60.0 / heart_rate
    beat_time = 0.0
    beat_num = 0

    while beat_time < duration_s:
        idx = int(beat_time * fs)
        if idx >= n_samples:
            break

        # Determine if this beat is a PVC
        is_pvc = add_pvcs and beat_num % 4 == 2  # Every 4th beat

        # QRS complex (sharp spike)
        qrs_width = 0.04 if not is_pvc else 0.12  # PVCs are wider
        qrs_amplitude = 1.0 if not is_pvc else -1.5  # PVCs often inverted/larger

        for i in range(n_samples):
            dt = t[i] - beat_time
            # QRS spike (Gaussian)
            signal[i] += qrs_amplitude * math.exp(-(dt ** 2) / (2 * (qrs_width / 3) ** 2))
            # T wave
            t_delay = 0.2 if not is_pvc else 0.3
            signal[i] += 0.3 * math.exp(-((dt - t_delay) ** 2) / (2 * 0.04 ** 2))
            # P wave  
            if not is_pvc:
                signal[i] += 0.15 * math.exp(-((dt + 0.12) ** 2) / (2 * 0.02 ** 2))

        beat_time += beat_interval + random.gauss(0, 0.01)
        beat_num += 1

    # Add baseline noise
    signal += np.random.normal(0, 0.02, n_samples)

    return signal.tolist()


# -- Streaming Client --
def run_client(host: str = "localhost", port: int = 50051,
               device_id: str = "SIM_DEVICE_001",
               duration_s: float = 30.0, chunk_s: float = 1.0,
               heart_rate: float = 75.0, send_pvcs: bool = False):
    """
    Stream synthetic ECG data to the gRPC server and print alerts.
    """
    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = ecg_pb2_grpc.ECGServiceStub(channel)

    fs = 125
    chunk_size = int(fs * chunk_s)

    print("=" * 60)
    print(f"  ECG gRPC Test Client")
    print(f"  Server: {host}:{port}")
    print(f"  Device: {device_id}")
    print(f"  Duration: {duration_s}s | Chunk: {chunk_s}s | HR: {heart_rate} bpm")
    print(f"  PVCs: {'YES (every 4th beat)' if send_pvcs else 'No'}")
    print("=" * 60)

    # Generate full signal
    print("\n[*] Generating synthetic ECG... ", end="")
    full_signal = generate_synthetic_ecg(
        duration_s=duration_s, fs=fs,
        heart_rate=heart_rate, add_pvcs=send_pvcs
    )
    print(f"Done ({len(full_signal)} samples)")

    # Request generator: yields ECGData chunks
    def request_stream():
        offset = 0
        chunk_num = 0
        while offset < len(full_signal):
            chunk = full_signal[offset:offset + chunk_size]
            msg = ecg_pb2.ECGData(
                values=chunk,
                device_id=device_id,
                timestamp=int(time.time() * 1000),
                sample_rate=fs
            )
            chunk_num += 1
            elapsed = offset / fs
            print(f"  -> Sent chunk #{chunk_num} ({len(chunk)} samples, t={elapsed:.1f}s)")
            yield msg
            offset += chunk_size
            time.sleep(chunk_s)  # Real-time pacing

        print("\n[*] All data sent. Waiting for final alerts...")

    # Call the bidirectional streaming RPC
    try:
        print("\n[>] Connecting to server...\n")
        responses = stub.StreamECG(request_stream())

        alert_count = 0
        for alert in responses:
            alert_count += 1
            ts = time.strftime("%H:%M:%S", time.localtime(alert.timestamp / 1000))
            print(f"\n  [!] ALERT #{alert_count}:")
            print(f"     Type:       {alert.arrhythmia_type}")
            print(f"     Confidence: {alert.confidence:.2%}")
            print(f"     Message:    {alert.message}")
            print(f"     Time:       {ts}")

        print(f"\n{'=' * 60}")
        print(f"  Session complete. Total alerts received: {alert_count}")
        print(f"{'=' * 60}")

    except grpc.RpcError as e:
        print(f"\n[X] gRPC Error: {e.code()} - {e.details()}")
    except KeyboardInterrupt:
        print("\n[STOP] Client stopped by user")
    finally:
        channel.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG gRPC Test Client")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument("--device", default="SIM_DEVICE_001", help="Device ID")
    parser.add_argument("--duration", type=float, default=30.0, help="Stream duration (seconds)")
    parser.add_argument("--hr", type=float, default=75.0, help="Heart rate (bpm)")
    parser.add_argument("--pvcs", action="store_true", help="Inject PVCs every 4th beat")
    args = parser.parse_args()

    run_client(
        host=args.host, port=args.port,
        device_id=args.device, duration_s=args.duration,
        heart_rate=args.hr, send_pvcs=args.pvcs
    )
