"""
gRPC Streaming Server for Real-Time ECG Arrhythmia Detection
=============================================================
Implements the ECGService defined in ecg.proto.
Bi-directional streaming: receives ECGData, returns ArrhythmiaAlert.

Usage:
    python grpc_server.py              # starts on port 50051
    python grpc_server.py --port 50052 # custom port
"""

import sys
import os
import time
import asyncio
import argparse
import logging
import uuid
import collections
import csv
import threading
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np

# ── Project path setup ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from grpc_gen import ecg_pb2, ecg_pb2_grpc

# ── Import pipeline components (same as dashboard/app.py) ───────────
from signal_processing.cleaning import clean_signal
from scipy.signal import resample_poly

TARGET_FS = 125
SEGMENT_DURATION_S = 10.0
SEGMENT_LENGTH = int(TARGET_FS * SEGMENT_DURATION_S)  # 1250 samples

# ── Lazy-loaded ML components ──────────────────────────────────────
_orchestrator = None
_xai_loaded = False


def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from decision_engine.rhythm_orchestrator import RhythmOrchestrator
        _orchestrator = RhythmOrchestrator()
    return _orchestrator


def _ensure_xai():
    global _xai_loaded
    if not _xai_loaded:
        import xai.xai  # noqa: F401  triggers model loading
        _xai_loaded = True


# ── Signal Processing (mirrored from app.py) ──────────────────────
def _preprocess(signal: np.ndarray, original_fs: int) -> np.ndarray:
    if original_fs != TARGET_FS:
        signal = resample_poly(signal, TARGET_FS, original_fs).astype(np.float32)
    else:
        signal = signal.astype(np.float32)
    return clean_signal(signal, TARGET_FS)


def _detect_r_peaks(signal: np.ndarray, fs: int) -> np.ndarray:
    try:
        import neurokit2 as nk
        _, info = nk.ecg_peaks(signal, sampling_rate=fs, method="neurokit")
        return np.array(info.get("ECG_R_Peaks", []))
    except Exception:
        return np.array([])


def _extract_features(segment: np.ndarray, r_peaks: np.ndarray) -> dict:
    """Extract basic clinical features from a 10s segment."""
    features = {
        "segment_index": 0,
        "mean_amplitude": float(np.mean(segment)),
        "std_amplitude": float(np.std(segment)),
    }

    rr_intervals_ms = np.array([])
    if len(r_peaks) >= 2:
        rr_intervals_ms = np.diff(r_peaks) * 1000.0 / TARGET_FS

    if rr_intervals_ms.size > 0:
        features["rr_intervals_ms"] = rr_intervals_ms.tolist()
        features["mean_rr"] = float(np.mean(rr_intervals_ms))
        features["mean_hr"] = float(60000.0 / features["mean_rr"]) if features["mean_rr"] > 0 else 0.0
        features["SDNN"] = float(np.std(rr_intervals_ms))
        diff_rr = np.diff(rr_intervals_ms)
        features["RMSSD"] = float(np.sqrt(np.mean(diff_rr ** 2))) if diff_rr.size > 0 else 0.0
        features["pNN50"] = float(np.sum(np.abs(diff_rr) > 50) / diff_rr.size) if diff_rr.size > 0 else 0.0
    else:
        features.update({"rr_intervals_ms": [], "mean_rr": 0, "mean_hr": 0, "SDNN": 0, "RMSSD": 0, "pNN50": 0})

    # QRS durations (simplified)
    features["qrs_durations_ms"] = []
    features["pr_interval"] = 0.0

    return features


# ── Per-device buffer ──────────────────────────────────────────────
class DeviceBuffer:
    """Accumulates ECG samples for one device until a full segment is ready."""

    def __init__(self, device_id: str, sample_rate: int = 125):
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.buffer = collections.deque()
        self.segment_length = int(sample_rate * SEGMENT_DURATION_S)
        self.overlap = self.segment_length // 2  # 50% overlap

    def add_samples(self, values: list):
        self.buffer.extend(values)

    def has_segment(self) -> bool:
        return len(self.buffer) >= self.segment_length

    def pop_segment(self) -> np.ndarray:
        """Extract one segment, keeping overlap samples for continuity."""
        segment = np.array([self.buffer[i] for i in range(self.segment_length)])
        # Remove first half (keep overlap)
        for _ in range(self.segment_length - self.overlap):
            self.buffer.popleft()
        return segment


# ── CSV Logging ────────────────────────────────────────────────────
class CSVLogger:
    def __init__(self, filename="logs/arrhythmia_alerts.csv"):
        self.filename = Path(filename)
        self.lock = threading.Lock()
        
        # Ensure directory exists
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        
        # Write header if file is new
        if not self.filename.exists():
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "DeviceID", "ArrhythmiaType", "Confidence", "Message"])

    def log_alert(self, device_id, arrhythmia_type, confidence, message):
        with self.lock:
            with open(self.filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([timestamp, device_id, arrhythmia_type, f"{confidence:.2f}", message])


# ── gRPC Servicer ──────────────────────────────────────────────────
class ECGServiceServicer(ecg_pb2_grpc.ECGServiceServicer):
    """
    Implements the StreamECG bi-directional streaming RPC.
    
    For each connected device:
    1. Buffers incoming ECGData chunks
    2. When 10s of data is ready → runs the full pipeline
    3. Streams back ArrhythmiaAlert messages
    """

    def __init__(self):
        self.logger = logging.getLogger("ECGServicer")
        self.device_buffers: dict[str, DeviceBuffer] = {}
        self.csv_logger = CSVLogger()

    def StreamECG(self, request_iterator, context):
        """Handle bi-directional streaming."""
        self.logger.info("New client connected")
        peer = context.peer()
        self.logger.info(f"  Peer: {peer}")

        try:
            for ecg_data in request_iterator:
                device_id = ecg_data.device_id or "unknown"
                sample_rate = ecg_data.sample_rate or TARGET_FS
                timestamp = ecg_data.timestamp
                values = list(ecg_data.values)

                if not values:
                    continue

                # Get or create buffer
                if device_id not in self.device_buffers:
                    self.device_buffers[device_id] = DeviceBuffer(device_id, sample_rate)
                    self.logger.info(f"  New device registered: {device_id} @ {sample_rate}Hz")

                buf = self.device_buffers[device_id]
                buf.add_samples(values)

                # Process all ready segments
                while buf.has_segment():
                    segment_raw = buf.pop_segment()
                    alerts = self._analyze_segment(segment_raw, sample_rate, device_id, timestamp)
                    for alert in alerts:
                        yield alert

        except grpc.RpcError as e:
            self.logger.warning(f"Client disconnected: {e}")
        except Exception as e:
            self.logger.error(f"Stream error: {e}", exc_info=True)
        finally:
            self.logger.info(f"Stream ended for peer {peer}")

    def _analyze_segment(self, raw_signal: np.ndarray, sample_rate: int,
                         device_id: str, timestamp: int) -> list:
        """Run the full detection pipeline on a 10s segment."""
        alerts = []
        try:
            # 1. Preprocess
            cleaned = _preprocess(raw_signal, sample_rate)

            # 2. R-peak detection
            r_peaks = _detect_r_peaks(cleaned, TARGET_FS)

            # 3. Feature extraction
            features = _extract_features(cleaned, r_peaks)

            # 4. Signal Quality check (basic)
            sqi = {"is_acceptable": True, "overall_sqi": 0.9}
            if features["mean_hr"] == 0 or len(r_peaks) < 2:
                sqi["is_acceptable"] = False
                sqi["overall_sqi"] = 0.2

            # 5. ML Inference
            try:
                _ensure_xai()
                from xai.xai import explain_segment
                ml_result = explain_segment(cleaned, features)
            except Exception as e:
                self.logger.warning(f"ML inference failed, using rule-only: {e}")
                ml_result = {"label": "Unknown", "confidence": 0.0, "probabilities": {}}

            # 6. Decision Engine
            orchestrator = _get_orchestrator()
            decision = orchestrator.decide(
                ml_prediction=ml_result,
                clinical_features=features,
                sqi_result=sqi,
                segment_index=0
            )

            # 7. Convert events → ArrhythmiaAlert protobuf messages
            for event in decision.final_display_events:
                etype = event.event_type if hasattr(event, 'event_type') else str(event.get("event_type", "Unknown"))
                conf = 0.0
                if hasattr(event, 'ml_evidence') and event.ml_evidence:
                    conf = event.ml_evidence.get("confidence", 0.0)
                elif isinstance(event, dict):
                    conf = event.get("ml_evidence", {}).get("confidence", 0.0)

                # Skip normal sinus rhythm
                if etype in ("Sinus Rhythm", "Unknown"):
                    continue

                hr_str = f" (HR: {features.get('mean_hr', 0):.0f} bpm)" if features.get("mean_hr") else ""
                alert = ecg_pb2.ArrhythmiaAlert(
                    arrhythmia_type=etype,
                    confidence=float(conf),
                    message=f"[{device_id}] {etype} detected{hr_str}",
                    timestamp=timestamp or int(time.time() * 1000)
                )
                alerts.append(alert)
                self.logger.info(f"  [ALERT]: {etype} (conf={conf:.2f}) for device {device_id}")
                
                # Log to CSV
                self.csv_logger.log_alert(device_id, etype, conf, alert.message)

            if not alerts:
                self.logger.debug(f"  [OK] Normal rhythm for device {device_id}")

        except Exception as e:
            self.logger.error(f"Analysis error: {e}", exc_info=True)
            alerts.append(ecg_pb2.ArrhythmiaAlert(
                arrhythmia_type="Error",
                confidence=0.0,
                message=f"Analysis error: {str(e)}",
                timestamp=int(time.time() * 1000)
            ))

        return alerts


# ── Server Launch ──────────────────────────────────────────────────
def serve(port: int = 50051, max_workers: int = 4):
    """Start the gRPC server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("gRPC-Server")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    ecg_pb2_grpc.add_ECGServiceServicer_to_server(ECGServiceServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()

    logger.info("=" * 60)
    logger.info(f"  ECG gRPC Server started on port {port}")
    logger.info(f"  Workers: {max_workers}")
    logger.info(f"  Segment size: {SEGMENT_LENGTH} samples ({SEGMENT_DURATION_S}s)")
    logger.info(f"  Target sample rate: {TARGET_FS} Hz")
    logger.info("=" * 60)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop(grace=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG gRPC Streaming Server")
    parser.add_argument("--port", type=int, default=50051, help="Port (default: 50051)")
    parser.add_argument("--workers", type=int, default=4, help="Max worker threads (default: 4)")
    args = parser.parse_args()
    serve(port=args.port, max_workers=args.workers)
