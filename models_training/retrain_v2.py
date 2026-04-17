#!/usr/bin/env python3
"""
retrain_v2.py — Enhanced Training with Signal + Extracted Features
===================================================================

SAFE ROLLOUT: This script sits alongside the original retrain.py.
              The original retrain.py is UNCHANGED and still works.

DIFFERENCES FROM retrain.py:
  1. Uses CNNTransformerWithFeatures (models_v2.py) instead of CNNTransformerClassifier
  2. Extracts feature vectors from each signal window during dataset loading
  3. Passes (signal, features) to the model during training
  4. Can optionally bootstrap from a v1 checkpoint (transfer signal pathway weights)
  5. Saves checkpoints with "_v2" suffix — never overwrites v1 checkpoints

USAGE:
  python retrain_v2.py --task rhythm  --mode initial
  python retrain_v2.py --task rhythm  --mode finetune
  python retrain_v2.py --task ectopy  --mode initial
  python retrain_v2.py --task ectopy  --mode finetune

  # Optional: bootstrap from v1 checkpoint
  python retrain_v2.py --task rhythm --mode initial --bootstrap-v1
"""

import os
import sys
import json
import psycopg2
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import joblib
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "models_training"))

from data_loader import (
    normalize_label,
    RHYTHM_CLASS_NAMES, get_rhythm_label_idx,
    ECTOPY_CLASS_NAMES, get_ectopy_label_idx, ECTOPY_INDEX,
)
from models_v2 import CNNTransformerWithFeatures, load_v2_from_v1_checkpoint
from signal_processing_v3 import process_ecg_v3
from signal_processing_v3.features.extraction import (
    feature_dict_to_vector,
    FEATURE_NAMES_V3 as FEATURE_NAMES,
)
NUM_FEATURES = len(FEATURE_NAMES)


def clean_signal(sig, fs):
    """V3 preprocessing replaces V2 clean_signal."""
    from signal_processing_v3.preprocessing.pipeline import preprocess_v3
    return preprocess_v3(sig, fs=fs)["cleaned"].astype(np.float32)


def extract_feature_vector(sig, fs=125, r_peaks=None):
    """V3 feature extractor — drop-in replacement for V2 extract_feature_vector."""
    result = process_ecg_v3(sig, fs=fs, min_quality=0.0)
    return feature_dict_to_vector(result["features"])

# ---------------------------------------------------------------------------
# Paths & DB
# ---------------------------------------------------------------------------
OUTPUT      = Path(__file__).resolve().parent / "outputs"
CHECKPOINTS = OUTPUT / "checkpoints"
LOGS        = OUTPUT / "logs"
for d in (OUTPUT, CHECKPOINTS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

DB_PARAMS = {
    "host":     "127.0.0.1",
    "dbname":   "ecg_analysis",
    "user":     "ecg_user",
    "password": "sais",
    "port":     "5432",
}

CARDIOLOGIST_OVERSAMPLE = 5

# ---------------------------------------------------------------------------
# Logger with Error Tracking
# ---------------------------------------------------------------------------
class TeeLogger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log      = open(path, "w", encoding="utf-8")
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()

# Data loading error tracker
LOAD_ERRORS = {
    "signal_parse_errors": [],
    "signal_length_errors": [],
    "event_parse_errors": [],
    "rpeak_detect_errors": [],
    "feature_extract_errors": [],
    "resample_errors": [],
    "invalid_event_times": [],
    "unknown_annotation_source": [],
    "invalid_labels": [],
}

def log_load_error(error_type: str, segment_id: int, error_msg: str):
    """Log data loading errors for analysis."""
    if error_type in LOAD_ERRORS:
        LOAD_ERRORS[error_type].append({
            "segment_id": segment_id,
            "error": str(error_msg)[:100]
        })

# ---------------------------------------------------------------------------
# Focal Loss (same as retrain.py)
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce   = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

# ---------------------------------------------------------------------------
# Dataset — same event-based loading as retrain.py, plus feature extraction
# ---------------------------------------------------------------------------
class ECGEventDatasetV2(torch.utils.data.Dataset):
    """
    Same as ECGEventDataset from retrain.py, but each sample also includes
    a feature vector extracted from the signal window.

    Tuple: (signal_np, feature_np, label_idx, annotation_source, seg_id, filename)
    """

    TARGET_FS      = 125
    WINDOW_SAMPLES = 1250
    SLIDE_STEP     = 1250

    def __init__(self, task="rhythm", source_filter="all", augment=False):
        self.augment = augment
        self.task    = task
        self.samples = []  # (signal, features, label, source, seg_id, filename)
        self.scaler  = None  # Set by run_initial/run_finetune after fitting
        self.training_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n[DatasetV2] task={task}  filter={source_filter}")
        print(f"[DatasetV2] Training session: {self.training_session_id}")
        print( "[DatasetV2] Fetching segments from DB...")

        # ARCHITECTURAL FIX: Only fetch segments that are cardiologist-verified (is_corrected=TRUE)
        # This prevents reusing stale annotations from prior training runs.
        with psycopg2.connect(**DB_PARAMS) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT segment_id, signal_data, events_json, arrhythmia_label,
                           segment_fs, filename, is_corrected, features_json, sqi_score
                    FROM   ecg_features_annotatable
                    WHERE  signal_data IS NOT NULL
                      AND  is_corrected = TRUE
                    ORDER BY segment_id
                """)
                rows = cur.fetchall()

        print(f"[DatasetV2] Fetched {len(rows)} cardiologist-verified segments")

        skipped_null, skipped_short, skipped_label, total_windows = 0, 0, 0, 0
        used_segment_ids = set()  # Track segments used for None-class to prevent duplication

        for row_data in rows:
            seg_id, signal_raw, events_json_raw, arrhythmia_label, fs, filename, is_corrected, features_json_raw, sqi_score = row_data

            # ── SIGNAL PARSING WITH ERROR LOGGING ──
            try:
                if isinstance(signal_raw, str):
                    signal = np.array(json.loads(signal_raw), dtype=np.float32)
                else:
                    signal = np.array(signal_raw, dtype=np.float32)
            except Exception as e:
                log_load_error("signal_parse_errors", seg_id, str(e))
                skipped_null += 1
                continue

            # ── SIGNAL LENGTH VALIDATION ──
            if signal is None or len(signal) == 0:
                log_load_error("signal_length_errors", seg_id, f"empty signal array")
                skipped_short += 1
                continue

            if len(signal) < self.WINDOW_SAMPLES:
                log_load_error("signal_length_errors", seg_id, f"len={len(signal)} < {self.WINDOW_SAMPLES}")
                skipped_short += 1
                continue

            fs = int(fs) if fs else self.TARGET_FS

            # ── RESAMPLE WITH ERROR HANDLING ──
            if fs != self.TARGET_FS and fs > 0:
                try:
                    from scipy.signal import resample as sci_resample
                    target_len = int(len(signal) * self.TARGET_FS / fs)
                    signal = sci_resample(signal, target_len).astype(np.float32)
                    if len(signal) < self.WINDOW_SAMPLES:
                        log_load_error("resample_errors", seg_id, f"resampled len={len(signal)} < {self.WINDOW_SAMPLES}")
                        continue
                    fs = self.TARGET_FS
                except Exception as e:
                    log_load_error("resample_errors", seg_id, str(e))
                    continue

            try:
                signal = clean_signal(signal, self.TARGET_FS).astype(np.float32)
            except Exception as e:
                log_load_error("signal_parse_errors", seg_id, f"cleaning failed: {str(e)}")
                continue

            # Pre-load r_peaks from features_json if available
            r_peaks_hint = None
            if features_json_raw:
                try:
                    fj = features_json_raw if isinstance(features_json_raw, dict) else json.loads(features_json_raw)
                    rp = fj.get("r_peaks", [])
                    if rp and isinstance(rp, list) and len(rp) > 0:
                        r_peaks_hint = np.array(rp, dtype=int)
                except Exception as e:
                    log_load_error("rpeak_detect_errors", seg_id, f"r_peaks parse: {str(e)}")
                    pass

            # ── EVENTS PARSING WITH ERROR LOGGING ──
            try:
                if isinstance(events_json_raw, str):
                    ev_data = json.loads(events_json_raw)
                else:
                    ev_data = events_json_raw or []
            except Exception as e:
                log_load_error("event_parse_errors", seg_id, str(e))
                ev_data = []

            if isinstance(ev_data, list):
                events = ev_data
            elif isinstance(ev_data, dict):
                events = ev_data.get("events", []) if "events" in ev_data else []
            else:
                events = []

            # Corrected rhythm segments (same logic as retrain.py)
            if self.task == "rhythm" and is_corrected and arrhythmia_label:
                has_existing = any(
                    ev.get("event_category") == "RHYTHM" and ev.get("annotation_source") == "cardiologist"
                    for ev in events if isinstance(ev, dict)
                )
                if not has_existing:
                    label_idx = get_rhythm_label_idx(arrhythmia_label)
                    if label_idx is not None:
                        try:
                            windows = self._slide_windows(signal, 0.0, 10.0, fs)
                            # Use stored V3 features from features_json when available.
                            # The full segment window is identical to what was backfilled,
                            # so re-running process_ecg_v3 is redundant and slow.
                            stored_feat = None
                            if features_json_raw:
                                try:
                                    fj = features_json_raw if isinstance(features_json_raw, dict) else json.loads(features_json_raw)
                                    stored_feat = feature_dict_to_vector(fj)
                                except Exception:
                                    stored_feat = None
                            for win in windows:
                                if win is not None:
                                    if stored_feat is not None:
                                        feat = stored_feat
                                    else:
                                        feat = extract_feature_vector(win, fs=self.TARGET_FS, r_peaks=None)
                                    self.samples.append((
                                        win, feat, label_idx, "cardiologist",
                                        seg_id, filename or ""
                                    ))
                                    total_windows += 1
                        except Exception as e:
                            log_load_error("feature_extract_errors", seg_id, f"rhythm window: {str(e)}")
                    else:
                        log_load_error("invalid_labels", seg_id, f"unknown rhythm label: {arrhythmia_label}")

            # ── EVENT-LEVEL PROCESSING WITH VALIDATION ──
            for event in events:
                if not isinstance(event, dict):
                    continue

                ann_source  = event.get("annotation_source", "unknown")
                event_type  = event.get("event_type", "")
                start_s     = float(event.get("start_time", 0.0)) if event.get("start_time") is not None else 0.0
                end_s       = float(event.get("end_time",  10.0)) if event.get("end_time") is not None else 10.0

                # ── ANNOTATION SOURCE VALIDATION ──
                if ann_source == "unknown" or not ann_source:
                    log_load_error("unknown_annotation_source", seg_id, f"event {event_type}")
                    continue

                if not event_type or not isinstance(event_type, str) or len(event_type.strip()) == 0:
                    continue

                if source_filter == "cardiologist" and ann_source != "cardiologist":
                    continue

                # ── EVENT TIME RANGE VALIDATION ──
                # Clip floating-point overflow (e.g. 10.036 → 10.0) before rejecting
                start_s = max(0.0, min(start_s, 10.0))
                end_s   = max(0.0, min(end_s,   10.0))
                if start_s >= end_s:
                    log_load_error("invalid_event_times", seg_id, f"{event_type}: [{start_s}, {end_s}] after clip")
                    continue

                # Resolve label
                if self.task == "rhythm":
                    if event.get("event_category") == "ECTOPY": continue
                    label_idx = get_rhythm_label_idx(event_type)
                else:
                    if event.get("event_category") == "RHYTHM": continue
                    label_idx = get_ectopy_label_idx(event_type)

                if label_idx is None:
                    log_load_error("invalid_labels", seg_id, f"{event_type} (task={self.task})")
                    skipped_label += 1
                    continue

                # Window extraction (same as retrain.py)
                try:
                    event_duration_s = end_s - start_s
                    narrow_threshold = (self.WINDOW_SAMPLES / fs) * 1.5

                    if event_duration_s <= narrow_threshold:
                        windows = [self._center_window(signal, (start_s + end_s) / 2.0, fs)]
                    else:
                        windows = self._slide_windows(signal, start_s, end_s, fs)

                    for win in windows:
                        if win is not None:
                            feat = extract_feature_vector(win, fs=self.TARGET_FS, r_peaks=None)
                            self.samples.append((
                                win, feat, label_idx, ann_source,
                                seg_id, filename or ""
                            ))
                            total_windows += 1
                except Exception as e:
                    log_load_error("feature_extract_errors", seg_id, f"{event_type}: {str(e)}")
                    continue

        # ── ECTOPY ONLY: Add None-class beats from unannotated sinus segments ──
        # Use unannotated sinus-labeled segments (15K+ available) instead of the
        # verified set — no data leakage with rhythm model, no manual annotation needed.
        # These segments are safe ground truth for "normal beat" since they are labeled
        # Sinus Rhythm by the importer. SQI filter drops garbage signal.
        if self.task == "ectopy":
            none_candidates = []
            from scipy.signal import find_peaks as _find_peaks

            # Compute cap up front so we can break early
            pvc_count = sum(1 for s in self.samples if s[2] == ECTOPY_INDEX["PVC"])
            cap = max(pvc_count * 2, 5000)

            import random as _random

            # Fetch unannotated sinus segments — separate from the verified training set
            with psycopg2.connect(**DB_PARAMS) as _conn:
                with _conn.cursor() as _cur:
                    _cur.execute("""
                        SELECT segment_id, signal_data, events_json, arrhythmia_label,
                               segment_fs, filename, is_corrected, features_json, sqi_score
                        FROM   ecg_features_annotatable
                        WHERE  signal_data IS NOT NULL
                          AND  (is_corrected = FALSE OR is_corrected IS NULL)
                          AND  arrhythmia_label IN ('Sinus Rhythm','Sinus Bradycardia','Sinus Tachycardia','Normal')
                          AND  (sqi_score IS NULL OR sqi_score >= 0.5)
                        ORDER BY RANDOM()
                        LIMIT  %s
                    """, (cap * 3,))  # fetch 3× cap so early-break has enough variety
                    sinus_rows = _cur.fetchall()

            print(f"[DatasetV2] None-class pool: {len(sinus_rows)} unannotated sinus segments (cap={cap})")
            _random.shuffle(sinus_rows)

            for row_data in sinus_rows:
                if len(none_candidates) >= cap:
                    break

                seg_id, signal_raw, events_json_raw, arrhythmia_label, fs, filename, is_corrected, features_json_raw, sqi_score = row_data

                try:
                    sig = np.array(json.loads(signal_raw) if isinstance(signal_raw, str) else signal_raw, dtype=np.float32)
                except Exception as e:
                    log_load_error("signal_parse_errors", seg_id, f"none-class: {str(e)}")
                    continue

                if sig is None or len(sig) < self.WINDOW_SAMPLES:
                    continue

                seg_fs = int(fs) if fs else self.TARGET_FS
                try:
                    if seg_fs != self.TARGET_FS and seg_fs > 0:
                        from scipy.signal import resample as _resample
                        sig = _resample(sig, int(len(sig) * self.TARGET_FS / seg_fs)).astype(np.float32)
                except Exception as e:
                    log_load_error("resample_errors", seg_id, f"none-class: {str(e)}")
                    continue

                try:
                    sig = clean_signal(sig, self.TARGET_FS).astype(np.float32)
                except Exception as e:
                    log_load_error("signal_parse_errors", seg_id, f"none-class clean: {str(e)}")
                    continue

                # Skip if segment has any ectopy events — avoid mislabeling normal beats
                try:
                    ev_raw = json.loads(events_json_raw) if isinstance(events_json_raw, str) else (events_json_raw or [])
                    evs = ev_raw.get("events", []) if isinstance(ev_raw, dict) else (ev_raw if isinstance(ev_raw, list) else [])
                except Exception as e:
                    log_load_error("event_parse_errors", seg_id, f"none-class: {str(e)}")
                    evs = []

                if any(e.get("event_category", "").upper() == "ECTOPY" for e in evs if isinstance(e, dict)):
                    continue

                # R-peaks: prefer stored hint from features_json, fall back to find_peaks
                r_peaks = None
                try:
                    if features_json_raw:
                        fj = features_json_raw if isinstance(features_json_raw, dict) else json.loads(features_json_raw)
                        rp = fj.get("r_peaks", [])
                        if rp and isinstance(rp, list) and len(rp) > 0:
                            r_peaks = np.array(rp, dtype=int)
                except Exception as e:
                    log_load_error("rpeak_detect_errors", seg_id, f"none-class r_peaks: {str(e)}")

                if r_peaks is None or len(r_peaks) == 0:
                    try:
                        height = float(np.percentile(sig, 75))
                        r_peaks, _ = _find_peaks(sig, height=height, distance=int(0.5 * self.TARGET_FS))
                    except Exception as e:
                        log_load_error("rpeak_detect_errors", seg_id, f"find_peaks: {str(e)}")
                        continue

                # ── PREVENT SEGMENT DUPLICATION ──
                if seg_id in used_segment_ids:
                    continue
                used_segment_ids.add(seg_id)

                for r in r_peaks:
                    if len(none_candidates) >= cap:
                        break
                    try:
                        win = self._center_window(sig, r / self.TARGET_FS, self.TARGET_FS)
                        if win is not None:
                            feat = extract_feature_vector(win, fs=self.TARGET_FS, r_peaks=None)
                            none_candidates.append((win, feat, ECTOPY_INDEX["None"], "imported", seg_id, filename or ""))
                    except Exception as e:
                        log_load_error("feature_extract_errors", seg_id, f"none-class beat: {str(e)}")
                        continue

            self.samples.extend(none_candidates)
            total_windows += len(none_candidates)
            print(f"[DatasetV2] Added {len(none_candidates)} None-class beats from {len(used_segment_ids)} sinus segments (cap={cap})")

        print(f"[DatasetV2] {total_windows} windows extracted")
        print(f"            skipped: null={skipped_null} short={skipped_short} no_label={skipped_label}")

        sources = Counter(s[3] for s in self.samples)
        print(f"            sources: {dict(sources)}")

        label_counts = Counter(s[2] for s in self.samples)
        cls_names = ECTOPY_CLASS_NAMES if self.task == "ectopy" else RHYTHM_CLASS_NAMES
        print(f"            class distribution:")
        for idx, name in enumerate(cls_names):
            count = label_counts.get(idx, 0)
            print(f"              {idx}: {name} = {count}")
            if count == 0 and idx in [0, 1, 2, 3]:  # Critical classes
                if self.task == "rhythm" and idx in [4, 5, 6, 7, 8, 9]:  # VF, VT rare classes
                    continue  # Skip warning for ultra-rare classes
                print(f"              [WARNING] Critical class has 0 samples!")

    # -- Window helpers (identical to retrain.py) --------------------------------

    def _center_window(self, signal, center_s, fs):
        center_i = int(center_s * fs)
        half     = self.WINDOW_SAMPLES // 2
        s_i      = center_i - half
        e_i      = center_i + half
        if s_i < 0:
            win = signal[0:max(0, e_i)]
            return np.pad(win, (abs(s_i), 0)).astype(np.float32)
        elif e_i > len(signal):
            win = signal[s_i:len(signal)]
            return np.pad(win, (0, e_i - len(signal))).astype(np.float32)
        return signal[s_i:e_i].astype(np.float32)

    def _slide_windows(self, signal, start_s, end_s, fs):
        start_i = int(start_s * fs)
        end_i   = min(len(signal), int(end_s * fs))
        wins    = []
        pos     = start_i
        while pos + self.WINDOW_SAMPLES <= end_i:
            wins.append(self._pad_or_crop(signal[pos:pos + self.WINDOW_SAMPLES]))
            pos += self.SLIDE_STEP
        wins.append(self._center_window(signal, (start_s + end_s) / 2.0, fs))
        return wins

    def _pad_or_crop(self, signal):
        if len(signal) > self.WINDOW_SAMPLES:
            return signal[:self.WINDOW_SAMPLES]
        elif len(signal) < self.WINDOW_SAMPLES:
            return np.pad(signal, (0, self.WINDOW_SAMPLES - len(signal)), mode='constant')
        return signal

    def _augment_signal(self, sig):
        if not self.augment:
            return sig
        sig = sig.copy()
        n = len(sig)
        t = np.arange(n, dtype=np.float32) / self.TARGET_FS
        sig_std = max(float(np.std(sig)), 1e-6)

        if np.random.rand() < 0.5:
            sig = sig * np.random.uniform(0.85, 1.15)
        if np.random.rand() < 0.5:
            sig = sig + np.random.normal(0, 0.02 * sig_std, sig.shape).astype(np.float32)
        if np.random.rand() < 0.5:
            freq = np.random.uniform(0.1, 0.5)
            amp = np.random.uniform(0.05, 0.20) * sig_std
            sig = sig + (amp * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi))).astype(np.float32)
        if np.random.rand() < 0.3:
            freq = np.random.choice([50.0, 60.0])
            if freq < 0.5 * self.TARGET_FS:
                amp = np.random.uniform(0.01, 0.05) * sig_std
                sig = sig + (amp * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi))).astype(np.float32)

        return sig.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sig, feat, label_idx, source, seg_id, filename = self.samples[idx]
        if self.augment:
            sig = self._augment_signal(sig.copy())
            # Re-extract features from augmented signal for consistency
            feat = extract_feature_vector(sig, fs=self.TARGET_FS, r_peaks=None)
        # Apply feature scaler if fitted (prevents high-magnitude features from
        # dominating gradient descent; scaler is fit on training split only)
        if self.scaler is not None:
            feat = self.scaler.transform(feat.reshape(1, -1))[0].astype(np.float32)
        return {
            "signal":   sig,
            "features": feat,
            "label":    label_idx,
            "source":   source,
            "seg_id":   seg_id,
            "filename": filename,
        }


# ---------------------------------------------------------------------------
# Collate — produces (signal_tensor, feature_tensor, label_tensor)
# ---------------------------------------------------------------------------
def collate_fn_v2(batch):
    xs = torch.stack([
        torch.from_numpy(b["signal"]).float().unsqueeze(0) for b in batch
    ])
    fs = torch.stack([
        torch.from_numpy(b["features"]).float() for b in batch
    ])
    ys = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return xs, fs, ys


# ---------------------------------------------------------------------------
# Train / eval — now pass features to model
# ---------------------------------------------------------------------------
def train_epoch(model, opt, criterion, loader, device):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []

    for x, f, y in tqdm(loader, desc="  train", ncols=80, leave=False):
        x, f, y = x.to(device), f.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x, f)  # ← pass features
        loss   = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * x.size(0)
        with torch.no_grad():
            y_true += y.cpu().tolist()
            y_pred += logits.argmax(1).cpu().tolist()

    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    return {"loss": total_loss / max(len(y_true), 1), "acc": acc}


def eval_epoch(model, criterion, loader, device, num_classes, class_names=None):
    from sklearn.metrics import confusion_matrix, classification_report
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, f, y in tqdm(loader, desc="  val  ", ncols=80, leave=False):
            x, f, y = x.to(device), f.to(device), y.to(device)
            logits = model(x, f)  # ← pass features
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            y_true += y.cpu().tolist()
            y_pred += logits.argmax(1).cpu().tolist()

    y_arr  = np.array(y_true)
    yp_arr = np.array(y_pred)
    per_cls = {
        i: float((yp_arr[y_arr == i] == i).mean()) if (y_arr == i).sum() > 0 else 0.0
        for i in range(num_classes)
    }

    labels = list(range(num_classes))
    cm = confusion_matrix(y_arr, yp_arr, labels=labels)
    names = class_names or [str(i) for i in labels]

    present = sorted(set(y_arr.tolist()) | set(yp_arr.tolist()))
    if len(present) > 0:
        print("\n  Confusion Matrix (rows=true, cols=pred):")
        hdr = "  {:>20s}".format("") + "".join(f" {names[c][:8]:>8s}" for c in present)
        print(hdr)
        for r in present:
            row_str = "  {:>20s}".format(names[r][:20])
            for c in present:
                row_str += f" {cm[r, c]:>8d}"
            print(row_str)

    try:
        report_str = classification_report(
            y_arr, yp_arr, labels=labels, target_names=names, zero_division=0
        )
        print(f"\n{report_str}")
    except Exception:
        pass

    return {
        "loss":            total_loss / max(len(y_true), 1),
        "acc":             float((y_arr == yp_arr).mean()),
        "balanced_acc":    float(np.mean(list(per_cls.values()))),
        "per_class":       per_cls,
        "confusion_matrix": cm,
        "y_true":          y_arr,
        "y_pred":          yp_arr,
    }


# ---------------------------------------------------------------------------
# Helpers (same as retrain.py)
# ---------------------------------------------------------------------------
def _recording_id(filename: str) -> str:
    stem = Path(filename).stem.lower()
    if "mitdb" in stem:
        parts = stem.split("_")
        if len(parts) >= 2: return f"mitdb_{parts[1]}"
    if "ptb" in stem:
        parts = stem.split("_")
        if len(parts) >= 2: return f"ptbxl_{parts[1]}"
    parts = stem.split("_seg_")
    return parts[0] if len(parts) > 1 else stem


def filename_split(samples, val_ratio=0.15):
    """
    Stratified split: group windows by (recording_id, class_label) so that
    every class that has training data is guaranteed to have at least one
    window in the validation set. Falls back to pure filename split for
    recordings that only have one window (can't split).
    """
    # Group indices by (recording_id, label)
    from collections import defaultdict
    by_class_rec = defaultdict(list)   # (label, rid) -> [idx, ...]
    for i, s in enumerate(samples):
        rid   = _recording_id(s[5])    # s[5] = filename
        label = s[2]                    # s[2] = label index
        by_class_rec[(label, rid)].append(i)

    train_set, val_set = set(), set()

    # For each class, shuffle recordings and split val_ratio into val
    by_label = defaultdict(list)
    for (label, rid), idxs in by_class_rec.items():
        by_label[label].append((rid, idxs))

    for label, rec_list in by_label.items():
        np.random.shuffle(rec_list)
        n_val_recs = max(1, int(len(rec_list) * val_ratio))
        for rid, idxs in rec_list[-n_val_recs:]:
            val_set.update(idxs)
        for rid, idxs in rec_list[:-n_val_recs]:
            train_set.update(idxs)

    # Any window that ended up in both (shouldn't happen) → keep in train
    overlap = train_set & val_set
    val_set -= overlap

    # If val is still empty (very small dataset), fall back to last 15% of all
    if not val_set:
        all_idx = list(range(len(samples)))
        cut = int(len(all_idx) * (1 - val_ratio))
        train_set = set(all_idx[:cut])
        val_set   = set(all_idx[cut:])

    train_idx = sorted(train_set)
    val_idx   = sorted(val_set)

    # Report
    val_labels   = Counter(samples[i][2] for i in val_idx)
    n_train_recs = len({_recording_id(samples[i][5]) for i in train_idx})
    n_val_recs   = len({_recording_id(samples[i][5]) for i in val_idx})
    print(f"[Split] {n_train_recs} train recordings -> {len(train_idx)} windows  |  "
          f"{n_val_recs} val recordings -> {len(val_idx)} windows")
    print(f"[Split] Val classes present: "
          f"{sum(1 for c in val_labels.values() if c > 0)}"
          f"/{len(set(s[2] for s in samples))} "
          f"({', '.join(f'{k}:{v}' for k,v in sorted(val_labels.items()))})")
    return train_idx, val_idx


def build_sampler(samples, train_idx, num_classes, oversample_factor=2):
    labels = [samples[i][2] for i in train_idx]  # s[2] = label (shifted)
    counts = Counter(labels)
    ca     = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float32)
    ca[ca == 0] = 1.0
    class_w = 1.0 / ca
    weights = []
    for i in train_idx:
        lbl    = samples[i][2]
        source = samples[i][3]
        cw     = float(class_w[lbl])
        sw     = float(CARDIOLOGIST_OVERSAMPLE) if source == "cardiologist" else 1.0
        weights.append(cw * sw)
    n = len(train_idx) * oversample_factor
    return WeightedRandomSampler(weights, num_samples=n, replacement=True)


def _build_criterion(labels, num_classes, device):
    counts = Counter(labels)
    ca     = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float32)
    ca[ca == 0] = 1.0
    alpha  = torch.tensor(
        np.clip(np.sqrt(ca.sum() / (num_classes * ca)), 0.5, 15.0),  # raised cap 10→15 for very rare classes
        dtype=torch.float32,
    ).to(device)
    return FocalLoss(alpha=alpha, gamma=3.0)  # gamma 2.5→3.0: harder penalty on confident wrong predictions


# ---------------------------------------------------------------------------
# Database Update Helper
# ---------------------------------------------------------------------------
def _update_training_metadata(dataset, training_round_increment=True):
    """
    Update database after training completes:
    1. Increment training_round for segments used in this training run
    2. Keep used_for_training=TRUE (marks segment was in a training set)
    3. Add timestamp for audit trail

    This prevents stale annotations from being reused in future training runs
    while still maintaining training history.
    """
    if not hasattr(dataset, 'samples') or len(dataset.samples) == 0:
        print("[DB] No training data to update")
        return

    # Collect unique segment IDs used in this training run
    used_segment_ids = set(int(s[4]) for s in dataset.samples if len(s) > 4)
    print(f"[DB] Updating metadata for {len(used_segment_ids)} segments")

    if len(used_segment_ids) == 0:
        print("[DB] No segments to update")
        return

    try:
        conn = psycopg2.connect(**DB_PARAMS)
        with conn.cursor() as cur:
            # Build WHERE clause for segment IDs
            seg_list = ",".join(str(sid) for sid in used_segment_ids)

            if training_round_increment:
                # Increment training_round and keep used_for_training=TRUE
                cur.execute(f"""
                    UPDATE ecg_features_annotatable
                    SET training_round = training_round + 1,
                        used_for_training = TRUE
                    WHERE segment_id IN ({seg_list})
                """)

        conn.commit()
        print(f"[DB] Updated training_round for {len(used_segment_ids)} segments")
    except Exception as e:
        print(f"[DB ERROR] Failed to update training metadata: {e}")
    finally:
        try:
            conn.close()
        except:
            pass


# ---------------------------------------------------------------------------
# INITIAL TRAINING (v2)
# ---------------------------------------------------------------------------
def run_initial(task, num_epochs, batch_size, lr, bootstrap_v1=False):
    print(f"\n{'='*65}")
    print(f"  V2 INITIAL TRAINING  |  task={task.upper()}  epochs={num_epochs}")
    print(f"  Features: {NUM_FEATURES} ({', '.join(FEATURE_NAMES[:5])}...)")
    print(f"{'='*65}")

    class_names = RHYTHM_CLASS_NAMES if task == "rhythm" else ECTOPY_CLASS_NAMES
    num_classes = len(class_names)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path   = CHECKPOINTS / f"best_model_{task}_v2.pth"  # ← _v2 suffix!

    ds = ECGEventDatasetV2(task=task, source_filter="all", augment=True)
    if len(ds) < 20:
        print(f"[ABORT] Only {len(ds)} windows - not enough to train.")
        return

    # Check for empty critical classes before training.
    # For rhythm: only class 0 (Sinus Rhythm) is truly required — rare arrhythmias
    #             (AFL, VF, Junctional, etc.) may legitimately have 0 samples.
    # For ectopy: all 3 classes (None=0, PVC=1, PAC=2) must be present.
    label_counts = Counter(s[2] for s in ds.samples)
    critical_classes = [0, 1, 2] if task == "ectopy" else [0]
    empty_critical = [cls for cls in critical_classes if label_counts.get(cls, 0) == 0 and cls < num_classes]

    if empty_critical:
        cls_names = ECTOPY_CLASS_NAMES if task == "ectopy" else RHYTHM_CLASS_NAMES
        empty_names = [f"{cls}={cls_names[cls]}" for cls in empty_critical if cls < len(cls_names)]
        print(f"[ABORT] Empty critical classes detected: {', '.join(empty_names)}")
        print(f"        Please add training data for these classes before retraining.")
        return

    # Warn (but do not abort) about rhythm classes with 0 samples
    if task == "rhythm":
        cls_names = RHYTHM_CLASS_NAMES
        zero_classes = [f"{i}={cls_names[i]}" for i in range(num_classes) if label_counts.get(i, 0) == 0 and i < len(cls_names)]
        if zero_classes:
            print(f"[INFO] Classes with no training data (will not be predicted): {', '.join(zero_classes)}")

    train_idx, val_idx = filename_split(ds.samples)
    if len(val_idx) == 0:
        val_idx   = train_idx[int(0.9 * len(train_idx)):]
        train_idx = train_idx[:int(0.9 * len(train_idx))]

    train_labels = [ds.samples[i][2] for i in train_idx]
    counts       = Counter(train_labels)

    print(f"\nClass distribution (train windows):")
    for i, name in enumerate(class_names):
        print(f"  {i:02d}  {name:<40}  {counts.get(i, 0):>6}")

    # ── Fit StandardScaler on training features ──────────────────────────────
    # Feature vector mixes wildly different scales (SDNN ~20–200 ms vs ST
    # elevation ~±0.5 mV).  Without normalization gradient descent ignores
    # low-magnitude clinical features.  Scaler is fit on train split only
    # (never val) and saved for inference.
    X_train_feats = np.stack([ds.samples[i][1] for i in train_idx])
    scaler = StandardScaler()
    scaler.fit(X_train_feats)
    scaler_path = CHECKPOINTS / f"feature_scaler_{task}.joblib"
    joblib.dump(scaler, scaler_path)
    ds.scaler = scaler   # applied in __getitem__ for both train and val
    print(f"[Scaler] Fit on {len(train_idx)} train windows → {scaler_path.name}")

    eff_batch = min(batch_size, max(2, len(train_idx) // 4))

    sampler   = build_sampler(ds.samples, train_idx, num_classes, oversample_factor=6)  # 2→6
    train_ds  = torch.utils.data.Subset(ds, train_idx)
    val_ds    = torch.utils.data.Subset(ds, val_idx)
    train_ldr = DataLoader(train_ds, batch_size=eff_batch, sampler=sampler,   collate_fn=collate_fn_v2)
    val_ldr   = DataLoader(val_ds,   batch_size=eff_batch, shuffle=False,      collate_fn=collate_fn_v2)

    # Create model — optionally bootstrap signal pathway from v1
    if bootstrap_v1:
        v1_ckpt = CHECKPOINTS / f"best_model_{task}.pth"
        if v1_ckpt.exists():
            print(f"\n[Bootstrap] Loading signal pathway from v1: {v1_ckpt}")
            model, _ = load_v2_from_v1_checkpoint(v1_ckpt, num_classes, NUM_FEATURES, device)
        else:
            print(f"[Bootstrap] v1 checkpoint not found at {v1_ckpt}, starting fresh")
            model = CNNTransformerWithFeatures(num_classes=num_classes, num_features=NUM_FEATURES).to(device)
    else:
        model = CNNTransformerWithFeatures(num_classes=num_classes, num_features=NUM_FEATURES).to(device)

    criterion = _build_criterion(train_labels, num_classes, device)
    opt       = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)  # 5e-4→5e-3: stronger regularisation
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)  # track bal_acc, not loss

    best_bal_acc  = 0.0
    no_improve    = 0
    early_stop_patience = 15  # stop if no improvement for 15 epochs

    print(f"\nDevice={device}  Batch={eff_batch}  "
          f"Train_windows={len(train_idx)}  Val_windows={len(val_idx)}\n")

    for ep in range(1, num_epochs + 1):
        tr = train_epoch(model, opt, criterion, train_ldr, device)
        va = eval_epoch(model, criterion, val_ldr, device, num_classes, class_names)
        scheduler.step(va["balanced_acc"])

        print(f"Ep {ep:02d}/{num_epochs}  "
              f"train loss={tr['loss']:.4f} acc={tr['acc']:.3f}  "
              f"val loss={va['loss']:.4f} bal_acc={va['balanced_acc']:.3f}")

        if va["balanced_acc"] > best_bal_acc:
            best_bal_acc = va["balanced_acc"]
            no_improve   = 0
            torch.save({
                "epoch":        ep,
                "model_state":  model.state_dict(),
                "balanced_acc": best_bal_acc,
                "class_names":  class_names,
                "num_features": NUM_FEATURES,
                "feature_names": FEATURE_NAMES,
                "mode":         "initial",
                "version":      "v2",
            }, ckpt_path)
            print(f"  → Saved  bal_acc={best_bal_acc:.4f}")
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"\n[Early Stop] No improvement for {early_stop_patience} epochs. Stopping at ep {ep}.")
                break

    print(f"\n[DONE] Best balanced acc: {best_bal_acc:.4f}  |  Checkpoint: {ckpt_path}")

    # ARCHITECTURAL FIX: Update training_round and clear used_for_training flag
    # This ensures segments can be reused in future training runs with fresh annotations
    _update_training_metadata(ds, training_round_increment=True)


# ---------------------------------------------------------------------------
# FINE-TUNE (v2)
# ---------------------------------------------------------------------------
def run_finetune(task, num_epochs, batch_size, lr):
    print(f"\n{'='*65}")
    print(f"  V2 FINE-TUNE  |  task={task.upper()}  epochs={num_epochs}")
    print(f"{'='*65}")

    class_names = RHYTHM_CLASS_NAMES if task == "rhythm" else ECTOPY_CLASS_NAMES
    num_classes = len(class_names)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path   = CHECKPOINTS / f"best_model_{task}_v2.pth"

    if not ckpt_path.exists():
        print(f"[ERROR] No v2 checkpoint at {ckpt_path}")
        print("        Run --mode initial first.")
        sys.exit(1)

    ds = ECGEventDatasetV2(task=task, source_filter="cardiologist", augment=True)
    if len(ds) < 5:
        print(f"[ABORT] Only {len(ds)} cardiologist events.")
        return

    # Check for empty critical classes (fine-tune: warn only, never abort)
    label_counts_check = Counter(s[2] for s in ds.samples)
    critical_classes = [0, 1, 2] if task == "ectopy" else [0]
    empty_critical = [cls for cls in critical_classes if label_counts_check.get(cls, 0) == 0 and cls < len(class_names)]
    if empty_critical:
        cls_names_list = ECTOPY_CLASS_NAMES if task == "ectopy" else RHYTHM_CLASS_NAMES
        empty_names = [f"{cls}={cls_names_list[cls]}" for cls in empty_critical if cls < len(cls_names_list)]
        print(f"[WARN] Fine-tune has no cardiologist events for: {', '.join(empty_names)}")
        print(f"       Proceeding with available classes only.")

    train_idx, val_idx = filename_split(ds.samples)
    if len(val_idx) == 0:
        val_idx   = train_idx[-max(1, len(train_idx) // 10):]
        train_idx = train_idx[:-len(val_idx)]

    train_labels = [ds.samples[i][2] for i in train_idx]
    counts       = Counter(train_labels)

    print(f"\nCardiologist event distribution ({task}):")
    for i, name in enumerate(class_names):
        if counts.get(i, 0) > 0:
            print(f"  {i:02d}  {name:<40}  {counts[i]:>6}")

    # ── Load or refit feature scaler ─────────────────────────────────────────
    scaler_path = CHECKPOINTS / f"feature_scaler_{task}.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print(f"[Scaler] Loaded existing scaler from {scaler_path.name}")
    else:
        # Fallback: fit from fine-tune training data (e.g. if initial never ran)
        X_train_feats = np.stack([ds.samples[i][1] for i in train_idx])
        scaler = StandardScaler()
        scaler.fit(X_train_feats)
        joblib.dump(scaler, scaler_path)
        print(f"[Scaler] Fit from fine-tune data (no initial scaler found) → {scaler_path.name}")
    ds.scaler = scaler

    eff_batch = min(batch_size, max(2, len(train_idx)))

    sampler   = build_sampler(ds.samples, train_idx, num_classes, oversample_factor=10)
    train_ds  = torch.utils.data.Subset(ds, train_idx)
    val_ds    = torch.utils.data.Subset(ds, val_idx)
    train_ldr = DataLoader(train_ds, batch_size=eff_batch, sampler=sampler,  collate_fn=collate_fn_v2)
    val_ldr   = DataLoader(val_ds,   batch_size=max(1, eff_batch), shuffle=False, collate_fn=collate_fn_v2)

    # Load v2 checkpoint
    model = CNNTransformerWithFeatures(num_classes=num_classes, num_features=NUM_FEATURES).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    prev_acc = state.get("balanced_acc", 0.0)
    print(f"\nLoaded v2 checkpoint - prev bal_acc={prev_acc:.4f}  device={device}")

    criterion    = _build_criterion(train_labels, num_classes, device)
    total_params = sum(p.numel() for p in model.parameters())
    P1_EPOCHS    = max(1, num_epochs // 2)
    P2_EPOCHS    = num_epochs - P1_EPOCHS
    best_bal_acc = prev_acc

    # Phase 1: freeze signal pathway, train feature_net + classifier
    print(f"\n-- Phase 1: Feature net + Classifier ({P1_EPOCHS} epochs) --------")
    for name, param in model.named_parameters():
        param.requires_grad = ("classifier" in name or "feature_net" in name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable: {trainable:,} / {total_params:,} params")

    opt_p1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr * 2, weight_decay=5e-4,
    )
    for ep in range(1, P1_EPOCHS + 1):
        tr = train_epoch(model, opt_p1, criterion, train_ldr, device)
        va = eval_epoch(model, criterion, val_ldr, device, num_classes, class_names)
        print(f"  P1 ep {ep:02d}  loss={tr['loss']:.4f} acc={tr['acc']:.3f}  "
              f"val bal_acc={va['balanced_acc']:.3f}")
        if va["balanced_acc"] > best_bal_acc:
            best_bal_acc = va["balanced_acc"]
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "balanced_acc": best_bal_acc, "class_names": class_names,
                        "num_features": NUM_FEATURES, "feature_names": FEATURE_NAMES,
                        "mode": "finetune", "version": "v2"}, ckpt_path)
            print(f"  → Saved (improvement) bal_acc={best_bal_acc:.4f}")

    # Phase 2: unfreeze all
    print(f"\n-- Phase 2: Full model fine-tune ({P2_EPOCHS} epochs) --------")
    for p in model.parameters():
        p.requires_grad = True
    print(f"   Trainable: {total_params:,} / {total_params:,} params")

    opt_p2    = torch.optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p2, T_max=max(1, P2_EPOCHS))

    for ep in range(1, P2_EPOCHS + 1):
        tr = train_epoch(model, opt_p2, criterion, train_ldr, device)
        va = eval_epoch(model, criterion, val_ldr, device, num_classes, class_names)
        scheduler.step()
        print(f"  P2 ep {ep:02d}  loss={tr['loss']:.4f} acc={tr['acc']:.3f}  "
              f"val bal_acc={va['balanced_acc']:.3f}")
        if va["balanced_acc"] > best_bal_acc:
            best_bal_acc = va["balanced_acc"]
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "balanced_acc": best_bal_acc, "class_names": class_names,
                        "num_features": NUM_FEATURES, "feature_names": FEATURE_NAMES,
                        "mode": "finetune", "version": "v2"}, ckpt_path)
            print(f"  → Saved (improvement) bal_acc={best_bal_acc:.4f}")

    if best_bal_acc <= prev_acc:
        print(f"\n--  Fine-tune did NOT improve.  prev={prev_acc:.4f}  best={best_bal_acc:.4f}")
    else:
        print(f"\n[DONE] Improved: {prev_acc:.4f} → {best_bal_acc:.4f}")

    # Update training metadata
    _update_training_metadata(ds, training_round_increment=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ECG V2 Training (Signal + Features)")
    parser.add_argument("--task",   choices=["rhythm", "ectopy"], default="rhythm")
    parser.add_argument("--mode",   choices=["initial", "finetune"], default="finetune")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch",  type=int, default=32)
    parser.add_argument("--lr",     type=float, default=1e-4)  # 5e-4→1e-4: slower learning prevents overfitting
    parser.add_argument("--bootstrap-v1", action="store_true",
                        help="Bootstrap signal pathway weights from v1 checkpoint (initial mode only)")
    args = parser.parse_args()

    if args.epochs is None:
        args.epochs = 60 if args.mode == "initial" else 20  # 30→60: more epochs + early stopping handles overfitting

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS / f"retrain_v2_{args.task}_{args.mode}_{ts}.log"
    logger = TeeLogger(log_path)
    sys.stdout = logger

    print(f"[retrain_v2] {datetime.now().isoformat()}")
    print(f"[retrain_v2] task={args.task}  mode={args.mode}  epochs={args.epochs}  "
          f"batch={args.batch}  lr={args.lr}  bootstrap_v1={args.bootstrap_v1}")
    print(f"[retrain_v2] Log: {log_path}")
    print(f"[retrain_v2] Features: {NUM_FEATURES} dimensions")

    if args.mode == "initial":
        run_initial(args.task, args.epochs, args.batch, args.lr, args.bootstrap_v1)
    else:
        run_finetune(args.task, args.epochs, args.batch, args.lr)

    # Print data loading error summary
    if any(LOAD_ERRORS.values()):
        print("\n" + "="*65)
        print("DATA LOADING ERROR SUMMARY")
        print("="*65)
        for error_type, errors in LOAD_ERRORS.items():
            if errors:
                print(f"\n{error_type}: {len(errors)} occurrences")
                for i, err in enumerate(errors[:5]):  # Show first 5
                    print(f"  {i+1}. seg_id={err['segment_id']}: {err['error']}")
                if len(errors) > 5:
                    print(f"  ... and {len(errors) - 5} more")

    sys.stdout = logger.terminal
    logger.close()
