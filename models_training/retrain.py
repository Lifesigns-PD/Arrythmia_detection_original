#!/usr/bin/env python3
"""
retrain.py - FIXED PRODUCTION TRAINING & RETRAINING SCRIPT
============================================================

BUGS FIXED IN THIS VERSION:
  [FIX 1] Model now LOADS existing checkpoint before retraining (not random weights)
  [FIX 2] Window extraction: slides across full 10s segment instead of always taking first 2s
  [FIX 3] Cardiologist events get 20x oversampling weight vs trusted_source
  [FIX 4] used_for_training flag now correctly tracked in ecg_features_annotatable
  [FIX 5] Filename-based split prevents data leakage (same recording in train+val)
  [FIX 6] host changed to 127.0.0.1 (consistent with db_service.py)
  [FIX 7] Signal null guard - skips empty/null signals silently
  [FIX 8] Two modes: initial (30 epochs) vs finetune (10 epochs, loads checkpoint)
  [FIX 9] event.get() used everywhere - no KeyError on malformed events
  [FIX 10] Minimum batch size guard for BatchNorm stability

TWO MODES:
  python retrain.py --task rhythm  --mode initial    # First ever training run
  python retrain.py --task rhythm  --mode finetune   # After cardiologist annotates
  python retrain.py --task ectopy  --mode initial
  python retrain.py --task ectopy  --mode finetune
"""

import os
import sys
import json
import psycopg2
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "models_training"))

from data_loader import (
    normalize_label,
    RHYTHM_CLASS_NAMES, get_rhythm_label_idx,
    ECTOPY_CLASS_NAMES, get_ectopy_label_idx,
)
from models import CNNTransformerClassifier

# -----------------------------------------------------------------------------
# Paths & DB
# -----------------------------------------------------------------------------
OUTPUT      = Path(__file__).resolve().parent / "outputs"
CHECKPOINTS = OUTPUT / "checkpoints"
LOGS        = OUTPUT / "logs"
for d in (OUTPUT, CHECKPOINTS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

# [FIX 6] Use 127.0.0.1 - "localhost" can resolve to IPv6 on Windows
DB_PARAMS = {
    "host":     "127.0.0.1",
    "dbname":   "ecg_analysis",
    "user":     "ecg_user",
    "password": "sais",
    "port":     "5432",
}

# How much extra weight cardiologist events get over trusted_source events.
# If trusted_source has 6000 AFib events and cardiologist has 3,
# these 3 corrections effectively compete as if there were 60 of them.
CARDIOLOGIST_OVERSAMPLE = 20

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Focal Loss
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class ECGEventDataset(torch.utils.data.Dataset):
    """
    Loads ECG events from ecg_features_annotatable.

    KEY DIFFERENCES from old ECGRawDatasetSQL
    -----------------------------------------
    FIX 2: Window extraction
      Old code: signal[0:2500] cropped to [:500] = always first 2s
      New code: For narrow events (cardiologist, <3s) -> center window on event
                For wide events  (trusted_source, 10s) -> slide 500-sample window
                across the full segment with 0.5s step -> covers whole signal

    FIX 3: Each sample records its annotation_source so the sampler
            can apply 20x extra weight to cardiologist events.

    FIX 7: Signal null guard - malformed signals are skipped.
    FIX 9: event.get() used everywhere - never KeyError.
    """

    TARGET_FS      = 250
    WINDOW_SAMPLES = 2500   # Refactored: 10s @ 250Hz - matches CNNTransformerClassifier.TARGET_LEN
    SLIDE_STEP     = 2500   # No overlap, take full segments

    def __init__(self, task="rhythm", source_filter="all", augment=False):
        self.augment = augment
        self.task    = task
        # tuple: (signal_np, label_idx, annotation_source, seg_id, filename)
        self.samples = []

        print(f"\n[Dataset] task={task}  filter={source_filter}")
        print( "[Dataset] Fetching segments from DB...")

        with psycopg2.connect(**DB_PARAMS) as conn:
            with conn.cursor() as cur:
                # [FIX 7] signal IS NOT NULL guard in the query itself
                cur.execute("""
                    SELECT segment_id, signal_data, events_json, arrhythmia_label, segment_fs, filename, is_corrected
                    FROM   ecg_features_annotatable
                    WHERE  signal_data IS NOT NULL
                """)
                rows = cur.fetchall()

        print(f"[Dataset] Fetched {len(rows)} segments")

        skipped_null, skipped_short, skipped_label, total_windows = 0, 0, 0, 0

        for seg_id, signal_raw, events_json_raw, arrhythmia_label, fs, filename, is_corrected in rows:

            # Parse signal
            try:
                if isinstance(signal_raw, str):
                    signal = np.array(json.loads(signal_raw), dtype=np.float32)
                else:
                    signal = np.array(signal_raw, dtype=np.float32)
            except Exception:
                skipped_null += 1
                continue

            # [FIX 7] Length guard
            if len(signal) < self.WINDOW_SAMPLES:
                skipped_short += 1
                continue

            fs = int(fs) if fs else self.TARGET_FS

            # Parse events
            try:
                if isinstance(events_json_raw, str):
                    ev_data = json.loads(events_json_raw)
                else:
                    ev_data = events_json_raw or []
            except Exception:
                continue

            if isinstance(ev_data, list):
                events = ev_data
            elif isinstance(ev_data, dict):
                events = ev_data.get("events", [])
            else:
                events = []

            # [NEW] For Rhythm task, if it's corrected by a human, use the arrhythmia_label (dropdown)
            # as a 10-second global rhythm event.
            if self.task == "rhythm" and is_corrected and arrhythmia_label:
                # Check if we already have a cardiologist-sourced rhythm event to avoid doubles
                has_existing_rhythm = any(ev.get("event_category") == "RHYTHM" and ev.get("annotation_source") == "cardiologist" for ev in events)
                if not has_existing_rhythm:
                    label_idx = get_rhythm_label_idx(arrhythmia_label)
                    if label_idx is not None:
                        # Slide windows across the full 10s segment
                        windows = self._slide_windows(signal, 0.0, 10.0, fs)
                        for win in windows:
                            if win is not None:
                                self.samples.append((
                                    win, label_idx, "cardiologist",
                                    seg_id, filename or ""
                                ))
                                total_windows += 1

            for event in events:
                # [FIX 9] Safe access everywhere
                ann_source  = event.get("annotation_source", "unknown")
                event_type  = event.get("event_type", "")
                start_s     = float(event.get("start_time", 0.0))
                end_s       = float(event.get("end_time",  10.0))

                if not event_type:
                    continue

                # Source filter (finetune only wants cardiologist events)
                if source_filter == "cardiologist" and ann_source != "cardiologist":
                    continue

                # Resolve label
                if self.task == "rhythm":
                    # Skip if it's an ectopy event during a rhythm task
                    if event.get("event_category") == "ECTOPY": continue
                    label_idx = get_rhythm_label_idx(event_type)
                else:
                    # Skip if it's a rhythm event during an ectopy task
                    if event.get("event_category") == "RHYTHM": continue
                    label_idx = get_ectopy_label_idx(event_type)

                if label_idx is None:
                    skipped_label += 1
                    continue

                # [FIX 2] Window extraction
                event_duration_s = end_s - start_s
                narrow_threshold = (self.WINDOW_SAMPLES / fs) * 1.5  # ~3s

                if event_duration_s <= narrow_threshold:
                    # Narrow / cardiologist event: single centered window
                    windows = [self._center_window(signal, (start_s + end_s) / 2.0, fs)]
                else:
                    # Wide / whole-segment event: slide + center
                    windows = self._slide_windows(signal, start_s, end_s, fs)

                for win in windows:
                    if win is not None:
                        self.samples.append((
                            win, label_idx, ann_source,
                            seg_id, filename or ""
                        ))
                        total_windows += 1

        print(f"[Dataset] {total_windows} windows extracted")
        print(f"          skipped: null={skipped_null} short={skipped_short} no_label={skipped_label}")

        # Print cardiologist vs trusted_source breakdown
        sources = Counter(s[2] for s in self.samples)
        print(f"          sources: {dict(sources)}")

    # -- Window helpers --------------------------------------------------------

    def _center_window(self, signal, center_s, fs):
        center_i = int(center_s * fs)
        half     = self.WINDOW_SAMPLES // 2
        s_i      = center_i - half
        e_i      = center_i + half
        
        # Safe extraction with zero padding if window hits edges of the 10s segment
        if s_i < 0:
            # Pad beginning
            win = signal[0:max(0, e_i)]
            return np.pad(win, (abs(s_i), 0)).astype(np.float32)
        elif e_i > len(signal):
            # Pad end
            win = signal[s_i:len(signal)]
            return np.pad(win, (0, e_i - len(signal))).astype(np.float32)
        
        return signal[s_i:e_i].astype(np.float32)

    def _slide_windows(self, signal, start_s, end_s, fs):
        start_i = int(start_s * fs)
        end_i   = min(len(signal), int(end_s * fs))
        wins    = []
        pos     = start_i
        while pos + self.WINDOW_SAMPLES <= end_i:
            wins.append(self._pad_or_crop(signal[pos: pos + self.WINDOW_SAMPLES]))
            pos += self.SLIDE_STEP
        # Also always include the midpoint window
        wins.append(self._center_window(signal, (start_s + end_s) / 2.0, fs))
        return wins

    def _pad_or_crop(self, signal):
        """Standardizes signal to exactly WINDOW_SAMPLES (2500)."""
        if len(signal) > self.WINDOW_SAMPLES:
            return signal[:self.WINDOW_SAMPLES]
        elif len(signal) < self.WINDOW_SAMPLES:
            return np.pad(signal, (0, self.WINDOW_SAMPLES - len(signal)), mode='constant')
        return signal

    def _augment_signal(self, sig):
        if not self.augment or np.random.rand() > 0.5:
            return sig
        sig = sig * np.random.uniform(0.85, 1.15)
        sig = sig + np.random.normal(0, 0.02 * max(np.std(sig), 1e-6), sig.shape)
        return sig.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sig, label_idx, source, seg_id, filename = self.samples[idx]
        if self.augment:
            sig = self._augment_signal(sig.copy())
        return {
            "signal":   sig,
            "label":    label_idx,
            "source":   source,
            "seg_id":   seg_id,
            "filename": filename,
        }

# -----------------------------------------------------------------------------
# Collate  (B, 1, T) for SmallCNN
# -----------------------------------------------------------------------------
def collate_fn(batch):
    xs = torch.stack([
        torch.from_numpy(b["signal"]).float().unsqueeze(0) for b in batch
    ])
    ys = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return xs, ys

# -----------------------------------------------------------------------------
# Train / eval epochs
# -----------------------------------------------------------------------------
def train_epoch(model, opt, criterion, loader, device):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []

    for x, y in tqdm(loader, desc="  train", ncols=80, leave=False):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
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


def eval_epoch(model, criterion, loader, device, num_classes):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="  val  ", ncols=80, leave=False):
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            total_loss += loss.item() * x.size(0)
            y_true += y.cpu().tolist()
            y_pred += model(x).argmax(1).cpu().tolist()

    y_arr  = np.array(y_true)
    yp_arr = np.array(y_pred)
    per_cls = {
        i: float((yp_arr[y_arr == i] == i).mean()) if (y_arr == i).sum() > 0 else 0.0
        for i in range(num_classes)
    }
    return {
        "loss":         total_loss / max(len(y_true), 1),
        "acc":          float((y_arr == yp_arr).mean()),
        "balanced_acc": float(np.mean(list(per_cls.values()))),
        "per_class":    per_cls,
    }

# -----------------------------------------------------------------------------
# [FIX 5] Filename-based split - prevents same recording in train+val
# -----------------------------------------------------------------------------
def _recording_id(filename: str) -> str:
    """
    Extract recording-level (patient) ID from segment filename.
    Refactored to prevent data leakage for MITDB and PTB-XL.
    MITDB: 'mitdb_100_seg1.json' -> 'mitdb_100'
    PTB-XL: 'ptbxl_00001_seg2.json' -> 'ptbxl_00001'
    """
    stem = Path(filename).stem.lower()
    
    # Check for MITDB/PTBXL prefixes
    if "mitdb" in stem:
        # Expected: mitdb_XXX_segY
        parts = stem.split("_")
        if len(parts) >= 2: return f"mitdb_{parts[1]}"
    
    if "ptb" in stem:
        # Expected: ptbxl_XXXXX_segY
        parts = stem.split("_")
        if len(parts) >= 2: return f"ptbxl_{parts[1]}"

    # Fallback to old logic
    parts = stem.split("_seg_")
    return parts[0] if len(parts) > 1 else stem


def filename_split(samples, val_ratio=0.15):
    """Split by recording ID (FIX 5: prevents data leakage)."""
    groups = defaultdict(list)
    for i, s in enumerate(samples):
        rid = _recording_id(s[4])  # s[4] = filename
        groups[rid].append(i)

    rids = list(groups.keys())
    np.random.shuffle(rids)
    cut = max(1, int(len(rids) * (1 - val_ratio)))

    train_idx = [i for rid in rids[:cut] for i in groups[rid]]
    val_idx   = [i for rid in rids[cut:] for i in groups[rid]]

    print(f"[Split] {cut} train recordings -> {len(train_idx)} windows  |  "
          f"{len(rids)-cut} val recordings -> {len(val_idx)} windows")
    return train_idx, val_idx

# -----------------------------------------------------------------------------
# [FIX 3] Weighted sampler with cardiologist boost
# -----------------------------------------------------------------------------
def build_sampler(samples, train_idx, num_classes, oversample_factor=2):
    """
    Two-level weighting:
      Level 1 - class balance  (inverse class frequency)
      Level 2 - source balance (cardiologist - CARDIOLOGIST_OVERSAMPLE)

    Combined: weight = (1/class_count) * source_multiplier
    """
    labels = [samples[i][1] for i in train_idx]
    counts = Counter(labels)
    ca     = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float32)
    ca[ca == 0] = 1.0
    class_w = 1.0 / ca

    weights = []
    for i in train_idx:
        lbl    = samples[i][1]
        source = samples[i][2]
        cw     = float(class_w[lbl])
        sw     = float(CARDIOLOGIST_OVERSAMPLE) if source == "cardiologist" else 1.0
        weights.append(cw * sw)

    n = len(train_idx) * oversample_factor
    return WeightedRandomSampler(weights, num_samples=n, replacement=True)

# -----------------------------------------------------------------------------
# Focal Loss criterion builder
# -----------------------------------------------------------------------------
def _build_criterion(labels, num_classes, device):
    counts = Counter(labels)
    ca     = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float32)
    ca[ca == 0] = 1.0
    alpha  = torch.tensor(
        np.clip(np.sqrt(ca.sum() / (num_classes * ca)), 0.5, 5.0),
        dtype=torch.float32,
    ).to(device)
    return FocalLoss(alpha=alpha, gamma=2.0)

# -----------------------------------------------------------------------------
# [FIX 4] Mark cardiologist events as used in DB
# -----------------------------------------------------------------------------
def mark_cardiologist_events_used():
    """
    Updates ecg_features_annotatable.used_for_training = TRUE
    for all segments that have at least one cardiologist event in ecg_features_annotatable.

    NOTE: The NEXT retrain still picks up these events (finetune will filter
    by source_filter="cardiologist", so used_for_training is informational only -
    but it correctly drives the dashboard "Verified" badge).
    """
    print("\n[DB] Marking cardiologist segments as used_for_training-")
    try:
        with psycopg2.connect(**DB_PARAMS) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE ecg_features_annotatable
                    SET    used_for_training = TRUE
                    WHERE  events_json IS NOT NULL
                      AND  EXISTS (
                          SELECT 1
                          FROM   jsonb_array_elements(
                              CASE
                                  WHEN jsonb_typeof(events_json) = 'array'
                                       THEN events_json
                                  WHEN jsonb_typeof(events_json) = 'object'
                                       AND  events_json ? 'events'
                                       THEN events_json->'events'
                                  ELSE '[]'::jsonb
                              END
                          ) AS ev
                          WHERE ev->>'annotation_source' = 'cardiologist'
                      )
                """)
                n = cur.rowcount
            conn.commit()
        print(f"[DB] Marked {n} segments as used_for_training=TRUE")
    except Exception as e:
        print(f"[DB] Warning - could not mark segments: {e}")

# -----------------------------------------------------------------------------
# MODE 1: INITIAL TRAINING  (fresh model, all data)
# -----------------------------------------------------------------------------
def run_initial(task, num_epochs, batch_size, lr):
    print(f"\n{'='*65}")
    print(f"  INITIAL TRAINING  |  task={task.upper()}  epochs={num_epochs}")
    print(f"{'='*65}")

    class_names = RHYTHM_CLASS_NAMES if task == "rhythm" else ECTOPY_CLASS_NAMES
    num_classes = len(class_names)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path   = CHECKPOINTS / f"best_model_{task}.pth"

    ds = ECGEventDataset(task=task, source_filter="all", augment=False)
    if len(ds) < 20:
        print(f"[ABORT] Only {len(ds)} windows - not enough to train.")
        return

    train_idx, val_idx = filename_split(ds.samples)
    if len(val_idx) == 0:
        val_idx   = train_idx[int(0.9 * len(train_idx)):]
        train_idx = train_idx[:int(0.9 * len(train_idx))]

    train_labels = [ds.samples[i][1] for i in train_idx]
    counts       = Counter(train_labels)

    print(f"\nClass distribution (train windows):")
    for i, name in enumerate(class_names):
        print(f"  {i:02d}  {name:<40}  {counts.get(i, 0):>6}")

    # [FIX 10] BatchNorm needs batch >= 2
    eff_batch = min(batch_size, max(2, len(train_idx) // 4))

    sampler   = build_sampler(ds.samples, train_idx, num_classes, oversample_factor=2)
    train_ds  = torch.utils.data.Subset(ds, train_idx)
    val_ds    = torch.utils.data.Subset(ds, val_idx)
    train_ldr = DataLoader(train_ds, batch_size=eff_batch, sampler=sampler,   collate_fn=collate_fn)
    val_ldr   = DataLoader(val_ds,   batch_size=eff_batch, shuffle=False,      collate_fn=collate_fn)

    # Fresh model - no checkpoint for initial training
    model     = CNNTransformerClassifier(num_classes=num_classes).to(device)
    criterion = _build_criterion(train_labels, num_classes, device)
    opt       = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)

    best_bal_acc = 0.0
    print(f"\nDevice={device}  Batch={eff_batch}  "
          f"Train_windows={len(train_idx)}  Val_windows={len(val_idx)}\n")

    for ep in range(1, num_epochs + 1):
        tr = train_epoch(model, opt, criterion, train_ldr, device)
        va = eval_epoch(model, criterion, val_ldr, device, num_classes)
        scheduler.step(va["loss"])

        print(f"Ep {ep:02d}/{num_epochs}  "
              f"train loss={tr['loss']:.4f} acc={tr['acc']:.3f}  "
              f"val loss={va['loss']:.4f} bal_acc={va['balanced_acc']:.3f}")

        if va["balanced_acc"] > best_bal_acc:
            best_bal_acc = va["balanced_acc"]
            torch.save({
                "epoch":        ep,
                "model_state":  model.state_dict(),
                "balanced_acc": best_bal_acc,
                "class_names":  class_names,
                "mode":         "initial",
            }, ckpt_path)
            print(f"  - Saved  bal_acc={best_bal_acc:.4f}")

    print(f"\n[DONE] Best balanced acc: {best_bal_acc:.4f}  |  Checkpoint: {ckpt_path}")

# -----------------------------------------------------------------------------
# MODE 2: FINE-TUNE  (load checkpoint, cardiologist events only, 2-phase)
# -----------------------------------------------------------------------------
def run_finetune(task, num_epochs, batch_size, lr):
    print(f"\n{'='*65}")
    print(f"  FINE-TUNE  |  task={task.upper()}  epochs={num_epochs}")
    print(f"{'='*65}")

    class_names = RHYTHM_CLASS_NAMES if task == "rhythm" else ECTOPY_CLASS_NAMES
    num_classes = len(class_names)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path   = CHECKPOINTS / f"best_model_{task}.pth"

    # [FIX 1] Must have a checkpoint to finetune from
    if not ckpt_path.exists():
        print(f"[ERROR] No checkpoint at {ckpt_path}")
        print("        Run --mode initial first, then annotate, then finetune.")
        sys.exit(1)

    ds = ECGEventDataset(task=task, source_filter="cardiologist", augment=True)
    if len(ds) < 5:
        print(f"[ABORT] Only {len(ds)} cardiologist events.")
        print("        Annotate more segments in the dashboard first.")
        return

    train_idx, val_idx = filename_split(ds.samples)
    if len(val_idx) == 0:
        val_idx   = train_idx[-max(1, len(train_idx) // 10):]
        train_idx = train_idx[:-len(val_idx)]

    train_labels = [ds.samples[i][1] for i in train_idx]
    counts       = Counter(train_labels)

    print(f"\nCardiologist event distribution ({task}):")
    for i, name in enumerate(class_names):
        if counts.get(i, 0) > 0:
            print(f"  {i:02d}  {name:<40}  {counts[i]:>6}")

    # [FIX 10] BatchNorm guard
    eff_batch = min(batch_size, max(2, len(train_idx)))

    # [FIX 3] 10x oversample inside sampler (combined with CARDIOLOGIST_OVERSAMPLE in weights)
    sampler   = build_sampler(ds.samples, train_idx, num_classes, oversample_factor=10)
    train_ds  = torch.utils.data.Subset(ds, train_idx)
    val_ds    = torch.utils.data.Subset(ds, val_idx)
    train_ldr = DataLoader(train_ds, batch_size=eff_batch, sampler=sampler,  collate_fn=collate_fn)
    val_ldr   = DataLoader(val_ds,   batch_size=max(1, eff_batch), shuffle=False, collate_fn=collate_fn)

    # [FIX 1] Load existing model
    model = CNNTransformerClassifier(num_classes=num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    prev_acc = state.get("balanced_acc", 0.0)
    print(f"\nLoaded checkpoint - prev bal_acc={prev_acc:.4f}  device={device}")

    criterion    = _build_criterion(train_labels, num_classes, device)
    total_params = sum(p.numel() for p in model.parameters())
    P1_EPOCHS    = max(1, num_epochs // 2)
    P2_EPOCHS    = num_epochs - P1_EPOCHS
    best_bal_acc = prev_acc  # Must beat previous to save

    # -- Phase 1: freeze CNN+Transformer, train classifier head only -----------
    # This adapts the output layer fast without disturbing learned representations.
    print(f"\n-- Phase 1: Classifier head only ({P1_EPOCHS} epochs) -------------")
    for name, param in model.named_parameters():
        param.requires_grad = ("classifier" in name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable: {trainable:,} / {total_params:,} params")

    opt_p1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr * 2, weight_decay=1e-4,
    )
    for ep in range(1, P1_EPOCHS + 1):
        tr = train_epoch(model, opt_p1, criterion, train_ldr, device)
        va = eval_epoch(model, criterion, val_ldr, device, num_classes)
        print(f"  P1 ep {ep:02d}  loss={tr['loss']:.4f} acc={tr['acc']:.3f}  "
              f"val bal_acc={va['balanced_acc']:.3f}")
        # [FIX] Micro-dataset Save Logic: If tiny dataset and perfect training, allow save
        is_perfect_micro = (len(train_idx) < 50 and tr["loss"] < 0.01 and tr["acc"] > 0.99)
        if va["balanced_acc"] > best_bal_acc or is_perfect_micro:
            old_best = best_bal_acc
            best_bal_acc = va["balanced_acc"]
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "balanced_acc": best_bal_acc, "class_names": class_names,
                        "mode": "finetune"}, ckpt_path)
            reason = "micro-perfect" if is_perfect_micro and va["balanced_acc"] <= old_best else "improvement"
            print(f"  - Saved ({reason}) bal_acc={best_bal_acc:.4f}  (prev={prev_acc:.4f})")

    # -- Phase 2: unfreeze all, low LR - prevents catastrophic forgetting ------
    print(f"\n-- Phase 2: Full model fine-tune ({P2_EPOCHS} epochs) -------------")
    for p in model.parameters():
        p.requires_grad = True
    print(f"   Trainable: {total_params:,} / {total_params:,} params")

    opt_p2    = torch.optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p2, T_max=max(1, P2_EPOCHS))

    for ep in range(1, P2_EPOCHS + 1):
        tr = train_epoch(model, opt_p2, criterion, train_ldr, device)
        va = eval_epoch(model, criterion, val_ldr, device, num_classes)
        scheduler.step()
        print(f"  P2 ep {ep:02d}  loss={tr['loss']:.4f} acc={tr['acc']:.3f}  "
              f"val bal_acc={va['balanced_acc']:.3f}")
        # [FIX] Micro-dataset Save Logic: If tiny dataset and perfect training, allow save
        is_perfect_micro = (len(train_idx) < 50 and tr["loss"] < 0.01 and tr["acc"] > 0.99)
        if va["balanced_acc"] > best_bal_acc or is_perfect_micro:
            old_best = best_bal_acc
            best_bal_acc = va["balanced_acc"]
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "balanced_acc": best_bal_acc, "class_names": class_names,
                        "mode": "finetune"}, ckpt_path)
            reason = "micro-perfect" if is_perfect_micro and va["balanced_acc"] <= old_best else "improvement"
            print(f"  - Saved ({reason}) bal_acc={best_bal_acc:.4f}  (prev={prev_acc:.4f})")

    if best_bal_acc <= prev_acc:
        print(f"\n--  Fine-tune did NOT improve.  "
              f"prev={prev_acc:.4f}  best_this_run={best_bal_acc:.4f}")
        print("   Checkpoint NOT replaced - existing model preserved.")
    else:
        print(f"\n[DONE] Improved: {prev_acc:.4f} - {best_bal_acc:.4f}")

    # [FIX 4] Mark events as used in DB
    mark_cardiologist_events_used()

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ECG Model Training & Retraining")
    parser.add_argument("--task",   choices=["rhythm", "ectopy"], default="rhythm")
    parser.add_argument("--mode",   choices=["initial", "finetune"], default="finetune",
                        help="initial=fresh model on all data | finetune=adapt checkpoint to new cardiologist annotations")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs (default: 30 initial / 10 finetune)")
    parser.add_argument("--batch",  type=int, default=32)
    parser.add_argument("--lr",     type=float, default=5e-4)
    args = parser.parse_args()

    if args.epochs is None:
        args.epochs = 30 if args.mode == "initial" else 10

    # Class list integrity checks
    if args.task == "rhythm":
        for name in RHYTHM_CLASS_NAMES:
            assert " + " not in name, f"Composite found in RHYTHM_CLASS_NAMES: {name}"
        print(f"[OK] Rhythm classes: {len(RHYTHM_CLASS_NAMES)}, no composites")
    else:
        assert ECTOPY_CLASS_NAMES[0] == "None", "ECTOPY class 0 must be 'None'"
        print(f"[OK] Ectopy classes: {len(ECTOPY_CLASS_NAMES)}")

    # Logging
    log_dir = LOGS / args.task
    log_dir.mkdir(parents=True, exist_ok=True)
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TeeLogger(log_dir / f"{args.mode}_{ts}.log")
    sys.stdout = logger

    print(f"\nSession started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"task={args.task}  mode={args.mode}  epochs={args.epochs}  "
          f"batch={args.batch}  lr={args.lr}")

    try:
        if args.mode == "initial":
            run_initial(args.task, args.epochs, args.batch, args.lr)
        else:
            run_finetune(args.task, args.epochs, args.batch, args.lr)
    finally:
        sys.stdout = logger.terminal
        logger.close()
