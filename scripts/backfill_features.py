#!/usr/bin/env python3
"""
backfill_features.py — Populate features_json with extracted ECG features
==========================================================================

SAFE: Only UPDATES the existing features_json column. Does NOT modify
      signal_data, arrhythmia_label, events_json, or any other column.

This script:
  1. Reads all segments from ecg_features_annotatable that have signal_data
  2. Extracts 13 numeric features from each signal (HR, QRS, QTc, ST, etc.)
  3. Merges the new features into the existing features_json (preserves r_peaks)
  4. Updates the database

Run ONCE before training with retrain_v2.py. Safe to re-run (idempotent).

Usage:
    python scripts/backfill_features.py
    python scripts/backfill_features.py --limit 100    # test on 100 segments first
    python scripts/backfill_features.py --force         # re-extract even if already done
"""

import sys
import json
import psycopg2
import numpy as np
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from signal_processing.feature_extraction import extract_feature_dict, FEATURE_NAMES
from signal_processing.cleaning import clean_signal

DB_PARAMS = {
    "host":     "127.0.0.1",
    "dbname":   "ecg_analysis",
    "user":     "ecg_user",
    "password": "sais",
    "port":     "5432",
}

TARGET_FS = 125
TARGET_LEN = 1250


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backfill features_json with extracted ECG features")
    parser.add_argument("--limit", type=int, default=None, help="Process only N segments (for testing)")
    parser.add_argument("--force", action="store_true", help="Re-extract even if features already exist")
    args = parser.parse_args()

    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = True
    cur = conn.cursor()

    # Fetch segments
    query = """
        SELECT segment_id, signal_data, segment_fs, features_json
        FROM ecg_features_annotatable
        WHERE signal_data IS NOT NULL
    """
    if not args.force:
        # Skip segments that already have our features
        query += " AND (features_json IS NULL OR NOT (features_json ? 'mean_hr_bpm'))"

    query += " ORDER BY segment_id"
    if args.limit:
        query += f" LIMIT {args.limit}"

    cur.execute(query)
    rows = cur.fetchall()

    print(f"[Backfill] Found {len(rows)} segments to process")
    if len(rows) == 0:
        print("[Backfill] Nothing to do. Use --force to re-extract.")
        cur.close()
        conn.close()
        return

    updated = 0
    errors = 0

    for seg_id, signal_raw, fs, existing_features_raw in tqdm(rows, desc="Extracting features"):
        try:
            # Parse signal
            sig = np.array(signal_raw, dtype=np.float32)
            fs = int(fs or TARGET_FS)

            # Resample if needed
            if fs != TARGET_FS and len(sig) > 1:
                from scipy.signal import resample
                new_len = int(len(sig) * TARGET_FS / fs)
                sig = resample(sig, new_len).astype(np.float32)

            # Pad/truncate
            if len(sig) < TARGET_LEN:
                sig = np.pad(sig, (0, TARGET_LEN - len(sig)))
            elif len(sig) > TARGET_LEN:
                sig = sig[:TARGET_LEN]

            # Clean signal
            sig = clean_signal(sig, TARGET_FS).astype(np.float32)

            # Get existing r_peaks if available
            r_peaks = None
            existing_dict = {}
            if existing_features_raw:
                try:
                    if isinstance(existing_features_raw, str):
                        existing_dict = json.loads(existing_features_raw)
                    else:
                        existing_dict = existing_features_raw or {}
                    rp = existing_dict.get("r_peaks", [])
                    if rp:
                        r_peaks = np.array(rp, dtype=int)
                except Exception:
                    existing_dict = {}

            # Extract features
            new_features = extract_feature_dict(sig, fs=TARGET_FS, r_peaks=r_peaks)

            # Merge: keep existing keys (like r_peaks), add new features
            merged = {**existing_dict, **new_features}

            # Update database
            cur.execute(
                "UPDATE ecg_features_annotatable SET features_json = %s WHERE segment_id = %s",
                (json.dumps(merged), seg_id)
            )
            updated += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n[Error] segment_id={seg_id}: {e}")

    cur.close()
    conn.close()

    print(f"\n[Backfill] Done!")
    print(f"  Updated:  {updated}")
    print(f"  Errors:   {errors}")
    print(f"  Features: {', '.join(FEATURE_NAMES)}")
    print(f"\nYou can now train with: python models_training/retrain_v2.py --task rhythm --mode initial")


if __name__ == "__main__":
    main()
