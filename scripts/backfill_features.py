#!/usr/bin/env python3
"""
backfill_features.py — Populate features_json with V3 ECG features (60 dimensions)
====================================================================================
SAFE: Only updates features_json. Does NOT touch signal_data, labels, or events.

Detects V3 features by the presence of 'sdnn_ms' key.
Segments already having V3 features are skipped (idempotent).

Usage:
    python scripts/backfill_features.py
    python scripts/backfill_features.py --limit 100   # test on 100 first
    python scripts/backfill_features.py --force        # re-extract everything
    python scripts/backfill_features.py --corrected    # only is_corrected=TRUE
"""

import sys
import json
import psycopg2
import numpy as np
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

DB_PARAMS = {
    "host":     "127.0.0.1",
    "dbname":   "ecg_analysis",
    "user":     "ecg_user",
    "password": "sais",
    "port":     "5432",
}

TARGET_FS  = 125
TARGET_LEN = 1250


def _resample_and_fix(sig, orig_fs):
    if orig_fs != TARGET_FS and len(sig) > 1:
        from scipy.signal import resample as sp_resample
        sig = sp_resample(sig, int(len(sig) * TARGET_FS / orig_fs)).astype(np.float32)
    if len(sig) < TARGET_LEN:
        sig = np.pad(sig, (0, TARGET_LEN - len(sig)))
    elif len(sig) > TARGET_LEN:
        sig = sig[:TARGET_LEN]
    return sig.astype(np.float32)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backfill features_json with V3 features")
    parser.add_argument("--limit",     type=int,  default=None)
    parser.add_argument("--force",     action="store_true", help="Re-extract even if V3 features exist")
    parser.add_argument("--corrected", action="store_true", help="Only process is_corrected=TRUE segments")
    parser.add_argument("--source",    type=str,  default=None, help="Filter by dataset_source (e.g. mitdb, ptbxl)")
    args = parser.parse_args()

    # Import V3 pipeline
    from signal_processing_v3 import process_ecg_v3
    from signal_processing_v3.features.extraction import FEATURE_NAMES_V3

    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = True
    cur  = conn.cursor()

    # Build query
    where = ["signal_data IS NOT NULL"]
    if not args.force:
        # Skip segments that already have V3 features (sdnn_ms is a V3-only key)
        where.append("(features_json IS NULL OR NOT (features_json ? 'sdnn_ms'))")
    if args.corrected:
        where.append("is_corrected = TRUE")
    if args.source:
        where.append(f"dataset_source = '{args.source}'")

    query = ("SELECT segment_id, signal_data, segment_fs, features_json "
             "FROM ecg_features_annotatable "
             f"WHERE {' AND '.join(where)} "
             "ORDER BY segment_id")
    if args.limit:
        query += f" LIMIT {args.limit}"

    cur.execute(query)
    rows = cur.fetchall()

    print(f"[Backfill V3] {len(rows)} segments to process  "
          f"({'force' if args.force else 'skip existing'})")
    if not rows:
        print("Nothing to do. Use --force to re-extract.")
        return

    updated = errors = skipped = 0

    for seg_id, signal_raw, fs, feat_json_raw in tqdm(rows, desc="V3 features"):
        try:
            sig = np.array(signal_raw, dtype=np.float32)
            fs  = int(fs or TARGET_FS)
            sig = _resample_and_fix(sig, fs)

            # Preserve existing metadata (r_peaks, etc.)
            existing = {}
            if feat_json_raw:
                try:
                    existing = json.loads(feat_json_raw) if isinstance(feat_json_raw, str) else (feat_json_raw or {})
                except Exception:
                    existing = {}

            # Run V3 pipeline (min_quality=0 so we always get features)
            result = process_ecg_v3(sig, fs=TARGET_FS, min_quality=0.0)

            # Build feature dict to store
            feat_dict = {k: (float(v) if v is not None else None)
                         for k, v in result["features"].items()}
            feat_dict["sqi_v3"]   = result["sqi"]
            feat_dict["method_v3"] = result["method"]
            # Store r_peaks for future use
            if len(result["r_peaks"]) > 0:
                feat_dict["r_peaks"] = result["r_peaks"].tolist()

            # Merge: V3 features overwrite V2, but keep any extra existing keys
            merged = {**existing, **feat_dict}

            cur.execute(
                "UPDATE ecg_features_annotatable SET features_json = %s WHERE segment_id = %s",
                (json.dumps(merged), seg_id)
            )
            updated += 1

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"\n[Error] seg {seg_id}: {e}")

    cur.close()
    conn.close()

    print(f"\n[Backfill V3] Done!")
    print(f"  Updated : {updated}")
    print(f"  Errors  : {errors}")
    print(f"  Features: {len(FEATURE_NAMES_V3)} dimensions")
    print(f"\nNext step:")
    print(f"  python models_training/retrain_v2.py --task rhythm --mode finetune")
    print(f"  (ectopy model does not need retraining — it takes raw signal, not features)")


if __name__ == "__main__":
    main()
