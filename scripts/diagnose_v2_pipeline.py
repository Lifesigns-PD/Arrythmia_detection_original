#!/usr/bin/env python3
"""
diagnose_v2_pipeline.py
========================
Comprehensive diagnostic tool to verify V2 pipeline connectivity:
  1. Dashboard → Database connection
  2. Database → Data Loader pipeline
  3. Data integrity checks
  4. Feature availability
  5. Label consistency

Usage:
  python scripts/diagnose_v2_pipeline.py
  python scripts/diagnose_v2_pipeline.py --detailed
  python scripts/diagnose_v2_pipeline.py --fix-backfill
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2
from models_training.data_loader import (
    get_rhythm_label_idx, get_ectopy_label_idx, RHYTHM_CLASS_NAMES, ECTOPY_CLASS_NAMES
)

DB_PARAMS = {
    "host": "127.0.0.1",
    "dbname": "ecg_analysis",
    "user": "ecg_user",
    "password": "sais",
    "port": "5432",
}

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")

def print_check(name, status, details=""):
    status_color = Colors.GREEN if status else Colors.RED
    status_text = "✅ PASS" if status else "❌ FAIL"
    symbol = "✓" if status else "✗"
    print(f"  {symbol} {name:50s} {status_color}{status_text:15s}{Colors.RESET}")
    if details:
        print(f"    └─ {details}")

def check_database_connection():
    """Verify database is accessible"""
    print_header("1. Database Connection")

    try:
        conn = psycopg2.connect(**DB_PARAMS)
        conn.close()
        print_check("Database Connection", True, "PostgreSQL 127.0.0.1:5432")
        return True
    except Exception as e:
        print_check("Database Connection", False, str(e))
        return False

def check_table_schema():
    """Verify ecg_features_annotatable table exists and has required columns"""
    print_header("2. Database Schema")

    required_columns = [
        "segment_id", "signal_data", "arrhythmia_label", "events_json",
        "is_corrected", "used_for_training", "features_json", "corrected_at",
        "r_peaks_in_segment", "segment_fs", "filename", "training_round"
    ]

    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        # Check table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'ecg_features_annotatable'
            )
        """)
        exists = cur.fetchone()[0]
        print_check("Table exists: ecg_features_annotatable", exists)

        if not exists:
            conn.close()
            return False

        # Check columns
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'ecg_features_annotatable'
        """)
        columns = [row[0] for row in cur.fetchall()]

        missing = [c for c in required_columns if c not in columns]
        if missing:
            print_check("All required columns", False, f"Missing: {missing}")
        else:
            print_check("All required columns", True, f"{len(required_columns)} columns found")

        conn.close()
        return len(missing) == 0

    except Exception as e:
        print_check("Table Schema", False, str(e))
        return False

def check_annotated_data():
    """Check count of cardiologist-verified segments"""
    print_header("3. Annotated Data Status")

    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        # Count by is_corrected flag
        cur.execute("""
            SELECT is_corrected, COUNT(*) as cnt
            FROM ecg_features_annotatable
            GROUP BY is_corrected
            ORDER BY is_corrected
        """)
        rows = cur.fetchall()

        total = sum(row[1] for row in rows)
        corrected = sum(row[1] for row in rows if row[0])
        uncorrected = sum(row[1] for row in rows if not row[0])

        print(f"  Total segments: {total}")
        print(f"    ✓ Cardiologist-verified (is_corrected=TRUE): {corrected}")
        print(f"    ○ Not verified (is_corrected=FALSE): {uncorrected}")

        print_check("Verified segments exist", corrected > 0, f"{corrected} segments ready for training")

        # Check recent annotations
        cur.execute("""
            SELECT COUNT(*) as cnt
            FROM ecg_features_annotatable
            WHERE corrected_at > NOW() - INTERVAL '24 hours'
        """)
        recent = cur.fetchone()[0]
        print_check("Recent annotations (< 24h)", recent > 0, f"{recent} segments annotated in last 24 hours")

        conn.close()
        return corrected > 0

    except Exception as e:
        print_check("Annotated Data Status", False, str(e))
        return False

def check_feature_vectors():
    """Check if feature vectors have been backfilled"""
    print_header("4. Feature Vector Status (Backfill Check)")

    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        cur.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN features_json IS NOT NULL THEN 1 ELSE 0 END) as with_features,
                SUM(CASE WHEN features_json->>'mean_hr_bpm' IS NOT NULL THEN 1 ELSE 0 END) as with_hr,
                SUM(CASE WHEN (features_json->>'pr_interval_ms')::float < 100
                         OR (features_json->>'pr_interval_ms')::float > 400 THEN 1 ELSE 0 END) as corrupted_pr
            FROM ecg_features_annotatable
        """)
        total, with_features, with_hr, corrupted_pr = cur.fetchone()

        print(f"  Total segments: {total}")
        print(f"    ✓ With features_json: {with_features} ({100*with_features//total if total else 0}%)")
        print(f"    ✓ With HR measurements: {with_hr}")
        print(f"    ⚠ Corrupted PR intervals (< 100ms or > 400ms): {corrupted_pr}")

        backfill_complete = (with_features == total) and (corrupted_pr == 0)
        print_check("Feature backfill complete", backfill_complete,
                   "Run 'python scripts/backfill_features.py' if incomplete")

        conn.close()
        return backfill_complete

    except Exception as e:
        print_check("Feature Vector Status", False, str(e))
        return False

def check_label_consistency():
    """Check for invalid label mismatches"""
    print_header("5. Label Consistency (Invalid Label Check)")

    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        # Check for ectopy labels in rhythm position
        cur.execute("""
            SELECT COUNT(*) as cnt
            FROM ecg_features_annotatable
            WHERE is_corrected = TRUE
            AND (arrhythmia_label IN ('PVC', 'PAC', 'PVC Bigeminy', 'PAC Bigeminy',
                                       'PVC Trigeminy', 'PAC Trigeminy', 'Ventricular Triplet',
                                       'Atrial Triplet', 'PVC Couplet', 'Atrial Couplet'))
        """)
        ectopy_as_rhythm = cur.fetchone()[0]

        # Check for unknown labels
        cur.execute("""
            SELECT COUNT(DISTINCT arrhythmia_label) as cnt
            FROM ecg_features_annotatable
            WHERE is_corrected = TRUE
        """)
        unique_labels = cur.fetchone()[0]

        # Check for NULL labels in corrected segments
        cur.execute("""
            SELECT COUNT(*) as cnt
            FROM ecg_features_annotatable
            WHERE is_corrected = TRUE AND arrhythmia_label IS NULL
        """)
        null_labels = cur.fetchone()[0]

        print(f"  Unique corrected labels: {unique_labels}")
        print(f"    ⚠ Ectopy labels as rhythm: {ectopy_as_rhythm}")
        print(f"    ⚠ NULL labels in corrected segments: {null_labels}")

        valid = (ectopy_as_rhythm == 0) and (null_labels == 0)
        print_check("Label consistency", valid,
                   "Invalid labels detected - manual review needed" if not valid else "All labels valid")

        # Sample invalid labels if found
        if ectopy_as_rhythm > 0:
            cur.execute("""
                SELECT segment_id, arrhythmia_label, events_json
                FROM ecg_features_annotatable
                WHERE is_corrected = TRUE
                AND arrhythmia_label IN ('PVC', 'PAC')
                LIMIT 5
            """)
            samples = cur.fetchall()
            print(f"\n    Sample invalid segments:")
            for seg_id, label, events in samples:
                print(f"      - seg_id={seg_id}: label='{label}'")

        conn.close()
        return valid

    except Exception as e:
        print_check("Label Consistency", False, str(e))
        return False

def check_data_loader_integration():
    """Test that data loader can fetch V2 training data"""
    print_header("6. Data Loader Integration")

    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        # Simulate what V2 trainer fetches
        cur.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN signal_data IS NOT NULL THEN 1 ELSE 0 END) as with_signal,
                SUM(CASE WHEN events_json IS NOT NULL THEN 1 ELSE 0 END) as with_events
            FROM ecg_features_annotatable
            WHERE is_corrected = TRUE
        """)
        total, with_signal, with_events = cur.fetchone()

        print(f"  Segments ready for V2 training: {total}")
        print(f"    ✓ With signal data: {with_signal}")
        print(f"    ✓ With event annotations: {with_events}")

        # Get a sample segment and test label mapping
        cur.execute("""
            SELECT segment_id, arrhythmia_label
            FROM ecg_features_annotatable
            WHERE is_corrected = TRUE
            AND signal_data IS NOT NULL
            LIMIT 1
        """)
        result = cur.fetchone()

        if result:
            seg_id, label = result
            rhythm_idx = get_rhythm_label_idx(label)
            print(f"\n  Sample segment: seg_id={seg_id}")
            print(f"    Label: '{label}' → Rhythm index: {rhythm_idx}")
            print_check("Label mapping works", rhythm_idx is not None,
                       f"Mapped to class index {rhythm_idx}")
        else:
            print_check("Sample segment available", False, "No corrected segments found")

        conn.close()
        return True

    except Exception as e:
        print_check("Data Loader Integration", False, str(e))
        return False

def check_recent_training():
    """Check if training has used recently annotated data"""
    print_header("7. Training Round Tracking")

    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        # Get training round distribution
        cur.execute("""
            SELECT training_round, COUNT(*) as cnt
            FROM ecg_features_annotatable
            WHERE training_round > 0
            GROUP BY training_round
            ORDER BY training_round DESC
        """)
        rows = cur.fetchall()

        if rows:
            print(f"  Training round distribution:")
            for round_num, count in rows:
                print(f"    Round {round_num}: {count} segments")

            # Check if any segments were annotated but not trained yet
            cur.execute("""
                SELECT COUNT(*) as cnt
                FROM ecg_features_annotatable
                WHERE is_corrected = TRUE
                AND training_round = 0
            """)
            untrained = cur.fetchone()[0]

            print(f"\n  ⚠ Newly annotated (not yet in any training round): {untrained}")
            print_check("Training loop active", untrained >= 0,
                       f"{untrained} segments waiting for next training run")
        else:
            print("  No training rounds recorded yet")
            print_check("First training run", True, "Ready to start")

        conn.close()
        return True

    except Exception as e:
        print_check("Training Round Tracking", False, str(e))
        return False

def run_full_diagnostic():
    """Run all diagnostic checks"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔" + "="*68 + "╗")
    print("║" + "V2 PIPELINE COMPREHENSIVE DIAGNOSTIC".center(68) + "║")
    print("║" + "Dashboard → Database → Data Loader → Training".center(68) + "║")
    print("╚" + "="*68 + "╝")
    print(Colors.RESET)

    results = {
        "Database Connection": check_database_connection(),
        "Table Schema": check_table_schema(),
        "Annotated Data": check_annotated_data(),
        "Feature Vectors": check_feature_vectors(),
        "Label Consistency": check_label_consistency(),
        "Data Loader": check_data_loader_integration(),
        "Training Tracking": check_recent_training(),
    }

    # Summary
    print_header("SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status_color = Colors.GREEN if result else Colors.YELLOW
        status = "✅ PASS" if result else "⚠️  WARN"
        print(f"  {status_color}{status}{Colors.RESET} {name}")

    print(f"\n  {Colors.BOLD}Overall: {passed}/{total} checks passed{Colors.RESET}")

    if passed == total:
        print(f"\n  {Colors.GREEN}✓ Pipeline is fully operational{Colors.RESET}")
        print(f"  {Colors.GREEN}✓ Ready to train V2 models{Colors.RESET}\n")
        return 0
    else:
        print(f"\n  {Colors.YELLOW}⚠ Issues detected - review above for action items{Colors.RESET}")
        if not results["Feature Vectors"]:
            print(f"  {Colors.YELLOW}→ Run: python scripts/backfill_features.py{Colors.RESET}")
        if not results["Label Consistency"]:
            print(f"  {Colors.YELLOW}→ Review invalid labels (see details above){Colors.RESET}\n")
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V2 Pipeline Diagnostic Tool")
    parser.add_argument("--detailed", action="store_true", help="Show detailed output")
    parser.add_argument("--fix-backfill", action="store_true", help="Run backfill if needed")
    args = parser.parse_args()

    exit_code = run_full_diagnostic()

    if args.fix_backfill:
        print(f"{Colors.BLUE}Running feature backfill...{Colors.RESET}")
        import subprocess
        result = subprocess.run([
            sys.executable,
            str(Path(__file__).resolve().parent / "backfill_features.py")
        ])
        exit_code = result.returncode

    sys.exit(exit_code)
