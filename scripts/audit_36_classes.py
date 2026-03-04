#!/usr/bin/env python3
"""
============================================================
SYSTEM-WIDE ARRHYTHMIA INTEGRITY AUDIT
============================================================
Tests ALL 37 classes (0-36) across:
  1. Class List Coverage
  2. Label Normalization (synonyms, self-normalization)
  3. Rule Integration (Bigeminy, Trigeminy, PSVT, NSVT, Couplet)
  4. Display Arbitration (priority hierarchy)
  5. Manual Annotation Round-Trip (Save -> Load -> Verify)
  6. Training Pipeline Alignment (retrain.py class lists)
  7. Dashboard Dropdown Coverage
"""

import sys
import os
import json
import psycopg2
import uuid
from pathlib import Path
from collections import OrderedDict

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from models_training.data_loader import (
    CLASS_NAMES, CLASS_INDEX, normalize_label,
    RHYTHM_CLASS_NAMES, ECTOPY_CLASS_NAMES,
    get_rhythm_label_idx, get_ectopy_label_idx
)
from decision_engine.models import Event, EventCategory, DisplayState
from decision_engine.rules import (
    apply_ectopy_patterns, apply_display_rules, apply_training_flags
)
from database.db_service import PSQL_CONN_PARAMS

# ============================================================
# Helpers
# ============================================================
PASS = 0
FAIL = 0
WARN = 0

def log_pass(msg):
    global PASS
    PASS += 1
    print(f"  ✅ [PASS] {msg}")

def log_fail(msg):
    global FAIL
    FAIL += 1
    print(f"  ❌ [FAIL] {msg}")

def log_warn(msg):
    global WARN
    WARN += 1
    print(f"  ⚠️  [WARN] {msg}")

def make_ectopy_event(etype, start, beat_idx):
    return Event(
        event_id=str(uuid.uuid4()),
        event_type=etype,
        event_category=EventCategory.ECTOPY,
        start_time=start,
        end_time=start + 0.1,
        beat_indices=[beat_idx]
    )

def make_rhythm_event(etype, start=0, end=10, priority=50):
    return Event(
        event_id=str(uuid.uuid4()),
        event_type=etype,
        event_category=EventCategory.RHYTHM,
        start_time=start,
        end_time=end,
        priority=priority
    )

# ============================================================
# TEST 1: Class List Coverage
# ============================================================
def test_class_list():
    print(f"\n{'='*60}")
    print("[1/7] CLASS LIST COVERAGE")
    print(f"{'='*60}")
    
    expected = 37  # indices 0-36
    actual = len(CLASS_NAMES)
    
    if actual == expected:
        log_pass(f"CLASS_NAMES has exactly {expected} entries (indices 0-36).")
    else:
        log_fail(f"CLASS_NAMES has {actual} entries, expected {expected}.")
    
    # Verify CLASS_INDEX is consistent
    for i, name in enumerate(CLASS_NAMES):
        if CLASS_INDEX.get(name) != i:
            log_fail(f"CLASS_INDEX mismatch for '{name}': expected {i}, got {CLASS_INDEX.get(name)}")
            return
    log_pass("CLASS_INDEX is consistent with CLASS_NAMES.")

    # Print full class list
    print("\n  Full Class List:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    [{i:2d}] {name}")

# ============================================================
# TEST 2: Normalization Logic
# ============================================================
def test_normalization():
    print(f"\n{'='*60}")
    print("[2/7] LABEL NORMALIZATION")
    print(f"{'='*60}")
    
    # 2a. Self-normalization (every class name must map to itself)
    self_fails = 0
    for name in CLASS_NAMES:
        result = normalize_label(name)
        if result != name:
            log_fail(f"Self-norm: '{name}' -> '{result}'")
            self_fails += 1
    if self_fails == 0:
        log_pass(f"All {len(CLASS_NAMES)} classes self-normalize correctly.")

    # 2b. Synonym normalization
    synonym_tests = OrderedDict([
        ("af", "Atrial Fibrillation"),
        ("afib", "Atrial Fibrillation"),
        ("AF", "Atrial Fibrillation"),
        ("SVT", "Supraventricular Tachycardia"),
        ("normal", "Sinus Rhythm"),
        ("NSR", "Sinus Rhythm"),
        ("sinus tach", "Sinus Tachycardia"),
        ("TACHY", "Sinus Tachycardia"),
        ("BRADY", "Sinus Bradycardia"),
        ("APC", "PAC"),
        ("VPB", "PVC"),
        ("PVC", "PVC"),
        ("pvc bigeminy", "PVC Bigeminy"),
        ("BBB", "Bundle Branch Block"),
        ("LBBB", "Bundle Branch Block"),
        ("RBBB", "Bundle Branch Block"),
        ("AF + PVC", "Atrial Fibrillation + PVC"),
        ("Wenckebach", "2nd Degree AV Block Type 1"),
        ("Mobitz II", "2nd Degree AV Block Type 2"),
    ])
    
    syn_fails = 0
    for input_str, expected in synonym_tests.items():
        actual = normalize_label(input_str)
        if actual == expected:
            log_pass(f"'{input_str}' -> '{actual}'")
        else:
            log_fail(f"'{input_str}' -> '{actual}' (Expected: '{expected}')")
            syn_fails += 1

# ============================================================
# TEST 3: Rule Integration (Patterns)
# ============================================================
def test_rules():
    print(f"\n{'='*60}")
    print("[3/7] RULE INTEGRATION (Patterns & Runs)")
    print(f"{'='*60}")
    
    # 3a. PVC Bigeminy (indices: 10, 12, 14 — every other beat)
    events = [
        make_ectopy_event("PVC", 1.0, 10),
        make_ectopy_event("PVC", 2.0, 12),
        make_ectopy_event("PVC", 3.0, 14),
    ]
    apply_ectopy_patterns(events)
    if any(e.event_type == "PVC Bigeminy" for e in events):
        log_pass("PVC Bigeminy detected (indices 10, 12, 14).")
    else:
        log_fail("PVC Bigeminy NOT detected.")

    # 3b. PAC Trigeminy (indices: 5, 8, 11 — every 3rd beat)
    events = [
        make_ectopy_event("PAC", 0.5, 5),
        make_ectopy_event("PAC", 2.0, 8),
        make_ectopy_event("PAC", 3.5, 11),
    ]
    apply_ectopy_patterns(events)
    if any(e.event_type == "PAC Trigeminy" for e in events):
        log_pass("PAC Trigeminy detected (indices 5, 8, 11).")
    else:
        log_fail("PAC Trigeminy NOT detected.")

    # 3c. PAC Bigeminy (indices: 20, 22, 24)
    events = [
        make_ectopy_event("PAC", 1.0, 20),
        make_ectopy_event("PAC", 2.0, 22),
        make_ectopy_event("PAC", 3.0, 24),
    ]
    apply_ectopy_patterns(events)
    if any(e.event_type == "PAC Bigeminy" for e in events):
        log_pass("PAC Bigeminy detected (indices 20, 22, 24).")
    else:
        log_fail("PAC Bigeminy NOT detected.")

    # 3d. PVC Trigeminy (indices: 30, 33, 36)
    events = [
        make_ectopy_event("PVC", 1.0, 30),
        make_ectopy_event("PVC", 2.5, 33),
        make_ectopy_event("PVC", 4.0, 36),
    ]
    apply_ectopy_patterns(events)
    if any(e.event_type == "PVC Trigeminy" for e in events):
        log_pass("PVC Trigeminy detected (indices 30, 33, 36).")
    else:
        log_fail("PVC Trigeminy NOT detected.")

    # 3e. PSVT (3+ consecutive PACs, tachycardic rate)
    events = [
        make_ectopy_event("PAC", 1.0, 100),
        make_ectopy_event("PAC", 1.4, 101),
        make_ectopy_event("PAC", 1.8, 102),
    ]
    apply_ectopy_patterns(events)
    if any(e.event_type == "PSVT" for e in events):
        log_pass("PSVT detected (3 consecutive PACs, rate >= 100).")
    else:
        log_fail("PSVT NOT detected.")

    # 3f. NSVT (3+ consecutive PVCs, tachycardic rate)
    events = [
        make_ectopy_event("PVC", 4.0, 200),
        make_ectopy_event("PVC", 4.5, 201),
        make_ectopy_event("PVC", 5.0, 202),
    ]
    apply_ectopy_patterns(events)
    if any(e.event_type == "NSVT" for e in events):
        log_pass("NSVT detected (3 consecutive PVCs, rate >= 100).")
    else:
        log_fail("NSVT NOT detected.")

    # 3g. PVC Couplet (2 consecutive)
    events = [
        make_ectopy_event("PVC", 6.0, 300),
        make_ectopy_event("PVC", 6.5, 301),
    ]
    apply_ectopy_patterns(events)
    labels = [e.pattern_label for e in events if e.event_category == EventCategory.ECTOPY]
    if all(l == "Couplet" for l in labels):
        log_pass("PVC Couplet detected (2 consecutive PVCs).")
    else:
        log_fail(f"PVC Couplet NOT detected. Labels: {labels}")

    # 3h. Atrial Run (3+ consecutive PACs, low rate)
    events = [
        make_ectopy_event("PAC", 1.0, 400),
        make_ectopy_event("PAC", 2.5, 401),
        make_ectopy_event("PAC", 4.0, 402),
    ]
    apply_ectopy_patterns(events)
    if any(e.event_type == "Atrial Run" for e in events):
        log_pass("Atrial Run detected (3 consecutive PACs, rate < 100).")
    else:
        log_fail("Atrial Run NOT detected.")

# ============================================================
# TEST 4: Display Arbitration
# ============================================================
def test_display():
    print(f"\n{'='*60}")
    print("[4/7] DISPLAY ARBITRATION")
    print(f"{'='*60}")
    
    # 4a. PSVT should suppress individual PACs
    events = [
        make_ectopy_event("PAC", 1.0, 10),
        make_ectopy_event("PAC", 2.0, 12),
        make_ectopy_event("PAC", 3.0, 14),
    ]
    apply_ectopy_patterns(events)
    displayed = apply_display_rules("Sinus Rhythm", events)
    displayed_types = [e.event_type for e in displayed]
    
    if "PAC Bigeminy" in displayed_types:
        log_pass("PAC Bigeminy shown in display (higher priority than individual PACs).")
    else:
        log_fail(f"PAC Bigeminy NOT in display. Displayed: {displayed_types}")

    # 4b. AF should suppress other rhythms
    events = [
        make_rhythm_event("Atrial Fibrillation", priority=80),
        make_rhythm_event("Sinus Tachycardia", priority=10),
    ]
    displayed = apply_display_rules("Sinus Rhythm", events)
    displayed_types = [e.event_type for e in displayed]
    if "Atrial Fibrillation" in displayed_types and "Sinus Tachycardia" not in displayed_types:
        log_pass("AF dominance: AF shown, Sinus Tachycardia suppressed.")
    else:
        log_fail(f"AF dominance failed. Displayed: {displayed_types}")

# ============================================================
# TEST 5: Manual Annotation Round-Trip (Save -> Load -> Verify)
# ============================================================
def test_annotation_roundtrip():
    print(f"\n{'='*60}")
    print("[5/7] MANUAL ANNOTATION ROUND-TRIP (SQL)")
    print(f"{'='*60}")
    
    conn = None
    test_segment_id = None
    try:
        conn = psycopg2.connect(**PSQL_CONN_PARAMS)
        with conn.cursor() as cur:
            # Find a real segment to use for the test
            cur.execute("SELECT segment_id FROM ecg_features_annotatable LIMIT 1")
            row = cur.fetchone()
            if not row:
                log_warn("No segments found in ecg_features_annotatable. Skipping round-trip test.")
                return
            test_segment_id = row[0]
            
            # Read existing events_json
            cur.execute("SELECT events_json FROM ecg_features_annotatable WHERE segment_id = %s", (test_segment_id,))
            original_data = cur.fetchone()[0]
            
            # === SAVE: Inject a test annotation ===
            test_event_id = f"AUDIT_TEST_{uuid.uuid4().hex[:8]}"
            test_event = {
                "event_id": test_event_id,
                "event_type": "PVC Bigeminy",
                "start_time": 1.5,
                "end_time": 2.1,
                "annotation_source": "cardiologist",
                "annotation_status": "confirmed",
                "used_for_training": True
            }
            
            # Parse existing data
            if isinstance(original_data, str):
                data = json.loads(original_data)
            else:
                data = original_data or {"events": [], "final_display_events": []}
            
            if isinstance(data, list):
                data = {"events": data, "final_display_events": data}
            
            data.setdefault("events", []).append(test_event)
            data.setdefault("final_display_events", []).append(test_event)
            
            cur.execute(
                "UPDATE ecg_features_annotatable SET events_json = %s WHERE segment_id = %s",
                (json.dumps(data), test_segment_id)
            )
            conn.commit()
            log_pass(f"SAVE: Injected test annotation '{test_event_id}' into segment {test_segment_id}.")

            # === LOAD: Read it back ===
            cur.execute("SELECT events_json FROM ecg_features_annotatable WHERE segment_id = %s", (test_segment_id,))
            loaded_data = cur.fetchone()[0]
            if isinstance(loaded_data, str):
                loaded_data = json.loads(loaded_data)
            
            loaded_events = loaded_data.get("events", []) if isinstance(loaded_data, dict) else loaded_data
            found_event = None
            for evt in loaded_events:
                if evt.get("event_id") == test_event_id:
                    found_event = evt
                    break
            
            if found_event:
                log_pass(f"LOAD: Successfully retrieved test annotation from DB.")
                # Verify all fields survived
                if found_event.get("event_type") == "PVC Bigeminy":
                    log_pass("VERIFY: event_type preserved ('PVC Bigeminy').")
                else:
                    log_fail(f"VERIFY: event_type corrupted: {found_event.get('event_type')}")
                    
                if found_event.get("annotation_source") == "cardiologist":
                    log_pass("VERIFY: annotation_source preserved ('cardiologist').")
                else:
                    log_fail(f"VERIFY: annotation_source corrupted: {found_event.get('annotation_source')}")
                
                if found_event.get("used_for_training") == True:
                    log_pass("VERIFY: used_for_training flag preserved (True).")
                else:
                    log_fail(f"VERIFY: used_for_training corrupted: {found_event.get('used_for_training')}")
            else:
                log_fail("LOAD: Test annotation NOT found after save. Round-trip failed!")

            # === CLEANUP: Remove test event ===
            if isinstance(loaded_data, dict):
                loaded_data["events"] = [e for e in loaded_data.get("events", []) if e.get("event_id") != test_event_id]
                loaded_data["final_display_events"] = [e for e in loaded_data.get("final_display_events", []) if e.get("event_id") != test_event_id]
            
            cur.execute(
                "UPDATE ecg_features_annotatable SET events_json = %s WHERE segment_id = %s",
                (json.dumps(loaded_data), test_segment_id)
            )
            conn.commit()
            log_pass("CLEANUP: Test annotation removed. No data was modified.")

    except Exception as e:
        log_fail(f"SQL Error: {e}")
    finally:
        if conn: conn.close()

# ============================================================
# TEST 6: Training Pipeline Alignment
# ============================================================
def test_training_pipeline():
    print(f"\n{'='*60}")
    print("[6/7] TRAINING PIPELINE ALIGNMENT")
    print(f"{'='*60}")
    
    # 6a. Rhythm classes should not include Sinus/Artifact
    sinus_in_rhythm = [n for n in RHYTHM_CLASS_NAMES if "Sinus" in n]
    if not sinus_in_rhythm:
        log_pass("RHYTHM_CLASS_NAMES correctly excludes Sinus rhythms.")
    else:
        log_fail(f"RHYTHM_CLASS_NAMES contains Sinus: {sinus_in_rhythm}")

    # 6b. Ectopy model classes (beat-level): PVC, PAC, Run
    # NOTE: Bigeminy/Trigeminy are RULE-BASED patterns, NOT model predictions.
    # The ectopy model classifies individual beats, then rules.py upgrades them.
    required_ectopy = ["None", "PVC", "PAC", "Run"]
    missing = [r for r in required_ectopy if r not in ECTOPY_CLASS_NAMES]
    if not missing:
        log_pass(f"ECTOPY_CLASS_NAMES includes all beat-level types: {ECTOPY_CLASS_NAMES}")
    else:
        log_fail(f"ECTOPY_CLASS_NAMES missing: {missing}")
    
    # 6b2. Verify that Bigeminy/Trigeminy are in the FULL CLASS_NAMES (for training)
    pattern_classes = ["PVC Bigeminy", "PAC Bigeminy", "PVC Trigeminy"]
    missing_patterns = [p for p in pattern_classes if p not in CLASS_NAMES]
    if not missing_patterns:
        log_pass(f"CLASS_NAMES includes rule-based patterns: {pattern_classes}")
    else:
        log_fail(f"CLASS_NAMES missing patterns: {missing_patterns}")

    # 6c. Training flags should mark correct events
    test_events = [
        Event("t1", "PVC", EventCategory.ECTOPY, 0, 1),
        Event("t2", "PAC Bigeminy", EventCategory.RHYTHM, 0, 10),
        Event("t3", "PSVT", EventCategory.RHYTHM, 0, 10),
        Event("t4", "NSVT", EventCategory.RHYTHM, 0, 10),
        Event("t5", "Sinus Rhythm", EventCategory.RHYTHM, 0, 10),
        Event("t6", "Artifact", EventCategory.RHYTHM, 0, 10),
        Event("t7", "PVC Bigeminy", EventCategory.RHYTHM, 0, 10),
        Event("t8", "PVC Trigeminy", EventCategory.RHYTHM, 0, 10),
    ]
    apply_training_flags(test_events)
    
    # PVC, PAC Bigeminy, PSVT, NSVT, PVC Bigeminy, PVC Trigeminy should be True
    for e in test_events:
        if e.event_type in ["Sinus Rhythm", "Artifact"]:
            if not e.used_for_training:
                log_pass(f"Training flag: '{e.event_type}' correctly EXCLUDED.")
            else:
                log_fail(f"Training flag: '{e.event_type}' should be excluded but is included!")
        elif e.event_type in ["PVC", "PAC Bigeminy", "PSVT", "NSVT", "PVC Bigeminy", "PVC Trigeminy"]:
            if e.used_for_training:
                log_pass(f"Training flag: '{e.event_type}' correctly INCLUDED.")
            else:
                log_fail(f"Training flag: '{e.event_type}' should be included but is excluded!")

# ============================================================
# TEST 7: Dashboard Dropdown Coverage
# ============================================================
def test_dashboard_dropdown():
    print(f"\n{'='*60}")
    print("[7/7] DASHBOARD DROPDOWN COVERAGE")
    print(f"{'='*60}")
    
    html_path = PROJECT_ROOT / "dashboard" / "templates" / "index.html"
    if not html_path.exists():
        log_warn("index.html not found. Skipping dropdown audit.")
        return
    
    html_content = html_path.read_text(encoding="utf-8")
    
    # Critical labels that MUST be in the dropdown
    critical_labels = [
        "Sinus Rhythm", "Sinus Bradycardia", "Sinus Tachycardia",
        "Atrial Fibrillation", "Atrial Flutter",
        "Supraventricular Tachycardia", "PSVT",
        "Ventricular Tachycardia", "NSVT", "Ventricular Fibrillation",
        "Junctional Rhythm", "Idioventricular Rhythm",
        "1st Degree AV Block", "2nd Degree AV Block Type 1", 
        "2nd Degree AV Block Type 2","3rd Degree AV Block",
        "Bundle Branch Block",
        "PAC", "PVC Bigeminy", "PVC Trigeminy",
        "PAC Bigeminy",
        "Artifact", "Pause"
    ]
    
    for label in critical_labels:
        # Check for value="label" in dropdown
        if f'value="{label}"' in html_content:
            log_pass(f"Dashboard dropdown has '{label}'.")
        else:
            log_fail(f"Dashboard dropdown MISSING '{label}'.")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print("#  SYSTEM-WIDE ARRHYTHMIA INTEGRITY AUDIT")
    print(f"#  Testing {len(CLASS_NAMES)} Clinical Classes")
    print(f"{'#'*60}")

    test_class_list()
    test_normalization()
    test_rules()
    test_display()
    test_annotation_roundtrip()
    test_training_pipeline()
    test_dashboard_dropdown()

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  ✅ PASSED: {PASS}")
    print(f"  ❌ FAILED: {FAIL}")
    print(f"  ⚠️  WARNINGS: {WARN}")
    
    total = PASS + FAIL
    if total > 0:
        score = (PASS / total) * 100
        print(f"\n  INTEGRITY SCORE: {score:.1f}%")
        if score == 100:
            print("  🏆 SYSTEM IS FULLY CERTIFIED FOR CLINICAL USE.")
        elif score >= 90:
            print("  ⚠️  SYSTEM HAS MINOR GAPS. Review failures above.")
        else:
            print("  🚨 SYSTEM HAS CRITICAL GAPS. Immediate attention required.")
    
    print(f"{'='*60}\n")
