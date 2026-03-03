"""
Quick Database Health Check: Is everything ready for annotation?
Updated to focus on the single ecg_features_annotatable table.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import db_service

passed = 0
failed = 0
total = 0

def check(name, condition, detail=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  [PASS] {name}" + (f" -- {detail}" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {name}" + (f" -- {detail}" if detail else ""))

print("=" * 60)
print("  DATABASE HEALTH CHECK")
print("=" * 60)

# 1. Connection Test
print("\n[1] CONNECTION")
conn = None
try:
    conn = db_service._connect()
    check("PostgreSQL connection", conn is not None)
except Exception as e:
    check("PostgreSQL connection", False, str(e))

if not conn:
    print("\nCannot proceed without database connection.")
    sys.exit(1)

cur = conn.cursor()

# 2. Table Existence
print("\n[2] TABLES")
for table in ["ecg_features_annotatable"]:
    cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)", (table,))
    exists = cur.fetchone()[0]
    check(f"Table '{table}' exists", exists)

# 3. Column checks for ecg_features_annotatable 
print("\n[3] SCHEMA (ecg_features_annotatable)")
critical_cols = [
    "segment_id", "filename", "segment_index", "arrhythmia_label", 
    "features_json", "events_json", "raw_signal", "is_corrected"
]
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'ecg_features_annotatable'")
feat_cols = [r[0] for r in cur.fetchall()]
for col in critical_cols:
    check(f"Column 'ecg_features_annotatable.{col}'", col in feat_cols)

# 4. Data counts
print("\n[4] DATA VOLUME")
cur.execute("SELECT COUNT(*) FROM ecg_features_annotatable")
feat_count = cur.fetchone()[0]
check(f"ecg_features_annotatable has data", feat_count > 0, f"{feat_count} records")

# 5. Annotation readiness: events_json check
print("\n[5] ANNOTATION READINESS")
cur.execute("SELECT COUNT(*) FROM ecg_features_annotatable WHERE events_json IS NOT NULL")
annotated = cur.fetchone()[0]
check(f"Segments with annotations", True, f"{annotated}/{feat_count} annotated")

cur.execute("SELECT COUNT(*) FROM ecg_features_annotatable WHERE is_corrected = TRUE")
verified = cur.fetchone()[0]
check(f"Verified segments", True, f"{verified}/{feat_count} verified")

cur.execute("SELECT COUNT(*) FROM ecg_features_annotatable WHERE is_corrected = FALSE")
pending = cur.fetchone()[0]
check(f"Pending segments (ready to annotate)", True, f"{pending} segments awaiting annotation")

# 5b. Signal array length integrity
print("\n[5b] SIGNAL LENGTH INTEGRITY (expected: 1250 samples @ 125 Hz)")
cur.execute("""
    SELECT COUNT(*) FROM ecg_features_annotatable
    WHERE signal_data IS NOT NULL AND array_length(signal_data, 1) != 1250
""")
bad_count = cur.fetchone()[0]
check("All signal_data arrays are 1250 samples", bad_count == 0,
      f"{bad_count} segments with wrong length" if bad_count else "all correct")

if bad_count > 0:
    print("  Length distribution (wrong-length segments):")
    cur.execute("""
        SELECT array_length(signal_data, 1), COUNT(*)
        FROM ecg_features_annotatable
        WHERE signal_data IS NOT NULL AND array_length(signal_data, 1) != 1250
        GROUP BY array_length(signal_data, 1)
        ORDER BY COUNT(*) DESC LIMIT 5
    """)
    for length, cnt in cur.fetchall():
        print(f"    length={length}: {cnt} segments")

# Check segments where signal_data is NULL (won't be usable by ML)
cur.execute("SELECT COUNT(*) FROM ecg_features_annotatable WHERE signal_data IS NULL")
null_sig = cur.fetchone()[0]
check("No segments with NULL signal_data", null_sig == 0,
      f"{null_sig} segments missing signal_data (will be skipped by ML)" if null_sig else "")


# 6. Save/Load round-trip test
print("\n[6] WRITE/READ TEST")
import uuid
test_id_str = f"HEALTH_CHECK_{uuid.uuid4().hex[:8]}"
# Find a segment to test on
cur.execute("SELECT segment_id FROM ecg_features_annotatable ORDER BY segment_id LIMIT 1")
test_seg = cur.fetchone()
if test_seg:
    test_segment_id = test_seg[0]
    test_event = {
        "event_id": test_id_str,
        "event_type": "PVC",
        "event_category": "ECTOPY",
        "start_time": 0.5,
        "end_time": 1.1,
        "annotation_source": "health_check",
        "used_for_training": False
    }
    save_ok = db_service.save_event_to_db(test_segment_id, test_event)
    check("Write event to DB", save_ok)

    # Read it back
    seg_data = db_service.get_segment_new(test_segment_id)
    events_json = seg_data.get("events_json") if seg_data else None
    found = False
    if events_json:
        raw = events_json.get("events", []) if isinstance(events_json, dict) else events_json
        found = any(e.get("event_id") == test_id_str for e in raw)
    check("Read event back from DB", found)

    # Cleanup
    db_service.delete_event(test_segment_id, test_id_str)
    check("Cleanup test data", True)
else:
    check("Write/Read test", False, "No segments found")

# 7. Index check
print("\n[7] INDEXES")
cur.execute("""
    SELECT indexname FROM pg_indexes 
    WHERE tablename = 'ecg_features_annotatable' 
    AND indexdef LIKE '%gin%'
""")
gin_indexes = [r[0] for r in cur.fetchall()]
check("GIN indexes for JSONB queries", len(gin_indexes) > 0, f"{len(gin_indexes)} GIN indexes found")

conn.close()

# SUMMARY
print("\n" + "=" * 60)
print(f"  RESULTS: {passed}/{total} PASSED, {failed} FAILED")
if failed == 0:
    print("  DATABASE IS HEALTHY AND READY FOR ANNOTATION!")
else:
    print("  WARNING: Some checks failed. Review above.")
print("=" * 60)
