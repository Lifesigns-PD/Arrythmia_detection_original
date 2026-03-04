
import psycopg2
from verify_db_checklist import DB_PARAMS

conn = psycopg2.connect(**DB_PARAMS)
cur = conn.cursor()

print("--- SUMMARY ---")
cur.execute("""
    SELECT annotation_type, mistake_target, used_for_training, COUNT(*)
    FROM ecg_features_annotatable
    GROUP BY annotation_type, mistake_target, used_for_training
    ORDER BY annotation_type, mistake_target;
""")
rows = cur.fetchall()
print(f"{'Type':<20} | {'Target':<10} | {'Used':<5} | {'Count'}")
for r in rows:
    print(f"{str(r[0]):<20} | {str(r[1]):<10} | {str(r[2]):<5} | {r[3]}")

print("\n--- INTEGRITY CHECKS ---")
# 1. Null annotation_type checks
cur.execute("SELECT COUNT(*) FROM ecg_features_annotatable WHERE corrected_by IS NOT NULL AND annotation_type IS NULL")
print(f"Annotated but NULL type: {cur.fetchone()[0]}")

# 2. FN/FP without target
cur.execute("SELECT COUNT(*) FROM ecg_features_annotatable WHERE annotation_type IN ('FALSE_NEGATIVE','FALSE_POSITIVE') AND mistake_target IS NULL")
print(f"FN/FP no target: {cur.fetchone()[0]}")

conn.close()
