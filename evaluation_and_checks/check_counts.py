
import psycopg2
from verify_db_checklist import DB_PARAMS

conn = psycopg2.connect(**DB_PARAMS)
cur = conn.cursor()

def count(label, where):
    cur.execute(f"SELECT COUNT(*) FROM ecg_features_annotatable WHERE {where}")
    print(f"{label}: {cur.fetchone()[0]}")

print("--- COUNTS ---")
count("Total Rows", "TRUE")
count("Annotated Total", "corrected_by IS NOT NULL")
count("Training Candidates (Rhythm)", "annotation_type IN ('FALSE_NEGATIVE','FALSE_POSITIVE','BORDERLINE') AND mistake_target = 'RHYTHM' AND used_for_training = FALSE")
count("Training Candidates (Ectopy)", "annotation_type IN ('FALSE_NEGATIVE','FALSE_POSITIVE','BORDERLINE') AND mistake_target = 'ECTOPY' AND used_for_training = FALSE")
count("Confirmed Correct", "annotation_type = 'CONFIRMED_CORRECT'")
count("Null Annotation Type", "annotation_type IS NULL")

conn.close()
