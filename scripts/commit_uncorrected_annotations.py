"""
Commit all uncorrected PAC annotations to is_corrected=TRUE
Run after cardiologist finishes marking beats in dashboard
"""
import psycopg2
from datetime import datetime

params = {
    'dbname': 'ecg_analysis',
    'user': 'ecg_user',
    'password': 'sais',
    'host': '127.0.0.1',
    'port': '5432'
}

conn = psycopg2.connect(**params)
cur = conn.cursor()

# Commit all uncorrected PAC with notes
cur.execute('''
UPDATE ecg_features_annotatable
SET 
    is_corrected = TRUE,
    corrected_by = 'cardiologist',
    corrected_at = %s,
    used_for_training = TRUE
WHERE arrhythmia_label ILIKE '%PAC%' 
  AND is_corrected = FALSE
  AND cardiologist_notes IS NOT NULL
RETURNING segment_id, arrhythmia_label
''', (datetime.now(),))

rows = cur.fetchall()
conn.commit()

print(f"Committed {len(rows)} PAC annotations")
for segment_id, label in rows[:10]:
    print(f"  - segment_id={segment_id}, label={label}")
if len(rows) > 10:
    print(f"  ... and {len(rows) - 10} more")

conn.close()
