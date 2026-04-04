import psycopg2
from collections import Counter

DB_PARAMS = {
    "host":     "127.0.0.1",
    "dbname":   "ecg_analysis",
    "user":     "ecg_user",
    "password": "sais",
    "port":     "5432",
}

try:
    with psycopg2.connect(**DB_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT arrhythmia_label FROM ecg_features_annotatable WHERE is_corrected = TRUE;")
            labels = [r[0] for r in cur.fetchall()]
            counts = Counter(labels)
            print("Label distribution for is_corrected=TRUE:")
            for label, count in counts.items():
                print(f"  {label}: {count}")

except Exception as e:
    print(f"Error: {e}")
