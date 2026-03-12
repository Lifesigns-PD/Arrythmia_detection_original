import psycopg2
import json
import numpy as np

PSQL_CONN_PARAMS = {
    "dbname": "ecg_analysis",
    "user": "ecg_user",
    "password": "sais",
    "host": "127.0.0.1",
    "port": "5432"
}

def check_db_shape():
    try:
        conn = psycopg2.connect(**PSQL_CONN_PARAMS)
        cur = conn.cursor()
        
        # Check the first available segment
        # Using signal_data which is the REAL[] array
        cur.execute("""
            SELECT segment_duration_s, array_length(signal_data, 1), filename 
            FROM ecg_features_annotatable 
            WHERE signal_data IS NOT NULL 
            LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            print("No segments found in database. Please run ecgprocessor.py first.")
            return

        duration, length, filename = row
        print(f"Record: {filename}")
        print(f"Duration: {duration}s (Expected: 10.0)")
        print(f"Signal Length: {length} samples (Expected: 1250)")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_db_shape()
