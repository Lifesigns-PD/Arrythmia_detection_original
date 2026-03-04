
import psycopg2

DB_PARAMS = {
    "dbname": "ecg_analysis",
    "user": "ecg_user",
    "password": "sais",
    "host": "127.0.0.1",
    "port": "5432"
}

def manual_verification():
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    print("--- 1. Presidency Columns Verification ---")
    cur.execute("SELECT segment_id, arrhythmia_label, is_verified, corrected_by FROM ecg_features_annotatable LIMIT 5;")
    rows = cur.fetchall()
    print(f"{'segment_id':<12} | {'label':<15} | {'verified':<10} | {'corrected_by'}")
    print("-" * 60)
    for r in rows:
        print(f"{r[0]:<12} | {str(r[1]):<15} | {str(r[2]):<10} | {str(r[3])}")

    print("\n--- 2. Ghost Segments Check ---")
    cur.execute("SELECT COUNT(*) FROM ecg_features_annotatable WHERE raw_signal IS NULL;")
    ghost_count = cur.fetchone()[0]
    print(f"Ghost segments (raw_signal IS NULL): {ghost_count}")

    print("\n--- 3. Class Balance Audit ---")
    cur.execute("""
        SELECT arrhythmia_label, COUNT(*) 
        FROM ecg_features_annotatable 
        GROUP BY arrhythmia_label 
        ORDER BY COUNT(*) DESC;
    """)
    balance = cur.fetchall()
    print(f"{'Arrhythmia Label':<30} | {'Count'}")
    print("-" * 45)
    for b in balance:
        print(f"{str(b[0]):<30} | {b[1]}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    manual_verification()
