import psycopg2

# Match your exact DB params from db_service.py
PSQL_CONN_PARAMS = {
    "dbname": "ecg_analysis",
    "user": "ecg_user",
    "password": "sais",
    "host": "127.0.0.1",
    "port": "5432"
}

def wipe_database():
    print("="*60)
    print("DANGER: DATABASE WIPE INITIATED")
    print("="*60)
    print("This will completely destroy ALL old ECG segments,")
    print("features, and cardiologist annotations in the database.")
    print("="*60)
    
    confirm = input("Type 'YES' to permanently wipe the database: ")
    
    if confirm != 'YES':
        print("Safety triggered. Aborting wipe.")
        return

    try:
        conn = psycopg2.connect(**PSQL_CONN_PARAMS)
        cur = conn.cursor()
        
        print("\nExecuting TRUNCATE on ecg_features_annotatable...")
        # CASCADE drops dependent table links if any exist
        # RESTART IDENTITY resets segment_id back to 1
        cur.execute("TRUNCATE TABLE ecg_features_annotatable RESTART IDENTITY CASCADE;")
        
        conn.commit()
        print("✅ Success: Database successfully wiped and ID counter reset to 1!")
        
    except Exception as e:
        print(f"❌ Error wiping database: {e}")
    finally:
        if 'conn' in locals() and conn:
            cur.close()
            conn.close()

if __name__ == "__main__":
    wipe_database()
