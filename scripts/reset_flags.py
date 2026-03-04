import psycopg2

def reset_all_verifications():
    # Connect to local database
    params = {
        "dbname": "ecg_analysis",
        "user": "ecg_user",
        "password": "sais",         
        "host": "127.0.0.1",
        "port": "5432"
    }
    conn = psycopg2.connect(**params)
    conn.autocommit = True
    cur = conn.cursor()
    
    print("Resetting all cardiologist verifications...")
    # This safely resets workflow flags without touching signal or feature data
    cur.execute("""
        UPDATE ecg_features_annotatable 
        SET is_corrected = FALSE, 
            used_for_training = FALSE,
            corrected_at = NULL,
            corrected_by = NULL;
    """)
    
    print("SUCCESS: All segments are now set to pending.")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    reset_all_verifications()
