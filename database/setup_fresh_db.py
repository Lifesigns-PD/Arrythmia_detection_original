import psycopg2

def setup_hybrid_database():
    print("Connecting to database...")
    try:
        conn = psycopg2.connect(
            host="127.0.0.1",
            database="ecg_analysis",
            user="ecg_user",
            password="sais"
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Wipe existing tables to start fresh
        cursor.execute("DROP TABLE IF EXISTS ecg_features_annotatable CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS ecg_segments CASCADE;")

        print("Building Unified Hybrid Schema (Single Table Source of Truth)...")

        create_table_query = """
        CREATE TABLE ecg_features_annotatable (
            -- 1. Core Identifiers
            segment_id SERIAL PRIMARY KEY,
            dataset_source VARCHAR(50),
            patient_id VARCHAR(100),
            filename VARCHAR(255) NOT NULL,
            segment_index INT NOT NULL,
            segment_start_s FLOAT DEFAULT 0.0,
            segment_duration_s FLOAT DEFAULT 10.0,
            segment_fs INT DEFAULT 125,

            -- 2. HIGH-SPEED ML DATA (For PyTorch & gRPC)
            signal_data REAL[],
            model_pred_probs JSONB,

            -- 3. DASHBOARD UI DATA (For Flask, db_loader.py, and UI rendering)
            raw_signal JSONB,
            features_json JSONB,
            events_json JSONB DEFAULT '[]'::jsonb,
            r_peaks_in_segment TEXT,
            pr_interval FLOAT,

            -- 4. ANNOTATION & RETRAINING LOOP
            arrhythmia_label VARCHAR(50) DEFAULT 'Unlabeled',
            arrhythmia_text_notes TEXT DEFAULT '',
            model_pred_label TEXT,
            cardiologist_notes TEXT DEFAULT '',
            corrected_by TEXT,
            corrected_at TIMESTAMP,
            is_corrected BOOLEAN DEFAULT FALSE,
            is_verified BOOLEAN DEFAULT FALSE,
            used_for_training BOOLEAN DEFAULT FALSE,
            training_round INT DEFAULT 0,
            mistake_target TEXT,

            -- 5. Timestamps
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_table_query)

        # Create Indexes
        print("Creating Dashboard & ML Indexes...")
        cursor.execute("CREATE UNIQUE INDEX idx_unique_segment ON ecg_features_annotatable (filename, segment_index);")
        cursor.execute("CREATE INDEX idx_label ON ecg_features_annotatable(arrhythmia_label);")
        cursor.execute("CREATE INDEX idx_training ON ecg_features_annotatable(used_for_training);")
        cursor.execute("CREATE INDEX idx_gin_events ON ecg_features_annotatable USING GIN (events_json);")

        print("Hybrid Database Setup Complete! Dashboard and ML Pipeline are now fully supported.")

    except Exception as e:
        print(f"Database Setup Failed: {e}")
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    setup_hybrid_database()
