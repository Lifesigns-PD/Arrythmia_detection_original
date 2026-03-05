import os
import json
import wfdb
import pandas as pd
import numpy as np
import psycopg2
import neurokit2 as nk
from scipy.signal import resample
from tqdm import tqdm

# Configuration
DATA_DIR = 'raw_data/ptbxl'
FS_ORIGINAL = 500
FS_TARGET = 125
TARGET_LENGTH = 1250  # 10 seconds * 125 Hz

def connect_db():
    return psycopg2.connect(
        host="127.0.0.1", database="ecg_analysis", user="ecg_user", password="sais"
    )

def main():
    print("Loading PTB-XL metadata CSV...")
    # The CSV contains the labels and the paths to the .dat files
    csv_path = os.path.join(DATA_DIR, 'ptbxl_database.csv')
    df = pd.read_csv(csv_path, index_col='ecg_id')
    
    conn = connect_db()
    conn.autocommit = True
    cur = conn.cursor()

    INSERT_SQL = """
        INSERT INTO ecg_features_annotatable (
            dataset_source, filename, segment_index, signal_data, raw_signal, 
            features_json, arrhythmia_label, segment_fs
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (filename, segment_index) DO NOTHING;
    """

    success_count = 0
    
    # Iterate through the dataset
    print("Processing and inserting records into PostgreSQL...")
    for ecg_id, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # The 'filename_hr' column points to the 500Hz files
            file_path = os.path.join(DATA_DIR, row['filename_hr'])
            
            # 1. Extract Label (PTB-XL uses scp_codes dictionaries, we grab the string for now)
            # You can write a better parser for this later to map specific arrhythmias!
            raw_labels = row['scp_codes'] 
            label = "Normal" if "NORM" in raw_labels else "Arrhythmia"
            
            # 2. Read the 500Hz signal
            record = wfdb.rdrecord(file_path)
            
            # Lead II is usually channel 1 in PTB-XL (0-indexed: I, II, III, aVR, aVL, aVF, V1-V6)
            lead_ii = record.p_signal[:, 1]
            
            # 3. Resample to 125 Hz (1,250 points)
            signal_125 = resample(lead_ii, TARGET_LENGTH)
            
            # 4. Extract R-peaks using Neurokit2 for the Dashboard UI
            # We wrap it in a try-except because some extremely noisy signals fail peak detection
            try:
                _, info = nk.ecg_peaks(signal_125, sampling_rate=FS_TARGET)
                r_peaks = info["ECG_R_Peaks"].tolist()
            except:
                r_peaks = []
                
            features_dict = {"r_peaks": r_peaks}
            
            # 5. Insert into Database
            filename = f"ptbxl_{ecg_id}"
            signal_list = signal_125.tolist()
            
            cur.execute(
                INSERT_SQL,
                (
                    "ptbxl",
                    filename,
                    0,                      # segment_index is 0 because PTB-XL is only 10s long
                    signal_list,            # REAL[] for PyTorch
                    json.dumps(signal_list),# JSONB for Dashboard
                    json.dumps(features_dict),
                    label,
                    FS_TARGET
                )
            )
            success_count += 1
            
        except Exception as e:
            # Skip records if the file doesn't exist or is corrupted
            pass

    print(f"✅ Finished! Successfully processed and loaded {success_count} PTB-XL segments into the database.")
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()