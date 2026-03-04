import os
import json
import wfdb
import numpy as np
import psycopg2
import neurokit2 as nk
from scipy.signal import resample
from tqdm import tqdm

# Configuration
DATA_DIR = r"C:\Users\admin\Documents\porject\Project_Submission_Clean\raw_data\mitdb"
FS_ORIGINAL = 360
FS_TARGET = 125
SEGMENT_SECONDS = 10
TARGET_LENGTH = SEGMENT_SECONDS * FS_TARGET
ORIGINAL_LENGTH = SEGMENT_SECONDS * FS_ORIGINAL

def connect_db():
    return psycopg2.connect(
        host="127.0.0.1", database="ecg_analysis", user="ecg_user", password="sais"
    )

def determine_label(symbols):
    """
    Maps MIT-BIH beat annotations to a segment-level label.
    Standard AAMI mapping.
    """
    if not symbols:
        return "Unlabeled"
    
    symbols_set = set(symbols)
    
    # Priority 1: Ventricular Ectopics
    if 'V' in symbols_set or 'E' in symbols_set:
        return "PVC"
    # Priority 2: Supraventricular Ectopics
    elif 'A' in symbols_set or 'a' in symbols_set or 'S' in symbols_set or 'J' in symbols_set:
        return "PAC"
    # Priority 3: Normal
    elif 'N' in symbols_set or 'L' in symbols_set or 'R' in symbols_set:
        return "Sinus Rhythm"
    else:
        return "Other Arrhythmia"

def main():
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

    # Get all unique record names from the downloaded folder (e.g., '100', '101')
    records = sorted(list(set([f.split('.')[0] for f in os.listdir(DATA_DIR) if f.endswith('.dat')])))
    
    print(f"Found {len(records)} MIT-BIH records. Processing into 10-second segments...")
    success_count = 0

    for record_name in tqdm(records):
        try:
            record_path = os.path.join(DATA_DIR, record_name)
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            
            # Channel 0 is MLII (Modified Lead II) in almost all MIT-BIH records
            signal = record.p_signal[:, 0]
            total_samples = len(signal)
            
            # Calculate how many full 10-second segments we can extract
            num_segments = total_samples // ORIGINAL_LENGTH
            
            for seg_idx in range(num_segments):
                start_idx = seg_idx * ORIGINAL_LENGTH
                end_idx = start_idx + ORIGINAL_LENGTH
                
                # 1. Slice original 360Hz signal
                segment_360 = signal[start_idx:end_idx]
                
                # 2. Resample to 125 Hz (1,250 points)
                segment_125 = resample(segment_360, TARGET_LENGTH)
                
                # 3. Find annotations that fall into this specific 10-second window
                ann_indices = np.where((annotation.sample >= start_idx) & (annotation.sample < end_idx))[0]
                window_symbols = [annotation.symbol[i] for i in ann_indices]
                
                label = determine_label(window_symbols)
                
                # 4. Extract R-peaks for the UI dashboard
                try:
                    _, info = nk.ecg_peaks(segment_125, sampling_rate=FS_TARGET)
                    r_peaks = info["ECG_R_Peaks"].tolist()
                except:
                    r_peaks = []
                    
                features_dict = {"r_peaks": r_peaks}
                
                # 5. Insert to DB
                filename = f"mitdb_{record_name}"
                signal_list = segment_125.tolist()
                
                cur.execute(
                    INSERT_SQL,
                    (
                        "mitdb",
                        filename,
                        seg_idx,                 # Now this tracks which 10s chunk it is (0 to 179)
                        signal_list,             # Native array
                        json.dumps(signal_list), # JSON for UI
                        json.dumps(features_dict),
                        label,
                        FS_TARGET
                    )
                )
                success_count += 1
                
        except Exception as e:
            print(f"Error processing record {record_name}: {e}")

    print(f"✅ Finished! Successfully extracted and loaded {success_count} MIT-BIH segments into the database.")
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()