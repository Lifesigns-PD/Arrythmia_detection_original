
import os
import json
import numpy as np
import psycopg2
import wfdb
from pathlib import Path
from tqdm import tqdm

# Config
DB_PARAMS = {
    "host": "127.0.0.1",
    "database": "ecg_analysis",
    "user": "ecg_user",
    "password": "sais",
    "port": 5432
}
BASE_DIR = Path(__file__).resolve().parent.parent
MITDB_DIR = BASE_DIR / "data" / "mitdb_data"
SEG_LEN = 2500 # 10 seconds at 250Hz
ORIGINAL_FS = 360
TARGET_FS = 250
SCALE_FACTOR = TARGET_FS / ORIGINAL_FS

def get_db_samples(n=10):
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    # Get random MITDB samples
    cur.execute("""
        SELECT segment_id, filename, raw_signal, segment_index
        FROM ecg_features_annotatable
        WHERE dataset_source = 'MITDB'
        AND raw_signal IS NOT NULL
        ORDER BY RANDOM()
        LIMIT %s;
    """, (n,))
    rows = cur.fetchall()
    conn.close()
    return rows

def verify_pas():
    print(f"--- Peak Alignment Score (PAS) Verification ---")
    print(f"Goal: Average distance < 20ms (+/- 5 samples at 250Hz)")
    
    samples = get_db_samples(10)
    if not samples:
        print("No MITDB samples found in DB.")
        return

    distances = []
    
    for seg_id, filename, signal_data, seg_idx in samples:
        # 1. Parse filename to get record
        # MITDB__100_seg_0000.json
        try:
            record_name = filename.split("__")[1].split("_")[0]
        except:
            continue
            
        # 2. Load original WFDB annotations
        ann_path = MITDB_DIR / record_name
        if not ann_path.exists():
            # Try with .hea extension
            if not (MITDB_DIR / f"{record_name}.hea").exists():
                print(f"Original record {record_name} not found at {MITDB_DIR}")
                continue
        
        ann = wfdb.rdann(str(ann_path), 'atr')
        
        # 3. Get signal from DB
        if isinstance(signal_data, str):
            signal = np.array(list(map(float, signal_data.split(","))))
        else:
            signal = np.array(signal_data)
            
        # 4. Find annotations in this segment's window
        # Window in original FS:
        start_orig = seg_idx * (SEG_LEN / SCALE_FACTOR)
        end_orig = start_orig + (SEG_LEN / SCALE_FACTOR)
        
        # Original indices in this window
        window_ann = [idx for idx in ann.sample if start_orig <= idx < end_orig]
        
        if not window_ann:
            continue
            
        # 5. Calculate PAS for each beat
        for orig_idx in window_ann:
            # Scaled index where it SHOULD be in the DB signal
            target_idx = int((orig_idx - start_orig) * SCALE_FACTOR)
            
            if target_idx < 0 or target_idx >= len(signal):
                continue
                
            # Find nearest peak in signal (search +/- 10 samples)
            search_start = max(0, target_idx - 10)
            search_end = min(len(signal), target_idx + 10)
            search_window = signal[search_start:search_end]
            
            if len(search_window) == 0: continue
            
            local_peak_offset = np.argmax(np.abs(search_window))
            actual_peak_idx = search_start + local_peak_offset
            
            dist = abs(target_idx - actual_peak_idx)
            distances.append(dist)
            
    if not distances:
        print("No beat alignments found.")
        return
        
    avg_dist_samples = np.mean(distances)
    avg_dist_ms = avg_dist_samples * (1000 / TARGET_FS)
    
    print(f"\nRESULTS:")
    print(f"  Samples checked: {len(distances)} beats")
    print(f"  Average Distance: {avg_dist_samples:.2f} samples")
    print(f"  Average Distance: {avg_dist_ms:.2f} ms")
    
    if avg_dist_ms < 20:
        print("\nPAS PASSED: Data is perfectly aligned.")
    else:
        print("\nPAS FAILED: Labels are drifting.")

if __name__ == "__main__":
    verify_pas()
