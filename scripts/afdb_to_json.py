#!/usr/bin/env python3
import wfdb
import json
import numpy as np
from pathlib import Path
from scipy.signal import resample
from tqdm import tqdm

# Relative path to project root
INPUT = Path(r"data/afdb_data")
OUTPUT = Path("data/converted_ecg")
OUTPUT.mkdir(parents=True, exist_ok=True)

SEG_LEN = 1250  # 10 sec @ 125 Hz

def convert_record(record_name):
    try:
        record_path = str(INPUT / record_name)
        
        # 1. Read Signal
        sig, fields = wfdb.rdsamp(record_path)
        ecg = sig[:, 0]
        fs = fields.get('fs', 250) if isinstance(fields, dict) else fields.fs

        # 2. Resample Signal
        scale_factor = 1.0
        if fs != 125:
            scale_factor = 125 / fs
            ecg = resample(ecg, int(len(ecg) * scale_factor))
        
        # 3. Build Rhythm Mask (The Fix)
        # 0 = Sinus/Other, 1 = AFIB
        # We assume the patient starts in Normal/Sinus unless told otherwise
        rhythm_mask = np.zeros(len(ecg), dtype=int)
        
        ann = wfdb.rdann(record_path, "atr")
        
        # Iterate through annotations to fill the mask
        # AFDB format: "(AFIB" starts AF, "(N" ends it (or starts Normal)
        current_state = 0 # 0 for Sinus, 1 for AFIB
        
        # Loop through all annotations except the last one
        for i in range(len(ann.sample) - 1):
            start_idx = int(ann.sample[i] * scale_factor)
            end_idx = int(ann.sample[i+1] * scale_factor)
            
            note = ann.aux_note[i]
            
            # Update State
            if "(AFIB" in note:
                current_state = 1
            elif "(N" in note or "(AFL" in note or "(J" in note:
                current_state = 0
            
            # Fill the mask for this duration
            if current_state == 1:
                # Clamp indices to signal boundaries just in case
                s = max(0, start_idx)
                e = min(len(rhythm_mask), end_idx)
                rhythm_mask[s:e] = 1

        # Handle the very last annotation to end of file
        last_idx = int(ann.sample[-1] * scale_factor)
        last_note = ann.aux_note[-1]
        if "(AFIB" in last_note:
            rhythm_mask[last_idx:] = 1

        # 4. Segment based on Mask Density
        n_segments = len(ecg) // SEG_LEN
        saved_count = 0

        for i in range(n_segments):
            start = i * SEG_LEN
            end = start + SEG_LEN
            seg_signal = ecg[start:end].tolist()

            # CRITICAL CHECK: Look at the mask, not just change points
            # If more than 50% of the segment is AFIB, label it AFIB
            af_density = np.sum(rhythm_mask[start:end]) / SEG_LEN
            
            if af_density > 0.5:
                label = "Atrial Fibrillation"
            else:
                label = "Sinus Rhythm"

            out = {
                "ECG_CH_A": seg_signal,
                "fs": 250,
                "label": label,
                "dataset": "AFDB",
                "record": record_name,
                "segment_index": i
            }

            fname = OUTPUT / f"AFDB_{record_name}_seg_{i:04d}.json"
            fname.write_text(json.dumps(out))
            saved_count += 1
        
        return saved_count
        
    except FileNotFoundError:
        print(f" Record {record_name} not found")
        return 0
    except Exception as e:
        print(f" Error processing {record_name}: {e}")
        return 0

def main():
    # List of records (update as needed)
    records = ["04015", "04043", "04048", "04126", "04746", "04908", "04936", 
               "05091", "05121", "05261", "06426", "06453", "06997", "07162", 
               "07859", "07879", "08215", "08219", "08378", "08405", "08434", "08455"]
    
    total_segments = 0
    print(f" Converting AFDB records (Mask Logic)...")
    
    for rec in tqdm(records, desc="Records"):
        segments = convert_record(rec)
        total_segments += segments
    
    print(f"\n Conversion complete!")
    print(f"   • Output: {OUTPUT.absolute()}")

if __name__ == "__main__":
    main()
