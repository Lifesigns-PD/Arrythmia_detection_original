#!/usr/bin/env python3
import wfdb
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

INPUT = Path("../raw_data/mitdb")
OUTPUT = Path("data/converted_ecg")
OUTPUT.mkdir(parents=True, exist_ok=True)

MITDB_LABEL_MAP = {
    "N": "Sinus Rhythm",
    "A": "Atrial Premature Contraction",  # FIXED: 'A' is APC, not AFIB
    "V": "PVCs",
    "/": "Paced Rhythm",                 # FIXED: '/' is Paced, not SVT
    "f": "Atrial Fibrillation",         # (Note: AFIB is usually auxiliary, but 'f' is sometimes used)
    "!": "Ventricular Flutter/Fibrillation"
}

# Priority Hierarchy (Higher number = Higher Priority)
LABEL_PRIORITY = {
    "Ventricular Flutter/Fibrillation": 100,
    "Supraventricular Tachycardia": 90,
    "Atrial Fibrillation": 80,
    "Atrial Flutter": 75,
    "PVCs": 50,
    "Atrial Premature Contraction": 40,
    "Paced Rhythm": 20,
    "Sinus Rhythm": 1
}

SEG_LEN = 1250  # 10 sec * 125 Hz

def convert_record(record_name):
    try:
        record_path = str(INPUT / record_name)
        sig, fields = wfdb.rdsamp(record_path)
        ann = wfdb.rdann(record_path, "atr")

        ecg = sig[:, 0]  # channel A
        fs = fields.get('fs', 125) if isinstance(fields, dict) else fields.fs

        scale_factor = 1.0
        if fs != 125:
            # resample to 125
            from scipy.signal import resample
            # Calculate scale factor using original fs BEFORE updating it
            scale_factor = 125 / fs
            ecg = resample(ecg, int(len(ecg) * scale_factor))
            fs = 125

        # assign per-beat labels → convert to segment labels
        beat_labels = {}
        for idx, sym in zip(ann.sample, ann.symbol):
            # FIXED: Scale the annotation index
            new_idx = int(idx * scale_factor)
            beat_labels[new_idx] = MITDB_LABEL_MAP.get(sym, "Sinus Rhythm")

        n_segments = len(ecg) // SEG_LEN
        
        for i in range(n_segments):
            start = i * SEG_LEN
            end = start + SEG_LEN
            seg = ecg[start:end].tolist()

            seg_beats = [beat_labels[s] for s in beat_labels if start <= s < end]

            if seg_beats:
                # PRIORITY LOGIC: Take the most severe arrhythmia in the segment
                # instead of simple majority, which hides short runs of arrhythmia.
                label = max(seg_beats, key=lambda x: LABEL_PRIORITY.get(x, 0))
            else:
                label = "Sinus Rhythm"

            out = {
                "ECG_CH_A": seg,
                "fs": 250,
                "label": label,
                "dataset": "MITDB",
                "record": record_name,
                "segment_index": i
            }

            fname = OUTPUT / f"MITDB__{record_name}_seg_{i:04d}.json"
            fname.write_text(json.dumps(out))
        
        return n_segments
        
    except FileNotFoundError:
        print(f"[WARN]  Record {record_name} not found")
        return 0
    except Exception as e:
        print(f"[ERROR] Error processing {record_name}: {e}")
        return 0

def main():
    records = ["100", "101", "102", "103", "104", "105", "106", "107", "108"]
    total_segments = 0
    
    print(f"[PROCESSING] Converting {len(records)} MITDB records to JSON...")
    
    for rec in tqdm(records, desc="Records", unit="rec"):
        segments = convert_record(rec)
        total_segments += segments
    
    print(f"\n[DONE] Conversion complete!")
    print(f"   - Records: {len(records)}")
    print(f"   - Total segments: {total_segments}")
    print(f"   - Output: {OUTPUT}")

if __name__ == "__main__":
    main()
