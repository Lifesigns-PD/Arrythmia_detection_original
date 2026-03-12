import os
import ast
import json
import wfdb
import pandas as pd
import numpy as np
import psycopg2
import neurokit2 as nk
from scipy.signal import resample
from tqdm import tqdm
from pathlib import Path

# Configuration
DATA_DIR = str(Path(__file__).resolve().parent.parent / "raw_data" / "ptbxl")
FS_ORIGINAL = 500
FS_TARGET = 125
TARGET_LENGTH = 1250  # 10 seconds * 125 Hz

# ---------------------------------------------------------------------------
# PTB-XL SCP Code -> Clinical Label Mapping
# Only rhythm/conduction/ectopy codes are recognized.
# Records with ONLY non-arrhythmia codes (MI, STTC, etc.) are SKIPPED.
# ---------------------------------------------------------------------------
SCP_TO_LABEL = {
    # Normal
    "NORM":  "Sinus Rhythm",

    # Sinus rate variants (map to Sinus Rhythm for rhythm model)
    "SR":    "Sinus Rhythm",
    "SBRAD": "Sinus Bradycardia",
    "STACH": "Sinus Tachycardia",
    "SARRH": "Sinus Rhythm",       # Sinus arrhythmia is benign

    # Atrial
    "AFIB":  "Atrial Fibrillation",
    "AFLT":  "Atrial Flutter",
    "SVT":   "Supraventricular Tachycardia",
    "PSVT":  "Supraventricular Tachycardia",
    "SVTAC": "Supraventricular Tachycardia",

    # Junctional
    "JIDR":  "Junctional Rhythm",

    # Ventricular
    "IVRT":  "Idioventricular Rhythm",
    "VT":    "Ventricular Tachycardia",
    "VFIB":  "Ventricular Fibrillation",

    # AV Blocks
    "1AVB":  "1st Degree AV Block",
    "2AVB":  "2nd Degree AV Block Type 1",
    "3AVB":  "3rd Degree AV Block",

    # Bundle Branch Blocks
    "LBBB":  "Bundle Branch Block",
    "RBBB":  "Bundle Branch Block",
    "ILBBB": "Bundle Branch Block",
    "IRBBB": "Bundle Branch Block",
    "CLBBB": "Bundle Branch Block",
    "CRBBB": "Bundle Branch Block",
    "LAFB":  "Bundle Branch Block",
    "LPFB":  "Bundle Branch Block",

    # Ectopy
    "PVC":   "PVC",
    "BIGU":  "PVC Bigeminy",
    "TRIGU": "PVC Trigeminy",
    "PAC":   "PAC",
    "SVARR": "PAC",

    # Pause
    "PACE":  "Other Arrhythmia",
}


def parse_scp_label(scp_codes_str):
    """
    Parse PTB-XL scp_codes column and return a recognized clinical label.
    Returns None if the record has no recognized arrhythmia/rhythm codes.
    """
    try:
        codes = ast.literal_eval(scp_codes_str)
    except Exception:
        return None

    if not isinstance(codes, dict):
        return None

    # Find the highest-likelihood recognized code
    best_label = None
    best_likelihood = -1.0

    for code, likelihood in codes.items():
        code_upper = code.strip().upper()
        if code_upper in SCP_TO_LABEL and float(likelihood) > best_likelihood:
            best_likelihood = float(likelihood)
            best_label = SCP_TO_LABEL[code_upper]

    return best_label  # None if no recognized codes found


def connect_db():
    return psycopg2.connect(
        host="127.0.0.1", database="ecg_analysis", user="ecg_user", password="sais"
    )


def main():
    print("Loading PTB-XL metadata CSV...")
    csv_path = os.path.join(DATA_DIR, 'ptbxl_database.csv')
    df = pd.read_csv(csv_path, index_col='ecg_id')

    conn = connect_db()
    conn.autocommit = True
    cur = conn.cursor()

    INSERT_SQL = """
        INSERT INTO ecg_features_annotatable (
            dataset_source, filename, segment_index, signal_data, raw_signal,
            features_json, arrhythmia_label, segment_fs, events_json
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (filename, segment_index) DO NOTHING;
    """

    # Labels that represent ectopy (need both RHYTHM and ECTOPY events)
    ECTOPY_LABELS = {"PVC", "PAC", "PVC Bigeminy", "PVC Trigeminy", "PVC Quadrigeminy",
                     "PAC Bigeminy", "PAC Trigeminy"}

    success_count = 0
    skipped_no_label = 0

    print("Processing and inserting records into PostgreSQL...")
    for ecg_id, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # 1. Parse label from scp_codes
            label = parse_scp_label(row['scp_codes'])
            if label is None:
                skipped_no_label += 1
                continue

            # 2. Read the 500Hz signal
            file_path = os.path.join(DATA_DIR, row['filename_hr'])
            record = wfdb.rdrecord(file_path)

            # Lead I (channel 0)
            lead_i = record.p_signal[:, 0]

            # 3. Resample to 125 Hz (1,250 points)
            signal_resampled = resample(lead_i, TARGET_LENGTH)

            # 4. Extract R-peaks at target sampling rate
            try:
                _, info = nk.ecg_peaks(signal_resampled, sampling_rate=FS_TARGET)
                r_peaks = info["ECG_R_Peaks"].tolist()
            except Exception:
                r_peaks = []

            features_dict = {"r_peaks": r_peaks}

            # 5. Build events_json (RHYTHM event always, ECTOPY if applicable)
            events_list = [{
                "event_type": label,
                "event_category": "RHYTHM",
                "start_time": 0.0,
                "end_time": 10.0,
                "annotation_source": "imported"
            }]
            if label in ECTOPY_LABELS:
                events_list.append({
                    "event_type": label,
                    "event_category": "ECTOPY",
                    "start_time": 0.0,
                    "end_time": 10.0,
                    "annotation_source": "imported"
                })

            # 6. Insert into Database
            filename = f"ptbxl_{ecg_id}"
            signal_list = signal_resampled.tolist()

            cur.execute(
                INSERT_SQL,
                (
                    "ptbxl",
                    filename,
                    0,
                    signal_list,
                    json.dumps(signal_list),
                    json.dumps(features_dict),
                    label,
                    FS_TARGET,
                    json.dumps(events_list)
                )
            )
            success_count += 1

        except Exception:
            pass

    print(f"Finished! Loaded {success_count} PTB-XL segments. Skipped {skipped_no_label} (no recognized arrhythmia/rhythm code).")
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
