"""
data_loader.py
--------------
Loads ECG JSON dataset for training CNN+Transformer model.

✔ Reads all *.json files inside dataset folder
✔ Normalizes label text to CLASS_NAMES
✔ Converts to numpy arrays suitable for collate_fn
✔ Handles corrupted JSON files gracefully
✔ Ensures every signal = 250 Hz, 10 sec (2500 samples)
"""

import json
import numpy as np
import psycopg2
from pathlib import Path
from scipy.signal import resample
import sys
# Ensure we can find the project root features (signal_processing)
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ============================================================
# EXPLICIT WINDOWING CONSTANTS (MANDATORY)
# ============================================================
WINDOW_SEC = 2.0

def extract_fixed_window(signal, fs, start_s, end_s):
    """
    Extracts a fixed-length window for XAI explanation.
    - For narrow events (cardiologist annotation): center on event
    - For wide events (whole-segment): find window with highest variance as proxy for actual anomaly location
    - fs: sampling rate
    - start_s, end_s: clinical event boundaries (in seconds)
    """
    window_samples = int(WINDOW_SEC * fs)

    start_i = int(start_s * fs)
    end_i   = min(len(signal), int(end_s * fs))

    event_duration = end_i - start_i

    if event_duration <= window_samples * 1.5:
        # Narrow event (cardiologist annotation) — center on event
        center_i = (start_i + end_i) // 2
    else:
        # Wide/whole-segment event — find highest variance window (best signal feature location)
        best_var = -1
        best_pos = start_i
        step = window_samples // 4  # 0.5s step for 2s window at 250Hz
        pos  = start_i
        while pos + window_samples <= end_i:
            var = np.var(signal[pos : pos + window_samples])
            if var > best_var:
                best_var = var
                best_pos = pos
            pos += step
        center_i = best_pos + window_samples // 2

    half  = window_samples // 2
    s_i   = max(0, center_i - half)
    e_i   = min(len(signal), center_i + half)
    win   = signal[s_i:e_i]

    # Pad if needed
    if len(win) < window_samples:
        pad = window_samples - len(win)
        win = np.pad(win, (0, pad), mode="constant")

    return win[:window_samples]

# ============================================================
# ============================================================
# ============================================================
# FINAL FIXED CLASS LIST (COMPREHENSIVE + COMBINATIONS)
# ============================================================

CLASS_NAMES = [
    # 0-21: Standard
    "Sinus Rhythm",                  # 0
    "Sinus Bradycardia",             # 1
    "Sinus Tachycardia",             # 2
    "Supraventricular Tachycardia",  # 3
    "Atrial Fibrillation",           # 4
    "Atrial Flutter",                # 5
    "Junctional Rhythm",             # 6
    "Idioventricular Rhythm",        # 7
    "Ventricular Tachycardia",       # 8
    "Ventricular Fibrillation",      # 9
    "1st Degree AV Block",           # 10
    "2nd Degree AV Block Type 1",    # 11 
    "2nd Degree AV Block Type 2",    # 12 
    "3rd Degree AV Block",           # 13
    "PVC",                          # 14
    "PVC Bigeminy",                  # 15
    "PVC Trigeminy",                 # 16
    "PVC Couplet",                   # 17
    "PAC",                           # 18
    "PAC Bigeminy",                  # 19
    "Bundle Branch Block",           # 20
    "Artifact",                      # 21
    
    # NEW COMBINATIONS
    "Sinus Bradycardia + PVC",       # 22
    "Sinus Tachycardia + PVC",       # 23
    "Sinus Bradycardia + PAC",       # 24
    "Sinus Tachycardia + PAC",       # 25
    "Atrial Fibrillation + PVC",     # 26
    "Atrial Flutter + PVC",          # 27
    "1st Degree AV Block + PVC",     # 28
    "Sinus Bradycardia + PVC Bigeminy", # 29
    "Sinus Tachycardia + PVC Bigeminy", # 30
    
    # COMPLEX PATTERNS (Rules-Based / Advanced)
    "Atrial Couplet",                # 31
    "Atrial Run",                    # 32
    "Ventricular Run",               # 33
    "NSVT",                          # 34
    "PSVT",                          # 35
    "Pause",                         # 36
]

CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# ============================================================
# TASK-SPECIFIC CLASS LISTS (CHANGE 3)
# ============================================================

# RHYTHM MODEL: Detects primary pathology
ECTOPY_TERMS = [
    "PVC", "PAC",
    "Bigeminy", "Trigeminy",
    "Couplet", "Run", "NSVT"
]

RHYTHM_CLASS_NAMES = [
    "Supraventricular Tachycardia",
    "Atrial Fibrillation",
    "Atrial Flutter",
    "Junctional Rhythm",
    "Idioventricular Rhythm",
    "Ventricular Tachycardia",
    "Ventricular Fibrillation",
    "1st Degree AV Block",
    "2nd Degree AV Block Type 1",
    "2nd Degree AV Block Type 2",
    "3rd Degree AV Block",
    "Bundle Branch Block",
    "Artifact",
    "PSVT",
    "Pause"
]

# Hard Safety Assertion
for name in RHYTHM_CLASS_NAMES:
    for term in ECTOPY_TERMS:
        assert term not in name, f"ECTOPY LEAK in Rhythm model: {name}"

# Reverse Safety Check: No Rhythm terms in Ectopy model
RHYTHM_TERMS = ["AF", "Atrial Fibrillation", "Atrial Flutter", "Block", "SVT", "AFib"]
ECTOPY_CLASS_NAMES = [
    "None",   # 0
    "PVC",    # 1
    "PAC",    # 2
    "Run"     # 3
]

for name in ECTOPY_CLASS_NAMES:
    for term in RHYTHM_TERMS:
        assert term not in name, f"RHYTHM LEAK in Ectopy model: {name}"

RHYTHM_INDEX = {name: i for i, name in enumerate(RHYTHM_CLASS_NAMES)}
ECTOPY_INDEX = {name: i for i, name in enumerate(ECTOPY_CLASS_NAMES)}

def get_rhythm_label_idx(original_label_name):
    """
    RHYTHM TASK: Focuses on the base pathology.
    - AF + PVC -> AF (KEEP)
    - Sinus + PVC -> Sinus -> None (DROPPED)
    - PVCs -> None (DROPPED - Ectopy is not a rhythm)
    """
    if original_label_name is None: return None
    label = normalize_label(original_label_name)

    # 1. Strip everything after ' + ' to find the base rhythm
    if " + " in label:
        label = label.split(" + ")[0]

    # 2. Return index only if base rhythm is in our targeted list
    return RHYTHM_INDEX.get(label, None)

def get_ectopy_label_idx(original_label_name):
    """
    ECTOPY TASK: Focuses exclusively on events.
    - AF + PVC -> PVC
    - Sinus + PVC -> PVC
    - AF -> None
    - Sinus -> None
    """
    if original_label_name is None: return ECTOPY_INDEX["None"]
    label = normalize_label(original_label_name).upper()

    # Priority 1: Runs (Highest severity ectopy)
    if any(t in label for t in ["RUN", "NSVT"]):
        return ECTOPY_INDEX["Run"]
        
    # Priority 2: PACs (including Atrial Couplet/Bigeminy)
    if any(t in label for t in ["PAC", "ATRIAL"]):
        return ECTOPY_INDEX["PAC"]

    # Priority 3: PVCs (including bigeminy/couplets)
    if any(t in label for t in ["PVC", "BIGEMINY", "TRIGEMINY", "COUPLET", "VPB"]):
        return ECTOPY_INDEX["PVC"]

    # Default: No ectopy detected
    return ECTOPY_INDEX["None"]



TARGET_FS = 250
SEG_LEN = TARGET_FS * 10 

# ============================================================
# SIMPLE LABEL NORMALIZATION
# ============================================================

LABEL_MAP = {
    # Normals
    "NORMAL": "Sinus Rhythm", "NSR": "Sinus Rhythm", "NORM": "Sinus Rhythm",
    "SB": "Sinus Bradycardia", "BRADY": "Sinus Bradycardia", "SINUS BRADYCARDIA": "Sinus Bradycardia",
    "ST": "Sinus Tachycardia", "TACHY": "Sinus Tachycardia", "SINUS TACHYCARDIA": "Sinus Tachycardia", "SINUS TACH": "Sinus Tachycardia",
    
    # SVT / Atrial
    "SVT": "Supraventricular Tachycardia", 
    "AF": "Atrial Fibrillation", "AFIB": "Atrial Fibrillation", "ATRIAL FIBRILLATION": "Atrial Fibrillation",
    "AFL": "Atrial Flutter", "ATRIAL FLUTTER": "Atrial Flutter",
    
    # Junctional
    "JUNCTIONAL": "Junctional Rhythm", 
    
    # Ventricular
    "IVR": "Idioventricular Rhythm",
    "VT": "Ventricular Tachycardia", 
    "VF": "Ventricular Fibrillation", 
    
    # Blocks
    "1AVB": "1st Degree AV Block", "1' AV BLOCK": "1st Degree AV Block", 
    "WENCKEBACH": "2nd Degree AV Block Type 1", 
    "MOBITZ II": "2nd Degree AV Block Type 2", 
    "3AVB": "3rd Degree AV Block", 
    "BBB": "Bundle Branch Block", "LBBB": "Bundle Branch Block", "RBBB": "Bundle Branch Block",
    
    # Ectopy
    "PVC": "PVC", "VPB": "PVC",
    "PVC BIGEMINY": "PVC Bigeminy", 
    "PVC TRIGEMINY": "PVC Trigeminy", 
    "PVC COUPLET": "PVC Couplet", 
    "PAC": "PAC", 
    "PAC BIGEMINY": "PAC Bigeminy", 
    
    # Synonyms for MITDB/Clinical terminology
    "ATRIAL PREMATURE CONTRACTION": "PAC", 
    "APC": "PAC",
    
    "ARTIFACT": "Artifact"
}


def normalize_label(label: str):
    """Convert any dataset label into one of the comprehensive classes."""
    if label is None: return "Sinus Rhythm"
    if not isinstance(label, str): label = str(label)

    L = label.strip().upper()

    # Direct passthrough if already correct
    for c in CLASS_NAMES:
        if c.upper() == L:
            return c
    
    # Also Check exact map
    if L in LABEL_MAP: return LABEL_MAP[L]
    
    # Heuristic Fallbacks
    if "WENCKEBACH" in L: return "2nd Degree AV Block Type 1"
    if "MOBITZ" in L: return "2nd Degree AV Block Type 2"
    if "BIGEMINY" in L: 
        return "PVC Bigeminy" if "PVC" in L or "VENTRICULAR" in L else "PAC Bigeminy"
    if "FLUTTER" in L: return "Atrial Flutter"
    if "FIBRILLATION" in L:
        return "Ventricular Fibrillation" if "VENTRICULAR" in L else "Atrial Fibrillation"

    # NEW: Handle composite strings that might not be in LABEL_MAP directly (e.g. "AF+PVC")
    if "+" in L:
        parts = [p.strip() for p in L.split("+")]
        norm_parts = [normalize_label(p) for p in parts]
        return " + ".join(norm_parts)
    
    return "Sinus Rhythm" # Default fallback


# ============================================================
# DATASET - Lazy and robust JSON reading
# ============================================================

# ============================================================
# DATASET - SQL based loading
# ============================================================

class ECGDataset:
    """
    SQL-based dataset that loads ECG segments from ecg_features_annotatable.
    Replaces the old file-strolling JSON loader.
    """

    def __init__(self, mode='all', limit=None, task='all', **kwargs):
        """
        mode: 'all' to load everything, 'retrain' to load only clinician-corrected segments.
        limit: Max segments to load (int or None).
        task: Filter for task-specific labels (e.g. 'rhythm', 'ectopy').
        """
        self.samples = []
        self.task = task
        
        # Connection params (Match db_service.py/retrain.py)
        DB_PARAMS = {
            "host":     "127.0.0.1",
            "dbname":   "ecg_analysis",
            "user":     "ecg_user",
            "password": "sais",
            "port":     "5432",
        }

        print(f"[Dataset] Initializing SQL dataset (mode={mode}, limit={limit}, task={task})...")
        try:
            conn = psycopg2.connect(**DB_PARAMS)
            cur = conn.cursor()
            
            # Fetch signal_data (REAL[]), label, and fs
            base_query = """
                SELECT segment_id, signal_data, arrhythmia_label, segment_fs, filename
                FROM ecg_features_annotatable 
            """
            
            where_clauses = ["signal_data IS NOT NULL"]
            if mode == 'retrain':
                where_clauses.append("is_corrected = TRUE")
            
            query = base_query + " WHERE " + " AND ".join(where_clauses)
            query += " ORDER BY segment_id DESC" # Get recent ones first if limited
            
            if limit:
                query += f" LIMIT {limit}"
            
            cur.execute(query)
            rows = cur.fetchall()
            
            for seg_id, signal_raw, label_txt, fs, filename in rows:
                # 1. Parse signal (Postgres REAL[] comes back as list)
                sig = np.array(signal_raw, dtype=np.float32)
                
                # 2. Resample & Fix Length (10s @ 250Hz = 2500 samples)
                fs = int(fs or 125)
                sig = self._resample_and_fixlen(sig, fs)
                
                # 3. Resolve label index based on actual task
                if self.task == 'rhythm':
                    y = get_rhythm_label_idx(label_txt)
                elif self.task == 'ectopy':
                    y = get_ectopy_label_idx(label_txt)
                else:
                    label_norm = normalize_label(label_txt or "Sinus Rhythm")
                    y = CLASS_INDEX.get(label_norm, 0)
                
                # Skip if label is invalid for the specified task
                if y is None: 
                    continue

                self.samples.append({
                    "signal": sig,
                    "label": int(y),
                    "id": seg_id,
                    "filename": filename
                })
                
            cur.close()
            conn.close()
            print(f"[Dataset] Loaded {len(self.samples)} samples from SQL.")
            
        except Exception as e:
            print(f"[ERROR] Failed to load SQL dataset: {e}")
            raise

    def _resample_and_fixlen(self, sig, orig_fs):
        """Standardizes signal to 250 Hz and 2500 samples."""
        if orig_fs != TARGET_FS and len(sig) > 1:
            try:
                new_len = int(len(sig) * float(TARGET_FS) / float(orig_fs))
                sig = resample(sig, new_len).astype(np.float32)
            except Exception:
                # Fallback: numpy interp
                idx_old = np.arange(len(sig))
                idx_new = np.linspace(0, len(sig) - 1, int(len(sig) * float(TARGET_FS) / float(orig_fs)))
                sig = np.interp(idx_new, idx_old, sig).astype(np.float32)

        # Pad/truncate to SEG_LEN (2500)
        if len(sig) < SEG_LEN:
            pad = int(SEG_LEN - len(sig))
            sig = np.pad(sig, (0, pad))
        elif len(sig) > SEG_LEN:
            sig = sig[:int(SEG_LEN)]

        return sig.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "signal": s["signal"], 
            "label": s["label"], 
            "meta": {"id": s["id"], "filename": s["filename"]}
        }


def collate_fn(batch):
    """Batches signals and labels for training."""
    signals = np.stack([b["signal"] for b in batch])
    labels = np.array([b["label"] for b in batch])
    return signals, labels

# Backward Compatibility Alias
ECGRawDatasetSQL = ECGDataset

# ============================================================
# END OF DATA_LOADER
# ============================================================

