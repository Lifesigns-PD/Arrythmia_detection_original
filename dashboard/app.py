from flask import Flask, render_template, jsonify, request, redirect, url_for
import sys
import os
from pathlib import Path

# --- FOLDER RESTRUCTURE FIX ---
# Add project root and sibling folders to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "database"))
sys.path.append(str(BASE_DIR / "xai"))
sys.path.append(str(BASE_DIR / "models_training"))

from database import db_service
import json
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import resample_poly, butter, filtfilt, find_peaks, welch
from scipy.interpolate import interp1d
from werkzeug.utils import secure_filename
from typing import List, Dict, Any, Tuple
import warnings
import psycopg2
import subprocess

# XAI – Option A (clinical text + model prediction)
from xai import explain_segment, explain_decision, reset_model
from decision_engine.rhythm_orchestrator import RhythmOrchestrator
from decision_engine.models import SegmentDecision
from models_training.data_loader import CLASS_NAMES, RHYTHM_CLASS_NAMES, ECTOPY_CLASS_NAMES

# Suppress harmless scipy warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# Flask App & Global Config
# =========================================================

app = Flask(__name__)

TARGET_FS = 250
SEGMENT_DURATION_S = 10.0
SEGMENT_LENGTH = int(TARGET_FS * SEGMENT_DURATION_S)
HRV_INTERP_FS = 4.0

# Where individual uploaded JSONs go
# Point to shared 'data/ecg_data' folder
DATA_ROOT = BASE_DIR / "data"
app.config["UPLOAD_FOLDER"] = str(DATA_ROOT / "ecg_data")
DATA_ROOT_DIR = Path(app.config["UPLOAD_FOLDER"])
os.makedirs(DATA_ROOT_DIR, exist_ok=True)

# Folder that holds the bulk JSON datasets already converted
DATASET_JSON_DIR = DATA_ROOT / "input_segments"
os.makedirs(DATASET_JSON_DIR, exist_ok=True)


# =========================================================
# ECG Loading & Preprocessing
# =========================================================

def _load_data_from_json(file_path: Path) -> Tuple[np.ndarray, int]:
    """
    Loads ECG data from a JSON file and returns (signal, original_fs).
    Supports:
      - SensorData[0]["ECG_CH_A"]
      - Top-level "ECG_CH_A" / "ECG_CH_B"
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    signal = None
    original_fs = 250
    filename = file_path.name

    # Structure 1: SensorData list (PTB-XL style, etc.)
    if isinstance(data.get("SensorData"), list) and data["SensorData"]:
        row = data["SensorData"][0]
        if "ECG_CH_A" in row:
            signal = np.array(row["ECG_CH_A"], dtype=float)

        # Heuristics for fs
        if "PTBXL" in filename.upper() or "PTB-XL" in filename.upper():
            original_fs = 500  # many PTB-XL records
        elif "MITDB" in filename.upper() or "MIT-BIH" in filename.upper():
            original_fs = 360

    # Structure 2: top-level "ECG_CH_A"
    elif "ECG_CH_A" in data:
        signal = np.array(data["ECG_CH_A"], dtype=float)
        if "MITDB" in filename.upper() or "MIT-BIH" in filename.upper():
            original_fs = 360

    elif "ECG_CH_B" in data:
        signal = np.array(data["ECG_CH_B"], dtype=float)

    if signal is None:
        raise ValueError(f"Could not find valid ECG channel in JSON file: {filename}")

    return signal, original_fs


from signal_processing.cleaning import clean_signal

def _preprocess(signal: np.ndarray, original_fs: int) -> np.ndarray:
    """
    Standard preprocessing using signal_processing module.
    1. Resample to Target Rate
    2. Clean (Baseline Removal + Powerline Removal)
    """
    # Resample first if needed
    if original_fs != TARGET_FS:
        signal = resample_poly(signal, TARGET_FS, original_fs).astype(np.float32)
    else:
        signal = signal.astype(np.float32)
        
    # Apply centralized cleaning
    return clean_signal(signal, TARGET_FS)


def _detect_r_peaks_neurokit(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Robust R-peak detection using NeuroKit2.
    """
    try:
        # 1. Clean signal (removes baseline wander, powerline noise)
        cleaned = nk.ecg_clean(signal, sampling_rate=fs, method="neurokit")
        # 2. Find Peaks
        # method='neurokit' is steep-slope based, very good for QRS
        signals, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit")
        peaks = info.get("ECG_R_Peaks", [])
        # Ensure we return int array
        return np.array(peaks, dtype=int)
    except Exception as e:
        print(f"NeuroKit Peak Detection Failed: {e}")
        return np.array([], dtype=int)

def _r_peak_detection(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Wrapper for NeuroKit detection to maintain compatibility.
    """
    return _detect_r_peaks_neurokit(signal, fs)


# =========================================================
# HRV, Morphology & PR/QRS Features
# =========================================================

def _calculate_frequency_hrv(rr_intervals_ms: np.ndarray) -> Dict[str, float]:
    """
    Frequency-domain HRV (VLF, LF, HF, LF/HF) using Welch.
    """
    out = {"VLF": 0.0, "LF": 0.0, "HF": 0.0, "LF_HF_ratio": 0.0}
    if len(rr_intervals_ms) < 5:
        return out

    rr_s = rr_intervals_ms / 1000.0
    t = np.cumsum(rr_s)
    t -= t[0]

    try:
        f_interp = interp1d(t, rr_intervals_ms, kind="cubic")
        t_new = np.arange(t[0], t[-1], 1.0 / HRV_INTERP_FS)
        rr_interp = f_interp(t_new)
    except ValueError:
        return out

    n = len(rr_interp)
    if n < 16:
        return out

    nperseg = min(n, 256)
    noverlap = nperseg // 2

    fxx, pxx = welch(
        rr_interp,
        fs=HRV_INTERP_FS,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
    )

    def band_power(f, p, band):
        idx = (f >= band[0]) & (f < band[1])
        if not np.any(idx):
            return 0.0
        return float(np.trapz(p[idx], f[idx]))

    vlf = band_power(fxx, pxx, (0.0, 0.04))
    lf = band_power(fxx, pxx, (0.04, 0.15))
    hf = band_power(fxx, pxx, (0.15, 0.4))

    out["VLF"] = vlf
    out["LF"] = lf
    out["HF"] = hf
    out["LF_HF_ratio"] = float(lf / hf) if lf > 0 and hf > 0 else 0.0
    return out


def _calculate_nonlinear_hrv(rr_intervals_ms: np.ndarray) -> Dict[str, float]:
    """
    Poincaré SD1/SD2.
    """
    out = {"SD1": 0.0, "SD2": 0.0}
    if len(rr_intervals_ms) < 2:
        return out

    rr_n = rr_intervals_ms[:-1]
    rr_n1 = rr_intervals_ms[1:]

    sd1 = np.sqrt(0.5 * np.var(rr_n - rr_n1))
    sd2 = np.sqrt(2 * np.var(rr_intervals_ms) - sd1**2)

    out["SD1"] = float(sd1)
    out["SD2"] = float(sd2)
    return out


def _compute_qrs_durations(segment: np.ndarray, segment_r_peaks: np.ndarray, fs: int) -> np.ndarray:
    """
    Estimate QRS durations using NeuroKit2 Delineation.
    Returns array of durations in ms for each QRS detected.
    """
    if segment_r_peaks is None or len(segment_r_peaks) == 0:
        return np.array([])
    
    try:
        # NeuroKit DWT (Discrete Wavelet Transform) is robust for delineation
        # It needs R-peaks. We pass current R-peaks to help it.
        # IMPROVEMENT: Apply 40Hz lowpass before delineation to ignore sharp noise spikes
        nyq = 0.5 * fs
        b, a = butter(2, 40/nyq, btype='low')
        smooth_seg = filtfilt(b, a, segment)
        
        _, waves = nk.ecg_delineate(smooth_seg, segment_r_peaks, sampling_rate=fs, method="dwt", show=False)
        
        # waves dictionary contains "ECG_R_Onsets" and "ECG_R_Offsets"
        # These are lists with NaNs for missing waves
        r_onsets = np.array(waves.get("ECG_R_Onsets", []))
        r_offsets = np.array(waves.get("ECG_R_Offsets", []))
        
        # Ensure we have data
        if len(r_onsets) == 0 or len(r_offsets) == 0:
            return np.array([])
            
        # Create mask for valid pairs
        valid_mask = ~pd.isna(r_onsets) & ~pd.isna(r_offsets)
        
        if np.sum(valid_mask) == 0:
            return np.array([])
            
        # Calculate Durations
        durations_ms = (r_offsets[valid_mask] - r_onsets[valid_mask]) * 1000.0 / fs
        
        # Filter physiological range (clinically 50ms to 250ms)
        durations_ms = durations_ms[(durations_ms >= 50) & (durations_ms <= 250)]
        
        return durations_ms

    except Exception as e:
        print(f"NeuroKit QRS Calc Failed: {e}")
        return np.array([80.0]) # Fallback default



def _calculate_morphology_features(segment: np.ndarray, segment_r_peaks: np.ndarray) -> Dict[str, Any]:
    """
    QRS energy + QRS duration distribution (ms).
    """
    out: Dict[str, Any] = {
        "QRS_Avg_Energy": 0.0,
        "QRS_Energy_Std": 0.0,
        "qrs_durations_ms": [],
    }
    if len(segment_r_peaks) == 0:
        return out

    window_samples = int(0.100 * TARGET_FS)
    energies = []

    for r in segment_r_peaks:
        start = max(0, r - window_samples)
        end = min(len(segment), r + window_samples)
        qrs_seg = segment[start:end]
        energies.append(np.sum(qrs_seg**2))

    if energies:
        out["QRS_Avg_Energy"] = float(np.mean(energies))
        out["QRS_Energy_Std"] = float(np.std(energies))

    qrs_list = _compute_qrs_durations(segment, segment_r_peaks, TARGET_FS)
    out["qrs_durations_ms"] = qrs_list.tolist() if qrs_list.size > 0 else []
    return out


def _sanitize_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure no NaN/Inf for JSONB."""
    clean = {}
    for k, v in features.items():
        if isinstance(v, (float, np.floating)):
            if np.isnan(v) or np.isinf(v):
                clean[k] = 0.0
            else:
                clean[k] = float(v)
        else:
            clean[k] = v
    return clean


def _calculate_pr_interval(signal: np.ndarray, r_peaks: np.ndarray, fs: int) -> float:
    """
    Estimate PR interval using NeuroKit2 Delineation.
    Returns median PR interval in ms.
    """
    if r_peaks is None or len(r_peaks) == 0:
        return 0.0

    try:
        # Use NeuroKit's DWT method for delineation
        # IMPROVEMENT: Apply 40Hz lowpass before delineation to ignore high-frequency noise
        nyq = 0.5 * fs
        b, a = butter(2, 40/nyq, btype='low')
        smooth_sig = filtfilt(b, a, signal)

        _, waves = nk.ecg_delineate(smooth_sig, r_peaks, sampling_rate=fs, method="dwt", show=False)
        
        # P-onset to R-onset (beginning of QRS)
        p_onsets = np.array(waves.get("ECG_P_Onsets", []))
        r_onsets = np.array(waves.get("ECG_R_Onsets", []))
        
        if len(p_onsets) == 0 or len(r_onsets) == 0:
            return 0.0
            
        valid = ~pd.isna(p_onsets) & ~pd.isna(r_onsets)
        
        if np.sum(valid) == 0:
            return 0.0
            
        pr_vals = (r_onsets[valid] - p_onsets[valid]) * 1000.0 / fs
        
        # Filter valid range (100 - 400 ms) - Short PR < 100 is rare in normal sinus
        pr_vals = pr_vals[(pr_vals >= 100) & (pr_vals <= 400)]
        
        if len(pr_vals) == 0:
             return 0.0
             
        return float(np.nanmedian(pr_vals))
        
    except Exception as e:
        print(f"NK PR Failed: {e}")
        return 0.0


def _extract_segment_features(
    segment: np.ndarray, segment_r_peaks: np.ndarray, segment_idx: int
) -> Dict[str, Any]:
    """
    Full time-domain, HRV, and morphology features per 10 s segment.
    """
    features: Dict[str, Any] = {}

    features["segment_index"] = int(segment_idx)
    features["mean_amplitude"] = float(np.mean(segment))
    features["std_amplitude"] = float(np.std(segment))

    rr_intervals_ms = np.array([])
    if len(segment_r_peaks) >= 2:
        rr_samples = np.diff(segment_r_peaks)
        rr_intervals_ms = rr_samples * 1000.0 / TARGET_FS

    if rr_intervals_ms.size > 0:
        features["rr_intervals_ms"] = rr_intervals_ms.tolist()
        features["mean_rr"] = float(np.mean(rr_intervals_ms))
        features["mean_hr"] = (
            float(60.0 / (features["mean_rr"] / 1000.0))
            if features["mean_rr"] > 0
            else 0.0
        )
        features["SDNN"] = float(np.std(rr_intervals_ms))
        diff_rr = np.diff(rr_intervals_ms)
        features["RMSSD"] = (
            float(np.sqrt(np.mean(diff_rr**2))) if diff_rr.size > 0 else 0.0
        )
        features["pNN50"] = (
            float(np.sum(np.abs(diff_rr) > 50) / diff_rr.size)
            if diff_rr.size > 0
            else 0.0
        )
    else:
        features["rr_intervals_ms"] = []
        features["mean_rr"] = 0.0
        features["mean_hr"] = 0.0
        features["SDNN"] = 0.0
        features["RMSSD"] = 0.0
        features["pNN50"] = 0.0

    features.update(_calculate_morphology_features(segment, segment_r_peaks))
    features.update(_calculate_frequency_hrv(rr_intervals_ms))
    features.update(_calculate_nonlinear_hrv(rr_intervals_ms))

    # Calculate PR interval from waveform
    pr_interval = _calculate_pr_interval(segment, segment_r_peaks, TARGET_FS)
    features["pr_interval"] = float(pr_interval)
    
    return features


# =========================================================
# Ingestion: process an uploaded file & save features to SQL
# =========================================================

def process_and_save_record(file_path: Path) -> str:
    """
    Process one uploaded ECG JSON file:
      - preprocess
      - R-peaks
      - segment into 10s
      - compute features
      - store in ecg_features_annotatable
    """
    filename_key = str(file_path.relative_to(DATA_ROOT_DIR))

    try:
        raw_signal, original_fs = _load_data_from_json(file_path)
        processed_signal = _preprocess(raw_signal, original_fs)
        r_peaks_all = _r_peak_detection(processed_signal, TARGET_FS)
    except Exception as e:
        raise Exception(f"Processing failed for {filename_key}: {e}")

    db_service.setup_database()

    conn = None
    try:
        conn = db_service._connect()
        conn.commit()

        n_segments = len(processed_signal) // SEGMENT_LENGTH

        with conn.cursor() as cur:
            for i in range(n_segments):
                start = i * SEGMENT_LENGTH
                end = (i + 1) * SEGMENT_LENGTH
                segment = processed_signal[start:end]

                seg_r_peaks_abs = r_peaks_all[
                    (r_peaks_all >= start) & (r_peaks_all < end)
                ]
                seg_r_peaks_rel = seg_r_peaks_abs - start

                # Push raw_signal and segment_fs into ecg_features_annotatable
                cur.execute(
                    """
                    INSERT INTO ecg_features_annotatable (raw_signal, segment_fs)
                    VALUES (%s, %s)
                    """,
                    (raw_signal.tolist(), original_fs)
                )

        conn.commit()
        return filename_key

    except psycopg2.Error as e:
        raise Exception(f"Database error during insert: {e}")
    finally:
        if conn:
            conn.close()


# =========================================================
# Utility: load a segment signal from disk for plotting/XAI
# =========================================================

def _load_and_segment_raw_data(relative_path: str, segment_index: int) -> List[float]:
    """
    Load raw ECG segment used for plotting and XAI.

    1. Try ecg_data/<relative_path>
    2. If missing, try input_segments/<relative_path>.json
    """
    # Default location (uploads)
    file_path = DATA_ROOT_DIR / relative_path

    # Fallback for dataset JSONs imported via import_json_segments_to_sql
    if not file_path.exists():
        alt1 = DATASET_JSON_DIR / (relative_path + ".json")
        alt2 = DATASET_JSON_DIR / relative_path  # in case filename already has .json
        if alt1.exists():
            file_path = alt1
        elif alt2.exists():
            file_path = alt2
        else:
            raise FileNotFoundError(
                f"ECG file not found at {file_path} or {alt1} or {alt2}"
            )

    full_signal, original_fs = _load_data_from_json(file_path)
    full_signal = _preprocess(full_signal, original_fs)

    start = segment_index * SEGMENT_LENGTH
    end = (segment_index + 1) * SEGMENT_LENGTH
    segment = full_signal[start:end]
    return segment.tolist()


# =========================================================
# Flask Routes
# =========================================================

@app.route("/")
def index():
    """
    Main dashboard view – loads first segment from SQL.
    """
    row = db_service.fetch_one("SELECT MIN(segment_id) FROM ecg_features_annotatable;")
    first_segment_id = row[0] if row and row[0] else 1

    load_segment_id = request.args.get("load_segment_id", first_segment_id)

    try:
        file_list = sorted([p.name for p in DATA_ROOT_DIR.iterdir() if p.is_file()])
    except Exception:
        file_list = []

    all_segments = db_service.get_all_segments()

    return render_template(
        "index.html",
        file_list=file_list,
        all_segments=all_segments,
        initial_segment_id=int(load_segment_id),
        TARGET_FS=TARGET_FS,
    )


@app.route("/upload_and_process", methods=["POST"])
def upload_and_process():
    """
    Optional: upload a JSON ECG and immediately index segments into the DB.
    """
    if "file" not in request.files or request.files["file"].filename == "":
        return redirect(url_for("index"))

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = DATA_ROOT_DIR / filename

    try:
        file.save(filepath)
        filename_key = process_and_save_record(filepath)
        new_segment_id = db_service.get_first_segment_id_by_filename(filename_key)
        if not new_segment_id:
            new_segment_id = 1
        return redirect(f"/?load_segment_id={new_segment_id}")
    except Exception as e:
        return (
            f"ERROR: Processing or Database Insertion Failed: {e}. "
            "Please ensure the JSON file is valid.",
            500,
        )


# =========================================================
# XAI Clinical Explanation Endpoint (Option A)
# =========================================================

@app.route("/api/xai/<int:segment_id>")
def api_xai(segment_id: int):
    """
    Standardized Decision Engine & XAI Endpoint.
    Leverages pre-computed results from ecg_features_annotatable if available.
    """
    new_data = db_service.get_segment_new(segment_id)
    # Success from new table (Migration is handled inside get_segment_new)
    if not new_data:
        return jsonify({"error": "Segment not found"}), 404

    events_json = new_data["events_json"]
    bg_rhythm = new_data.get("background_rhythm") or "Sinus Rhythm"
    
    from decision_engine.models import Event, EventCategory, DisplayState, SegmentDecision, SegmentState
    from decision_engine.rules import apply_display_rules
    import uuid
    
    # 1. Recover Event Objects
    event_objs = []
    raw_events = events_json.get("events", []) if isinstance(events_json, dict) else events_json
    
    # ... (rest of the logic for processing events)
    for e_dict in raw_events:
        temp_e = e_dict.copy()
        if "event_id" not in temp_e: temp_e["event_id"] = str(uuid.uuid4())
        if "start_time" not in temp_e: temp_e["start_time"] = 0.0
        if "end_time" not in temp_e: temp_e["end_time"] = 0.0
        if "event_type" not in temp_e: temp_e["event_type"] = "Unknown"
        
        if "event_category" not in temp_e:
            etype = temp_e["event_type"]
            etype_upper = etype.upper()
            if any(term in etype_upper for term in ["PVC", "PAC", "BIGEMINY", "TRIGEMINY", "COUPLET"]):
                temp_e["event_category"] = EventCategory.ECTOPY
            else:
                temp_e["event_category"] = EventCategory.RHYTHM
        elif isinstance(temp_e["event_category"], str):
            temp_e["event_category"] = EventCategory(temp_e["event_category"])
            
        if "display_state" in temp_e and isinstance(temp_e["display_state"], str):
            temp_e["display_state"] = DisplayState(temp_e["display_state"])
        
        valid_keys = Event.__annotations__.keys()
        e_filtered = {k: v for k, v in temp_e.items() if k in valid_keys}
        event_objs.append(Event(**e_filtered))
    
    from decision_engine.rules import apply_ectopy_patterns, apply_display_rules
    apply_ectopy_patterns(event_objs)
    final_display = apply_display_rules(bg_rhythm, event_objs)
    
    _raw_state = new_data.get("segment_state") or "ANALYZED"
    try:
        _state = SegmentState(_raw_state)
    except ValueError:
        _state = SegmentState.ANALYZED

    decision = SegmentDecision(
        segment_index=new_data.get("segment_index") or segment_id,
        segment_state=_state,
        background_rhythm=bg_rhythm,
        events=event_objs,
        final_display_events=final_display,
        xai_notes=new_data.get("features_json") or {}
    )
    
    explanation_text = explain_decision(decision)
    if isinstance(events_json, dict) and events_json.get("explanation"):
        explanation_text = events_json["explanation"]
        
    response = decision.to_dict()
    response["explanation"] = explanation_text
    return jsonify(response)


# =========================================================
# Segment Fetch (ECG + Features + Annotation) for Dashboard
# =========================================================



@app.route("/api/segment/<int:segment_id>")
def get_segment_api(segment_id: int):
    """
    Fetch all necessary info for a specific segment ID.
    Prioritizes the optimized ecg_features_annotatable table.
    """
    meta = db_service.get_segment_new(segment_id)
    # Success from new table (Migration is handled inside get_segment_new)
    if not meta:
        return jsonify({"error": "Segment not found"}), 404
        
    # get_segment_new returns key "signal" (not "raw_signal")
    raw_signal = meta.get("signal")
    # If signal is still missing (NULL in both tables), try disk load
    if not raw_signal or len(raw_signal) == 0:
        try:
            raw_signal = _load_and_segment_raw_data(meta["filename"], meta["segment_index"])
        except Exception as e:
            return jsonify({"error": f"Failed to load ECG: {e}"}), 500

    # get_segment_new returns key "features" (not "features_json")
    features = meta.get("features") or {}
    mean_hr = float(features.get("mean_hr", 0.0))

    # Parse r-peaks from DB (Prioritize manual edits)
    r_peaks_for_frontend = meta.get("r_peaks_in_segment")

    # Use the actual fs stored in the DB — critical for coordinate sync
    # MIT-BIH data is at 125 Hz; uploading a file via app might be 250 Hz.
    # Using a hard-coded TARGET_FS here would cause a 2× coordinate shift.
    seg_fs = int(meta.get("segment_fs") or 125)

    if not r_peaks_for_frontend:
        # 1. Calculate FRESH R-peaks on the fly ONLY if missing from DB
        try:
            r_peaks_arr = _detect_r_peaks_neurokit(np.array(raw_signal), seg_fs)
            # Convert to string for JSON
            r_peaks_for_frontend = ",".join(str(x) for x in r_peaks_arr)
        except Exception:
            r_peaks_arr = np.array([], dtype=int)
            r_peaks_for_frontend = ""
    else:
        # Load existing peaks from DB
        try:
            r_peaks_arr = np.array([int(x) for x in r_peaks_for_frontend.split(",") if x.strip()], dtype=int)
        except:
            r_peaks_arr = np.array([], dtype=int)

    # Calculate LIVE Heart Rate
    if len(r_peaks_arr) >= 2:
        rr_intervals = np.diff(r_peaks_arr)
        mean_rr = np.mean(rr_intervals)
        if mean_rr > 0:
            mean_hr = 60.0 * seg_fs / mean_rr
    try:
        pr_interval_ms = _calculate_pr_interval(np.array(raw_signal), r_peaks_arr, seg_fs)
    except Exception:
        pr_interval_ms = 0.0

    # QRS width from features (robust to None/NaN)
    # QRS width from features (robust to None/NaN) -- RECALCULATED ON THE FLY with NeuroKit
    # Note: We prioritize recomputing it to fix old data issues in dashboard
    try:
        qrs_durations = _compute_qrs_durations(np.array(raw_signal), r_peaks_arr, seg_fs)
        if len(qrs_durations) > 0:
            # Use MEDIAN for robustness against delineation errors
            qrs_mean_ms = float(np.nanmedian(qrs_durations))
        else:
             # Fallback to stored features if NeuroKit returns nothing (rare)
             qrs_mean_ms = 0.0 
             qrs_list = features.get("qrs_durations_ms")
             if isinstance(qrs_list, list):
                q_clean = [float(v) for v in qrs_list if v is not None and not np.isnan(float(v))]
                if q_clean:
                    qrs_mean_ms = float(sum(q_clean)/len(q_clean))
    except Exception as e:
        qrs_mean_ms = float(features.get("mean_qrs", 0.0))

    # NaN Guards for JSON compliance
    if np.isnan(qrs_mean_ms): qrs_mean_ms = 0.0
    if np.isnan(pr_interval_ms): pr_interval_ms = 0.0

    # constructing structured response for frontend
    vitals = {
        "bpm": round(mean_hr, 1),
        "pr_interval": round(pr_interval_ms, 0),
        "qrs_duration": round(qrs_mean_ms, 0)
    }

    labels = {
        "ai_prediction": meta.get("arrhythmia_label") or "Unlabeled",
        "imported_label": meta.get("dataset_source") or "Unknown"
    }

    return jsonify(
        {
            "segment_id": segment_id,
            "filename": meta["filename"],
            "segment_index": meta["segment_index"],
            "raw_signal": raw_signal,
            "fs": seg_fs,
            "length": len(raw_signal),
            "vitals": vitals,
            "labels": labels,
            "r_peaks": r_peaks_for_frontend,
            "notes": meta.get("cardiologist_notes") or "",
            "corrected_by": meta.get("corrected_by"),
            "corrected_at": meta.get("corrected_at")
        }
    )

@app.route("/api/next_unverified/<int:current_id>")
def get_next_unverified_api(current_id: int):
    """Finds the next unverified segment ID (is_corrected = FALSE) starting from current_id."""
    conn = db_service._connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT segment_id FROM ecg_features_annotatable 
                WHERE segment_id > %s AND is_corrected = FALSE 
                ORDER BY segment_id ASC LIMIT 1
            """, (current_id,))
            row = cur.fetchone()
            if row:
                return jsonify({"ok": True, "next": row[0]})
            
            # If nothing found after current, wrap around to start
            cur.execute("SELECT segment_id FROM ecg_features_annotatable WHERE is_corrected = FALSE ORDER BY segment_id ASC LIMIT 1")
            row = cur.fetchone()
            if row:
                return jsonify({"ok": True, "next": row[0]})
                
            return jsonify({"ok": False, "msg": "No unverified segments left!"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        conn.close()


# =========================================================
# Annotation Save Endpoint
# =========================================================

# LEGACY SEGMENT ANNOTATION DROPPED IN FAVOR OF EVENT ANNOTATION

import uuid
@app.route("/api/annotate_beats", methods=["POST"])
def annotate_beats():
    """
    Bulk apply label to selected beat indices.
    Each beat is converted to a strict ±0.3s (0.6s total) window.
    """
    data = request.json
    segment_id = data.get("segment_id")
    beat_indices = data.get("beat_indices", [])
    label = data.get("label")

    if not segment_id or not beat_indices or not label:
        return jsonify({"error": "Missing required fields"}), 400

    # Strict Medical Window: ±0.3s around peak
    WINDOW_S = 0.3
    
    # Determine event_category from label
    label_upper = label.upper()
    if any(term in label_upper for term in ["PVC", "PAC", "BIGEMINY", "TRIGEMINY", "COUPLET"]):
        event_category = "ECTOPY"
    else:
        event_category = "RHYTHM"

    # Convert beat_indices to R-peak index within the segment
    # We need the R-peaks to compute the beat_index (ordinal position)
    r_peaks_str = None
    try:
        conn = db_service._connect()
        with conn.cursor() as cur:
            cur.execute("SELECT r_peaks_in_segment FROM ecg_features_annotatable WHERE segment_id = %s", (segment_id,))
            row = cur.fetchone()
            if row and row[0]:
                r_peaks_str = row[0]
        conn.close()
    except Exception:
        pass
    
    r_peak_list = []
    if r_peaks_str:
        r_peak_list = sorted([int(x) for x in r_peaks_str.split(',') if x.strip()])

    success_count = 0
    for idx_rel in beat_indices:
        # Convert relative index (samples) to relative time (seconds)
        peak_time = idx_rel / TARGET_FS
        
        # Compute beat_index: which R-peak number is this beat?
        beat_index = None
        if r_peak_list:
            for i, rp in enumerate(r_peak_list):
                if abs(rp - idx_rel) < 50:  # Within 50 samples (~200ms at 250Hz)
                    beat_index = i
                    break
        
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": label,
            "event_category": event_category,
            "start_time": max(0, peak_time - WINDOW_S),
            "end_time": min(10.0, peak_time + WINDOW_S),
            "beat_indices": [beat_index] if beat_index is not None else [],
            "annotation_source": "cardiologist",
            "annotation_status": "confirmed",
            "used_for_training": True
        }
        
        if db_service.save_event_to_db(segment_id, event):
            success_count += 1
    return jsonify({"status": "ok", "applied": success_count})


@app.route("/api/delete_annotation", methods=["POST"])
def delete_annotation():
    """
    Remove a specific annotation event.
    Payload: { "segment_id": int, "event_id": str }
    """
    data = request.json
    segment_id = data.get("segment_id")
    event_id = data.get("event_id")

    if not segment_id or not event_id:
        return jsonify({"error": "Missing segment_id or event_id"}), 400

    if db_service.delete_event(segment_id, event_id):
        return jsonify({"status": "ok", "message": "Annotation deleted"})
    else:
        return jsonify({"error": "Failed to delete or event not found"}), 500


@app.route("/api/clear_all_annotations", methods=["POST"])
def api_clear_all_annotations():
    """Wipes all annotation events and resets verification status for a segment."""
    data = request.json
    segment_id = data.get("segment_id")

    if not segment_id:
        return jsonify({"error": "Missing segment_id"}), 400

    if db_service.clear_all_annotations(segment_id):
        return jsonify({"status": "ok", "message": "All annotations cleared"})
    else:
        return jsonify({"error": "Failed to clear annotations"}), 500


# =========================================================
# Export Corrected Segments → retraining_data/ (JSON)
# =========================================================

# EXPORT TO JSON DROPPED - TRAINING USES DIRECT SQL CONNECTION


# =========================================================
# Next / Previous Navigation
# =========================================================

@app.route("/api/next_segment/<int:segment_id>")
def api_next_segment(segment_id: int):
    """
    Return the next available segment_id after the given one.
    If none, wrap to the minimum segment_id.
    """
    conn = db_service._connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT MIN(segment_id)
                FROM ecg_features_annotatable
                WHERE segment_id > %s
                """,
                (segment_id,),
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                return jsonify({"ok": True, "next": int(row[0])})

            cur.execute("SELECT MIN(segment_id) FROM ecg_features_annotatable")
            row = cur.fetchone()
            if row and row[0] is not None:
                return jsonify({"ok": True, "next": int(row[0])})

        return jsonify({"ok": False, "error": "No segments"}), 404
    finally:
        conn.close()


@app.route("/api/prev_segment/<int:segment_id>")
def api_prev_segment(segment_id: int):
    """
    Return the previous available segment_id before the given one.
    If none, wrap to the maximum segment_id.
    """
    conn = db_service._connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT MAX(segment_id)
                FROM ecg_features_annotatable
                WHERE segment_id < %s
                """,
                (segment_id,),
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                return jsonify({"ok": True, "prev": int(row[0])})

            cur.execute("SELECT MAX(segment_id) FROM ecg_features_annotatable")
            row = cur.fetchone()
            if row and row[0] is not None:
                return jsonify({"ok": True, "prev": int(row[0])})

        return jsonify({"ok": False, "error": "No segments"}), 404
    finally:
        conn.close()


# =========================================================
# Retrain Model Endpoint (Button in UI)
# =========================================================

@app.route("/api/retrain_model", methods=["GET", "POST"])
def api_retrain_model():
    """
    Called by dashboard "Retrain Model Using Corrected Segments" button.

    Pipeline:
      1) export_corrected_segments()  -> retraining_data/
      2) run retrain_model.py         -> outputs/checkpoints/best_model.pth
      3) xai.reset_model()            -> reload new weights on next XAI call
    """
    try:
        # 🔒 ISSUE 4: Retraining Gate enforcement
        count = db_service.count_confirmed_cardiologist_events()
        if count < 10: 
            return jsonify({
                "error": f"Insufficient data. Need at least 10 cardiologist-confirmed events (Current: {count})."
            }), 400

        script_path = BASE_DIR / "models_training" / "retrain.py"

        with open("training_log.txt", "w") as log_file:
            for task in ["rhythm", "ectopy"]:
                subprocess.Popen(
                    [sys.executable, str(script_path), "--task", task],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=str(BASE_DIR / "models_training")
                )

        return jsonify({"status": "ok", "message": "Training started for both rhythm and ectopy! Check training_log.txt for progress."})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/update_rpeaks", methods=["POST"])
def update_rpeaks():
    data = request.json
    segment_id = data.get("segment_id")
    new_peaks = data.get("r_peaks", [])

    if not segment_id:
        return jsonify({"error": "Missing segment_id"}), 400

    peaks_str = ",".join(map(str, sorted(new_peaks)))
    conn = db_service._connect()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE ecg_features_annotatable SET r_peaks_in_segment = %s WHERE segment_id = %s", (peaks_str, segment_id))
        conn.commit()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route("/api/verify_segment", methods=["POST"])
def verify_segment():
    data = request.json
    segment_id = data.get("segment_id")
    bg_rhythm = data.get("background_rhythm")

    if not segment_id:
        return jsonify({"error": "Missing segment_id"}), 400

    success = db_service.update_segment_status(segment_id, "ANALYZED", bg_rhythm)
    if success:
        return jsonify({"status": "ok"})
    else:
        return jsonify({"error": "Database update failed"}), 500


@app.route("/api/segment/<int:segment_id>/update", methods=["POST"])
def update_segment_events(segment_id: int):
    """
    Overwrites the saved events for a segment with a new array from the UI.
    """
    data = request.json
    events = data.get("events", [])
    
    # 1. Clear existing annotations
    db_service.clear_all_annotations(segment_id)
    
    # 2. Insert new ones
    success_count = 0
    import uuid
    for evt in events:
        # Format the event for DB
        event_dict = {
            "event_id": str(uuid.uuid4()),
            "event_type": evt.get("event_type", "Unknown"),
            "event_category": evt.get("event_category", "ECTOPY"),
            "start_time": float(evt.get("start_time", 0.0)),
            "end_time": float(evt.get("end_time", 0.0)),
            "beat_indices": [],
            "annotation_source": evt.get("annotation_source", "cardiologist"),
            "annotation_status": "confirmed",
            "used_for_training": True
        }
        if db_service.save_event_to_db(segment_id, event_dict):
            success_count += 1
            
    return jsonify({"status": "ok", "saved": success_count})

# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    # Run on 0.0.0.0 so you can view from other machines in LAN if needed
    app.run(host="0.0.0.0", port=5000, debug=True)
