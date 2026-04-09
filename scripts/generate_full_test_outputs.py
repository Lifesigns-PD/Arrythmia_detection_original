import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import psycopg2

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from xai.xai import explain_segment
from decision_engine.rhythm_orchestrator import RhythmOrchestrator
from decision_engine.models import DisplayState
from models_training.data_loader import CLASS_NAMES, normalize_label
from signal_processing.cleaning import clean_signal

def get_primary_label(decision):
    displayed_events = [e for e in decision.events if e.display_state == DisplayState.DISPLAYED]
    if displayed_events:
        displayed_events.sort(key=lambda e: e.priority, reverse=True)
        return displayed_events[0].event_type
    return decision.background_rhythm

def main():
    print("Initializing directories...")
    output_dir = BASE_DIR / "test_output_full"
    output_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("Querying PostgreSQL Database...")
    db_params = {
        "host": "127.0.0.1",
        "dbname": "ecg_analysis",
        "user": "ecg_user",
        "password": "sais",
        "port": "5432"
    }

    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    
    # We will exclude missing signals and limit if necessary. 
    # To process all ~25k it will take maybe 10-20 min.
    cur.execute("""
        SELECT segment_id, signal_data, arrhythmia_label, segment_fs, filename, features_json
        FROM ecg_features_annotatable
        WHERE signal_data IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 2500
    """)
    rows = cur.fetchall()
    print(f"Loaded {len(rows)} segments from DB.")

    orchestrator = RhythmOrchestrator()

    y_true = []
    y_pred = []
    metadata = []
    signals_stored = []
    output_results = {name: [] for name in CLASS_NAMES}

    class_counts_found = {name: 0 for name in CLASS_NAMES}
    
    print("Running Inference Pipeline...")
    for idx, (seg_id, signal_raw, label_txt, fs, filename, feats_json) in enumerate(tqdm(rows, desc="Evaluating Pipeline")):
        # Parse signal
        try:
            if isinstance(signal_raw, str):
                signal = np.array(json.loads(signal_raw), dtype=np.float32)
            else:
                signal = np.array(signal_raw, dtype=np.float32)
        except Exception:
            continue
            
        fs = int(fs) if fs else 125
        signal = clean_signal(signal, fs).astype(np.float32)
        
        # Parse features
        feat_dict = {}
        if feats_json:
            try:
                if isinstance(feats_json, str):
                    feat_dict = json.loads(feats_json)
                else:
                    feat_dict = feats_json
            except Exception:
                pass
                
        # Make sure "mean_hr" is at least present
        if "mean_hr_bpm" in feat_dict and "mean_hr" not in feat_dict:
            feat_dict["mean_hr"] = feat_dict["mean_hr_bpm"]

        # 1. AI Evidence
        ml_prediction = explain_segment(signal, feat_dict)
        
        # 2. Rules Engine
        sqi = {"is_acceptable": True}
        decision = orchestrator.decide(ml_prediction, feat_dict, sqi)
        
        # 3. Label Extraction
        pred_label_raw = get_primary_label(decision)
        pred_label_txt = normalize_label(pred_label_raw)
        true_label_txt = normalize_label(label_txt)
        
        y_true.append(true_label_txt)
        y_pred.append(pred_label_txt)
        
        meta = {
            "id": seg_id,
            "filename": filename,
            "true_label": true_label_txt,
            "predicted_label": pred_label_txt,
            "match": (true_label_txt == pred_label_txt)
        }
        metadata.append(meta)
        
        # Collect top 5 examples for plots dynamically
        if class_counts_found[pred_label_txt] < 5:
            class_counts_found[pred_label_txt] += 1
            output_results[pred_label_txt].append(meta)
            
            # Save plot
            plt.figure(figsize=(10, 3))
            plt.plot(signal)
            plt.title(f"Segment ID: {seg_id} - True: {true_label_txt} | Pred: {pred_label_txt}")
            plt.ylabel("Amplitude (mV)")
            plt.xlabel("Samples")
            plt.tight_layout()
            safe_name = pred_label_txt.replace("/", "_").replace(" ", "_").replace("+", "plus")
            plt.savefig(plots_dir / f"{safe_name}_example_{class_counts_found[pred_label_txt]}_seg_{seg_id}.png")
            plt.close()

    cur.close()
    conn.close()

    # Filter classes that actually exist in the result
    unique_classes_true = set(y_true)
    unique_classes_pred = set(y_pred)
    used_classes = sorted(list(unique_classes_true | unique_classes_pred))
    
    # Map them to indices for confusion matrix
    used_indices = {name: i for i, name in enumerate(used_classes)}
    
    y_true_idx = [used_indices[name] for name in y_true]
    y_pred_idx = [used_indices[name] for name in y_pred]

    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=range(len(used_classes)))
    
    plt.figure(figsize=(12, 10))
    cax = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(cax)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                plt.text(j, i, format(val, 'd'),
                         ha="center", va="center",
                         color="white" if val > thresh else "black", fontsize=8)
                         
    plt.xticks(np.arange(len(used_classes)), used_classes, rotation=60, ha="right")
    plt.yticks(np.arange(len(used_classes)), used_classes)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix (End-to-End Pipeline)')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()
    
    with open(output_dir / "confusion_matrix.txt", "w") as f:
        f.write(classification_report(y_true_idx, y_pred_idx, target_names=used_classes, labels=range(len(used_classes)), zero_division=0))
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    with open(output_dir / "used_segments_and_files.json", "w") as f:
        # filter out empty arrays
        filtered_results = {k: v for k, v in output_results.items() if len(v) > 0}
        json.dump(filtered_results, f, indent=4)
        
    print(f"Done! Outputs saved to {output_dir}")

if __name__ == "__main__":
    main()
