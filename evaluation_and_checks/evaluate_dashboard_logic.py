#!/usr/bin/env python3
"""
evaluate_dashboard_logic.py

Runs the end-to-end Dashboard/XAI pipeline (ML Models + Clinical Rules Engine)
on the database segments, exactly as the dashboard and decision engine do,
and calculates the final accuracy against the cardiologist truth labels.
"""

import sys
import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "models_training"))

from database import db_service
from xai.xai import explain_segment, _init_device
from decision_engine.rhythm_orchestrator import RhythmOrchestrator
from models_training.data_loader import normalize_label
import argparse
import matplotlib.pyplot as plt

def plot_segment_to_png(seg_dict, decision, pred_label, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    signal = seg_dict["signal"]
    fs = seg_dict["features"].get("fs", 125)
    r_peaks = seg_dict["features"].get("r_peaks", [])
    t = np.arange(len(signal)) / fs

    plt.figure(figsize=(15, 5))
    plt.plot(t, signal, color='black', linewidth=1.0)
    plt.title(f"Segment {seg_dict['id']} | Ground Truth: {seg_dict['truth_label']} | Pred: {pred_label}\nBackground Rhythm: {decision.background_rhythm}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    if len(r_peaks) > 0:
        peak_times = np.array(r_peaks) / fs
        peak_vals = signal[r_peaks] if max(r_peaks) < len(signal) else [0]*len(r_peaks)
        plt.scatter(peak_times, peak_vals, color='blue', marker='v', s=30, label='R-Peaks')

    for e in decision.final_display_events:
        st = e.start_time
        et = e.end_time
        if st is None or et is None: continue
        # Ignore full-segment events for shading to prevent entire background being colored, except if they explicitly denote an area.
        if st <= 0.1 and et >= 9.9 and e.event_type == decision.background_rhythm:
            continue
            
        color = 'red' if e.event_category.name == 'ECTOPY' else 'orange'
        plt.axvspan(st, et, color=color, alpha=0.3, label=e.event_type)
        plt.text(st, max(signal)*0.8, e.event_type, color=color, fontsize=9, rotation=90)

    # De-duplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    filename = os.path.join(output_dir, f"segment_{seg_dict['id']}_True_{seg_dict['truth_label'].replace(' ', '')}_Pred_{pred_label.replace(' ', '')}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate dashboard logic")
    parser.ArgumentParser = argparse.ArgumentParser
    parser.add_argument("--plot", action="store_true", help="Generate PNG plots of segments")
    parser.add_argument("--plot-dir", default="testing_final/plots", help="Directory to save plots")
    parser.add_argument("--plot-max", type=int, default=20, help="Maximum number of plots to generate overall")
    parser.add_argument("--export-json", action="store_true", help="Export the raw segments and features to JSON files")
    parser.add_argument("--json-dir", default="testing_final/inputs", help="Directory to save exported JSONs")
    args = parser.parse_args()

    print("="*80)
    print("EVALUATING FULL SYSTEM PIPELINE (DASHBOARD LOGIC)")
    print("="*80)

    _init_device()
    orchestrator = RhythmOrchestrator()

    # 1. Fetch segments directly using db_service
    # We want segments that have been corrected/annotated
    print("Fetching segments from the database...")
    conn = db_service._connect()
    segments = []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT segment_id, raw_signal, features_json, arrhythmia_label 
                FROM ecg_features_annotatable
                WHERE is_corrected = TRUE AND raw_signal IS NOT NULL
            """)
            rows = cur.fetchall()
            for r in rows:
                seg_id, sig_raw, features_raw, gt_label = r
                if not gt_label or gt_label == "Unlabeled":
                    continue
                
                # Parse
                sig = np.array(sig_raw if isinstance(sig_raw, list) else json.loads(sig_raw), dtype=np.float32)
                feats_raw = features_raw if isinstance(features_raw, dict) else (json.loads(features_raw) if features_raw else {})
                
                # Map to format expected by orchestrator/rules
                feats = {
                    "mean_hr": feats_raw.get("mean_hr_bpm", 0),
                    "pr_interval": feats_raw.get("pr_interval_ms", 0),
                    "rr_intervals_ms": feats_raw.get("rr_intervals_ms", []),
                    "qrs_durations_ms": [feats_raw.get("qrs_duration_ms")] if feats_raw.get("qrs_duration_ms") else [],
                    "p_wave_present_ratio": feats_raw.get("p_wave_present_ratio", 1.0),
                    "per_beat_pr_ms": feats_raw.get("per_beat_pr_ms", []),
                    "fs": feats_raw.get("fs", 125),
                    "r_peaks": feats_raw.get("r_peaks", []),
                    "RMSSD": feats_raw.get("rmssd_ms", 0)
                }
                
                segments.append({
                    "id": seg_id,
                    "signal": sig,
                    "features": feats,
                    "truth_label": gt_label,
                    "raw_features_payload": feats_raw # Store the raw json metadata for export
                })
    except Exception as e:
        print(f"Error fetching segments: {e}")
        return
    finally:
        conn.close()
        
    print(f"Loaded {len(segments)} annotated segments.")
    if not segments:
        print("No ground truth annotated segments found. Cannot evaluate.")
        return

    # 2. Run Pipeline
    y_true = []
    y_pred = []
    
    results = []
    plots_generated = 0
    jsons_generated = 0

    if args.export_json:
        os.makedirs(args.json_dir, exist_ok=True)
    
    print("Running through XAI Models + Decision Engine...")
    pbar = tqdm(segments)
    for seg in pbar:
        # Update progress bar description to show which segment we are processing
        pbar.set_description(f"Processing segment {seg['id']}")
        
        sig = seg["signal"]
        feats = seg["features"]
        gt_label_raw = seg["truth_label"]
        
        # 1. Get ML Evidence (the outputs of the CNN models, per-beat ectopy, etc.)
        ml_prediction = explain_segment(sig, feats)
        
        if "error" in ml_prediction:
            print(f"Error on segment {seg['id']}: {ml_prediction['error']}")
            continue
            
        # 2. Orchestrate Decision
        decision = orchestrator.decide(
            ml_prediction=ml_prediction,
            clinical_features=feats,
            sqi_result={"is_acceptable": True}, # assume clean for now
            segment_index=0
        )
        
        # 3. Determine final predicted label from the Display Events / Background
        predicted_primary = decision.background_rhythm
        display_events = sorted(decision.final_display_events, key=lambda x: x.priority, reverse=True)
        if display_events:
            top_event = display_events[0]
            if top_event.priority >= 60:
                predicted_primary = top_event.event_type
                
        norm_true = normalize_label(gt_label_raw)
        norm_pred = normalize_label(predicted_primary)

        # Output logic
        if args.plot and plots_generated < args.plot_max:
            plot_segment_to_png(seg, decision, norm_pred, args.plot_dir)
            plots_generated += 1

        if args.export_json and jsons_generated < args.plot_max:
            j_filename = os.path.join(args.json_dir, f"segment_{seg['id']}_inputs.json")
            payload = {
                "segment_id": seg["id"],
                "truth_label": gt_label_raw,
                "signal_length": len(sig),
                "features_json_raw": seg["raw_features_payload"],
                "mapped_clinical_features": feats,
                "ml_prediction_evidence": ml_prediction,
                "signal_array": sig.tolist()
            }
            with open(j_filename, 'w', encoding='utf-8') as jf:
                json.dump(payload, jf, indent=4)
            jsons_generated += 1


        
        y_true.append(norm_true)
        y_pred.append(norm_pred)
        
        results.append({
            "id": seg["id"],
            "true_raw": gt_label_raw,
            "true_norm": norm_true,
            "pred_raw": predicted_primary,
            "pred_norm": norm_pred,
            "bg_rhythm": decision.background_rhythm,
            "events": [e.event_type for e in display_events]
        })

    # 3. Report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("PIPELINE EVALUATION RESULTS")
    report_lines.append("="*80)
    
    # Global Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    acc = correct / total if total > 0 else 0
    report_lines.append(f"Overall Pipeline Accuracy: {acc:.4f} ({correct}/{total})\n")
    
    # Get unique classes present in either truth or predictions
    unique_classes = sorted(list(set(y_true) | set(y_pred)))
    
    report_lines.append(classification_report(y_true, y_pred, labels=unique_classes, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    report_lines.append("\nCONFUSION MATRIX:")
    header = "{:<25s}".format("") + "".join(f"{c[:8]:>9s}" for c in unique_classes)
    report_lines.append(header)
    for i, row_class in enumerate(unique_classes):
        row_str = "{:<25s}".format(row_class[:24])
        for j in range(len(unique_classes)):
            row_str += f"{cm[i, j]:>9d}"
        report_lines.append(row_str)
        
    report_lines.append("\n" + "="*80)
    
    final_report_str = "\n".join(report_lines)
    print(final_report_str)
    
    os.makedirs("testing_final", exist_ok=True)
    with open("testing_final/evaluation_report.txt", "w", encoding="utf-8") as rf:
        rf.write(final_report_str)
    print("Report saved to testing_final/evaluation_report.txt")


if __name__ == "__main__":
    main()
