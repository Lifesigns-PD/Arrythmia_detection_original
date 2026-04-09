import os
import glob
import json
import traceback
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "models_training"))

from ecg_processor import process, _segment, _detect_r_peaks, _extract_morphology, SAMPLING_RATE, WINDOW_SAMPLES
from xai.xai import explain_segment, _init_device
from decision_engine.rhythm_orchestrator import RhythmOrchestrator

def plot_segment_to_png(signal, decision, pred_label, admission_id, segment_idx, output_dir, ml_pred, morph, fs=125):
    os.makedirs(output_dir, exist_ok=True)
    t = np.arange(len(signal)) / fs
    plt.figure(figsize=(15, 7))
    plt.plot(t, signal, color='black', linewidth=0.8)
    
    # ML data extraction
    rhythm_ml = ml_pred.get("rhythm", {})
    ml_label = rhythm_ml.get("label", "Unknown")
    ml_conf = rhythm_ml.get("confidence", 0.0)
    ectopy_beats = ml_pred.get("ectopy", {}).get("beat_events", [])
    ect_str = ", ".join([f"{b['label']} ({b['conf']:.2f})" for b in ectopy_beats[:5]])
    if not ect_str: 
        # Fallback to segment label if beat_events is empty
        seg_ectopy = ml_pred.get("ectopy", {}).get("label", "None")
        seg_conf = ml_pred.get("ectopy", {}).get("confidence", 0.0)
        ect_str = f"{seg_ectopy} ({seg_conf:.2f})" if seg_ectopy != "None" else "None"

    title_str = (f"Admission: {admission_id} | Segment: {segment_idx}\n"
                 f"Primary Decision: {pred_label} | Background: {decision.background_rhythm}\n"
                 f"ML Prediction: {ml_label} ({ml_conf:.2f}) | HR: {morph.get('heart_rate_bpm', 0):.1f} bpm\n"
                 f"Ectopy Model: {ect_str}")
    
    plt.title(title_str, fontsize=11, loc='left', color='maroon')
    plt.xlabel("Time (s)")
    plt.ylabel("Potential (mV)")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Shading
    for e in decision.final_display_events:
        st, et = e.start_time, e.end_time
        if st is not None and et is not None:
             cat_name = getattr(e.event_category, 'name', str(e.event_category))
             color = 'red' if cat_name == 'ECTOPY' else 'darkorange'
             plt.axvspan(st, et, color=color, alpha=0.15)
             plt.text(st + 0.1, max(signal)*0.3 if max(signal)>0 else 0.1, f"{e.event_type}", color=color, fontsize=8, fontweight='bold', rotation=90)

    out_file = os.path.join(output_dir, f"diag_VtoMV_{admission_id}_{segment_idx}.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()

def extract_master_signal(data):
    master = []
    if not isinstance(data, list): return None
    for pkt in data:
        if not isinstance(pkt, dict): continue
        val = pkt.get("value")
        if isinstance(val, str): 
            try: val = json.loads(val)
            except: continue
        
        chunk = None
        if isinstance(val, list) and len(val) > 0:
            chunk = val[0] if isinstance(val[0], list) else val
        elif isinstance(val, dict):
            chunk = val.get("ecgData")
            
        if isinstance(chunk, list) and len(chunk) > 0:
            if isinstance(chunk[0], (int, float)):
                # CONVERSION: Multiply by 1000.0 (Assuming input is in Volts, as per 0.08 values)
                master.extend([float(x) * 1000.0 for x in chunk])
                
    return np.array(master, dtype=np.float32) if master and len(master) > 10 else None

def process_extracts():
    extracts_dir = os.path.join(str(BASE_DIR), "ECG_Data_Extracts")
    plot_dir = os.path.join(str(BASE_DIR), "testing_final", "VtoMV_diagnostic_strips")
    out_dir = os.path.join(str(BASE_DIR), "testing_final", "VtoMV_extracts_output")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    _init_device()
    orchestrator = RhythmOrchestrator()
    files = [f for f in glob.glob(os.path.join(extracts_dir, "*.json")) if "727425540" not in f]
    
    print(f"Analyzing {len(files)} files with Volts->mV conversion...")
    for fpath in files:
        adm_id = os.path.basename(fpath).replace(".json", "")
        print(f"   [File] {adm_id}")
        
        try:
            with open(fpath, 'r') as f: data = json.load(f)
            master_sig = extract_master_signal(data)
            if master_sig is None: continue
            
            # Save chunk payloads
            payloads = []
            chunk_size = 7500 # 1 minute
            for i in range(0, len(master_sig), chunk_size):
                chunk = master_sig[i:i+chunk_size]
                if len(chunk) < 1250: continue
                try:
                    res = process(chunk.tolist(), adm_id, "extract_VtoMV", 0)
                    # We don't delete ecgData here to keep the record full if needed
                    payloads.append(res)
                except Exception: pass

            # Plot strips
            all_windows = _segment(master_sig)
            plots_saved = 0
            for wi, win in enumerate(all_windows):
                if plots_saved >= 5: break
                rpeaks = _detect_r_peaks(win)
                if not rpeaks: continue
                
                morph = _extract_morphology(win, rpeaks)
                m_summary = morph.get("summary", {})
                m_summary["r_peaks"] = rpeaks # CRITICAL: Pass peaks for beat-by-beat ectopy
                
                pred = explain_segment(win, m_summary)
                dec = orchestrator.decide(pred, m_summary, {"is_acceptable": True})
                
                lbl = dec.background_rhythm
                best = sorted(dec.final_display_events, key=lambda x: x.priority, reverse=True)
                if best and best[0].priority >= 60: lbl = best[0].event_type
                
                plot_segment_to_png(win, dec, lbl, adm_id, f"seg_{wi}", plot_dir, pred, m_summary)
                plots_saved += 1
            
            if payloads:
                with open(os.path.join(out_dir, f"vtomv_processed_{adm_id}.json"), 'w') as f_out:
                    json.dump(payloads, f_out, indent=2)

        except Exception as e:
            print(f"Error on {adm_id}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    process_extracts()
