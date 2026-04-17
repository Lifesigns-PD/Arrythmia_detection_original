import os
import json
import numpy as np
import math
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
if not hasattr(np, 'math'):
    np.math = math
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "models_training"))

from signal_processing_v3 import process_ecg_v3
from models_v2 import CNNTransformerWithFeatures
from models import CNNTransformerClassifier
from data_loader import RHYTHM_CLASS_NAMES, ECTOPY_CLASS_NAMES

# Paths
EXTRACTS_DIR = PROJECT_ROOT / "ECG_Data_Extracts"
CHECKPOINTS_DIR = PROJECT_ROOT / "models_training" / "outputs" / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hybrid_reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    print("[Report Gen] Loading Models...")
    from signal_processing_v3.features.extraction import FEATURE_NAMES_V3

    # ── Rhythm model: prefer V3 (60-feat), fall back to V1 (signal-only) ──────
    rhythm_ckpt_v3 = CHECKPOINTS_DIR / "best_model_rhythm_v3.pth"
    rhythm_ckpt_v2 = CHECKPOINTS_DIR / "best_model_rhythm_v2.pth"
    rhythm_ckpt_v1 = CHECKPOINTS_DIR / "best_model_rhythm.pth"
    num_classes_rhythm = len(RHYTHM_CLASS_NAMES)

    if rhythm_ckpt_v3.exists():
        state = torch.load(rhythm_ckpt_v3, map_location=DEVICE, weights_only=False)
        num_feat_r = state.get("num_features", 60)
        rhythm_model = CNNTransformerWithFeatures(num_classes=num_classes_rhythm, num_features=num_feat_r).to(DEVICE)
        key = "model_state_dict" if "model_state_dict" in state else "model_state"
        rhythm_model.load_state_dict(state[key], strict=False)
        print(f"  - Rhythm V3 loaded from {rhythm_ckpt_v3.name} ({num_classes_rhythm} classes, {num_feat_r} features)")
        rhythm_uses_features = True
    elif rhythm_ckpt_v2.exists():
        state = torch.load(rhythm_ckpt_v2, map_location=DEVICE, weights_only=False)
        num_feat_r = state.get("num_features", 60)
        rhythm_model = CNNTransformerWithFeatures(num_classes=num_classes_rhythm, num_features=num_feat_r).to(DEVICE)
        key = "model_state_dict" if "model_state_dict" in state else "model_state"
        rhythm_model.load_state_dict(state[key], strict=False)
        print(f"  - Rhythm V2 loaded from {rhythm_ckpt_v2.name} ({num_classes_rhythm} classes, {num_feat_r} features)")
        rhythm_uses_features = True
    else:
        state = torch.load(rhythm_ckpt_v1, map_location=DEVICE, weights_only=False)
        rhythm_model = CNNTransformerClassifier(num_classes=num_classes_rhythm).to(DEVICE)
        rhythm_model.load_state_dict(state["model_state"], strict=False)
        print(f"  - Rhythm V1 loaded from {rhythm_ckpt_v1.name} ({num_classes_rhythm} classes, signal-only)")
        rhythm_uses_features = False
    rhythm_model.eval()

    # ── Ectopy model: prefer V3 (60-feat), fall back to V2 (15-feat) ──────────
    ectopy_ckpt_v3 = CHECKPOINTS_DIR / "best_model_ectopy_v3.pth"
    ectopy_ckpt_v2 = CHECKPOINTS_DIR / "best_model_ectopy_v2.pth"

    if ectopy_ckpt_v3.exists():
        state_e = torch.load(ectopy_ckpt_v3, map_location=DEVICE, weights_only=False)
        num_feat_e = state_e.get("num_features", 60)
        expected_features = state_e.get("feature_names", FEATURE_NAMES_V3)
        print(f"  - Ectopy V3 loaded from {ectopy_ckpt_v3.name} (3 classes, {num_feat_e} features)")
    else:
        state_e = torch.load(ectopy_ckpt_v2, map_location=DEVICE, weights_only=False)
        # Old V2 checkpoint may not have feature_names saved; default to first 15 V3 names
        expected_features = state_e.get("feature_names", FEATURE_NAMES_V3[:15])
        num_feat_e = len(expected_features)
        print(f"  - Ectopy V2 loaded from {ectopy_ckpt_v2.name} (3 classes, {num_feat_e} features)")

    ectopy_model = CNNTransformerWithFeatures(num_classes=3, num_features=num_feat_e).to(DEVICE)
    key_e = "model_state_dict" if "model_state_dict" in state_e else "model_state"
    ectopy_model.load_state_dict(state_e[key_e], strict=False)
    ectopy_model.eval()

    return rhythm_model, rhythm_uses_features, ectopy_model, expected_features

def load_signal_from_json(json_path):
    """Stitch packets from JSON extract."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    all_samples = []
    # Sort by packetNo if available, but usually they are already sorted
    for packet in data:
        if "value" in packet and len(packet["value"]) > 0:
            all_samples.extend(packet["value"][0])
            
    return np.array(all_samples, dtype=np.float32)

def generate_report_for_file(json_path, rhythm_model, rhythm_uses_features, ectopy_model, expected_features):
    admission_id = Path(json_path).stem
    pdf_path = OUTPUT_DIR / f"Report_{admission_id}.pdf"
    
    # if pdf_path.exists():
    #     print(f"[Report Gen] Skipping {admission_id} (PDF already exists)")
    #     return
    
    # Full classes (15)
    V1_RHYTHM_CLASSES = RHYTHM_CLASS_NAMES
    
    print(f"[Report Gen] Processing {admission_id}...")
    full_signal = load_signal_from_json(json_path)
    fs = 125 
    
    # Segment into 10s windows (1250 samples)
    window_len = 1250
    num_windows = len(full_signal) // window_len
    
    if num_windows == 0:
        print(f"  [Warning] Signal too short in {admission_id}")
        return

    with PdfPages(pdf_path) as pdf:
        for i in range(num_windows):
            start = i * window_len
            end = start + window_len
            chunk = full_signal[start:end]
            
            # Process with V3 pipeline
            result = process_ecg_v3(chunk, fs=fs, min_quality=0.1)
            cleaned = result["signal"]
            all_features = result["features"] # Dict
            r_peaks = result["r_peaks"]
            
            # Map V3 feature dict to the feature names the loaded checkpoint expects
            features_vec = np.array([float(all_features.get(name, 0.0) or 0.0) for name in expected_features], dtype=np.float32)
            feat_tensor = torch.from_numpy(features_vec).float().unsqueeze(0).to(DEVICE)
            
            # Inference - Rhythm (10s window; uses features if V2/V3 checkpoint)
            with torch.no_grad():
                sig_tensor = torch.from_numpy(cleaned).float().unsqueeze(0).to(DEVICE)
                rhythm_logits = rhythm_model(sig_tensor, feat_tensor) if rhythm_uses_features else rhythm_model(sig_tensor)
                rhythm_idx = torch.argmax(rhythm_logits, dim=1).item()
                rhythm_label = V1_RHYTHM_CLASSES[rhythm_idx]
                rhythm_conf = torch.softmax(rhythm_logits, dim=1)[0, rhythm_idx].item()

            # Inference - Ectopy V2 (Per-Beat) - Using 2s window (250 samples) as in xai.py
            beat_predictions = []
            print(f"    - Processing {len(r_peaks)} beats...", flush=True)
            for beat_i, peak_idx in enumerate(r_peaks):
                # Extract 2s window centered on peak
                half = 125 # 250 // 2
                s_idx = peak_idx - half
                e_idx = peak_idx + half
                
                # Extract with padding
                if s_idx < 0:
                    beat_win = np.pad(cleaned[0:max(0, e_idx)], (abs(s_idx), 0))
                elif e_idx > len(cleaned):
                    beat_win = np.pad(cleaned[s_idx:len(cleaned)], (0, e_idx - len(cleaned)))
                else:
                    beat_win = cleaned[s_idx:e_idx]
                
                beat_win = beat_win[:250] # ensure exact length (2s)
                
                with torch.no_grad():
                    bx = torch.from_numpy(beat_win).float().unsqueeze(0).to(DEVICE)
                    be_logits = ectopy_model(bx, feat_tensor)
                    be_probs = torch.softmax(be_logits, dim=1)[0]
                    be_idx = torch.argmax(be_probs).item()
                    
                    # Very low threshold (0.3) to force markers if there's ANY evidence
                    active_probs = be_probs[1:].cpu().numpy() # PVC, PAC
                    max_act = np.max(active_probs)
                    if be_idx != 0 and be_probs[be_idx].item() < 0.3:
                        be_idx = 0
                        
                    if be_idx != 0:
                        beat_predictions.append({
                            "peak_idx": peak_idx,
                            "label": ECTOPY_CLASS_NAMES[be_idx],
                            "conf": be_probs[be_idx].item()
                        })
                    
                    if beat_i % 20 == 0: # Print every 20 beats to avoid flood
                         print(f"      Beat {beat_i}: max(PVC,PAC)={max_act:.3f}, top={ECTOPY_CLASS_NAMES[torch.argmax(be_probs).item()]}", flush=True)

            # Check if any ectopy in the strip
            has_ectopy = any(b["label"] != "None" for b in beat_predictions)
            ectopy_summary = "None"
            if has_ectopy:
                # Pick the most frequent or highest confidence ectopy for the title
                counts = {}
                for b in beat_predictions:
                    if b["label"] != "None":
                        counts[b["label"]] = counts.get(b["label"], 0) + 1
                ectopy_summary = max(counts, key=counts.get)
                print(f"    - Window {i+1}: Ectopy Detected ({counts})", flush=True)

            # Plotting
            fig, ax = plt.subplots(figsize=(15, 6))
            time = np.arange(len(cleaned)) / fs
            ax.plot(time, cleaned, color='black', linewidth=0.8)
            
            # Mark R-peaks and Ectopy
            # Create a list of all peaks indices that have been processed
            all_peak_indices = [b["peak_idx"] for b in beat_predictions]
            
            # Draw standard red peaks for those with no ectopy prediction record 
            # (though here every peak has a prediction, we only kept ectopy ones in beat_predictions for plotting)
            for p_idx in r_peaks:
                p_time = p_idx / fs
                if p_idx not in all_peak_indices:
                     ax.scatter(p_time, cleaned[p_idx], color='red', s=20)
            
            # Now draw ectopy markers
            for b in beat_predictions:
                p_idx = b["peak_idx"]
                p_time = p_idx / fs
                color = 'darkorange' if b["label"] == 'PVC' else 'darkviolet'
                ax.scatter(p_time, cleaned[p_idx], color=color, s=50, edgecolors='black', zorder=10)
                ax.text(p_time, cleaned[p_idx] + 0.3, f"{b['label']}\n{b['conf']:.0%}", 
                        color=color, fontweight='bold', ha='center', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
            
            # Display Rhythm
            title = f"Window {i+1} | {admission_id}\n"
            title += f"V1 Rhythm: {rhythm_label} ({rhythm_conf:.1%})"
            if has_ectopy:
                title += f" | V2 Ectopy detected in strip"
            ax.set_title(title, fontweight='bold', fontsize=12)
            
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (mV)")
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Mark the whole rhythm highlight as requested
            start_s_total = i * 10
            end_s_total = (i + 1) * 10
            ax.text(0.02, 0.95, f"Rhythm active: {start_s_total}s - {end_s_total}s", 
                    transform=ax.transAxes, color='blue', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8))

            if has_ectopy:
                # We no longer highlight the whole strip per user request
                pass

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"  [Success] Report saved to {pdf_path.name}")

def main():
    r_model, r_uses_feats, e_model, expected_feats = load_models()
    
    json_files = sorted(list(EXTRACTS_DIR.glob("*.json")))
    
    # Clean previous reports if we want fresh ones with new marking
    # (Optional: user didn't ask but it's better for consistency)
    import shutil
    # shutil.rmtree(OUTPUT_DIR)
    # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for jf in json_files:
        # Skip exceptionally large files (e.g. > 20MB) to prevent massive reports
        if jf.stat().st_size > 20 * 1024 * 1024:
            print(f"[Report Gen] Skipping {jf.name} (File too large: {jf.stat().st_size / 1024 / 1024:.1f} MB)")
            continue
            
        try:
            generate_report_for_file(jf, r_model, r_uses_feats, e_model, expected_feats)
        except Exception as e:
            print(f"  [Error] Failed to process {jf.name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
