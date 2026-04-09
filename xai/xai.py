from pathlib import Path
import sys

# --- FIX IMPORTS FOR FOLDER RESTRUCTURE ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "models_training"))

import numpy as np
import torch
import torch.nn.functional as F

from models_training.models_v2 import CNNTransformerWithFeatures
from models_training.models import CNNTransformerClassifier  # V1 fallback only
from models_training.data_loader import CLASS_NAMES, RHYTHM_CLASS_NAMES, ECTOPY_CLASS_NAMES, extract_fixed_window, WINDOW_SEC
from signal_processing.feature_extraction import FEATURE_NAMES, NUM_FEATURES
from decision_engine.models import SegmentDecision, SegmentState, DisplayState



# ---------------------------------------------------------------------
# GLOBALS & CHECKPOINT
# ---------------------------------------------------------------------

# Make path relative to this file (xai.py)
# BASE_DIR is defined above as project root
# CKPT is in models_training/outputs/checkpoints
# Checkpoints for split models
RHYTHM_CKPT    = BASE_DIR / "models_training" / "outputs" / "checkpoints" / "best_model_rhythm_v2.pth"
ECTOPY_CKPT    = BASE_DIR / "models_training" / "outputs" / "checkpoints" / "best_model_ectopy_v2.pth"
RHYTHM_CKPT_V1 = BASE_DIR / "models_training" / "outputs" / "checkpoints" / "best_model_rhythm.pth"
ECTOPY_CKPT_V1 = BASE_DIR / "models_training" / "outputs" / "checkpoints" / "best_model_ectopy.pth"

_device = None
_model_rhythm = None
_model_ectopy = None

_last_cnn_featuremap = None
_last_attention = None
_is_model_untrained = False  # Track if we are using fallback random models


def _init_device():
    """Initialize device safely."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] XAI device initialized: {_device}")
    return _device


# ---------------------------------------------------------------------
# HOOKS for CNN feature maps & Transformer attention
# ---------------------------------------------------------------------

def _cnn_hook(module, inp, out):
    global _last_cnn_featuremap
    _last_cnn_featuremap = out.detach().cpu().numpy()


def _attention_hook(module, inp, out):
    global _last_attention
    # Handle case where out is a tuple (attn_output, attn_weights) or just attn_output
    if isinstance(out, tuple) and len(out) > 1:
        attn = out[1]
    else:
        attn = out

    if attn is not None:
        _last_attention = attn.detach().cpu().numpy()
    else:
        _last_attention = None


# ---------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------

# Global to track loaded model timestamps
_rhythm_mtime = 0
_ectopy_mtime = 0

def _load_model(task="rhythm"):
    """
    Lazily load either the Rhythm or Ectopy specialist model.
    """
    global _model_rhythm, _model_ectopy, _rhythm_mtime, _ectopy_mtime, _is_model_untrained

    ckpt    = RHYTHM_CKPT    if task == "rhythm" else ECTOPY_CKPT
    ckpt_v1 = RHYTHM_CKPT_V1 if task == "rhythm" else ECTOPY_CKPT_V1
    classes = RHYTHM_CLASS_NAMES if task == "rhythm" else ECTOPY_CLASS_NAMES

    # Check modification time
    current_mtime = ckpt.stat().st_mtime if ckpt.exists() else 0

    # Selection
    current_model = _model_rhythm if task == "rhythm" else _model_ectopy
    stored_mtime = _rhythm_mtime if task == "rhythm" else _ectopy_mtime

    # Reload if model is None OR file has changed
    if current_model is not None and current_mtime == stored_mtime:
        return current_model

    device = _init_device()

    # Try V2 checkpoint first, fall back to V1 if not available
    if ckpt.exists():
        model = CNNTransformerWithFeatures(num_classes=len(classes), num_features=NUM_FEATURES)
        try:
            state = torch.load(ckpt, map_location=device, weights_only=False)
            sd = state["model_state"] if "model_state" in state else state
            model.load_state_dict(sd)
            print(f"[OK] {task.upper()} V2 model loaded from {ckpt}")
        except Exception as e:
            print(f"[WARN] {task.upper()} V2 checkpoint mismatch: {e}. Using untrained V2.")
            _is_model_untrained = True
    elif ckpt_v1.exists():
        print(f"[INFO] {task.upper()} V2 checkpoint not found, falling back to V1: {ckpt_v1}")
        model = CNNTransformerClassifier(num_classes=len(classes))
        try:
            state = torch.load(ckpt_v1, map_location=device, weights_only=False)
            sd = state["model_state"] if "model_state" in state else state
            model.load_state_dict(sd)
            print(f"[OK] {task.upper()} V1 model loaded (fallback)")
        except Exception as e:
            print(f"[WARN] {task.upper()} V1 fallback mismatch: {e}. Using untrained.")
            _is_model_untrained = True
    else:
        print(f"[WARN] No checkpoint found for {task.upper()}. Using untrained V2.")
        model = CNNTransformerWithFeatures(num_classes=len(classes), num_features=NUM_FEATURES)
        _is_model_untrained = True
    
    model.to(device)
    model.eval()

    # Attach XAI hooks only to Rhythm model (it's the complex one)
    if task == "rhythm":
        _model_rhythm = model
        _rhythm_mtime = current_mtime
        try:
            if hasattr(model, "cnn"):
                model.cnn.register_forward_hook(_cnn_hook)
            if hasattr(model, "transformer_encoder"):
                model.transformer_encoder.layers[-1].self_attn.register_forward_hook(_attention_hook)
        except: pass
    else:
        _model_ectopy = model
        _ectopy_mtime = current_mtime

    return model

def reset_model():
    """Forces reload from updated checkpoints on next call."""
    global _model_rhythm, _model_ectopy
    _model_rhythm = None
    _model_ectopy = None
    return True

from decision_engine.models import SegmentDecision, Event, EventCategory, DisplayState


# ---------------------------------------------------------------------
# CLINICAL RULES (Option A explanation)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# CLINICAL RULES (Option A explanation)
# ---------------------------------------------------------------------

# DEPRECATED: Rules have been migrated to decision_engine/rules.py
# This function is kept as a stub or removed entirely in favor of the Orchestrator.
def _apply_clinical_rules(features: dict) -> tuple:
    """
    DEPRECATED. Logic moved to decision_engine/rules.py.
    """
    return (None, None, None)


def _clinical_explanation(label: str, features: dict, attention_context: str = "") -> str:
    """
    Returns a textual explanation that is contextual, detailed, and 'clever'.
    It synthesizes quantitative data (features) with clinical logic.
    """

    # Define default values to prevent NameError
    cond_str = "normal"
    rhythm_desc = "regular"

    if not label:
        return "Analysis: No specific arrhythmia detected. The signal appears to be within normal limits."

    text = label.lower()
    
    # -------------------------------------------------------------
    # 1. EXTRACT AND VALIDATE DATA
    # -------------------------------------------------------------
    hr_val = features.get("mean_hr")
    hr = float(hr_val) if hr_val is not None else 0.0
    
    pr_val = features.get("pr_interval")
    pr = float(pr_val) if pr_val is not None else 0.0
    
    rr_intervals = features.get("rr_intervals_ms", [])
    if isinstance(rr_intervals, list) and len(rr_intervals) > 0:
        cv = np.std(rr_intervals) / np.mean(rr_intervals)
        rmssd = float(features.get("RMSSD", 0))
    else:
        cv = 0.0
        rmssd = 0.0

    # Robust QRS mean
    qrs_mean = 0.0
    try:
        raw_qrs = features.get("qrs_durations_ms")
        if isinstance(raw_qrs, list):
            q_list = [x for x in raw_qrs if isinstance(x, (int, float))]
            if q_list:
                qrs_mean = float(sum(q_list) / len(q_list))
    except Exception:
        pass

    # -------------------------------------------------------------
    # 2. GENERATE INTELLIGENT CONTEXT
    # -------------------------------------------------------------
    
    # Rate descriptors
    if hr == 0:
        rate_desc = "unmeasurable"
        intro = f"**Clinical Context**: The heart rate is {rate_desc}. Conduction analysis shows {cond_str}."
    else:
        if hr < 40: rate_desc = "profoundly bradycardic"
        elif hr < 60: rate_desc = "bradycardic"
        elif hr < 100: rate_desc = "normal range"
        elif hr < 150: rate_desc = "tachycardic"
        else: rate_desc = "severely tachycardic"
        intro = f"**Clinical Context**: The rhythm is {rate_desc} ({hr:.0f} bpm) and {rhythm_desc}, with {cond_str}."
    
    # Helper function to add attention context
    def enhance(base_text):
        if attention_context:
            return f"{base_text}\n\n**Model Focus**: {attention_context}"
        return base_text

    # -------------------------------------------------------------
    # 3. ARRHYTHMIA-SPECIFIC NARRATIVES
    # -------------------------------------------------------------
    
    # Atrial Fibrillation
    if "fibrillation" in text and "atrial" in text:
        analysis = (f"**Analysis**: **Atrial Fibrillation** is characterized by:\n"
                   f"1. Chaotic irregularity (CV={cv:.2f}, RMSSD={rmssd:.0f}ms)\n"
                   f"2. Absent organized P-waves (replaced by f-waves)\n"
                   f"This combination confirms the diagnosis despite the {rate_desc} ventricular response.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # Atrial Flutter
    if "flutter" in text:
        analysis = (f"**Analysis**: **Atrial Flutter** exhibits characteristic 'sawtooth' F-waves at ~300 bpm "
                   f"with structured AV conduction (typically 2:1 or 4:1 block).")
        return enhance(f"{intro}\n\n{analysis}")
    
    # 3rd Degree AV Block
    if "3rd degree" in text or "complete" in text:
        analysis = (f"**Analysis**: **Complete (3rd Degree) AV Block** - CRITICAL finding.\n"
                   f"Complete AV dissociation is present. The ventricles beat independently at {hr:.0f} bpm "
                   f"(escape rhythm), unrelated to atrial activity. The regularity (CV={cv:.2f}) confirms "
                   f"the independent ventricular pacemaker.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # 2nd Degree AV Block Type 1 (Wenckebach)
    if "wenckebach" in text or "type 1" in text:
        analysis = (f"**Analysis**: **2nd Degree AV Block Type I (Wenckebach)**.\n"
                   f"Progressive PR prolongation culminates in a dropped QRS. This 'grouped beating' pattern "
                   f"indicates AV nodal fatigability rather than structural damage.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # 2nd Degree AV Block Type 2 (Mobitz II)
    if "mobitz" in text or "type 2" in text:
        analysis = (f"**Analysis**: **2nd Degree AV Block Type II (Mobitz II)** - HIGH RISK.\n"
                   f"Intermittent dropped beats occur WITHOUT prior PR prolongation. "
                   f"The wide QRS ({qrs_mean:.0f}ms) localizes this to the His-Purkinje system. "
                   f"Risk of progression to complete heart block.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # 1st Degree AV Block
    if "1st degree" in text:
        analysis = (f"**Analysis**: **1st Degree AV Block**.\n"
                   f"All atrial impulses conduct to ventricles, but with delay. PR interval is {pr:.0f}ms "
                   f"(normal <200ms). The rhythm remains {rhythm_desc}, making this typically benign.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # PSVT/SVT
    if "psvt" in text or ("svt" in text and "nsvt" not in text):
        analysis = (f"**Analysis**: **Paroxysmal Supraventricular Tachycardia**.\n"
                   f"Rapid ({hr:.0f} bpm), regular tachycardia with narrow QRS ({qrs_mean:.0f}ms) confirms "
                   f"supraventricular origin. Sudden onset suggests re-entrant mechanism (AVNRT or AVRT).")
        return enhance(f"{intro}\n\n{analysis}")
    
    # NSVT
    if "nsvt" in text:
        analysis = (f"**Analysis**: **Non-Sustained Ventricular Tachycardia** - SIGNIFICANT finding.\n"
                   f"A run of ≥3 consecutive wide-complex beats at tachycardic rate. "
                   f"Indicates ventricular irritability and warrants monitoring.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # Sinus Tachycardia
    if "sinus tachycardia" in text:
        analysis = (f"**Analysis**: **Sinus Tachycardia**.\n"
                   f"Physiological acceleration of the sinus node. P-waves are normal, PR intact. "
                   f"Typically a response to stress, exercise, fever, or hypovolemia rather than primary arrhythmia.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # Sinus Bradycardia
    if "sinus bradycardia" in text or "bradycardia" in text:
        analysis = (f"**Analysis**: **Sinus Bradycardia**.\n"
                   f"Slow but organized sinus rhythm. All conduction intervals normal. "
                   f"May be physiological (athletes, sleep) or pathological (medications, sick sinus).")
        return enhance(f"{intro}\n\n{analysis}")
    
    # PVC Bigeminy
    if "bigeminy" in text:
        analysis = (f"**Analysis**: **Ventricular Bigeminy**.\n"
                   f"Alternating pattern: Normal beat → PVC → Normal beat → PVC. "
                   f"Wide QRS complexes (>120ms) in every other beat create characteristic coupling.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # PVC Trigeminy
    if "trigeminy" in text:
        analysis = (f"**Analysis**: **Ventricular Trigeminy**.\n"
                   f"Pattern: Normal → Normal → PVC (repeating). "
                   f"Indicates stable ventricular ectopic focus with 3:1 coupling.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # PVCs (general)
    if "pvc" in text and "bigeminy" not in text and "trigeminy" not in text:
        analysis = (f"**Analysis**: **Premature Ventricular Contractions**.\n"
                   f"Ectopic beats with wide QRS ({qrs_mean:.0f}ms) arising from ventricular focus. "
                   f"Arrive early in the cardiac cycle, often followed by compensatory pause.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # PAC Trigeminy
    if "pac trigeminy" in text:
        analysis = (f"**Analysis**: **PAC Trigeminy**.\n"
                   f"Rhythm Pattern: Normal → Normal → PAC. "
                   f"Every third beat acts as a premature atrial stimulus. Common in high adrenergic states.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # PACs (general)
    if "pac" in text.lower():
        analysis = (f"**Analysis**: **Premature Atrial Contractions**.\n"
                   f"Early beats originating from atrial ectopic focus. QRS remains narrow ({qrs_mean:.0f}ms), "
                   f"but P-wave morphology may be abnormal.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # Pause
    if "pause" in text:
        analysis = (f"**Analysis**: **Significant Sinus Pause**.\n"
                   f"Prolonged interval (>2.0s) between beats detected. "
                   f"May indicate sinus arrest, exit block, or non-conducted PAC.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # Ventricular Fibrillation
    if "ventricular fibrillation" in text:
        analysis = (f"**CRITICAL**: **Ventricular Fibrillation**.\n"
                   f"Chaotic, disorganized electrical activity with no discernible QRS complexes. "
                   f"Cardiac output is absent - IMMEDIATE defibrillation required.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # Bundle Branch Blocks
    if "bundle branch" in text:
        analysis = (f"**Analysis**: **Bundle Branch Block**.\n"
                   f"Intraventricular conduction delay evident by wide QRS ({qrs_mean:.0f}ms). "
                   f"Rhythm is supraventricular, differentiating from ventricular ectopy.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # Sinus Rhythm (Normal)
    if "sinus rhythm" in text:
        base = f"**Analysis**: **Normal Sinus Rhythm**.\nPhysiological rhythm with normal intervals."
        if hr > 100:
            base += f"\n*Note: Rate is elevated ({hr:.0f} bpm) - consider sinus tachycardia.*"
        if hr < 50 and hr > 0:
            base += f"\n*Note: Rate is low ({hr:.0f} bpm) - consider sinus bradycardia.*"
        if pr > 200:
            base += f"\n*Note: PR prolonged ({pr:.0f}ms) - suggests 1st degree AV block.*"
        return enhance(f"{intro}\n\n{base}")
    
    # Generic fallback
    analysis = (f"**Analysis**: **{label}** detected.\n"
               f"Based on rhythm analysis (CV={cv:.2f}), rate ({hr:.0f} bpm), "
               f"and morphology (QRS {qrs_mean:.0f}ms).")
    return enhance(f"{intro}\n\n{analysis}")


# ---------------------------------------------------------------------
# SALIENCY (vanilla gradient)
# ---------------------------------------------------------------------

def _build_features_tensor(features: dict, device) -> torch.Tensor:
    """Build the canonical 13-dim features tensor from the segment's features dict."""
    vec = np.array(
        [float(features.get(name) or 0.0) for name in FEATURE_NAMES],
        dtype=np.float32,
    )
    return torch.from_numpy(vec[None, :]).to(device)  # shape (1, 13)


def _compute_saliency(model, x, target_idx: int, features_t=None):
    """
    Vanilla gradient saliency on the 1-D ECG trace.
    x: torch tensor of shape (1, 1, T) with requires_grad=True
    features_t: optional features tensor (1, F) for V2 model — no gradient needed
    """
    model.zero_grad()
    x = x.clone().detach().requires_grad_(True)

    if features_t is not None and isinstance(model, CNNTransformerWithFeatures):
        logits = model(x, features_t.detach())
    else:
        logits = model(x)
    score = logits[0, target_idx]
    score.backward()

    grad = x.grad.detach().cpu().numpy()[0, 0]
    sal = np.abs(grad)
    sal = sal / (sal.max() + 1e-6)
    return sal.tolist()


def _analyze_attention(model) -> str:
    """
    Analyzes the last captured transformer attention weights to find
    where the model was 'looking' in time.
    
    Attention shape is typically (Batch, T, T) or (Batch, Heads, T, T).
    We sum/mean over heads and query dim to get a 1D 'importance' over time.
    
    Returns: A string description, e.g. "Focus concentrated at 2.4s and 7.1s"
    """
    global _last_attention
    if _last_attention is None:
        return ""
        
    try:
        # _last_attention is numpy, shape usually (1, T, T) or (1, Heads, T, T)
        attn = _last_attention
        if attn.ndim == 4: # (B, H, T, T)
             attn = attn.mean(axis=1) # Average heads -> (B, T, T)
             
        # Take the 0th element in batch
        # attn is now (T, T) - self attention matrix
        # We want to know which 'keys' (source positions) were attended to most.
        # Summing over the query dimension (axis=0) gives total attention received by each token
        attn_1d = attn[0].sum(axis=0) # (T,)
        
        # Normalize
        if attn_1d.max() > 0:
            attn_1d = attn_1d / attn_1d.max()
        
        # Threshold: Find peaks > 0.8
        peaks = np.where(attn_1d > 0.7)[0]
        
        if len(peaks) == 0:
            # Try lower threshold
            peaks = np.where(attn_1d > 0.5)[0]
        
        # Map to seconds
        # Total T is ~312 for 10s. 1 index = 10/312 s
        T = len(attn_1d)
        secs_per_step = 10.0 / T
        
        peak_secs = [f"{p * secs_per_step:.1f}s" for p in peaks]
        
        # Cluster nearby peaks (simple logic)
        unique_zones = []
        if peak_secs:
            last_sec = -999
            for p_idx in peaks:
                sec = p_idx * secs_per_step
                if sec - last_sec > 0.8: # Distinct zone if > 0.8s apart
                    unique_zones.append(f"{sec:.1f}s")
                    last_sec = sec
        
        if not unique_zones:
            return "Diffuse attention across the segment."
            
        if len(unique_zones) > 4:
            return "Attention distributed across multiple points in the segment."
            
        return f"Model focused closely on events at " + ", ".join(unique_zones) + "."
        
    except Exception as e:
        print(f"Attention analysis failed: {e}")
        return ""


# ---------------------------------------------------------------------
# MAIN API FUNCTION FOR /api/xai/<segment_id>
# ---------------------------------------------------------------------

def explain_segment(signal_1d: np.ndarray, features: dict) -> dict:
    """
    ML Evidence Generator for Split-Model Architecture.
    Runs Rhythm and Ectopy specialists and returns evidence for the Orchestrator.
    """
    try:
        if len(signal_1d) < 100:
             return {"error": "Signal too short"}

        device = _init_device()
        model_rhythm = _load_model(task="rhythm")
        model_ectopy = _load_model(task="ectopy")

        arr = np.asarray(signal_1d, dtype=np.float32)

        # Target FS is 125Hz as per system standards
        fs = 125

        # Build features tensor once — reused for every inference call (V2 models)
        features_t = _build_features_tensor(features, device)

        # ── 1. Inference: Ectopy (Beat-by-Beat) — runs FIRST so we can
        #    identify non-ectopic beats for clean rhythm-model windowing.
        BEAT_ECTOPY_THRESHOLD = 0.7  # per-beat confidence floor — model over-predicts ectopy

        r_peaks = features.get("r_peaks", [])
        if not r_peaks:
            # Fallback: no R-peaks available — run ectopy on default window
            default_window = extract_fixed_window(arr, fs, 0.0, 10.0)
            x_default = torch.from_numpy(default_window[None, None, :]).to(device)
            with torch.no_grad():
                if isinstance(model_ectopy, CNNTransformerWithFeatures):
                    e_logits = model_ectopy(x_default, features_t)
                else:
                    e_logits = model_ectopy(x_default)
                e_probs = F.softmax(e_logits, dim=1)[0].cpu().numpy()
            e_idx = int(np.argmax(e_probs))
            if e_idx != 0 and e_probs[e_idx] < BEAT_ECTOPY_THRESHOLD:
                e_idx = 0
            e_label = ECTOPY_CLASS_NAMES[e_idx]
            beat_records = []
        else:
            # Multi-beat inference — keep per-beat records for bigeminy/trigeminy detection
            all_e_probs  = []
            beat_records = []
            for beat_idx, peak_idx in enumerate(r_peaks):
                # Center 2s window on this peak
                start_s = (peak_idx / fs) - 1.0
                end_s = (peak_idx / fs) + 1.0
                beat_window = extract_fixed_window(arr, fs, start_s, end_s)
                bx = torch.from_numpy(beat_window[None, None, :]).to(device)

                with torch.no_grad():
                    if isinstance(model_ectopy, CNNTransformerWithFeatures):
                        be_logits = model_ectopy(bx, features_t)
                    else:
                        be_logits = model_ectopy(bx)
                    be_probs = F.softmax(be_logits, dim=1)[0].cpu().numpy()
                all_e_probs.append(be_probs)

                b_idx_top = int(np.argmax(be_probs))
                # Per-beat confidence threshold: require strong evidence for ectopy
                if b_idx_top != 0 and be_probs[b_idx_top] < BEAT_ECTOPY_THRESHOLD:
                    b_idx_top = 0  # Default to "None" (normal beat)
                beat_records.append({
                    "beat_idx":    beat_idx,             # sequential position in r_peaks list
                    "peak_sample": int(peak_idx),        # sample index in the segment
                    "label":       ECTOPY_CLASS_NAMES[b_idx_top],
                    "conf":        float(be_probs[b_idx_top]),
                    "probs":       be_probs.tolist(),
                })

            # Aggregate: Take max probability across all beats for segment-level label
            all_e_probs = np.array(all_e_probs)  # (N_beats, N_classes)
            max_probs = np.max(all_e_probs, axis=0)

            ectopy_indices = [1, 2]  # PVC, PAC (Run removed — rules-only)
            if np.max(max_probs[ectopy_indices]) > 0.6:
                e_idx = int(np.argmax(max_probs))
            else:
                e_idx = 0  # Default to "None"

            e_probs = max_probs
            e_label = ECTOPY_CLASS_NAMES[e_idx]

        # ── 2. Inference: Rhythm — select a clean window excluding PVC/PAC beats
        #    PVC beats have wide QRS (high variance) which biases the variance-based
        #    window selection toward ectopic morphology, causing false BBB detections.
        has_ectopy_beats = any(rec["label"] != "None" for rec in beat_records) if beat_records else False
        normal_peaks = [
            r_peaks[rec["beat_idx"]]
            for rec in beat_records
            if rec["label"] == "None"
        ] if beat_records else []

        if has_ectopy_beats and normal_peaks:
            # Center rhythm window on the normal beat closest to segment center
            seg_center = len(arr) / 2
            best_peak = min(normal_peaks, key=lambda p: abs(p - seg_center))
            center_s = best_peak / fs
            rhythm_window = extract_fixed_window(arr, fs, center_s - 1.0, center_s + 1.0)
        else:
            # No ectopy found OR no normal beats — use default variance-based window
            rhythm_window = extract_fixed_window(arr, fs, 0.0, 10.0)

        x = torch.from_numpy(rhythm_window[None, None, :]).to(device)
        with torch.no_grad():
            if isinstance(model_rhythm, CNNTransformerWithFeatures):
                r_logits = model_rhythm(x, features_t)
            else:
                r_logits = model_rhythm(x)
            r_probs = F.softmax(r_logits, dim=1)[0].cpu().numpy()

        r_idx = int(np.argmax(r_probs))
        r_label = RHYTHM_CLASS_NAMES[r_idx]

        # 3. Evidence Gathering
        saliency = _compute_saliency(model_rhythm, x, r_idx, features_t=features_t)
        attention_desc = _analyze_attention(model_rhythm)
        
        return {
            "rhythm": {
                "label": r_label,
                "confidence": float(r_probs[r_idx]),
                "probs": r_probs.tolist(),
                "attention": attention_desc
            },
            "ectopy": {
                "label": e_label,
                "confidence": float(e_probs[e_idx]),
                "probs": e_probs.tolist(),
                "beat_events": beat_records,   # per-beat predictions for bigeminy/trigeminy detection
            },
            "saliency": saliency
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def generate_detailed_ledger(base_rhythm, events):
    """
    Translates the AI's findings and the Rules Engine logic into a detailed, 
    time-stamped clinical paragraph for the XAI Reasoning Ledger.
    """
    if not events:
        return f"**AI Ledger**: The primary CNN+Transformer model analyzed the morphology and classified the base rhythm as **{base_rhythm}**. No ectopic beats or secondary arrhythmias were detected in this 10-second window."

    # 1. Sort events chronologically to trace the AI's "thought process"
    events = sorted(events, key=lambda x: x.get('start_time', 0.0))
    
    # 2. Extract specific beat timings
    pvc_times = [f"{e['start_time']:.1f}s" for e in events if "PVC" in e.get('event_type', '') or "Ventricular" in e.get('event_type', '')]
    pac_times = [f"{e['start_time']:.1f}s" for e in events if "PAC" in e.get('event_type', '') or "Atrial" in e.get('event_type', '')]
    
    # 3. Build the narrative
    narrative = f"**AI Ledger**: The baseline rhythm was identified as **{base_rhythm}**. "
    
    if pvc_times:
        narrative += f"\n* **Ventricular Ectopy**: The model detected {len(pvc_times)} abnormal ventricular morphology beat(s) at exactly **{', '.join(pvc_times)}**. "
    if pac_times:
        narrative += f"\n* **Atrial Ectopy**: The model detected {len(pac_times)} premature atrial beat(s) at exactly **{', '.join(pac_times)}**. "

    # 4. Explain the Rules Engine Output
    if len(pvc_times) == 2:
        narrative += "\n* **Rule Triggered**: Because 2 PVCs were detected consecutively, the Clinical Rules Engine upgraded the secondary diagnosis to a **PVC Couplet**."
    elif len(pvc_times) >= 3:
        narrative += f"\n* **Rule Triggered**: A sequence of {len(pvc_times)} consecutive PVCs triggered the high-priority **Ventricular Run / NSVT** rule."
    elif len(pac_times) >= 3:
        narrative += f"\n* **Rule Triggered**: A sequence of {len(pac_times)} consecutive PACs triggered the **PSVT** rule."

    narrative += "\n* **Action**: Please verify the timings on the ECG trace above. Use the dropdown to confirm or override the final ground-truth label."

    return narrative


def explain_decision(decision: SegmentDecision) -> str:
    """
    PURE EXPLAINER:
    Synthesizes the SegmentDecision into a logical clinical narrative.
    Explains: WHERE, WHAT ML SAW, WHAT RULES FIRED, WHY arbitration happened.
    """
    
    parts = []
    
    # 1. OVERALL STATE
    if decision.segment_state == SegmentState.UNRELIABLE:
        return "**Signal Quality Alert**: Interpretation suspended due to excessive noise or artifacts."
    elif decision.segment_state == SegmentState.WARMUP:
        return "**System Warmup**: Stabilizing signal analysis..."

    # 2. BASIC RHYTHM & BACKGROUND
    parts.append(f"**Background**: Segment analyzed as **{decision.background_rhythm}**.")
    
    # 3. EVENT ANALYSIS
    displayed = decision.final_display_events
    hidden = [e for e in decision.events if e.display_state == DisplayState.HIDDEN]
    
    if not decision.events:
        parts.append("No transient arrhythmia events detected.")
    else:
        # A) Explain Displayed Events
        parts.append("\n**Detected Events**:")
        for e in displayed:
            beat_info = f" at beats {e.beat_indices}" if e.beat_indices else ""
            pattern_info = f" ({e.pattern_label})" if e.pattern_label else ""
            
            # Source attribution
            if e.rule_evidence:
                source = "Clinical Rule"
            elif getattr(e, "annotation_source", "") == "cardiologist":
                source = "Cardiologist"
            else:
                source = "ML Model"
            
            parts.append(f"- **{e.event_type}**{pattern_info}{beat_info}: Verified by {source}.")
            
            # Evidence detail
            if e.rule_evidence:
                 # Extract specific rule text if available
                 rule_name = e.rule_evidence.get("rule", "Rule Engine")
                 parts.append(f"  *Reasoning*: {rule_name} fired based on interval analysis.")
            elif e.ml_evidence:
                 conf = e.ml_evidence.get("confidence", 0)
                 if isinstance(conf, float):
                     parts.append(f"  *ML Confidence*: {conf:.1%}")

        # B) Explain Suppression (The "Why")
        if hidden:
            parts.append("\n**Arbitration Notes**:")
            for e in hidden:
                reason = e.suppressed_by or "clinical hierarchy"
                parts.append(f"- **{e.event_type}** was detected but **suppressed** due to: {reason}.")

    # 4. XAI NARRATIVE Synthesizer (Time-stamped Ledger)
    event_dicts = []
    for e in decision.events:
        if hasattr(e, 'to_dict'):
            event_dicts.append(e.to_dict())
        else:
            event_dicts.append({
                'start_time': getattr(e, 'start_time', 0.0),
                'event_type': getattr(e, 'event_type', '')
            })
    narrative = generate_detailed_ledger(decision.background_rhythm, event_dicts)
    
    parts.append("\n---\n" + narrative)
    
    return "\n".join(parts)


# ---------------------------------------------------------------------
# FULL RAW XAI FOR /api/xai_raw (optional)
# ---------------------------------------------------------------------

def predict_and_explain(signal_1d: np.ndarray) -> dict:
    """
    Extended XAI including CNN feature maps and transformer attention.
    You can wire this to /api/xai_raw if desired.

    Returns:
      {
        "pred_label": <str>,
        "probabilities": [...],
        "saliency": [...],
        "cnn_featuremap": [...],
        "transformer_attention": [...]
      }
    """
    global _last_cnn_featuremap, _last_attention

    try:
        _last_cnn_featuremap = None
        _last_attention = None

        device = _init_device()
        model = _load_model()

        arr = np.asarray(signal_1d, dtype=np.float32)
        x = torch.from_numpy(arr[None, None, :]).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = CLASS_NAMES[pred_idx]

        saliency = _compute_saliency(model, x, pred_idx)

        cnn_map = []
        if _last_cnn_featuremap is not None:
            cnn_map = _last_cnn_featuremap[0].tolist()

        attn = []
        if _last_attention is not None:
            attn = _last_attention[0].tolist()

        return {
            "pred_label": pred_label,
            "probabilities": probs.tolist(),
            "saliency": saliency,
            "cnn_featuremap": cnn_map,
            "transformer_attention": attn,
        }
    except Exception as e:
        print(f"❌ XAI predict_and_explain error: {e}")
        return {
            "error": str(e),
            "pred_label": "Unknown",
            "probabilities": [],
            "saliency": [],
        }
