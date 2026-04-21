"""
ECG Arrhythmia Analysis — Comprehensive Interactive Dashboard
==============================================================
Tabs:
  1. Signal Processing (6 stages with biological context)
  2. R-Peak Detection (Pan-Tompkins + Hilbert + Wavelet ensemble)
  3. Delineation (P/Q/R/S/T with biological meaning + CWT/DWT)
  4. ML Inference (model architecture + inference detail)
  5. Decision Engine (sinus rules + ML fusion)
  6. Database & SQL
  7. System Architecture

Run: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Project imports
from signal_processing_v3 import process_ecg_v3
from signal_processing_v3.preprocessing.pipeline import preprocess_v3
from signal_processing_v3.detection.ensemble import detect_r_peaks_ensemble, refine_peaks_subsample
from signal_processing_v3.delineation.hybrid import delineate_v3
from signal_processing_v3.features.extraction import extract_features_v3, FEATURE_NAMES_V3
from decision_engine.sinus_detector import SinusDetector
from models_training.data_loader import RHYTHM_CLASS_NAMES, ECTOPY_CLASS_NAMES

st.set_page_config(page_title="ECG Analysis Dashboard", layout="wide", initial_sidebar_state="expanded")

# ─── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.bio-box { background:#0d1b2a; border-left:4px solid #1e90ff; padding:10px 14px; border-radius:4px; margin:6px 0; }
.algo-box { background:#1a2e1a; border-left:4px solid #32cd32; padding:10px 14px; border-radius:4px; margin:6px 0; }
.warn-box { background:#2e1a0d; border-left:4px solid #ff8c00; padding:10px 14px; border-radius:4px; margin:6px 0; }
</style>
""", unsafe_allow_html=True)

st.title("🫀 ECG Arrhythmia Analysis — Interactive Dashboard")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def safe_float(v, default=0.0):
    if v is None:
        return default
    if isinstance(v, np.ndarray):
        return float(v.item()) if v.size == 1 else float(np.mean(v))
    try:
        return float(v)
    except Exception:
        return default

def normalize_segment(seg):
    seg = dict(seg) if isinstance(seg, dict) else {}
    if "signal" not in seg and "ecgData" in seg:
        seg["signal"] = seg["ecgData"]
    if "signal" in seg:
        seg["signal"] = [float(x) for x in seg["signal"]]
    seg.setdefault("fs", 125)
    seg.setdefault("label", seg.get("patientId", seg.get("admissionId", "Unknown")))
    return seg

def normalize_packets(packets):
    if not isinstance(packets, list) or len(packets) < 10:
        return None
    if "packetNo" not in packets[0]:
        return None
    packets = sorted(packets, key=lambda p: p.get("packetNo", 0))
    full = []
    for pkt in packets:
        v = pkt.get("value", [])
        full.extend(v[0] if (isinstance(v, list) and v and isinstance(v[0], list)) else v)
    adm = packets[0].get("admissionId", "Unknown")
    return {"signal": full[:1250], "fs": 125, "label": f"Adm {adm}"}

ECG_WAVE_BIO = {
    "P": "**P Wave** — Atrial depolarization. Normal: 80–120 ms duration, <0.25 mV. Absent in AF. Inverted in junctional rhythm.",
    "PR": "**PR Interval** — AV conduction time (120–200 ms). Prolonged (>200 ms) = 1st degree AV block. Short (<120 ms) = WPW/junctional.",
    "QRS": "**QRS Complex** — Ventricular depolarization (60–120 ms). Wide (>120 ms) = BBB or PVC. Notched upstroke = delta wave (WPW).",
    "ST": "**ST Segment** — Early repolarization plateau. Elevation >0.1 mV = ischemia/STEMI. Depression = NSTEMI/strain.",
    "T": "**T Wave** — Ventricular repolarization. Inversion = ischemia, LVH, BBB. Peaked = hyperkalemia. Flattened = hypokalemia.",
    "QT": "**QTc Interval** — Total ventricular activation/recovery (Bazett corrected). Prolonged >450 ms (M) / >460 ms (F) = arrhythmia risk.",
}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Input ECG")
    use_sample = st.checkbox("Use demo ECG", value=True)
    if use_sample:
        sp = BASE_DIR / "ecg_analysis_package" / "sample_ecg.json"
        with open(sp) as f:
            uploaded_data = json.load(f)
        st.success("Loaded sample_ecg.json")
    else:
        uf = st.file_uploader("Upload ECG JSON", type=["json"])
        uploaded_data = json.load(uf) if uf else None

    if not uploaded_data:
        st.warning("No data loaded")
        st.stop()

    # Format detection
    if isinstance(uploaded_data, list) and len(uploaded_data) > 10 and "packetNo" in uploaded_data[0]:
        seg = normalize_packets(uploaded_data)
        st.info(f"Detected: {len(uploaded_data)} ECG packets")
    else:
        seg = normalize_segment(uploaded_data if isinstance(uploaded_data, dict) else uploaded_data[0])

    if seg is None:
        st.error("Unsupported format")
        st.stop()

    signal_arr = np.array(seg["signal"], dtype=np.float64)
    fs = seg["fs"]
    label = seg["label"]

    if len(signal_arr) < 100:
        st.error("Signal too short (<100 samples)")
        st.stop()

    st.metric("Samples", len(signal_arr))
    st.metric("Duration", f"{len(signal_arr)/fs:.1f}s")
    st.metric("Fs", f"{fs} Hz")
    st.caption(f"Label: {label}")

# ─────────────────────────────────────────────────────────────────────────────
# PROCESS PIPELINE (run once, cache)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_pipeline(signal_bytes, fs):
    signal = np.frombuffer(signal_bytes, dtype=np.float64).copy()
    prep = preprocess_v3(signal, fs)
    cleaned = prep.get("cleaned", signal)
    r_peaks = detect_r_peaks_ensemble(cleaned, fs)
    r_peaks_float = refine_peaks_subsample(cleaned, r_peaks)
    delineation = delineate_v3(cleaned, r_peaks, fs)
    features = extract_features_v3(cleaned, r_peaks_float, delineation, fs)
    return cleaned, r_peaks, r_peaks_float, delineation, features, prep

with st.spinner("Running signal processing pipeline..."):
    cleaned, r_peaks, r_peaks_float, delineation, features, prep = run_pipeline(signal_arr.tobytes(), fs)

per_beat = delineation.get("per_beat", [])
clean_features = {k: safe_float(v) for k, v in features.items()}

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔬 Signal Pipeline",
    "📍 R-Peak Detection",
    "🌊 Delineation",
    "🤖 ML Inference",
    "⚙️ Decision Engine",
    "🗄️ Database",
    "🏗️ Architecture",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIGNAL PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Signal Processing Pipeline — 6 Stages")
    t = np.arange(len(signal_arr)) / fs

    # Stage 1: Raw
    with st.expander("**Stage 1 — Raw ECG Signal**", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=signal_arr, mode='lines', name='Raw ECG',
                                     line=dict(color='#aaaaaa', width=1)))
            fig.update_layout(title="Raw ECG (unfiltered)", xaxis_title="Time (s)",
                               yaxis_title="Voltage (mV)", height=280, margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig, width='stretch')
        with col2:
            st.markdown("""
**File:** `ecg_processor.py`
**Function:** `_segment(ecg_data, 125)`

Splits 60s (7500 samples) into 6×10s windows (1250 samples each).
            """)

    # Stage 2: Preprocessing
    with st.expander("**Stage 2 — Adaptive Preprocessing (Baseline + Denoising)**"):
        col1, col2 = st.columns([3, 1])
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=signal_arr, mode='lines', name='Raw',
                                     opacity=0.4, line=dict(color='grey')))
            tc = np.arange(len(cleaned)) / fs
            fig.add_trace(go.Scatter(x=tc, y=cleaned, mode='lines', name='Cleaned',
                                     line=dict(color='#00c8ff', width=1.5)))
            fig.update_layout(title="Before vs After Preprocessing",
                               xaxis_title="Time (s)", yaxis_title="mV",
                               height=280, margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig, width='stretch')
        with col2:
            st.markdown("""
**File:** `signal_processing_v3/preprocessing/pipeline.py`
**Function:** `preprocess_v3()`

**4 stages inside:**
1. Baseline wander removal (Butterworth HP 0.15 Hz + SG + Morphological opening)
2. Powerline notch (auto-detects 50 or 60 Hz)
3. Lowpass Butterworth 45 Hz (order 4)
4. NaN/Inf cleanup
            """)
        qual = prep.get("quality_score", 0.0)
        issues = prep.get("quality_issues", [])
        st.progress(min(1.0, qual), text=f"Signal Quality: {qual:.0%}")
        if issues:
            for i in issues:
                st.caption(f"⚠ {i}")

    # Stage 3: R-peaks overview
    with st.expander("**Stage 3 — R-Peak Detection**"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tc, y=cleaned, mode='lines', name='ECG',
                                 line=dict(color='#00c8ff', width=1.2)))
        if len(r_peaks) > 0:
            rp_y = cleaned[np.clip(r_peaks.astype(int), 0, len(cleaned)-1)]
            fig.add_trace(go.Scatter(x=r_peaks/fs, y=rp_y, mode='markers', name='R-peaks',
                                     marker=dict(color='#ff4444', size=8, symbol='x')))
        fig.update_layout(title=f"R-Peaks Detected: {len(r_peaks)} beats ({safe_float(features.get('mean_hr_bpm')):.0f} bpm)",
                          xaxis_title="Time (s)", yaxis_title="mV", height=280,
                          margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig, width='stretch')
        st.caption("Details → **R-Peak Detection** tab")

    # Stage 4: SQI
    with st.expander("**Stage 4 — Signal Quality Index (SQI)**"):
        sqi = safe_float(features.get("sqi", prep.get("quality_score", 0.0)))
        col1, col2, col3 = st.columns(3)
        col1.metric("SQI Score", f"{sqi:.2f}")
        col2.metric("R-peaks", len(r_peaks))
        col3.metric("Status", "✅ Good" if sqi >= 0.3 else "❌ Poor")

    # Stage 5: Delineation overview
    with st.expander("**Stage 5 — Wave Delineation (P/Q/R/S/T)**"):
        st.caption(f"Method: {delineation.get('method', 'unknown')} | Beats delineated: {len(per_beat)}")
        st.caption("Full interactive delineation → **Delineation** tab")

    # Stage 6: Features
    with st.expander("**Stage 6 — Feature Extraction (60 features)**"):
        feat_data = []
        for k in FEATURE_NAMES_V3:
            v = features.get(k)
            domain = "HRV-Time" if k in FEATURE_NAMES_V3[:11] else \
                     "HRV-Freq" if k in FEATURE_NAMES_V3[11:19] else \
                     "Nonlinear" if k in FEATURE_NAMES_V3[19:27] else \
                     "Morphology" if k in FEATURE_NAMES_V3[27:40] else "Beat"
            feat_data.append({"Feature": k, "Domain": domain, "Value": round(v, 4) if v is not None else None})
        df = pd.DataFrame(feat_data)
        st.dataframe(df, width='stretch', height=350)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — R-PEAK DETECTION DETAIL
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("R-Peak Detection — Ensemble Algorithm")

    st.markdown("""
**File:** `signal_processing_v3/detection/ensemble.py`
**Function:** `detect_r_peaks_ensemble(signal, fs)`

The system uses **ensemble voting across 3 independent detectors** then confirms peaks
that ≥2 detectors agree on within an adaptive agreement window.
    """)

    st.subheader("1. Pan-Tompkins Algorithm (Classic ECG Standard)")
    st.markdown("""
<div class='algo-box'>

**Step 1 — Bandpass Filter (5–15 Hz):**
Isolates QRS frequency content. Removes P/T waves (low freq) and noise (high freq).

**Step 2 — Differentiation:**
```
y[n] = (1/8fs) × (-x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2])
```
Emphasizes the steep QRS slope (~200 mV/s) over slow P/T waves (~20 mV/s).

**Step 3 — Squaring:**
```
y[n] = x[n]²
```
Amplifies large QRS peaks, suppresses smaller baseline fluctuations. All values become positive.

**Step 4 — Moving Window Integration (150 ms):**
```
y[n] = (1/N) × Σ x[n-N+1..n]   where N = 150ms × fs
```
Smooths the squared signal, produces broad peak around each QRS.

**Step 5 — Adaptive Thresholding:**
- SPKI (signal peak): running estimate of QRS peak level
- NPKI (noise peak): running estimate of noise level
- Threshold = NPKI + 0.25 × (SPKI − NPKI)
- Threshold auto-adjusts after every beat
</div>
    """, unsafe_allow_html=True)

    st.subheader("2. Hilbert Transform Envelope Detector")
    st.markdown("""
<div class='algo-box'>

Extracts instantaneous amplitude envelope of the ECG.

```
H(x)[n] = x[n] convolved with 1/(π·n)  (Hilbert transform)
envelope[n] = √(x[n]² + H(x)[n]²)     (analytic signal magnitude)
```

Peaks in the envelope correspond to QRS complexes. Works well for:
- Inverted QRS complexes (negative R-waves)
- Wide QRS (BBB, PVC) where Pan-Tompkins may fire early
</div>
    """, unsafe_allow_html=True)

    st.subheader("3. Wavelet (CWT) Detector")
    st.markdown("""
<div class='algo-box'>

Uses **Continuous Wavelet Transform** with **Mexican Hat wavelet** (ψ):

```
ψ(t) = (1 - t²) × e^(-t²/2)        ← 2nd derivative of Gaussian
CWT(a, b) = ∫ x(t) × ψ((t-b)/a) dt  ← scale 'a', position 'b'
```

QRS detection scale: a = **5 samples = 40 ms** at 125 Hz.
At this scale the wavelet matches QRS width → maximum response at R-peak.

Zero-crossings of CWT correspond to onset/offset of waves.
</div>
    """, unsafe_allow_html=True)

    st.subheader("4. Ensemble Voting & Validation")
    st.markdown("""
<div class='algo-box'>

**Agreement window:** 50 ms base, expands to 80 ms if RR-interval CV > 0.15 (AF, PVCs)

```
For each reference peak P:
    supporters = [P]
    For each other detector D:
        if nearest_peak(D) is within agreement_window:
            supporters.add(nearest_peak(D))
    if len(supporters) >= 2:
        confirmed_peak = median(supporters)  ← consensus position
```

**RR Validation removes false peaks:**
- RR < 200 ms → impossible (>300 bpm) → remove
- RR > 3× median RR → likely missed beat / artifact → flag

**Sub-sample refinement** (3-point parabolic interpolation):
```
refined = p + 0.5 × (y[p-1] - y[p+1]) / (2×y[p] - y[p-1] - y[p+1])
```
Reduces 8 ms quantization jitter from 125 Hz sampling.
</div>
    """, unsafe_allow_html=True)

    # Visualise RR intervals
    if len(r_peaks) > 1:
        st.subheader("RR Interval Analysis")
        rr_ms = np.diff(r_peaks) / fs * 1000
        rr_ms = rr_ms[(rr_ms > 200) & (rr_ms < 3000)]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("RR Intervals Over Time", "RR Histogram"))
        fig.add_trace(go.Scatter(y=rr_ms, mode='lines+markers', name='RR', line=dict(color='#00c8ff')), row=1, col=1)
        fig.add_hline(y=float(np.mean(rr_ms)), line_dash="dash", line_color="yellow", annotation_text="Mean", row=1, col=1)
        fig.add_trace(go.Histogram(x=rr_ms, nbinsx=15, name='Distribution', marker_color='#00c8ff'), row=1, col=2)
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, width='stretch')

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean RR", f"{np.mean(rr_ms):.0f} ms")
        c2.metric("SDNN", f"{np.std(rr_ms):.1f} ms")
        c3.metric("HR", f"{60000/np.mean(rr_ms):.0f} bpm")
        c4.metric("RR CV", f"{np.std(rr_ms)/np.mean(rr_ms):.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DELINEATION
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Wave Delineation — P/Q/R/S/T Boundaries")

    for wave, bio in ECG_WAVE_BIO.items():
        st.markdown(f"<div class='bio-box'>{bio}</div>", unsafe_allow_html=True)

    st.divider()

    if per_beat:
        # Full signal with all wave annotations
        fig = go.Figure()
        tc = np.arange(len(cleaned)) / fs
        fig.add_trace(go.Scatter(x=tc, y=cleaned, mode='lines', name='ECG',
                                 line=dict(color='#cccccc', width=1)))

        # Color map for each wave
        COLORS = {
            'p': ('#ffaa00', 'triangle-up', 'P'), 'q': ('#aa00ff', 'triangle-down', 'Q'),
            'r': ('#ff4444', 'x', 'R'), 's': ('#00ff88', 'triangle-down', 'S'),
            't': ('#00aaff', 'triangle-up', 'T'),
        }

        for wave_key, (color, sym, name) in COLORS.items():
            xs, ys = [], []
            for beat in per_beat:
                peak_key = f"{wave_key}_peak"
                if beat.get(peak_key) is not None:
                    idx = int(beat[peak_key])
                    if 0 <= idx < len(cleaned):
                        xs.append(idx / fs)
                        ys.append(cleaned[idx])
            if xs:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', name=f"{name}-peak",
                                         marker=dict(color=color, size=9, symbol=sym)))

        # QRS shading on first 3 beats
        for beat in per_beat[:3]:
            qon = beat.get("qrs_onset")
            qoff = beat.get("qrs_offset")
            if qon is not None and qoff is not None:
                fig.add_vrect(x0=qon/fs, x1=qoff/fs, fillcolor="#ff4444", opacity=0.12,
                              annotation_text="QRS", annotation_position="top left")
            pon = beat.get("p_onset")
            poff = beat.get("p_offset")
            if pon is not None and poff is not None:
                fig.add_vrect(x0=pon/fs, x1=poff/fs, fillcolor="#ffaa00", opacity=0.12,
                              annotation_text="P", annotation_position="top left")
            ton = beat.get("t_onset")
            toff = beat.get("t_offset")
            if ton is not None and toff is not None:
                fig.add_vrect(x0=ton/fs, x1=toff/fs, fillcolor="#00aaff", opacity=0.10,
                              annotation_text="T", annotation_position="top left")

        fig.update_layout(title="Full ECG with P/Q/R/S/T Annotations",
                          xaxis_title="Time (s)", yaxis_title="mV",
                          height=380, legend=dict(orientation="h"),
                          margin=dict(l=40, r=20, t=50, b=40))
        st.plotly_chart(fig, width='stretch')

        # Zoomed-in on single beat
        st.subheader("Single Beat Zoom (Beat 1)")
        beat = per_beat[0]
        r = int(r_peaks[0]) if len(r_peaks) > 0 else 0
        win = int(0.5 * fs)
        s0, s1 = max(0, r - win), min(len(cleaned), r + win)
        beat_sig = cleaned[s0:s1]
        bt = np.arange(s0, s1) / fs

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=bt, y=beat_sig, mode='lines', name='ECG',
                                  line=dict(color='white', width=2)))
        labels = [('p_peak','P','#ffaa00','triangle-up'), ('q_peak','Q','#aa00ff','triangle-down'),
                  ('r_peak' if 'r_peak' in beat else None,'R','#ff4444','x'),
                  ('s_peak','S','#00ff88','triangle-down'), ('t_peak','T','#00aaff','triangle-up')]
        for key, name, color, sym in labels:
            if key and beat.get(key) is not None:
                idx = int(beat[key])
                if s0 <= idx < s1:
                    fig2.add_trace(go.Scatter(x=[idx/fs], y=[cleaned[idx]], mode='markers+text',
                                             text=[name], textposition="top center",
                                             name=name, marker=dict(color=color, size=12, symbol=sym)))

        # Label intervals
        qon = beat.get("qrs_onset"); qoff = beat.get("qrs_offset")
        if qon and qoff:
            fig2.add_vrect(x0=qon/fs, x1=qoff/fs, fillcolor="#ff4444", opacity=0.15)
            fig2.add_annotation(x=(qon+qoff)/(2*fs), y=max(beat_sig)*0.8,
                                text=f"QRS {(qoff-qon)/fs*1000:.0f}ms", showarrow=False,
                                font=dict(color="#ff8888"))

        fig2.update_layout(title="Single Beat Delineation", xaxis_title="Time (s)",
                           yaxis_title="mV", height=320,
                           margin=dict(l=40, r=20, t=50, b=40))
        st.plotly_chart(fig2, width='stretch')

    st.subheader("CWT Delineation Algorithm")
    st.markdown("""
<div class='algo-box'>

**File:** `signal_processing_v3/delineation/wavelet_delineation.py`
**Wavelet:** Mexican Hat (2nd derivative of Gaussian)

```
ψ(t) = (1 - t²) × e^(-t²/2)
```

**Scales used:**
| Wave   | Scale     | Why this scale |
|--------|-----------|---------------|
| QRS    | 5 samples (40 ms) | Matches QRS width, zero-crossings = onset/offset |
| P-wave | 2–5 samples      | Smaller scale → sharper features |
| T-wave | 7.5 samples (60 ms) | Wider scale → broad T-wave |

**Zero-crossing interpretation:**
- `neg → pos` crossing: wave onset (upslope begins)
- `pos → neg` crossing: wave offset (downslope)

**P-wave AF rejection:**
```
p_energy = Σ(P_window - mean)² / len
baseline_energy = Σ(TP_window - mean)² / len

if p_energy < 2.5 × baseline_energy → p = "absent"
if peak_amplitude < 0.04 mV → p = "absent"
```

**Template Matching refinement** (on top of wavelet):
- Takes first 8 beats, builds median template
- Cross-correlates each beat against template (±60 ms search)
- Final position = 60% template + 40% wavelet (blended)
- Corrects T-P overlap at HR >150 bpm (tachycardia)
</div>
    """, unsafe_allow_html=True)

    # Morphology table
    if per_beat:
        st.subheader("Per-Beat Morphology Table")
        rows = []
        for i, beat in enumerate(per_beat[:8]):
            qon = beat.get("qrs_onset"); qoff = beat.get("qrs_offset")
            qrs_dur = (qoff - qon)/fs*1000 if qon and qoff else None
            rows.append({
                "Beat": i+1,
                "QRS ms": round(qrs_dur, 1) if qrs_dur else "—",
                "P": beat.get("p_morphology", "—"),
                "T inv": "Yes" if beat.get("t_inverted") else "No",
                "Delta": "Yes" if beat.get("delta_wave") else "No",
                "Q depth mV": round(beat.get("q_depth", 0) or 0, 3),
                "S depth mV": round(beat.get("s_depth", 0) or 0, 3),
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ML INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("ML Inference — CNNTransformerWithFeatures")

    st.subheader("Model Architecture")
    st.markdown("""
<div class='algo-box'>

**File:** `models_training/models_v2.py`
**Class:** `CNNTransformerWithFeatures`

```
Input Signal: (B, 1250)                ← 10s @ 125Hz raw ECG
    ↓
SmallCNN (3 blocks, channels=32→64→128)
  Block 1: Conv1d(1, 32, k=7) → BN → ReLU → MaxPool1d(2)   → (B, 32, 625)
  Block 2: Conv1d(32, 64, k=7) → BN → ReLU → MaxPool1d(2)  → (B, 64, 312)
  Block 3: Conv1d(64, 128, k=7) → BN → ReLU → MaxPool1d(2) → (B, 128, 156)
    ↓
Conv1d(128, 128, kernel=1) [channel projection]              → (B, 128, 156)
    ↓
TransformerEncoder (2 layers, 8 attention heads, d_ff=256)
  Each layer: MultiHeadAttention(128, 8h) + FeedForward(256) + LayerNorm
              → captures long-range rhythm patterns across 156 time steps
    ↓
Global Average Pooling (mean over time dimension)             → z_sig (B, 128)

──────────────────────────────────────────────────────────────
Input Features: (B, 60)                ← 60 clinical features
    ↓
LayerNorm(60)   [different scales: HR~80 vs mV~0.1]
    ↓
Linear(60 → 64) → BatchNorm → ReLU → Dropout(0.1)
    ↓
Linear(64 → 64) → ReLU                                       → z_feat (B, 64)

──────────────────────────────────────────────────────────────
Fusion: concat([z_sig, z_feat])                               → (B, 192)
    ↓
LayerNorm(192) → Linear(192→64) → ReLU → Dropout(0.2) → Linear(64→9)
    ↓
Output logits (B, 9 classes)   ← Rhythm: or (B, 3) ← Ectopy: None/PVC/PAC
```
</div>
    """, unsafe_allow_html=True)

    st.subheader("Why Dual Pathway?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Signal Pathway (CNN + Transformer)**
- CNN detects local morphology: QRS shape, P-wave presence, ST changes
- Transformer captures global rhythm: irregularity, rate, inter-beat patterns
- Works even when clinical features fail (bad signal quality)
        """)
    with col2:
        st.markdown("""
**Feature Pathway (Dense Network)**
- Pre-computed clinical measurements: QRS duration, PR interval, HRV metrics
- Forces model to use expert-designed discriminators
- PVC score, PAC score, p_absent_fraction — purpose-built for arrhythmia
        """)

    st.subheader("Checkpoints & Training")
    d1, d2 = st.columns(2)
    d1.info("**Rhythm Model**\nFile: `best_model_rhythm_v2.pth`\nClasses: 9\nFeatures: 36\nBest bal_acc: 0.3534")
    d2.info("**Ectopy Model**\nFile: `best_model_ectopy_v2.pth`\nClasses: 3\nFeatures: 47\nBest bal_acc: 0.5868")

    st.subheader("Live ML Inference")
    st.info("Loads model checkpoints (~3 MB total). Click to run.")
    if st.button("🔄 Run ML Inference"):
        with st.spinner("Loading models and running inference..."):
            try:
                from xai.xai import explain_segment
                feat_copy = dict(features)
                feat_copy["r_peaks"] = r_peaks
                feat_copy["_signal"] = cleaned
                ml_result = explain_segment(cleaned, feat_copy)

                rhythm = ml_result.get("rhythm", {})
                probs = rhythm.get("probs", [])

                # Rhythm bar chart
                if probs:
                    while len(probs) < len(RHYTHM_CLASS_NAMES):
                        probs.append(0.0)
                    probs = probs[:len(RHYTHM_CLASS_NAMES)]
                    fig = go.Figure(go.Bar(x=RHYTHM_CLASS_NAMES, y=probs,
                                          marker=dict(color=probs, colorscale='Blues'),
                                          text=[f"{p:.0%}" for p in probs], textposition="outside"))
                    fig.update_layout(title=f"Rhythm: {rhythm.get('label')} ({rhythm.get('confidence',0):.1%})",
                                      xaxis_tickangle=-35, height=350)
                    st.plotly_chart(fig, width='stretch')

                # Ectopy beat table
                beats = ml_result.get("ectopy", {}).get("beat_events", [])
                if beats:
                    df_b = pd.DataFrame([{
                        "Beat": b.get("beat_idx"),
                        "Sample": b.get("peak_sample"),
                        "Label": b.get("label"),
                        "Confidence": f"{b.get('conf', 0):.1%}"
                    } for b in beats[:15]])
                    st.dataframe(df_b, width='stretch', hide_index=True)

            except Exception as e:
                st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DECISION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.header("Decision Engine — Rules + ML Fusion")

    st.markdown("""
**File:** `decision_engine/rhythm_orchestrator.py`
**Hierarchy:** Signal Processing Rules run FIRST (before ML). ML only handles non-sinus rhythms.
    """)

    with st.expander("**Sinus Rhythm Detector — 9 Criteria**", expanded=True):
        detector = SinusDetector()
        is_sinus, reason = detector.is_sinus_rhythm(clean_features)

        p_absent = safe_float(features.get("p_absent_fraction"), 1.0)
        qrs_width = safe_float(features.get("mean_qrs_duration_ms"))
        pr = safe_float(features.get("pr_interval_ms"))
        rr_cv = safe_float(features.get("rr_cv"), 0.5)
        qrs_wf = safe_float(features.get("qrs_wide_fraction"))
        pvc_s = safe_float(features.get("pvc_score_mean"))
        pac_s = safe_float(features.get("pac_score_mean"))
        lf_hf = safe_float(features.get("lf_hf_ratio"))
        hr = safe_float(features.get("mean_hr_bpm"))

        rows = [
            ("P-waves present", f"{p_absent:.2f}", "≤ 0.20", p_absent <= 0.20,
             "AF shows P_absent > 0.20. Normal sinus has visible P before every QRS."),
            ("QRS width normal", f"{qrs_width:.0f} ms", "< 120 ms", qrs_width < 120,
             "BBB or PVC widens QRS > 120 ms. Sinus rhythm has narrow complex."),
            ("PR interval normal", f"{pr:.0f} ms", "100–250 ms", 100 <= pr <= 250,
             "Short PR = WPW/junctional. Long PR > 200ms = 1st degree AV block."),
            ("RR regularity", f"{rr_cv:.3f}", "≤ 0.15", rr_cv <= 0.15,
             "AF has CV > 0.15 (irregular). Sinus is regular. PVCs cause brief irregularity."),
            ("No wide beats", f"{qrs_wf:.2f}", "≤ 0.10", qrs_wf <= 0.10,
             "< 10% wide beats in segment. Frequent PVCs would fail this."),
            ("Low PVC score", f"{pvc_s:.2f}", "≤ 2.0", pvc_s <= 2.0,
             "pvc_score_mean: composite of QRS width, compensatory pause, T-discordance."),
            ("Low PAC score", f"{pac_s:.2f}", "≤ 2.0", pac_s <= 2.0,
             "pac_score_mean: early P, short PR, narrow QRS pattern score."),
            ("LF/HF ratio", f"{lf_hf:.2f}", "≥ 0.5", lf_hf >= 0.5,
             "AF has chaotic spectral pattern, LF/HF collapses toward 0."),
            ("HR in range", f"{hr:.0f} bpm", "40–150 bpm", 40 <= hr <= 150,
             "Escape rhythms < 40 bpm. Above 150 = tachyarrhythmia territory."),
        ]

        for crit, val, thresh, passes, explanation in rows:
            col1, col2, col3, col4 = st.columns([2.5, 1.2, 1.2, 0.8])
            col1.write(crit)
            col2.write(val)
            col3.write(thresh)
            col4.write("✅" if passes else "❌")
            with st.expander(f"Why: {crit}", expanded=False):
                st.caption(explanation)

        if is_sinus:
            variant = detector.classify_sinus_variant(clean_features)
            st.success(f"✅ Detected: **{variant}** (confidence: 0.95)")
        else:
            st.info(f"Not Sinus — {reason}")
            st.info("→ ML model runs for abnormal rhythm classification")

    with st.expander("**Rules Engine — Pause / AF Safety Net / Atrial Flutter**"):
        r_peaks_arr = r_peaks
        rr = np.diff(r_peaks_arr) / fs * 1000 if len(r_peaks_arr) > 1 else np.array([])
        rr = rr[~np.isnan(rr)] if len(rr) > 0 else rr

        max_rr = float(max(rr)) if len(rr) > 0 else 0
        rr_std = float(np.std(rr)) if len(rr) > 0 else 0

        st.markdown(f"**Rule 1 — Pause Detection** (any RR > 2000 ms)\n"
                    f"Max RR = {max_rr:.0f} ms → {'🔴 FIRED' if max_rr > 2000 else '⚪ not fired'}")
        st.markdown(f"**Rule 2 — AF Safety Net** (RR_std > 160 ms AND p_absent < 0.4)\n"
                    f"RR_std = {rr_std:.0f} ms, P_absent = {p_absent:.2f} → {'🔴 FIRED' if rr_std > 160 and p_absent < 0.4 and not is_sinus else '⚪ not fired'}")
        st.markdown(f"**Rule 3 — Atrial Flutter** (HR 130–175 + FFT 4–6 Hz peak)\n"
                    f"HR = {hr:.0f} bpm → {'🟡 POSSIBLE' if 130 <= hr <= 175 else '⚪ not fired'}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.header("Database & SQL Reference")
    st.markdown("**Single table:** `ecg_features_annotatable` (PostgreSQL, host 127.0.0.1:5432, db: ecg_analysis)")

    schema = [
        ("signal_data", "REAL[]", "1250-sample float array — used for model training"),
        ("raw_signal", "JSONB", "ECG voltages for dashboard rendering"),
        ("features_json", "JSONB", "60-feature dict (HRV, morphology, beat discriminators)"),
        ("arrhythmia_label", "VARCHAR(50)", "Ground-truth label: Sinus Rhythm, AF, BBB, etc."),
        ("events_json", "JSONB", "Beat-level events: PVC/PAC annotations"),
        ("r_peaks_in_segment", "TEXT", "Comma-separated R-peak indices"),
        ("sqi_score", "FLOAT", "Signal Quality Index (0–1), min 0.5 for None-class pool"),
        ("is_corrected", "BOOLEAN", "Cardiologist verified = TRUE"),
        ("used_for_training", "BOOLEAN", "Included in ML training"),
        ("training_round", "INT", "How many training runs used this row"),
    ]
    st.dataframe(pd.DataFrame(schema, columns=["Column", "Type", "Purpose"]), width='stretch', hide_index=True)

    st.subheader("Key SQL Queries")
    st.code("""-- TRAINING (rhythm model)
SELECT segment_id, signal_data, arrhythmia_label, features_json
FROM ecg_features_annotatable
WHERE signal_data IS NOT NULL AND arrhythmia_label IS NOT NULL
ORDER BY segment_id;
-- File: models_training/retrain_v2.py → ECGEventDatasetV2.__init__()""", language="sql")

    st.code("""-- NONE-CLASS POOL (ectopy model — unannotated sinus beats)
SELECT segment_id, signal_data, features_json
FROM ecg_features_annotatable
WHERE (is_corrected = FALSE OR is_corrected IS NULL)
  AND arrhythmia_label IN ('Sinus Rhythm','Sinus Bradycardia','Sinus Tachycardia')
  AND (sqi_score IS NULL OR sqi_score >= 0.5)
ORDER BY RANDOM() LIMIT %s;
-- File: models_training/retrain_v2.py → ECGEventDatasetV2.__init__()""", language="sql")

    st.code("""-- DASHBOARD SEGMENT LOAD
SELECT raw_signal, features_json, arrhythmia_label, events_json,
       r_peaks_in_segment, cardiologist_notes, is_corrected
FROM ecg_features_annotatable WHERE segment_id = %s;
-- File: database/db_service.py → get_segment_new(segment_id)""", language="sql")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

with tab7:
    st.header("System Architecture — Full Call Chain")
    st.code("""
INPUT JSON (signal/ecgData/packet-array)
  │
  ├─ ecg_processor.py:process()
  │    └─ _segment() → 6 × 1250-sample windows
  │
  └─ Per window:
      ├─ signal_processing_v3/__init__.py:process_ecg_v3()
      │    ├─ preprocessing/pipeline.py:preprocess_v3()
      │    │    ├─ adaptive_baseline.py (HP + SG + Morphological opening)
      │    │    ├─ adaptive_denoising.py (notch 50/60 Hz + LP 45 Hz)
      │    │    └─ quality pre-check
      │    │
      │    ├─ detection/ensemble.py:detect_r_peaks_ensemble()
      │    │    ├─ Pan-Tompkins (bandpass → diff → square → integrate → threshold)
      │    │    ├─ Hilbert envelope detector
      │    │    ├─ CWT (Mexican Hat) detector
      │    │    ├─ Ensemble voting (≥2/3 agree within 50-80 ms window)
      │    │    └─ refine_peaks_subsample() (3-point parabolic interpolation)
      │    │
      │    ├─ quality/signal_quality.py:compute_sqi_v3() (SQI gate ≥ 0.3)
      │    │
      │    ├─ delineation/hybrid.py:delineate_v3()
      │    │    ├─ wavelet_delineation.py (CWT Mexican Hat per beat)
      │    │    │    ├─ QRS onset/offset (slope-flatness J-point)
      │    │    │    ├─ P-wave (energy gating + morphology classification)
      │    │    │    ├─ T-wave (CWT peak + zero-crossings)
      │    │    │    └─ Q, S waves (argmin in windows)
      │    │    └─ template_matching.py (8-beat median template + xcorr refinement)
      │    │
      │    └─ features/extraction.py:extract_features_v3()
      │         ├─ HRV time domain (11): mean_rr, sdnn, rmssd, pnn50 ...
      │         ├─ HRV frequency (8): vlf, lf, hf, lf_hf_ratio ...
      │         ├─ Nonlinear (8): sample_entropy, DFA, sd1, sd2 ...
      │         ├─ Morphology (13): qrs_dur, pr_interval, qtc, st_elevation ...
      │         └─ Beat discriminators (20): pvc_score, pac_score, wide_frac ...
      │
      ├─ xai/xai.py:explain_segment()
      │    ├─ _load_model("ectopy") → best_model_ectopy_v2.pth (3 classes)
      │    ├─ _load_model("rhythm") → best_model_rhythm_v2.pth (9 classes)
      │    ├─ StandardScaler.transform() (feature_scaler_*.joblib)
      │    ├─ ECTOPY: 2s windows per R-peak → per-beat None/PVC/PAC (gate: 0.97)
      │    └─ RHYTHM: full 10s window → 9-class probs
      │
      └─ decision_engine/rhythm_orchestrator.py:RhythmOrchestrator.decide()
           ├─ sinus_detector.py:detect_sinus_and_rhythm() [9 criteria, BEFORE ML]
           ├─ ML veto: if ML ≥ 0.88 on dangerous rhythm (AF/VF/3°AVB) → override
           ├─ rules.py:derive_rule_events() [Pause / AF / AFL]
           ├─ rules.py:apply_ectopy_patterns() [Couplet/Bigeminy/NSVT/VT]
           └─ returns SegmentDecision(background_rhythm, final_display_events)

OUTPUT: MongoDB document with analysis, events, HR, rhythm label
    """, language="text")

    st.info("""
**Key Design Decisions:**
- **Hierarchical detection**: Sinus identified by signal-processing rules first (cheap + explainable)
- **ML skipped for sinus**: Only AF, blocks, BBB, artifact go through ML (reduces false positives)
- **Dual-pathway model**: CNN handles raw waveform, Dense handles engineered features
- **Task-specific features**: Rhythm=36 features, Ectopy=47 features (noise reduction)
- **FocalLoss γ=3**: Handles severe class imbalance (20K sinus vs 135 type-2 AV block)
- **WeightedRandomSampler 5×**: Cardiologist-verified data oversampled during training
    """)
