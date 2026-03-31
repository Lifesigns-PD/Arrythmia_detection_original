"""Build grpc_output_example.json with 3 example types."""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# Read a real report for the "both" example
both_path = BASE / "outputs/patient_ecg/ADM640316196/ADM640316196_seg10_report.json"
if not both_path.exists():
    sys.exit(f"Run ingest first: {both_path}")

with open(both_path) as f:
    real_both = json.load(f)

ts = real_both["timestamp"]

# =====================================================
# EXAMPLE 1: RHYTHM ONLY — AF, no ectopy
# =====================================================
ex1 = {
    "_example_type": "RHYTHM ONLY - No ectopic beats, only rhythm-level arrhythmia",
    "arrhythmia_type": "Atrial Fibrillation",
    "confidence": 0.76,
    "message": "[ADM640316196] Atrial Fibrillation detected (HR: 92 bpm)",
    "patient_id": "ADM640316196",
    "timestamp": ts,
    "mean_hr": 92.4,
    "p_wave_duration_ms": 0.0,
    "p_wave_amplitude_mv": 0.0,
    "p_wave_present_ratio": 0.07,
    "pr_interval_ms": 0.0,
    "pr_segment_ms": 0.0,
    "qrs_duration_ms": 96.0,
    "qrs_amplitude_mv": 0.8412,
    "st_segment_ms": 112.0,
    "st_deviation_mv": 0.0185,
    "t_wave_duration_ms": 80.0,
    "t_wave_amplitude_mv": 0.1923,
    "qt_interval_ms": 340.0,
    "qtc_bazett_ms": 378.2,
    "rr_interval_ms": 652.0,
    "sdnn_ms": 89.3,
    "rmssd_ms": 102.1,
    "rr_intervals_ms": [680.0, 612.0, 704.0, 588.0, 640.0, "...13 values total"],
    "detection_window": {
        "description": "Rhythm-level arrhythmia - spans entire 10-second analysis window",
        "beat_events": {
            "event_flag": "True",
            "events": [
                {"beat_idx": 0, "peak_sample": 62, "start_pos": 25, "end_pos": 99, "label": "None", "confidence": 0.94},
                {"beat_idx": 1, "peak_sample": 147, "start_pos": 110, "end_pos": 184, "label": "None", "confidence": 0.91},
                {"beat_idx": 2, "peak_sample": 235, "start_pos": 198, "end_pos": 272, "label": "None", "confidence": 0.88},
                {"...": "14 beats total, all label=None (no ectopy)"},
            ],
        },
    },
    "xai_explanation": {
        "rhythm": {
            "label": "Atrial Fibrillation",
            "class_probabilities": {
                "Sinus Rhythm": 0.05,
                "Atrial Fibrillation": 0.76,
                "Atrial Flutter": 0.08,
                "Bundle Branch Block": 0.04,
                "1st Degree AV Block": 0.03,
                "Other Arrhythmia": 0.04,
            },
            "transformer_attention": "Model focus on irregularly irregular RR intervals and absent P-waves",
        },
        "ectopy": {
            "label": "None",
            "confidence": 0.94,
            "beat_events": [
                {"beat_idx": 0, "peak_sample": 62, "start_pos": 25, "end_pos": 99, "label": "None", "confidence": 0.94},
                {"...": "14 beats total, all None"},
            ],
        },
        "saliency": {
            "description": "Gradient saliency map - normalized 0.0 to 1.0 per sample.",
            "values": [0.03, 0.07, 0.42, 0.88, 0.91, 0.72, 0.28, 0.04, "...1250 values total"],
        },
        "clinical_reasoning": (
            "**AI Ledger**: The primary CNN+Transformer model analyzed the morphology and "
            "classified the base rhythm as **Atrial Fibrillation**. No ectopic beats or "
            "secondary arrhythmias were detected in this 10-second window."
        ),
    },
    "_detailed": {
        "segment_index": 10,
        "filename": "ADM640316196",
        "segment_offset_seconds": 100.0,
        "timestamp_start": "2026-03-12T10:39:08",
        "prediction": {
            "rhythm_label": "Atrial Fibrillation",
            "rhythm_confidence": 0.76,
            "ectopy_label": "None",
            "ectopy_confidence": 0.94,
        },
        "rules_engine": {
            "background_rhythm": "AF",
            "final_events": ["AF"],
            "primary_conclusion": "AF",
            "beat_markers": [],
        },
        "morphology": {
            "summary": {
                "heart_rate_bpm": 92.4,
                "qrs_duration_ms": 96.0,
                "pr_interval_ms": 0.0,
                "p_wave_present_ratio": 0.07,
                "num_beats": 14,
            },
            "per_beat": [
                {
                    "beat_index": 0, "r_peak_sample": 62,
                    "p_wave_duration_ms": None, "p_wave_present": False,
                    "p_wave_amplitude_mv": None, "pr_interval_ms": None,
                    "pr_segment_ms": None, "qrs_duration_ms": 96.0,
                    "qrs_amplitude_mv": 0.84, "st_segment_ms": 112.0,
                    "st_deviation_mv": 0.018, "t_wave_duration_ms": 80.0,
                    "t_wave_amplitude_mv": 0.19, "qt_interval_ms": 288.0,
                    "qtc_bazett_ms": 346.1, "rr_interval_ms": 680.0,
                    "heart_rate_bpm": 88.2,
                },
                {"...": "14 beats total"},
            ],
        },
    },
}

# =====================================================
# EXAMPLE 2: ECTOPY ONLY — PVC on sinus background
# =====================================================
ex2 = {
    "_example_type": "ECTOPY ONLY - Normal sinus rhythm with ventricular ectopy (PVC)",
    "arrhythmia_type": "PVC",
    "confidence": 0.81,
    "message": "[ADM640316196] PVC detected (HR: 78 bpm)",
    "patient_id": "ADM640316196",
    "timestamp": ts,
    "mean_hr": 78.3,
    "p_wave_duration_ms": 72.0,
    "p_wave_amplitude_mv": 0.119,
    "p_wave_present_ratio": 0.86,
    "pr_interval_ms": 164.0,
    "pr_segment_ms": 80.0,
    "qrs_duration_ms": 112.0,
    "qrs_amplitude_mv": 0.264,
    "st_segment_ms": 120.0,
    "st_deviation_mv": 0.0023,
    "t_wave_duration_ms": 64.0,
    "t_wave_amplitude_mv": 0.192,
    "qt_interval_ms": 296.0,
    "qtc_bazett_ms": 352.8,
    "rr_interval_ms": 768.0,
    "sdnn_ms": 42.5,
    "rmssd_ms": 38.7,
    "rr_intervals_ms": [768.0, 760.0, 772.0, 544.0, 992.0, 764.0, "...14 values total"],
    "detection_window": {
        "description": "Ectopy-level arrhythmia - localized to specific beats",
        "beat_events": {
            "event_flag": "True",
            "events": [
                {"beat_idx": 0, "peak_sample": 87, "start_pos": 50, "end_pos": 124, "label": "None", "confidence": 0.92},
                {"beat_idx": 1, "peak_sample": 183, "start_pos": 146, "end_pos": 220, "label": "None", "confidence": 0.89},
                {"beat_idx": 2, "peak_sample": 279, "start_pos": 242, "end_pos": 316, "label": "None", "confidence": 0.91},
                {"beat_idx": 3, "peak_sample": 347, "start_pos": 310, "end_pos": 384, "label": "PVC", "confidence": 0.81},
                {"beat_idx": 4, "peak_sample": 471, "start_pos": 434, "end_pos": 508, "label": "None", "confidence": 0.93},
                {"...": "14 beats total, 1 PVC at beat_idx=3"},
            ],
        },
    },
    "xai_explanation": {
        "rhythm": {
            "label": "Sinus Rhythm",
            "class_probabilities": {
                "Sinus Rhythm": 0.91,
                "Bundle Branch Block": 0.04,
                "Atrial Fibrillation": 0.02,
                "1st Degree AV Block": 0.02,
                "Other Arrhythmia": 0.01,
            },
            "transformer_attention": "Uniform attention across regular RR intervals, consistent with sinus rhythm",
        },
        "ectopy": {
            "label": "PVC",
            "confidence": 0.81,
            "beat_events": [
                {"beat_idx": 0, "peak_sample": 87, "start_pos": 50, "end_pos": 124, "label": "None", "confidence": 0.92},
                {"beat_idx": 3, "peak_sample": 347, "start_pos": 310, "end_pos": 384, "label": "PVC", "confidence": 0.81},
                {"...": "14 beats total"},
            ],
        },
        "saliency": {
            "description": "Gradient saliency map - normalized 0.0 to 1.0 per sample.",
            "values": [0.02, 0.04, 0.11, 0.08, 0.03, 0.85, 0.97, 0.91, "...1250 values total, peak at PVC location"],
        },
        "clinical_reasoning": (
            "**AI Ledger**: The baseline rhythm was identified as **Sinus Rhythm**. \n"
            "* **Ventricular Ectopy**: The model detected 1 abnormal ventricular morphology "
            "beat(s) at exactly **2.8s**. \n"
            "* **Action**: Please verify the timings on the ECG trace above."
        ),
    },
    "_detailed": {
        "segment_index": 0,
        "filename": "ADM640316196",
        "segment_offset_seconds": 0.0,
        "timestamp_start": "2026-03-12T10:38:48",
        "prediction": {
            "rhythm_label": "Sinus Rhythm",
            "rhythm_confidence": 0.91,
            "ectopy_label": "PVC",
            "ectopy_confidence": 0.81,
        },
        "rules_engine": {
            "background_rhythm": "Sinus Rhythm",
            "final_events": ["PVC"],
            "primary_conclusion": "PVC",
            "beat_markers": [{"peak_sample": 347, "label": "PVC", "conf": 0.81}],
        },
        "morphology": {
            "summary": {
                "heart_rate_bpm": 78.3,
                "qrs_duration_ms": 112.0,
                "pr_interval_ms": 164.0,
                "p_wave_present_ratio": 0.86,
                "num_beats": 14,
            },
            "per_beat": [
                {
                    "beat_index": 0, "r_peak_sample": 87,
                    "p_wave_duration_ms": 72.0, "p_wave_present": True,
                    "p_wave_amplitude_mv": 0.119, "pr_interval_ms": 164.0,
                    "pr_segment_ms": 80.0, "qrs_duration_ms": 88.0,
                    "qrs_amplitude_mv": 0.264, "st_segment_ms": 120.0,
                    "st_deviation_mv": 0.002, "t_wave_duration_ms": 64.0,
                    "t_wave_amplitude_mv": 0.192, "qt_interval_ms": 296.0,
                    "qtc_bazett_ms": 352.8, "rr_interval_ms": 768.0,
                    "heart_rate_bpm": 78.1,
                },
                {
                    "beat_index": 3, "r_peak_sample": 347,
                    "p_wave_duration_ms": None, "p_wave_present": False,
                    "p_wave_amplitude_mv": None, "pr_interval_ms": None,
                    "pr_segment_ms": None, "qrs_duration_ms": 152.0,
                    "qrs_amplitude_mv": 0.41, "st_segment_ms": 96.0,
                    "st_deviation_mv": -0.08, "t_wave_duration_ms": 88.0,
                    "t_wave_amplitude_mv": -0.21, "qt_interval_ms": 336.0,
                    "qtc_bazett_ms": 455.6, "rr_interval_ms": 544.0,
                    "heart_rate_bpm": 110.3,
                },
                {"...": "14 beats total"},
            ],
        },
    },
}

# =====================================================
# EXAMPLE 3: RHYTHM + ECTOPY — AF + PVC (from real data)
# =====================================================
ex3 = dict(real_both)
ex3["_example_type"] = "RHYTHM + ECTOPY - Both rhythm arrhythmia and ectopic beats detected"

# Trim arrays for readability
dw_events = ex3["detection_window"]["beat_events"]["events"]
n_beats = len(dw_events)
ex3["detection_window"]["beat_events"]["events"] = dw_events[:4] + [{"...": f"{n_beats} beats total"}]

ec_events = ex3["xai_explanation"]["ectopy"]["beat_events"]
ex3["xai_explanation"]["ectopy"]["beat_events"] = ec_events[:3] + [{"...": f"{n_beats} beats total"}]

sal = ex3["xai_explanation"]["saliency"]["values"]
ex3["xai_explanation"]["saliency"]["values"] = sal[:8] + ["...1250 values total"]

rri = ex3["rr_intervals_ms"]
ex3["rr_intervals_ms"] = rri[:5] + [f"...{len(rri)} values total"]

d = ex3["_detailed"]
fe = d["rules_engine"]["final_events"]
d["rules_engine"]["final_events"] = fe[:4] + [f"...{len(fe)} total"]

bm = d["rules_engine"]["beat_markers"]
d["rules_engine"]["beat_markers"] = bm[:2] + [{"...": f"{len(bm)} total"}]

pb = d["morphology"]["per_beat"]
num_b = d["morphology"]["summary"]["num_beats"]
d["morphology"]["per_beat"] = pb[:2] + [{"...": f"{num_b} beats total"}]

# =====================================================
# Combine
# =====================================================
output = {
    "input": {
        "device_id": "monitor_ward3_bed7",
        "patient_id": "ADM640316196",
        "sample_rate": 125,
        "timestamp": ts,
        "values": [
            0.12, 0.15, 0.18,
            "...continuous ECG voltage stream in mV (1250 samples = 10s at 125Hz)",
        ],
    },
    "examples": [
        ex1,
        ex2,
        ex3,
    ],
    "notes": {
        "normal_sinus": "No output is sent for normal sinus rhythm - silence means healthy",
        "arrhythmia_type_logic": (
            "If only rhythm detected: rhythm label (e.g. 'Atrial Fibrillation'). "
            "If only ectopy: ectopy label (e.g. 'PVC'). "
            "If both: 'rhythm + ectopy' (e.g. 'AF + PAC')."
        ),
        "confidence": "0.0 to 1.0 - values above 0.7 are high confidence",
        "timestamp": "Milliseconds Unix epoch - echoed from the client input",
        "start_pos_end_pos": (
            "Each beat event has start_pos and end_pos (sample indices within the "
            "1250-sample segment at 125Hz). Computed as peak_sample +/- 37 "
            "(0.3s x 125Hz), clamped to [0, 1250]."
        ),
        "detection_window": (
            "Rhythm arrhythmias span the full segment (0-1250). "
            "Ectopy patterns span only affected beats range."
        ),
        "morphology_features": (
            "Top-level features are AVERAGED across all beats in the segment. "
            "Per-beat values available in _detailed.morphology.per_beat."
        ),
        "xai_note": (
            "xai_explanation is the internal AI reasoning block. "
            "Full XAI detail also available via dashboard: GET /api/xai/<segment_id>"
        ),
    },
}

out_path = BASE / "grpc_output_example.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, default=str)

print(f"Written to {out_path}")
print(f"  Example 1: {ex1['_example_type']}")
print(f"  Example 2: {ex2['_example_type']}")
print(f"  Example 3: {ex3['_example_type']}")
