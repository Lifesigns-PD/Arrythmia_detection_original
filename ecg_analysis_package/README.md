# ECG Pipeline Analysis Package

Standalone tool — no database required.
Upload any ECG as a JSON file and get a full visual + mathematical comparison
of the Custom pipeline vs NeuroKit2.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Quick Start

**Step 1 — Generate sample data (first time only):**
```bash
python generate_sample.py
```

**Step 2 — Run the report:**
```bash
python ecg_pipeline_report.py sample_ecg.json
```

**Save plots as PNG instead of displaying:**
```bash
python ecg_pipeline_report.py sample_ecg.json --save
```

---

## JSON Format

**Single segment:**
```json
{
    "signal": [0.12, 0.15, 0.18, ...],
    "fs": 125,
    "label": "Sinus Rhythm"
}
```

**Multiple segments:**
```json
[
    {"signal": [...], "fs": 125, "label": "Sinus Rhythm"},
    {"signal": [...], "fs": 125, "label": "Atrial Fibrillation"},
    {"signal": [...], "fs": 125, "label": "Bundle Branch Block"}
]
```

`signal` — list of float values in mV
`fs`     — sampling rate in Hz (default 125)
`label`  — any string describing the rhythm

---

## What the Report Shows

| Panel | Description |
|---|---|
| ① Raw Signal | Original unprocessed ECG |
| ② Preprocessing | Custom (median + Butterworth) vs NK2 |
| ③ R-peak Detection | Custom Pan-Tompkins vs NK2, count compared |
| ④ Delineation | P/Q/R/S/T markers, QRS + P-wave regions shaded |
| ⑤ Feature Table | HR, PR, QRS, SDNN, RMSSD, p_absent vs clinical norms |

---

## Mathematical Formulas Shown

**Preprocessing:**
- Baseline: `x_clean = x - median_{0.6s}(x)`
- Bandpass: Butterworth 4th order [0.5–40 Hz]

**Pan-Tompkins R-peak detection:**
- Derivative: `y[n] = (-x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2]) / 8`
- Square: `y[n] = y[n]²`
- Moving average: `W = 0.15s`
- Threshold: `θ = 0.5 × max(MA)`

**QRS Offset (J-point):**
- Slope criterion: `|x[n+1] - x[n]| < 0.025 mV/sample`
- Level criterion: `x[n] < 0.15 × R_amplitude`
- Minimum wait: `40 ms after R`

**P-wave energy ratio:**
- `E = Σ(x - mean)² / N`
- Accept P if: `E_p / E_baseline ≥ 2.5` AND `peak_amp ≥ 0.04 mV`

**HRV features:**
- `HR = 60000 / mean(RR)`
- `SDNN = std(RR)` — overall variability
- `RMSSD = √mean(ΔRR²)` — short-term variability
- `pNN50 = count(|ΔRR| > 50ms) / N`

---

## Status Legend in Feature Table

| Status | Meaning |
|---|---|
| Both correct ✓ | Both pipelines within clinical normal range |
| Custom ✓ NK2 ✗ | Custom pipeline more accurate |
| Custom ✗ NK2 ✓ | NK2 more accurate |
| Values differ ⚠ | Both outside range or disagree by >15% |

---

## Refinement Guide

If you see `⚠` on a specific feature:

**QRS too wide (>120ms for SR):**
- Check `level_thresh` in `wavelet_delineation.py`
- Adjust `r_amp * 0.15` multiplier

**PR too short (<120ms):**
- P-wave onset being detected too late
- Increase P-search window backward from QRS onset

**p_absent too low for AF:**
- Energy ratio threshold too lenient
- Increase `2.5` multiplier in P-wave gate

**HR wrong:**
- R-peak count mismatch between pipelines
- Check refractory period (`0.2s` minimum distance)
