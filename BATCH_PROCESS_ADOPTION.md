# BATCH_PROCESS Adoption Plan — What Gets Ported, Replaced, or Removed

This document lists every change being made to the V3 pipeline, explaining what is being adopted from
`BATCH_PROCESS/lifesigns_engine.py`, what is being changed in the existing architecture, and exactly
which files and lines are affected. Nothing is deferred as future work except the beat classifier,
which depends on the lethal detector being stable first.

---

## Summary Table

| Change | Type | Priority | Files |
|---|---|---|---|
| VTach/VFib spectral detection | **Port from BATCH_PROCESS** | P0 — Critical | CREATE `decision_engine/lethal_detector.py` |
| Lethal detector wired into pipeline | **New wiring** | P0 — Critical | MODIFY `rhythm_orchestrator.py` |
| Preprocessed signal passed to lethal detector | **New field** | P0 — Critical | MODIFY `ecg_processor.py` |
| BBB renamed to IVCD | **Label fix** | P1 | MODIFY `data_loader.py` + `rhythm_orchestrator.py` |
| ML veto at 0.88 conf removed | **Architecture fix** | P1 | MODIFY `rhythm_orchestrator.py` |
| Idioventricular Rhythm removed | **Architecture fix** | P1 | MODIFY `rhythm_orchestrator.py` |
| SVT detection via signal processing | **Port from BATCH_PROCESS** | P1 | MODIFY `decision_engine/lethal_detector.py` |
| PVC/PAC template correlation refinement | **Port from BATCH_PROCESS** | P2 | CREATE `decision_engine/beat_classifier.py` |

---

## P0: VTach / VFib Detection — `decision_engine/lethal_detector.py` (CREATE)

### Why
- VFib has ZERO detection in the current system
- VTach is only caught if the ectopy ML model labels ≥11 consecutive beats as PVC, then rules count them
- The rhythm ML model has 3 VT training samples and 0 VFib samples — it literally cannot learn these
- BATCH_PROCESS has 4 signal-processing algorithms that detect both conditions deterministically

### What Is Being Ported

#### Algorithm 1 — Spectral Lethal Pre-check (from `spectral_lethal_precheck()` lines 260–343)

**Two-stage detection on the preprocessed (bandpass-filtered) signal:**

**Stage 1 — Organization gate** (false-positive prevention using V3 r_peaks):
- Bigeminy fingerprint: even/odd RR alternation >12% AND each group CV <20% → SURVIVABLE
- Very regular + slow: rr_cv <0.20 AND rate <130 BPM AND ≥6 beats → SURVIVABLE
- Flutter guard (V3 addition): rr_cv <0.08 → SURVIVABLE (regular narrow rhythms like flutter/SVT)
- All other patterns fall through to spectral stage

**Stage 2 — Welch PSD spectral power ratio:**
```
SPI = power(1.5–7 Hz) / total power(0.5–40 Hz)
concentration = peak_bin_power / band_lethal_power

SPI > 0.75 → lethal zone:
  concentration > 0.35 → VENTRICULAR TACHYCARDIA (organized single-frequency power)
  concentration ≤ 0.35 → VENTRICULAR FIBRILLATION (diffuse chaotic power)
SPI ≤ 0.75 → SURVIVABLE
```
Returns: `(label: str | None, confidence: float)`
Confidence: 0.92 (spectral is the most reliable method)

#### Algorithm 2 — Kinetic VTach (from `calculate_metrics()` lines 951–972)

Wide-complex + AV dissociation criteria using V3 features:
```
HR > 100 bpm
AND QRS > 120 ms (relaxed to >100 ms when HR > 150)
AND wide_qrs_fraction > 0.75 (relaxed to >0.60 when HR > 150)
AND p_absent_frac > 0.70
```
Note: `pr_int_sd` not in V3 features — use P-absent arm only (>0.70). This is the stronger criterion.
Returns: `bool`
Confidence: 0.85

#### Algorithm 3 — Polymorphic VTach / Torsades (from `calculate_metrics()` lines 905–915)

```
HR > 150 bpm
AND 0.08 < rr_cv < 0.55 (not SVT-regular, not VFib-chaotic)
AND p_absent_frac > 0.60
```
Returns: `bool`
Confidence: 0.82

#### Algorithm 4 — Coarse VFib (from `calculate_metrics()` lines 896–903)

```
rr_cv > 0.50 (completely chaotic)
AND HR > 80 bpm
AND p_absent_frac > 0.60
```
Returns: `bool`
Confidence: 0.80

### Public API

```python
def detect_lethal_rhythm(
    signal: np.ndarray,    # preprocessed (cleaned) 10s window
    r_peaks: np.ndarray,   # from V3 ensemble detector
    features: dict,        # V3 60-feature dict from extract_features_v3()
    fs: int = 125,
) -> tuple[str | None, float, str]:
    """
    Returns (label, confidence, reason).
    label = "Ventricular Tachycardia" | "Ventricular Fibrillation" | None
    None = no lethal rhythm found, proceed to ML.
    """
```

### Feature Mapping (lifesigns_engine variable → V3 features key)

| lifesigns_engine | V3 features key | Notes |
|---|---|---|
| `metrics["hr"]` | `features["mean_hr_bpm"]` | Direct |
| `metrics["hrv_cv"]` | `features["rr_cv"]` | Direct |
| `metrics["qrs_dur"]` | `features["qrs_duration_ms"]` | Direct |
| `qrs_wide_fraction` | `features["wide_qrs_fraction"]` | Direct |
| `p_absent_frac` | `1.0 - features["p_wave_present_ratio"]` | Invert |
| `metrics["pr_int_sd"]` | *(not in V3)* | Skip — use P-absent arm only |

---

## P0: Wire Lethal Detector — `decision_engine/rhythm_orchestrator.py` (MODIFY)

### Insertion point

In `decide()`, after Step 3 sinus detection (line ~98), before Step 4A rule events (line ~101).

The variables `_signal`, `_r_peaks`, `_fs` are already extracted at lines 102–104 — move that extraction
block to BEFORE the lethal detector call.

```python
# Step 3.5 — Lethal rhythm pre-check (signal processing, no ML)
if decision.background_rhythm == "Unknown":
    _signal  = clinical_features.get("_signal_clean") or clinical_features.get("_signal")
    _r_peaks = clinical_features.get("r_peaks")
    _fs      = int(clinical_features.get("fs", 125))
    if _signal is not None and _r_peaks is not None:
        from decision_engine.lethal_detector import detect_lethal_rhythm
        _lethal_label, _lethal_conf, _lethal_reason = detect_lethal_rhythm(
            signal=np.asarray(_signal, dtype=np.float32),
            r_peaks=np.asarray(_r_peaks, dtype=int),
            features=clinical_features,
            fs=_fs,
        )
        if _lethal_label:
            decision.background_rhythm = _lethal_label
            print(f"[Lethal Detection] {_lethal_label} (conf={_lethal_conf:.2f}) — {_lethal_reason}")
```

The existing Step 4B ML block is already gated on `decision.background_rhythm == "Unknown"` — so if
lethal detection fires, ML rhythm is automatically skipped. No other changes needed in the orchestrator
for this.

---

## P0: Pass Preprocessed Signal — `ecg_processor.py` (MODIFY)

### What

In `_run_orchestrator()`, add `_signal_clean` to `clinical_features` so the lethal detector gets the
bandpass-filtered signal (not raw). The BATCH_PROCESS spectral check explicitly requires a filtered
signal to prevent EMG noise corrupting the 1.5–7 Hz band.

### Where

After line `v3 = process_ecg_v3(window, fs=SAMPLING_RATE, min_quality=0.2)` in `_run_orchestrator()`:

```python
# existing: clinical_features["_signal"] = window.tolist()
# add:
clinical_features["_signal_clean"] = v3.get("cleaned", window).tolist()
```

---

## P1: SVT Detection via Signal Processing (MODIFY `lethal_detector.py`)

### Why

SVT has only 27 training samples (0 corrected) in the database. ML cannot learn it reliably.
BATCH_PROCESS detects SVT from signal features directly.

### Criteria (from `calculate_metrics()` svt_flag block)

```
HR > 100 bpm (supraventricular rate)
AND QRS < 120 ms (narrow complex — not ventricular)
AND rr_cv < 0.10 (very regular — not AF)
AND p_absent_frac > 0.60 (P-waves absent/hidden in T — AV node re-entry)
```

### Where to add

As Algorithm 5 in `detect_lethal_rhythm()`, after the coarse VFib check:
- If SVT → return "SVT", confidence 0.80, reason string

Note: SVT is **not lethal** but is caught here before ML because ML has no training data for it.
The function name `detect_lethal_rhythm` may be broadened to `detect_signal_rhythm` or the SVT
logic can be in a separate sub-function called from the same orchestrator step.

---

## P1: Remove ML Veto Block — `rhythm_orchestrator.py` lines 81–94 (MODIFY)

### Why

The ML veto allows the rhythm model to override the sinus gate if it sees a dangerous rhythm at ≥0.88
confidence. This is backwards: the sinus gate uses 10 signal-processing criteria and is more reliable
than a rhythm model with 56% balanced accuracy. The veto introduces false alarms — a sinus tachycardia
labelled as VT by a weak model would override a correct sinus detection.

With the lethal detector now in place (Step 3.5), any truly dangerous rhythm at high confidence will be
caught by signal processing before ML even runs. The ML veto is redundant and harmful.

### What to delete

Lines 81–94 in `rhythm_orchestrator.py`:
```python
_DANGEROUS_RHYTHMS = {
    "Atrial Fibrillation", "AF", "Atrial Flutter",
    "3rd Degree AV Block", "2nd Degree AV Block Type 2",
    "Ventricular Fibrillation", "Ventricular Tachycardia", "VT",
}
_rhythm_block_early = ml_prediction.get("rhythm") or {}
_ml_label_early     = _rhythm_block_early.get("label", "Unknown")
_ml_conf_early      = float(_rhythm_block_early.get("confidence", 0.0))
if _ml_label_early in _DANGEROUS_RHYTHMS and _ml_conf_early >= 0.88:
    decision.background_rhythm = _ml_label_early
    print(f"[ML Veto] {_ml_label_early} (conf={_ml_conf_early:.2f}) overrides {sinus_label}")
```

Delete entirely — no replacement needed.

---

## P1: Remove Idioventricular Rhythm — `rhythm_orchestrator.py` lines 261–263 (MODIFY)

### Why

Idioventricular Rhythm (wide-QRS escape at HR <60) has zero corrected training samples. The detection
block (P_ratio <0.2, QRS >120ms, HR <60) will fire incorrectly on BBB/IVCD patients with slow rates.
With the lethal detector now covering wide-QRS fast rhythms (VTach), the slow wide-complex scenario
is marginal and not reliably detectable without data. Removing it prevents false labelling.

### What to delete

In `_detect_background_rhythm()`, lines 261–263:
```python
if p_ratio < 0.2 and qrs_ms > 120:
    return "Idioventricular Rhythm"   # Wide-complex ventricular escape
```

Delete these 2 lines. The code path that follows (Junctional Rhythm at lines 263–265) remains unchanged.

---

## P1: Rename BBB → IVCD — `data_loader.py` + `rhythm_orchestrator.py` (MODIFY)

### Why

"Bundle Branch Block" (BBB) is ambiguous — it bundles LBBB, RBBB, LAFB, and incomplete blocks.
"Intraventricular Conduction Delay" (IVCD) is the clinically accurate umbrella term used in modern
cardiology and is what device reports should say. This is a label rename only — no retraining needed
since model weights are class-index based, not string based.

### Changes in `models_training/data_loader.py`

Replace all instances of `"Bundle Branch Block"` with `"Intraventricular Conduction Delay"`:

| Line | Old | New |
|---|---|---|
| 351 | `"BBB": "Bundle Branch Block"` | `"BBB": "Intraventricular Conduction Delay"` |
| 355 | `"LBBB": "Bundle Branch Block"` | `"LBBB": "Intraventricular Conduction Delay"` |
| 355 | `"RBBB": "Bundle Branch Block"` | `"RBBB": "Intraventricular Conduction Delay"` |
| 381 | `"L": "Bundle Branch Block"` | `"L": "Intraventricular Conduction Delay"` |
| 382 | `"R": "Bundle Branch Block"` | `"R": "Intraventricular Conduction Delay"` |

### Changes in `decision_engine/rhythm_orchestrator.py`

Replace the confidence threshold key:
```python
# Old:
"Bundle Branch Block":          0.80,
# New:
"Intraventricular Conduction Delay": 0.80,
```

Also update any string comparisons or priority blocks that reference `"Bundle Branch Block"`.

---

## P2: PVC/PAC Template Correlation Refinement — `decision_engine/beat_classifier.py` (CREATE)

### Why

The current system uses the ectopy ML model alone for PVC/PAC classification. This fails in two cases:
1. Aberrant conduction (PAC with wide QRS looks like PVC to ML)
2. Ashman phenomenon (PAC at end of long-short RR sequence appears morphologically different)

BATCH_PROCESS uses a multi-criteria scoring system with template correlation as the primary discriminator.
This significantly reduces PVC/PAC confusion and is especially important for clinical reports.

### What Is Being Ported (from `classify_beats()` lines 672–835)

**Step 1 — Build sinus template**: Average all beats labelled "Normal" by ML in the same 10s window.
Build a fixed-length beat template (±200ms around R-peak).

**Step 2 — For each premature beat** (any beat with coupling ratio <0.92):

**Compute:**
- Template correlation (Pearson) against sinus template
- Coupling interval (ms)
- Compensatory/non-compensatory pause ratio
- QRS duration (from V3 delineation)
- T-wave discordance (polarity relative to QRS)
- P-wave presence/polarity (from V3 delineation)
- Ashman check: `(rr_pre_prev > 1.30 × median_rr) AND (rr_prev < 0.92 × median_rr)`
- T-wave deformation on preceding beat: `t_amp_prev > 1.30 × ref_t_amp`

**PVC score (threshold ≥3.0):**
- +2.0 QRS >120ms
- +1.5 T-wave discordant
- +1.5 P-wave absent
- +1.0 coupling <88% RR
- +1.0 compensatory pause >1.85×RR
- +2.5 template corr <0.60
- -1.0 if template corr ≥0.85 (soft penalty — sinus morphology disfavours PVC)

**PAC score (threshold ≥3.0, requires coupling <0.92):**
- +1.0 QRS <110ms
- +1.5 P-wave inverted
- +0.5 P-wave present
- +1.0 coupling <88% RR
- +1.0 non-compensatory pause <1.90×RR
- +2.5 template corr ≥0.85 (KEY DISCRIMINATOR)
- +0.5 corr soft boost
- +1.5 Ashman phenomenon
- +1.0 preceding T-wave deformation

**Decision:**
- PVC ≥3.0 AND PVC > PAC → override to "PVC"
- PAC ≥3.0 AND coupling <0.92 → override to "PAC"
- Otherwise → keep ML label

**Boundary rule:** First and last beat of every 10s segment are never flagged as ectopic.

### Integration point

Called in `rhythm_orchestrator.py` Step 4C, **after** ML ectopy gives per-beat predictions. Takes the
ML `beat_events` list and returns a refined list where low-confidence ML calls are replaced by the
template-correlation decision.

Only overrides when:
1. ML ectopy confidence is below 0.99 (high-confidence ML is trusted as-is)
2. Beat has a coupling ratio <0.92 (premature beat — Ashman/template analysis is applicable)
3. ≥3 normal beats exist in the same window to build a reliable sinus template

### Files to create

- `decision_engine/beat_classifier.py` — full implementation
- Wire into `rhythm_orchestrator.py` Step 4C

---

## What Is NOT Being Changed

| Component | Why kept |
|---|---|
| V3 ensemble R-peak detector (3 detectors, voting) | Better than BATCH_PROCESS entropy method — voting eliminates false positives |
| V3 adaptive baseline removal with BBB guard | Better than BATCH_PROCESS 600ms median filter — 600ms median corrupts wide QRS |
| V3 NeuroKit2 DWT delineation | Better than BATCH_PROCESS prominence-based — DWT is the clinical standard |
| V3 60-feature extraction | Already feeding both ML and signal processing checks |
| Sinus detection (10 criteria, sinus_detector.py) | Already correct and working — signal processing before ML |
| VT/NSVT from consecutive PVC rules | Kept as secondary path — complementary to lethal detector |
| Atrial Flutter spectral detection in rules.py | Already working |
| Display arbitration suppression in rules.py | Display-only, not permanent — all events in `decision.events`; suppression is clinically correct |
| 2nd Degree AV Block Type 1 (Wenckebach) | In schema, alias mapped, but zero training data — inactive until data exists |

---

## Final Architecture After All Changes

```
10s ECG window (raw)
       ↓
[V3 Signal Processing] — preprocess_v3 → ensemble R-peaks → DWT delineation → 60 features
       ↓
[Layer 1 — Sinus Gate]      signal processing, 10 criteria
  → Sinus/Brady/Tachy detected? YES → done, ML skipped
                               NO  → ↓
[Layer 2 — Lethal Gate]     signal processing, spectral + kinetic + polymorphic + coarse VFib + SVT
  → VTach/VFib/SVT detected? YES → done, ML skipped, high-priority
                              NO  → ↓
[Layer 3 — ML Rhythm]       CNNTransformerWithFeatures (9 classes, 56% balanced accuracy)
  → AF / AFl / 3rd AVB / 2nd AVB Type 2 / IVCD / NSVT / etc.
       ↓
[Layer 4 — Rule Events]     Pause, Atrial Flutter spectral, PVC/PAC from ML ectopy
       ↓
[Layer 4B — Beat Refinement] Template correlation → override low-confidence PVC/PAC calls
       ↓
[Layer 5 — Pattern Rules]   Bigeminy/Trigeminy/Couplet/NSVT/VT from consecutive beats
       ↓
[Layer 6 — Priority Promotion] VF > VT > AF > NSVT promote to background
       ↓
[Layer 7 — Display Rules]   Suppression hierarchy (display-only)
       ↓
final_display_events → UI / PDF report
```

---

## Verification Tests

| Test | Expected Result |
|---|---|
| `ADM441825561.json` (Atrial Flutter ~145 BPM) | Flutter detected, NOT VTach — rr_cv <0.08 flutter guard fires |
| `ADM1196270205.json` (IVCD + 1st Degree AVB) | IVCD label (was BBB), 1st AVB — NOT VTach (HR normal, kinetic check fails) |
| Synthetic VFib signal (high-variance 3–6 Hz noise) | VFib detected via spectral SPI >0.75, concentration ≤0.35 |
| Synthetic VTach (150 BPM, wide QRS) | VTach detected via spectral SPI >0.75, concentration >0.35 OR kinetic check |
| Bigeminy pattern (alternating PVC every beat) | Organization gate fires → SURVIVABLE → not VTach |
| Normal sinus at 130 BPM (sinus tach) | Regular, rate <130 gate → SURVIVABLE (or sinus gate fires first) |
