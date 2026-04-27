# Plan: VTach / VFib Detection via Signal Processing (No ML)

---

## Executive Summary

**The Problem**: The V3 pipeline cannot detect Ventricular Tachycardia (VTach) or Ventricular Fibrillation (VFib) directly from the ECG signal. These are the two most life-threatening cardiac arrhythmias. Currently, VTach is only caught *indirectly* — the ectopy ML model must label individual beats as PVCs first, then the rules engine counts 4+ consecutive PVCs to raise an NSVT/VT flag. If the ectopy model misses even one beat, VTach is missed entirely. VFib is not detected at all.

**The Solution**: The `BATCH_PROCESS/lifesigns_engine.py` file — built separately for the LifeSigns device — contains four robust signal-processing algorithms for lethal rhythm detection that work *without any ML model*. These algorithms use spectral analysis, QRS morphology, and RR interval statistics to identify VTach and VFib directly from the ECG waveform. This plan ports those four algorithms into the V3 pipeline as a **pre-ML safety gate**, so lethal rhythms are caught even if the ML model fails.

**Why Signal Processing Instead of ML for VTach/VFib?**
- The ML rhythm model has only 3 VT training samples in the entire database — it cannot reliably learn VTach
- VFib has 0 training samples — ML literally cannot detect it
- Spectral methods (Welch PSD) and wide-QRS kinetic criteria are electrophysiology gold standards used in real ICU monitors
- Signal processing is deterministic, explainable, and does not degrade with distribution shift
- It runs in milliseconds — no inference overhead

**What Is NOT Being Changed**: The V3 signal processing pipeline (ensemble R-peaks, adaptive baseline, NeuroKit2 delineation, 60-feature extraction) is *better* than BATCH_PROCESS's methods in every category except lethal detection. Only the lethal detection logic is being ported. Everything else stays exactly as it is.

---

## Context

The current V3 pipeline has **no standalone VTach or VFib detection**. VT is only caught indirectly — the ectopy ML model must first label individual beats as PVCs, and then the rules engine counts 4+ consecutive PVCs to raise NSVT/VT. VFib is **not detected at all**.

The `BATCH_PROCESS/lifesigns_engine.py` file contains battle-tested spectral and kinetic VTach/VFib detection that works **purely from signal processing** — no ML model required. This plan ports that logic into the V3 pipeline as a pre-ML gate, mirroring the existing sinus detection architecture:

```
CURRENT FLOW:
  Sinus rules (signal processing)  →  if YES: done
                                    →  if NO:  pass to ML

NEW FLOW:
  Sinus rules (signal processing)  →  if YES: done
                                    →  if NO:  ↓
  Lethal rules (signal processing) →  if VTach/VFib: done (CRITICAL, no ML needed)
                                    →  if safe:  pass to ML
```

---

## What the BATCH_PROCESS Engine Has (That V3 Lacks)

### 1. Spectral Lethal Pre-Check — `spectral_lethal_precheck()` (lines 260–343)

Two-stage detection on the **preprocessed/filtered** signal:

**Stage 1 — Organization gate** (false-positive prevention):
- Uses pre-computed R-peaks (NOT independent detection — critical design decision)
- **Bigeminy fingerprint**: even/odd RR alternation >12% AND each group CV <20% → return SURVIVABLE
- **Very regular + slow**: CV <0.20 AND rate <130 BPM AND ≥6 beats → return SURVIVABLE
- All other patterns fall through to spectral stage (covers VTach, VFib, rapid AFib)

**Stage 2 — Spectral power ratio** (Welch PSD):
- `SPI = power(1.5–7 Hz) / total power(0.5–40 Hz)`
- SPI >0.75 → compute concentration = peak_bin / band_lethal
  - concentration >0.35 → **VENTRICULAR TACHYCARDIA** (organized power = one dominant frequency)
  - concentration ≤0.35 → **VENTRICULAR FIBRILLATION** (diffuse power = chaotic signal)
- SPI ≤0.75 → SURVIVABLE

### 2. Kinetic VTach Check — from `calculate_metrics()` (lines 951–972)

Wide complex + AV dissociation criteria:
- HR >100 bpm
- QRS duration >120 ms (relaxed to >100 ms when HR >150)
- Wide QRS fraction >75% of beats (relaxed to >60% when HR >150)
- AV dissociation: P-absent fraction >70% OR (PR-SD >40 ms AND P-absent >40%)

### 3. Polymorphic VTach / Torsades Check — from `calculate_metrics()` (lines 905–915)

Catches Torsades de Pointes and polymorphic VT that evade spectral check:
- HR >150 bpm
- RR CV between 0.08 and 0.55 (not regular like SVT, not chaotic like VFib)
- P-absent fraction >60%

### 4. Coarse VFib Check — from `calculate_metrics()` (lines 896–903)

Catches coarse VFib that has some organized-looking RR intervals:
- RR CV >0.50 (completely chaotic)
- HR >80 bpm
- P-absent fraction >60%

---

## What the V3 Pipeline Already Has (Keep As-Is)

| Component | Status |
|---|---|
| Sinus detection via 9 criteria | ✅ Keep in `sinus_detector.py` |
| VT/NSVT from consecutive PVC beats | ✅ Keep in `rules.py` (secondary path) |
| Atrial Flutter via spectral flutter waves | ✅ Keep in `rules.py` |
| Ensemble R-peak detection (3 detectors, voting) | ✅ Better than BATCH_PROCESS entropy method |
| Adaptive baseline removal with BBB guard | ✅ Better than BATCH_PROCESS median filter |
| NeuroKit2 DWT delineation | ✅ Better than BATCH_PROCESS prominence-based |
| 60-feature extraction | ✅ Keep — feeds both ML and lethal checks |

---

## Files to Create / Modify

### 1. CREATE — `decision_engine/lethal_detector.py`

New module, single responsibility: detect VTach/VFib from signal + V3 features, no ML.

**Public function:**
```python
def detect_lethal_rhythm(
    signal: np.ndarray,    # preprocessed (cleaned) 10s window
    r_peaks: np.ndarray,   # from ensemble detector
    features: dict,        # V3 60-feature dict from extract_features_v3()
    fs: int = 125,
) -> tuple[str | None, float, str]:
    """
    Returns (label, confidence, reason):
      label = "Ventricular Tachycardia" | "Ventricular Fibrillation" | None
      confidence = 0.0–1.0
      reason = human-readable string for logging
    None = no lethal rhythm, proceed to ML.
    """
```

**Internal detection order:**
1. `_spectral_lethal_check(signal, r_peaks, fs)` — port of `spectral_lethal_precheck()`
   - If lethal → return label at confidence **0.92**
2. `_kinetic_vtach_check(features)` — from `calculate_metrics()` vtach block
   - If True → return "Ventricular Tachycardia" at confidence **0.85**
3. `_polymorphic_vtach_check(features)` — from `calculate_metrics()` polymorphic block
   - If True → return "Ventricular Tachycardia" at confidence **0.82**
4. `_coarse_vfib_check(features)` — from `calculate_metrics()` coarse_vfib block
   - If True → return "Ventricular Fibrillation" at confidence **0.80**
5. Return `(None, 0.0, "No lethal rhythm detected")`

**Feature mapping** (lifesigns_engine → V3 feature keys):
| lifesigns_engine | V3 features key | Notes |
|---|---|---|
| `metrics["hr"]` | `features["mean_hr_bpm"]` | Direct |
| `metrics["hrv_cv"]` | `features["rr_cv"]` | Direct |
| `metrics["qrs_dur"]` | `features["qrs_duration_ms"]` | Direct |
| `qrs_wide_fraction` | `features["wide_qrs_fraction"]` | Direct |
| `p_absent_frac` | `1.0 - features["p_wave_present_ratio"]` | Invert |
| `pr_int_sd` | Not in V3 features | Use P-absent arm only (>0.70 threshold) |

---

### 2. MODIFY — `decision_engine/rhythm_orchestrator.py`

**Where**: In `decide()`, after sinus detection (line 98), before Step 4 rule events (line 100).

**What to add** (Step 3.5):
```python
# Step 3.5 — Lethal rhythm pre-check (VTach/VFib via signal processing, before ML)
_lethal_label = None
if decision.background_rhythm == "Unknown":
    _signal_data = clinical_features.get("_signal_clean") or clinical_features.get("_signal")
    _rp          = clinical_features.get("r_peaks")
    if _signal_data and _rp:
        from decision_engine.lethal_detector import detect_lethal_rhythm
        _lethal_label, _lethal_conf, _lethal_reason = detect_lethal_rhythm(
            signal=np.asarray(_signal_data, dtype=np.float32),
            r_peaks=np.asarray(_rp, dtype=int),
            features=clinical_features,
            fs=int(clinical_features.get("fs", 125)),
        )
        if _lethal_label:
            decision.background_rhythm = _lethal_label
            print(f"[Lethal Detection] {_lethal_label} (conf={_lethal_conf:.2f}) — {_lethal_reason}")
```

The existing Step 4B ML block is already gated on `decision.background_rhythm == "Unknown"` — so if lethal detection fires and sets background_rhythm, ML rhythm is automatically skipped. **No other changes needed in the orchestrator.**

---

### 3. MODIFY — `ecg_processor.py`

**Where**: In `_run_orchestrator()`, after `v3 = process_ecg_v3(...)` is called.

**What to add**: Store the cleaned signal for lethal detection:
```python
# After: v3 = process_ecg_v3(window, fs=SAMPLING_RATE, min_quality=0.2)
# Add:
clinical_features["_signal_clean"] = v3.get("cleaned", window).tolist()
```

This gives the lethal detector the bandpass-filtered signal (not raw) — same as what BATCH_PROCESS uses. The BATCH_PROCESS doc explicitly states spectral check must use filtered signal to avoid EMG noise corrupting the 1.5–7 Hz band.

---

## Detection Logic Flow (After Implementation)

```
10s ECG window (preprocessed by V3)
       ↓
[Sinus Detector] — 9 signal processing criteria
  → Sinus/Brady/Tachy detected?
    YES → background_rhythm set, ML skipped
    NO  → ↓
[Lethal Detector] — spectral + kinetic + polymorphic + coarse VFib
  → VTach or VFib detected?
    YES → background_rhythm set, ML skipped, high-priority event added
    NO  → ↓
[ML Model — Rhythm] — CNNTransformerWithFeatures (9 classes)
  → label + confidence → if above threshold → background_rhythm set
       ↓
[Rule Events] — Pause, AF safety net, Atrial Flutter
       ↓
[ML Model — Ectopy] — per-beat PVC/PAC labels
       ↓
[Pattern Rules] — Bigeminy/Trigeminy/Couplet/NSVT/VT from consecutive beats
       ↓
[Display Arbitration] → final_display_events
```

---

## Why Not Use the BATCH_PROCESS R-Peak Detector or Preprocessing?

After full code review:

| Component | BATCH_PROCESS | V3 | Verdict |
|---|---|---|---|
| R-peak detection | Entropy-based, single algorithm | 3-detector ensemble voting | **V3 wins** — voting eliminates false positives |
| Preprocessing | Fixed 600ms median filter | Adaptive 3-method + BBB guard | **V3 wins** — 600ms median corrupts wide QRS (BBB) |
| Delineation | Custom prominence-based (disabled HR>140) | NeuroKit2 DWT | **V3 wins** — DWT is gold standard |
| Lethal detection | Spectral + kinetic + polymorphic + coarse VFib | Nothing | **BATCH_PROCESS wins** — V3 has none |
| Ashman phenomenon | Yes — PAC looks wide after long-short RR | No | Nice-to-have, can add later |
| Template correlation | Yes — Pearson vs sinus template | No | Nice-to-have, can add later |

**Only the lethal detection logic is worth porting. Everything else in V3 is already better.**

---

## Confidence Thresholds Summary

| Detection method | Label | Confidence |
|---|---|---|
| Spectral (SPI >0.75, organized) | VTach | 0.92 |
| Spectral (SPI >0.75, diffuse) | VFib | 0.92 |
| Kinetic (wide + AV dissociation) | VTach | 0.85 |
| Polymorphic (fast + mild irregular + no P) | VTach | 0.82 |
| Coarse VFib (CV >0.50, fast, no P) | VFib | 0.80 |

---

## Verification Steps

1. **Regression — Atrial Flutter**: Run `ADM441825561.json` → must still report Atrial Flutter, NOT VTach (flutter rate ~145 bpm, but organization gate should see regular rhythm or spectral peak at 2.4 Hz = 145 bpm, which is outside 1.5–7 Hz VTach band... actually 145 bpm = 2.4 Hz which IS in 1.5–7 Hz. Need to verify SPI <0.75 for flutter signal. Flutter has dominant atrial frequency at 5 Hz but ventricular response is 2:1 at ~2.5 Hz — spectral energy is spread, SPI unlikely >0.75. If false alarm: organization gate should catch it as very regular + rate = 145 > 130 so gate won't fire. May need SPI threshold adjustment.)

2. **Regression — BBB**: Run `ADM1196270205.json` → must still report BBB + 1st Degree AV Block, NOT VTach (BBB has wide QRS but slow/normal HR, kinetic VTach check requires HR >100)

3. **Lethal detection smoke test**: Verify with a known VTach/VFib pattern from `ecg_data_extracts/` folder if one exists, or synthesize a test signal

4. **PDF report**: Regenerate PDF for both files — labels in strips should match regression results

---

## Risk: Atrial Flutter False Alarm

**Issue**: Atrial flutter ventricular rate ~150 BPM = 2.5 Hz, which falls inside the 1.5–7 Hz band used for VTach spectral detection. If flutter signal has SPI >0.75, it would be falsely labelled VTach.

**Mitigation** (already in BATCH_PROCESS organization gate):
- If R-peaks produce CV <0.20 AND rate <130 → SURVIVABLE (gate fires)
- Flutter at 150 BPM: rate =150 >130 so this gate **does not fire**
- However flutter has very narrow, organized spectral peak at 2.5 Hz, so concentration >0.35 is likely → would be labelled VTach (wrong!)

**Proposed guard**: Add a flutter exclusion — if `rules.py _detect_flutter_waves()` already fires in the same segment, skip lethal detection. OR: add HR guard — only run spectral check if HR >120 AND signal is irregular (CV >0.10). Flutter is regular (CV <0.10).

**Recommended fix in `lethal_detector.py`**: Before spectral stage, check regularity:
```python
rr_cv = features.get("rr_cv", 0)
if rr_cv < 0.08:
    # Very regular — likely flutter or SVT or sinus tach, not VTach/VFib
    # Skip spectral (organization gate equivalent using V3 features)
    return None, 0.0, "Regular rhythm — not lethal"
```
This is safer than the BATCH_PROCESS gate (which uses CV <0.20 and rate <130) because V3's `rr_cv` is computed from cleaned RR intervals.
