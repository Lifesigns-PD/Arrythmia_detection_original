# ECG Arrhythmia System — Complete Architecture & Detection Map

---

## 1. Every Arrhythmia Output the System Can Produce

### Background Rhythm Labels (one per segment)
| Label | Source |
|---|---|
| Sinus Rhythm | Signal processing — sinus_detector.py |
| Sinus Bradycardia | Signal processing — sinus_detector.py |
| Sinus Tachycardia | Signal processing — sinus_detector.py |
| Atrial Fibrillation | ML rhythm model → rules safety net backup |
| Atrial Flutter | ML rhythm model → rules spectral backup |
| 1st Degree AV Block | ML rhythm model only |
| 3rd Degree AV Block | ML rhythm model only |
| Bundle Branch Block | ML rhythm model only |
| Artifact | ML rhythm model only |
| 2nd Degree AV Block Type 2 | ML rhythm model only |
| Ventricular Tachycardia | Rules (consecutive PVC count) — no direct ML |
| Junctional Rhythm | Fallback heuristic in orchestrator (HR 60–100, no P) |
| Idioventricular Rhythm | Fallback heuristic in orchestrator (slow, wide QRS, no P) |
| Unknown | No method fired with enough confidence |

### Event Labels (zero or more per segment, on top of background)
| Label | Category | Source |
|---|---|---|
| PVC | ECTOPY | ML ectopy model (per-beat) |
| PAC | ECTOPY | ML ectopy model (per-beat) |
| PVC Couplet | RHYTHM | Rules — 2 consecutive PVC beats |
| Ventricular Run | RHYTHM | Rules — 3 consecutive PVC beats |
| NSVT | RHYTHM | Rules — 4–10 consecutive PVC beats at ≥100 bpm |
| VT | RHYTHM | Rules — 11+ consecutive PVC beats at ≥100 bpm |
| PVC Bigeminy | RHYTHM | Rules — every 2nd beat is PVC |
| PVC Trigeminy | RHYTHM | Rules — every 3rd beat is PVC |
| PVC Quadrigeminy | RHYTHM | Rules — every 4th beat is PVC |
| Atrial Couplet | RHYTHM | Rules — 2 consecutive PAC beats |
| Atrial Run | RHYTHM | Rules — 3–5 consecutive PAC beats |
| PSVT | RHYTHM | Rules — 6–10 consecutive PAC beats at ≥100 bpm |
| SVT | RHYTHM | Rules — 11+ consecutive PAC beats at ≥100 bpm |
| PAC Bigeminy | RHYTHM | Rules — every 2nd beat is PAC |
| PAC Trigeminy | RHYTHM | Rules — every 3rd beat is PAC |
| PAC Quadrigeminy | RHYTHM | Rules — every 4th beat is PAC |
| Pause | RHYTHM | Rules — any RR interval >2000 ms |
| Atrial Flutter | RHYTHM | Rules — spectral flutter waves at 4–6.5 Hz in QRS-blanked signal |
| Atrial Fibrillation | RHYTHM | Rules — AF safety net (RR std >160 ms + P-wave ratio <0.4) |
| Artifact | RHYTHM | Orchestrator — when SQI <0.3 |

---

## 2. Complete Detection Architecture — Layer by Layer

```
RAW SIGNAL (mV, 125 Hz, ≥500 samples)
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 0 — SIGNAL PROCESSING V3 (runs on every segment)    │
│                                                             │
│  preprocess_v3()          preprocessing/pipeline.py         │
│    ├─ Adaptive baseline removal (3-method + BBB guard)      │
│    ├─ Adaptive denoising (powerline + LP)                   │
│    └─ Artifact removal (muscle spikes)                      │
│                                                             │
│  detect_r_peaks_ensemble() detection/ensemble.py            │
│    ├─ Pan-Tompkins detector                                 │
│    ├─ Hilbert envelope detector                             │
│    ├─ Mexican Hat wavelet detector                          │
│    └─ VOTE: ≥2/3 agree within ±50ms → confirmed R-peak     │
│                                                             │
│  delineate_v3()           delineation/hybrid.py             │
│    ├─ wavelet_delineation.py (CWT — primary)                │
│    ├─ template_matching.py (patient template — refine)      │
│    └─ NeuroKit2 DWT fallback (missing fiducial points)      │
│                                                             │
│  extract_features_v3()    features/extraction.py            │
│    ├─ HRV time domain: 11 features (SDNN, RMSSD, pNN50…)   │
│    ├─ HRV frequency: 8 features (LF, HF, LF/HF…)          │
│    ├─ Nonlinear: 8 features (entropy, DFA, Poincaré…)      │
│    ├─ Morphology: 13 features (QRS/PR/QT, ST, T-amp…)      │
│    └─ Beat discriminators: 20 features (PVC/PAC scores…)   │
│    → 60 features total                                      │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — SINUS DETECTION  decision_engine/sinus_detector  │
│  (Signal processing only. Fires BEFORE ML.)                 │
│                                                             │
│  9 criteria (ALL must pass for sinus):                      │
│   1. P-wave present ratio ≥ 0.6                             │
│   2. QRS duration < 120 ms                                  │
│   3. PR interval 100–250 ms                                 │
│   4. RR coefficient of variation < 0.15                     │
│   5. PVC score < threshold                                  │
│   6. PAC score < threshold                                  │
│   7. LF/HF ratio > 0.5                                      │
│   8. HR plausible for sinus (40–180 bpm)                    │
│   9. Wide-QRS fraction < 0.3                                │
│                                                             │
│  If PASS → Sinus Rhythm / Sinus Bradycardia / Sinus Tachy   │
│    └─ ML rhythm model SKIPPED                               │
│    └─ ML can veto if it sees dangerous rhythm at ≥0.88 conf │
│                                                             │
│  If FAIL → pass to Layer 2                                  │
└─────────────────────────────────────────────────────────────┘
           │ (only if not Sinus)
           ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2 — ML RHYTHM MODEL   xai/xai.py → rhythm model     │
│  (CNNTransformerWithFeatures, 9 classes)                    │
│                                                             │
│  Input: float32[1250] signal + float32[36] features         │
│  (features normalized by feature_scaler_rhythm.joblib)      │
│                                                             │
│  9 output classes with per-class confidence thresholds:     │
│   Sinus Rhythm         → not used (Layer 1 handles this)    │
│   Atrial Fibrillation  → threshold 0.85                     │
│   Atrial Flutter       → threshold 0.85                     │
│   1st Degree AV Block  → threshold 0.80                     │
│   3rd Degree AV Block  → threshold 0.85                     │
│   Bundle Branch Block  → threshold 0.80                     │
│   Artifact             → threshold 0.80                     │
│   Sinus Bradycardia    → not used (Layer 1 handles this)    │
│   2nd Degree AVB Type2 → threshold 0.85                     │
│                                                             │
│  If ML fires → sets background_rhythm                       │
│  If ML below threshold → background_rhythm stays "Unknown"  │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3 — ML ECTOPY MODEL   xai/xai.py → ectopy model     │
│  (CNNTransformerWithFeatures, 3 classes — per beat)         │
│                                                             │
│  Input: float32[250] — 2-second window centered on R-peak   │
│         float32[47] features (feature_scaler_ectopy.joblib) │
│                                                             │
│  Per-beat classification (one per R-peak):                  │
│   0: None  (normal / no ectopy)                             │
│   1: PVC   → if confidence ≥ 0.97                           │
│   2: PAC   → if confidence ≥ 0.97                           │
│                                                             │
│  Beat gates (suppress hallucinations):                      │
│   • Base gate: conf ≥ 0.97                                  │
│   • Rhythm trust gate: if sinus + <3 beats → conf ≥ 0.99   │
│   • Density gate: if >60% beats ectopic on sinus bg → drop  │
│     (unless 3+ consecutive indices = real run)              │
│                                                             │
│  Passes beat_events list with beat_idx to Layer 4           │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 4 — RULES ENGINE   decision_engine/rules.py          │
│                                                             │
│  A) derive_rule_events() — signal-level rules               │
│   • Pause: any RR > 2000 ms → Event("Pause")               │
│   • AF safety net: RR_std >160 ms AND P-ratio <0.4 →        │
│     Event("Atrial Fibrillation") [only if not sinus]        │
│   • Atrial Flutter: spectral peak at 4–6.5 Hz in           │
│     QRS-blanked signal > 2.5× median PSD                   │
│                                                             │
│  B) apply_ectopy_patterns() — beat-index rules on PVC/PAC   │
│   Clustering: gap ≤ 2s between beats of same type           │
│                                                             │
│   PVC patterns (from beat_indices diffs):                   │
│    diffs all=2 (≥2 cycles)  → PVC Bigeminy                  │
│    diffs all=3 (≥2 cycles)  → PVC Trigeminy                 │
│    diffs all=4 (≥2 cycles)  → PVC Quadrigeminy              │
│    consecutive + count=2    → PVC Couplet                   │
│    consecutive + count=3    → Ventricular Run               │
│    consecutive + count 4-10 + rate≥100 → NSVT              │
│    consecutive + count≥11 + rate≥100  → VT                  │
│    (if QRS <110 ms: SVT instead of VT/NSVT)                 │
│                                                             │
│   PAC patterns (same logic):                                │
│    → PAC Bigeminy / Trigeminy / Quadrigeminy                │
│    → Atrial Couplet / Atrial Run                            │
│    → PSVT (4–10 consecutive) / SVT (11+ consecutive)        │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 5 — PRIORITY PROMOTION  rhythm_orchestrator.py       │
│                                                             │
│  Escalates background_rhythm if high-priority event found:  │
│   VF event found      → background = "Ventricular Fibrillation"│
│   VT event found      → background = "Ventricular Tachycardia"│
│   AF/AFL event found  → background = event label            │
│   NSVT event found    → background = "NSVT"                 │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 6 — DISPLAY ARBITRATION  rules.apply_display_rules() │
│                                                             │
│  Suppression hierarchy:                                     │
│   AF/AFL present  → hide isolated PVC/PAC events            │
│   VT/NSVT present → hide isolated PVC events                │
│   SVT/PSVT present→ hide isolated PAC events                │
│   Background = sinus → hide background-rhythm events        │
│                                                             │
│  → final_display_events (what the UI/PDF shows)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. What Has a Backup (Redundancy Map)

| Arrhythmia | Primary Detection | Backup Detection | If Both Fail |
|---|---|---|---|
| Sinus Rhythm | Signal processing (sinus_detector, 9 criteria) | None — sinus is the default if nothing else fires | Classified as "Unknown" or by ML (which it shouldn't override) |
| Sinus Bradycardia | Signal processing (same + HR <60) | Orchestrator fallback heuristic (HR <60, P present) | "Unknown" |
| Sinus Tachycardia | Signal processing (same + HR >100) | Orchestrator fallback heuristic (HR >100) | "Unknown" |
| Atrial Fibrillation | ML rhythm model (threshold 0.85) | Rules safety net: RR_std >160ms + P-ratio <0.4 | Missed |
| Atrial Flutter | ML rhythm model (threshold 0.85) | Rules: spectral flutter waves 4–6.5 Hz | Missed |
| 1st Degree AV Block | ML rhythm model (threshold 0.80) | None | Missed |
| 3rd Degree AV Block | ML rhythm model (threshold 0.85) | None | Missed |
| Bundle Branch Block | ML rhythm model (threshold 0.80) | None | Missed |
| 2nd Degree AV Block Type 2 | ML rhythm model (threshold 0.85) | None | Missed |
| PVC (isolated) | ML ectopy model (conf ≥0.97) | None | Missed |
| PAC (isolated) | ML ectopy model (conf ≥0.97) | None | Missed |
| PVC Bigeminy/Trigeminy | Rules (beat index pattern from ML ectopy) | None | Missed if ML ectopy fails |
| Couplet | Rules (2 consecutive PVCs from ML ectopy) | None | Missed if ML ectopy fails |
| NSVT | Rules (4–10 consecutive PVCs from ML ectopy) | None | Missed if ML ectopy fails |
| VT | Rules (11+ consecutive PVCs from ML ectopy) | None | Missed if ML ectopy fails |
| Pause | Rules (RR >2000 ms — deterministic) | None needed — pure threshold | Cannot fail |
| VFib | **NOTHING — no detection at all** | N/A | Always missed |

### Critical Gaps (No Detection or No Backup)
- **VFib**: Zero detection. No signal processing, no ML, no rules.
- **VTach**: Only detected if ML ectopy labels ≥11 consecutive beats as PVC. If the ectopy model misses even 1 beat in the run, NSVT/VT is never raised.
- **SVT**: Only raised when 11+ consecutive PAC beats detected by ectopy model. No direct signal processing route.
- **All blocks, BBB**: Single point of failure = ML rhythm model. No backup.
- **PAC patterns**: PAC recall is ~75% (after finetune). Pattern rules will miss bigeminy/trigeminy frequently.

---

## 4. PQRST Wave Detection — BATCH_PROCESS vs V3 Pipeline — Full Expose

### R-Peak Detection

| Aspect | BATCH_PROCESS (lifesigns_engine) | V3 Pipeline (ensemble.py) | Winner |
|---|---|---|---|
| Algorithm | Single: entropy-of-slopes on Butterworth 0.5–40 Hz signal | Three-detector ensemble: Pan-Tompkins + Hilbert + Mexican Hat CWT | **V3** |
| False positive protection | Moving-average threshold + prominence 0.01 | Voting: ≥2/3 detectors must agree within ±50 ms | **V3** |
| Inverted QRS | `argmax(|signal|)` — finds absolute max | Polarity detection (neg >1.3× pos → flip), then `argmin` | **V3** |
| Irregular rhythms (AF, PVC) | Fixed 200 ms distance constraint | Adaptive window: expands from 50 ms to 80 ms when CV >0.15 | **V3** |
| Peak position precision | Refine to argmax(|fiducial|) within ±1 window | Refine to argmax/argmin within ±20 ms + sub-sample parabolic interpolation | **V3** |
| VFib / flat signal | Entropy threshold catches semi-random peaks | Ensemble vote fails cleanly → returns empty array | **DRAW** |
| High rate (>140 bpm) | Works — 200 ms distance allows up to 300 bpm | Works — 50 ms window allows detection; RR <200 ms filtered out | **DRAW** |

**Verdict: V3 ensemble wins clearly. Three-detector voting eliminates false positives that single-algorithm systems produce on PVC T-waves and baseline artifacts.**

---

### QRS Onset/Offset (Boundary Detection)

| Aspect | BATCH_PROCESS | V3 (wavelet_delineation.py) | Winner |
|---|---|---|---|
| Method | 15% amplitude threshold from local PR baseline | CWT zero-crossing (onset) + slope-flatness 4-sample J-point (offset) | **V3** |
| Local baseline | PR-segment median (120–280 ms before R) — **clever** | Physiological bounds only (no explicit local baseline) | **BATCH_PROCESS** |
| Inverted QRS | Dual branches (≥0 / <0) with sign-flipped thresholds | Flip signal before delineation; unified logic | **V3 simpler** |
| Offset criterion | Slope <0.02 mV/sample — single sample guard | 4 consecutive samples <0.025 mV/sample — stronger plateau | **V3** |
| Wide QRS (BBB, PVC) | Works — threshold-based adapts to any amplitude | Works — slope-flatness adapts to wide QRS | **DRAW** |
| Fallback | None — uses fixed 80 ms before R if scan fails | Falls back to 40 ms before R (onset) or CWT zero-crossing (offset) | **V3** |
| High rate (>140 bpm) | Skips QRS boundary delineation entirely | Delineates normally — no cutoff | **V3** |

**Key insight**: BATCH_PROCESS's PR-segment local baseline is genuinely better for QRS amplitude correction — if the signal has baseline drift, 15% of `(R_amp - local_baseline)` is more accurate than 15% of raw R amplitude. V3's wavelet_delineation does not do this. However V3's 4-sample slope-flatness for QRS offset is more robust than BATCH_PROCESS's single-sample guard.

**Verdict: V3 wins on robustness. BATCH_PROCESS has one superior detail (local baseline) worth adopting.**

---

### P-Wave Detection

| Aspect | BATCH_PROCESS | V3 (wavelet_delineation.py) | Winner |
|---|---|---|---|
| Signal used | `t_elim` — signal with QRS+T blanked (separate filtered channel) | Work signal (preprocessed, T-region bounded by search window) | **BATCH_PROCESS** |
| Detection criterion | Prominence ≥0.005 mV on blanked signal | Energy ratio ≥2.5× baseline + amplitude ≥0.04 mV | **V3** |
| AF f-wave rejection | SNR <1.0 or noise_var >0.005 mV² → "Review (Noisy)" | Energy ratio <2.5× (f-waves have ratio ~1.0–2.0) → "absent" | **V3** |
| Absent P-wave | Explicitly marked — "Absent" or "Explicitly Absent" | `p_morphology = "absent"`, all indices = None | **DRAW** |
| Morphology | Normal / Inverted (binary) | Normal / Inverted / Biphasic / Absent (4 classes) | **V3** |
| P-onset/offset | Not computed — only peak location | CWT zero-crossing before/after P-peak | **V3** |
| High rate (>140 bpm) | P-wave search skipped — flag set | P-wave searched but window may overlap T of previous beat | **BATCH_PROCESS** |
| Noisy P validation | SNR calculated against local noise variance | Energy ratio vs TP-segment baseline energy | **DRAW** |

**Key insight**: BATCH_PROCESS uses a separate T-eliminated channel (`t_elim`) built by blanking the QRS+T complex on a filtered signal. This is the correct approach — searching for P-waves in a signal where T-waves have been zeroed out eliminates the biggest source of P-wave false positives. V3 does not do this — it searches in the raw preprocessed signal with only a time-window bound.

**Verdict: DRAW with different strengths. V3 has better P-morphology classification and AF-rejection. BATCH_PROCESS has better P-wave isolation via T-elimination channel. BATCH_PROCESS correctly handles high rate (skips). V3 delineates P-onset and P-offset (V3 advantage for clinical intervals).**

---

### T-Wave Detection

| Aspect | BATCH_PROCESS | V3 (wavelet_delineation.py) | Winner |
|---|---|---|---|
| Signal used | `morph` — Savitzky-Golay smoothed 0.5–15 Hz signal | Work signal with 60–500 ms post-QRS window | **BATCH_PROCESS** |
| Detection criterion | Prominence ≥0.015 mV | Max `|amplitude|` in 60–500 ms post-QRS | **BATCH_PROCESS** |
| Inverted T | Negative prominence > 1.2× positive → "Inverted" | `t_amp < -0.05` → `t_inverted = True` | **DRAW** |
| T-onset | Not computed | CWT minimum before T-peak | **V3** |
| T-offset | Not computed | CWT zero-crossing after T-peak | **V3** |
| High rate (>140 bpm) | T-wave search skipped | Searched normally — may overlap next P-wave | **BATCH_PROCESS** |
| Flat T-wave | Prominence threshold catches it | Max absolute — may pick noise instead | **BATCH_PROCESS** |

**Verdict: BATCH_PROCESS wins on T-wave peak detection (smoothed signal + prominence is more noise-resistant). V3 wins on T-onset/offset computation (needed for QTc measurement).**

---

### Q and S Waves

| Aspect | BATCH_PROCESS | V3 (wavelet_delineation.py) | Winner |
|---|---|---|---|
| Q-wave | Within QRS bounds on `fiducial` (bandpass signal) | Absolute minimum between QRS-onset and R | **DRAW** |
| S-wave | Within QRS bounds on `fiducial` | Absolute minimum between R and QRS-offset | **DRAW** |
| Depth measurement | From fiducial signal directly | Signed — adjusted for QRS polarity | **V3** |
| Boundary | Uses q_on/q_off from QRS detection | Uses qrs_onset/qrs_offset from same delineation | **DRAW** |

**Verdict: Draw. Both are equivalent minimum-finding approaches.**

---

### Summary Score

| Component | BATCH_PROCESS | V3 Pipeline | Winner |
|---|---|---|---|
| R-peak detection | ❌ Single algorithm, no voting | ✅ 3-detector ensemble, adaptive window | **V3** |
| Preprocessing/filter | ❌ Fixed 600 ms median (corrupts BBB) | ✅ Adaptive 3-method + BBB guard | **V3** |
| QRS boundary | ✅ Local PR baseline correction | ✅ 4-sample slope-flatness J-point | **V3 (more robust)** |
| Q/S waves | ✅ Adequate | ✅ Adequate, polarity-adjusted | **V3 (slight)** |
| P-wave isolation | ✅ T-eliminated channel | ❌ Raw signal, time-window only | **BATCH_PROCESS** |
| P-wave classification | ❌ Binary (normal/inverted) | ✅ 4-class (normal/inverted/biphasic/absent) | **V3** |
| P-onset/offset | ❌ Not computed | ✅ CWT zero-crossing | **V3** |
| T-wave detection | ✅ Smoothed + prominence | ✅ Max absolute | **BATCH_PROCESS (slight)** |
| T-onset/offset | ❌ Not computed | ✅ CWT zero-crossing | **V3** |
| High rate (>140 bpm) | ✅ Skips P/T safely | ❌ Attempts P/T (overlap risk) | **BATCH_PROCESS** |
| VTach detection | ✅ Spectral + kinetic + polymorphic | ❌ None | **BATCH_PROCESS** |
| VFib detection | ✅ Spectral + coarse-VFib check | ❌ None | **BATCH_PROCESS** |
| SVT detection | ✅ Narrow + fast + regular + no P | ❌ Only via consecutive PAC count | **BATCH_PROCESS** |
| AFib detection | ✅ Bounded CV + no P (explicit flag) | ✅ ML + AF safety net | **DRAW** |

---

## 5. What Should Be Adopted from BATCH_PROCESS

### Must Adopt (Critical Gaps)
1. **`spectral_lethal_precheck()`** — VTach/VFib spectral detection. V3 has zero VFib detection and VTach only through consecutive beat counting. Port to `decision_engine/lethal_detector.py`.
2. **Kinetic VTach check** — Wide QRS + AV dissociation + fast HR. Port to same module.
3. **Polymorphic VTach / Torsades check** — Catches what spectral misses. Port to same module.
4. **Coarse VFib check** — Last-resort VFib via extreme RR chaos. Port to same module.

### Should Adopt (Improves Existing)
5. **High-rate mode (HR >140 bpm)** — Skip P and T delineation above 140 bpm in `wavelet_delineation.py`. Currently V3 tries to delineate and gets unreliable results that cascade into wrong features.
6. **SVT explicit detection** — Narrow + fast + regular + no P. Add to `lethal_detector.py` as non-lethal fast-path.

### Nice to Have (Future)
7. **T-eliminated channel for P-wave search** — Build `t_elim` (blank QRS+T from morphology signal) and use it to search for P-waves instead of raw signal. Improves P-wave accuracy especially in patients with large T-waves.
8. **Ashman phenomenon detection** — Long-short RR before a wide beat → PAC with aberrant conduction. Would improve PAC/PAC vs PVC discrimination.
9. **Local PR baseline for QRS amplitude** — Use median of 120–280 ms before R as baseline reference for 15% QRS boundary threshold. More accurate than raw amplitude in drifting signal.

### Do NOT Adopt (V3 Is Better)
- BATCH_PROCESS R-peak detection (entropy, single algorithm) → **V3 ensemble is superior**
- BATCH_PROCESS preprocessing (600 ms median) → **V3 adaptive baseline is superior, critical for BBB**
- BATCH_PROCESS QRS offset (single-sample slope guard) → **V3 4-sample plateau is more robust**
- BATCH_PROCESS P-morphology (binary) → **V3 4-class is more complete**

---

## 6. Proposed Final Architecture (After Planned Changes)

```
LAYER 0: V3 Signal Processing (unchanged)
  preprocess_v3 → ensemble R-peaks → delineate_v3 → extract_features_v3

LAYER 1: Sinus Detection (signal processing only — unchanged)
  sinus_detector.py → Sinus / Brady / Tachy or pass to next

LAYER 2: Lethal + SVT Detection [NEW — from BATCH_PROCESS]
  lethal_detector.py →
    spectral_lethal_precheck()   → VTach / VFib (SPI >0.75)
    _kinetic_vtach_check()       → VTach (wide + AV dissociation)
    _polymorphic_vtach_check()   → VTach (Torsades pattern)
    _coarse_vfib_check()         → VFib (extreme RR chaos)
    _svt_check()                 → SVT (narrow + fast + regular + no P)

LAYER 3: ML Rhythm Model (only if Layers 1+2 gave no answer)
  7 classes: AFib / AFL / 1st AVB / 3rd AVB / BBB / Artifact / 2nd AVB Type2
  (Sinus Bradycardia and Sinus Rhythm REMOVED — handled by Layer 1)

LAYER 4: ML Ectopy Model (always runs — per-beat)
  3 classes: None / PVC / PAC

LAYER 5: Rules (always runs on top of all layers)
  Pause / AF safety net / Atrial Flutter spectral
  Pattern rules: Bigeminy / Trigeminy / Couplet / Run / NSVT / VT from beat indices

LAYER 6: Priority Promotion + Display Arbitration (unchanged)
```

### One Method Per Arrhythmia (No Overlap)

| Arrhythmia | Exclusive Method |
|---|---|
| Sinus Rhythm / Brady / Tachy | Signal processing (Layer 1) |
| VTach | Signal processing (Layer 2) |
| VFib | Signal processing (Layer 2) |
| SVT | Signal processing (Layer 2) |
| AFib | ML (Layer 3) + rules safety net (Layer 5) |
| AFL | ML (Layer 3) + rules spectral (Layer 5) |
| 1st/3rd Degree AV Block | ML (Layer 3) |
| Bundle Branch Block | ML (Layer 3) |
| 2nd Degree AV Block Type 2 | ML (Layer 3) |
| Artifact | ML (Layer 3) |
| PVC / PAC isolated | ML ectopy (Layer 4) |
| NSVT / VT (from beats) | Rules pattern (Layer 5) |
| Bigeminy / Trigeminy | Rules pattern (Layer 5) |
| Pause | Rules threshold (Layer 5) |

AFib and AFL are the only arrhythmias with a backup (ML + rules safety net). Everything else is single-method. If the single method fails, it is missed.

---

## 7. Files to Create/Modify for Planned Changes

| File | Change | Priority |
|---|---|---|
| `decision_engine/lethal_detector.py` | CREATE — port 4 lethal + SVT detection from lifesigns_engine.py | CRITICAL |
| `decision_engine/rhythm_orchestrator.py` | MODIFY — add Step 3.5 lethal pre-check after sinus gate (~15 lines) | CRITICAL |
| `ecg_processor.py` | MODIFY — store cleaned signal in clinical_features (1 line) | CRITICAL |
| `signal_processing_v3/delineation/wavelet_delineation.py` | MODIFY — add high-rate mode skip for P/T above 140 bpm | MEDIUM |
| `models_training/data_loader.py` | MODIFY — remove Sinus Rhythm (0) and Sinus Bradycardia (7) from RHYTHM_CLASS_NAMES for next fresh training run | FUTURE |
