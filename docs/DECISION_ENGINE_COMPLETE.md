# Decision Engine — Complete Documentation
## Rules, Confidence Gates, Event Hierarchy, Display Logic

---

## 1. What the Decision Engine Does

After the models make predictions, the decision engine:
1. Applies clinical rules that the model cannot be expected to learn (rare, rule-definable)
2. Corrects PVC/PAC labels using electrophysiology criteria
3. Detects complex arrhythmia patterns from per-beat events (bigeminy, runs, VT)
4. Promotes high-priority patterns to background rhythm
5. Arbitrates display: which events are shown, which are suppressed
6. Sets training flags: which events contribute to model retraining

```
ml_prediction + clinical_features + sqi_result
                    │
                    ▼
         ┌──────────────────────┐
         │  RhythmOrchestrator  │
         │      .decide()       │
         └──────────┬───────────┘
                    │
         ┌──────────▼───────────────────────────┐
         │  Step 1: SQI gate — mark UNRELIABLE? │
         └──────────┬───────────────────────────┘
                    │
         ┌──────────▼──────────────────────────────┐
         │  Step 2: Background rhythm detection     │
         │  (HR + P-wave + QRS width → Sinus/Brady/ │
         │   Tachy/Junctional/Idioventricular)      │
         └──────────┬──────────────────────────────┘
                    │
         ┌──────────▼──────────────────────────────┐
         │  Step 3A: Rule-derived events            │
         │  - Pause (RR > 2000 ms)                 │
         │  - AF safety net (RR std + P-absent)     │
         │  - Atrial Flutter (spectral FFT)         │
         └──────────┬──────────────────────────────┘
                    │
         ┌──────────▼──────────────────────────────┐
         │  Step 3B: ML rhythm event (if confident) │
         │  Thresholds: VF=0.90, VT=0.88, AF=0.85  │
         └──────────┬──────────────────────────────┘
                    │
         ┌──────────▼──────────────────────────────┐
         │  Step 3C: Per-beat ectopy events         │
         │  Threshold: conf ≥ 0.97                  │
         │  + Rhythm trust gate (0.99 if sinus)     │
         │  + Density gate (>60% ectopy → suppress) │
         └──────────┬──────────────────────────────┘
                    │
         ┌──────────▼──────────────────────────────┐
         │  Step 4: apply_ectopy_patterns()         │
         │  - PVC/PAC correction (QRS + V3 scores)  │
         │  - Bigeminy / Trigeminy / Quadrigeminy   │
         │  - Couplet / Ventricular Run             │
         │  - NSVT / VT / SVT / PSVT               │
         └──────────┬──────────────────────────────┘
                    │
         ┌──────────▼──────────────────────────────┐
         │  Step 5: Background rhythm promotion     │
         │  VF > VT > AF/AFL > NSVT (hierarchy)    │
         └──────────┬──────────────────────────────┘
                    │
         ┌──────────▼──────────────────────────────┐
         │  Step 6: apply_display_rules()           │
         │  → final_display_events (sorted by prio) │
         └──────────┬──────────────────────────────┘
                    │
         ┌──────────▼──────────────────────────────┐
         │  Step 7: apply_training_flags()          │
         │  → which events mark used_for_training   │
         └──────────────────────────────────────────┘
                    │
                    ▼
             SegmentDecision object
             (background_rhythm, events, final_display_events)
```

---

## 2. Background Rhythm Detection (Rule-Based)

The background rhythm is determined BEFORE model events, using only HR + clinical features.
This ensures there is always a sensible "baseline" even if the model fails.

```
HR < 60:
  p_wave_present_ratio < 0.2 AND qrs_duration_ms > 120 → Idioventricular Rhythm
  p_wave_present_ratio < 0.3                            → Junctional Rhythm
  else                                                  → Sinus Bradycardia

HR > 100:
  → Sinus Tachycardia

HR 60–100:
  p_wave_present_ratio < 0.2 → Junctional Rhythm (normal rate, no P-waves)
  else                       → Sinus Rhythm
```

**Why rule-based background**: The model might fail on edge cases, but HR + P-wave is
always derivable from clinical features. The background gives the annotator a sane starting
point even if ML is wrong.

---

## 3. Rule-Derived Events (derive_rule_events)

These events fire from explicit clinical rules, independent of the ML model.

### 3.1 Pause Detection
```
Criterion: any RR interval > 2000 ms
Event:     Pause (category=RHYTHM, priority=85, used_for_training=False)
Rationale: RR > 2s is clinically significant pause regardless of HR or rhythm
```

### 3.2 AF Safety Net
```
Criteria: 
  - len(rr_intervals) >= 4
  - std(RR) > 160 ms       ← irregularly irregular
  - p_wave_present_ratio < 0.40  ← absent P-waves

Event: Atrial Fibrillation (category=RHYTHM, priority=88, used_for_training=False)
Why rule-based backup exists:
  AF is the most important arrhythmia to detect. When only 5 corrected AF segments
  were in the training DB, the ML model could not learn AF reliably. The rule acts
  as a safety net: even if ML misses AF, the rule catches it.
  
Electrophysiology basis:
  - AF produces irregularly irregular RR intervals (high RR standard deviation)
  - AF has no organized atrial activity → P-waves absent
  - 160 ms RR std chosen from clinical literature (Rizos et al., 2012)
```

### 3.3 Atrial Flutter (Spectral Detection)
```
Criteria:
  - mean HR between 130–175 bpm (typical 2:1 AFL conduction)
  - FFT of QRS-blanked signal has dominant peak at 4–6 Hz
    (= atrial rate 240–360 bpm, typical AFL range)
  - Peak power > 2.5× median spectral power

Event: Atrial Flutter (category=RHYTHM, priority=87, used_for_training=False)
Method:
  QRS blanking: zero out ±100 ms around each R-peak in signal
  Then FFT to reveal atrial baseline oscillation without QRS contamination
  
Why spectral detection:
  AFL flutter waves are often too small to detect as individual P-waves.
  But they produce a very regular, high-frequency oscillation (sawtooth) that
  is dominant in the FFT after QRS removal.
```

---

## 4. ML-Derived Events

### 4.1 Confidence Thresholds

Each ML rhythm label fires an event only if model confidence exceeds a per-class threshold.

```
Label                           Threshold  Rationale
───────────────────────────────────────────────────────────────────────
Ventricular Fibrillation          0.90     Critical: false alarm is catastrophic
VT / Ventricular Tachycardia      0.88     Critical: requires urgent response
NSVT                              0.85     High significance
Atrial Fibrillation / AF          0.85     Common but high FP rate at lower thresholds
Atrial Flutter                    0.85     AFL shares features with AF
3rd Degree AV Block               0.85     Complete block — pacemaker decision
2nd Degree AV Block Type 2        0.85     High risk of progression to 3rd degree
2nd Degree AV Block Type 1        0.82     Lower risk Wenckebach
1st Degree AV Block               0.80     Informational only
Bundle Branch Block               0.80     Chronic finding, lower urgency
Sinus Bradycardia                 0.75     Benign; model is highly accurate on this
Sinus Rhythm / Unknown            (never fire an event — these are normal)
```

**Why per-class thresholds instead of one global threshold**:
A global 0.80 threshold is too low for VF (0.80 VF confidence can still be wrong 20% of the time — at 10 patients/day that is 2 false VF alarms per day, unacceptable). But 0.80 for Sinus Brady is fine (benign, annotator can ignore a false alarm easily).

### 4.2 Per-Beat Ectopy Confidence Gates

The ectopy model produces per-beat predictions with 3-layer gating:

**Layer 1 — Base threshold**: `conf >= 0.97`
- Raises from default 0.95 to cut false-positive PVCs in normal sinus rhythms
- Only beats the model is very sure about pass through

**Layer 2 — Rhythm trust gate**:
```
If rhythm label is Sinus/Brady/Tachy/Unknown AND rhythm confidence > 0.65:
  If candidate beat count < 3:
    Require conf >= 0.99 instead of 0.97
```
Rationale: A high-confidence Sinus Rhythm prediction contradicts having multiple PVCs.
1–2 isolated ectopic beat predictions in a confidently-sinus segment are likely false
positives (hallucinations). Require 0.99 to override the sinus context.
Exception: 3+ ectopic beats are allowed through (a genuine bigeminy in a sinus background
is clinically plausible).

**Layer 3 — Density gate**:
```
If rhythm = Sinus AND rhythm confidence > 0.70:
  If (candidate_beats / total_beats) > 0.60:
    Check for consecutive run (3+ sequential beat indices)
    If no consecutive run → SUPPRESS ALL candidates (scattered hallucination)
```
Rationale: Genuine isolated ectopy in sinus rhythm is < 30% of beats.
If > 60% of beats are flagged, the ectopy model is confused. Exception: a real
VT run can have 100% consecutive ectopic beats — the consecutive-run check protects
genuine runs from being suppressed.

---

## 5. Ectopy Pattern Recognition (apply_ectopy_patterns)

### 5.1 PVC/PAC Label Correction (3-Stage)

Before pattern clustering, individual PVC/PAC labels are validated:

**Stage 1 — Hard QRS width rule**:
```
PVC event AND qrs_duration_ms < 80 ms → relabel to PAC
  (QRS < 80ms is physiologically impossible for ventricular origin)

PAC event AND qrs_duration_ms > 150 ms → relabel to PVC
  (QRS > 150ms is too wide for atrial origin; must be ventricular)
```

**Stage 2 — Ambiguous width (80–150 ms) + V3 discriminator scores**:
```
If pvc_score_mean available (from V3 beat discriminators):
  if pvc_score > pac_score + 0.15 → relabel to PVC
  if pac_score > pvc_score + 0.15 → relabel to PAC
  else: keep ML label (scores too close)

pvc_score_mean and pac_score_mean are medians across all beats, computed from:
  PVC criteria: wide QRS (+3.0/+2.0), no P (+2.5), compensatory pause (+2.0),
                T discordant (+2.0), short coupling (+1.0), neg polarity (+1.0)
  PAC criteria: narrow QRS (+2.5), inverted/biphasic P (+3.0),
                non-compensatory (+1.5), short coupling (+1.5), concordant T (+1.0)
```

**Stage 3 — Compensatory pause fallback** (when V3 scores unavailable):
```
Beat before: RR_before
Beat after:  RR_after
Normal RR:   median(all RR intervals)

RR_before + RR_after >= 1.85 × normal_RR → FULL compensatory → PVC
  (SA node NOT reset — ventricular origin)
RR_before + RR_after <= 1.60 × normal_RR → INCOMPLETE compensatory → PAC
  (SA node IS reset — atrial origin)
1.60 < ratio < 1.85 → ambiguous → keep ML label
```

### 5.2 Clustering Logic

After label correction, same-type ectopy events within 2 seconds of each other are grouped:
```
MAX_GAP = 2.0 seconds between consecutive ectopy events to remain in same cluster
```

### 5.3 Pattern Detection from Beat Indices

Using the sequential beat index (`beat_idx`) of each ectopy event:
```
diffs = diff(sorted beat indices)

Bigeminy:    all diffs == 2  (every other beat is ectopic)
Trigeminy:   all diffs == 3  (every third beat is ectopic)
Quadrigeminy: all diffs == 4 (every fourth beat is ectopic)

Minimum 2 cycles (3+ beats) required for Bigeminy/Trigeminy/Quadrigeminy
  (cannot confirm a pattern from a single cycle)

Consecutive:  all diffs == 1 (beats are sequential — a "run")
```

If beat indices are unavailable (events without beat_indices), time-gap CV is used
as a fallback, but Bigeminy/Trigeminy/Quadrigeminy are NEVER detected without indices
(cannot distinguish these patterns from time-based data alone).

### 5.4 PVC Pattern Count Rules

```
Consecutive PVCs in cluster:
  Count  Rate        Result        Priority  Training
  ─────────────────────────────────────────────────────
  ≥ 11   ≥ 100 bpm  VT            100       No (rules-only)
         [if QRS < 110ms → SVT instead — cannot be ventricular if narrow complex]
  4–10   ≥ 100 bpm  NSVT           90       No (rules-only)
         [if QRS < 110ms → PSVT instead]
  3       any        Ventricular   40        No (rules-only)
                     Run
  2       any        PVC Couplet   30        Yes

Interspersed PVCs:
  Bigeminy   (diff=2)  → PVC Bigeminy   55  Yes
  Trigeminy  (diff=3)  → PVC Trigeminy  55  Yes
  Quadrigeminy (diff=4) → PVC Quadrigeminy 55 Yes
```

### 5.5 PAC Pattern Count Rules

```
Consecutive PACs in cluster:
  Count  Rate        Result        Priority  Training
  ─────────────────────────────────────────────────────
  ≥ 11   ≥ 100 bpm  SVT            80       No (rules-only)
  6–10   ≥ 100 bpm  PSVT           85       No (rules-only)
  3–5    any         Atrial Run     40       No (rules-only)
  2      any         Atrial Couplet 30       Yes

Interspersed PACs:
  Bigeminy   (diff=2)  → PAC Bigeminy   55  Yes
  Trigeminy  (diff=3)  → PAC Trigeminy  55  Yes
  Quadrigeminy (diff=4) → PAC Quadrigeminy 55 Yes
```

---

## 6. Background Rhythm Promotion

After ectopy patterns are resolved, high-priority events can upgrade the background rhythm:

```
Priority (highest wins):
  1. Ventricular Fibrillation → background = "Ventricular Fibrillation"
  2. VT / Ventricular Tachycardia → background = "Ventricular Tachycardia"
  3. AF / Atrial Fibrillation / Atrial Flutter → background = event_type (e.g. "Atrial Flutter")
  4. NSVT / Ventricular Run → background = event_type (e.g. "NSVT")
  5. Otherwise: background stays as rule-based detection from Step 2
```

---

## 7. Display Rules (apply_display_rules)

Not all events are shown — a hierarchy determines what is displayed:

### Rule A — Life-Threatening Always Show
```
event.priority >= 95 → ALWAYS DISPLAYED
(VF events have priority 100; VT events 95)
```

### Rule B — AF Dominance
```
When any AF/Atrial Flutter event is present:
  - AF/Flutter events: DISPLAYED
  - Ectopy events (PVC/PAC): DISPLAYED (concurrent findings, important)
  - Other RHYTHM events: HIDDEN (suppressed by "AF Dominance")
  
Reason: You can have PVCs on top of AF — both are clinically important
and both should be shown. But you cannot also show "Junctional Rhythm"
at the same time as AF — only one background rhythm makes sense.
```

### Rule C — Run Dominance
```
When SVT/PSVT/Atrial Run is present:
  Individual PAC events → HIDDEN ("SVT/PSVT Dominance")
  
When VT/NSVT/Ventricular Run is present:
  Individual PVC events → HIDDEN ("VT/NSVT Dominance")
  
Reason: Showing "PVC, PVC, PVC, PVC... VT" is redundant. The VT event
already communicates there are multiple PVCs. Individual markers clutter.
```

### Rule D — Background Suppression
```
"Sinus Rhythm" events (from rule derivation, not cardiologist annotation):
  → HIDDEN (it's the default/background — no need to show)
  
Exception: if annotation_source == "cardiologist" → SHOW
  (cardiologist explicitly confirmed normal sinus — show as confirmation)
```

### Rule E — Artifact Suppression
```
If ANY non-artifact, non-sinus event is displayed:
  → Artifact events HIDDEN
  
If NO other events:
  → Artifact events DISPLAYED (the only finding)
  
Reason: Artifact is irrelevant when there's a real arrhythmia. But if the
whole segment is artifact, showing "Artifact" is the correct output.
```

Final display list is sorted by priority (highest first).

---

## 8. Training Flags (apply_training_flags)

After display arbitration, training flags are set for model retraining:

### Events That Train the Ectopy Model (used_for_training = True)
```
PAC, Atrial Couplet, PAC Bigeminy, PAC Trigeminy, PAC Quadrigeminy
PVC, PVC Couplet, PVC Bigeminy, PVC Trigeminy, PVC Quadrigeminy
```

### Events That Train the Rhythm Model (used_for_training = True)
```
AF, Atrial Fibrillation, Atrial Flutter
Ventricular Fibrillation
VT, Ventricular Tachycardia, NSVT
1st/2nd/3rd Degree AV Block
Bundle Branch Block
```

### Events That NEVER Train (used_for_training = False)
```
Sinus Rhythm, Sinus Bradycardia, Sinus Tachycardia → avoid baseline bias
Artifact → confounds training
SVT, PSVT → rules-only derived; no model class
Ventricular Run, Atrial Run → rules-only derived; no model class
AF Safety Net, AFL Spectral → rule-derived events, not cardiologist-verified
Pause → rule-derived
```

Rule-derived events are never trained because:
1. They are generated by rules that never make mistakes (by definition)
2. Training the model on rule outputs would create a circular dependency
3. Only cardiologist-annotated labels (via `is_corrected=TRUE`) are training ground truth

---

## 9. Priority Reference Table

| Event Type | Priority | Category | Displayed | Trains |
|------------|----------|----------|-----------|--------|
| Ventricular Fibrillation | 100 | RHYTHM | Always | Yes (cardiologist-annotated) |
| Ventricular Tachycardia (ML) | 95 | RHYTHM | Always | Yes (if annotated) |
| Pause (rule) | 85 | RHYTHM | Yes | No |
| NSVT (ML) | 90 | RHYTHM | Yes (unless VT present) | Yes (if annotated) |
| SVT (rule) | 80 | RHYTHM | Yes | No |
| PSVT (rule) | 85 | RHYTHM | Yes | No |
| Atrial Fibrillation (ML+rule) | 88 | RHYTHM | Yes | Yes (if annotated) |
| Atrial Flutter (ML+rule) | 87 | RHYTHM | Yes | Yes (if annotated) |
| AV Block (ML) | 70 | RHYTHM | Yes | Yes (if annotated) |
| PVC Bigeminy / Trigeminy | 55 | RHYTHM | Yes | Yes |
| PAC Bigeminy / Trigeminy | 55 | RHYTHM | Yes | Yes |
| Ventricular Run | 40 | RHYTHM | Yes | No |
| Atrial Run | 40 | RHYTHM | Yes | No |
| PVC Couplet | 30 | ECTOPY | Yes | Yes |
| Atrial Couplet | 30 | ECTOPY | Yes | Yes |
| PVC (individual) | 10 | ECTOPY | Yes (unless VT/Run) | Yes |
| PAC (individual) | 10 | ECTOPY | Yes (unless SVT/Run) | Yes |
| Sinus Bradycardia (ML) | 20 | RHYTHM | Hidden (background) | Yes (if annotated) |
| Artifact (rule) | 0 | RHYTHM | Only if nothing else | No |
