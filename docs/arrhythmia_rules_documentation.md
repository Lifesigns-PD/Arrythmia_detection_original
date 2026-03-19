# Arrhythmia Detection Rules — Complete Documentation

## System Architecture

The system uses a **two-layer approach**: ML models predict base arrhythmia types, and a rule-based decision engine refines those predictions using clinical features and beat-level patterns.

```
Raw ECG Signal
    │
    ├──► ML Rhythm Model (17 classes) ──► Base rhythm label + confidence
    ├──► ML Ectopy Model (4 classes)  ──► Per-beat PVC/PAC/Run/None
    │
    └──► Clinical Feature Extraction   ──► HR, PR interval, QRS width, RR intervals
            │
            ▼
    ┌─────────────────────────────┐
    │   RhythmOrchestrator.decide()   │
    │                                 │
    │  1. Background rhythm (HR)      │
    │  2. Rule-derived events         │
    │  3. ML-derived events           │
    │  4. Ectopy pattern detection    │
    │  5. Display arbitration         │
    │  6. Training flag assignment    │
    └─────────────────────────────┘
            │
            ▼
    Final SegmentDecision (events + background rhythm + display list)
```

---

## 1. Background Rhythm Detection

**File:** `rhythm_orchestrator.py` → `_detect_background_rhythm()`

Determines the sinus background from heart rate alone. This is always set regardless of other findings.

| Heart Rate (bpm) | Background Rhythm    |
|-------------------|----------------------|
| 0 (no signal)     | Unknown              |
| < 60              | Sinus Bradycardia    |
| 60–100            | Sinus Rhythm         |
| 100–150           | Sinus Tachycardia    |
| > 150             | Sinus Tachycardia    |

**Note:** If Atrial Fibrillation is detected later (step 4), the background rhythm is overridden to "Atrial Fibrillation".

---

## 2. Rule-Derived Rhythm Events

**File:** `rules.py` → `derive_rule_events()`

These rules fire based on clinical features extracted from the signal (HR, PR interval, QRS duration, RR variability). They produce RHYTHM-category events.

### 2.1 Atrial Fibrillation (AF)

| Parameter         | Threshold       |
|-------------------|-----------------|
| RR intervals      | > 3 beats available |
| CV (RR variability) | > 0.15        |
| PR interval       | < 10 ms (absent P-waves) |

- **Priority:** 90
- **Training flag:** Yes
- **Logic:** Irregular rhythm + absent P-waves = AF

### 2.2 Supraventricular Tachycardia (SVT)

| Parameter         | Threshold       |
|-------------------|-----------------|
| Heart Rate        | > 130 bpm       |
| QRS duration      | < 120 ms (narrow) |
| CV (RR variability) | < 0.08 (regular) |
| AF detected?      | No (mutually exclusive) |

- **Priority:** 80
- **Training flag:** Yes
- **Logic:** Fast + regular + narrow QRS + not AF = SVT

### 2.3 Ventricular Tachycardia (VT)

| Parameter         | Threshold       |
|-------------------|-----------------|
| Heart Rate        | > 100 bpm       |
| QRS duration      | >= 120 ms (wide) |

- **Priority:** 100
- **Training flag:** Yes
- **Logic:** Fast + wide QRS = VT

### 2.4 1st Degree AV Block

| Parameter         | Threshold       |
|-------------------|-----------------|
| PR interval       | > 200 ms        |

- **Priority:** 50
- **Training flag:** No (never train on AV block from rules)
- **Logic:** Prolonged PR = 1st degree block

### 2.5 Pause

| Parameter         | Threshold       |
|-------------------|-----------------|
| Any RR interval   | > 2000 ms (2 seconds) |

- **Priority:** 85
- **Training flag:** No (never train on pause from rules)
- **Logic:** Any single RR interval exceeding 2 seconds

---

## 3. ML-Derived Events

**File:** `rhythm_orchestrator.py` → `decide()` (steps 4B and 4C)

### 3.1 Rhythm Model Event

The ML rhythm model outputs one of 17 classes. If the prediction is NOT "Sinus Rhythm" or "Unknown", a RHYTHM event is created with the predicted label.

**Rhythm Model Classes (17):**

| Index | Class Name                    |
|-------|-------------------------------|
| 0     | Sinus Rhythm                  |
| 1     | Supraventricular Tachycardia  |
| 2     | Atrial Fibrillation           |
| 3     | Atrial Flutter                |
| 4     | Junctional Rhythm             |
| 5     | Idioventricular Rhythm        |
| 6     | Ventricular Tachycardia       |
| 7     | Ventricular Fibrillation      |
| 8     | 1st Degree AV Block           |
| 9     | 2nd Degree AV Block Type 1    |
| 10    | 2nd Degree AV Block Type 2    |
| 11    | 3rd Degree AV Block           |
| 12    | Bundle Branch Block           |
| 13    | Artifact                      |
| 14    | PSVT                          |
| 15    | Pause                         |
| 16    | Other Arrhythmia              |

### 3.2 Ectopy Model (Per-Beat) Events

The ML ectopy model classifies each beat individually. Non-"None" beats generate ECTOPY events with beat index and timing.

**Ectopy Model Classes (4):**

| Index | Class Name |
|-------|------------|
| 0     | None       |
| 1     | PVC        |
| 2     | PAC        |
| 3     | Run        |

Each ectopy beat event has:
- `start_time`: beat peak - 0.3s
- `end_time`: beat peak + 0.3s
- `beat_indices`: [sequential beat index] — used for pattern detection
- **Priority:** 10

---

## 4. Ectopy Pattern Recognition (Rules)

**File:** `rules.py` → `apply_ectopy_patterns()`

After ML produces per-beat ectopy events, this rule engine clusters them and detects clinical patterns. Runs separately for PVC and PAC event types.

### Clustering

Events of the same type (PVC or PAC) are grouped into clusters with a maximum inter-event gap of **2.0 seconds**.

### 4.1 Bigeminy

| Condition               | Requirement                    |
|-------------------------|--------------------------------|
| Beat indices available  | Yes (all events in cluster)    |
| Index differences       | All exactly 2 (every other beat) |
| Minimum beats           | 3 (at least 2 cycles)         |

- **Output:** "PVC Bigeminy" or "PAC Bigeminy"
- **Category:** RHYTHM
- **Priority:** 55
- **Training flag:** Yes
- **Clinical meaning:** Ectopic beat alternating with normal beat (1:1 pattern)

### 4.2 Trigeminy

| Condition               | Requirement                    |
|-------------------------|--------------------------------|
| Beat indices available  | Yes                            |
| Index differences       | All exactly 3 (every 3rd beat) |
| Minimum beats           | 3 (at least 2 cycles)         |

- **Output:** "PVC Trigeminy" or "PAC Trigeminy"
- **Category:** RHYTHM
- **Priority:** 55
- **Training flag:** Yes
- **Clinical meaning:** Ectopic beat every 3rd beat (2:1 pattern)

### 4.3 Quadrigeminy

| Condition               | Requirement                    |
|-------------------------|--------------------------------|
| Beat indices available  | Yes                            |
| Index differences       | All exactly 4 (every 4th beat) |
| Minimum beats           | 3 (at least 2 cycles)         |

- **Output:** "PVC Quadrigeminy" or "PAC Quadrigeminy"
- **Category:** RHYTHM
- **Priority:** 55
- **Training flag:** Yes
- **Clinical meaning:** Ectopic beat every 4th beat (3:1 pattern)

### 4.4 NSVT (Non-Sustained Ventricular Tachycardia)

| Condition               | Requirement                    |
|-------------------------|--------------------------------|
| Beat type               | PVC                            |
| Consecutive beats       | > 3 (4 or more)               |
| Rate                    | >= 100 bpm                     |

- **Output:** "NSVT"
- **Category:** RHYTHM
- **Priority:** 90
- **Training flag:** Yes
- **Clinical meaning:** Short burst of rapid ventricular beats

### 4.5 PSVT (Paroxysmal Supraventricular Tachycardia)

| Condition               | Requirement                    |
|-------------------------|--------------------------------|
| Beat type               | PAC                            |
| Consecutive beats       | > 3 (4 or more)               |
| Rate                    | >= 100 bpm                     |

- **Output:** "PSVT"
- **Category:** RHYTHM
- **Priority:** 85
- **Training flag:** Yes
- **Clinical meaning:** Short burst of rapid supraventricular beats

### 4.6 Ventricular Triplet / Atrial Triplet

| Condition               | Requirement                    |
|-------------------------|--------------------------------|
| Consecutive beats       | Exactly 3                      |

- **Output:** "Ventricular Triplet" (PVC) or "Atrial Triplet" (PAC)
- **Category:** RHYTHM
- **Priority:** 40
- **Training flag:** Yes
- **Clinical meaning:** Three consecutive ectopic beats

### 4.7 Couplet

| Condition               | Requirement                    |
|-------------------------|--------------------------------|
| Consecutive beats       | Exactly 2                      |

- **Output:** "PVC Couplet" or "Atrial Couplet"
- **Category:** ECTOPY
- **Priority:** 30
- **Training flag:** Yes
- **Clinical meaning:** Two consecutive ectopic beats

### Pattern Detection: Beat Indices vs Time Fallback

| Method           | Can detect                                              |
|------------------|---------------------------------------------------------|
| Beat indices     | Bigeminy, Trigeminy, Quadrigeminy, Couplet, Triplet, NSVT/PSVT |
| Time fallback    | Couplet, Triplet, NSVT/PSVT only                       |

**Important:** Without beat indices, Bigeminy/Trigeminy/Quadrigeminy CANNOT be detected. The time-based fallback treats all regular clusters as consecutive runs to avoid false positives.

---

## 5. Display Arbitration Rules

**File:** `rules.py` → `apply_display_rules()`

After all events are gathered, display rules determine which events are shown to the cardiologist and which are suppressed.

### Priority Hierarchy

| Rule        | Condition                        | Action                              |
|-------------|----------------------------------|-------------------------------------|
| AI Audit    | Event is from AI (not cardiologist) | Always display (audit mode active) |
| Life-Threat | Priority >= 95                   | Always display                      |
| AF Dominance| AF present + event is RHYTHM     | Suppress other rhythms              |
| AF + Ectopy | AF present + event is ECTOPY     | Display (ectopy on top of AF)       |
| SVT Dominance | SVT/PSVT present + isolated PAC | Suppress individual PACs           |
| VT Dominance  | VT/NSVT present + isolated PVC  | Suppress individual PVCs           |
| Background  | Sinus variants (non-cardiologist) | Suppress (background only)         |
| Artifact    | Artifact + other events displayed | Suppress artifact                  |

### Event Priority Scale

| Priority | Event Types                                    |
|----------|------------------------------------------------|
| 100      | VT, Ventricular Fibrillation                   |
| 90       | AF, Atrial Fibrillation, NSVT                  |
| 85       | Pause, PSVT                                    |
| 80       | SVT                                            |
| 70       | AV Blocks                                      |
| 55       | Bigeminy, Trigeminy, Quadrigeminy              |
| 50       | Other ML predictions                           |
| 40       | Triplets                                       |
| 30       | Couplets                                       |
| 10       | Isolated PVC, PAC                              |
| 0        | Artifact                                       |

---

## 6. Training Flag Assignment

**File:** `rules.py` → `apply_training_flags()`

Determines which events are eligible to be used for model retraining.

### Events USED for training:

| Category       | Event Types                                                         |
|----------------|---------------------------------------------------------------------|
| Atrial         | PAC, PAC Couplet, Atrial Triplet, PSVT, PAC Bigeminy/Trigeminy/Quadrigeminy |
| AF/Flutter     | AF, Atrial Fibrillation, Atrial Flutter, SVT                       |
| Ventricular    | PVC, PVC Couplet, Ventricular Triplet, NSVT, PVC Bigeminy/Trigeminy/Quadrigeminy |
| VT/VF          | VT, Ventricular Tachycardia, Ventricular Fibrillation              |
| Blocks         | 1st Degree AV Block, 2nd Degree AV Block Type 1, 3rd Degree AV Block |

### Events NOT used for training:

| Event Type          | Reason                                    |
|---------------------|-------------------------------------------|
| Sinus Rhythm        | Baseline bias — too dominant in dataset    |
| Sinus Bradycardia   | Rate variant, not separate pathology       |
| Sinus Tachycardia   | Rate variant, not separate pathology       |
| Artifact            | Noise, not a clinical finding              |
| All other unlisted   | Unknown/unvalidated labels                |

---

## 7. Label Normalization for Training

**File:** `data_loader.py` → `normalize_label()`

Maps raw database labels to canonical class names before training.

### Key Mappings:

| Raw Label / Code          | Normalized To                  |
|---------------------------|--------------------------------|
| N, NORM, NSR              | Sinus Rhythm                   |
| AFIB, AF                  | Atrial Fibrillation            |
| AFLUT                     | Atrial Flutter                 |
| VT                        | Ventricular Tachycardia        |
| VF, VFIB                  | Ventricular Fibrillation       |
| SVT                       | Supraventricular Tachycardia   |
| V (single char)           | PVC                            |
| A (single char)           | PAC                            |
| F (single char)           | PVC (fusion grouped with PVC)  |
| Wenckebach                | 2nd Degree AV Block Type 1     |
| Mobitz                    | 2nd Degree AV Block Type 2     |
| *Bigeminy (contains)      | PVC Bigeminy                   |
| *Trigeminy (contains)     | PVC Trigeminy                  |
| Sinus Bradycardia         | Sinus Rhythm (via alias)       |
| Sinus Tachycardia         | Sinus Rhythm (via alias)       |
| Unrecognized label        | Other Arrhythmia (with warning)|

### Composite Labels ("+"):

Labels containing " + " (e.g., "Atrial Fibrillation + PVC") are split. The rhythm model takes the first part (base rhythm), the ectopy model handles the second part independently.

---

## 8. Complete Arrhythmia Summary Table

| Arrhythmia               | Detected By     | Rule Criteria                                     | Model Class Index |
|--------------------------|-----------------|---------------------------------------------------|-------------------|
| Sinus Rhythm             | Background rule | HR 60–100                                         | Rhythm: 0         |
| Sinus Bradycardia        | Background rule | HR < 60                                           | → maps to Rhythm: 0 |
| Sinus Tachycardia        | Background rule | HR > 100                                          | → maps to Rhythm: 0 |
| Atrial Fibrillation      | Rule + ML       | CV > 0.15, no P-waves / ML prediction             | Rhythm: 2         |
| Atrial Flutter           | ML only         | ML prediction                                     | Rhythm: 3         |
| SVT                      | Rule + ML       | HR > 130, narrow QRS, regular, not AF / ML        | Rhythm: 1         |
| PSVT                     | Ectopy rules    | 4+ consecutive PACs, rate >= 100                  | Rhythm: 14        |
| Ventricular Tachycardia  | Rule + ML       | HR > 100, wide QRS / ML prediction                | Rhythm: 6         |
| NSVT                     | Ectopy rules    | 4+ consecutive PVCs, rate >= 100                  | —                 |
| Ventricular Fibrillation | ML only         | ML prediction                                     | Rhythm: 7         |
| Junctional Rhythm        | ML only         | ML prediction                                     | Rhythm: 4         |
| Idioventricular Rhythm   | ML only         | ML prediction                                     | Rhythm: 5         |
| 1st Degree AV Block      | Rule + ML       | PR > 200 ms / ML prediction                       | Rhythm: 8         |
| 2nd Degree AV Block T1   | ML only         | ML prediction                                     | Rhythm: 9         |
| 2nd Degree AV Block T2   | ML only         | ML prediction                                     | Rhythm: 10        |
| 3rd Degree AV Block      | ML only         | ML prediction                                     | Rhythm: 11        |
| Bundle Branch Block      | ML only         | ML prediction                                     | Rhythm: 12        |
| PVC (isolated)           | ML ectopy       | Per-beat classification                           | Ectopy: 1         |
| PAC (isolated)           | ML ectopy       | Per-beat classification                           | Ectopy: 2         |
| PVC Bigeminy             | Ectopy rules    | Beat index diffs all = 2, min 3 beats             | —                 |
| PVC Trigeminy            | Ectopy rules    | Beat index diffs all = 3, min 3 beats             | —                 |
| PVC Quadrigeminy         | Ectopy rules    | Beat index diffs all = 4, min 3 beats             | —                 |
| PAC Bigeminy             | Ectopy rules    | Beat index diffs all = 2, min 3 beats             | —                 |
| PAC Trigeminy            | Ectopy rules    | Beat index diffs all = 3, min 3 beats             | —                 |
| PAC Quadrigeminy         | Ectopy rules    | Beat index diffs all = 4, min 3 beats             | —                 |
| PVC Couplet              | Ectopy rules    | 2 consecutive PVCs                                | —                 |
| Atrial Couplet           | Ectopy rules    | 2 consecutive PACs                                | —                 |
| Ventricular Triplet      | Ectopy rules    | 3 consecutive PVCs                                | —                 |
| Atrial Triplet           | Ectopy rules    | 3 consecutive PACs                                | —                 |
| Pause                    | Rule            | Any RR > 2000 ms                                  | Rhythm: 15        |
| Artifact                 | Orchestrator    | SQI check fails                                   | Rhythm: 13        |
| Other Arrhythmia         | ML / fallback   | Unrecognized labels                               | Rhythm: 16        |

---

## 9. Rule Overlap Analysis

The rules are designed to be **mutually exclusive** where possible:

| Potential Overlap       | Resolution                                          |
|-------------------------|-----------------------------------------------------|
| AF vs SVT               | SVT requires `not is_af` — explicit exclusion       |
| SVT vs VT               | SVT requires QRS < 120; VT requires QRS >= 120      |
| AF vs VT                | Can co-fire (different criteria, both valid)         |
| Bigeminy vs NSVT        | Bigeminy checked first; elif prevents NSVT overlap  |
| Bigeminy vs Couplet     | Bigeminy needs 3+ beats; Couplet needs exactly 2    |
| Triplet vs NSVT         | Triplet = exactly 3; NSVT = 4+ consecutive          |
| Rule AF vs ML AF        | Both can produce AF events; display arbiter deduplicates |
| Background rhythm vs AF | AF overrides background rhythm when detected         |
