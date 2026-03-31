# Output Logic — Simplified Decision System

## Core Principle

The output has **two independent channels**, plus a **rule override layer**:

```
┌─────────────────────────────────────────────────┐
│             10-Second ECG Segment                │
│                                                  │
│   ┌──────────────┐     ┌──────────────┐         │
│   │ Rhythm Model │     │ Ectopy Model │         │
│   │ (17 classes) │     │ (per-beat)   │         │
│   └──────┬───────┘     └──────┬───────┘         │
│          │                    │                   │
│          ▼                    ▼                   │
│   Rhythm Output         Ectopy Output            │
│   (by confidence)       (PVC/PAC/None)           │
│          │                    │                   │
│          └────────┬───────────┘                   │
│                   │                               │
│                   ▼                               │
│          ┌────────────────┐                       │
│          │  Rules Engine  │                       │
│          │  (overrides)   │                       │
│          └────────┬───────┘                       │
│                   │                               │
│                   ▼                               │
│            FINAL OUTPUT                           │
└─────────────────────────────────────────────────┘
```

---

## The Simple Logic

| Step | Logic |
|------|-------|
| **1** | Run Rhythm Model → take the class with **highest confidence** |
| **2** | Run Ectopy Model on each beat → label each beat as PVC, PAC, or None |
| **3** | Run Rules Engine on clinical features + ectopy beats |
| **4** | If a **rule fires** → that rule's output overrides the rhythm label |
| **5** | Display: Rhythm output + Ectopy output (if any) + Rule output (if any) |

---

## ALL Output Combinations

### CASE 1: Rhythm Only (Ectopy = None, No Rules)

**What happened:** Rhythm model gives a prediction, ectopy model says all beats are normal, no rule conditions met.

| Rhythm Model | Ectopy Model | Rule | Final Output |
|-------------|--------------|------|--------------|
| Sinus Rhythm (0.92) | None | — | **Rhythm: Sinus Rhythm** |
| Sinus Bradycardia (0.88) | None | — | **Rhythm: Sinus Bradycardia** |
| Sinus Tachycardia (0.85) | None | — | **Rhythm: Sinus Tachycardia** |
| Atrial Fibrillation (0.78) | None | — | **Rhythm: Atrial Fibrillation** |
| Junctional Rhythm (0.71) | None | — | **Rhythm: Junctional Rhythm** |

**Display:** Only the rhythm label. Nothing else.

---

### CASE 2: Rhythm + Ectopy (No Rules)

**What happened:** Rhythm model gives a prediction, ectopy model detected beats, but no rule condition met (not enough consecutive beats for patterns).

| Rhythm Model | Ectopy Model | Rule | Final Output |
|-------------|--------------|------|--------------|
| Sinus Rhythm (0.91) | 1 PVC at beat #4 | — | **Rhythm: Sinus Rhythm, Ectopy: PVC** |
| Sinus Rhythm (0.89) | 1 PAC at beat #7 | — | **Rhythm: Sinus Rhythm, Ectopy: PAC** |
| Sinus Rhythm (0.87) | PVC at #3, PAC at #8 | — | **Rhythm: Sinus Rhythm, Ectopy: PVC + PAC** |
| Sinus Tachycardia (0.83) | 1 PVC at beat #5 | — | **Rhythm: Sinus Tachycardia, Ectopy: PVC** |
| Atrial Fibrillation (0.76) | 1 PVC at beat #2 | — | **Rhythm: Atrial Fibrillation, Ectopy: PVC** |

**Display:** Both rhythm and ectopy labels, with beat positions marked on ECG.

---

### CASE 3: Rule Fires (Overrides Rhythm)

**What happened:** Clinical features or ectopy patterns trigger a rule. The rule output becomes the primary diagnosis.

#### 3A. Rules from Clinical Features (no ectopy needed)

| Rhythm Model | Ectopy Model | Rule Fired | Final Output |
|-------------|--------------|------------|--------------|
| Sinus Rhythm (0.90) | None | AF Rule (CV>0.15, no P-waves) | **Rhythm: AF (Rule)** |
| Sinus Rhythm (0.88) | None | 1st Degree AV Block (PR>200ms) | **Rhythm: 1st Degree AV Block (Rule)** |
| Sinus Rhythm (0.85) | None | Pause (RR>2000ms) | **Rhythm: Sinus Rhythm, Event: Pause (Rule)** |

**Why Pause is different:** Pause is an *event*, not a rhythm. The rhythm is still Sinus — there's just a long gap between two beats.

#### 3B. Rules from Ectopy Patterns (ectopy model + pattern rules)

| Rhythm Model | Ectopy Model | Rule Fired | Final Output |
|-------------|--------------|------------|--------------|
| Sinus Rhythm (0.91) | PVC at beats #2, #4 | PVC Couplet (2 consecutive) | **Rhythm: Sinus Rhythm, Ectopy: PVC Couplet (Rule)** |
| Sinus Rhythm (0.89) | PVC at beats #1,#3,#5 | PVC Bigeminy (every 2nd beat) | **Rhythm: PVC Bigeminy (Rule)** |
| Sinus Rhythm (0.87) | PVC at beats #1,#4,#7 | PVC Trigeminy (every 3rd beat) | **Rhythm: PVC Trigeminy (Rule)** |
| Sinus Rhythm (0.92) | PVC at beats #3,#4,#5 | Ventricular Run (3 consecutive) | **Rhythm: Ventricular Run (Rule)** |
| Sinus Rhythm (0.88) | PVC at beats #2,#3,#4,#5,#6 | NSVT (4-10 consecutive PVCs) | **Rhythm: NSVT (Rule)** |
| Sinus Rhythm (0.85) | PVC ×11+ consecutive | VT (11+ consecutive PVCs) | **Rhythm: VT (Rule)** |
| Sinus Rhythm (0.90) | PAC at beats #2, #4 | Atrial Couplet (2 consecutive) | **Rhythm: Sinus Rhythm, Ectopy: Atrial Couplet (Rule)** |
| Sinus Rhythm (0.88) | PAC at beats #1,#3,#5 | PAC Bigeminy (every 2nd beat) | **Rhythm: PAC Bigeminy (Rule)** |
| Sinus Rhythm (0.91) | PAC at beats #3,#4,#5 | Atrial Run (3-5 consecutive PACs) | **Rhythm: Atrial Run (Rule)** |
| Sinus Rhythm (0.87) | PAC ×6-10 consecutive | PSVT (6-10 consecutive PACs) | **Rhythm: PSVT (Rule)** |
| Sinus Rhythm (0.83) | PAC ×11+ consecutive | SVT (11+ consecutive PACs) | **Rhythm: SVT (Rule)** |

---

### CASE 4: ML Rhythm + Rule Both Agree

| Rhythm Model | Ectopy Model | Rule Fired | Final Output |
|-------------|--------------|------------|--------------|
| Atrial Fibrillation (0.82) | None | AF Rule (CV>0.15) | **Rhythm: Atrial Fibrillation (ML + Rule agree)** |

**Strongest case:** Both the ML model and the clinical rule independently agree. High confidence in diagnosis.

---

### CASE 5: ML Rhythm + Rule Disagree

| Rhythm Model | Ectopy Model | Rule Fired | Who Wins? | Final Output |
|-------------|--------------|------------|-----------|--------------|
| Sinus Rhythm (0.90) | None | AF Rule fires | **Rule wins** | **Rhythm: AF (Rule)** |
| Atrial Fibrillation (0.65) | None | No AF rule (CV<0.15) | **ML wins** (no rule to override) | **Rhythm: Atrial Fibrillation (ML, conf: 0.65)** |

**Simple principle: If a rule fires, rule wins. If no rule fires, ML confidence determines the output.**

---

### CASE 6: Multiple Rules Fire Simultaneously

| Rhythm Model | What Happened | Rules Fired | Final Output |
|-------------|---------------|-------------|--------------|
| Sinus (0.88) | PVCs in bigeminy + PR>200ms | PVC Bigeminy + 1st Degree AV Block | **Rhythm: PVC Bigeminy (Rule), Event: 1st Degree AV Block (Rule)** |
| Sinus (0.85) | AF features + RR>2s gap | AF + Pause | **Rhythm: AF (Rule), Event: Pause (Rule)** |
| Sinus (0.90) | PVC bigeminy + PAC at #9 | PVC Bigeminy | **Rhythm: PVC Bigeminy (Rule), Ectopy: PAC** |

**When multiple rules fire:** Show all of them. The rhythm-type rule becomes the primary diagnosis. Event-type rules (Pause) and ectopy are shown alongside.

---

### CASE 7: Bad Signal (Artifact)

| Rhythm Model | Ectopy Model | SQI Check | Final Output |
|-------------|--------------|-----------|--------------|
| (not trusted) | (not trusted) | FAILED | **Artifact — Signal Unreliable** |

**When SQI fails:** Nothing from ML is trusted. Just show "Artifact".

---

## Summary Decision Table

| # | Rhythm Model Says | Ectopy Model Says | Rule Fires? | What Gets Displayed |
|---|-------------------|-------------------|-------------|---------------------|
| 1 | Sinus Rhythm | None | No | Rhythm: Sinus Rhythm |
| 2 | Sinus Rhythm | PVC/PAC (isolated) | No | Rhythm: Sinus Rhythm + Ectopy: PVC/PAC |
| 3 | Sinus Rhythm | None | AF Rule | Rhythm: AF (Rule overrides) |
| 4 | Sinus Rhythm | None | AV Block Rule | Rhythm: 1st Degree AV Block (Rule overrides) |
| 5 | Sinus Rhythm | None | Pause Rule | Rhythm: Sinus Rhythm + Event: Pause |
| 6 | Sinus Rhythm | 2 consecutive PVCs | Couplet Rule | Rhythm: Sinus Rhythm + Ectopy: PVC Couplet |
| 7 | Sinus Rhythm | PVCs every 2nd beat | Bigeminy Rule | Rhythm: PVC Bigeminy (Rule overrides) |
| 8 | Sinus Rhythm | PVCs every 3rd beat | Trigeminy Rule | Rhythm: PVC Trigeminy (Rule overrides) |
| 9 | Sinus Rhythm | 3 consecutive PVCs | Ventricular Run | Rhythm: Ventricular Run (Rule overrides) |
| 10 | Sinus Rhythm | 4-10 consecutive PVCs | NSVT Rule | Rhythm: NSVT (Rule overrides) |
| 11 | Sinus Rhythm | 11+ consecutive PVCs | VT Rule | Rhythm: VT (Rule overrides) |
| 12 | Sinus Rhythm | PACs every 2nd beat | PAC Bigeminy Rule | Rhythm: PAC Bigeminy (Rule overrides) |
| 13 | Sinus Rhythm | 3-5 consecutive PACs | Atrial Run | Rhythm: Atrial Run (Rule overrides) |
| 14 | Sinus Rhythm | 6-10 consecutive PACs | PSVT Rule | Rhythm: PSVT (Rule overrides) |
| 15 | Sinus Rhythm | 11+ consecutive PACs | SVT Rule | Rhythm: SVT (Rule overrides) |
| 16 | AF (ML, 0.82) | None | AF Rule also fires | Rhythm: AF (ML + Rule agree) |
| 17 | AF (ML, 0.82) | PVC detected | AF Rule fires | Rhythm: AF + Ectopy: PVC |
| 18 | AF (ML, 0.65) | None | No rule | Rhythm: AF (ML only, conf: 0.65) |
| 19 | VT (ML, 0.70) | None | No rule | Rhythm: VT (ML only, conf: 0.70) |
| 20 | Any | Any | SQI Fail | Artifact — Signal Unreliable |

---

## What This Means for Priority

**Priority number is NO LONGER NEEDED.** The logic is now:

1. **Rule fires?** → Rule output is the answer (deterministic, no confidence score needed)
2. **No rule?** → ML model's highest confidence class is the answer
3. **Ectopy detected?** → Always shown alongside, regardless of rhythm
4. **Everything detected is displayed** — no suppression, no hiding

The only ordering needed is: **Rules > ML** for the primary diagnosis. That's it.
