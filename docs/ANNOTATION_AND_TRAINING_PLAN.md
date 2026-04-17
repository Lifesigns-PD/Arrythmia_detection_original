# Annotation and Training Plan
## Complete Guide: How Annotated Data Trains the ECG Models

---

## 1. The Core Principle

The models start knowing nothing. They improve only through cardiologist corrections.

```
Model makes a prediction
         │
         ▼
Cardiologist reviews in dashboard
         │
    ┌────┴────┐
  Wrong?    Correct?
    │           │
    ▼           ▼
 Correct     Confirm
 the label   (no change needed)
    │
    ▼
Saved to DB with is_corrected = TRUE
         │
         ▼
Retrain with all is_corrected segments
         │
         ▼
Model improves on that class
```

The critical gate is `is_corrected = TRUE`. Only segments that a cardiologist
has explicitly reviewed and approved are used for training. Unreviewed model
outputs are stored but never trained on — they may be wrong.

---

## 2. What the Annotator Sees in the Dashboard

### Display Elements
1. **Cleaned ECG strip** (10 seconds): V3-preprocessed signal with detected R-peaks marked
2. **Background rhythm**: model's rhythm prediction (e.g. "Atrial Fibrillation (87%)")
3. **Beat markers**: colored dots at each R-peak
   - Gray = Normal beat
   - Orange = PVC (model + rules detected)
   - Purple = PAC (model + rules detected)
4. **Pattern labels**: if rules detected bigeminy/couplet, these are shown as overlays
5. **XAI explanation**: top 3 contributing features for the rhythm prediction
6. **Annotation dropdown**: cardiologist selects the correct label

### Annotation Workflow
1. Review the ECG strip visually
2. If model output matches clinical assessment → save as-is (tick is_corrected)
3. If model is wrong → select correct label from dropdown
4. Optionally click individual beats to label specific beats
5. Press Save → data goes to DB with `is_corrected = TRUE`

---

## 3. Label Routing: What Trains Which Model

This is the most critical concept. Saving a label in the dashboard triggers
different training depending on the label type:

### Rhythm Labels → Rhythm Model
These labels train the **Rhythm model** (15 classes):

```
Sinus Rhythm          → class 0
Atrial Fibrillation   → class 1
Atrial Flutter        → class 2
Junctional Rhythm     → class 3
Idioventricular       → class 4
Ventricular Fibrillation → class 5
Ventricular Tachycardia → class 6
  └─ NSVT              → folded into class 6
1st Degree AV Block   → class 7
2nd Degree AV Block T1 → class 8
2nd Degree AV Block T2 → class 9
3rd Degree AV Block   → class 10
Bundle Branch Block   → class 11
Artifact              → class 12
Pause                 → class 13
Sinus Bradycardia     → class 14
```

### Ectopy Labels → Ectopy Model
These labels train the **Ectopy model** (3 classes):

```
PVC                   → class 1 (PVC)
PVC Bigeminy          → class 1 (PVC) — pattern collapses to beat type
PVC Trigeminy         → class 1 (PVC)
PVC Couplet           → class 1 (PVC)
Ventricular Run       → class 1 (PVC) — run = multiple consecutive PVCs
NSVT, VT             → class 1 (PVC) — ectopy model learns beat morphology

PAC                   → class 2 (PAC)
PAC Bigeminy          → class 2 (PAC)
PAC Trigeminy         → class 2 (PAC)
Atrial Couplet        → class 2 (PAC)
Atrial Run            → class 2 (PAC)
SVT, PSVT            → class 2 (PAC)
```

### Labels That Train BOTH Models
Combination labels are split:
```
"Sinus Rhythm + PVC" → Rhythm model: Sinus Rhythm (class 0)
                      → Ectopy model: PVC (class 1)

"AF + PVC"           → Rhythm model: Atrial Fibrillation (class 1)
                      → Ectopy model: PVC (class 1)

"Sinus Bradycardia"  → Rhythm model: Sinus Bradycardia (class 14)
                      → Ectopy model: None (class 0) — no ectopy
```

The split is handled by `get_rhythm_label_idx()` and `get_ectopy_label_idx()`
in `data_loader.py`. They each parse the label independently.

### Labels That Train NEITHER Model
These are excluded from training:
```
"Other Arrhythmia"   → No clinical definition; excluded
"SVT" / "PSVT"       → Rules-only output; excluded from rhythm model
```
SVT/PSVT are never trained because:
1. We don't have enough verified examples
2. The rule for SVT is: PAC rate > 150 bpm → `apply_ectopy_patterns()` detects it
3. The model would learn "high HR + PAC → SVT" which is already a rule

---

## 4. The `is_corrected` Gate

```sql
-- Training query (in retrain_v2.py)
SELECT *
FROM ecg_features_annotatable
WHERE is_corrected = TRUE   -- ← ONLY verified segments
```

**Why this gate exists**:
- Model outputs are stored in the DB as the initial `arrhythmia_label`
- Before a cardiologist reviews it, that label may be WRONG
- Training on wrong labels would teach the model to repeat its mistakes
- Only after human review and correction does the label become ground truth

**What happens to unreviewed segments**:
- They remain in DB with `is_corrected = FALSE`
- They are shown to the next available annotator
- They may have features_json already populated (from backfill) — ready to train once corrected

---

## 5. Dashboard Routing Safeguard (JavaScript)

The dashboard JavaScript has a dedicated `ECTOPY_LABELS` set that prevents
ectopy pattern labels from being saved as the background rhythm:

```javascript
const ECTOPY_LABELS = new Set([
    "PVC", "PAC", "PVC Bigeminy", "PVC Trigeminy", "PVC Couplet",
    "PVC Quadrigeminy", "PAC Bigeminy", "PAC Trigeminy", "PAC Quadrigeminy",
    "PAC Couplet", "Atrial Couplet", "Ventricular Run", "Atrial Run",
    "NSVT", "PSVT", "SVT", "Ventricular Tachycardia"
]);
```

When saving, if the selected label is in ECTOPY_LABELS:
- `arrhythmia_label` in DB gets the detected BACKGROUND rhythm (not the ectopy label)
- The ectopy label goes into `events_json` as an event

This separation ensures:
- Rhythm model gets trained on background rhythms (AF, SR, blocks)
- Ectopy model gets trained on beat events (PVC, PAC)

---

## 6. Annotation Priority (What to Annotate First)

For maximum model improvement per hour of annotation time:

### Priority 1: Rare, High-Risk Classes
Annotate these first — model accuracy is lowest here and clinical impact is highest:
- Ventricular Tachycardia / NSVT (need 50+ examples minimum)
- Ventricular Fibrillation (20+ examples)
- 3rd Degree AV Block (30+ examples)
- 2nd Degree AV Block Type 2 (30+ examples)

### Priority 2: Common with High Annotation Volume
These classes need many examples for the model to learn fine-grained boundaries:
- Atrial Fibrillation (target: 200+ examples)
- Sinus Rhythm (target: 300+ examples — needed as strong negative class)
- Sinus Bradycardia (currently 806 corrected — good)

### Priority 3: Ectopy Patterns
- PVC Bigeminy / Trigeminy / Couplet (target: 100+ each)
- PAC variants (target: 80+ each)

### What to Correct vs Confirm
- If model says "Sinus Rhythm" and it IS sinus → just confirm (tick, no change)
- If model says "AF" but it's actually "Atrial Flutter" → correct the label
- If model says "PVC" but beat pattern is clearly bigeminy → select "PVC Bigeminy"
  (this teaches the ectopy model more PVC examples)

---

## 7. Training Workflow (Step by Step)

### Step 1: Backfill V3 Features
Run once after major signal processing changes:
```bash
python scripts/backfill_features.py
```
This populates `features_json` with 60 V3 features for every existing DB segment.
Segments with existing V3 features (detected by `sdnn_ms` key) are skipped.

### Step 2: Train Rhythm Model
```bash
python models_training/retrain_v2.py --task rhythm --mode initial
```
- Loads all `is_corrected=TRUE` segments from DB
- Extracts signal + V3 feature vector (60 dims)
- Routes labels via `get_rhythm_label_idx()`
- Trains CNN+Transformer for 50 epochs (early stopping on val balanced accuracy)
- Saves checkpoint to `models_training/outputs/checkpoints/best_model_rhythm_v3.pth`

### Step 3: Train Ectopy Model
```bash
python models_training/retrain_v2.py --task ectopy --mode initial
```
- Same process but uses 2-second windows around detected R-peaks
- Routes labels via `get_ectopy_label_idx()`
- Saves checkpoint to `models_training/outputs/checkpoints/best_model_ectopy_v3.pth`

### Step 4: Evaluate
```bash
python scripts/compare_models.py --task rhythm
python scripts/compare_models.py --task ectopy
```
Prints side-by-side accuracy table. If V3 model beats V2 on balanced accuracy → deploy.

### Step 5: Deploy
The running `ecg_processor.py` automatically picks up the new checkpoint at startup
(or after restart). No code change needed — checkpoint path is configured in `xai/xai.py`.

### Step 6: Ongoing Annotation → Incremental Retrain
After every 50–100 new `is_corrected` segments:
```bash
python models_training/retrain_v2.py --task rhythm --mode retrain
```
`--mode retrain` uses all corrected data (including old). Starting from the last
checkpoint is optional but can speed convergence for minor corrections.

---

## 8. How Many Examples Are Needed?

| Class | Minimum Usable | Good | Excellent |
|-------|----------------|------|-----------|
| Common classes (Sinus, AF) | 100 | 300 | 1000+ |
| Moderately rare (AV Blocks) | 30 | 80 | 200 |
| Rare (VT, VF, IVR) | 20 | 50 | 150 |
| Ectopy (PVC, PAC) | 50 per class | 150 | 500 |

Below minimum → model may achieve >80% accuracy by always predicting the majority class.
FocalLoss + WeightedRandomSampler mitigates this, but cannot fully compensate for <20 examples.

**Current Status** (approximate):
- Sinus Bradycardia: 806 corrected (excellent)
- AF: unknown (check DB)
- VT/NSVT: unknown — likely <20 (first priority to annotate)

---

## 9. What "Annotated Segments" vs "All Segments" Means for Training

| Mode | SQL Filter | Use Case |
|------|-----------|----------|
| `--mode initial` | `is_corrected = TRUE` | Default; cardiologist-verified only |
| `--mode all` | No filter | Bootstrap only when <20 verified examples for key classes |

**`--mode all` is risky**: model outputs that haven't been reviewed may be wrong.
Training on wrong labels teaches the model to repeat errors.

**Rule**: Always use `--mode initial` unless you explicitly want to bootstrap a class
that has zero verified examples. Even then, review a sample of what gets included.

---

## 10. Label Distribution Check (Before Training)

Before running retrain, always check class distribution:
```bash
python -c "
import psycopg2
conn = psycopg2.connect(host='127.0.0.1', dbname='ecg_analysis', user='ecg_user', password='sais', port=5432)
cur = conn.cursor()
cur.execute('SELECT arrhythmia_label, COUNT(*) FROM ecg_features_annotatable WHERE is_corrected=TRUE GROUP BY arrhythmia_label ORDER BY COUNT(*) DESC')
for row in cur.fetchall(): print(row)
"
```

If any class has < 20 examples, add them to annotation priority before training.
A model trained with 1 VF example and 1000 Sinus examples will never learn VF.
