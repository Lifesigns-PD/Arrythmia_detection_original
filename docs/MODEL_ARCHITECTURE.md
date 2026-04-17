# Model Architecture — CNN + Transformer with Feature Fusion
## Why This Architecture, Design Rationale, Pros/Cons

---

## 1. High-Level Architecture

```
Input (10-second ECG, 1250 samples @ 125 Hz)
         │
         ▼
   ┌─────────────────────────────────────┐
   │        CNN Feature Extractor        │  ← learns local waveform shapes
   │  Conv1D(1→32→64→128) + Pooling     │    (QRS morphology, T-wave shape)
   └─────────────────────────────────────┘
                    │
                    ▼ temporal feature map (128, T')
   ┌─────────────────────────────────────┐
   │    Transformer Encoder (4 heads)    │  ← learns long-range dependencies
   │    Positional encoding + 2 layers   │    (beat-to-beat patterns, AF rhythm)
   └─────────────────────────────────────┘
                    │
                    ▼ context vector (128,)
         ┌──────────┴──────────────┐
         │                         │
    ┌────▼─────┐          ┌────────▼──────────┐
    │ CLS token│          │  60 V3 Features   │
    │ (signal  │          │  (clinical feats) │
    │  pathway)│          └────────┬──────────┘
    └────┬─────┘                   │
         │             ┌───────────▼──────────────┐
         │             │  Feature Projection Layer │
         │             │  Linear(60 → 64) + GELU  │
         └──────┬───────┘
                │ concatenate [128 + 64 = 192]
                ▼
   ┌─────────────────────────────────────┐
   │     Classification Head            │
   │     Linear(192 → 256) + GELU       │
   │     Dropout(0.3)                    │
   │     Linear(256 → num_classes)       │
   └─────────────────────────────────────┘
                │
         logits (num_classes)
              [softmax → probabilities]
```

Two separate model instances are trained:
- **Rhythm model**: 9 classes (Sinus Rhythm, AF, Atrial Flutter, 1st/3rd/2nd Degree AV Block, BBB, Artifact, Sinus Bradycardia)
- **Ectopy model**: 3 classes (None / PVC / PAC), same architecture

---

## 2. Why CNN + Transformer (Not Pure CNN, Not LSTM)

### Option A: Pure CNN
Classic approach for ECG (e.g. Stanford AHA paper). 

**Problem**: A 10-second segment at 125 Hz = 1250 samples.
At the final conv layer the receptive field is typically 500–800 samples.
Patterns like AF (no P-waves anywhere in the segment) require looking at
the WHOLE segment simultaneously. CNNs struggle with:
- Global rhythm patterns that span the entire segment
- Varying inter-beat intervals (you need the RELATIONSHIP between all beats, not just local shapes)
- Variable segment lengths when padding is needed

### Option B: LSTM / GRU
Good at sequences; handles variable length. 

**Problem**: 
- LSTM is sequential — slow to train and infer (can't parallelize across time)
- Vanishing gradient over 1250 steps — earlier beats have weak gradient signal
- LSTM captures local context well but has difficulty with long-range skip-patterns (e.g., 2nd degree block where every 3rd beat is dropped)

### Why CNN + Transformer is Better Here

- **CNN stage**: extracts LOCAL features efficiently (30ms QRS is well-captured by a kernel of size 32 = 256 ms). Reduces sequence length from 1250 → 78 before the transformer.
- **Transformer stage**: global attention across all time positions simultaneously (no sequential bottleneck). Perfectly suited to "look at all beats at once" for rhythm classification.
- **Feature fusion**: the explicit 60 V3 features act as a "clinical bias" — they inject domain knowledge that the CNN/Transformer might not extract from a limited training set. This dramatically improves rare class performance.

### Why Feature Fusion Helps

With only a few hundred examples of "Ventricular Tachycardia", the CNN+Transformer alone
has insufficient data to learn a reliable VT detector. But the V3 features provide:
- `qrs_wide_fraction` = 1.0 (all beats wide)
- `compensatory_pause_fraction` ≈ 0 (regular tachycardia)
- `mean_hr_bpm` > 100
- `pvc_score_mean` high

The model can use these features to boost the VT probability even with few training examples.

---

## 3. Training Configuration

### Loss Function: FocalLoss

Standard cross-entropy suffers from **class imbalance**: if 80% of segments are "Sinus Rhythm",
the model learns to predict Sinus Rhythm for everything and still achieves 80% accuracy.

**FocalLoss** down-weights well-classified easy examples:
```
FL(p_t) = -(1 - p_t)^γ × log(p_t)
```
With γ = 2.0: a correctly classified easy example (p_t = 0.9) gets weight (1-0.9)^2 = 0.01.
A hard misclassified example (p_t = 0.3) gets weight (1-0.3)^2 = 0.49 — 49× more focus.

This forces the model to learn from rare, hard examples (VT, 3rd degree block)
rather than over-fitting to the majority class.

### Sampler: WeightedRandomSampler

In addition to FocalLoss, each training batch is **resampled** to ensure rare classes appear
proportionally. For a class with 20 examples vs 2000 examples:
- Without sampler: rare class appears in ~1% of batches
- With sampler: rare class appears in ~equal frequency via inverse-frequency weighting

Together, FocalLoss + WeightedRandomSampler give the model approximately balanced exposure
to all classes during every epoch.

### Why Two Separate Models (Rhythm + Ectopy)

**Option A: One model, many classes (40+ classes)**
- Would need to simultaneously learn "is there AF" AND "is this beat a PVC" from one segment
- These are orthogonal tasks: AF is a 10-second rhythm; PVC is a 0.5-second beat event
- Multi-task confusion reduces both tasks' accuracy
- Training data with combined labels is sparse (e.g., "AF + PVC Bigeminy" might have 5 examples)

**Option B: Two separate models (chosen)**
- Rhythm model sees whole segment: learns rhythm characteristics (RR variability, P-wave presence, QRS width)
- Ectopy model sees 2-second per-beat windows: learns beat-level waveform shapes
- Each model is trained on its own label set — no interference
- Can update one without affecting the other

---

## 4. Rhythm Model — 15 Classes

```
Class  Name                    Training Source
──────────────────────────────────────────────────────────────
  0    Sinus Rhythm            Cardiologist-verified normal segments
  1    Atrial Fibrillation     is_corrected=TRUE + arrhythmia_label="Atrial Fibrillation"
  2    Atrial Flutter          Cardiologist-annotated
  3    Junctional Rhythm       Annotated (HR 40–60, no P-waves, narrow QRS)
  4    Idioventricular Rhythm  Annotated (HR <40, wide QRS, no P-waves)
  5    Ventricular Fibrillation Annotated (chaotic, no discernible QRS)
  6    Ventricular Tachycardia Annotated VT + folded NSVT (NSVT → VT alias)
  7    1st Degree AV Block     Annotated (PR > 200 ms)
  8    2nd Degree AV Block T1  Annotated (Wenckebach)
  9    2nd Degree AV Block T2  Annotated (Mobitz II)
 10    3rd Degree AV Block     Annotated (complete dissociation)
 11    Bundle Branch Block     Annotated (wide QRS ≥ 120 ms, LBBB or RBBB pattern)
 12    Artifact                Annotated (unreadable signal)
 13    Pause                   Annotated (RR > 2.0s)
 14    Sinus Bradycardia       Annotated (HR < 60, normal P-QRS-T)
```

**Labels NOT trained on Rhythm model** (routed to None):
- PVC, PAC, any Bigeminy/Trigeminy/Couplet (→ ectopy model only)
- NSVT → folded into VT (class 6)
- SVT/PSVT → rules-only, no rhythm model class
- "Other Arrhythmia" → excluded (no clinical definition)
- Sinus Tachycardia → folded into Sinus Rhythm (insufficient examples)

---

## 5. Ectopy Model — 3 Classes

```
Class  Name   Training Source
──────────────────────────────────────────────────────────────────────────
  0    None   2s windows around normal beats (labeled by annotation context)
  1    PVC    Segments with: PVC, PVC Bigeminy, PVC Trigeminy, PVC Couplet,
              Ventricular Run, NSVT, VT (all collapsed to PVC class)
  2    PAC    Segments with: PAC, PAC Bigeminy, PAC Trigeminy, Atrial Run,
              PSVT, SVT, Atrial Couplet (all collapsed to PAC class)
```

Training window: 2 seconds (250 samples) centered on each R-peak.
The model learns the beat-level morphology (QRS width, P-wave presence, T-wave axis).

**Important caveat**: for a segment labeled "PVC Bigeminy", the ectopy model trains
ALL beats in that segment as PVC. This introduces some label noise (normal alternating
beats in a bigeminy pattern also get labeled PVC). However:
- The model still learns from the strong PVC signal present
- The label noise is systematic and the model learns to tolerate it
- Beat-level annotation (labeling individual beats) would fix this but requires 10× more annotator effort

---

## 6. Inference Path

```
10-second segment
       │
       ├── V3 Signal Processing ──────────► 60-feature vector
       │         │
       │         └─ r_peaks ──────────────► 2s per-beat windows
       │
       ├── Rhythm Model (whole segment)
       │   Input: signal (1250,) + features (60,)
       │   Output: logits (15,) → argmax → rhythm_label + confidence
       │
       └── Ectopy Model (per beat, 2s window)
           Input: beat_window (250,) + features (60,)
           Output: logits (3,) → argmax → beat_label + confidence per beat
                                           ↓
                                  xai.py aggregates → beat_events list
                                           ↓
                              Decision Engine uses beat_events
                              to detect Bigeminy/Trigeminy/Couplet/Run
```

---

## 7. Per-Class Confidence Thresholds (Inference Gates)

Not every model output is acted on. Each class has a minimum confidence before
the decision engine creates an event:

| Class | Threshold | Rationale |
|-------|-----------|-----------|
| Ventricular Fibrillation | 0.90 | Critical — false alarm is extremely disruptive |
| Ventricular Tachycardia | 0.88 | Critical — false positive triggers urgent alarm |
| NSVT | 0.85 | High clinical significance |
| Atrial Fibrillation | 0.85 | Most common arrhythmia — very high FP rate for AF at lower thresholds |
| Atrial Flutter | 0.85 | Similar to AF |
| 3rd Degree AV Block | 0.85 | Requires immediate pacemaker — low FP tolerance |
| 2nd Degree Type 2 | 0.85 | High risk progression |
| 2nd Degree Type 1 | 0.82 | Lower risk — slightly lower bar |
| 1st Degree AV Block | 0.80 | Informational — acceptable FP rate |
| Bundle Branch Block | 0.80 | Common, lower urgency |
| Sinus Bradycardia | 0.75 | Lowest bar — benign, commonly encountered |

Sinus Rhythm and Unknown are never turned into events (they ARE the absence of pathology).

---

## 8. XAI — Explainability

`xai/xai.py` uses SHAP (SHapley Additive exPlanations) to explain each model decision:

```python
from signal_processing_v3.features.extraction import FEATURE_NAMES_V3 as FEATURE_NAMES
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(input)
```

For each prediction, the top 5 contributing features are surfaced in the API response.
This allows a cardiologist reviewing the dashboard to see not just "AF predicted at 87%"
but also "LF/HF ratio = 0.1 (+), P-absent fraction = 0.9 (+), SDNN = 180 ms (+)".

---

## 9. Pros / Cons of the Full ML Architecture

| | Pros | Cons |
|-|------|------|
| CNN+Transformer | Global + local context; parallelisable | Needs >500 examples per class to train reliably |
| Feature fusion | Compensates for rare classes; clinically interpretable | Feature quality is pipeline-dependent |
| FocalLoss | Prevents majority-class dominance | γ needs tuning (2.0 is standard starting point) |
| Separate rhythm/ectopy models | Clean separation; independent update | Double inference time vs single model |
| SHAP explainability | Cardiologist trust; identifies feature bugs | SHAP is approximate for deep models |
| 2-second ectopy windows | Captures full PVC/PAC morphology | Misses patterns requiring >2s context (e.g. bigeminy) — pattern detection is rules-based |
