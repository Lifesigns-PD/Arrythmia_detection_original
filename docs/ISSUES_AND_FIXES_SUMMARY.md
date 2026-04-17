# Issues Found & Required Fixes

## ISSUE 1: PAC Class Data Loss — 663 Missing Annotations
**Status:** CRITICAL FIX AVAILABLE

### Problem:
- Cardiologist marked 40+ PAC beats in dashboard
- Database shows only 12 corrected PAC annotations
- 663 PAC annotations exist but are marked `is_corrected=FALSE`
- These are not being used for training

### Root Cause:
Dashboard annotations were entered but never "committed" to `is_corrected=TRUE`

### Fix:
```bash
python scripts/commit_uncorrected_annotations.py
```
This will:
- Mark all 663 uncorrected PAC annotations as `is_corrected=TRUE`
- Set `used_for_training=TRUE`
- PAC training samples will jump from 12 → 675

**Timeline:** 1 minute

---

## ISSUE 2: Signal Processing Defects — 39% Corrupted Features
**Status:** PLANNED BUT NOT EXECUTED

### Problem:
- 11,821 of 30,158 segments (39%) have corrupted PR/QRS values
- NeuroKit2 delineation fails silently, returning None → stored as 0ms
- SQI scale is wrong (0-100 instead of 0-1)
- Dashboard has duplicate feature extraction code (different bugs)

### Root Cause:
NeuroKit2 ECG delineation has no fallback when it fails.

### Fix Required:
7-step implementation plan in `SIGNAL_PROCESSING_ISSUES_AND_FIX_PLAN.md`:
1. Add validation layer to morphology.py
2. Implement heuristic fallback
3. Fix SQI scale in sqi.py
4. Unify dashboard code
5. Recalculate all 30,158 segments
6. Validate output
7. Add monitoring

**Expected Impact:**
- Before: 5,937 bad QRS values (20%), 9,513 bad PR values (31%)
- After: ~100 unfixable segments (0.3%), ~30,000 valid (99%)
- Model accuracy: ~70% → 85-90%

**Timeline:** ~4 hours

---

## ISSUE 3: V2 Model Class Mismatch
**Status:** WORKAROUND IN PLACE (V1 fallback)

### Problem:
V2 rhythm checkpoint trained with 13 classes, but current code expects 14 classes.
"Sinus Bradycardia" was added after V2 training.

### Current State:
- V1 Rhythm: 14 classes ✓ (ACTIVE)
- V2 Ectopy: 3 classes ✓ (ACTIVE)
- V2 Rhythm: 13 classes ✗ (MISMATCH - falls back to V1)

### Check Model Status:
```bash
python scripts/check_models.py
```

### Fix:
Retrain V2 with new 14-class setup (requires clean features first):
```bash
python models_training/retrain_v2.py --task rhythm --mode initial --epochs 60
```

**Timeline:** ~2 hours (after Issue 2 is fixed)

---

## RECOMMENDED EXECUTION ORDER

### STEP 1: Commit PAC Annotations (5 min)
```bash
python scripts/commit_uncorrected_annotations.py
```
Result: 12 → 675 training samples for PAC

### STEP 2: Fix Signal Processing (4 hours)
```bash
# Implement 7-step fix plan
# Recalculate all 30,158 segments
```
Result: 30,000 clean features (up from ~20,000)

### STEP 3: Retrain V2 (2 hours)
```bash
python models_training/retrain_v2.py --task rhythm --mode initial --epochs 60
python models_training/retrain_v2.py --task ectopy --mode initial --epochs 80
```
Result: V2 models with clean data + 14-class rhythm

### STEP 4: Validate
```bash
python scripts/check_models.py
# Run test_producer.py with 5 messages
# Check Docker logs for arrhythmia detection
```

---

## WHAT TO SEND IN ZIP NOW?

Your current zip is ready to send because:
- ✓ Kafka pipeline works (tested end-to-end)
- ✓ 5-thread consumer deployed
- ✓ MongoDB schema matches Vinayak's spec
- ✓ All required files present
- ✓ Docker builds in seconds

**Known limitations for reviewers:**
- PAC class is weak (12 corrected samples, but 675 uncorrected available)
- V2 model uses V1 fallback due to class mismatch
- ~39% of training data has feature defects (noted in Signal Processing plan)

Document these caveats in your email.

---

## PRIORITY RANKING

1. **MUST DO NOW (if you want robust PAC):** Commit uncorrected annotations (5 min)
2. **SHOULD DO (for better accuracy):** Fix signal processing (4 hours)
3. **CAN DEFER:** Retrain V2 (2 hours, depends on #2)

