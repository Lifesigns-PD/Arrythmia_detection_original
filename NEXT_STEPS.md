# Next Steps - After Fixes Applied

## Immediate Actions Required

### 1. **Update Existing Database (if not using fresh init)**
If you already have a running database, add the new `sqi_score` column:

```bash
psql -h 127.0.0.1 -U ecg_user -d ecg_analysis -f database/migrate_sqi_column.sql
```

Or manually:
```sql
ALTER TABLE ecg_features_annotatable
ADD COLUMN IF NOT EXISTS sqi_score FLOAT DEFAULT NULL;
```

### 2. **Verify Data Before Training**
Check that you have cardiologist-verified segments:

```sql
-- Count verified segments
SELECT COUNT(*) as verified_segments FROM ecg_features_annotatable 
WHERE is_corrected = TRUE;

-- Should return > 0 to proceed with training
```

### 3. **Run V2 Training Again**
The architectural fixes prevent stale data reuse. Start training with bootstrap from v1:

```bash
python models_training/retrain_v2.py --task rhythm --mode initial --bootstrap-v1 --epochs 30
```

**Expected behavior:**
- Dataset only loads segments with `is_corrected=TRUE`
- Error summary printed at end showing any data issues
- After training completes, `training_round` incremented for used segments
- Can run same script again later without duplicate training data

### 4. **Check Error Report**
Look for data loading errors in the log:
```
DATA LOADING ERROR SUMMARY
=================================================================
signal_parse_errors: X occurrences
  1. seg_id=1234: [error details]
...
```

This helps identify bad data that needs correction.

---

## Verification Checklist

- [ ] Database migration applied (sqi_score column exists)
- [ ] `is_corrected=TRUE` segments exist in DB (verified via SQL)
- [ ] retrain_v2.py compiles: `python -m py_compile models_training/retrain_v2.py`
- [ ] First training run completes without [ABORT] errors
- [ ] Error summary shows no critical failures
- [ ] After training, check: `SELECT DISTINCT training_round FROM ecg_features_annotatable WHERE used_for_training=TRUE`
  - Should show `training_round = 1` for first run
  - Should show `training_round = 2` for second run (for re-used segments)

---

## What Changed (Summary)

### Database
- ✅ Added `is_corrected=TRUE` filter to training data loader
- ✅ Added `training_round` increment after each training run
- ✅ Added error logging for all silent failures (22 points)
- ✅ Added validation for signals, events, times, sources
- ✅ Added `sqi_score` column to schema for future quality checks

### Code Quality
- ✅ No more silent failures - every error is logged with segment_id
- ✅ No more stale data reuse - only verified segments used
- ✅ No more ambiguous training history - training_round tracks each run
- ✅ Safer training - validates all data before processing

### Backward Compatibility
- ✅ All existing models still work
- ✅ No breaking changes
- ✅ Can run old and new training code on same database

---

## Troubleshooting

### "Only 0 windows extracted"
- Check that segments have `is_corrected=TRUE`
- Verify at least one cardiologist annotation exists
- Run: `SELECT COUNT(*) FROM ecg_features_annotatable WHERE is_corrected=TRUE`

### "[ABORT] Empty critical classes"
- For Ectopy: Need at least one PVC, PAC, and None (normal beat) sample
- For Rhythm: Need at least one sample for critical classes (check class names in output)
- Add more annotations via dashboard and set `is_corrected=TRUE`

### "Data loading errors: signal_parse_errors: 50 occurrences"
- Indicates malformed signal data in database
- Check: `SELECT segment_id, LENGTH(signal_data) FROM ecg_features_annotatable WHERE signal_data IS NOT NULL LIMIT 5`
- May need to re-import data or fix signal encoding

---

## Timeline

1. **Now:** Apply fixes (DONE ✅)
2. **Step 1:** Migrate database (5 min)
3. **Step 2:** Verify data (2 min)
4. **Step 3:** Run v2 training (30-60 min depending on data size)
5. **Step 4:** Review error summary (5 min)
6. **Step 5:** Discuss results and next training steps

---

## Questions?

Refer to `FIXES_APPLIED.md` for detailed technical documentation of each fix.

All files modified:
- `models_training/retrain_v2.py` - Error logging, is_corrected filter, training_round update
- `database/db_service.py` - Documentation clarifications
- `database/init_db.sql` - Added sqi_score column
- `database/migrate_sqi_column.sql` - Migration script (NEW)
