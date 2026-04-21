"""
test_dataset.py — Pre-training dataset validation
Run from models_training/ directory: python test_dataset.py
"""
import sys
import numpy as np
sys.path.insert(0, '..')

from retrain_v2 import ECGEventDatasetV2
from collections import Counter

print("\n" + "="*60)
print("  ECTOPY DATASET CHECK")
print("="*60)
ds_e = ECGEventDatasetV2(task='ectopy', source_filter='all', augment=False)
print(f'Total windows : {len(ds_e)}')
print(f'Feature dim   : {ds_e.num_features}  (expect 47)')
dist_e = Counter(s[2] for s in ds_e.samples)
print(f'Class dist    : {dict(sorted(dist_e.items()))}')
print(f'  0=None : {dist_e.get(0,0)}')
print(f'  1=PVC  : {dist_e.get(1,0)}')
print(f'  2=PAC  : {dist_e.get(2,0)}')

# NaN check
feats_e = np.stack([s[1] for s in ds_e.samples])
nan_e = int(np.isnan(feats_e).sum())
inf_e = int(np.isinf(feats_e).sum())
print(f'NaN count     : {nan_e}  (must be 0)')
print(f'Inf count     : {inf_e}  (must be 0)')
print(f'Feature range : {feats_e.mean(axis=0).min():.2f} to {feats_e.mean(axis=0).max():.2f}')

print("\n" + "="*60)
print("  RHYTHM DATASET CHECK")
print("="*60)
ds_r = ECGEventDatasetV2(task='rhythm', source_filter='all', augment=False)
print(f'Total windows : {len(ds_r)}')
print(f'Feature dim   : {ds_r.num_features}  (expect 36)')
dist_r = Counter(s[2] for s in ds_r.samples)
print(f'Class dist    : {dict(sorted(dist_r.items()))}')

from data_loader import RHYTHM_CLASS_NAMES
for idx, name in enumerate(RHYTHM_CLASS_NAMES):
    print(f'  {idx}={name[:30]:<30} : {dist_r.get(idx,0)}')

# NaN check
feats_r = np.stack([s[1] for s in ds_r.samples])
nan_r = int(np.isnan(feats_r).sum())
inf_r = int(np.isinf(feats_r).sum())
print(f'NaN count     : {nan_r}  (must be 0)')
print(f'Inf count     : {inf_r}  (must be 0)')
print(f'Feature range : {feats_r.mean(axis=0).min():.2f} to {feats_r.mean(axis=0).max():.2f}')

print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
issues = []
if ds_e.num_features != 47:   issues.append(f"Ectopy features = {ds_e.num_features}, expected 47")
if ds_r.num_features != 36:   issues.append(f"Rhythm features = {ds_r.num_features}, expected 36")
if len(ds_e) < 5000:          issues.append(f"Ectopy only {len(ds_e)} windows — too few")
if len(ds_r) < 10000:         issues.append(f"Rhythm only {len(ds_r)} windows — too few")
if nan_e > 0:                  issues.append(f"Ectopy has {nan_e} NaN values — will cause NaN loss")
if nan_r > 0:                  issues.append(f"Rhythm has {nan_r} NaN values — will cause NaN loss")
if dist_e.get(1,0) == 0:      issues.append("Ectopy has NO PVC samples!")
if dist_e.get(2,0) == 0:      issues.append("Ectopy has NO PAC samples!")

if issues:
    print("ISSUES FOUND:")
    for i in issues:
        print(f"  ❌ {i}")
else:
    print("  All checks passed — safe to train!")
    print(f"  Ectopy: {len(ds_e)} windows, {ds_e.num_features} features")
    print(f"  Rhythm: {len(ds_r)} windows, {ds_r.num_features} features")
