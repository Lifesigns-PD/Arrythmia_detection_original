# V3 Test Suite

Run each test independently from the project root.

| File | What it tests |
|------|--------------|
| `test_preprocessing.py` | Baseline removal, noise removal, artifact removal, quality gate |
| `test_detection.py` | R-peak ensemble vs V2 Pan-Tompkins: precision/recall/F1 |
| `test_delineation.py` | Fiducial coverage, QRS duration, QTc, P-wave detection |
| `test_features.py` | 40-feature coverage, HRV plausibility, vector shape/dtype |
| `compare_v2_v3.py` | Side-by-side V2 vs V3 on synthetic + optional DB data |

## Quick start

```bash
# From project root
python signal_processing_v3/tests/test_preprocessing.py
python signal_processing_v3/tests/test_detection.py
python signal_processing_v3/tests/test_delineation.py
python signal_processing_v3/tests/test_features.py

# V2 vs V3 comparison (synthetic, 10 cases)
python signal_processing_v3/tests/compare_v2_v3.py

# V2 vs V3 comparison using real DB data
python signal_processing_v3/tests/compare_v2_v3.py --db
```
