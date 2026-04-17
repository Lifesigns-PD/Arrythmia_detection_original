# Complete Pipeline Documentation
## Every Folder, Every File, How They Work Together

---

## 1. Full Data Flow (End to End)

```
REAL-TIME DEPLOYMENT
────────────────────
Device (125 Hz ECG)
    │
    ▼ gRPC stream
grpc_gen/grpc_server.py   ← receives raw ECG packets via gRPC
    │
    ▼ enqueues to
kafka_consumer.py          ← Kafka consumer (optional streaming layer)
    │                         Deserializes ECG packets
    ▼
ecg_processor.py           ← MAIN ORCHESTRATION FILE
    │
    ├── signal_processing_v3/ ── process_ecg_v3()
    │       ├── preprocess_v3()      (baseline, denoising, artifact removal)
    │       ├── detect_r_peaks_ensemble()   (3-detector vote)
    │       ├── delineate_v3()       (P/Q/R/S/T per beat)
    │       └── extract_features_v3()  → 60 features
    │
    ├── xai/xai.py ──────────────────── run inference
    │       ├── Rhythm model → rhythm_label, confidence
    │       ├── Ectopy model (per beat) → beat_events [{label, conf, peak_sample}]
    │       └── SHAP → top-feature explanations
    │
    └── decision_engine/ ────────────── RhythmOrchestrator.decide()
            ├── rules.py              (Pause, AF net, AFL spectral)
            ├── rhythm_orchestrator.py (orchestrates all steps)
            └── → SegmentDecision     (background_rhythm, events, final_display_events)
                │
                ▼
          database/db_service.py ←── store SegmentDecision to PostgreSQL
                │
                ▼
          dashboard/app.py ──────── serve /api/process endpoint
                │
                ▼
          Cardiologist views in dashboard/templates/index.html
                │
          [Annotate → Save]
                │
                ▼
          DB: is_corrected = TRUE, arrhythmia_label updated

RETRAINING
──────────
scripts/backfill_features.py ──── populate features_json for all segments
models_training/retrain_v2.py ─── train CNN+Transformer on DB data
scripts/compare_models.py ──────── evaluate old vs new checkpoint
```

---

## 2. Folder Descriptions

### `signal_processing_v3/` — V3 Signal Processing Pipeline
**Purpose**: Convert raw ECG into clean signal + 60 features + delineation.
**Entry point**: `process_ecg_v3(raw_signal, fs=125)` in `__init__.py`

```
signal_processing_v3/
├── __init__.py                  Entry point: process_ecg_v3()
│                                Returns: signal, r_peaks, delineation,
│                                         features (dict, 60 keys), feature_vector (60,),
│                                         sqi, sqi_issues, skipped
│
├── preprocessing/
│   ├── quality_check.py         assess_signal_quality(signal, fs)
│   │                            → (score, issues_list) — pre-flight check
│   ├── adaptive_baseline.py     remove_baseline(signal, fs)
│   │                            Morphological opening + Savitzky-Golay
│   ├── adaptive_denoising.py    remove_powerline(signal, fs)
│   │                            Auto-detects 50/60 Hz, applies notch filter
│   ├── artifact_removal.py      remove_artifacts(signal, fs)
│   │                            Wavelet-based HF artifact suppression
│   └── pipeline.py              preprocess_v3(signal, fs, skip_if_unusable)
│                                Chains all 4 stages; returns cleaned signal
│
├── detection/
│   ├── pan_tompkins.py          detect_pan_tompkins(signal, fs) → peaks
│   ├── hilbert_detector.py      detect_hilbert(signal, fs) → peaks
│   ├── wavelet_detector.py      detect_wavelet(signal, fs) → peaks
│   └── ensemble.py              detect_r_peaks_ensemble(signal, fs) → peaks
│                                Vote across 3 detectors + polarity check
│                                + RR validation + peak refinement
│
├── delineation/
│   ├── wavelet_delineation.py   delineate_beats_cwt(signal, r_peaks, fs)
│   │                            Per beat: P/Q/R/S/T via CWT ridge tracking
│   │                            Outputs p_morphology, delta_wave, t_inverted,
│   │                            q_depth, s_depth, qrs_polarity
│   ├── template_matching.py     delineate_template(signal, r_peaks, fs)
│   │                            Patient-specific template correlation
│   └── hybrid.py                delineate_v3(signal, r_peaks, fs)
│                                Tries wavelet first, falls back to template;
│                                Returns per_beat list + summary dict + method
│
├── features/
│   ├── hrv_time_domain.py       compute_hrv_time_domain(r_peaks, fs) → 11 features
│   ├── hrv_frequency.py         compute_hrv_frequency(r_peaks, fs) → 8 features
│   ├── nonlinear.py             compute_nonlinear_features(r_peaks, fs) → 8 features
│   ├── morphology_features.py   compute_morphology_features(signal, per_beat, r_peaks, fs) → 13
│   ├── beat_morphology.py       compute_beat_discriminators(signal, per_beat, r_peaks, fs) → 20
│   │                            Also: BEAT_DISC_FEATURES (20 names)
│   │                            pvc_score/_pac_score per beat, aggregated to means
│   ├── extraction.py            extract_features_v3(...) → dict (60 keys)
│   │                            feature_dict_to_vector(dict) → float32 ndarray (60,)
│   │                            FEATURE_NAMES_V3 — 60-element canonical list
│   └── __init__.py              Exports: extract_features_v3, feature_dict_to_vector,
│                                         FEATURE_NAMES_V3, compute_beat_discriminators
│
├── quality/
│   └── signal_quality.py        compute_sqi_v3(signal, r_peaks, fs)
│                                 → (score float, issues list)
│                                 10-criteria scoring: flatline, SNR, noise, clipping, etc.
│
└── tests/
    ├── test_preprocessing.py    6 tests for all preprocessing stages
    ├── test_detection.py        3 tests: sensitivity/PPV/F1 on synthetic ECG
    ├── test_delineation.py      5 tests: P/Q/R/S/T boundary accuracy
    ├── test_features.py         6 tests: 60-feature count, HRV plausibility
    └── compare_v2_v3.py         Side-by-side V2 vs V3 comparison
                                 Flags: --synthetic (no DB needed), --db, --n N
```

---

### `decision_engine/` — Clinical Rules + Orchestration
**Purpose**: Turn ML predictions into validated clinical events.

```
decision_engine/
├── models.py              Pydantic data models:
│                          - SegmentDecision: top-level result container
│                          - SegmentState: ANALYZED / UNRELIABLE / WARMUP
│                          - Event: one arrhythmia event
│                            Fields: event_id, event_type, event_category,
│                                    start_time, end_time, priority,
│                                    used_for_training, display_state,
│                                    beat_indices, ml_evidence, rule_evidence
│                          - EventCategory: RHYTHM / ECTOPY
│                          - DisplayState: DISPLAYED / HIDDEN / PENDING
│
├── rules.py               All clinical rules:
│   ├── _detect_flutter_waves()   FFT of QRS-blanked signal for AFL
│   ├── _classify_compensatory_pause()  Full vs incomplete pause → PVC/PAC
│   ├── derive_rule_events()      Pause, AF safety net, AFL spectral
│   ├── apply_ectopy_patterns()   3-stage PVC/PAC correction + bigeminy/run detection
│   ├── apply_display_rules()     Show/hide arbitration logic
│   └── apply_training_flags()    Set used_for_training per event type
│
└── rhythm_orchestrator.py  RhythmOrchestrator class:
    └── decide()            Full 7-step orchestration (see DECISION_ENGINE_COMPLETE.md)
        ├── SQI gate → UNRELIABLE?
        ├── Background rhythm (HR + P-wave rules)
        ├── Rule events (Pause/AF/AFL)
        ├── ML rhythm event (with confidence threshold)
        ├── Per-beat ectopy (3-layer confidence gate)
        ├── apply_ectopy_patterns()
        ├── Background promotion (VF > VT > AF > NSVT)
        ├── apply_display_rules()
        └── apply_training_flags()
```

---

### `models_training/` — Model Definition and Training
**Purpose**: Define CNN+Transformer architecture and train on DB data.

```
models_training/
├── models.py              V1 model: CNNTransformerClassifier (signal only)
│                          Used only for comparison (best_model_rhythm.pth checkpoint)
│
├── models_v2.py           V2+ model: CNNTransformerWithFeatures
│                          Input: (signal_batch, feature_batch)
│                          Architecture: CNN → Transformer → feature fusion → classifier
│                          Used for both rhythm and ectopy training
│
├── data_loader.py         CRITICAL FILE — all training configuration:
│   ├── RHYTHM_CLASS_NAMES    15 rhythm model classes (see MODEL_ARCHITECTURE.md)
│   ├── ECTOPY_CLASS_NAMES    3 ectopy model classes [None, PVC, PAC]
│   ├── RHYTHM_LABEL_ALIASES  NSVT→VT, Sinus Tachy→Sinus Rhythm, etc.
│   ├── get_rhythm_label_idx() Routes any label string to rhythm class index
│   ├── get_ectopy_label_idx() Routes any label string to ectopy class index
│   ├── normalize_label()      Converts raw DB strings to canonical class names
│   └── LABEL_MAP              Abbreviation expansion (AF→Atrial Fibrillation, etc.)
│
├── retrain_v2.py          Training script:
│   ├── Loads DB segments (is_corrected=TRUE)
│   ├── Extracts 60 V3 features per segment
│   ├── FocalLoss + WeightedRandomSampler for class imbalance
│   ├── 50-epoch training with early stopping on balanced accuracy
│   └── Saves checkpoint to outputs/checkpoints/best_model_{task}_v3.pth
│
├── retrain.py             Legacy V1 training (signal-only, 4 classes) — deprecated
│
├── calibration.py         Temperature scaling for probability calibration
│
├── metrics.py             Balanced accuracy, per-class F1 computation
│
├── balance_dataset.py     Manual oversampling utility (replaced by WeightedRandomSampler)
│
├── inspect_ckpt.py        Inspect checkpoint metadata (epoch, acc, features, classes)
│
└── outputs/checkpoints/
    ├── best_model_rhythm.pth      V1 signal-only checkpoint (15 classes old)
    ├── best_model_rhythm_v2.pth   V2 with 60 features (if trained)
    ├── best_model_rhythm_v3.pth   V3 target checkpoint (60 features, 15 classes)
    ├── best_model_ectopy.pth      V1 ectopy checkpoint
    ├── best_model_ectopy_v2.pth   V2 ectopy with 15 features (old)
    └── best_model_ectopy_v3.pth   V3 target checkpoint (60 features, 3 classes)
```

---

### `ecg_processor.py` — Main Real-Time Processor
**Purpose**: Single file that wires signal processing → models → decision engine.

```
ECGProcessor class:
  __init__():
    - Loads rhythm + ectopy model checkpoints
    - Initializes RhythmOrchestrator

  process(raw_signal, fs=125):
    1. Segment signal into 10-second windows (1250 samples each)
    2. For each window:
       a. process_ecg_v3(window) → cleaned signal + r_peaks + 60 features
       b. If sqi < threshold → skip window (UNRELIABLE)
       c. xai.run_inference(window) → rhythm_pred + beat_events
       d. Merge 60 V3 features into clinical_features dict
       e. orchestrator.decide(ml_pred, clinical_features, sqi) → SegmentDecision
       f. Store SegmentDecision to DB
    3. Return list of SegmentDecisions

  _segment(signal):
    Splits signal into 10s windows with optional overlap

  _run_orchestrator(window, v3_result):
    Bridges V3 output to orchestrator.decide() signature
```

---

### `xai/xai.py` — Inference + Explainability
**Purpose**: Run both models on a segment and produce structured predictions.

```
Key functions:
  run_inference(signal, fs):
    1. process_ecg_v3(signal) → features, r_peaks
    2. Rhythm model forward pass → {label, confidence, probabilities}
    3. For each R-peak: ectopy model on 2s window → {label, conf, beat_idx, peak_sample}
    4. SHAP explanation → top-3 features contributing to rhythm prediction
    5. Returns: {rhythm: {...}, ectopy: {beat_events: [...]}, xai_notes: {...}}

Feature names:
  from signal_processing_v3.features.extraction import FEATURE_NAMES_V3 as FEATURE_NAMES
  NUM_FEATURES = 60   ← must match model input size
```

---

### `dashboard/` — Annotation Dashboard
**Purpose**: Web interface for cardiologist annotation and retraining.

```
dashboard/
├── app.py                 Flask application:
│   ├── GET /              Renders index.html
│   ├── POST /api/process  Accepts signal_data → runs ecg_processor → returns events
│   ├── POST /api/save     Saves annotation to DB (sets is_corrected=TRUE)
│   ├── GET /api/segments  Returns segments needing review (is_corrected=FALSE)
│   └── POST /api/retrain  Triggers retrain_v2.py in subprocess
│
└── templates/index.html   Full-stack single-page app:
    ├── ECG strip renderer (canvas-based, 25mm/s scale)
    ├── Diagnosis dropdown (all labels grouped by category)
    │   Groups: Sinus Rhythms / Atrial / AV Blocks / Ventricular / Ectopy Patterns
    │   Values use canonical DB labels (e.g. "PVC Bigeminy", not "Ventricular Bigeminy")
    ├── Beat marker overlay (colored dots per R-peak)
    ├── Pattern event overlays (bigeminy brackets, run highlights)
    ├── XAI panel (top-3 SHAP features)
    ├── ECTOPY_LABELS set (JS routing guard — prevents ectopy labels in arrhythmia_label)
    └── Save logic:
        - Reads background rhythm from diagnosis-select dropdown
        - Sends arrhythmia_label (rhythm) + events_json (ectopy events) separately
        - Backend splits these into rhythm model training data + ectopy model training data
```

---

### `database/` — PostgreSQL Interface
**Purpose**: All DB operations (read/write segments, features, annotations).

```
database/
├── db_service.py          Main service class:
│   ├── save_segment()     Insert or update segment with features + annotation
│   ├── get_segment()      Fetch segment by ID
│   ├── get_pending_review() Segments with is_corrected=FALSE
│   └── update_annotation() Set arrhythmia_label + is_corrected=TRUE
│
├── db_loader.py           Batch loader for training:
│   └── load_training_batch(task, limit) → [(signal, features, label), ...]
│
├── setup_fresh_db.py      Creates ecg_features_annotatable table schema
│
├── import_to_sql.py       One-time import of legacy JSON data to PostgreSQL
│
├── auto_backup.py         Scheduled DB backup script
│
└── export_sql_segments_to_json.py  Export segments back to JSON (for external use)
```

**Key DB table**: `ecg_features_annotatable`
```sql
segment_id          UUID PRIMARY KEY
signal_data         FLOAT[]         -- raw ECG samples
segment_fs          INTEGER         -- sampling rate (125)
arrhythmia_label    TEXT            -- annotated or model-predicted label
is_corrected        BOOLEAN         -- TRUE = cardiologist verified
features_json       JSONB           -- 60 V3 features + r_peaks + sqi
events_json         JSONB           -- ectopy events from decision engine
created_at          TIMESTAMP
updated_at          TIMESTAMP
```

---

### `data/` — Data Ingestion
**Purpose**: Import ECG data from external sources into PostgreSQL.

```
data/
├── ingest_json.py         Ingest raw ECG JSON files (from device exports)
├── ingest_ecg_extracts.py Ingest from ECG_Data_Extracts/ folder
├── mit_download.py        Download MIT-BIH arrhythmia database via wfdb
├── mitb_database.py       Parse MITDB and import to SQL
├── ptbxl_download.py      Download PTB-XL database
└── ptbxl_database.py      Parse PTB-XL and import to SQL
```

---

### `scripts/` — Utility Scripts
**Purpose**: One-time operations, evaluation, debugging.

```
scripts/
├── backfill_features.py   IMPORTANT: Populate V3 features for all DB segments
│                          Run before retraining after signal processing changes
│                          Uses process_ecg_v3(); skips segments with sdnn_ms already present
│
├── compare_models.py      Evaluate V1 vs V2 vs V3 checkpoint accuracy side by side
│                          Loads each checkpoint, runs on validation set, prints accuracy table
│
├── generate_hybrid_reports.py  PDF report generation from ECG JSON files
│                               Auto-selects best available checkpoint (V3 > V2 > V1)
│
├── check_models.py        Quick sanity check on loaded checkpoint shape/classes
│
├── commit_uncorrected_annotations.py  Utility to batch-confirm model outputs as correct
│                                      USE WITH CAUTION — sets is_corrected=TRUE in bulk
│
├── diagnose_v2_pipeline.py  Debug V2 pipeline compatibility issues
│
├── import_mitdb_only.py   Import only MITDB beats to SQL (for beat-level training)
│
├── wfdb_to_json.py        Convert wfdb format to JSON for ingest
│
├── afdb_to_json.py        Convert AF Database (PhysioNet) to JSON
│
├── reset_flags.py         Reset is_corrected=FALSE for specified segments
│
├── wipe_and_reset_db.py   DANGEROUS: clear DB and reimport from scratch
│
├── verify_db_shape.py     Check DB row count, feature dimensions, class distribution
│
└── log_to_json.py         Convert grpc log files to JSON format
```

---

### `grpc_gen/` — gRPC Communication Layer
**Purpose**: Receive real-time ECG data from device via gRPC.

```
grpc_gen/
├── ecg_pb2.py             Auto-generated protobuf classes (do not edit)
├── ecg_pb2_grpc.py        Auto-generated gRPC service stubs (do not edit)
└── grpc_server.py         gRPC server: receives ECG packets, calls ecg_processor.py
```

---

### `signal_processing/` — V2 Signal Processing (Legacy)
**Purpose**: Old 15-feature pipeline. Kept for backward compatibility.
**Status**: Deprecated. All new code uses signal_processing_v3.

---

### `evaluation_and_checks/` — Evaluation Scripts
**Purpose**: Various diagnostic and evaluation scripts from development.

Key files:
- `evaluate_rhythm.py` — Per-class rhythm model accuracy on DB
- `evaluate_ectopy.py` — Per-class ectopy model accuracy on DB
- `db_health_check.py` — Database integrity checks
- `test_pattern_detection.py` — Unit tests for bigeminy/trigeminy rule detection

---

## 3. Configuration Files

```
config.py          Global constants: DB connection params, checkpoint paths,
                   model class counts, SQI thresholds

signal_processing/config.yaml   V2 signal processing parameters (legacy)
```

---

## 4. Key Cross-Cutting Constants

These constants MUST be consistent across all files:

| Constant | Value | Location | Used By |
|----------|-------|----------|---------|
| NUM_FEATURES | 60 | signal_processing_v3/features/extraction.py | xai.py, retrain_v2.py, data_loader.py |
| FEATURE_NAMES_V3 | [60 names] | extraction.py | All above |
| RHYTHM_CLASS_NAMES | [15 names] | data_loader.py | retrain_v2.py, compare_models.py, dashboard |
| ECTOPY_CLASS_NAMES | [3 names] | data_loader.py | Same |
| WINDOW_SEC | 2.0 s (ectopy) | data_loader.py | xai.py beat windowing |
| SEG_LEN | 1250 samples | data_loader.py | ecg_processor, dashboard |
| TARGET_FS | 125 Hz | data_loader.py | All signal processing |

**If you change any of these, you MUST update all files that use them and retrain.**

---

## 5. How to Run the Full System

### Local Development (Dashboard Only)

```bash
# 1. Start database
# (PostgreSQL must be running on 127.0.0.1:5432)

# 2. Start dashboard
cd dashboard
python app.py
# Opens http://localhost:5000

# 3. Upload an ECG segment JSON in the dashboard
#    → it calls /api/process → ecg_processor → models → decision engine → displays
```

### Retraining After Annotation

```bash
# Step 1: Backfill V3 features for any new segments
python scripts/backfill_features.py

# Step 2: Check class distribution before training
python -c "
import psycopg2
conn = psycopg2.connect(host='127.0.0.1', dbname='ecg_analysis', user='ecg_user', password='sais', port=5432)
cur = conn.cursor()
cur.execute('SELECT arrhythmia_label, COUNT(*) FROM ecg_features_annotatable WHERE is_corrected=TRUE GROUP BY arrhythmia_label ORDER BY COUNT(*) DESC')
for r in cur.fetchall(): print(r)
"

# Step 3: Train rhythm model
python models_training/retrain_v2.py --task rhythm --mode initial

# Step 4: Train ectopy model
python models_training/retrain_v2.py --task ectopy --mode initial

# Step 5: Compare new vs old
python scripts/compare_models.py --task rhythm
python scripts/compare_models.py --task ectopy

# Step 6: If improved, restart dashboard to pick up new checkpoint
```

### Generate PDF Reports (From ECG JSON Files)

```bash
python scripts/generate_hybrid_reports.py
# Reads ECG_Data_Extracts/*.json
# Outputs PDF per patient to outputs/hybrid_reports/
```

---

## 6. File Dependency Map

```
ecg_processor.py
    ├── signal_processing_v3/__init__.py (process_ecg_v3)
    │       └── [all submodules]
    ├── xai/xai.py (run_inference)
    │       ├── models_training/models_v2.py (CNNTransformerWithFeatures)
    │       ├── signal_processing_v3/features/extraction.py (FEATURE_NAMES_V3)
    │       └── [rhythm + ectopy checkpoint .pth files]
    ├── decision_engine/rhythm_orchestrator.py (RhythmOrchestrator)
    │       ├── decision_engine/rules.py
    │       └── decision_engine/models.py
    └── database/db_service.py (save_segment)

models_training/retrain_v2.py
    ├── models_training/data_loader.py (RHYTHM/ECTOPY_CLASS_NAMES, get_*_label_idx)
    ├── models_training/models_v2.py (CNNTransformerWithFeatures)
    └── signal_processing_v3 (process_ecg_v3, feature_dict_to_vector, FEATURE_NAMES_V3)

dashboard/app.py
    └── ecg_processor.py (ECGProcessor)
```
