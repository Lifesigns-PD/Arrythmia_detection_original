# Project Files — ECG Arrhythmia Detection System

Only files that are actively used by the running system are listed here.
Files and folders that are unused, archived, or superseded are excluded.

---

## Root

| File | Purpose |
|---|---|
| `ecg_processor.py` | Main entry point — segments signal, runs full pipeline, returns analysis dict |
| `kafka_consumer.py` | Kafka consumer — receives live ECG packets, calls `ecg_processor.py` |
| `config.py` | Global constants: `SAMPLING_RATE=125`, `WINDOW_SAMPLES=1250`, model paths |
| `streamlit_dashboard.py` | Interactive local dashboard — visualises every pipeline stage |
| `requirements.txt` | Python dependencies |
| `.env` | DB credentials, Kafka broker URL (not committed to git) |
| `.env.template` | Template for `.env` |
| `Dockerfile` | Docker image definition |
| `docker-compose.yml` | Orchestrates app + Kafka + PostgreSQL containers |

---

## `signal_processing_v3/` — V3 Signal Processing Package

### Entry point
| File | Purpose |
|---|---|
| `__init__.py` | Exports `process_ecg_v3()` — runs all 5 stages in sequence |

### `preprocessing/`
| File | Purpose |
|---|---|
| `pipeline.py` | `preprocess_v3()` — baseline wander removal, notch filter, lowpass, NaN cleanup |
| `adaptive_baseline.py` | Butterworth HP 0.15 Hz + Savitzky-Golay + morphological opening |
| `adaptive_denoising.py` | Lowpass Butterworth 45 Hz (order 4) |
| `artifact_removal.py` | Clips extreme spikes before filtering |
| `quality_check.py` | Pre-filter signal quality assessment |

### `detection/`
| File | Purpose |
|---|---|
| `ensemble.py` | `detect_r_peaks_ensemble()` — votes across 3 detectors (≥2/3); `refine_peaks_subsample()` |
| `pan_tompkins.py` | Pan-Tompkins detector: bandpass → diff → square → MWI → adaptive threshold |
| `hilbert_detector.py` | Hilbert envelope detector |
| `wavelet_detector.py` | CWT Mexican Hat wavelet detector |

### `delineation/`
| File | Purpose |
|---|---|
| `hybrid.py` | `delineate_v3()` — combines wavelet + template matching, returns P/Q/R/S/T per beat |
| `wavelet_delineation.py` | CWT zero-crossing delineation for P/T onset/offset |
| `template_matching.py` | 8-beat median template, cross-correlation refinement (±60 ms) |

### `features/`
| File | Purpose |
|---|---|
| `extraction.py` | `extract_features_v3()` — produces 60-feature dict; `feature_dict_to_vector()`; `FEATURE_NAMES_V3` |
| `hrv_time_domain.py` | RMSSD, SDNN, pNN50, triangular index |
| `hrv_frequency.py` | LF/HF ratio, VLF/LF/HF power via Welch PSD |
| `nonlinear.py` | SD1/SD2 (Poincaré), SampEn, DFA α1 |
| `beat_morphology.py` | QRS duration, P amplitude, T amplitude, ST level per beat |
| `morphology_features.py` | Aggregate morphology stats across all beats in window |

### `quality/`
| File | Purpose |
|---|---|
| `signal_quality.py` | `compute_sqi_v3()` — SNR, kurtosis, flatline fraction → composite SQI score |

---

## `decision_engine/` — Rules + ML Fusion

| File | Purpose |
|---|---|
| `__init__.py` | Package init |
| `models.py` | `SegmentDecision` dataclass — final output container |
| `rhythm_orchestrator.py` | `RhythmOrchestrator.decide()` — hierarchical: sinus rules first, ML second |
| `sinus_detector.py` | `SinusDetector.is_sinus_rhythm()` — 9 criteria checklist; `classify_sinus_variant()` |
| `rules.py` | `derive_rule_events()` — Pause / AF safety net / Atrial Flutter rules; `apply_ectopy_patterns()` — Couplet / Bigeminy / NSVT; `apply_display_rules()` |

---

## `models_training/` — ML Model Definition & Training

| File | Purpose |
|---|---|
| `__init__.py` | Package init |
| `models_v2.py` | `CNNTransformerWithFeatures` architecture — SmallCNN + TransformerEncoder + feature MLP + fusion head |
| `retrain_v2.py` | Training script — `ECGEventDatasetV2`, FocalLoss, AdamW, WeightedRandomSampler |
| `data_loader.py` | `RHYTHM_CLASS_NAMES`, `ECTOPY_CLASS_NAMES`, label mappings |
| `metrics.py` | Balanced accuracy, per-class F1, confusion matrix utilities |
| `outputs/checkpoints/best_model_rhythm_v2.pth` | Active rhythm model checkpoint (9 classes) |
| `outputs/checkpoints/best_model_ectopy_v2.pth` | Active ectopy model checkpoint (3 classes: None/PVC/PAC) |
| `outputs/checkpoints/feature_scaler_rhythm.joblib` | StandardScaler for rhythm model input features |
| `outputs/checkpoints/feature_scaler_ectopy.joblib` | StandardScaler for ectopy model input features |

---

## `database/` — PostgreSQL Interface

| File | Purpose |
|---|---|
| `__init__.py` | Package init |
| `db_service.py` | `get_segment_new()`, `save_model_prediction()`, `update_segment_status()` |
| `db_loader.py` | Batch loader for dashboard and evaluation |
| `init_db.sql` | Table + index creation: `ecg_features_annotatable`, view `v_segment_summary` |
| `import_to_sql.py` | Imports labelled JSON segments into the database |
| `setup_fresh_db.py` | One-time DB initialisation script |
| `auto_backup.py` | Scheduled PostgreSQL dump |
| `migrate_sqi_column.sql` | Migration: adds `sqi_score` column |
| `export_sql_segments_to_json.py` | Exports DB rows to JSON for offline use |

---

## `xai/` — Explainability & Inference

| File | Purpose |
|---|---|
| `__init__.py` | Package init |
| `xai.py` | `explain_segment()` — loads both models, runs ectopy per-beat + rhythm inference, returns saliency |
| `label_report.py` | Generates per-label accuracy reports from DB |
| `analyze_dataset.py` | Dataset distribution analysis against DB |

---

## `ecg_analysis_package/` — Standalone Reporting Tool

| File | Purpose |
|---|---|
| `ecg_pipeline_report.py` | CLI: runs full pipeline on a JSON file, generates PDF/PNG report |
| `sample_ecg.json` | Demo ECG signal (1250 samples, 125 Hz) used by dashboard default |
| `generate_sample.py` | Generates synthetic ECG for testing |
| `visualise_pipeline.py` | Matplotlib visualisation of all pipeline stages |

---

## `ECG_Data_Extracts/` — Real Patient ECG Input Files

Raw ECG recordings from the hospital system. Each file is a JSON array of 659 packets:
```
[ { "packetNo": N, "admissionId": "ADM...", "value": [[samples...]] }, ... ]
```
Used as input to `ecg_pipeline_report.py` and `streamlit_dashboard.py`.

---

## `scripts/` — Data Utilities

| File | Purpose |
|---|---|
| `wfdb_to_json.py` | Converts PhysioNet WFDB format to JSON for DB import |
| `afdb_to_json.py` | Converts MIT-BIH AF database to JSON |
| `import_mitdb_only.py` | Imports MIT-BIH arrhythmia DB segments |
| `backfill_features.py` | Re-extracts features for older DB rows missing them |
| `verify_db_shape.py` | Checks DB row count and feature column completeness |
| `reset_flags.py` | Resets `used_for_training` flags for re-training runs |
| `check_models.py` | Prints checkpoint metadata and class counts |

---

## Not Used / Excluded

| Name | Reason excluded |
|---|---|
| `signal_processing/` | Superseded by `signal_processing_v3/` |
| `NO_USE/` | Archived older versions of database, models, scripts |
| `dashboard/` | Replaced by `streamlit_dashboard.py` |
| `models_training/models.py` | Superseded by `models_v2.py` |
| `models_training/retrain.py` | Superseded by `retrain_v2.py` |
| `models_training/outputs/checkpoints/best_model_ectopy.pth` | Old v1 checkpoint, not loaded |
| `models_training/outputs/checkpoints/best_model_rhythm.pth` | Old v1 checkpoint, not loaded |
| `models_training/outputs/checkpoints/best_model_rhythm_v2_poor.pth` | Rejected checkpoint |
| `models_training/outputs/logs/` | Training logs — reference only, not runtime |
| `models_training/calibration.py` | Not wired into inference |
| `models_training/balance_dataset.py` | One-time utility, not part of pipeline |
| `models_training/inspect_ckpt.py` | Debug utility |
| `signal_processing_v3/pipeline_nk2.py` | Experimental NeuroKit2 pipeline, not used |
| `pong.py` | Test file |
| `check_qrs.py` | One-off debug script |
| `test_producer.py` | Kafka test producer, not production |
| `mongo_writer.py` | MongoDB writer — system uses PostgreSQL |
| `grpc_gen/` | gRPC generated stubs, not active |
| `visualiser_bundle/` | Old visualiser, replaced by dashboard |
| `raw_data/`, `data/`, `sample/`, `input_segments/`, `outputs/`, `converted_ecg/` | Local data folders, not code |
| `*.zip`, `*.log`, `*.pdf`, `*.txt` (root) | Build artifacts, reports, logs |
| `evaluation_and_checks/` | One-off evaluation scripts, not runtime |
| `testing_final/` | Manual test scripts |
