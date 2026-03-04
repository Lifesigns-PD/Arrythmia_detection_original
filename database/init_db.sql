-- init_db.sql
-- Unified Schema for ECG Analysis Dashboard (Single Table Source of Truth)

-- 1. Hybrid Table: ecg_features_annotatable
-- This table stores everything: raw signal, features, labels, and cardiologist annotations.
CREATE TABLE IF NOT EXISTS ecg_features_annotatable (
    segment_id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    segment_index INT NOT NULL,
    segment_start_s FLOAT DEFAULT 0.0,
    segment_duration_s FLOAT DEFAULT 10.0,
    arrhythmia_label VARCHAR(50) DEFAULT 'Unlabeled',
    arrhythmia_text_notes TEXT DEFAULT '',
    r_peaks_in_segment JSONB, -- Stored as JSON array
    features_json JSONB,
    model_pred_label TEXT,
    model_pred_probs JSONB,
    cardiologist_notes TEXT DEFAULT '',
    corrected_by TEXT,
    corrected_at TIMESTAMP,
    is_corrected BOOLEAN DEFAULT FALSE,
    used_for_training BOOLEAN DEFAULT FALSE,
    training_round INT DEFAULT 0,
    raw_signal JSONB, -- The actual 10-second ECG voltage values
    pr_interval FLOAT,
    segment_fs INT DEFAULT 125, -- Standardized to 125Hz
    dataset_source TEXT,
    is_verified BOOLEAN DEFAULT FALSE,
    mistake_target TEXT,
    events_json JSONB DEFAULT '[]'::jsonb, -- Unified beat-level event storage (PVC/PAC)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Performance Indexes
-- Ensure we don't duplicate the same 10-second chunk of the same file
CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_legacy_segment ON ecg_features_annotatable (filename, segment_index);

-- GIN Index for high-speed JSONB searches (Crucial for finding PACs/PVCs across thousands of segments)
CREATE INDEX IF NOT EXISTS idx_gin_events_legacy ON ecg_features_annotatable USING GIN (events_json);

-- 3. Utility View for Analytics
CREATE OR REPLACE VIEW v_segment_summary AS
SELECT 
    segment_id,
    filename,
    segment_index,
    arrhythmia_label as background_rhythm,
    CASE WHEN is_corrected THEN 'VERIFIED' ELSE 'PENDING' END as segment_state,
    CASE 
        WHEN jsonb_typeof(events_json) = 'array' THEN jsonb_array_length(events_json)
        ELSE 0 
    END as event_count
FROM ecg_features_annotatable;
