-- Migration: Add sqi_score column for signal quality validation
-- Run this if you have an existing database that needs the new column

ALTER TABLE ecg_features_annotatable
ADD COLUMN IF NOT EXISTS sqi_score FLOAT DEFAULT NULL;

-- Add index for faster SQI-based queries
CREATE INDEX IF NOT EXISTS idx_sqi_score ON ecg_features_annotatable (sqi_score);

-- Verify the column was added
SELECT column_name, data_type FROM information_schema.columns
WHERE table_name = 'ecg_features_annotatable' AND column_name = 'sqi_score';
