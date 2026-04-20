# ECG Analysis Package

Standalone signal processing and visualization toolkit for ECG data.

## Files

- visualise_pipeline.py — Main visualization script
- sample_ecg.json — Sample ECG data for testing  
- requirements.txt — Python dependencies
- SIGNAL_PROCESSING_METHODS.md — Technical methodology

## Quick Start

Install dependencies:
  pip install -r requirements.txt

Visualize ECG data (Simple JSON):
  python visualise_pipeline.py --json sample_ecg.json --output report.png

Visualize ECG data (MongoDB export):
  python visualise_pipeline.py --json ECG_Data_Extracts/ADM1014424580.json --record 0

From database (requires psycopg2):
  python visualise_pipeline.py --db
  python visualise_pipeline.py --db --seg 1234

## JSON Input Formats

Format 1 - Simple object:
  {"admissionId": "ADM001", "ecgData": [...], "timestamp": 1712600000}

Format 2 - MongoDB export (array):
  [{"admissionId": "ADM1014...", "value": [[...]], "facilityId": "CF..."}]

## Output

Generates console report + PNG/PDF plot showing:
  Row 1: Raw ECG signal
  Row 2: Preprocessed signal + R-peaks detected
  Row 3: Full waveform delineation (P/Q/R/S/T)

## Signal Processing Pipeline

See SIGNAL_PROCESSING_METHODS.md for technical details.
