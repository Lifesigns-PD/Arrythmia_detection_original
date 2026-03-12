
import neurokit2 as nk
import numpy as np
import pandas as pd
import psycopg2
import sys
import os
from pathlib import Path

# Add project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from dashboard.app import _preprocess, _detect_r_peaks_neurokit, _calculate_pr_interval, _compute_qrs_durations

def test_real_data():
    conn = psycopg2.connect("host=localhost dbname=ecg_analysis user=ecg_user password=sais")
    cur = conn.cursor()
    
    # Get a segment that has been confirmed or just a random one
    cur.execute("SELECT segment_id, raw_signal FROM ecg_features_annotatable LIMIT 1")
    res = cur.fetchone()
    if not res:
        print("No segments found in DB")
        return
        
    seg_id, raw_signal = res
    print(f"Testing Segment ID: {seg_id}")
    
    signal = np.array(raw_signal)
    fs = 125
    
    # 1. Pipeline Cleaning
    # (Note: _preprocess in app.py uses clean_signal from signal_processing)
    cleaned = _preprocess(signal, fs)
    
    # 2. Peaks
    r_peaks = _detect_r_peaks_neurokit(cleaned, fs)
    
    # 3. Delineate using NEW functions from app.py
    try:
        pr_median = _calculate_pr_interval(cleaned, r_peaks, fs)
        qrs_durations = _compute_qrs_durations(cleaned, r_peaks, fs)
        
        print(f"\nResults for {len(r_peaks)} beats:")
        if len(qrs_durations) > 0:
            print(f" - QRS Median: {np.median(qrs_durations):.1f} ms")
            print(f" - QRS Range:  {np.min(qrs_durations):.1f} to {np.max(qrs_durations):.1f} ms")
        else:
            print(" - QRS: No valid intervals found (outside filter range)")
            
        if pr_median > 0:
            print(f" - PR Median (Calculated):  {pr_median:.1f} ms")
        else:
            print(" - PR: No valid intervals found (outside filter range)")

    except Exception as e:
        print(f"Delineation failed: {e}")
        
    cur.close()
    conn.close()

if __name__ == "__main__":
    test_real_data()
