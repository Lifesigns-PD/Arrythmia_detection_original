
import neurokit2 as nk
import numpy as np
import pandas as pd

def compare_indices():
    # Produce a synthetic ECG signal with clear Q and S waves
    # We use a lower HR to ensure clear separation
    ecg = nk.ecg_simulate(duration=10, sampling_rate=250, heart_rate=60)
    
    # Find R-peaks
    _, info = nk.ecg_peaks(ecg, sampling_rate=250)
    r_peaks = info["ECG_R_Peaks"]
    
    # Delineate
    _, waves = nk.ecg_delineate(ecg, r_peaks, sampling_rate=250, method="dwt")
    
    print("Delineation Indices for first 3 beats:")
    for i in range(3):
        print(f"\nBeat {i+1}:")
        p_on = waves["ECG_P_Onsets"][i]
        q_peak = waves["ECG_Q_Peaks"][i]
        r_on = waves["ECG_R_Onsets"][i]
        r_peak = r_peaks[i]
        r_off = waves["ECG_R_Offsets"][i]
        s_peak = waves["ECG_S_Peaks"][i]
        
        print(f" - P Onset:  {p_on}")
        print(f" - Q Peak:   {q_peak}")
        print(f" - R Onset:  {r_on}")
        print(f" - R Peak:   {r_peak}")
        print(f" - R Offset: {r_off}")
        print(f" - S Peak:   {s_peak}")
        
        if not pd.isna(q_peak) and not pd.isna(r_on):
            if q_peak < r_on:
                print(" !!! WARNING: Q Peak is BEFORE R Onset. R Onset is NOT the start of QRS.")
            else:
                print(" OK: R Onset is before/at Q Peak (maybe it includes Q).")
                
        if not pd.isna(s_peak) and not pd.isna(r_off):
            if s_peak > r_off:
                print(" !!! WARNING: S Peak is AFTER R Offset. R Offset is NOT the end of QRS.")
            else:
                print(" OK: R Offset is after/at S Peak (maybe it includes S).")

if __name__ == "__main__":
    compare_indices()
