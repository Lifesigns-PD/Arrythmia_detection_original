
import neurokit2 as nk
import numpy as np
import pandas as pd

def test_delineation():
    # Produce a synthetic ECG signal
    ecg = nk.ecg_simulate(duration=10, sampling_rate=250, heart_rate=70)
    
    # Find R-peaks
    _, info = nk.ecg_peaks(ecg, sampling_rate=250)
    r_peaks = info["ECG_R_Peaks"]
    
    # Delineate
    _, waves = nk.ecg_delineate(ecg, r_peaks, sampling_rate=250, method="dwt")
    
    print("Waves keys found:")
    for k in waves.keys():
        print(f" - {k}: {len(waves[k])} items")
        
    # Check for R onsets/offsets
    print("\nSpecific keys check:")
    for key in ["ECG_R_Onsets", "ECG_R_Offsets", "ECG_P_Onsets", "ECG_Q_Onsets", "ECG_S_Offsets"]:
        val = waves.get(key)
        if val is not None:
             valid_count = np.sum(~pd.isna(val))
             print(f" - {key}: {valid_count} valid values")
        else:
             print(f" - {key}: NOT FOUND")

if __name__ == "__main__":
    test_delineation()
