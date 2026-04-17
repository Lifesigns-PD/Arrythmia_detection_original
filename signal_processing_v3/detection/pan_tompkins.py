"""
pan_tompkins.py — Re-export from V2 (already solid, kept unchanged)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from signal_processing.pan_tompkins import detect_r_peaks  # noqa: F401
