"""
pipeline.py — V3 Preprocessing Orchestrator
============================================
Runs all preprocessing stages in order:
  0. Quality pre-check
  1. Adaptive baseline removal
  2. Adaptive denoising (powerline + LP)
  3. Artifact removal (muscle + spikes)

Returns cleaned signal + quality metadata.
"""

import numpy as np
from typing import Dict, Any

from .quality_check      import assess_signal_quality
from .adaptive_baseline  import remove_baseline_adaptive
from .adaptive_denoising import remove_noise_adaptive
from .artifact_removal   import remove_artifacts


def preprocess_v3(
    signal: np.ndarray,
    fs: int = 125,
    skip_if_unusable: bool = False,
) -> Dict[str, Any]:
    """
    Full V3 preprocessing pipeline.

    Parameters
    ----------
    signal           : 1-D ECG array (mV)
    fs               : sampling rate
    skip_if_unusable : if True and quality_score < 0.4, return raw signal unchanged

    Returns
    -------
    dict with keys:
      "cleaned"        : np.ndarray  — cleaned signal
      "quality_score"  : float       — 0–1
      "quality_issues" : list[str]   — flagged problems
      "was_skipped"    : bool        — True if preprocessing was bypassed
    """
    # Stage 0: Pre-flight quality check
    quality_score, quality_issues = assess_signal_quality(signal, fs)

    if skip_if_unusable and quality_score < 0.4:
        return {
            "cleaned":        signal.astype(np.float32),
            "quality_score":  quality_score,
            "quality_issues": quality_issues,
            "was_skipped":    True,
        }

    cleaned = signal.astype(np.float64)

    # Stage 1: Baseline
    try:
        cleaned = remove_baseline_adaptive(cleaned, fs)
    except Exception as e:
        quality_issues.append(f"baseline_removal_failed: {e}")

    # Stage 2: Noise
    try:
        cleaned = remove_noise_adaptive(cleaned, fs)
    except Exception as e:
        quality_issues.append(f"noise_removal_failed: {e}")

    # Stage 3: Artifacts
    try:
        cleaned = remove_artifacts(cleaned, fs)
    except Exception as e:
        quality_issues.append(f"artifact_removal_failed: {e}")

    # Final sanity: replace NaN/Inf with 0
    cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "cleaned":        cleaned.astype(np.float32),
        "quality_score":  quality_score,
        "quality_issues": quality_issues,
        "was_skipped":    False,
    }
