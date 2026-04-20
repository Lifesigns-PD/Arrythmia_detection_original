"""
sinus_detector.py — Signal Processing-Based Sinus Rhythm Detection
===================================================================
Detects Sinus Rhythm BEFORE ML model inference.

Hierarchy:
  1. Is it Sinus? (P-waves, regular, normal QRS/PR)
     YES → Classify as Sinus Brady/Normal/Tachy (by HR)
     NO  → Pass to ML model for abnormal rhythm

  2. If Sinus BUT has PVC/PAC → IGNORE Sinus, use ectopy label
     (Beat-level ectopy takes precedence over segment-level rhythm)

Sinus Rhythm Definition (ALL must be true):
  ✓ P-waves present in most beats (p_absent_fraction < 0.20)
  ✓ No atrial fibrillation (lf_hf_ratio > 0.5)
  ✓ Normal QRS width (mean_qrs < 120 ms)
  ✓ Normal PR interval (120 <= pr <= 200 ms, or 100-250 flexible)
  ✓ Regular RR intervals (rr_cv < 0.15 = coefficient of variation)
  ✓ No frequent ectopy (qrs_wide_fraction < 0.10, pvc_score_mean < 2.0)
"""

import numpy as np
from typing import Dict, Tuple, Optional


class SinusDetector:
    """Rule-based Sinus Rhythm detector using V3 features."""

    # Thresholds for Sinus Rhythm criteria
    SINUS_CRITERIA = {
        "p_absent_max":           0.20,   # Max 20% beats without P-wave
        "qrs_width_max":          120,    # Max 120 ms (narrow complex)
        "pr_interval_min":        100,    # Min 100 ms (can be slightly low)
        "pr_interval_max":        250,    # Max 250 ms (flexible upper bound)
        "rr_cv_max":              0.15,   # RR regularity (coeff of var < 15%)
        "qrs_wide_fraction_max":  0.10,   # Max 10% wide beats
        "pvc_score_mean_max":     2.0,    # PVC score < 2.0 (not frequent)
        "pac_score_mean_max":     2.0,    # PAC score < 2.0 (not frequent)
        "lf_hf_ratio_min":        0.5,    # LF/HF > 0.5 (not AF chaotic)
        "hr_min":                 40,     # Min HR (escape rhythm)
        "hr_max":                 150,    # Max HR (upper limit)
    }

    @staticmethod
    def is_sinus_rhythm(features: Dict) -> Tuple[bool, str]:
        """
        Determine if segment is Sinus Rhythm (normal or variant).

        Returns:
            (is_sinus: bool, reason: str)
            e.g., (True, "Normal Sinus Rhythm")
                  (False, "p_absent_fraction=0.45 (AF suspected)")
        """

        # Extract features
        p_absent = features.get("p_absent_fraction", 1.0)
        qrs_width = features.get("mean_qrs_duration_ms", 0)
        pr_interval = features.get("pr_interval_ms", 0)
        rr_cv = features.get("rr_cv", 0.5)  # RR coefficient of variation
        qrs_wide_frac = features.get("qrs_wide_fraction", 0)
        pvc_score = features.get("pvc_score_mean", 0)
        pac_score = features.get("pac_score_mean", 0)
        lf_hf = features.get("lf_hf_ratio", 0)
        hr = features.get("mean_hr_bpm", 0)

        # Check each criterion
        c = SinusDetector.SINUS_CRITERIA  # Alias for readability
        checks = {
            "p_waves": (p_absent <= c["p_absent_max"],
                       f"p_absent={p_absent:.2f} (threshold {c['p_absent_max']})"),

            "qrs_width": (qrs_width < c["qrs_width_max"],
                         f"qrs_width={qrs_width:.0f}ms (max {c['qrs_width_max']}ms)"),

            "pr_interval": ((c["pr_interval_min"] <= pr_interval <= c["pr_interval_max"]),
                           f"pr={pr_interval:.0f}ms ({c['pr_interval_min']}-{c['pr_interval_max']}ms)"),

            "rr_regular": (rr_cv <= c["rr_cv_max"],
                          f"rr_cv={rr_cv:.3f} (max {c['rr_cv_max']})"),

            "no_wide_qrs": (qrs_wide_frac <= c["qrs_wide_fraction_max"],
                           f"wide_qrs_frac={qrs_wide_frac:.2f} (max {c['qrs_wide_fraction_max']})"),

            "low_pvc": (pvc_score <= c["pvc_score_mean_max"],
                       f"pvc_score={pvc_score:.2f} (max {c['pvc_score_mean_max']})"),

            "low_pac": (pac_score <= c["pac_score_mean_max"],
                       f"pac_score={pac_score:.2f} (max {c['pac_score_mean_max']})"),

            "not_af": (lf_hf >= c["lf_hf_ratio_min"],
                      f"lf_hf={lf_hf:.2f} (min {c['lf_hf_ratio_min']})"),

            "hr_valid": (c["hr_min"] <= hr <= c["hr_max"],
                        f"hr={hr:.0f}bpm ({c['hr_min']}-{c['hr_max']})"),
        }

        # All must pass
        all_pass = all(check[0] for check in checks.values())

        if all_pass:
            return True, "Meets all Sinus Rhythm criteria"
        else:
            failed = [name for name, (passed, _) in checks.items() if not passed]
            reason = f"Failed: {', '.join(failed)}"
            return False, reason

    @staticmethod
    def classify_sinus_variant(features: Dict) -> str:
        """
        Given IS Sinus Rhythm, classify as Brady/Normal/Tachy.

        Returns: "Sinus Bradycardia", "Sinus Rhythm", or "Sinus Tachycardia"
        """
        hr = features.get("mean_hr_bpm", 70)

        if hr < 60:
            return "Sinus Bradycardia"
        elif hr > 100:
            return "Sinus Tachycardia"
        else:
            return "Sinus Rhythm"

    @staticmethod
    def check_ectopy_override(features: Dict) -> Optional[str]:
        """
        If Sinus but has significant ectopy, return ectopy label to override.

        Returns: "PVC", "PAC", or None
        """
        pvc_score = features.get("pvc_score_mean", 0)
        pac_score = features.get("pac_score_mean", 0)
        short_coupling_frac = features.get("short_coupling_fraction", 0)

        # PVC override: high PVC score AND short coupling (frequent, early beats)
        if pvc_score >= 3.0 and short_coupling_frac >= 0.20:
            return "PVC"

        # PAC override: high PAC score
        if pac_score >= 3.0:
            return "PAC"

        return None


def detect_sinus_and_rhythm(features: Dict) -> Tuple[str, float, str]:
    """
    Main decision function: Signal processing-based rhythm detection.

    Returns:
        (rhythm_label: str, confidence: float, reasoning: str)

    Logic:
      1. Check if Sinus Rhythm
      2. If YES: classify as Brady/Normal/Tachy, check for ectopy override
      3. If NO: return "Unknown" (pass to ML model)
    """
    detector = SinusDetector()

    # Step 1: Is it Sinus?
    is_sinus, sinus_reason = detector.is_sinus_rhythm(features)

    if not is_sinus:
        # Not Sinus → pass to ML model
        return "Unknown", 0.0, f"Not Sinus: {sinus_reason}"

    # Step 2: Classify Sinus variant
    sinus_variant = detector.classify_sinus_variant(features)
    confidence = 0.95  # High confidence when all criteria met

    # Step 3: Check for ectopy override
    ectopy_override = detector.check_ectopy_override(features)

    if ectopy_override:
        # Has Sinus + significant ectopy → use ectopy label
        return ectopy_override, 0.80, f"Sinus + {ectopy_override} (ectopy override)"

    # Return Sinus variant
    return sinus_variant, confidence, f"Sinus rhythm detected: {sinus_reason}"


# Debug: Print criteria
if __name__ == "__main__":
    print("Sinus Rhythm Detection Criteria:")
    print("="*70)
    for key, value in SinusDetector.SINUS_CRITERIA.items():
        print(f"  {key:30} {value}")
    print("\n" + "="*70)

    # Test with sample features
    test_features = {
        "mean_hr_bpm": 72,
        "p_absent_fraction": 0.05,
        "mean_qrs_duration_ms": 95,
        "pr_interval_ms": 160,
        "rr_cv": 0.08,
        "qrs_wide_fraction": 0.0,
        "pvc_score_mean": 0.5,
        "pac_score_mean": 0.2,
        "lf_hf_ratio": 1.2,
    }

    label, conf, reason = detect_sinus_and_rhythm(test_features)
    print(f"\nTest Result:")
    print(f"  Label: {label}")
    print(f"  Confidence: {conf:.2f}")
    print(f"  Reason: {reason}")
