import uuid
import numpy as np
from typing import Dict, Any

from decision_engine.models import (
    SegmentDecision,
    SegmentState,
    Event,
    EventCategory,
    DisplayState
)
from decision_engine.rules import (
    derive_rule_events,
    apply_ectopy_patterns,
    apply_display_rules,
    apply_training_flags
)
from decision_engine.sinus_detector import detect_sinus_and_rhythm

class RhythmOrchestrator:
    def __init__(self):
        pass

    def decide(self, 
               ml_prediction: Dict[str, Any],   
               clinical_features: Dict[str, Any], 
               sqi_result: Dict[str, Any],
               segment_index: int = 0) -> SegmentDecision:
        """
        Orchestrates the decision process for a single ECG segment.
        
        Args:
            ml_prediction: Dictionary containing model outputs (label, probs, confidence)
            clinical_features: Dictionary of calculated clinical features (HR, PR, etc.)
            sqi_result: Dictionary containing signal quality metrics
            segment_index: Index of the segment in the recording (default 0 for API calls)
            
        Returns:
            SegmentDecision: The final decision object containing all events and states.
        """
        
        # 1. Initialize SegmentDecision
        decision = SegmentDecision(
            segment_index=segment_index,
            segment_state=SegmentState.ANALYZED,
            background_rhythm="Unknown"
        )

        # 2. Segment state checks (warmup / unreliable)
        if not sqi_result.get('is_acceptable', True):
            decision.segment_state = SegmentState.UNRELIABLE
            # Fix 1: Background MUST stay Sinus/Brady/Tachy
            decision.background_rhythm = self._detect_background_rhythm(clinical_features)
            
            # Create a "Artifact" event (Fix 3: Create Artifact Event ✅)
            artifact_event = Event(
                event_id=str(uuid.uuid4()),
                event_type="Artifact",
                event_category=EventCategory.RHYTHM,
                start_time=0.0,
                end_time=10.0,
                priority=0,
                used_for_training=False, # Fix 3: used_for_training = False ✅
                display_state=DisplayState.DISPLAYED
            )
            decision.events.append(artifact_event)
            # We don't manually append to final_display_events here anymore; 
            # we let the display arbitrator handle it in step 5.

        # 3. SINUS RHYTHM DETECTION (Signal Processing - BEFORE ML Model)
        # Check if segment is Sinus Rhythm using V3 features
        # If YES → classify as Brady/Normal/Tachy
        # If NO → will use ML model below
        sinus_label, sinus_conf, sinus_reason = detect_sinus_and_rhythm(clinical_features)

        if sinus_label != "Unknown":
            # Sinus Rhythm detected by signal processing — NO ML needed.
            decision.background_rhythm = sinus_label
            print(f"[Sinus Detection] {sinus_label} (conf={sinus_conf:.2f}) - {sinus_reason}")
        else:
            # Not Sinus → will use ML model below
            decision.background_rhythm = "Unknown"
            print(f"[Sinus Detection] Not Sinus - passing to lethal detector then ML")

        # 3.5 — Lethal / non-ML rhythm gate (VTach / VFib / SVT via signal processing)
        # Runs BEFORE the ML rhythm model. If it fires, ML rhythm is automatically
        # skipped because Step 4B is gated on background_rhythm == "Unknown".
        _signal  = clinical_features.get("_signal_clean") or clinical_features.get("_signal")
        _r_peaks = clinical_features.get("r_peaks")
        _fs      = int(clinical_features.get("fs", 125))

        if decision.background_rhythm == "Unknown" and _signal is not None and _r_peaks is not None:
            from decision_engine.lethal_detector import detect_signal_rhythm
            _lethal_label, _lethal_conf, _lethal_reason = detect_signal_rhythm(
                signal=np.asarray(_signal, dtype=np.float32),
                r_peaks=np.asarray(_r_peaks, dtype=int),
                features=clinical_features,
                fs=_fs,
            )
            if _lethal_label:
                decision.background_rhythm = _lethal_label
                print(f"[Lethal/Signal Detection] {_lethal_label} (conf={_lethal_conf:.2f}) — {_lethal_reason}")

        # 4. Gather Events
        # A) Rule-Derived Events — pass signal + r_peaks for AFL spectral detection
        rule_events = derive_rule_events(
            clinical_features,
            signal=np.asarray(_signal, dtype=np.float32) if _signal else None,
            r_peaks=np.asarray(_r_peaks, dtype=int) if _r_peaks else None,
            fs=_fs,
            background_rhythm=decision.background_rhythm,
        )
        
        # B) ML-Derived Events (Only if NOT Sinus - Sinus already detected above)
        ml_events = []

        # Only use ML if Sinus wasn't detected by signal processing
        if decision.background_rhythm == "Unknown":
            _rhythm_block = ml_prediction.get("rhythm") or {}
            ml_label = _rhythm_block.get("label") or ml_prediction.get("label", "Unknown")
            ml_conf = _rhythm_block.get("confidence") or ml_prediction.get("confidence", 0.0)

            # Per-class confidence thresholds: rarer and more dangerous rhythms require
            # higher confidence to fire, reducing costly false positives.
            _RHYTHM_CONF_THRESHOLDS = {
                "Ventricular Fibrillation":      0.90,
                "VT":                            0.88,
                "Ventricular Tachycardia":       0.88,
                "NSVT":                          0.85,
                "Atrial Fibrillation":           0.85,
                "AF":                            0.85,
                "Atrial Flutter":                0.85,
                "3rd Degree AV Block":           0.85,
                "2nd Degree AV Block Type 2":    0.85,
                "2nd Degree AV Block Type 1":    0.82,
                "1st Degree AV Block":           0.80,
                "Intraventricular Conduction Delay": 0.80,
            }
            required_conf = _RHYTHM_CONF_THRESHOLDS.get(ml_label, 0.80)
            if ml_label not in ["Sinus Rhythm", "Unknown"] and ml_conf > required_conf:
                decision.background_rhythm = ml_label
                ml_events.append(self._create_event_from_ml(ml_label, ml_prediction))
        else:
            # Sinus already detected - don't override with ML
            _rhythm_block = ml_prediction.get("rhythm") or {}
            ml_label = _rhythm_block.get("label", "Unknown")
            ml_conf = _rhythm_block.get("confidence", 0.0)

        # C) Per-beat ectopy events — required for bigeminy/trigeminy pattern detection
        #    xai.py now returns beat_events: [{beat_idx, peak_sample, label, conf}, ...]
        beat_events = ml_prediction.get("ectopy", {}).get("beat_events", [])
        seg_fs = float(clinical_features.get("fs", 125))

        # Step 1: Base gate — raise threshold from 0.95 to 0.97.
        candidate_beats = [
            b for b in beat_events
            if b.get("label", "None") not in ("None",) and b.get("conf", 0.0) >= 0.97
        ]

        # Step 2: Rhythm trust gate — when the Rhythm model is confident about Sinus,
        # apply a stricter threshold for small beat counts (1-2 beats).
        # This prevents 1-2 hallucinated beats from escalating into Couplet/Run patterns
        # while still allowing genuine single beats (0.99+) and 3+ beat patterns through.
        # Non-Sinus backgrounds (AF, Block, VT) are unaffected — full 0.97 applies.
        rhythm_is_sinus = ml_label in (
            "Sinus Rhythm", "Sinus Bradycardia", "Sinus Tachycardia", "Unknown"
        )
        if rhythm_is_sinus and ml_conf > 0.65 and len(candidate_beats) < 3:
            candidate_beats = [b for b in candidate_beats if b.get("conf", 0.0) >= 0.99]

        # Step 3: Density gate — scattered hallucination suppression.
        # If >60% of all beats in the segment are labeled ectopic on a Sinus background,
        # the ectopy model is almost certainly confused (genuine isolated ectopy is <30%).
        #
        # CRITICAL EXCEPTION: skip this gate if 3+ candidates have consecutive beat_indices.
        # That means a real run is forming (SVT/VT/NSVT/Atrial Run) where 100% of beats
        # CAN legitimately be ectopic. Only suppress SCATTERED high-density patterns.
        total_beats = len(beat_events)
        if (rhythm_is_sinus and ml_conf > 0.70
                and total_beats > 4
                and len(candidate_beats) / total_beats > 0.60):
            sorted_indices = sorted(b.get("beat_idx", -1) for b in candidate_beats)
            has_consecutive_run = (
                len(sorted_indices) >= 3 and
                any(
                    sorted_indices[i+1] == sorted_indices[i] + 1 and
                    sorted_indices[i+2] == sorted_indices[i] + 2
                    for i in range(len(sorted_indices) - 2)
                )
            )
            if not has_consecutive_run:
                candidate_beats = []  # Scattered high-density ectopy → hallucination → suppress

        # Step 4: Template correlation refinement — override low-confidence ML calls
        # using BATCH_PROCESS morphological scoring (Pearson + Ashman + T-deformation).
        # Only fires when there are candidate beats and a clean signal available.
        if candidate_beats and _signal is not None and _r_peaks is not None:
            _per_beat = clinical_features.get("_per_beat_delineation", [])
            if _per_beat:
                from decision_engine.beat_classifier import refine_beat_labels
                candidate_beats = refine_beat_labels(
                    beat_events=candidate_beats,
                    signal=np.asarray(_signal, dtype=np.float32),
                    r_peaks=np.asarray(_r_peaks, dtype=int),
                    per_beat_delineation=_per_beat,
                    fs=_fs,
                )

        for b in candidate_beats:
            t_center = b["peak_sample"] / seg_fs
            ml_events.append(Event(
                event_id=str(uuid.uuid4()),
                event_type=b["label"],
                event_category=EventCategory.ECTOPY,
                start_time=max(0.0, t_center - 0.3),
                end_time=min(10.0, t_center + 0.3),
                beat_indices=[b["beat_idx"]],   # sequential index — rules use diffs to detect patterns
                priority=10,
                used_for_training=True,
            ))

        # Combine
        decision.events = rule_events + ml_events
        
        # 5. Apply Complex Logic (Phase 2)
        apply_ectopy_patterns(decision.events, clinical_features)
        
        # Promote high-priority events to background_rhythm
        # Priority: VF > VT > AF/AFL > NSVT > everything else
        def _has(event_types):
            return next((e for e in decision.events if e.event_type in event_types), None)

        vf_event = _has(["Ventricular Fibrillation"])
        vt_event = _has(["VT", "Ventricular Tachycardia"])
        af_event = _has(["AF", "Atrial Fibrillation", "Atrial Flutter"])
        nsvt_event = _has(["NSVT", "Ventricular Run"])

        if vf_event:
            decision.background_rhythm = "Ventricular Fibrillation"
        elif vt_event:
            decision.background_rhythm = "Ventricular Tachycardia"
        elif af_event:
            decision.background_rhythm = af_event.event_type
        elif nsvt_event:
            decision.background_rhythm = nsvt_event.event_type
        
        decision.final_display_events = apply_display_rules(
            decision.background_rhythm,
            decision.events
        )
        
        apply_training_flags(decision.events)
        
        # Add XAI notes for debugging/verification
        decision.xai_notes = {
            "initial_ml_label": ml_label,
            "initial_ml_conf": ml_conf,
            "clinical_hr": clinical_features.get("mean_hr")
        }

        return decision

    def _detect_background_rhythm(self, features: Dict[str, Any]) -> str:
        """
        Determines background rhythm from HR and P-wave ratio.
        Used only for the SQI-unreliable path — main detection is in sinus_detector + lethal_detector.
        """
        hr_val  = features.get("mean_hr") or features.get("mean_hr_bpm")
        hr      = float(hr_val) if hr_val is not None else 0.0
        p_ratio = float(features.get("p_wave_present_ratio") or 1.0)

        if hr == 0:
            return "Unknown"

        if hr < 60:
            # Slow rhythm — check for escape pacemakers
            if p_ratio < 0.3:
                return "Junctional Rhythm"        # Narrow, no P-waves, slow
            return "Sinus Bradycardia"
        elif hr > 100:
            return "Sinus Tachycardia"
        else:
            if p_ratio < 0.2:
                return "Junctional Rhythm"        # Normal rate but no P-waves
            return "Sinus Rhythm"

    def _create_event_from_ml(self, label: str, ml_prediction: Dict[str, Any]) -> Event:
        """Creates an Event object from ML prediction."""
        
        # Determine Category
        category = EventCategory.RHYTHM
        # Simple heuristic list - to be expanded
        if label in ["PVC", "PAC", "Bigeminy", "Trigeminy", "Couplet"]:
             category = EventCategory.ECTOPY

        # Determine Priority
        priority = 50
        if label in ["VF", "Ventricular Fibrillation"]:
            priority = 100
        elif label in ["VT", "Ventricular Tachycardia"]:
            priority = 95
        elif label == "NSVT":
            priority = 90
        elif label in ["Atrial Fibrillation", "AFib", "AF", "Atrial Flutter"]:
            priority = 88
        elif "Block" in label:
            priority = 70
        elif label in ["Sinus Bradycardia"]:
            priority = 20
        elif label in ["PVC", "PAC"]:
            priority = 10
            
        return Event(
            event_id=str(uuid.uuid4()),
            event_type=label,
            event_category=category,
            start_time=0.0, # Placeholder, needs segment start context passed to decide if we want absolute time
            end_time=10.0,  # Placeholder for 10s segment
            ml_evidence=ml_prediction,
            priority=priority,
            used_for_training=True 
        )
