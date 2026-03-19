import uuid
import numpy as np
from typing import Dict, List, Any, Optional

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

        # 3. Background rhythm FIRST (Simple Rule-Based for now)
        decision.background_rhythm = self._detect_background_rhythm(clinical_features)

        # 4. Gather Events
        # A) Rule-Derived Events
        rule_events = derive_rule_events(clinical_features)
        
        # B) ML-Derived Events
        ml_events = []
        # explain_segment() returns {"rhythm": {...}, "ectopy": {...}} — extract nested fields
        _rhythm_block = ml_prediction.get("rhythm") or {}
        ml_label = _rhythm_block.get("label") or ml_prediction.get("label", "Unknown")
        ml_conf = _rhythm_block.get("confidence") or ml_prediction.get("confidence", 0.0)

        if ml_label not in ["Sinus Rhythm", "Unknown"]:
            ml_events.append(self._create_event_from_ml(ml_label, ml_conf, ml_prediction, clinical_features))

        # C) Per-beat ectopy events — required for bigeminy/trigeminy pattern detection
        #    xai.py now returns beat_events: [{beat_idx, peak_sample, label, conf}, ...]
        beat_events = ml_prediction.get("ectopy", {}).get("beat_events", [])
        seg_fs = float(clinical_features.get("fs", 125))
        for b in beat_events:
            if b.get("label", "None") in ("None",):
                continue  # skip normal beats
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
        apply_ectopy_patterns(decision.events)
        
        # FIX: AF Dominance - Promote to background rhythm if detected
        af_event = next((e for e in decision.events if e.event_type in ["AF", "Atrial Fibrillation", "Atrial Flutter"]), None)
        if af_event:
            decision.background_rhythm = af_event.event_type
        
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
        """Determines background rhythm (Sinus variants) from HR."""
        hr_val = features.get("mean_hr")
        hr = float(hr_val) if hr_val is not None else 0.0
        
        if hr == 0:
            return "Unknown"
        
        if hr < 60:
             return "Sinus Bradycardia"
        elif hr > 150:
             return "Sinus Tachycardia" # Or SVT, but "Sinus Tachycardia" for background usually
        elif hr > 100:
             return "Sinus Tachycardia"
        else:
             return "Sinus Rhythm"

    def _create_event_from_ml(self, label: str, conf: float, ml_prediction: Dict[str, Any], features: Dict[str, Any]) -> Event:
        """Creates an Event object from ML prediction."""
        
        # Determine Category
        category = EventCategory.RHYTHM
        # Simple heuristic list - to be expanded
        if label in ["PVC", "PAC", "Bigeminy", "Trigeminy", "Couplet", "Run", "Ventricular Run", "Atrial Run"]:
             category = EventCategory.ECTOPY
        
        # Determine Priority (Stub)
        priority = 50
        if label in ["VT", "Ventricular Tachycardia", "VF", "Ventricular Fibrillation"]:
            priority = 100
        elif label in ["Atrial Fibrillation", "AFib", "AF", "Atrial Flutter"]:
            priority = 90
        elif label in ["SVT", "PSVT", "Supraventricular Tachycardia"]:
            priority = 80
        elif "Block" in label:
            priority = 70
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
