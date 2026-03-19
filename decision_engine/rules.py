import numpy as np
import uuid
from typing import List, Dict, Any, Optional
from decision_engine.models import Event, EventCategory, DisplayState

# =============================================================================
# 1. RULE-BASED EVENT DERIVATION
# =============================================================================

def derive_rule_events(features: Dict[str, Any]) -> List[Event]:
    """
    Analyzes clinical features to detect arrhythmias strictly via rules.
    Returns a list of 'Rule-Derived Events' (AF, SVT, VT, AV Blocks, Pauses).
    Sinus rhythms are handled as background only and NEVER produce events.
    """
    events = []
    
    # Extract features safely
    hr_val = features.get("mean_hr")
    hr = float(hr_val) if hr_val is not None else 0.0
    
    pr_val = features.get("pr_interval")
    pr = float(pr_val) if pr_val is not None else 0.0
    
    rr_intervals = features.get("rr_intervals_ms", [])
    rr_arr = np.array([])
    if isinstance(rr_intervals, list) and len(rr_intervals) > 2:
        rr_arr = np.array([x for x in rr_intervals if x is not None and isinstance(x, (int, float))])
        
    qrs_mean = 0.0
    try:
        raw_qrs = features.get("qrs_durations_ms")
        if isinstance(raw_qrs, list):
            q_list = [x for x in raw_qrs if isinstance(x, (int, float))]
            if q_list: 
                qrs_mean = float(sum(q_list) / len(q_list))
    except: pass
    
    cv = 0.0
    if len(rr_arr) > 3:
        rr_std = np.std(rr_arr)
        rr_mean = np.mean(rr_arr)
        cv = rr_std / rr_mean if rr_mean > 0 else 0

    # ---------------------------------------------------------
    # 1. Atrial Fibrillation (Irregular + No P-wave)
    # ---------------------------------------------------------
    is_af = False
    if len(rr_arr) > 3:
        p_waves_absent = (pr < 10)
        if cv > 0.15 and p_waves_absent:
            is_af = True
            events.append(Event(
                event_id=str(uuid.uuid4()),
                event_type="AF",
                event_category=EventCategory.RHYTHM,
                start_time=0.0, end_time=10.0,
                rule_evidence={"rule": "AF_Strict", "cv": cv, "pr": pr},
                priority=90,
                used_for_training=True
            ))

    # ---------------------------------------------------------
    # 2. SVT (Regular + Fast + Narrow + Not AF/VT)
    # ---------------------------------------------------------
    # Criteria: Regular (CV < 0.08) + Narrow QRS (< 120ms) + HR > 130 + Not AF
    if hr > 130 and qrs_mean < 120 and cv < 0.08 and not is_af:
        events.append(Event(
            event_id=str(uuid.uuid4()),
            event_type="SVT",
            event_category=EventCategory.RHYTHM,
            start_time=0.0, end_time=10.0,
            rule_evidence={"rule": "SVT_Strict", "hr": hr, "qrs": qrs_mean, "cv": cv},
            priority=80,
            used_for_training=True
        ))

    # ---------------------------------------------------------
    # 3. VT (Wide QRS + HR > 100)
    # ---------------------------------------------------------
    if hr > 100 and qrs_mean >= 120:
        events.append(Event(
            event_id=str(uuid.uuid4()),
            event_type="VT",
            event_category=EventCategory.RHYTHM,
            start_time=0.0, end_time=10.0,
            rule_evidence={"rule": "VT_Rule", "hr": hr, "qrs": qrs_mean},
            priority=100,
            used_for_training=True
        ))

    # ---------------------------------------------------------
    # 4. AV Blocks
    # ---------------------------------------------------------
    if pr > 200:
        events.append(Event(
            event_id=str(uuid.uuid4()),
            event_type="1st Degree AV Block",
            event_category=EventCategory.RHYTHM,
            start_time=0.0, end_time=10.0,
            rule_evidence={"rule": "1stDegBlock", "pr": pr},
            priority=50,
            used_for_training=False # Never train on AV block
        ))

    # ---------------------------------------------------------
    # 5. Pause
    # ---------------------------------------------------------
    if any(rr > 2000 for rr in rr_arr):
        events.append(Event(
            event_id=str(uuid.uuid4()),
            event_type="Pause",
            event_category=EventCategory.RHYTHM,
            start_time=0.0, end_time=10.0,
            rule_evidence={"rule": "Pause_Detected"},
            priority=85,
            used_for_training=False # Never train on Pause
        ))

    return events


# =============================================================================
# 2. ECTOPY PATTERN RECOGNITION
# =============================================================================

def apply_ectopy_patterns(events: List[Event]) -> None:
    """
    Scans ECTOPY events and applies pattern labels.
    
    Clinical Rules:
    - Couplet:    2 consecutive PVCs/PACs
    - Run:        3 consecutive (Ventricular Run / Atrial Run)
    - NSVT:       >3 (4+) consecutive PVCs, rate >= 100
    - PSVT:       >3 (4+) consecutive PACs, rate >= 100  
    - Bigeminy:   Every other beat is PVC/PAC (needs beat_indices)
    - Trigeminy:  Every 3rd beat is PVC/PAC (needs beat_indices)
    """
    ectopy = sorted(
        [e for e in events if e.event_category == EventCategory.ECTOPY],
        key=lambda e: e.start_time
    )
    
    if len(ectopy) < 2:
        return

    # Clustering Logic for PVCs and PACs
    for target_type in ["PVC", "PAC"]:
        # Clustering Logic (Increased gap to 2.0s for slow patterns)
        clusters = []
        current_cluster = []
        MAX_GAP = 2.0 # seconds
        
        target_events = [e for e in ectopy if target_type in e.event_type]
        
        for e in target_events:
            if not current_cluster:
                current_cluster.append(e)
            else:
                gap = e.start_time - current_cluster[-1].start_time
                if gap <= MAX_GAP:
                    current_cluster.append(e)
                else:
                    if len(current_cluster) >= 2: clusters.append(current_cluster)
                    current_cluster = [e]
        if len(current_cluster) >= 2: clusters.append(current_cluster)

        for cluster in clusters:
            count = len(cluster)
            duration = cluster[-1].start_time - cluster[0].start_time
            rate = (count - 1) * (60.0 / duration) if duration > 0 else 0
            
            # Pattern Recognition via Beat Indices
            indices = []
            for e in cluster:
                if e.beat_indices: indices.append(e.beat_indices[0])
            
            has_indices = len(indices) == count  # All events have beat_indices
            
            if has_indices and len(indices) >= 2:
                # Primary path: use beat indices for precise pattern detection
                diffs = np.diff(indices)
                is_consecutive = all(d == 1 for d in diffs)
                # Require at least 2 cycles (3 beats) for Bigeminy/Trigeminy as per new UI rules
                is_bigeminy = len(diffs) >= 2 and all(d == 2 for d in diffs)
                is_trigeminy = len(diffs) >= 2 and all(d == 3 for d in diffs)
                is_quadrigeminy = len(diffs) >= 2 and all(d == 4 for d in diffs)
            else:
                # Fallback path: use TIME intervals when beat_indices are missing
                # (common with manual cardiologist annotations)
                #
                # IMPORTANT: Without beat_indices, we CANNOT distinguish Bigeminy
                # (every other beat) from a slow Run (consecutive beats at ~60bpm).
                # Both look like events spaced ~1s apart.
                #
                # SAFE DEFAULT: Treat all clusters as consecutive runs.
                # Bigeminy/Trigeminy can ONLY be detected via beat_indices.
                time_gaps = [cluster[i+1].start_time - cluster[i].start_time for i in range(count - 1)]
                
                if len(time_gaps) >= 1:
                    mean_gap = np.mean(time_gaps)
                    std_gap = np.std(time_gaps)
                    cv = std_gap / mean_gap if mean_gap > 0 else float('inf')
                    
                    # All events with consistent spacing are treated as consecutive
                    # The rate calculation will determine Run vs NSVT/PSVT
                    is_consecutive = (cv < 0.35) if count >= 3 else (cv < 0.25)
                else:
                    is_consecutive = (count == 2)  # Pairs are always consecutive
                
                # Never detect Bigeminy/Trigeminy without beat_indices
                is_bigeminy = False
                is_trigeminy = False
                is_quadrigeminy = False


            # Rule 1: Bigeminy/Trigeminy/Quadrigeminy (Interspersed patterns)
            if is_bigeminy or is_trigeminy or is_quadrigeminy:
                if is_bigeminy: pattern_name = "Bigeminy"
                elif is_trigeminy: pattern_name = "Trigeminy"
                else: pattern_name = "Quadrigeminy"
                priority = 55 # Greater than isolated (10), less than Run (80)
                
                new_event = Event(
                    event_id=str(uuid.uuid4()),
                    event_type=f"{target_type} {pattern_name}",
                    event_category=EventCategory.RHYTHM,
                    start_time=cluster[0].start_time,
                    end_time=cluster[-1].end_time,
                    pattern_label=pattern_name,
                    rule_evidence={"rule": f"{target_type}_{pattern_name}_Pattern", "count": count},
                    priority=priority,
                    used_for_training=True
                )
                events.append(new_event)
                # Label individual beats for dashboard highlight
                for e in cluster: e.pattern_label = pattern_name

            # Rule 2: NSVT / PSVT (>3 consecutive beats, i.e. 4+, rate >= 100 bpm)
            # Rate guard: bigeminy PVC-to-PVC rate ~35 bpm, NSVT requires >= 100 bpm by definition
            elif is_consecutive and count > 3 and rate >= 100:
                event_type = "NSVT" if target_type == "PVC" else "PSVT"
                priority = 90 if target_type == "PVC" else 85
                
                new_event = Event(
                    event_id=str(uuid.uuid4()),
                    event_type=event_type,
                    event_category=EventCategory.RHYTHM,
                    start_time=cluster[0].start_time,
                    end_time=cluster[-1].end_time,
                    pattern_label="Run",
                    rule_evidence={"rule": f"{event_type}_Detected", "count": count, "rate": round(rate, 1)},
                    priority=priority,
                    used_for_training=True
                )
                events.append(new_event)

            # Rule 3: Triplet (exactly 3 consecutive beats)
            elif is_consecutive and count == 3:
                event_type = "Ventricular Triplet" if target_type == "PVC" else "Atrial Triplet"
                new_event = Event(
                    event_id=str(uuid.uuid4()),
                    event_type=event_type,
                    event_category=EventCategory.RHYTHM,
                    start_time=cluster[0].start_time,
                    end_time=cluster[-1].end_time,
                    pattern_label="Run",
                    rule_evidence={"rule": f"{event_type}_Detected", "count": 3, "rate": round(rate, 1)},
                    priority=40,
                    used_for_training=True
                )
                events.append(new_event)

            # Rule 4: Couplet (exactly 2 consecutive beats)
            elif is_consecutive and count == 2:
                couplet_type = "PVC Couplet" if target_type == "PVC" else "Atrial Couplet"
                new_event = Event(
                    event_id=str(uuid.uuid4()),
                    event_type=couplet_type,
                    event_category=EventCategory.ECTOPY,
                    start_time=cluster[0].start_time,
                    end_time=cluster[-1].end_time,
                    pattern_label="Couplet",
                    rule_evidence={"rule": f"{couplet_type}_Detected", "count": 2},
                    priority=30,
                    used_for_training=True
                )
                events.append(new_event)
                for e in cluster: e.pattern_label = "Couplet"
            


# =============================================================================
# 3. DISPLAY ARBITRATION RULES
# =============================================================================

def apply_display_rules(background_rhythm: str, events: List[Event]) -> List[Event]:
    # # DEBUG MODE: Show all AI events for transition.
    # # We temporarily disable the veto logic so the user can audit model performance.
    # has_manual_sinus = any(
    #     e.event_type == "Sinus Rhythm" and getattr(e, "annotation_source", "") == "cardiologist"
    #     for e in events
    # )

    has_manual_sinus = False # FORCE DISABLE VETO
    
    # Pass 1: Global Hierarchy & Veto
    has_af = any(e.event_type in ["Atrial Fibrillation", "Atrial Flutter"] for e in events)
    has_svt = any(e.event_type in ["SVT", "Atrial Run (PSVT)", "Atrial Triplet", "PSVT"] for e in events)
    has_vt = any(e.event_type in ["VT", "NSVT", "Ventricular Triplet"] for e in events)

    for event in events:
        should_display = True
        suppression_reason = None
        
        # AI Visibility: Never hide AI events during this validation phase
        is_ai_event = getattr(event, "annotation_source", "ai") != "cardiologist"
        
        if is_ai_event:
            should_display = True # Force display for AI auditing
            event.display_state = DisplayState.DISPLAYED
            continue

        # 2. APPLY VETO (Disabled)
        # if has_manual_sinus and is_ai_event and event.event_category == EventCategory.RHYTHM:
        #      should_display = False
        #      suppression_reason = "Cardiologist Veto (Sinus)"

        # Rule A: Life-Threatening
        if event.priority >= 95:
             should_display = True
            
        # Rule B: AF Dominance (Show AF as background, allow Ectopy on top)
        # CRITICAL: When both AFib and Ectopy are present, BOTH must be displayed.
        # AFib is the background rhythm (primary finding), Ectopy is concurrent (additional finding).
        elif has_af:
            if event.event_type in ["AF", "Atrial Fibrillation", "Atrial Flutter"]:
                # Always show the AF event itself (as it informs the background rhythm)
                should_display = True
            elif event.event_category == EventCategory.ECTOPY:
                # Always show Ectopy on top of AF (PVCs/PACs are secondary findings)
                should_display = True
            elif event.event_category == EventCategory.RHYTHM:
                # Suppress other conflicting RHYTHM types (only one rhythm/background at a time)
                should_display = False
                suppression_reason = "AF Dominance"
            else:
                should_display = True

        # Rule C: Run Dominance (New) - Suppress individual beats if a Run/Tachycardia is present
        elif has_svt and event.event_type == "PAC":
            should_display = False
            suppression_reason = "SVT/PSVT Dominance"
        elif has_vt and event.event_type == "PVC":
            should_display = False
            suppression_reason = "VT/NSVT Dominance"

        # Rule D: Background Suppression
        elif "Sinus" in event.event_type:
             if getattr(event, "annotation_source", "") == "cardiologist":
                 should_display = True # Show the doctor's manual tag
             else:
                 should_display = False
                 suppression_reason = "Background Rhythm"
             
        event.display_state = DisplayState.DISPLAYED if should_display else DisplayState.HIDDEN
        event.suppressed_by = suppression_reason

    # Pass 2: Artifact Suppression
    displayed_count = sum(1 for e in events if e.display_state == DisplayState.DISPLAYED and e.event_type != "Artifact" and e.event_type != "Sinus Rhythm")
    for event in events:
        if event.event_type == "Artifact":
            event.display_state = DisplayState.HIDDEN if displayed_count > 0 else DisplayState.DISPLAYED
    
    final_list = [e for e in events if e.display_state == DisplayState.DISPLAYED]
    final_list.sort(key=lambda x: x.priority, reverse=True)
    return final_list


# =============================================================================
# 4. TRAINING FLAG LOGIC
# =============================================================================

def apply_training_flags(events: List[Event]) -> None:
    """
    Sets used_for_training flag based on event type.
    We train the Morphology specialist on single beats/couplets,
    and the Rhythm specialist on Runs/Rhythms.
    """
    training_set = {
        # Atrial
        "PAC", "PAC Couplet", "Atrial Triplet", "PSVT",
        "PAC Bigeminy", "PAC Trigeminy", "PAC Quadrigeminy",
        "AF", "Atrial Fibrillation", "Atrial Flutter", "SVT",
        
        # Ventricular
        "PVC", "PVC Couplet", "Ventricular Triplet", "NSVT",
        "PVC Bigeminy", "PVC Trigeminy", "PVC Quadrigeminy",
        "VT", "Ventricular Tachycardia", "Ventricular Fibrillation",
        
        # Blocks
        "1st Degree AV Block", "2nd Degree AV Block Type 1", "3rd Degree AV Block"
    }
    for event in events:
        # Never train on Sinus or Artifact as primary labels to avoid baseline bias
        if event.event_type in ["Sinus Rhythm", "Sinus Bradycardia", "Sinus Tachycardia", "Artifact"]:
            event.used_for_training = False
        elif event.event_type in training_set:
            event.used_for_training = True
        else:
            event.used_for_training = False
