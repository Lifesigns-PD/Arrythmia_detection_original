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
    #    Requires: enough beats for reliable CV (>5),
    #              high RR variability (CV > 0.15),
    #              AND absent P-waves (either PR < 10 OR p_wave_present_ratio < 0.3).
    #    The P-wave ratio guard prevents false AF on sinus strips where
    #    P-wave detection fails (returning PR=0) due to noise.
    # ---------------------------------------------------------
    is_af = False
    p_wave_ratio = features.get("p_wave_present_ratio", None)
    if len(rr_arr) > 5:
        p_waves_absent_pr = (pr < 10)
        p_waves_absent_ratio = (p_wave_ratio is not None and p_wave_ratio < 0.3)
        # Both checks must agree, OR use ratio when available (more reliable)
        if p_wave_ratio is not None:
            p_waves_absent = p_waves_absent_ratio
        else:
            p_waves_absent = p_waves_absent_pr
        if cv > 0.15 and p_waves_absent:
            is_af = True
            events.append(Event(
                event_id=str(uuid.uuid4()),
                event_type="AF",
                event_category=EventCategory.RHYTHM,
                start_time=0.0, end_time=10.0,
                rule_evidence={"rule": "AF_Strict", "cv": cv, "pr": pr,
                               "p_wave_ratio": p_wave_ratio},
                priority=90,
                used_for_training=True
            ))

    # ---------------------------------------------------------
    # 2. SVT / VT — REMOVED from rules. Now purely derived from
    #    consecutive PAC/PVC beat counts in apply_ectopy_patterns().
    # ---------------------------------------------------------

    # Per-beat PR intervals (needed for 2nd/3rd degree block detection)
    per_beat_pr = []
    raw_pbpr = features.get("per_beat_pr_ms", [])
    if isinstance(raw_pbpr, list):
        per_beat_pr = [x for x in raw_pbpr if isinstance(x, (int, float)) and x > 0]

    # ---------------------------------------------------------
    # 3a. Bundle Branch Block (QRS > 120 ms)
    # ---------------------------------------------------------
    if qrs_mean > 120 and not is_af:
        events.append(Event(
            event_id=str(uuid.uuid4()),
            event_type="Bundle Branch Block",
            event_category=EventCategory.RHYTHM,
            start_time=0.0, end_time=10.0,
            rule_evidence={"rule": "BBB_WideQRS", "qrs_mean_ms": round(qrs_mean, 1)},
            priority=70,
            used_for_training=False
        ))

    # ---------------------------------------------------------
    # 3b. AV Blocks (1st, 2nd Type 1/2, 3rd)
    # Only fire when not AF (AF has no organised P-waves → PR is meaningless)
    # ---------------------------------------------------------
    block_fired = False  # track highest-grade block to avoid dual firing

    # 3rd Degree (Complete AV Block) — most specific, check first
    # Criteria: very slow escape HR, P-waves present, PR intervals highly variable (AV dissociation)
    if (not is_af and hr > 0 and hr < 50 and pr > 0
            and len(per_beat_pr) >= 3
            and float(np.std(per_beat_pr)) > 40):
        events.append(Event(
            event_id=str(uuid.uuid4()),
            event_type="3rd Degree AV Block",
            event_category=EventCategory.RHYTHM,
            start_time=0.0, end_time=10.0,
            rule_evidence={"rule": "3rdDegBlock_AVDissociation",
                           "hr": round(hr, 1), "pr_std": round(float(np.std(per_beat_pr)), 1)},
            priority=80,
            used_for_training=False
        ))
        block_fired = True

    # 2nd Degree Type 2 (Mobitz II) — fixed PR + sudden dropped beat
    # Criteria: PR intervals constant (CV < 10%), RR gap ≥ 1.7× median (dropped beat without warning)
    if (not is_af and not block_fired and len(per_beat_pr) >= 2 and len(rr_arr) >= 3):
        pr_mean = float(np.mean(per_beat_pr))
        pr_cv = float(np.std(per_beat_pr)) / pr_mean if pr_mean > 0 else 1.0
        median_rr = float(np.median(rr_arr))
        if pr_cv < 0.10 and float(np.max(rr_arr)) > 1.7 * median_rr:
            events.append(Event(
                event_id=str(uuid.uuid4()),
                event_type="2nd Degree AV Block Type 2",
                event_category=EventCategory.RHYTHM,
                start_time=0.0, end_time=10.0,
                rule_evidence={"rule": "2ndDegBlock_MobitzII",
                               "pr_cv": round(pr_cv, 3), "max_rr_ms": round(float(np.max(rr_arr)), 1)},
                priority=70,
                used_for_training=False
            ))
            block_fired = True

    # 2nd Degree Type 1 (Wenckebach) — progressive PR lengthening + dropped beat
    # Criteria: majority of consecutive PR diffs > +10 ms AND one RR gap > 1.5× median
    if (not is_af and not block_fired and len(per_beat_pr) >= 3 and len(rr_arr) >= 3):
        pr_diffs = np.diff(per_beat_pr)
        increasing = int(np.sum(pr_diffs > 10)) >= len(pr_diffs) // 2
        median_rr = float(np.median(rr_arr))
        if increasing and float(np.max(rr_arr)) > 1.5 * median_rr:
            events.append(Event(
                event_id=str(uuid.uuid4()),
                event_type="2nd Degree AV Block Type 1",
                event_category=EventCategory.RHYTHM,
                start_time=0.0, end_time=10.0,
                rule_evidence={"rule": "2ndDegBlock_Wenckebach",
                               "pr_increasing_beats": int(np.sum(pr_diffs > 10)),
                               "max_rr_ms": round(float(np.max(rr_arr)), 1)},
                priority=65,
                used_for_training=False
            ))
            block_fired = True

    # 1st Degree AV Block — only if no higher-grade block already fired
    if not is_af and not block_fired and pr > 200:
        events.append(Event(
            event_id=str(uuid.uuid4()),
            event_type="1st Degree AV Block",
            event_category=EventCategory.RHYTHM,
            start_time=0.0, end_time=10.0,
            rule_evidence={"rule": "1stDegBlock", "pr": pr},
            priority=50,
            used_for_training=False
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

    Count-Based Rules (no rate guard):
    PVC: 2=Couplet, 3=Ventricular Run, 4-10=NSVT, 11+=VT
    PAC: 2=Atrial Couplet, 3-5=Atrial Run, 6-10=PSVT, 11+=SVT
    Bigeminy/Trigeminy/Quadrigeminy: interspersed patterns via beat_indices diffs
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
                # Without beat_indices, we CANNOT distinguish Bigeminy from a slow Run.
                # SAFE DEFAULT: Treat all clusters as consecutive runs.
                time_gaps = [cluster[i+1].start_time - cluster[i].start_time for i in range(count - 1)]

                if len(time_gaps) >= 1:
                    mean_gap = np.mean(time_gaps)
                    std_gap = np.std(time_gaps)
                    cv = std_gap / mean_gap if mean_gap > 0 else float('inf')
                    is_consecutive = (cv < 0.35) if count >= 3 else (cv < 0.25)
                else:
                    is_consecutive = (count == 2)

                # Never detect Bigeminy/Trigeminy without beat_indices
                is_bigeminy = False
                is_trigeminy = False
                is_quadrigeminy = False


            # Rule 1: Bigeminy/Trigeminy/Quadrigeminy (Interspersed patterns)
            if is_bigeminy or is_trigeminy or is_quadrigeminy:
                if is_bigeminy: pattern_name = "Bigeminy"
                elif is_trigeminy: pattern_name = "Trigeminy"
                else: pattern_name = "Quadrigeminy"
                priority = 55

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
                for e in cluster: e.pattern_label = pattern_name

            # ── PVC consecutive count rules ──
            elif is_consecutive and target_type == "PVC":
                if count >= 11:
                    # VT: 11+ consecutive PVCs
                    new_event = Event(
                        event_id=str(uuid.uuid4()),
                        event_type="VT",
                        event_category=EventCategory.RHYTHM,
                        start_time=cluster[0].start_time,
                        end_time=cluster[-1].end_time,
                        pattern_label="Run",
                        rule_evidence={"rule": "VT_Detected", "count": count, "rate": round(rate, 1)},
                        priority=100,
                        used_for_training=False,
                    )
                    events.append(new_event)
                    for e in cluster: e.pattern_label = "Run"
                elif count >= 4:
                    # NSVT: 4-10 consecutive PVCs
                    new_event = Event(
                        event_id=str(uuid.uuid4()),
                        event_type="NSVT",
                        event_category=EventCategory.RHYTHM,
                        start_time=cluster[0].start_time,
                        end_time=cluster[-1].end_time,
                        pattern_label="Run",
                        rule_evidence={"rule": "NSVT_Detected", "count": count, "rate": round(rate, 1)},
                        priority=90,
                        used_for_training=False,
                    )
                    events.append(new_event)
                    for e in cluster: e.pattern_label = "Run"
                elif count == 3:
                    # Ventricular Run: exactly 3 consecutive PVCs
                    new_event = Event(
                        event_id=str(uuid.uuid4()),
                        event_type="Ventricular Run",
                        event_category=EventCategory.RHYTHM,
                        start_time=cluster[0].start_time,
                        end_time=cluster[-1].end_time,
                        pattern_label="Run",
                        rule_evidence={"rule": "Ventricular_Run_Detected", "count": 3, "rate": round(rate, 1)},
                        priority=40,
                        used_for_training=False,
                    )
                    events.append(new_event)
                    for e in cluster: e.pattern_label = "Run"
                elif count == 2:
                    # PVC Couplet: exactly 2 consecutive PVCs
                    new_event = Event(
                        event_id=str(uuid.uuid4()),
                        event_type="PVC Couplet",
                        event_category=EventCategory.ECTOPY,
                        start_time=cluster[0].start_time,
                        end_time=cluster[-1].end_time,
                        pattern_label="Couplet",
                        rule_evidence={"rule": "PVC_Couplet_Detected", "count": 2},
                        priority=30,
                        used_for_training=True,
                    )
                    events.append(new_event)
                    for e in cluster: e.pattern_label = "Couplet"

            # ── PAC consecutive count rules ──
            elif is_consecutive and target_type == "PAC":
                if count >= 11:
                    # SVT: 11+ consecutive PACs
                    new_event = Event(
                        event_id=str(uuid.uuid4()),
                        event_type="SVT",
                        event_category=EventCategory.RHYTHM,
                        start_time=cluster[0].start_time,
                        end_time=cluster[-1].end_time,
                        pattern_label="Run",
                        rule_evidence={"rule": "SVT_Detected", "count": count, "rate": round(rate, 1)},
                        priority=80,
                        used_for_training=False,
                    )
                    events.append(new_event)
                    for e in cluster: e.pattern_label = "Run"
                elif count >= 6:
                    # PSVT: 6-10 consecutive PACs
                    new_event = Event(
                        event_id=str(uuid.uuid4()),
                        event_type="PSVT",
                        event_category=EventCategory.RHYTHM,
                        start_time=cluster[0].start_time,
                        end_time=cluster[-1].end_time,
                        pattern_label="Run",
                        rule_evidence={"rule": "PSVT_Detected", "count": count, "rate": round(rate, 1)},
                        priority=85,
                        used_for_training=False,
                    )
                    events.append(new_event)
                    for e in cluster: e.pattern_label = "Run"
                elif count >= 3:
                    # Atrial Run: 3-5 consecutive PACs
                    new_event = Event(
                        event_id=str(uuid.uuid4()),
                        event_type="Atrial Run",
                        event_category=EventCategory.RHYTHM,
                        start_time=cluster[0].start_time,
                        end_time=cluster[-1].end_time,
                        pattern_label="Run",
                        rule_evidence={"rule": "Atrial_Run_Detected", "count": count, "rate": round(rate, 1)},
                        priority=40,
                        used_for_training=False,
                    )
                    events.append(new_event)
                    for e in cluster: e.pattern_label = "Run"
                elif count == 2:
                    # Atrial Couplet: exactly 2 consecutive PACs
                    new_event = Event(
                        event_id=str(uuid.uuid4()),
                        event_type="Atrial Couplet",
                        event_category=EventCategory.ECTOPY,
                        start_time=cluster[0].start_time,
                        end_time=cluster[-1].end_time,
                        pattern_label="Couplet",
                        rule_evidence={"rule": "Atrial_Couplet_Detected", "count": 2},
                        priority=30,
                        used_for_training=True,
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
    has_af = any(e.event_type in ["AF", "Atrial Fibrillation", "Atrial Flutter"] for e in events)
    has_svt = any(e.event_type in ["SVT", "PSVT", "Atrial Run"] for e in events)
    has_vt = any(e.event_type in ["VT", "NSVT", "Ventricular Run"] for e in events)

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
        # Atrial (beat-level ectopy — trains ectopy model)
        "PAC", "Atrial Couplet",           # "PAC Couplet" renamed to "Atrial Couplet" in rules
        "PAC Bigeminy", "PAC Trigeminy", "PAC Quadrigeminy",
        "AF", "Atrial Fibrillation", "Atrial Flutter",

        # Ventricular (beat-level ectopy — trains ectopy model)
        "PVC", "PVC Couplet",
        "PVC Bigeminy", "PVC Trigeminy", "PVC Quadrigeminy",
        "Ventricular Fibrillation",

        # Blocks — train the rhythm model (indices 6-10 in RHYTHM_CLASS_NAMES)
        "1st Degree AV Block", "2nd Degree AV Block Type 1",
        "2nd Degree AV Block Type 2", "3rd Degree AV Block",
        "Bundle Branch Block",

        # NOTE: SVT, VT, NSVT, PSVT, Ventricular Run, Atrial Run are
        # RULES-ONLY and should NOT train the ML models.
    }
    for event in events:
        # Never train on Sinus or Artifact as primary labels to avoid baseline bias
        if event.event_type in ["Sinus Rhythm", "Sinus Bradycardia", "Sinus Tachycardia", "Artifact"]:
            event.used_for_training = False
        elif event.event_type in training_set:
            event.used_for_training = True
        else:
            event.used_for_training = False
