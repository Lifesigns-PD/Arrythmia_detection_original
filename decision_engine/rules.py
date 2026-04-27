import numpy as np
import uuid
from typing import List, Dict, Any, Optional
from decision_engine.models import Event, EventCategory, DisplayState

# =============================================================================
# 1. RULE-BASED EVENT DERIVATION
# =============================================================================

def _detect_flutter_waves(signal: np.ndarray, r_peaks: np.ndarray, fs: int) -> bool:
    """
    Detects atrial flutter waves via spectral analysis of QRS-blanked signal.
    AFL = organized atrial rate 250-350 bpm → dominant FFT peak at 4-6 Hz.
    """
    if len(signal) < fs or len(r_peaks) == 0:
        return False
    try:
        blanked = signal.copy().astype(np.float64)
        for peak in r_peaks:
            s = max(0, int(peak) - int(0.10 * fs))
            e = min(len(blanked), int(peak) + int(0.15 * fs))
            blanked[s:e] = 0.0
        freqs = np.fft.rfftfreq(len(blanked), 1.0 / fs)
        psd   = np.abs(np.fft.rfft(blanked)) ** 2
        band  = (freqs >= 4.0) & (freqs <= 6.5)   # 240–390 atrial bpm
        if not band.any():
            return False
        nonzero = psd[psd > 0]
        if len(nonzero) == 0:
            return False
        return float(psd[band].max()) > 2.5 * float(np.median(nonzero))
    except Exception:
        return False


def _classify_compensatory_pause(
    beat_seq_idx: int,
    r_peaks: np.ndarray,
    normal_rr: float,
) -> Optional[str]:
    """
    Returns 'PVC', 'PAC', or None based on compensatory pause analysis.
    PVC → full compensatory pause (SA node NOT reset): RR_before + RR_after ≈ 2×normal_RR
    PAC → incomplete pause (SA node IS reset): sum < 1.8×normal_RR
    """
    if beat_seq_idx <= 0 or beat_seq_idx >= len(r_peaks) - 1 or normal_rr <= 0:
        return None
    rr_before = float(r_peaks[beat_seq_idx] - r_peaks[beat_seq_idx - 1])
    rr_after  = float(r_peaks[beat_seq_idx + 1] - r_peaks[beat_seq_idx])
    total = rr_before + rr_after
    if total >= 1.85 * normal_rr:   # full compensatory → ventricular origin
        return "PVC"
    if total <= 1.60 * normal_rr:   # incomplete → atrial origin
        return "PAC"
    return None  # ambiguous — trust ML


_SINUS_BACKGROUNDS = {"Sinus Rhythm", "Sinus Bradycardia", "Sinus Tachycardia"}

def derive_rule_events(
    features: Dict[str, Any],
    signal: Optional[np.ndarray] = None,
    r_peaks: Optional[np.ndarray] = None,
    fs: int = 125,
    background_rhythm: str = "Unknown",
) -> List[Event]:
    """
    Analyzes clinical features to detect arrhythmias strictly via rules.
    - Pause: detected via RR >2000ms
    - AF safety net: RR irregularity + absent P-waves (fallback when ML misses AF)
    - Atrial Flutter: spectral detection of flutter waves in QRS-blanked signal
    Pattern arrhythmias (VT/NSVT/Bigeminy etc.) are handled in apply_ectopy_patterns().
    """
    events = []

    rr_intervals = features.get("rr_intervals_ms", [])
    rr_arr = np.array([])
    if isinstance(rr_intervals, list) and len(rr_intervals) > 2:
        rr_arr = np.array([x for x in rr_intervals if x is not None and isinstance(x, (int, float))])

    # ---------------------------------------------------------
    # Pause
    # ---------------------------------------------------------
    if any(rr > 2000 for rr in rr_arr):
        events.append(Event(
            event_id=str(uuid.uuid4()),
            event_type="Pause",
            event_category=EventCategory.RHYTHM,
            start_time=0.0, end_time=10.0,
            rule_evidence={"rule": "Pause_Detected"},
            priority=85,
            used_for_training=False
        ))

    # ---------------------------------------------------------
    # AF Safety Net: irregularly irregular RR + absent P-waves
    # Fires when ML misses AF (only 5 corrected AF segments in training data).
    # Masterclass criterion: RR std >160ms + p_wave_present_ratio <0.4 = AF.
    # ---------------------------------------------------------
    if len(rr_arr) >= 4:
        rr_std = float(np.std(rr_arr))
        p_ratio = float(features.get("p_wave_present_ratio") or 1.0)
        # Only fire AF safety net if sinus was NOT already declared — avoids a
        # contradictory state where background=Sinus but an AF event also appears.
        _sinus_declared = background_rhythm in _SINUS_BACKGROUNDS
        if rr_std > 160 and p_ratio < 0.4 and not _sinus_declared:
            events.append(Event(
                event_id=str(uuid.uuid4()),
                event_type="Atrial Fibrillation",
                event_category=EventCategory.RHYTHM,
                start_time=0.0, end_time=10.0,
                rule_evidence={"rule": "AF_RR_Irregularity", "rr_std_ms": round(rr_std, 1), "p_wave_ratio": round(p_ratio, 2)},
                priority=88,
                used_for_training=False
            ))

    # ---------------------------------------------------------
    # Atrial Flutter: spectral detection (FFT on QRS-blanked signal)
    # AFL = ventricular rate 130-175 bpm (2:1 block) + flutter waves at 4-6 Hz.
    # ---------------------------------------------------------
    mean_hr = float(features.get("mean_hr_bpm") or 0)
    if 130 <= mean_hr <= 175 and signal is not None and r_peaks is not None and len(r_peaks) > 0:
        r_peaks_arr = np.asarray(r_peaks, dtype=int)
        if _detect_flutter_waves(np.asarray(signal, dtype=np.float32), r_peaks_arr, fs):
            events.append(Event(
                event_id=str(uuid.uuid4()),
                event_type="Atrial Flutter",
                event_category=EventCategory.RHYTHM,
                start_time=0.0, end_time=10.0,
                rule_evidence={"rule": "AFL_Spectral_Detected", "mean_hr_bpm": round(mean_hr, 1)},
                priority=87,
                used_for_training=False
            ))

    return events


# =============================================================================
# 2. ECTOPY PATTERN RECOGNITION
# =============================================================================

def apply_ectopy_patterns(events: List[Event], clinical_features: Optional[Dict[str, Any]] = None) -> None:
    """
    Scans ECTOPY events and applies pattern labels.

    Count-Based Rules (no rate guard):
    PVC: 2=Couplet, 3=Ventricular Run, 4-10=NSVT, 11+=VT
    PAC: 2=Atrial Couplet, 3-5=Atrial Run, 6-10=PSVT, 11+=SVT
    Bigeminy/Trigeminy/Quadrigeminy: interspersed patterns via beat_indices diffs

    Clinical validation (when clinical_features provided):
    - QRS width check: PVC must have QRS >80ms; PAC must have QRS ≤150ms
    - Compensatory pause: full=PVC, incomplete=PAC (overrides ML when ambiguous)
    - VT/NSVT require wide complex (QRS >110ms); narrow-complex → downgrade to SVT/PSVT
    """
    if clinical_features is None:
        clinical_features = {}

    # ── Clinical feature extraction for PVC/PAC validation ──
    qrs_ms      = float(clinical_features.get("qrs_duration_ms") or
                        clinical_features.get("mean_qrs_duration_ms") or 0)
    r_peaks_raw = clinical_features.get("r_peaks", [])
    r_peaks_arr = np.array(r_peaks_raw, dtype=int) if r_peaks_raw else np.array([], dtype=int)
    normal_rr   = float(np.median(np.diff(r_peaks_arr))) if len(r_peaks_arr) > 2 else 0.0

    # V3 beat discriminator scores (from feature vector — pre-computed)
    # These already aggregate p_absent, compensatory_pause, t_discordance etc.
    pvc_score = float(clinical_features.get("pvc_score_mean") or 0)
    pac_score = float(clinical_features.get("pac_score_mean") or 0)

    # ── Per-beat PVC/PAC correction via QRS width + V3 discriminators ────────
    for ev in events:
        if ev.event_category != EventCategory.ECTOPY:
            continue
        label = ev.event_type
        if label not in ("PVC", "PAC"):
            continue

        # Stage 1: Hard QRS-width rule (electrophysiology: PVC >120ms, PAC <120ms)
        if label == "PVC" and 0 < qrs_ms < 80:
            ev.event_type = "PAC"   # Too narrow for ventricular origin
            continue
        if label == "PAC" and qrs_ms > 150:
            ev.event_type = "PVC"   # Too wide for atrial origin
            continue

        # Stage 2: Ambiguous width (80–150 ms) — use V3 discriminator scores
        if 80 <= qrs_ms <= 150:
            if pvc_score > 0 or pac_score > 0:
                # V3 scores available: trust them
                if pvc_score > pac_score + 0.15:
                    ev.event_type = "PVC"
                elif pac_score > pvc_score + 0.15:
                    ev.event_type = "PAC"
                # else: keep ML label (scores too close to override)
            elif normal_rr > 0 and ev.beat_indices:
                # Fallback: compensatory pause tiebreaker
                pause_result = _classify_compensatory_pause(
                    ev.beat_indices[0], r_peaks_arr, normal_rr)
                if pause_result:
                    ev.event_type = pause_result

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
                if count >= 11 and rate >= 100:
                    # VT: 11+ consecutive PVCs at rate >= 100 bpm
                    # Masterclass: any wide-complex tachycardia is VT. Narrow-complex → SVT.
                    run_label = "VT"
                    if 0 < qrs_ms < 110:
                        run_label = "SVT"   # Narrow complex — cannot be VT
                    new_event = Event(
                        event_id=str(uuid.uuid4()),
                        event_type=run_label,
                        event_category=EventCategory.RHYTHM,
                        start_time=cluster[0].start_time,
                        end_time=cluster[-1].end_time,
                        pattern_label="Run",
                        rule_evidence={"rule": f"{run_label}_Detected", "count": count, "rate": round(rate, 1), "qrs_ms": round(qrs_ms, 1)},
                        priority=100,
                        used_for_training=False,
                    )
                    events.append(new_event)
                    for e in cluster: e.pattern_label = "Run"
                elif count >= 4 and rate >= 100:
                    # NSVT: 4-10 consecutive PVCs at rate >= 100 bpm
                    run_label = "NSVT"
                    if 0 < qrs_ms < 110:
                        run_label = "PSVT"  # Narrow complex — cannot be NSVT
                    new_event = Event(
                        event_id=str(uuid.uuid4()),
                        event_type=run_label,
                        event_category=EventCategory.RHYTHM,
                        start_time=cluster[0].start_time,
                        end_time=cluster[-1].end_time,
                        pattern_label="Run",
                        rule_evidence={"rule": f"{run_label}_Detected", "count": count, "rate": round(rate, 1), "qrs_ms": round(qrs_ms, 1)},
                        priority=90,
                        used_for_training=False,
                    )
                    events.append(new_event)
                    for e in cluster: e.pattern_label = "Run"
                elif count >= 3:
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
                if count >= 11 and rate >= 100:
                    # SVT: 11+ consecutive PACs at rate >= 100 bpm
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
                elif count >= 6 and rate >= 100:
                    # PSVT: 6-10 consecutive PACs at rate >= 100 bpm
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

def apply_display_rules(_background_rhythm: str, events: List[Event]) -> List[Event]:
    # Pass 1: Global Hierarchy & Veto
    has_af = any(e.event_type in ["AF", "Atrial Fibrillation", "Atrial Flutter"] for e in events)
    has_svt = any(e.event_type in ["SVT", "PSVT", "Atrial Run"] for e in events)
    has_vt = any(e.event_type in ["VT", "NSVT", "Ventricular Run"] for e in events)

    for event in events:
        should_display = True
        suppression_reason = None

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
        # Atrial ectopy — trains ectopy model (PAC class)
        "PAC", "Atrial Couplet",
        "PAC Bigeminy", "PAC Trigeminy", "PAC Quadrigeminy",

        # Ventricular ectopy — trains ectopy model (PVC class)
        "PVC", "PVC Couplet",
        "PVC Bigeminy", "PVC Trigeminy", "PVC Quadrigeminy",

        # Rhythm labels — train the rhythm model
        "AF", "Atrial Fibrillation", "Atrial Flutter",
        "Ventricular Fibrillation",
        # VT and NSVT annotated by cardiologist → train rhythm model (VT class)
        "VT", "Ventricular Tachycardia", "NSVT",

        # Blocks — train the rhythm model (indices 7-11 in RHYTHM_CLASS_NAMES)
        "1st Degree AV Block", "2nd Degree AV Block Type 1",
        "2nd Degree AV Block Type 2", "3rd Degree AV Block",
        "Intraventricular Conduction Delay",

        # NOTE: SVT, PSVT, Ventricular Run, Atrial Run are RULES-ONLY pattern
        # labels derived from clustering; they are NOT trained directly.
    }
    for event in events:
        # Never train on Sinus or Artifact as primary labels to avoid baseline bias
        if event.event_type in ["Sinus Rhythm", "Sinus Bradycardia", "Sinus Tachycardia", "Artifact"]:
            event.used_for_training = False
        elif event.event_type in training_set:
            event.used_for_training = True
        else:
            event.used_for_training = False
