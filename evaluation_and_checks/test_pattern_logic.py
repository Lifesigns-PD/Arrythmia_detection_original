import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from decision_engine.models import Event, EventCategory
from decision_engine.rules import apply_ectopy_patterns

def test_patterns():
    print("Testing Bigeminy/Trigeminy Logic...")
    
    # helper to create mock events
    def create_events(target_type, indices, start_times):
        events = []
        for idx, t in zip(indices, start_times):
            events.append(Event(
                event_id=f"e_{idx}",
                event_type=target_type,
                event_category=EventCategory.ECTOPY,
                start_time=t,
                end_time=t + 0.1,
                beat_indices=[idx]
            ))
        return events

    # 1. Test PVC Bigeminy (Indices: 10, 12, 14 - one normal beat between)
    print("\nCase 1: PVC Bigeminy")
    pvc_bigeminy = create_events("PVC", [10, 12, 14], [1.0, 2.0, 3.0])
    apply_ectopy_patterns(pvc_bigeminy)
    patterns = [e.pattern_label for e in pvc_bigeminy if e.event_category == EventCategory.ECTOPY]
    pattern_events = [e for e in pvc_bigeminy if e.event_category == EventCategory.RHYTHM]
    print(f"Individual Pattern Labels: {patterns}")
    if pattern_events:
        print(f"Detected Pattern Event: {pattern_events[0].event_type} (Priority: {pattern_events[0].priority})")
    
    # 2. Test PAC Trigeminy (Indices: 5, 8, 11 - two normal beats between)
    print("\nCase 2: PAC Trigeminy")
    pac_trigeminy = create_events("PAC", [5, 8, 11], [0.5, 2.0, 3.5])
    apply_ectopy_patterns(pac_trigeminy)
    patterns = [e.pattern_label for e in pac_trigeminy if e.event_category == EventCategory.ECTOPY]
    pattern_events = [e for e in pac_trigeminy if e.event_category == EventCategory.RHYTHM]
    print(f"Individual Pattern Labels: {patterns}")
    if pattern_events:
        print(f"Detected Pattern Event: {pattern_events[0].event_type} (Priority: {pattern_events[0].priority})")

    # 3. Test Ventricular Run (indices: 20, 21, 22 - consecutive)
    print("\nCase 3: Ventricular Run (Consecutive)")
    v_run = create_events("PVC", [20, 21, 22], [4.0, 4.5, 5.0]) # Gap 0.5s = 120 BPM
    apply_ectopy_patterns(v_run)
    patterns = [e.pattern_label for e in v_run if e.event_category == EventCategory.ECTOPY]
    pattern_events = [e for e in v_run if e.event_category == EventCategory.RHYTHM]
    print(f"Individual Pattern Labels: {patterns}")
    if pattern_events:
        print(f"Detected Pattern Event: {pattern_events[0].event_type} (Priority: {pattern_events[0].priority}, Rate: {pattern_events[0].rule_evidence.get('rate')} BPM)")

    # 4. Test PVC Couplet (consecutive 2)
    print("\nCase 4: PVC Couplet")
    couplet = create_events("PVC", [30, 31], [6.0, 6.5])
    apply_ectopy_patterns(couplet)
    patterns = [e.pattern_label for e in couplet if e.event_category == EventCategory.ECTOPY]
    print(f"Individual Pattern Labels: {patterns}")

if __name__ == "__main__":
    test_patterns()
