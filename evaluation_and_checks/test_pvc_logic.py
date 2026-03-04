
import sys
from pathlib import Path

# Fix paths for imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from decision_engine.models import Event, EventCategory
from decision_engine.rules import apply_ectopy_patterns

def test_pvc_logic():
    print("--- Testing PVC Pattern Logic ---")
    
    # 1. True Couplet (Gap 0.5s)
    events1 = [
        Event(event_id="1", event_type="PVC", event_category=EventCategory.ECTOPY, start_time=1.0, end_time=1.1),
        Event(event_id="2", event_type="PVC", event_category=EventCategory.ECTOPY, start_time=1.5, end_time=1.6)
    ]
    apply_ectopy_patterns(events1)
    assert events1[0].pattern_label == "Couplet", f"Expected Couplet, got {events1[0].pattern_label}"
    print("[PASS] True Couplet detected.")

    # 2. Teleporting PVC (Gap 5.0s)
    events2 = [
        Event(event_id="3", event_type="PVC", event_category=EventCategory.ECTOPY, start_time=1.0, end_time=1.1),
        Event(event_id="4", event_type="PVC", event_category=EventCategory.ECTOPY, start_time=6.0, end_time=6.1)
    ]
    apply_ectopy_patterns(events2)
    assert events2[0].pattern_label is None, f"Expected None, got {events2[0].pattern_label}"
    print("[PASS] Teleporting PVC correctly separated.")

    # 3. True NSVT (3 PVCs, 150 BPM)
    # Gap 0.4s -> 1.2s duration for 3 beats = high rate
    events3 = [
        Event(event_id="a", event_type="PVC", event_category=EventCategory.ECTOPY, start_time=1.0, end_time=1.1),
        Event(event_id="b", event_type="PVC", event_category=EventCategory.ECTOPY, start_time=1.4, end_time=1.5),
        Event(event_id="c", event_type="PVC", event_category=EventCategory.ECTOPY, start_time=1.8, end_time=1.9)
    ]
    apply_ectopy_patterns(events3)
    nsvt_events = [e for e in events3 if e.pattern_label == "NSVT"]
    assert len(nsvt_events) == 1, "Expected NSVT event to be created"
    print("[PASS] True NSVT detected.")

    # 4. Mixed Chain (Couplet then Far PVC)
    events4 = [
        Event(event_id="x", event_type="PVC", event_category=EventCategory.ECTOPY, start_time=1.0, end_time=1.1),
        Event(event_id="y", event_type="PVC", event_category=EventCategory.ECTOPY, start_time=1.5, end_time=1.6),
        Event(event_id="z", event_type="PAC", event_category=EventCategory.ECTOPY, start_time=2.0, end_time=2.1), # Breaks chain
        Event(event_id="w", event_type="PVC", event_category=EventCategory.ECTOPY, start_time=5.0, end_time=5.1)
    ]
    apply_ectopy_patterns(events4)
    assert events4[0].pattern_label == "Couplet", "First two should be couplet"
    assert events4[3].pattern_label is None, "Last PVC should be isolated"
    print("[PASS] Mixed chain handled correctly.")

    print("\nALL PVC LOGIC TESTS PASSED ✅")

if __name__ == "__main__":
    test_pvc_logic()
