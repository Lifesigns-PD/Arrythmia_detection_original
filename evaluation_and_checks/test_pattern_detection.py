"""
Test: Updated Pattern Detection Thresholds
===========================================
Verifies:
- 2 consecutive PVCs/PACs = Couplet
- 3 consecutive PVCs/PACs = Run (Ventricular/Atrial Run)
- 4+ consecutive PVCs fast = NSVT/PSVT
- 4+ consecutive PVCs slow = Ventricular/Atrial Run
- Bigeminy/Trigeminy = alternating (needs beat_indices)
"""
import sys, os, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_engine.models import Event, EventCategory
from decision_engine.rules import apply_ectopy_patterns

def make_event(etype, t, beat_idx=None):
    return Event(
        event_id=str(uuid.uuid4()),
        event_type=etype,
        event_category=EventCategory.ECTOPY,
        start_time=t, end_time=t + 0.1,
        beat_indices=[beat_idx] if beat_idx is not None else [],
        used_for_training=True
    )

results = []

def run_test(name, events, expected_type, should_exist=True):
    apply_ectopy_patterns(events)
    new_events = [e for e in events if e.event_category == EventCategory.RHYTHM or
                  (e.event_category == EventCategory.ECTOPY and e.pattern_label)]
    
    # Check for the expected event type
    found = any(expected_type.lower() in (getattr(e, 'event_type', '')).lower() for e in events)
    
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    print(f"  Input: {len([e for e in events if e.event_category == EventCategory.ECTOPY])} ectopy events")
    print(f"  Expected: {expected_type} = {'YES' if should_exist else 'NO'}")
    
    all_types = [e.event_type for e in events]
    print(f"  All event types: {all_types}")
    
    status = "PASS" if found == should_exist else "FAIL"
    print(f"  Result: [{status}] {expected_type} {'found' if found else 'NOT found'}")
    results.append((name, status))


# ── Test 1: 2 PVCs = Couplet ──
events = [make_event("PVC", 1.0), make_event("PVC", 1.2)]
run_test("2 PVCs -> PVC Couplet", events, "PVC Couplet")

# ── Test 2: 2 PACs = Atrial Couplet ──
events = [make_event("PAC", 1.0), make_event("PAC", 1.3)]
run_test("2 PACs -> Atrial Couplet", events, "Atrial Couplet")

# ── Test 3: 3 PVCs = Ventricular Run (NOT NSVT) ──
events = [make_event("PVC", 1.0), make_event("PVC", 1.2), make_event("PVC", 1.4)]
run_test("3 PVCs -> Ventricular Run", events, "Ventricular Run")

# ── Test 4: 3 PVCs should NOT be NSVT ──
events = [make_event("PVC", 1.0), make_event("PVC", 1.2), make_event("PVC", 1.4)]
run_test("3 PVCs -> NOT NSVT", events, "NSVT", should_exist=False)

# ── Test 5: 3 PACs = Atrial Run ──
events = [make_event("PAC", 1.0), make_event("PAC", 1.3), make_event("PAC", 1.6)]
run_test("3 PACs -> Atrial Run", events, "Atrial Run")

# ── Test 6: 3 PACs should NOT be PSVT ──
events = [make_event("PAC", 1.0), make_event("PAC", 1.3), make_event("PAC", 1.6)]
run_test("3 PACs -> NOT PSVT", events, "PSVT", should_exist=False)

# ── Test 7: 4 PVCs fast = NSVT ──
events = [make_event("PVC", 1.0), make_event("PVC", 1.2), make_event("PVC", 1.4), make_event("PVC", 1.6)]
run_test("4 PVCs fast -> NSVT", events, "NSVT")

# ── Test 8: 5 PVCs fast = NSVT ──
events = [make_event("PVC", 1.0), make_event("PVC", 1.15), make_event("PVC", 1.3),
          make_event("PVC", 1.45), make_event("PVC", 1.6)]
run_test("5 PVCs fast -> NSVT", events, "NSVT")

# ── Test 9: 4 PACs fast = PSVT ──
events = [make_event("PAC", 2.0), make_event("PAC", 2.3), make_event("PAC", 2.6), make_event("PAC", 2.9)]
run_test("4 PACs fast -> PSVT", events, "PSVT")

# ── Test 10: 4 PVCs slow = Ventricular Run (not NSVT) ──
events = [make_event("PVC", 1.0), make_event("PVC", 2.0), make_event("PVC", 3.0), make_event("PVC", 4.0)]
run_test("4 PVCs slow -> Ventricular Run", events, "Ventricular Run")

# ── Test 11: Bigeminy (beat indices 5,7,9 = diff of 2) ──
events = [make_event("PVC", 1.0, 5), make_event("PVC", 2.0, 7), make_event("PVC", 3.0, 9)]
run_test("PVC Bigeminy (indices 5,7,9)", events, "PVC Bigeminy")

# ── Test 12: Trigeminy (beat indices 3,6,9 = diff of 3) ──
events = [make_event("PVC", 1.0, 3), make_event("PVC", 2.0, 6), make_event("PVC", 3.0, 9)]
run_test("PVC Trigeminy (indices 3,6,9)", events, "PVC Trigeminy")

# ── Summary ──
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
passed = sum(1 for _, s in results if s == "PASS")
for name, status in results:
    print(f"  [{status}] {name}")
print(f"\n  {passed}/{len(results)} tests passed")
