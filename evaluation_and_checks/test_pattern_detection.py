"""
Test: Count-Based Pattern Detection Rules
==========================================
Verifies:
- PVC: 2=Couplet, 3=Ventricular Run, 4-10=NSVT, 11+=VT
- PAC: 2=Atrial Couplet, 3-5=Atrial Run, 6-10=PSVT, 11+=SVT
- Bigeminy/Trigeminy = alternating (needs beat_indices)
- No rate guard — count alone determines the label
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
    found = any(expected_type.lower() in (getattr(e, 'event_type', '')).lower() for e in events)

    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    all_types = [e.event_type for e in events]
    print(f"  All event types: {all_types}")

    status = "PASS" if found == should_exist else "FAIL"
    print(f"  Result: [{status}] {expected_type} {'found' if found else 'NOT found'}")
    results.append((name, status))


# ── PVC: 2 = Couplet ──
events = [make_event("PVC", 1.0), make_event("PVC", 1.2)]
run_test("2 PVCs -> PVC Couplet", events, "PVC Couplet")

# ── PVC: 3 = Ventricular Run ──
events = [make_event("PVC", 1.0), make_event("PVC", 1.2), make_event("PVC", 1.4)]
run_test("3 PVCs -> Ventricular Run", events, "Ventricular Run")

# ── PVC: 3 should NOT be NSVT ──
events = [make_event("PVC", 1.0), make_event("PVC", 1.2), make_event("PVC", 1.4)]
run_test("3 PVCs -> NOT NSVT", events, "NSVT", should_exist=False)

# ── PVC: 4 = NSVT (fast — with rate guard removed, count alone matters) ──
events = [make_event("PVC", 1.0), make_event("PVC", 1.2), make_event("PVC", 1.4), make_event("PVC", 1.6)]
run_test("4 PVCs fast -> NSVT", events, "NSVT")

# ── PVC: 4 slow = NSVT (no rate guard!) ──
events = [make_event("PVC", 1.0), make_event("PVC", 2.0), make_event("PVC", 3.0), make_event("PVC", 4.0)]
run_test("4 PVCs slow -> NSVT (no rate guard)", events, "NSVT")

# ── PVC: 10 = NSVT (upper boundary) ──
events = [make_event("PVC", i * 0.3) for i in range(10)]
run_test("10 PVCs -> NSVT", events, "NSVT")

# ── PVC: 11 = VT ──
events = [make_event("PVC", i * 0.3) for i in range(11)]
run_test("11 PVCs -> VT", events, "VT")

# ── PVC: 11 should NOT be NSVT ──
events = [make_event("PVC", i * 0.3) for i in range(11)]
run_test("11 PVCs -> NOT NSVT", events, "NSVT", should_exist=False)

# ── PAC: 2 = Atrial Couplet ──
events = [make_event("PAC", 1.0), make_event("PAC", 1.3)]
run_test("2 PACs -> Atrial Couplet", events, "Atrial Couplet")

# ── PAC: 3 = Atrial Run ──
events = [make_event("PAC", 1.0), make_event("PAC", 1.3), make_event("PAC", 1.6)]
run_test("3 PACs -> Atrial Run", events, "Atrial Run")

# ── PAC: 5 = Atrial Run (upper boundary) ──
events = [make_event("PAC", i * 0.4) for i in range(5)]
run_test("5 PACs -> Atrial Run", events, "Atrial Run")

# ── PAC: 6 = PSVT ──
events = [make_event("PAC", i * 0.4) for i in range(6)]
run_test("6 PACs -> PSVT", events, "PSVT")

# ── PAC: 10 = PSVT (upper boundary) ──
events = [make_event("PAC", i * 0.4) for i in range(10)]
run_test("10 PACs -> PSVT", events, "PSVT")

# ── PAC: 11 = SVT ──
events = [make_event("PAC", i * 0.4) for i in range(11)]
run_test("11 PACs -> SVT", events, "SVT")

# ── Bigeminy (beat indices 5,7,9 = diff of 2) ──
events = [make_event("PVC", 1.0, 5), make_event("PVC", 2.0, 7), make_event("PVC", 3.0, 9)]
run_test("PVC Bigeminy (indices 5,7,9)", events, "PVC Bigeminy")

# ── Trigeminy (beat indices 3,6,9 = diff of 3) ──
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
