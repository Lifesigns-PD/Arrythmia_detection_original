import sys
import os
import json
import uuid

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database import db_service

def test_db_interaction():
    print("Testing DB Interaction...")
    
    # 1. Get a segment ID
    conn = db_service._connect()
    with conn.cursor() as cur:
        cur.execute("SELECT segment_id FROM ecg_features_annotatable LIMIT 1")
        row = cur.fetchone()
        if not row:
            print("No segments found in DB. Run import first.")
            return
        segment_id = row[0]
    conn.close()
    print(f"Using Segment ID: {segment_id}")

    # 2. Initial Read
    print("2. Testing Initial Read (get_segment_new)...")
    try:
        data = db_service.get_segment_new(segment_id)
        print("   Success! Data loaded.")
    except Exception as e:
        print(f"   FAILED: {e}")
        return

    # 3. Save Event
    print("3. Testing Save Event...")
    event_id = str(uuid.uuid4())
    event = {
        "event_id": event_id,
        "event_type": "TEST_EVENT",
        "start_time": 0.5,
        "end_time": 0.6,
        "annotation_source": "test_script"
    }
    
    if db_service.save_event_to_db(segment_id, event):
        print("   Success! Event saved.")
    else:
        print("   FAILED to save event.")
        return

    # 4. Read After Save (Critical Check)
    print("4. Testing Read After Save...")
    try:
        data = db_service.get_segment_new(segment_id)
        events = data.get("events_json", {})
        
        # Check structure
        found = False
        if isinstance(events, list):
            print("   Warning: events_json is a list (Expected dict)")
            found = any(e['event_id'] == event_id for e in events)
        elif isinstance(events, dict):
            print("   events_json is a dict (Correct)")
            if "final_display_events" in events:
                found = any(e['event_id'] == event_id for e in events["final_display_events"])
            else:
                print("   Error: final_display_events missing")
        
        if found:
            print("   Success! Event found in DB.")
        else:
            print("   FAILED: Event not found in DB data.")

    except Exception as e:
        print(f"   FAILED: {e}")
        return

    # 5. Delete Event
    print("5. Testing Delete Event...")
    if db_service.delete_event(segment_id, event_id):
        print("   Success! Delete executed.")
    else:
        print("   FAILED to delete.")

    # 6. Verify Deletion
    print("6. Verifying Deletion...")
    data = db_service.get_segment_new(segment_id)
    events = data.get("events_json", {})
    
    found = False
    if isinstance(events, list):
        found = any(e['event_id'] == event_id for e in events)
    elif isinstance(events, dict):
         if "final_display_events" in events:
             found = any(e['event_id'] == event_id for e in events["final_display_events"])
    
    if not found:
        print("   Success! Event correctly removed.")
    else:
        print("   FAILED: Event still exists.")

if __name__ == "__main__":
    test_db_interaction()
