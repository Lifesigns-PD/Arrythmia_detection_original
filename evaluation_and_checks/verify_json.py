import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dashboard')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'database')))

from app import app as flask_app
import db_service
import json

def verify_json():
    print("=== Verification: JSON Serialization Fix ===")
    
    with flask_app.test_client() as client:
        # Test segment 47789 which failed for the user
        response = client.get('/api/xai/47789')
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.get_json()
                print("[OK] JSON is valid and serializable.")
                # Check for enums in final_display_events
                events = data.get("final_display_events", [])
                if events:
                    print(f"First event category: {events[0].get('event_category')}")
                    print(f"First event display_state: {events[0].get('display_state')}")
            except Exception as e:
                print(f"[FAIL] JSON parsing failed: {e}")
        else:
            print(f"[FAIL] API returned {response.status_code}")
            print(response.data.decode()[:500])

if __name__ == "__main__":
    verify_json()
