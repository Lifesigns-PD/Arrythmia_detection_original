import requests
import json
import uuid

# Configuration
BASE_URL = "http://127.0.0.1:5000"
TEST_SEGMENT_ID = 1  # Assuming segment 1 exists

def test_advanced_annotation():
    print("Testing /api/annotate_beats endpoint...")
    
    payload = {
        "segment_id": TEST_SEGMENT_ID,
        "beat_indices": [500, 1000], 
        "label": "AFib"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/annotate_beats", json=payload)
        if response.status_code == 200:
            print(f"Success! Response: {response.json()}")
        else:
            print(f"Failed. Status: {response.status_code}, Msg: {response.text}")
    except Exception as e:
        print(f"Error connecting to dashboard: {e}")

if __name__ == "__main__":
    test_advanced_annotation()
