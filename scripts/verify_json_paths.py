import sys
from pathlib import Path
import json

# Add project roots
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "dashboard"))
sys.path.append(str(BASE_DIR / "database"))

from app import app
import db_service

def verify_api_paths():
    print("--- Verifying API JSON Paths ---")
    
    # Use a test client
    with app.test_client() as client:
        # Get the first segment ID
        row = db_service.fetch_one("SELECT MIN(segment_id) FROM ecg_features_annotatable;")
        if not row or not row[0]:
            print("ERROR: No segments found in database to test.")
            return
        
        segment_id = row[0]
        print(f"Testing Segment ID: {segment_id}")
        
        response = client.get(f"/api/segment/{segment_id}")
        if response.status_code != 200:
            print(f"ERROR: API returned status {response.status_code}")
            return
        
        data = response.get_json()
        
        # Check vitals
        if "vitals" not in data:
            print("FAIL: 'vitals' key missing from response")
        else:
            v = data["vitals"]
            print(f"PASS: 'vitals' found: {v}")
            for key in ["bpm", "pr_interval", "qrs_duration"]:
                if key not in v:
                    print(f"FAIL: 'vitals.{key}' missing")
                else:
                    print(f"PASS: 'vitals.{key}' = {v[key]}")
        
        # Check labels
        if "labels" not in data:
            print("FAIL: 'labels' key missing from response")
        else:
            l = data["labels"]
            print(f"PASS: 'labels' found: {l}")
            for key in ["ai_prediction", "imported_label"]:
                if key not in l:
                    print(f"FAIL: 'labels.{key}' missing")
                else:
                    print(f"PASS: 'labels.{key}' = {l[key]}")

if __name__ == "__main__":
    verify_api_paths()
