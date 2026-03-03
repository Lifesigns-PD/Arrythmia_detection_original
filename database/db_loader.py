# src/db_loader.py
import json
from typing import List, Dict, Any, Optional
from db_service import PSQL_CONN_PARAMS
import psycopg2

def _connect():
    return psycopg2.connect(**PSQL_CONN_PARAMS)

def fetch_annotated_segments(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Return list of dict rows from ecg_features_annotatable for the UI.
    """
    try:
        conn = _connect()
    except Exception as ex:
        print(f"[db_loader] WARNING: DB connect failed: {ex}")
        return []

    out = []
    try:
        with conn.cursor() as cur:
            # segment_fs is critical for coordinate sync: sample_idx = time_s * segment_fs
            q = """
                SELECT segment_id, filename, segment_index, features_json, arrhythmia_label, events_json, segment_fs
                FROM ecg_features_annotatable
            """
            if limit:
                q += f" LIMIT {int(limit)}"
                
            cur.execute(q)
            rows = cur.fetchall()
            
            for seg_id, filename, seg_idx, features_json, arr_label, events_json, seg_fs in rows:
                # 1. Normalize features_json
                if isinstance(features_json, str):
                    try:
                        features = json.loads(features_json)
                    except:
                        features = {}
                else:
                    features = features_json or {}
                
                # 2. Extract r_peaks directly from the JSON dictionary!
                rp = features.get("r_peaks", [])
                
                # 3. Normalize events_json (for the red UI markers)
                if isinstance(events_json, str):
                    try:
                        events = json.loads(events_json)
                    except:
                        events = []
                else:
                    events = events_json or []

                out.append({
                    "segment_id": seg_id,
                    "filename": filename,
                    "segment_index": int(seg_idx) if seg_idx is not None else 0,
                    "features_json": features,
                    "arrhythmia_label": arr_label,
                    "r_peaks_in_segment": rp,
                    "events_json": events,
                    # segment_fs is essential for coordinate sync:
                    # sample_idx = event_time_seconds * segment_fs
                    # Do NOT hard-code 250 Hz; MIT-BIH data is at 125 Hz
                    "segment_fs": int(seg_fs) if seg_fs is not None else 125,
                })
    except Exception as ex:
        print(f"[db_loader] WARNING: query failed: {ex}")
    finally:
        try:
            conn.close()
        except:
            pass
            
    return out 
