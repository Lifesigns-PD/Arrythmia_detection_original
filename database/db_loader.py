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
    Includes robust JSON parsing for features_json and events_json.
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
            
            for row in rows:
                seg_id, filename, seg_idx, f_raw, arr_label, e_raw, seg_fs = row
                
                # Robust JSON loads
                def safe_load_dict(data):
                    if isinstance(data, dict): return data
                    try: return json.loads(data) if data else {}
                    except: return {}
                
                def safe_load_list(data):
                    if isinstance(data, list): return data
                    try: return json.loads(data) if data else []
                    except: return []

                features = safe_load_dict(f_raw)
                events = safe_load_list(e_raw)
                rp = features.get("r_peaks", [])

                out.append({
                    "segment_id": seg_id,
                    "filename": filename,
                    "segment_index": int(seg_idx) if seg_idx is not None else 0,
                    "features_json": features,
                    "arrhythmia_label": arr_label,
                    "r_peaks_in_segment": rp,
                    "events_json": events,
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
