import psycopg2
import json
from typing import List, Dict, Any, Optional

# ---------------------------------------
# PostgreSQL Connection Settings
# ---------------------------------------
PSQL_CONN_PARAMS = {
    "dbname": "ecg_analysis",
    "user": "ecg_user",
    "password": "sais",         
    "host": "127.0.0.1",
    "port": "5432"
}

def _connect():
    """Create a new PostgreSQL connection."""
    return psycopg2.connect(**PSQL_CONN_PARAMS)

def setup_database():
    """Initializes the database schema using init_db.sql."""
    from pathlib import Path
    sql_path = Path(__file__).parent / "init_db.sql"
    if not sql_path.exists():
        print(f"[WARN] init_db.sql not found at {sql_path}")
        return

    conn = _connect()
    try:
        with conn.cursor() as cur:
            with open(sql_path, "r") as f:
                cur.execute(f.read())
        conn.commit()
        print("[DB] Database schema initialized successfully.")
    except Exception as e:
        print(f"[ERROR] setup_database failed: {e}")
        conn.rollback()
    finally:
        conn.close()

# =====================================================================
# FETCH LIST OF FILES
# =====================================================================
def get_segment_list() -> List[Dict[str, Any]]:
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT filename,
                   COUNT(*) AS segment_count,
                   SUM(CASE WHEN arrhythmia_label IS NULL
                               OR arrhythmia_label='Unlabeled'
                            THEN 1 ELSE 0 END) AS unlabeled_count
            FROM ecg_features_annotatable
            GROUP BY filename
            ORDER BY filename;
        """)

        rows = cur.fetchall()
        return [
            {
                "filename": r[0],
                "segment_count": r[1],
                "unlabeled_count": r[2]
            }
            for r in rows
        ]
    except Exception as e:
        print(f"DB ERROR get_segment_list: {e}")
        return []
    finally:
        if conn:
            conn.close()

# =====================================================================
# FETCH A SINGLE SEGMENT
# =====================================================================
def get_segment_data(segment_id: int):
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    segment_id,
                    filename,
                    segment_index,
                    segment_start_s,
                    segment_duration_s,
                    arrhythmia_label,
                    arrhythmia_text_notes,
                    r_peaks_in_segment,
                    features_json,
                    cardiologist_notes,
                    corrected_by,
                    corrected_at,
                    training_round,
                    raw_signal,
                    pr_interval,
                    segment_fs,
                    dataset_source,
                    is_verified,
                    mistake_target
                FROM ecg_features_annotatable
                WHERE segment_id = %s
                """,
                (segment_id,),
            )
            row = cur.fetchone()
            if not row:
                return None

            cols = [
                "segment_id",
                "filename",
                "segment_index",
                "segment_start_s",
                "segment_duration_s",
                "arrhythmia_label",
                "arrhythmia_text_notes",
                "r_peaks_in_segment",
                "features_json",
                "cardiologist_notes",
                "corrected_by",
                "corrected_at",
                "training_round",
                "raw_signal",
                "pr_interval",
                "segment_fs",
                "dataset_source",
                "is_verified",
                "mistake_target"
            ]

            data = {cols[i]: row[i] for i in range(len(cols))}
            return data

    except Exception as e:
        print("DB ERROR get_segment_data:", e)
        return None
    finally:
        conn.close()
# =====================================================================
# FETCH FROM ecg_features_annotatable (Hybrid Table)
# =====================================================================
def get_segment_new(segment_id: int) -> Optional[Dict[str, Any]]:
    """Fetches the raw signal and UI events from the hybrid table for rendering."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            # We now query ecg_features_annotatable and use raw_signal, features_json, and arrhythmia_label
            cur.execute("""
                SELECT raw_signal, features_json, arrhythmia_label, events_json, segment_fs, filename, segment_index, r_peaks_in_segment, cardiologist_notes, dataset_source
                FROM ecg_features_annotatable
                WHERE segment_id = %s
            """, (segment_id,))
            
            row = cur.fetchone()
            if row:
                def safe_load(data):
                    if isinstance(data, (list, dict)): return data
                    try: return json.loads(data) if data else []
                    except: return []

                return {
                    "signal": safe_load(row[0]),
                    "features": safe_load(row[1]),
                    "background_rhythm": row[2] or "Unlabeled",
                    "events_json": safe_load(row[3]),
                    "segment_fs": row[4] or 125,
                    "filename": row[5],
                    "segment_index": row[6],
                    "r_peaks_in_segment": row[7],
                    "cardiologist_notes": row[8] or "",
                    "dataset_source": row[9] or "Unknown"
                }
    except Exception as e:
        print(f"CRITICAL DB ERROR in get_segment_new: {e}")
    finally:
        conn.close()
    return None

def get_segment(segment_id: int) -> Optional[Dict[str, Any]]:
    return get_segment_new(segment_id)


def save_event_to_db(segment_id: int, event: Dict[str, Any]) -> bool:
    """Appends a cardiologist event to the events_json list for a segment."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            # Fetch existing events_json from ecg_features_annotatable
            cur.execute("SELECT events_json FROM ecg_features_annotatable WHERE segment_id = %s", (segment_id,))
            row = cur.fetchone()
            if not row:
                return False
            
            raw_data = row[0]
            # Handle if it's already a dict (full decision) or a list (events only)
            if isinstance(raw_data, str):
                data = json.loads(raw_data)
            else:
                data = raw_data or []

            if isinstance(data, list):
                # Upgrade legacy list to dict
                data = {
                    "events": data + [event],
                    "final_display_events": data + [event]
                }
            elif isinstance(data, dict):
                if "events" not in data: data["events"] = []
                if "final_display_events" not in data: data["final_display_events"] = []
                
                data["events"].append(event)
                data["final_display_events"].append(event)
            else:
                # Initialize new dict
                data = {
                    "events": [event],
                    "final_display_events": [event]
                }
                
            cur.execute(
                "UPDATE ecg_features_annotatable SET events_json = %s WHERE segment_id = %s",
                (json.dumps(data), segment_id)
            )
            conn.commit()
            return True
    except Exception as e:
        print("DB ERROR save_event_to_db:", e)
        return False
    finally:
        conn.close()

def delete_event(segment_id: int, event_id: str) -> bool:
    """Removes a cardiologist event from the events_json list for a segment."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT events_json FROM ecg_features_annotatable WHERE segment_id = %s", (segment_id,))
            row = cur.fetchone()
            if not row:
                return False
            
            raw_data = row[0]
            if isinstance(raw_data, str):
                data = json.loads(raw_data)
            else:
                data = raw_data or []

            # Helper to filter list
            def remove_by_id(evt_list):
                return [e for e in evt_list if e.get("event_id") != event_id]

            if isinstance(data, list):
                data = remove_by_id(data)
            elif isinstance(data, dict):
                if "events" in data:
                    data["events"] = remove_by_id(data["events"])
                if "final_display_events" in data:
                    data["final_display_events"] = remove_by_id(data["final_display_events"])
            
            cur.execute(
                "UPDATE ecg_features_annotatable SET events_json = %s WHERE segment_id = %s",
                (json.dumps(data), segment_id)
            )
            conn.commit()
            return True
    except Exception as e:
        print("DB ERROR delete_event:", e)
        return False
    finally:
        conn.close()

def update_event_span(segment_id: int, event_id: str, start_time: float, end_time: float) -> bool:
    """Updates start_time and end_time for a specific event in events_json."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT events_json FROM ecg_features_annotatable WHERE segment_id = %s", (segment_id,))
            row = cur.fetchone()
            if not row:
                return False
            raw_data = row[0]
            data = json.loads(raw_data) if isinstance(raw_data, str) else (raw_data or [])

            def patch_list(evt_list):
                for ev in evt_list:
                    if ev.get("event_id") == event_id:
                        ev["start_time"] = start_time
                        ev["end_time"] = end_time
                return evt_list

            if isinstance(data, list):
                data = patch_list(data)
            elif isinstance(data, dict):
                if "events" in data:
                    data["events"] = patch_list(data["events"])
                if "final_display_events" in data:
                    data["final_display_events"] = patch_list(data["final_display_events"])

            cur.execute(
                "UPDATE ecg_features_annotatable SET events_json = %s WHERE segment_id = %s",
                (json.dumps(data), segment_id)
            )
            conn.commit()
            return True
    except Exception as e:
        print("DB ERROR update_event_span:", e)
        return False
    finally:
        conn.close()

def count_confirmed_cardiologist_events() -> int:
    """Counts how many events marked by a cardiologist exist in the ecg_features_annotatable table."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            # Check if events_json is a list of events or a decision dict
            cur.execute("""
                SELECT COUNT(*) 
                FROM ecg_features_annotatable, 
                LATERAL (
                    SELECT CASE 
                        WHEN jsonb_typeof(events_json) = 'array' THEN events_json
                        WHEN jsonb_typeof(events_json) = 'object' AND events_json ? 'events' THEN events_json->'events'
                        ELSE '[]'::jsonb
                    END as event_list
                ) AS l,
                jsonb_array_elements(l.event_list) AS event
                WHERE event->>'annotation_source' = 'cardiologist'
                  AND event->>'annotation_status' = 'confirmed';
            """)
            row = cur.fetchone()
            return row[0] if row else 0
    except Exception as e:
        print("DB ERROR count_confirmed_cardiologist_events:", e)
        return 0
    finally:
        conn.close()

# LEGACY update_annotation removed.

# =====================================================================
# SAVE MODEL PREDICTION (for XAI UI)
# =====================================================================
def save_model_prediction(segment_id: int, pred_label: str, probs_list):
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()

        cur.execute("""
            UPDATE ecg_features_annotatable
            SET model_pred_label = %s,
                model_pred_probs = %s
            WHERE segment_id = %s;
        """, (pred_label, json.dumps(probs_list), segment_id))

        conn.commit()

    except Exception as e:
        print("DB ERROR save_model_prediction:", e)
    finally:
        if conn:
            conn.close()
    return True
# =====================================================================
# FIND FIRST SEGMENT WITH raw_signal
# =====================================================================
def get_min_segment_id_with_signal() -> int:
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()

        cur.execute("""
            SELECT MIN(segment_id)
            FROM ecg_features_annotatable
            WHERE raw_signal IS NOT NULL;
        """)

        row = cur.fetchone()
        return int(row[0]) if row and row[0] else 0

    except Exception as e:
        print("DB ERROR get_min_segment_id_with_signal:", e)
        return 0
    finally:
        if conn:
            conn.close()

# =====================================================================
# GENERIC fetch_one() used by your app.py
# =====================================================================
def fetch_one(sql: str, params=None):
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur.fetchone()
    except Exception as e:
        print("DB fetch_one error:", e)
        return None
    finally:
        if conn:
            conn.close()

# =====================================================================
# Find first segment for a newly uploaded JSON
# =====================================================================
def get_first_segment_id_by_filename(filename_key: str) -> int:
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()

        cur.execute("""
            SELECT segment_id
            FROM ecg_features_annotatable
            WHERE filename = %s
            ORDER BY segment_index ASC
            LIMIT 1;
        """, (filename_key,))

        row = cur.fetchone()
        return row[0] if row else 0

    except Exception as e:
        print("DB ERROR get_first_segment_id_by_filename:", e)
        return 0

    finally:
        if conn:
            conn.close()
def get_all_segments() -> List[Dict[str, Any]]:
    """Fetch all segment IDs and their basic status for the dashboard sidebar.
    Uses ecg_features_annotatable as the single source of truth."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    segment_id,
                    filename,
                    segment_index,
                    is_corrected,
                    COALESCE(arrhythmia_label, 'Unlabeled') as arrhythmia_label
                FROM ecg_features_annotatable
                ORDER BY segment_id ASC
            """)
            rows = cur.fetchall()
            return [
                {
                    "id": r[0],
                    "filename": r[1],
                    "index": r[2] or 0,
                    "status": 'confirmed' if r[3] else 'pending'
                }
                for r in rows
            ]
    except Exception as e:
        print(f"DB ERROR get_all_segments: {e}")
        return []
    finally:
        conn.close()

# ---------------------------------------
# SINGLE SOURCE OF TRUTH ANNOTATIONS
# ---------------------------------------
def update_segment_status(segment_id: int, background_rhythm: str = 'Unlabeled', events: list = None, notes: str = '') -> bool:
    """
    Updates the unified table.
    Saves the rhythm, the specific beat events (PVCs/PACs), and marks it ready for ML Retraining.
    """
    conn = _connect()
    try:
        with conn.cursor() as cur:
            if events is not None:
                # If events are provided (from the UI), we overwrite the current JSON
                events_json_str = json.dumps({"events": events, "final_display_events": events})
                cur.execute("""
                    UPDATE ecg_features_annotatable 
                    SET arrhythmia_label = %s,
                        cardiologist_notes = %s,
                        events_json = %s,
                        is_corrected = TRUE,
                        used_for_training = TRUE,
                        corrected_at = CURRENT_TIMESTAMP
                    WHERE segment_id = %s
                """, (background_rhythm, notes, events_json_str, segment_id))
            else:
                # If events is None, we preserve the existing events_json
                cur.execute("""
                    UPDATE ecg_features_annotatable 
                    SET arrhythmia_label = %s,
                        cardiologist_notes = %s,
                        is_corrected = TRUE,
                        used_for_training = TRUE,
                        corrected_at = CURRENT_TIMESTAMP
                    WHERE segment_id = %s
                """, (background_rhythm, notes, segment_id))
            
        conn.commit()
        return True
    except Exception as e:
        print(f"DB ERROR update_segment_status: {e}")
        return False
    finally:
        conn.close()


def clear_all_annotations(segment_id: int) -> bool:
    """Wipes all annotation events and resets it so the ML model ignores it."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            # We ONLY reset the manual markers and training flags. 
            # We DO NOT overwrite the arrhythmia_label with "Unlabeled"!
            cur.execute("""
                UPDATE ecg_features_annotatable
                SET events_json = '{"events": [], "final_display_events": [], "cleared": true}'::jsonb,
                    cardiologist_notes = '',
                    is_corrected = FALSE,
                    used_for_training = FALSE,
                    corrected_at = NULL,
                    corrected_by = NULL
                WHERE segment_id = %s
            """, (segment_id,))
            
        conn.commit()
        return True
    except Exception as e:
        print(f"DB ERROR clear_all_annotations: {e}")
        return False
    finally:
        conn.close()


def mark_segment_corrected(segment_id: int) -> bool:
    """Set is_corrected=TRUE and used_for_training=TRUE after event edits."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE ecg_features_annotatable
                SET is_corrected = TRUE,
                    used_for_training = TRUE,
                    corrected_at = NOW()
                WHERE segment_id = %s
            """, (segment_id,))
        conn.commit()
        return True
    except Exception as e:
        print(f"DB ERROR mark_segment_corrected: {e}")
        return False
    finally:
        conn.close()
