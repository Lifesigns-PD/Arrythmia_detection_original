"""
One-time backfill: fixes arrhythmia_label for segments where an ectopy label
(PVC, PAC) was incorrectly stored as the segment-level label.

The rhythm model trains on arrhythmia_label and expects rhythm-level labels
(Sinus Rhythm, AF, VT, etc.) — NOT beat-level ectopy labels.

For segments with ectopy labels:
  - Computes actual background rhythm from heart rate (Sinus Rhythm/Brady/Tachy)
  - Updates arrhythmia_label to the correct rhythm label

Run:  python database/backfill_arrhythmia_labels.py
"""
import psycopg2
import json
import numpy as np

PSQL_CONN_PARAMS = {
    "dbname": "ecg_analysis",
    "user": "ecg_user",
    "password": "sais",
    "host": "127.0.0.1",
    "port": "5432"
}

# These labels should NEVER be in arrhythmia_label — they are ectopy/beat-level
ECTOPY_LABELS = {
    "PVC", "PAC", "None", "Run",
    "PVC Bigeminy", "PVC Trigeminy", "PVC Couplet", "PVC Quadrigeminy",
    "PAC Bigeminy", "PAC Trigeminy", "PAC Couplet", "PAC Quadrigeminy",
    "NSVT", "PSVT", "Ventricular Triplet", "Atrial Triplet",
    "Atrial Couplet", "Ventricular Run", "Atrial Run",
    # SVT/VT are now rules-only — must never be arrhythmia_label
    "SVT", "Supraventricular Tachycardia",
    "VT", "Ventricular Tachycardia",
}

DEFAULT_FS = 125


def compute_background_rhythm(features_json_raw, seg_fs):
    """Derive background rhythm from R-peak intervals → heart rate."""
    fs = int(seg_fs) if seg_fs else DEFAULT_FS

    if not features_json_raw:
        return "Sinus Rhythm"

    features = features_json_raw if isinstance(features_json_raw, dict) else json.loads(features_json_raw)
    r_peaks = features.get("r_peaks", [])

    if len(r_peaks) < 2:
        return "Sinus Rhythm"

    # Compute mean HR from R-R intervals
    r_peaks_sorted = sorted(r_peaks)
    rr_intervals = np.diff(r_peaks_sorted) / fs  # in seconds
    rr_intervals = rr_intervals[rr_intervals > 0.3]  # filter artifacts
    rr_intervals = rr_intervals[rr_intervals < 2.0]

    if len(rr_intervals) == 0:
        return "Sinus Rhythm"

    mean_rr = np.mean(rr_intervals)
    hr = 60.0 / mean_rr

    if hr < 60:
        return "Sinus Bradycardia"
    elif hr > 100:
        return "Sinus Tachycardia"
    else:
        return "Sinus Rhythm"


def backfill():
    conn = psycopg2.connect(**PSQL_CONN_PARAMS)
    cur = conn.cursor()

    # Find all segments with ectopy labels
    placeholders = ','.join(['%s'] * len(ECTOPY_LABELS))
    cur.execute(f"""
        SELECT segment_id, arrhythmia_label, features_json, segment_fs
        FROM ecg_features_annotatable
        WHERE arrhythmia_label IN ({placeholders})
    """, tuple(ECTOPY_LABELS))

    rows = cur.fetchall()
    print(f"Found {len(rows)} segments with ectopy labels in arrhythmia_label")

    updated = 0
    label_changes = {}

    for segment_id, old_label, features_json_raw, seg_fs in rows:
        new_label = compute_background_rhythm(features_json_raw, seg_fs)

        change_key = f"{old_label} -> {new_label}"
        label_changes[change_key] = label_changes.get(change_key, 0) + 1

        cur.execute(
            "UPDATE ecg_features_annotatable SET arrhythmia_label = %s WHERE segment_id = %s",
            (new_label, segment_id)
        )
        updated += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"\nDone! Updated {updated} segments")
    print("\nLabel changes:")
    for change, count in sorted(label_changes.items(), key=lambda x: -x[1]):
        print(f"  {change}: {count}")


if __name__ == "__main__":
    backfill()
