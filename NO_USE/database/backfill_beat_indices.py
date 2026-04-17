"""
One-time backfill script: reconstructs beat_indices for events
that were saved with empty beat_indices due to the bug in
update_segment_events().

Logic:
  - For each segment with events, load R-peaks from features_json.r_peaks
  - For each event with empty beat_indices, compute midpoint = (start_time + end_time) / 2
  - Convert midpoint to sample position: peak_sample = midpoint * fs
  - Find the closest R-peak index (ordinal position in the R-peak list)
  - Update beat_indices with the matched index

Run:  python database/backfill_beat_indices.py
"""
import psycopg2
import json

PSQL_CONN_PARAMS = {
    "dbname": "ecg_analysis",
    "user": "ecg_user",
    "password": "sais",
    "host": "127.0.0.1",
    "port": "5432"
}

DEFAULT_FS = 125


def backfill():
    conn = psycopg2.connect(**PSQL_CONN_PARAMS)
    cur = conn.cursor()

    # R-peaks are stored in features_json.r_peaks, NOT r_peaks_in_segment
    cur.execute("""
        SELECT segment_id, events_json, features_json, segment_fs
        FROM ecg_features_annotatable
        WHERE events_json IS NOT NULL
          AND features_json IS NOT NULL
    """)

    rows = cur.fetchall()
    print(f"Found {len(rows)} segments with events + features")

    updated = 0
    skipped = 0
    total_events_fixed = 0

    for segment_id, events_json_raw, features_json_raw, seg_fs in rows:
        fs = int(seg_fs) if seg_fs else DEFAULT_FS

        # Parse features_json to get R-peaks
        if isinstance(features_json_raw, str):
            features = json.loads(features_json_raw)
        else:
            features = features_json_raw or {}

        r_peaks = features.get("r_peaks", [])
        if not r_peaks:
            skipped += 1
            continue
        r_peaks = sorted([int(x) for x in r_peaks])

        # Parse events_json
        if isinstance(events_json_raw, str):
            data = json.loads(events_json_raw)
        else:
            data = events_json_raw

        if not isinstance(data, dict):
            # Legacy list format — wrap it
            data = {"events": data if isinstance(data, list) else [],
                    "final_display_events": data if isinstance(data, list) else []}

        changed = False

        # Process both event lists
        for key in ("events", "final_display_events"):
            events = data.get(key, [])
            if not isinstance(events, list):
                continue

            for evt in events:
                if not isinstance(evt, dict):
                    continue

                # Skip events that already have beat_indices
                existing = evt.get("beat_indices", [])
                if existing and len(existing) > 0:
                    continue

                # Skip segment-level events (full 10s span, pattern labels)
                start_t = float(evt.get("start_time", 0.0))
                end_t = float(evt.get("end_time", 10.0))

                # Only fix beat-level events (short window, ~0.6s)
                duration = end_t - start_t
                if duration > 2.0:
                    continue

                event_type = evt.get("event_type", "")

                # Compute midpoint → sample position
                midpoint = (start_t + end_t) / 2.0
                peak_sample = midpoint * fs

                # Find closest R-peak
                best_idx = None
                best_dist = float('inf')
                for i, rp in enumerate(r_peaks):
                    dist = abs(rp - peak_sample)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i

                # Only match if within 50 samples (~400ms at 125Hz)
                if best_idx is not None and best_dist < 50:
                    evt["beat_indices"] = [best_idx]
                    changed = True
                    total_events_fixed += 1
                else:
                    print(f"  [WARN] Segment {segment_id}: event '{event_type}' at {midpoint:.2f}s "
                          f"(sample {peak_sample:.0f}) — no R-peak match within 50 samples")

        if changed:
            cur.execute(
                "UPDATE ecg_features_annotatable SET events_json = %s WHERE segment_id = %s",
                (json.dumps(data), segment_id)
            )
            updated += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"\nDone!")
    print(f"  Segments updated: {updated}")
    print(f"  Segments skipped (no R-peaks in features): {skipped}")
    print(f"  Total events fixed: {total_events_fixed}")


if __name__ == "__main__":
    backfill()
