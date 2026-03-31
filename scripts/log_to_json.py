"""
log_to_json.py — Convert training logs to JSON with Unix timestamps.

Usage:
    # Full log
    python scripts/log_to_json.py --file models_training/outputs/logs/ectopy/initial_20260311_151701.log

    # Filter by time range (ISO or Unix)
    python scripts/log_to_json.py --file path/to/log --start "2026-03-11 15:20:00" --end "2026-03-11 16:00:00"
    python scripts/log_to_json.py --file path/to/log --start 1773430800 --end 1773434400

    # Filter by epoch range
    python scripts/log_to_json.py --file path/to/log --epoch-start 5 --epoch-end 15

    # Output to file instead of stdout
    python scripts/log_to_json.py --file path/to/log --output result.json

Output JSON:
{
    "source_file": "initial_20260311_151701.log",
    "session_start": "2026-03-11 15:17:01",
    "session_start_unix": 1773430621,
    "task": "ectopy",
    "mode": "initial",
    "total_epochs": 30,
    "filter": { "start_unix": ..., "end_unix": ... },
    "header": { ... dataset info ... },
    "epochs": [
        {
            "epoch": 1,
            "total_epochs": 30,
            "timestamp_unix": 1773430621,
            "timestamp_iso": "2026-03-11T15:17:01",
            "train_loss": 0.0315,
            "train_acc": 0.894,
            "val_loss": 0.0407,
            "bal_acc": 0.433,
            "saved_checkpoint": true
        },
        ...
    ],
    "best": { "epoch": 7, "bal_acc": 0.4632, "checkpoint": "..." },
    "class_distribution": { "00 None": 0, "01 PVC": 6992, ... }
}
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def _parse_timestamp(ts_str: str) -> float:
    """Parse 'YYYY-MM-DD HH:MM:SS' to Unix timestamp."""
    dt = datetime.strptime(ts_str.strip(), "%Y-%m-%d %H:%M:%S")
    return dt.timestamp()


def _to_unix(value: str) -> float:
    """Accept either ISO string or raw unix int/float."""
    try:
        return float(value)
    except ValueError:
        return _parse_timestamp(value)


def parse_log(log_path: Path) -> dict:
    """Parse a training log file into structured data."""
    lines = log_path.read_text(encoding="utf-8").splitlines()

    result = {
        "source_file": log_path.name,
        "session_start": None,
        "session_start_unix": None,
        "task": None,
        "mode": None,
        "total_epochs": None,
        "header": {},
        "epochs": [],
        "best": None,
        "class_distribution": {},
    }

    session_ts = None
    epoch_duration_est = None  # estimated seconds per epoch
    in_class_dist = False

    for line in lines:
        line = line.rstrip()

        # Session started: 2026-03-11 15:17:01
        m = re.match(r"Session started:\s*(.+)", line)
        if m:
            result["session_start"] = m.group(1).strip()
            session_ts = _parse_timestamp(m.group(1))
            result["session_start_unix"] = int(session_ts)
            continue

        # task=ectopy  mode=initial  epochs=30  batch=32  lr=0.0005
        m = re.match(r"task=(\w+)\s+mode=(\w+)\s+epochs=(\d+)", line)
        if m:
            result["task"] = m.group(1)
            result["mode"] = m.group(2)
            result["total_epochs"] = int(m.group(3))
            continue

        # [Dataset] ... lines
        m = re.match(r"\[Dataset\]\s+(.+)", line)
        if m:
            info = m.group(1)
            if "Fetched" in info:
                mm = re.search(r"(\d+)\s+segments", info)
                if mm:
                    result["header"]["total_segments"] = int(mm.group(1))
            elif "windows extracted" in info:
                mm = re.search(r"(\d+)\s+windows", info)
                if mm:
                    result["header"]["total_windows"] = int(mm.group(1))
            elif "task=" in info:
                result["header"]["dataset_info"] = info
            continue

        # [Split] 863 train recordings -> 9168 windows  |  153 val recordings -> 1790 windows
        m = re.match(r"\[Split\]\s+(\d+)\s+train.+?(\d+)\s+windows\s+\|\s+(\d+)\s+val.+?(\d+)\s+windows", line)
        if m:
            result["header"]["train_recordings"] = int(m.group(1))
            result["header"]["train_windows"] = int(m.group(2))
            result["header"]["val_recordings"] = int(m.group(3))
            result["header"]["val_windows"] = int(m.group(4))
            continue

        # Device=cpu  Batch=32  Train_windows=9168  Val_windows=1790
        m = re.match(r"Device=(\w+)\s+Batch=(\d+)", line)
        if m:
            result["header"]["device"] = m.group(1)
            result["header"]["batch_size"] = int(m.group(2))
            continue

        # Class distribution lines
        if "Class distribution" in line:
            in_class_dist = True
            continue
        if in_class_dist:
            m = re.match(r"\s+(\d+)\s+(\S+(?:\s+\S+)*?)\s+(\d+)\s*$", line)
            if m:
                cls_idx = m.group(1)
                cls_name = m.group(2).strip()
                cls_count = int(m.group(3))
                result["class_distribution"][f"{cls_idx} {cls_name}"] = cls_count
                continue
            elif line.strip() == "" or line.startswith("Device") or line.startswith("Ep"):
                in_class_dist = False
                # Fall through to check epoch below

        # sources: {'cardiologist': 38, 'imported': 10920}
        m = re.search(r"sources:\s*(\{.+\})", line)
        if m:
            try:
                result["header"]["sources"] = eval(m.group(1))
            except Exception:
                result["header"]["sources_raw"] = m.group(1)
            continue

        # skipped: null=0 short=0 no_label=0
        m = re.match(r"\s+skipped:\s+null=(\d+)\s+short=(\d+)\s+no_label=(\d+)", line)
        if m:
            result["header"]["skipped_null"] = int(m.group(1))
            result["header"]["skipped_short"] = int(m.group(2))
            result["header"]["skipped_no_label"] = int(m.group(3))
            continue

        # Ep 01/30  train loss=0.0315 acc=0.894  val loss=0.0407 bal_acc=0.433
        m = re.match(
            r"Ep\s+(\d+)/(\d+)\s+train\s+loss=([\d.]+)\s+acc=([\d.]+)\s+val\s+loss=([\d.]+)\s+bal_acc=([\d.]+)",
            line,
        )
        if m:
            ep_num = int(m.group(1))
            total = int(m.group(2))

            # Estimate timestamp: session_start + epoch_num * avg_epoch_time
            # We don't have real per-epoch timestamps, so we estimate linearly
            # across the file modification time
            ep_entry = {
                "epoch": ep_num,
                "total_epochs": total,
                "train_loss": float(m.group(3)),
                "train_acc": float(m.group(4)),
                "val_loss": float(m.group(5)),
                "bal_acc": float(m.group(6)),
                "saved_checkpoint": False,
            }
            result["epochs"].append(ep_entry)
            continue

        # - Saved  bal_acc=0.4328
        m = re.match(r"\s+-\s+Saved\s+bal_acc=([\d.]+)", line)
        if m and result["epochs"]:
            result["epochs"][-1]["saved_checkpoint"] = True
            result["epochs"][-1]["saved_bal_acc"] = float(m.group(1))
            continue

        # [DONE] Best balanced acc: 0.4632  |  Checkpoint: ...
        m = re.match(r"\[DONE\]\s+Best balanced acc:\s*([\d.]+)\s*\|\s*Checkpoint:\s*(.+)", line)
        if m:
            result["best"] = {
                "bal_acc": float(m.group(1)),
                "checkpoint": m.group(2).strip(),
            }
            continue

    # Estimate per-epoch timestamps using file mtime
    if session_ts and result["epochs"]:
        try:
            file_mtime = log_path.stat().st_mtime
            total_duration = file_mtime - session_ts
            n_epochs = len(result["epochs"])
            epoch_dur = total_duration / n_epochs if n_epochs > 0 else 0

            for ep in result["epochs"]:
                ep_ts = session_ts + (ep["epoch"] - 1) * epoch_dur
                ep["timestamp_unix"] = int(ep_ts)
                ep["timestamp_iso"] = datetime.fromtimestamp(ep_ts).strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            pass

    # Fill best epoch number
    if result["best"]:
        saved_eps = [e for e in result["epochs"] if e.get("saved_checkpoint")]
        if saved_eps:
            result["best"]["epoch"] = saved_eps[-1]["epoch"]

    return result


def filter_by_time(data: dict, start_unix: float = None, end_unix: float = None) -> dict:
    """Filter epochs by Unix timestamp range."""
    if start_unix is None and end_unix is None:
        return data

    filtered = data.copy()
    filtered["filter"] = {
        "start_unix": int(start_unix) if start_unix else None,
        "end_unix": int(end_unix) if end_unix else None,
        "start_iso": datetime.fromtimestamp(start_unix).strftime("%Y-%m-%dT%H:%M:%S") if start_unix else None,
        "end_iso": datetime.fromtimestamp(end_unix).strftime("%Y-%m-%dT%H:%M:%S") if end_unix else None,
    }

    filtered["epochs"] = [
        ep for ep in data["epochs"]
        if (start_unix is None or ep.get("timestamp_unix", 0) >= start_unix)
        and (end_unix is None or ep.get("timestamp_unix", float("inf")) <= end_unix)
    ]

    return filtered


def filter_by_epoch(data: dict, ep_start: int = None, ep_end: int = None) -> dict:
    """Filter by epoch number range."""
    if ep_start is None and ep_end is None:
        return data

    filtered = data.copy()
    filtered["filter"] = {
        "epoch_start": ep_start,
        "epoch_end": ep_end,
    }

    filtered["epochs"] = [
        ep for ep in data["epochs"]
        if (ep_start is None or ep["epoch"] >= ep_start)
        and (ep_end is None or ep["epoch"] <= ep_end)
    ]

    return filtered


def main():
    p = argparse.ArgumentParser(description="Convert training log to JSON with Unix timestamps")
    p.add_argument("--file", required=True, type=Path, help="Path to training log file")
    p.add_argument("--start", default=None, help="Start time filter (ISO 'YYYY-MM-DD HH:MM:SS' or Unix timestamp)")
    p.add_argument("--end", default=None, help="End time filter (ISO 'YYYY-MM-DD HH:MM:SS' or Unix timestamp)")
    p.add_argument("--epoch-start", default=None, type=int, help="Start epoch number filter")
    p.add_argument("--epoch-end", default=None, type=int, help="End epoch number filter")
    p.add_argument("--output", "-o", default=None, type=Path, help="Output JSON file (default: stdout)")
    args = p.parse_args()

    if not args.file.exists():
        sys.exit(f"[ERROR] File not found: {args.file}")

    # Parse
    data = parse_log(args.file.resolve())

    # Filter by time
    if args.start or args.end:
        start_u = _to_unix(args.start) if args.start else None
        end_u = _to_unix(args.end) if args.end else None
        data = filter_by_time(data, start_u, end_u)

    # Filter by epoch
    if args.epoch_start is not None or args.epoch_end is not None:
        data = filter_by_epoch(data, args.epoch_start, args.epoch_end)

    # Output
    json_str = json.dumps(data, indent=2, ensure_ascii=False)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json_str, encoding="utf-8")
        print(f"[OK] Saved to {args.output}")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
