"""
convert_json_to_signal.py
=========================
Converts various hospital ECG JSON formats into the standard pipeline format:
  {"admissionId": "...", "signal": [...floats in mV...], "fs": 125}

Handles:
  1. ADM MongoDB packet format  — array of docs, signal in "data" key, 125 samples/packet (already mV)
  2. ecg_export flat format     — single dict with "ecgData" key, already mV floats
  3. ECG_WITH_HR packet format  — array of docs with "ecgData" key, raw ADC integers → divide by 1000

Usage:
  python scripts/convert_json_to_signal.py --file ADM1249837065.json
  python scripts/convert_json_to_signal.py --file ecg_analysis_package/ecg_export_20260422_125530.json
  python scripts/convert_json_to_signal.py --all   (converts all known files)
"""

import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FS = 125


def convert_adm_packet_format(data: list) -> dict:
    """
    Handles ADM MongoDB packet format.
    Array of docs, each with 'data' key (125 samples, already mV).
    Sorts by packetNo and concatenates.
    """
    sorted_docs = sorted(data, key=lambda d: d.get("packetNo", 0))
    signal = []
    for doc in sorted_docs:
        signal.extend(doc.get("data", []))

    first = sorted_docs[0]
    return {
        "admissionId": first.get("admissionId", "unknown"),
        "deviceId":    first.get("deviceId", "unknown"),
        "patientId":   first.get("patientId", "unknown"),
        "facilityId":  first.get("facilityId", "unknown"),
        "timestamp":   first.get("timestamp") or first.get("utcTimestamp"),
        "fs":          FS,
        "signal":      signal,
        "source":      "adm_mongodb_packets",
    }


def convert_ecg_export_flat(data: dict) -> dict:
    """
    Handles ecg_export flat format.
    Single dict with 'ecgData' key (floats, already mV).
    """
    signal = data.get("ecgData", data.get("signal", []))
    return {
        "admissionId": data.get("admissionId", "unknown"),
        "deviceId":    data.get("deviceId", "unknown"),
        "patientId":   data.get("patientId", "unknown"),
        "facilityId":  data.get("facilityId", "unknown"),
        "timestamp":   data.get("timestamp"),
        "fs":          data.get("sampleRateHz", FS),
        "signal":      signal,
        "source":      "ecg_export_flat",
    }


def convert_ecg_with_hr_packets(data: list) -> dict:
    """
    Handles ECG_WITH_HR packet format.
    Array of docs with 'ecgData' key containing raw ADC integers.
    Divides by 1000 to convert ADC → mV. Sorts by timestamp1.
    """
    sorted_docs = sorted(data, key=lambda d: d.get("timestamp1", 0))
    signal = []
    for doc in sorted_docs:
        signal.extend([v / 1000.0 for v in doc.get("ecgData", [])])

    first = sorted_docs[0]
    return {
        "admissionId": first.get("admissionId", first.get("macID", "unknown")),
        "deviceId":    first.get("macID", "unknown"),
        "patientId":   first.get("patientId", "unknown"),
        "facilityId":  first.get("facilityId", "unknown"),
        "timestamp":   first.get("timestamp1") or first.get("timestamp"),
        "fs":          FS,
        "signal":      signal,
        "source":      "ecg_with_hr_packets_adc_div1000",
    }


def detect_and_convert(filepath: Path) -> dict:
    with open(filepath) as f:
        raw = json.load(f)

    # Format 1: ADM MongoDB packets — list of dicts with "data" key (already mV)
    if isinstance(raw, list) and len(raw) > 0 and "data" in raw[0]:
        print(f"  Detected: ADM MongoDB packet format ({len(raw)} packets)")
        result = convert_adm_packet_format(raw)

    # Format 3: ECG_WITH_HR packets — list of dicts with "ecgData" key (raw ADC integers)
    elif isinstance(raw, list) and len(raw) > 0 and "ecgData" in raw[0]:
        print(f"  Detected: ECG_WITH_HR packet format ({len(raw)} packets, ADC÷1000)")
        result = convert_ecg_with_hr_packets(raw)

    # Format 2: ecg_export flat — single dict with "ecgData"
    elif isinstance(raw, dict) and "ecgData" in raw:
        print(f"  Detected: ecg_export flat format ({len(raw.get('ecgData',[]))} samples)")
        result = convert_ecg_export_flat(raw)

    else:
        print(f"  [SKIP] Unknown or already-converted format in {filepath.name}")
        return None

    total = len(result["signal"])
    duration = total / result["fs"]
    print(f"  Signal: {total} samples = {duration:.1f}s at {result['fs']} Hz")
    print(f"  admissionId: {result['admissionId']}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Single file to convert")
    parser.add_argument("--all",  action="store_true", help="Convert all known files")
    args = parser.parse_args()

    targets = []
    if args.file:
        targets = [Path(args.file)]
    elif args.all:
        targets = [
            BASE_DIR / "ADM1249837065.json",
            BASE_DIR / "ADM1732567603.json",
            BASE_DIR / "ecg_analysis_package" / "ecg_export_20260422_125530.json",
            BASE_DIR / "ecg_analysis_package" / "ecg_export_20260422_123614.json",
        ]
    else:
        parser.print_help()
        return

    for fp in targets:
        if not fp.exists():
            print(f"[NOT FOUND] {fp}")
            continue

        print(f"\nConverting: {fp.name}")
        result = detect_and_convert(fp)
        if result is None:
            continue

        with open(fp, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {fp}")


if __name__ == "__main__":
    main()
