"""
generate_pdfs.py — Generate PDF reports from existing JSON report files.

Reads the *_report.json files already produced by ingest_ecg_extracts.py,
reconstructs the segment data, and generates only the PDF. No ML inference,
no PNG generation, no re-processing.

Usage:
    # Single patient folder
    python scripts/generate_pdfs.py --folder outputs/patient_ecg/ADM441825561

    # All patient folders under outputs/patient_ecg/
    python scripts/generate_pdfs.py --all outputs/patient_ecg

    # From raw ECG source (re-parses signal for ECG strips, but skips PNG/JSON)
    python scripts/generate_pdfs.py --source ECG_Data_Extracts/ADM441825561.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from data.ingest_ecg_extracts import (
    TARGET_FS,
    WINDOW_SAMPLES,
    _generate_pdf_report,
    _parse_extract_json,
    _resample,
    _segment,
    _detect_r_peaks,
    _run_inference,
    _run_rules_engine,
    _extract_morphology,
)


def _pdf_from_source(json_path: Path, output_dir: Path):
    """
    Re-parse raw ECG, run inference, generate PDF only (no PNG/JSON).
    """
    admission_id, runs = _parse_extract_json(json_path, None, None)
    if not runs:
        print(f"[pdf] No data in {json_path.name}. Skipping.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = json_path.stem
    all_pdf_segments = []
    total_segments = 0

    for run_idx, (signal, src_fs, t_start, t_end) in enumerate(runs):
        print(f"[pdf] Run {run_idx}: {len(signal)} samples, {t_start} to {t_end}")

        signal_125 = _resample(signal.tolist(), src_fs)
        windows = _segment(signal_125)
        print(f"[pdf] {len(windows)} segments")

        for idx, window in enumerate(windows):
            seg_name = f"{filename}_run{run_idx}" if len(runs) > 1 else filename
            r_peaks = _detect_r_peaks(window)

            rhythm_label, rhythm_conf, ectopy_label, ectopy_conf = _run_inference(window)

            rules_result = _run_rules_engine(
                window, r_peaks,
                rhythm_label, rhythm_conf,
                ectopy_label, ectopy_conf,
                fs=TARGET_FS,
            )

            morph_data = _extract_morphology(window, r_peaks)

            all_pdf_segments.append({
                "window": window,
                "r_peaks": r_peaks,
                "rhythm_label": rhythm_label,
                "rhythm_conf": rhythm_conf,
                "ectopy_label": ectopy_label,
                "ectopy_conf": ectopy_conf,
                "rules_result": rules_result,
                "morph_data": morph_data,
                "seg_idx": total_segments,
                "run_idx": run_idx,
                "t_start": t_start,
                "t_end": t_end,
            })
            total_segments += 1

    if all_pdf_segments:
        pdf_path = _generate_pdf_report(
            admission_id=admission_id,
            filename=filename,
            runs=runs,
            all_segments=all_pdf_segments,
            output_dir=output_dir,
        )
        if pdf_path:
            print(f"[pdf] PDF -> {pdf_path}")
    print(f"[pdf] Done. {total_segments} segments for {json_path.name}")


def _pdf_from_reports(folder: Path):
    """
    Read existing *_report.json files + reconstruct segment data for PDF.
    Needs the raw ECG source to draw the waveform strips.
    Falls back to --source mode if ECG_Data_Extracts file exists.
    """
    report_files = sorted(folder.glob("*_report.json"))
    report_files = [f for f in report_files if not f.name.endswith("_report.pdf")]

    if not report_files:
        print(f"[pdf] No *_report.json files in {folder}. Skipping.")
        return

    # Try to find the matching raw ECG source
    # Admission ID from first report
    with open(report_files[0], "r", encoding="utf-8") as f:
        first_report = json.load(f)
    adm_id = first_report.get("patient_id", "")
    if not adm_id:
        adm_id = first_report.get("_detailed", {}).get("prediction", {})
        adm_id = folder.name  # fallback to folder name

    # Look for raw source in ECG_Data_Extracts/
    source_candidates = [
        BASE_DIR / "ECG_Data_Extracts" / f"{adm_id}.json",
        BASE_DIR / "ECG_Data_Extracts" / f"{folder.name}.json",
    ]
    for src in source_candidates:
        if src.exists():
            print(f"[pdf] Found raw ECG source: {src.name}")
            _pdf_from_source(src, folder)
            return

    print(f"[pdf] No raw ECG source found for {adm_id}. "
          f"Use --source to specify the ECG_Data_Extracts JSON file.")


def main():
    p = argparse.ArgumentParser(
        description="Generate PDF reports from existing analysis outputs"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=Path,
                       help="Raw ECG JSON file (ECG_Data_Extracts/*.json) — re-runs inference, generates PDF only")
    group.add_argument("--folder", type=Path,
                       help="Patient output folder with *_report.json files")
    group.add_argument("--all", type=Path,
                       help="Parent folder containing patient subfolders")

    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output directory (default: same as input folder)")

    args = p.parse_args()

    if args.source:
        src = args.source.resolve()
        if not src.exists():
            sys.exit(f"[ERROR] File not found: {src}")
        out = args.output.resolve() if args.output else BASE_DIR / "outputs" / "patient_ecg" / src.stem
        _pdf_from_source(src, out)

    elif args.folder:
        folder = args.folder.resolve()
        if not folder.is_dir():
            sys.exit(f"[ERROR] Not a directory: {folder}")
        _pdf_from_reports(folder)

    elif args.all:
        parent = args.all.resolve()
        if not parent.is_dir():
            sys.exit(f"[ERROR] Not a directory: {parent}")
        subdirs = sorted([d for d in parent.iterdir() if d.is_dir()])
        if not subdirs:
            # Maybe files are directly in parent
            _pdf_from_reports(parent)
        else:
            print(f"[pdf] Found {len(subdirs)} patient folders")
            for d in subdirs:
                print(f"\n{'='*50}")
                _pdf_from_reports(d)
            print(f"\n{'='*50}")
            print(f"[pdf] BATCH COMPLETE. {len(subdirs)} patients processed.")


if __name__ == "__main__":
    main()
