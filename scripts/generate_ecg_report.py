"""
generate_ecg_report.py
======================
Processes an ECG_Data_Extracts JSON file through the full V3 pipeline
and saves a clinical PDF report with per-segment ECG strips annotated
with rhythm and ectopy labels.

Usage:
  python scripts/generate_ecg_report.py --file ecg_data_extracts/ADM1196270205.json
  python scripts/generate_ecg_report.py --file ecg_data_extracts/ADM441825561.json

Output:
  reports/<admissionId>_<timestamp>.pdf
"""

import sys
import os
import json
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from ecg_processor import process

FS = 125
SEG_LEN = 1250  # 10s × 125 Hz
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Colour scheme for rhythm/ectopy labels
RHYTHM_COLORS = {
    "Sinus Rhythm":              "#27ae60",
    "Sinus Bradycardia":         "#2ecc71",
    "Sinus Tachycardia":         "#f39c12",
    "Atrial Fibrillation":       "#e74c3c",
    "Atrial Flutter":            "#c0392b",
    "Bundle Branch Block":       "#8e44ad",
    "1st Degree AV Block":       "#2980b9",
    "2nd Degree AV Block Type 2":"#1a5276",
    "3rd Degree AV Block":       "#17202a",
    "Artifact":                  "#7f8c8d",
    "Unknown":                   "#95a5a6",
}
ECTOPY_COLORS = {
    "None":   "#27ae60",
    "PVC":    "#e74c3c",
    "PAC":    "#e67e22",
}


def load_signal(filepath: Path):
    with open(filepath) as f:
        raw = json.load(f)

    if isinstance(raw, list) and raw and "value" in raw[0]:
        # ECG_Data_Extracts format
        sorted_pkts = sorted(raw, key=lambda d: d.get("packetNo", 0))
        signal = []
        for pkt in sorted_pkts:
            v = pkt.get("value", [])
            if v and isinstance(v[0], list):
                for sub in v:
                    signal.extend(sub)
            else:
                signal.extend(v)
        meta = sorted_pkts[0]
        return np.array(signal, dtype=np.float32), meta

    elif isinstance(raw, dict) and "signal" in raw:
        # Already-converted standard format
        return np.array(raw["signal"], dtype=np.float32), raw

    elif isinstance(raw, list) and raw and "data" in raw[0]:
        # ADM MongoDB packet format
        sorted_pkts = sorted(raw, key=lambda d: d.get("packetNo", 0))
        signal = []
        for pkt in sorted_pkts:
            signal.extend(pkt.get("data", []))
        meta = sorted_pkts[0]
        return np.array(signal, dtype=np.float32), meta

    else:
        raise ValueError(f"Unknown format in {filepath.name}")


def rhythm_color(label: str) -> str:
    return RHYTHM_COLORS.get(label, "#95a5a6")


def ectopy_color(label: str) -> str:
    return ECTOPY_COLORS.get(label, "#95a5a6")


def make_segment_strip(signal_chunk: np.ndarray, seg: dict, tmp_path: Path) -> str:
    """Render a 10-second ECG strip with rhythm/ectopy annotations. Returns PNG path."""
    fig, ax = plt.subplots(figsize=(14, 2.2), dpi=100)
    t = np.arange(len(signal_chunk)) / FS

    rhythm = seg.get("rhythm_label", "Unknown")
    ectopy = seg.get("ectopy_label", "None")
    rconf  = seg.get("rhythm_confidence", 0)
    econf  = seg.get("ectopy_confidence", 0)
    hr     = seg.get("morphology", {}).get("hr_bpm")
    sqi    = seg.get("signal_quality", 0)

    # ECG line
    ax.plot(t, signal_chunk, color="#1a1a2e", linewidth=0.7, zorder=3)

    # Coloured background band for rhythm
    rclr = rhythm_color(rhythm)
    ax.axhspan(
        signal_chunk.min() - 0.1,
        signal_chunk.max() + 0.1,
        alpha=0.06, color=rclr, zorder=1
    )

    # Grid
    ax.set_facecolor("#fafafa")
    ax.grid(True, which="both", color="#e8c0c0", linewidth=0.4, zorder=0)
    ax.set_xlim(0, len(signal_chunk) / FS)

    # Labels on plot
    hr_str = f"  HR: {hr:.0f} bpm" if hr else ""
    ax.set_title(
        f"Seg {seg['segment_index']+1}  |  {seg['start_time_s']:.0f}s – {seg['end_time_s']:.0f}s"
        f"  |  Rhythm: {rhythm} ({rconf:.0%})  |  Ectopy: {ectopy} ({econf:.0%})"
        f"{hr_str}  |  SQI: {sqi:.2f}",
        fontsize=8, loc="left", pad=3,
        color="#2c3e50"
    )
    ax.set_ylabel("mV", fontsize=7)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.tick_params(labelsize=6)

    # Ectopy colour bar on right spine
    eclr = ectopy_color(ectopy)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.spines["right"].set_color(eclr)
    ax.spines["right"].set_linewidth(4)
    ax.spines["top"].set_color(rclr)
    ax.spines["top"].set_linewidth(4)

    plt.tight_layout(pad=0.4)
    out = str(tmp_path / f"seg_{seg['segment_index']:04d}.png")
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out


def make_summary_chart(segments: list, tmp_path: Path) -> str:
    """Bar chart of rhythm distribution across all segments."""
    from collections import Counter
    rhythms = [s.get("rhythm_label", "Unknown") for s in segments]
    counts = Counter(rhythms)
    labels = list(counts.keys())
    vals   = [counts[l] for l in labels]
    clrs   = [rhythm_color(l) for l in labels]

    fig, ax = plt.subplots(figsize=(10, 3), dpi=100)
    bars = ax.barh(labels, vals, color=clrs, edgecolor="white", height=0.6)
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xlabel("Number of segments", fontsize=8)
    ax.set_title("Rhythm distribution across all segments", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    out = str(tmp_path / "summary_chart.png")
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out


def build_pdf(result: dict, signal: np.ndarray, out_path: Path):
    import tempfile, shutil
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        _build_pdf_inner(result, signal, out_path, tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _build_pdf_inner(result: dict, signal: np.ndarray, out_path: Path, tmp_dir: Path):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm,
    )

    W = A4[0] - 30*mm   # usable width

    title_style = ParagraphStyle("title", fontSize=16, fontName="Helvetica-Bold",
                                  spaceAfter=4, alignment=TA_CENTER)
    sub_style   = ParagraphStyle("sub", fontSize=9, fontName="Helvetica",
                                  spaceAfter=2, alignment=TA_CENTER, textColor=colors.HexColor("#666666"))
    h2_style    = ParagraphStyle("h2", fontSize=11, fontName="Helvetica-Bold",
                                  spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#2c3e50"))
    body_style  = ParagraphStyle("body", fontSize=8, fontName="Helvetica", spaceAfter=2)
    label_style = ParagraphStyle("label", fontSize=7, fontName="Helvetica", spaceAfter=1,
                                  textColor=colors.HexColor("#555555"))

    analysis   = result["analysis"]
    summary    = analysis["summary"]
    segments   = analysis["segments"]
    admission  = result.get("admissionId", "N/A")
    device_id  = result.get("deviceId", "N/A")
    patient_id = result.get("patientId", "N/A")
    facility   = result.get("facilityId", "N/A")
    proc_time  = result.get("_processing_time_s", "N/A")
    gen_time   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    story = []

    # ── Cover / header ─────────────────────────────────────────────────────────
    story.append(Paragraph("ECG Arrhythmia Analysis Report", title_style))
    story.append(Paragraph("LifeSigns AI  •  V3 Pipeline", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2c3e50"), spaceAfter=6))

    meta_data = [
        ["Admission ID", admission,  "Generated",    gen_time],
        ["Patient ID",   patient_id, "Device ID",    device_id],
        ["Facility ID",  facility,   "Proc. Time",   f"{proc_time}s"],
        ["Total Segments", str(summary["total_segments"]),
         "Signal Duration", f"{summary['total_segments']*10}s"],
    ]
    meta_table = Table(meta_data, colWidths=[35*mm, 55*mm, 35*mm, 55*mm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",    (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",    (2,0), (2,-1), "Helvetica-Bold"),
        ("BACKGROUND",  (0,0), (-1,-1), colors.HexColor("#f4f6f7")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#eaf0fb"), colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#cccccc")),
        ("PADDING",     (0,0), (-1,-1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 4*mm))

    # ── Summary box ─────────────────────────────────────────────────────────────
    story.append(Paragraph("Summary", h2_style))
    dom_rhythm   = summary["dominant_rhythm"]
    arrhythmia   = "YES — Arrhythmia Detected" if summary["arrhythmia_detected"] else "No arrhythmia detected"
    events_found = ", ".join(summary["events_found"]) if summary["events_found"] else "None"
    hr_bpm       = analysis.get("heart_rate_bpm", "N/A")

    summ_clr = "#fdecea" if summary["arrhythmia_detected"] else "#eafbea"
    summ_data = [
        ["Dominant Rhythm",   dom_rhythm,    "Heart Rate",    f"{hr_bpm} bpm"],
        ["Arrhythmia Status", arrhythmia,    "Events Found",  events_found],
    ]
    summ_table = Table(summ_data, colWidths=[35*mm, 70*mm, 30*mm, 45*mm])
    summ_table.setStyle(TableStyle([
        ("FONTNAME",    (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",    (2,0), (2,-1), "Helvetica-Bold"),
        ("BACKGROUND",  (0,0), (-1,-1), colors.HexColor(summ_clr)),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#cccccc")),
        ("PADDING",     (0,0), (-1,-1), 5),
        ("TEXTCOLOR",   (1,1), (1,1), colors.HexColor("#c0392b") if summary["arrhythmia_detected"] else colors.HexColor("#27ae60")),
        ("FONTNAME",    (1,1), (1,1), "Helvetica-Bold"),
    ]))
    story.append(summ_table)
    story.append(Spacer(1, 4*mm))

    # ── Rhythm distribution chart ────────────────────────────────────────────────
    story.append(Paragraph("Rhythm Distribution", h2_style))
    chart_path = make_summary_chart(segments, tmp_dir)
    story.append(Image(chart_path, width=W, height=60*mm))
    story.append(Spacer(1, 4*mm))

    # ── Legend ───────────────────────────────────────────────────────────────────
    story.append(Paragraph("Legend", h2_style))
    legend_items = []
    for lbl, clr in list(RHYTHM_COLORS.items())[:8]:
        legend_items.append(
            Paragraph(f'<font color="{clr}">■</font>  {lbl}', label_style)
        )
    leg_data = [[legend_items[i] for i in range(min(4, len(legend_items)))]]
    if len(legend_items) > 4:
        leg_data.append([legend_items[i] for i in range(4, min(8, len(legend_items)))])
    leg_table = Table(leg_data, colWidths=[W/4]*4)
    leg_table.setStyle(TableStyle([("PADDING", (0,0), (-1,-1), 2)]))
    story.append(leg_table)
    story.append(Spacer(1, 2*mm))

    ectopy_leg = [
        Paragraph(f'<font color="{ECTOPY_COLORS["None"]}">■</font>  No Ectopy (right spine)', label_style),
        Paragraph(f'<font color="{ECTOPY_COLORS["PVC"]}">■</font>  PVC (right spine)', label_style),
        Paragraph(f'<font color="{ECTOPY_COLORS["PAC"]}">■</font>  PAC (right spine)', label_style),
        Paragraph('Top spine colour = Rhythm label', label_style),
    ]
    eleg_table = Table([ectopy_leg], colWidths=[W/4]*4)
    eleg_table.setStyle(TableStyle([("PADDING", (0,0), (-1,-1), 2)]))
    story.append(eleg_table)
    story.append(PageBreak())

    # ── Segment index table ──────────────────────────────────────────────────────
    story.append(Paragraph("Segment Index", h2_style))
    tbl_header = ["#", "Time", "Rhythm", "Conf", "Ectopy", "Conf", "HR (bpm)", "SQI"]
    tbl_rows   = [tbl_header]
    for seg in segments:
        hr_val = seg.get("morphology", {}).get("hr_bpm")
        tbl_rows.append([
            str(seg["segment_index"]+1),
            f"{seg['start_time_s']:.0f}s–{seg['end_time_s']:.0f}s",
            seg.get("rhythm_label", "?"),
            f"{seg.get('rhythm_confidence',0):.0%}",
            seg.get("ectopy_label", "?"),
            f"{seg.get('ectopy_confidence',0):.0%}",
            f"{hr_val:.0f}" if hr_val else "N/A",
            f"{seg.get('signal_quality',0):.2f}",
        ])

    col_w = [10*mm, 22*mm, 50*mm, 14*mm, 18*mm, 14*mm, 18*mm, 14*mm]
    idx_table = Table(tbl_rows, colWidths=col_w, repeatRows=1)
    row_styles = [
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 7),
        ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#cccccc")),
        ("PADDING",     (0,0), (-1,-1), 3),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f4f6f7")]),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("ALIGN",       (2,1), (2,-1), "LEFT"),
        ("ALIGN",       (4,1), (4,-1), "LEFT"),
    ]
    # Colour-code rhythm column
    for i, seg in enumerate(segments, start=1):
        r = seg.get("rhythm_label", "Unknown")
        e = seg.get("ectopy_label", "None")
        if r not in ("Sinus Rhythm", "Unknown"):
            idx_table_clr = colors.HexColor(rhythm_color(r) + "33")  # 20% opacity hex trick
            row_styles.append(("BACKGROUND", (2,i), (2,i), colors.HexColor(rhythm_color(r)+"22")))
        if e not in ("None",):
            row_styles.append(("BACKGROUND", (4,i), (4,i), colors.HexColor(ectopy_color(e)+"33")))

    idx_table.setStyle(TableStyle(row_styles))
    story.append(idx_table)
    story.append(PageBreak())

    # ── Per-segment ECG strips ───────────────────────────────────────────────────
    story.append(Paragraph("ECG Strips — Per Segment", h2_style))
    story.append(Spacer(1, 2*mm))

    strips_per_page = 4
    for i, seg in enumerate(segments):
        start = seg["segment_index"] * SEG_LEN
        end   = start + SEG_LEN
        chunk = signal[start:end]
        if len(chunk) < SEG_LEN:
            chunk = np.pad(chunk, (0, SEG_LEN - len(chunk)))

        png = make_segment_strip(chunk, seg, tmp_dir)
        story.append(Image(png, width=W, height=45*mm))
        story.append(Spacer(1, 1*mm))

        if (i + 1) % strips_per_page == 0 and i < len(segments) - 1:
            story.append(PageBreak())

    doc.build(story)


def main():
    parser = argparse.ArgumentParser(description="ECG pipeline → PDF report")
    parser.add_argument("--file", required=True, help="Path to ECG JSON file")
    args = parser.parse_args()

    fp = Path(args.file)
    if not fp.is_absolute():
        fp = BASE_DIR / fp
    if not fp.exists():
        print(f"[ERROR] File not found: {fp}")
        sys.exit(1)

    print(f"Loading: {fp.name}")
    signal, meta = load_signal(fp)
    print(f"Signal: {len(signal)} samples = {len(signal)/FS:.1f}s")

    print("Running pipeline...")
    result = process(
        ecg_data=signal.tolist(),
        admission_id=meta.get("admissionId", "unknown"),
        device_id=meta.get("deviceId", "unknown"),
        patient_id=meta.get("patientId", "unknown"),
        facility_id=meta.get("facilityId", "unknown"),
        timestamp=meta.get("utcTimestamp") or meta.get("timestamp"),
    )

    analysis = result["analysis"]
    print(f"Done: {analysis['summary']['total_segments']} segments | "
          f"Rhythm: {analysis['summary']['dominant_rhythm']} | "
          f"Events: {analysis['summary']['events_found']}")

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    adm = result.get("admissionId", "unknown").replace("/", "_")
    out = REPORTS_DIR / f"{adm}_{ts}.pdf"

    print(f"Building PDF: {out.name}")
    build_pdf(result, signal, out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
