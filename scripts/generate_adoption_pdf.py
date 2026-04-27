"""
Generate a clean, readable PDF of the BATCH_PROCESS adoption plan.
Output: docs/BATCH_PROCESS_ADOPTION_PLAN.pdf
"""
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak
)
from reportlab.platypus.flowables import BalancedColumns

# ── Colours ──────────────────────────────────────────────────────────────────
C_RED       = colors.HexColor("#C0392B")
C_ORANGE    = colors.HexColor("#E67E22")
C_BLUE      = colors.HexColor("#2980B9")
C_GREEN     = colors.HexColor("#27AE60")
C_DARK      = colors.HexColor("#1A1A2E")
C_GREY_BG   = colors.HexColor("#F4F6F8")
C_GREY_LINE = colors.HexColor("#BDC3C7")
C_P0        = colors.HexColor("#FDEDEC")   # light red
C_P1        = colors.HexColor("#FEF9E7")   # light yellow
C_P2        = colors.HexColor("#EBF5FB")   # light blue
C_HEAD_BG   = colors.HexColor("#2C3E50")
C_WHITE     = colors.white

PAGE_W, PAGE_H = A4
ML  = 18 * mm
MR  = 18 * mm
MT  = 20 * mm
MB  = 20 * mm
TW  = PAGE_W - ML - MR  # usable width

# ── Styles ────────────────────────────────────────────────────────────────────
ss = getSampleStyleSheet()

def S(name, **kw):
    base = ss[name] if name in ss else ss["Normal"]
    return ParagraphStyle(name + str(id(kw)), parent=base, **kw)

TITLE   = S("Title",  fontSize=22, textColor=C_DARK,  spaceAfter=4,  leading=28, alignment=TA_CENTER)
SUBTITLE= S("Normal", fontSize=11, textColor=C_GREY_LINE, spaceAfter=10, alignment=TA_CENTER)
H1      = S("Heading1", fontSize=15, textColor=C_WHITE,  spaceAfter=6, spaceBefore=14, leading=20)
H2      = S("Heading2", fontSize=12, textColor=C_DARK,   spaceAfter=4, spaceBefore=10, leading=16)
H3      = S("Heading3", fontSize=10, textColor=C_BLUE,   spaceAfter=3, spaceBefore=8,  leading=14)
BODY    = S("Normal",   fontSize=9,  textColor=C_DARK,   spaceAfter=3, leading=13, alignment=TA_JUSTIFY)
BULLET  = S("Normal",   fontSize=9,  textColor=C_DARK,   spaceAfter=2, leading=13, leftIndent=12,
            bulletIndent=4)
CODE    = S("Code",     fontSize=8,  fontName="Courier", textColor=colors.HexColor("#1a1a1a"),
            backColor=colors.HexColor("#F0F0F0"), spaceAfter=3, leading=12, leftIndent=8, rightIndent=8)
LABEL_P0= S("Normal",   fontSize=9,  textColor=C_RED,    fontName="Helvetica-Bold")
LABEL_P1= S("Normal",   fontSize=9,  textColor=C_ORANGE, fontName="Helvetica-Bold")
LABEL_P2= S("Normal",   fontSize=9,  textColor=C_BLUE,   fontName="Helvetica-Bold")
CAPTION = S("Normal",   fontSize=8,  textColor=colors.HexColor("#7F8C8D"), spaceAfter=2, alignment=TA_CENTER)
NOTE    = S("Normal",   fontSize=8,  textColor=colors.HexColor("#555555"), leftIndent=10, spaceAfter=2)

def hr(): return HRFlowable(width="100%", thickness=0.5, color=C_GREY_LINE, spaceAfter=6, spaceBefore=4)
def sp(h=4): return Spacer(1, h * mm)

def section_header(text, colour=C_HEAD_BG):
    tbl = Table([[Paragraph(text, H1)]], colWidths=[TW])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colour),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colour]),
    ]))
    return tbl

def pill(text, bg, fg=C_WHITE):
    tbl = Table([[Paragraph(f"<b>{text}</b>", S("Normal", fontSize=8, textColor=fg, leading=11))]],
                colWidths=[None])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), bg),
        ("ROUNDEDCORNERS",(0,0),(-1,-1), [3,3,3,3]),
        ("LEFTPADDING",  (0,0),(-1,-1), 5),
        ("RIGHTPADDING", (0,0),(-1,-1), 5),
        ("TOPPADDING",   (0,0),(-1,-1), 2),
        ("BOTTOMPADDING",(0,0),(-1,-1), 2),
    ]))
    return tbl

def two_col_table(data, col_ratio=(0.45, 0.55), header=None, bg_header=C_HEAD_BG):
    cw = [TW * r for r in col_ratio]
    rows = []
    if header:
        rows.append([Paragraph(f"<b>{header[0]}</b>",
                     S("Normal", fontSize=9, textColor=C_WHITE, leading=12)),
                     Paragraph(f"<b>{header[1]}</b>",
                     S("Normal", fontSize=9, textColor=C_WHITE, leading=12))])
    for row in data:
        rows.append([Paragraph(str(row[0]), S("Normal", fontSize=8.5, leading=12)),
                     Paragraph(str(row[1]), S("Normal", fontSize=8.5, leading=12))])
    tbl = Table(rows, colWidths=cw, repeatRows=1 if header else 0)
    style = [
        ("GRID",        (0,0),(-1,-1), 0.4, C_GREY_LINE),
        ("LEFTPADDING",  (0,0),(-1,-1), 6),
        ("RIGHTPADDING", (0,0),(-1,-1), 6),
        ("TOPPADDING",   (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_GREY_BG]),
        ("VALIGN",      (0,0),(-1,-1), "TOP"),
    ]
    if header:
        style += [
            ("BACKGROUND", (0,0), (-1,0), bg_header),
            ("TEXTCOLOR",  (0,0), (-1,0), C_WHITE),
        ]
    tbl.setStyle(TableStyle(style))
    return tbl

def four_col_table(data, col_widths, header=None):
    rows = []
    if header:
        rows.append([Paragraph(f"<b>{h}</b>", S("Normal", fontSize=8.5, textColor=C_WHITE, leading=12))
                     for h in header])
    for row in data:
        rows.append([Paragraph(str(c), S("Normal", fontSize=8, leading=12)) for c in row])
    tbl = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("GRID",         (0,0),(-1,-1), 0.4, C_GREY_LINE),
        ("LEFTPADDING",  (0,0),(-1,-1), 5),
        ("RIGHTPADDING", (0,0),(-1,-1), 5),
        ("TOPPADDING",   (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_GREY_BG]),
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
    ]
    if header:
        style += [("BACKGROUND",(0,0),(-1,0), C_HEAD_BG)]
    tbl.setStyle(TableStyle(style))
    return tbl

def criteria_box(items, bg=C_GREY_BG):
    rows = [[Paragraph(f"• {item}", S("Normal", fontSize=8.5, leading=13))] for item in items]
    tbl = Table(rows, colWidths=[TW - 20])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), bg),
        ("LEFTPADDING",  (0,0),(-1,-1), 10),
        ("RIGHTPADDING", (0,0),(-1,-1), 8),
        ("TOPPADDING",   (0,0),(-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ("BOX",          (0,0),(-1,-1), 0.5, C_GREY_LINE),
    ]))
    return tbl

def algo_card(title, source, confidence, conf_colour, criteria_items, note=None, bg=C_GREY_BG):
    header_row = [
        Paragraph(f"<b>{title}</b>", S("Normal", fontSize=9.5, textColor=C_DARK, leading=14)),
        Paragraph(f"<i>{source}</i>", S("Normal", fontSize=8, textColor=colors.HexColor("#7F8C8D"), leading=12)),
        Paragraph(f"<b>Conf: {confidence}</b>",
                  S("Normal", fontSize=9, textColor=conf_colour, fontName="Helvetica-Bold", leading=12)),
    ]
    crit_rows = [[Paragraph(f"• {c}", S("Normal", fontSize=8.5, leading=13))] for c in criteria_items]
    inner = Table(crit_rows, colWidths=[TW - 30])
    inner.setStyle(TableStyle([
        ("LEFTPADDING",  (0,0),(-1,-1), 0),
        ("RIGHTPADDING", (0,0),(-1,-1), 0),
        ("TOPPADDING",   (0,0),(-1,-1), 1),
        ("BOTTOMPADDING",(0,0),(-1,-1), 1),
    ]))
    content = [header_row[0], header_row[1], header_row[2], inner]
    if note:
        content.append(Paragraph(f"⚠ {note}", NOTE))
    rows = [[c] for c in content]
    tbl = Table(rows, colWidths=[TW - 16])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), bg),
        ("BOX",          (0,0),(-1,-1), 0.8, C_GREY_LINE),
        ("LEFTPADDING",  (0,0),(-1,-1), 10),
        ("RIGHTPADDING", (0,0),(-1,-1), 8),
        ("TOPPADDING",   (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]))
    return tbl

# ── Build document ────────────────────────────────────────────────────────────
def build_pdf(out_path: Path):
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=ML, rightMargin=MR,
        topMargin=MT, bottomMargin=MB,
        title="BATCH_PROCESS Adoption Plan",
        author="LifeSigns ECG System",
    )
    story = []

    # ── Cover ──────────────────────────────────────────────────────────────────
    story += [
        sp(10),
        Paragraph("BATCH_PROCESS → V3 Pipeline", SUBTITLE),
        Paragraph("Adoption Plan", TITLE),
        Paragraph("What gets ported, replaced, or removed — and why", SUBTITLE),
        sp(2),
        hr(),
        sp(2),
        Paragraph(
            "This document describes every change being made to the V3 ECG arrhythmia pipeline, "
            "detailing what is adopted from <b>BATCH_PROCESS/lifesigns_engine.py</b>, what existing "
            "behaviour is corrected, and exactly which files and lines are affected.",
            BODY),
        sp(4),
    ]

    # ── Summary table ──────────────────────────────────────────────────────────
    story.append(section_header("  OVERVIEW — All Changes at a Glance"))
    story.append(sp(2))

    summary_data = [
        ["VTach / VFib spectral detection",       "Port from BATCH_PROCESS", "P0 — Critical", "CREATE lethal_detector.py"],
        ["Lethal detector wired into pipeline",    "New wiring",              "P0 — Critical", "MODIFY rhythm_orchestrator.py"],
        ["Preprocessed signal to lethal detector", "New field",               "P0 — Critical", "MODIFY ecg_processor.py"],
        ["SVT detection via signal processing",    "Port from BATCH_PROCESS", "P1",            "MODIFY lethal_detector.py"],
        ["ML veto at 0.88 conf — removed",         "Architecture fix",        "P1",            "MODIFY rhythm_orchestrator.py"],
        ["Idioventricular Rhythm — removed",       "Architecture fix",        "P1",            "MODIFY rhythm_orchestrator.py"],
        ["BBB renamed to IVCD",                    "Label fix",               "P1",            "MODIFY data_loader.py + orchestrator"],
        ["PVC/PAC template correlation refinement","Port from BATCH_PROCESS", "P2",            "CREATE beat_classifier.py"],
    ]

    p_colours = {
        "P0 — Critical": (C_P0, C_RED),
        "P1":            (C_P1, C_ORANGE),
        "P2":            (C_P2, C_BLUE),
    }

    cw = [TW*0.30, TW*0.20, TW*0.12, TW*0.38]
    hdr = ["Change", "Type", "Priority", "File(s)"]
    rows = [[Paragraph(f"<b>{h}</b>", S("Normal", fontSize=9, textColor=C_WHITE, leading=12)) for h in hdr]]
    for row in summary_data:
        bg, fg = p_colours.get(row[2], (C_WHITE, C_DARK))
        rows.append([
            Paragraph(row[0], S("Normal", fontSize=8.5, leading=12)),
            Paragraph(row[1], S("Normal", fontSize=8.5, leading=12, textColor=colors.HexColor("#555555"))),
            Paragraph(f"<b>{row[2]}</b>", S("Normal", fontSize=8.5, leading=12, textColor=fg)),
            Paragraph(f"<code>{row[3]}</code>", S("Normal", fontSize=8, fontName="Courier", leading=12)),
        ])

    tbl = Table(rows, colWidths=cw, repeatRows=1)
    row_bgs = [C_HEAD_BG] + [p_colours.get(r[2], (C_WHITE, C_DARK))[0] for r in summary_data]
    tbl.setStyle(TableStyle([
        ("GRID",        (0,0),(-1,-1), 0.4, C_GREY_LINE),
        ("LEFTPADDING",  (0,0),(-1,-1), 6),
        ("RIGHTPADDING", (0,0),(-1,-1), 6),
        ("TOPPADDING",   (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("VALIGN",      (0,0),(-1,-1), "TOP"),
        ("BACKGROUND",  (0,0),(-1,0), C_HEAD_BG),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [p_colours.get(r[2], (C_WHITE, C_DARK))[0] for r in summary_data]),
    ]))
    story += [tbl, sp(6), PageBreak()]

    # ── P0: VTach/VFib lethal detector ────────────────────────────────────────
    story.append(section_header("  P0 — CREATE   decision_engine/lethal_detector.py", C_RED))
    story += [sp(2),
              Paragraph("<b>VTach / VFib / SVT Detection — Signal Processing, No ML</b>", H2)]

    story += [
        Paragraph("<b>Why this is needed</b>", H3),
        criteria_box([
            "VFib has ZERO detection in the current system",
            "VTach only caught if ectopy ML labels ≥11 consecutive beats as PVC — ML has 3 VT training samples, 0 VFib samples",
            "SVT has 27 training samples (0 corrected) — ML cannot learn it",
            "BATCH_PROCESS has 4 deterministic signal-processing algorithms for these conditions",
            "Signal processing is deterministic, explainable, runs in milliseconds, does not degrade with distribution shift",
        ], C_P0),
        sp(3),
    ]

    # Algo 1
    story += [
        Paragraph("<b>Algorithm 1 — Spectral Lethal Pre-check</b>  "
                  "<i>(ported from spectral_lethal_precheck(), lines 260–343)</i>", H3),
        Paragraph(
            "Two-stage detection on the <b>bandpass-filtered</b> (preprocessed) signal. "
            "Must use filtered signal — raw signal has EMG noise that corrupts the 1.5–7 Hz band.",
            BODY),
        sp(2),
    ]

    # Stage 1 table
    s1_data = [
        ["Bigeminy fingerprint",
         "Even/odd RR alternation >12% AND each group CV <20%  →  SURVIVABLE\n(prevents false alarm on bigeminal PVC rhythm)"],
        ["Very regular + slow",
         "rr_cv <0.20 AND rate <130 BPM AND ≥6 beats  →  SURVIVABLE\n(prevents false alarm on sinus tachycardia)"],
        ["Flutter / SVT guard  ★ V3 addition",
         "rr_cv <0.08  →  SURVIVABLE\n(catches atrial flutter ~145 BPM and SVT — very regular rhythms not VTach)"],
        ["All other patterns",
         "Fall through to Stage 2 spectral analysis"],
    ]
    story += [
        Paragraph("Stage 1 — Organization Gate  (false-positive prevention)", H3),
        two_col_table(s1_data, (0.30, 0.70), header=["Gate Rule", "Action"]),
        sp(3),
        Paragraph("Stage 2 — Welch PSD Spectral Power Ratio", H3),
        Paragraph(
            "Computes <b>SPI</b> = power(1.5–7 Hz) / total power(0.5–40 Hz) using Welch periodogram. "
            "The 1.5–7 Hz band covers ventricular rates from 90–420 BPM.", BODY),
        sp(2),
    ]

    spi_data = [
        ["SPI > 0.75  AND  concentration > 0.35",
         "<b>VENTRICULAR TACHYCARDIA</b> — organised single-frequency power (one dominant spectral peak)",
         "0.92"],
        ["SPI > 0.75  AND  concentration ≤ 0.35",
         "<b>VENTRICULAR FIBRILLATION</b> — diffuse chaotic power across the band",
         "0.92"],
        ["SPI ≤ 0.75",
         "SURVIVABLE — power outside lethal band, proceed to next algorithm",
         "—"],
    ]
    cw3 = [TW*0.32, TW*0.52, TW*0.16]
    rows3 = [[Paragraph(f"<b>{h}</b>", S("Normal", fontSize=8.5, textColor=C_WHITE, leading=12))
              for h in ["Spectral Condition", "Outcome", "Confidence"]]]
    for r in spi_data:
        rows3.append([Paragraph(r[0], S("Normal", fontSize=8.5, fontName="Courier", leading=12)),
                      Paragraph(r[1], S("Normal", fontSize=8.5, leading=12)),
                      Paragraph(r[2], S("Normal", fontSize=8.5, leading=12, alignment=TA_CENTER))])
    t3 = Table(rows3, colWidths=cw3, repeatRows=1)
    t3.setStyle(TableStyle([
        ("GRID",        (0,0),(-1,-1), 0.4, C_GREY_LINE),
        ("BACKGROUND",  (0,0),(-1,0), C_HEAD_BG),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_P0, colors.HexColor("#FDEBD0"), C_GREY_BG]),
        ("LEFTPADDING",  (0,0),(-1,-1), 6),
        ("RIGHTPADDING", (0,0),(-1,-1), 6),
        ("TOPPADDING",   (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("VALIGN",      (0,0),(-1,-1), "TOP"),
    ]))
    story += [t3, sp(3)]

    # Algos 2-5
    algo_data = [
        ("Algorithm 2 — Kinetic VTach",
         "calculate_metrics() lines 951–972",
         "0.85", C_RED,
         [
             "HR > 100 bpm  (fast ventricular rate)",
             "QRS > 120 ms  (wide complex) — relaxed to >100 ms when HR > 150",
             "wide_qrs_fraction > 0.75  (>75% of beats are wide) — relaxed to >0.60 when HR > 150",
             "p_absent_fraction > 0.70  (AV dissociation — P-waves absent in >70% of beats)",
         ],
         "pr_int_sd (PR interval SD) not in V3 features — using P-absent arm only (threshold 0.70). This is the stronger criterion."),
        ("Algorithm 3 — Polymorphic VTach / Torsades de Pointes",
         "calculate_metrics() lines 905–915",
         "0.82", C_ORANGE,
         [
             "HR > 150 bpm  (rapid ventricular rate)",
             "0.08 < rr_cv < 0.55  (mild irregularity — not regular like SVT, not chaotic like VFib)",
             "p_absent_fraction > 0.60  (no visible P-waves — AV dissociation)",
         ], None),
        ("Algorithm 4 — Coarse VFib",
         "calculate_metrics() lines 896–903",
         "0.80", C_ORANGE,
         [
             "rr_cv > 0.50  (completely chaotic RR intervals)",
             "HR > 80 bpm  (fast rate — not junctional escape)",
             "p_absent_fraction > 0.60  (no organised atrial activity)",
         ], None),
        ("Algorithm 5 — SVT  (Supraventricular Tachycardia)",
         "calculate_metrics() svt_flag block",
         "0.80", C_BLUE,
         [
             "HR > 100 bpm  (tachycardia)",
             "QRS < 120 ms  (narrow complex — not ventricular origin)",
             "rr_cv < 0.10  (very regular — not AF)",
             "p_absent_fraction > 0.60  (P-waves absent or hidden in T-wave — AV node re-entry)",
         ],
         "SVT is not lethal but caught here because ML has only 27 training samples (0 corrected). "
         "Function will be named detect_signal_rhythm() to reflect non-lethal rhythms are also detected."),
    ]

    bg_colours = [C_P0, C_P1, C_P1, C_P2]
    for i, (title, source, conf, ccolour, criteria, note) in enumerate(algo_data):
        story += [algo_card(title, source, conf, ccolour, criteria, note, bg_colours[i]), sp(3)]

    # Feature mapping
    story += [
        Paragraph("<b>Feature Mapping — lifesigns_engine → V3 features dict</b>", H3),
        two_col_table([
            ["`metrics[\"hr\"]`",            "`features[\"mean_hr_bpm\"]`  — direct"],
            ["`metrics[\"hrv_cv\"]`",         "`features[\"rr_cv\"]`  — direct"],
            ["`metrics[\"qrs_dur\"]`",         "`features[\"qrs_duration_ms\"]`  — direct"],
            ["`qrs_wide_fraction`",             "`features[\"wide_qrs_fraction\"]`  — direct"],
            ["`p_absent_frac`",                 "`1.0 - features[\"p_wave_present_ratio\"]`  — invert"],
            ["`metrics[\"pr_int_sd\"]`",        "Not in V3 — skipped, use P-absent arm only"],
        ], (0.35, 0.65), header=["lifesigns_engine variable", "V3 features key / mapping"]),
        sp(6), PageBreak(),
    ]

    # ── P0: Wire orchestrator ─────────────────────────────────────────────────
    story.append(section_header("  P0 — MODIFY   decision_engine/rhythm_orchestrator.py  (wiring)", C_RED))
    story += [
        sp(2),
        Paragraph("<b>Add Step 3.5 — Lethal Gate between Sinus Gate and ML Rhythm</b>", H2),
        Paragraph(
            "The lethal detector is inserted after sinus detection (Step 3) and before the ML rhythm model (Step 4B). "
            "Because Step 4B is already gated on <code>decision.background_rhythm == \"Unknown\"</code>, "
            "if the lethal detector fires it automatically skips the ML rhythm step.", BODY),
        sp(2),
        Paragraph("<b>New pipeline flow after this change:</b>", H3),
    ]

    flow_data = [
        ["Step 3",   "Sinus Gate",   "signal processing — 10 criteria (unchanged)",
         "Sinus / Brady / Tachy → done, ML skipped"],
        ["Step 3.5", "Lethal Gate",  "signal processing — spectral + kinetic + polymorphic + coarse VFib + SVT  ★ NEW",
         "VTach / VFib / SVT → done, ML skipped"],
        ["Step 4A",  "Rule Events",  "Pause, Atrial Flutter, QRS/ST events (unchanged)",
         "Events added to event list"],
        ["Step 4B",  "ML Rhythm",    "CNNTransformerWithFeatures — only runs if Unknown (unchanged)",
         "AF / AFL / AVB / IVCD / NSVT / etc."],
        ["Step 4C",  "ML Ectopy",    "Per-beat PVC / PAC from ectopy model (unchanged)",
         "Beat-level labels"],
        ["Step 4D",  "Beat Refine",  "Template correlation override — ★ NEW (see P2)",
         "Low-conf ML calls corrected"],
        ["Step 5",   "Patterns",     "Bigeminy / Trigeminy / Couplet / NSVT / VT (unchanged)",
         "Pattern events"],
        ["Step 6",   "Priority",     "VF > VT > AF > NSVT promote to background (unchanged)",
         "background_rhythm finalised"],
        ["Step 7",   "Display",      "Suppression hierarchy — display-only (unchanged)",
         "final_display_events"],
    ]
    cw4 = [TW*0.09, TW*0.14, TW*0.46, TW*0.31]
    hdr4 = ["Step", "Layer", "Method", "Output"]
    story += [
        four_col_table(flow_data, cw4, hdr4),
        sp(3),
        Paragraph("<b>Insertion code in decide():</b>", H3),
        Paragraph(
            "After line ~98 (sinus gate), before line ~101 (rule events):", BODY),
        Paragraph(
            "if decision.background_rhythm == 'Unknown':<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;_signal  = clinical_features.get('_signal_clean') or clinical_features.get('_signal')<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;_r_peaks = clinical_features.get('r_peaks')<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;_fs      = int(clinical_features.get('fs', 125))<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;if _signal is not None and _r_peaks is not None:<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_lethal_label, _lethal_conf, _lethal_reason = detect_lethal_rhythm(...)<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if _lethal_label:<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;decision.background_rhythm = _lethal_label",
            CODE),
        sp(6), PageBreak(),
    ]

    # ── P0: ecg_processor.py ──────────────────────────────────────────────────
    story.append(section_header("  P0 — MODIFY   ecg_processor.py  (signal field)", C_RED))
    story += [
        sp(2),
        Paragraph("<b>Pass Preprocessed (Cleaned) Signal to Lethal Detector</b>", H2),
        Paragraph(
            "The lethal detector's spectral algorithm must receive the <b>bandpass-filtered</b> signal, "
            "not the raw ECG. Raw signal contains EMG noise (20–200 Hz) which corrupts the 1.5–7 Hz "
            "VTach/VFib band and can cause false positives.", BODY),
        sp(2),
        two_col_table([
            ["Current state",
             "clinical_features[\"_signal\"] = window.tolist()  ← raw ECG, not filtered"],
            ["After change",
             "clinical_features[\"_signal_clean\"] = v3.get(\"cleaned\", window).tolist()  ← bandpass-filtered"],
            ["Fallback",
             "If _signal_clean is not in features dict, lethal detector falls back to _signal (still works, slightly noisier)"],
        ], (0.18, 0.82)),
        sp(6), PageBreak(),
    ]

    # ── P1 section ────────────────────────────────────────────────────────────
    story.append(section_header("  P1 — MODIFY   rhythm_orchestrator.py  (architecture fixes)", colors.HexColor("#D35400")))
    story += [sp(2)]

    # ML veto
    story += [
        Paragraph("<b>Remove ML Veto Block (lines 81–94)</b>", H2),
        Paragraph(
            "The current orchestrator has a block that allows the <b>rhythm ML model to override the "
            "sinus gate</b> if it sees a dangerous rhythm at ≥0.88 confidence. This is architecturally "
            "wrong for two reasons:", BODY),
        criteria_box([
            "The rhythm ML model has only 56% balanced accuracy — it regularly misclassifies sinus tachycardia as VT",
            "The sinus gate uses 10 signal-processing criteria (10+ clinical features) and is far more reliable",
            "With the new lethal detector (Step 3.5) in place, any truly dangerous rhythm is caught by signal "
            "processing BEFORE ML even runs — the veto is now completely redundant",
        ], C_P1),
        sp(2),
        Paragraph("<b>Block to delete in full:</b>", H3),
        Paragraph(
            "_DANGEROUS_RHYTHMS = { 'Atrial Fibrillation', 'AF', 'Atrial Flutter', '3rd Degree AV Block', "
            "'2nd Degree AV Block Type 2', 'Ventricular Fibrillation', 'Ventricular Tachycardia', 'VT' }<br/>"
            "_ml_label_early = ...<br/>"
            "_ml_conf_early  = ...<br/>"
            "if _ml_label_early in _DANGEROUS_RHYTHMS and _ml_conf_early >= 0.88:<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;decision.background_rhythm = _ml_label_early   # ← DELETE THIS ENTIRE BLOCK",
            CODE),
        sp(4),
    ]

    # Idioventricular
    story += [
        Paragraph("<b>Remove Idioventricular Rhythm Detection (lines 261–263)</b>", H2),
        Paragraph(
            "The current orchestrator detects Idioventricular Rhythm from a 3-line heuristic "
            "(P_ratio <0.2, QRS >120ms, HR <60). This has zero corrected training samples and "
            "fires incorrectly on IVCD patients who happen to have a slow rate.", BODY),
        criteria_box([
            "Zero corrected training samples — no validated evidence this criterion is reliable",
            "IVCD (formerly BBB) patients with slow sinus rate satisfy all 3 criteria → false alarm",
            "The lethal detector already handles fast wide-QRS rhythms (VTach) — the slow escape variant is marginal clinically",
        ], C_P1),
        Paragraph("<b>Lines to delete:</b>", H3),
        Paragraph(
            "# in _detect_background_rhythm():<br/>"
            "if p_ratio &lt; 0.2 and qrs_ms &gt; 120:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# ← DELETE<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;return 'Idioventricular Rhythm'&nbsp;&nbsp;&nbsp;# ← DELETE",
            CODE),
        sp(6), PageBreak(),
    ]

    # ── P1: BBB → IVCD ────────────────────────────────────────────────────────
    story.append(section_header("  P1 — MODIFY   data_loader.py + orchestrator  (label rename)", colors.HexColor("#D35400")))
    story += [
        sp(2),
        Paragraph("<b>Rename \"Bundle Branch Block\" → \"Intraventricular Conduction Delay\" (IVCD)</b>", H2),
        Paragraph(
            "\"Bundle Branch Block\" bundles LBBB, RBBB, LAFB, and incomplete blocks under one ambiguous label. "
            "\"Intraventricular Conduction Delay\" (IVCD) is the correct umbrella term used in modern cardiology "
            "device reports. This is a <b>label-only rename</b> — model weights are class-index based, not string "
            "based, so no retraining is needed.", BODY),
        sp(2),
        Paragraph("<b>Changes in models_training/data_loader.py:</b>", H3),
        four_col_table([
            ["351", '"BBB"',  '"Bundle Branch Block"',             '"Intraventricular Conduction Delay"'],
            ["355", '"LBBB"', '"Bundle Branch Block"',             '"Intraventricular Conduction Delay"'],
            ["355", '"RBBB"', '"Bundle Branch Block"',             '"Intraventricular Conduction Delay"'],
            ["381", '"L"',    '"Bundle Branch Block"',             '"Intraventricular Conduction Delay"'],
            ["382", '"R"',    '"Bundle Branch Block"',             '"Intraventricular Conduction Delay"'],
        ], [TW*0.08, TW*0.12, TW*0.38, TW*0.42],
           header=["Line", "Key", "Old value", "New value"]),
        sp(3),
        Paragraph("<b>Changes in decision_engine/rhythm_orchestrator.py:</b>", H3),
        two_col_table([
            ["Confidence threshold key (line 136)",
             '"Bundle Branch Block": 0.80  →  "Intraventricular Conduction Delay": 0.80'],
            ["Any priority or string comparison blocks",
             "Search for \"Bundle Branch Block\" and replace with \"Intraventricular Conduction Delay\""],
        ], (0.30, 0.70)),
        sp(6), PageBreak(),
    ]

    # ── P2: Beat classifier ───────────────────────────────────────────────────
    story.append(section_header("  P2 — CREATE   decision_engine/beat_classifier.py", C_BLUE))
    story += [
        sp(2),
        Paragraph("<b>PVC / PAC Template Correlation Refinement</b>  "
                  "<i>(ported from classify_beats(), lines 672–835)</i>", H2),
        Paragraph(
            "The current system uses the ectopy ML model alone for PVC/PAC classification. "
            "This fails in two common scenarios that BATCH_PROCESS handles correctly:", BODY),
        criteria_box([
            "Aberrant conduction: PAC with a wide QRS (due to Ashman phenomenon or bundle branch refractoriness) "
            "looks identical to PVC morphologically — ML misclassifies as PVC",
            "Low-confidence calls: ML returns 0.97 threshold but near-threshold calls (0.97–0.99) are frequently wrong "
            "when the beat has unusual timing",
        ], C_P2),
        sp(2),
        Paragraph("<b>How It Works — 3-Step Process</b>", H3),
    ]

    step_data = [
        ["Step 1 — Build sinus template",
         "Average all beats labelled 'Normal' by ML in the same 10s window. "
         "Creates a fixed-length waveform (±200ms around R-peak) as the reference morphology. "
         "Requires ≥3 normal beats to proceed."],
        ["Step 2 — Score each premature beat",
         "For every beat with coupling ratio <0.92, compute PVC score and PAC score "
         "using the multi-criteria system below. Template correlation is the primary discriminator."],
        ["Step 3 — Override if justified",
         "If PVC ≥3.0 AND PVC > PAC → override ML label to 'PVC'. "
         "If PAC ≥3.0 AND coupling <0.92 → override ML label to 'PAC'. "
         "Only overrides when ML ectopy confidence is below 0.99 (high-confidence ML is trusted)."],
    ]
    story += [two_col_table(step_data, (0.25, 0.75)), sp(3)]

    # Scoring tables
    pvc_data = [
        ["+2.0", "QRS duration > 120 ms", "Wide complex — ventricular origin"],
        ["+1.5", "T-wave discordant",      "Opposite polarity to QRS — typical PVC"],
        ["+1.5", "P-wave absent",           "No atrial depolarisation preceding beat"],
        ["+1.0", "Coupling interval < 88% RR", "Premature — fires early"],
        ["+1.0", "Compensatory pause > 1.85×RR", "Full compensatory pause — sinus node not reset"],
        ["+2.5", "Template corr < 0.60",   "Aberrant morphology vs sinus template — KEY discriminator"],
        ["−1.0", "Template corr ≥ 0.85 (soft penalty)", "Sinus morphology — disfavours PVC"],
    ]
    pac_data = [
        ["+1.0", "QRS < 110 ms",           "Narrow complex — supraventricular conduction"],
        ["+1.5", "P-wave inverted",         "Retrograde / ectopic atrial origin"],
        ["+0.5", "P-wave present",          "Atrial depolarisation visible"],
        ["+1.0", "Coupling interval < 88% RR", "Premature"],
        ["+1.0", "Non-compensatory pause < 1.90×RR", "Sinus node reset — partial pause"],
        ["+2.5", "Template corr ≥ 0.85",   "Sinus morphology identical — KEY discriminator"],
        ["+0.5", "Template corr soft boost","Near-sinus morphology"],
        ["+1.5", "Ashman phenomenon",       "Long-short RR → bundle branch refractoriness → wide PAC"],
        ["+1.0", "Preceding T deformation", "Retrograde P-wave embedded in preceding T-wave"],
    ]

    cw_score = [TW*0.10, TW*0.38, TW*0.52]
    hdr_score = ["Points", "Criterion", "Clinical Meaning"]

    story += [
        Paragraph("<b>PVC Scoring  (threshold ≥3.0):</b>", H3),
        four_col_table(pvc_data, cw_score, hdr_score),
        sp(3),
        Paragraph("<b>PAC Scoring  (threshold ≥3.0, requires coupling ratio <0.92):</b>", H3),
        four_col_table(pac_data, cw_score, hdr_score),
        sp(3),
        Paragraph(
            "<b>Boundary rule:</b> First and last beats of every 10-second segment are never flagged as "
            "ectopic — they lack full RR context on one side.", NOTE),
        sp(6), PageBreak(),
    ]

    # ── What is NOT changed ───────────────────────────────────────────────────
    story.append(section_header("  KEPT AS-IS — V3 Components That Are Already Better", C_GREEN))
    story += [sp(2)]

    kept_data = [
        ["V3 Ensemble R-peak detector\n(3 detectors, voting)",
         "BATCH_PROCESS uses a single entropy-based detector. V3's 3-detector voting (Pan-Tompkins + "
         "Hilbert + Mexican Hat CWT) eliminates false positives that any single detector would produce. "
         "Voting is the gold standard for R-peak detection in irregular rhythms."],
        ["V3 Adaptive baseline removal\n(with BBB guard)",
         "BATCH_PROCESS uses a fixed 600ms median filter. A 600ms window corrupts wide-QRS beats "
         "(IVCD/BBB patients) by treating the wide QRS as baseline drift. V3's adaptive 3-method "
         "approach with BBB guard prevents this."],
        ["V3 NeuroKit2 DWT delineation",
         "BATCH_PROCESS uses custom prominence-based detection and disables P/T at HR>140. "
         "NeuroKit2's discrete wavelet transform (DWT) is the clinical gold standard. "
         "P/T detection at high rates is handled by the kinetic VTach check instead."],
        ["V3 60-feature extraction",
         "Feeds both ML models and all signal-processing checks. Already the correct interface "
         "between signal processing and decision logic."],
        ["Sinus detection — 10 criteria\n(sinus_detector.py)",
         "Signal-processing first, ML second. This architecture is correct and working. "
         "All 10 criteria (P-wave ratio, RR regularity, QRS width, PR interval, etc.) are "
         "electrophysiology-grounded."],
        ["VT/NSVT from consecutive PVC rules\n(rules.py)",
         "Kept as secondary path — complements the lethal detector. 4+ consecutive PVCs from the "
         "ectopy model still generates NSVT/VT events even if spectral detection does not fire."],
        ["Atrial Flutter spectral detection\n(rules.py)",
         "Working correctly. Flutter waves detected via FFT peak at 4–6 Hz. Not touched."],
        ["Display arbitration suppression\n(rules.py)",
         "Suppression is display-only — all events remain stored in decision.events. "
         "Clinical UI correctly hides PVC events when NSVT already dominates. "
         "Downstream storage/API always has the full event list."],
        ["2nd Degree AV Block Type 1\n(Wenckebach)",
         "Schema has it, WENCKEBACH alias maps to it, but zero training data. "
         "Left inactive — will activate when training data is collected. No change needed."],
    ]
    story += [two_col_table(kept_data, (0.28, 0.72), header=["Component", "Why kept"]), sp(6), PageBreak()]

    # ── Verification ──────────────────────────────────────────────────────────
    story.append(section_header("  VERIFICATION — Regression Tests", C_HEAD_BG))
    story += [sp(2)]

    test_data = [
        ["ADM441825561.json\n(Atrial Flutter ~145 BPM)",
         "Flutter detected, NOT VTach",
         "rr_cv is very low (<0.08) → flutter guard fires at Stage 1 → SURVIVABLE → lethal detector returns None → ML detects flutter"],
        ["ADM1196270205.json\n(IVCD + 1st Degree AVB)",
         "Label shows IVCD (not BBB), 1st AVB — NOT VTach",
         "HR is normal (not >100) → kinetic VTach check fails → ML rhythm detects IVCD + 1st AVB"],
        ["Synthetic VFib segment\n(high-variance 3–6 Hz noise)",
         "Ventricular Fibrillation detected",
         "SPI > 0.75, concentration ≤ 0.35 → VFib at 0.92 confidence"],
        ["Synthetic VTach\n(150 BPM, wide QRS, no P)",
         "Ventricular Tachycardia detected",
         "SPI > 0.75, concentration > 0.35 → VTach at 0.92 confidence, OR kinetic check fires"],
        ["Bigeminy pattern\n(alternating PVC every beat)",
         "NOT labelled VTach",
         "Organization gate: even/odd RR alternation >12%, each group CV <20% → SURVIVABLE"],
        ["Sinus tachycardia at 125 BPM",
         "Sinus Tachycardia — NOT VTach",
         "Sinus gate fires first (P-waves present, regular, narrow QRS) → lethal detector never runs"],
    ]

    cw_test = [TW*0.25, TW*0.25, TW*0.50]
    story += [four_col_table(test_data, cw_test, header=["Test File / Signal", "Expected Output", "Why it should pass"])]

    doc.build(story)
    print(f"PDF saved: {out_path}")


if __name__ == "__main__":
    out_dir = BASE_DIR / "docs"
    out_dir.mkdir(exist_ok=True)
    build_pdf(out_dir / "BATCH_PROCESS_ADOPTION_PLAN.pdf")
