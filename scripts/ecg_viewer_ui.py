#!/usr/bin/env python3
"""
ecg_viewer_ui.py  —  Lightweight Flask ECG diagnostic viewer.

Signal processing only — NO ML model loaded.
Full V3 pipeline: preprocess → R-peaks → delineate → features → lethal detector.

Usage:
    python scripts/ecg_viewer_ui.py
    python scripts/ecg_viewer_ui.py --port 5050

Then open http://localhost:5001 in your browser.
Drop any ECG JSON file onto the page or type the path.
"""

import sys, json, argparse, threading, webbrowser
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

FS          = 125
WIN_SAMPLES = 1250   # 10 s

app = Flask(__name__)

# ── Signal loading (same as diagnostic_viewer) ────────────────────────────────

def load_signal(path: str):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list) and data and isinstance(data[0], dict) and "value" in data[0]:
        packets = sorted(data, key=lambda p: p.get("packetNo", 0))
        samples = []
        for pkt in packets:
            v = pkt["value"]
            if isinstance(v, list) and v and isinstance(v[0], list):
                samples.extend(v[0])
            else:
                samples.extend(v)
        return np.asarray(samples, dtype=np.float32), 500
    if isinstance(data, dict):
        fs = int(data.get("fs") or data.get("sampling_rate") or FS)
        for key in ("ECG_CH_A", "ecg_data", "signal", "ECG", "ecg", "data"):
            if key in data:
                return np.asarray(data[key], dtype=np.float32), fs
        raise ValueError(f"No signal key found. Keys: {list(data.keys())}")
    if isinstance(data, list):
        return np.asarray(data, dtype=np.float32), FS
    raise ValueError("Unrecognised JSON format")


def resample_to_125(sig, orig_fs):
    if orig_fs == FS:
        return sig
    from scipy.signal import resample as sp_resample
    return sp_resample(sig, int(len(sig) * FS / orig_fs)).astype(np.float32)


def get_segment(sig, idx):
    chunk = sig[idx * WIN_SAMPLES: (idx + 1) * WIN_SAMPLES]
    if len(chunk) < WIN_SAMPLES:
        chunk = np.pad(chunk, (0, WIN_SAMPLES - len(chunk)))
    return chunk


# ── V3 pipeline (no ML) ───────────────────────────────────────────────────────

def run_segment(window: np.ndarray) -> dict:
    from signal_processing_v3 import process_ecg_v3
    from decision_engine.lethal_detector import detect_signal_rhythm
    from decision_engine.beat_classifier import classify_beats_sp

    v3 = process_ecg_v3(window, fs=FS, min_quality=0.0)
    label, conf, reason = detect_signal_rhythm(
        v3["signal"], v3["r_peaks"], v3["features"], fs=FS
    )

    f        = v3["features"]
    per_beat = v3["delineation"].get("per_beat", [])

    def _idx(beats, key):
        return [int(b[key]) for b in beats if b.get(key) is not None]

    p_absent = f.get("p_absent_fraction")
    if p_absent is None:
        ppr = f.get("p_wave_present_ratio")
        p_absent = round(1.0 - float(ppr), 3) if ppr is not None else None

    # QRS regions as [onset, offset] pairs
    qrs_regions = []
    for b in per_beat:
        on  = b.get("qrs_onset")
        off = b.get("qrs_offset")
        if on is not None and off is not None:
            qrs_regions.append([int(on), int(off)])

    # Signal-processing PVC / PAC classification (no ML)
    beat_labels = []
    if len(v3["r_peaks"]) >= 3 and per_beat:
        try:
            beat_labels = classify_beats_sp(
                v3["signal"], v3["r_peaks"], per_beat, fs=FS
            )
        except Exception:
            beat_labels = []

    return {
        "signal":      [round(float(x), 4) for x in v3["signal"]],
        "fs":          FS,
        "r_peaks":     v3["r_peaks"].tolist(),
        "p_peaks":     _idx(per_beat, "p_peak"),
        "p_onsets":    _idx(per_beat, "p_onset"),
        "p_offsets":   _idx(per_beat, "p_offset"),
        "q_points":    _idx(per_beat, "q_point"),
        "s_points":    _idx(per_beat, "s_point"),
        "t_peaks":     _idx(per_beat, "t_peak"),
        "t_offsets":   _idx(per_beat, "t_offset"),
        "qrs_regions": qrs_regions,
        "beat_labels": beat_labels,   # [{beat_idx, peak_sample, label, pvc_score, pac_score, corr, coupling_ratio, qrs_dur_ms}]
        "label":       label,
        "conf":        round(conf, 3),
        "reason":      reason,
        "features": {
            "hr":       round(float(f.get("mean_hr_bpm") or 0), 1),
            "qrs_ms":   round(float(f.get("qrs_duration_ms") or 0), 1),
            "pr_ms":    round(float(f.get("pr_interval_ms") or 0), 1),
            "rr_cv":    round(float(f.get("rr_cv") or 0), 3),
            "sdnn_ms":  round(float(f.get("sdnn_ms") or 0), 1),
            "p_absent": round(p_absent, 3) if p_absent is not None else None,
            "sqi":      round(float(v3["sqi"]), 3),
            "n_beats":  len(v3["r_peaks"]),
            "method":   v3["method"],
        },
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/process", methods=["POST"])
def api_process():
    body     = request.get_json(force=True)
    filepath = body.get("file", "").strip()
    max_segs = int(body.get("max_segs", 20))

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    path = Path(filepath)
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.exists():
        return jsonify({"error": f"File not found: {path}"}), 404

    try:
        sig_raw, fs_orig = load_signal(str(path))
        sig = resample_to_125(sig_raw, fs_orig)
        n_total = max(1, len(sig) // WIN_SAMPLES)

        segments = []
        for idx in range(min(n_total, max_segs)):
            window = get_segment(sig, idx)
            seg    = run_segment(window)
            seg["seg_idx"] = idx
            segments.append(seg)

        return jsonify({
            "filename": path.name,
            "total_segments": n_total,
            "shown_segments": len(segments),
            "orig_fs": fs_orig,
            "segments": segments,
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ── HTML + JS (single-file, no external dependencies) ────────────────────────

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ECG Diagnostic Viewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; }

  #header {
    background: #16213e;
    padding: 14px 24px;
    display: flex; align-items: center; gap: 16px;
    border-bottom: 2px solid #0f3460;
    flex-wrap: wrap;
  }
  #header h1 { font-size: 16px; color: #e94560; font-weight: 700; white-space: nowrap; }
  #fileinput {
    flex: 1; min-width: 260px;
    background: #0f3460; border: 1px solid #e94560;
    color: #eee; padding: 6px 10px; border-radius: 4px; font-size: 13px;
  }
  #fileinput::placeholder { color: #888; }
  #maxsegs {
    width: 70px; background: #0f3460; border: 1px solid #555;
    color: #eee; padding: 6px 8px; border-radius: 4px; font-size: 13px;
  }
  button {
    background: #e94560; border: none; color: #fff;
    padding: 7px 18px; border-radius: 4px; font-size: 13px;
    cursor: pointer; font-weight: 600; white-space: nowrap;
  }
  button:hover { background: #c73652; }
  button:disabled { background: #555; cursor: default; }

  #status {
    padding: 6px 24px; font-size: 12px; color: #aaa;
    background: #111827; border-bottom: 1px solid #222;
    min-height: 26px;
  }
  #status.err { color: #ff6b6b; }

  #strips { padding: 12px 20px; overflow-y: auto; }

  .strip-card {
    background: #fff8f0;
    border-radius: 4px;
    margin-bottom: 18px;
    box-shadow: 0 2px 10px #0008;
    overflow: hidden;
  }
  .strip-header {
    display: flex; align-items: center; gap: 10px;
    padding: 5px 12px;
    background: #16213e;
    border-bottom: 1px solid #0f3460;
  }
  .strip-header .seg-label { font-size: 11px; color: #aaa; font-weight: 600; }
  .strip-header .rhythm-label { font-size: 13px; font-weight: 700; }
  .strip-header .feats {
    margin-left: auto; font-size: 10.5px; color: #bbb;
    font-family: monospace; display: flex; gap: 14px; flex-wrap: wrap;
  }
  .strip-header .feats span { white-space: nowrap; }

  canvas.ecg-canvas {
    display: block;
    width: 100%;
    cursor: crosshair;
  }

  #legend {
    display: flex; gap: 18px; flex-wrap: wrap;
    padding: 8px 20px 6px;
    background: #111827;
    border-top: 1px solid #222;
    font-size: 11px;
  }
  .leg-item { display: flex; align-items: center; gap: 5px; }
  .leg-dot {
    width: 10px; height: 10px; border-radius: 50%; display: inline-block;
  }
  .leg-tri { width: 0; height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
  }

  #tooltip {
    position: fixed; pointer-events: none;
    background: #111827ee; border: 1px solid #444;
    color: #eee; padding: 5px 10px; border-radius: 4px;
    font-size: 11px; font-family: monospace;
    display: none; z-index: 100;
  }
</style>
</head>
<body>

<div id="header">
  <h1>⚡ ECG Diagnostic Viewer</h1>
  <input id="fileinput" type="text"
    placeholder="ECG_Data_Extracts/ADM441825561.json  or  data/converted_ecg/MITDB__100_seg_0000.json" />
  <label style="font-size:12px;color:#aaa;white-space:nowrap">
    Max segs <input id="maxsegs" type="number" value="13" min="1" max="60" />
  </label>
  <button id="runbtn" onclick="loadFile()">Analyse</button>
</div>

<div id="status">Enter a file path and click Analyse.</div>
<div id="strips"></div>

<div id="legend">
  <div class="leg-item"><span class="leg-dot" style="background:#dd0000"></span> R-peak (Normal)</div>
  <div class="leg-item"><span class="leg-dot" style="background:#ff8800"></span> R-peak (PVC)</div>
  <div class="leg-item"><span class="leg-dot" style="background:#2299ff"></span> R-peak (PAC)</div>
  <div class="leg-item"><span class="leg-dot" style="background:#007700"></span> P-peak</div>
  <div class="leg-item"><span class="leg-dot" style="background:#005500;border-radius:0;width:2px;height:10px"></span> P on/off</div>
  <div class="leg-item"><span class="leg-dot" style="background:#0055cc"></span> Q-point</div>
  <div class="leg-item"><span class="leg-dot" style="background:#cc5500"></span> S-point</div>
  <div class="leg-item"><span class="leg-dot" style="background:#7700cc"></span> T-peak</div>
  <div class="leg-item"><span style="display:inline-block;width:14px;height:8px;background:#ffe06688;border:1px solid #ccaa00"></span> QRS region</div>
  <div class="leg-item"><span style="display:inline-block;width:20px;height:2px;background:#f4a0a0;vertical-align:middle"></span> Major grid (0.2 s / 0.5 mV)</div>
  <div class="leg-item"><span style="display:inline-block;width:20px;height:1px;background:#fad4d4;vertical-align:middle"></span> Minor grid (0.04 s / 0.1 mV)</div>
</div>

<div id="tooltip"></div>

<script>
// ── Config ────────────────────────────────────────────────────────────────────
const FS          = 125;
const WIN_SAMPLES = 1250;
const PAD_LEFT    = 64;   // px left margin (for mV axis)
const PAD_RIGHT   = 12;
const PAD_TOP     = 6;
const PAD_BOTTOM  = 24;
const STRIP_H_PX  = 180;  // canvas height in CSS px per strip

// ECG paper colours
const PAPER_BG    = "#fff8f0";
const GRID_MAJOR  = "#f09090";
const GRID_MINOR  = "#fadadd";

// ── State ─────────────────────────────────────────────────────────────────────
let allSegments = [];

// ── API call ──────────────────────────────────────────────────────────────────
async function loadFile() {
  const path    = document.getElementById("fileinput").value.trim();
  const maxSegs = parseInt(document.getElementById("maxsegs").value) || 13;
  if (!path) { setStatus("Please enter a file path.", true); return; }

  setStatus("Processing… this may take a few seconds per segment.");
  document.getElementById("runbtn").disabled = true;
  document.getElementById("strips").innerHTML = "";
  allSegments = [];

  try {
    const resp = await fetch("/api/process", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({file: path, max_segs: maxSegs}),
    });
    const data = await resp.json();
    if (data.error) { setStatus("Error: " + data.error, true); return; }

    allSegments = data.segments;
    setStatus(
      `${data.filename}  —  ${data.shown_segments} of ${data.total_segments} segments shown  ` +
      `(original ${data.orig_fs} Hz → resampled to 125 Hz)`
    );
    renderAll(data.segments);
  } catch(e) {
    setStatus("Network error: " + e, true);
  } finally {
    document.getElementById("runbtn").disabled = false;
  }
}

function setStatus(msg, err=false) {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.className = err ? "err" : "";
}

// ── Render all strips ─────────────────────────────────────────────────────────
function renderAll(segments) {
  const container = document.getElementById("strips");
  segments.forEach(seg => {
    const card = buildStripCard(seg);
    container.appendChild(card);
  });
}

function rhythmColor(label) {
  if (!label) return "#22aa44";
  if (label.includes("Fibrillation") || label.includes("Tachycardia")) return "#e94560";
  if (label === "SVT") return "#f5a623";
  return "#22aa44";
}

function buildStripCard(seg) {
  const f     = seg.features;
  const label = seg.label || "No arrhythmia (SP)";
  const color = rhythmColor(seg.label);

  const card = document.createElement("div");
  card.className = "strip-card";

  // Header
  const hdr = document.createElement("div");
  hdr.className = "strip-header";
  const nPVC = (seg.beat_labels || []).filter(b => b.label === "PVC").length;
  const nPAC = (seg.beat_labels || []).filter(b => b.label === "PAC").length;
  const ectopyStr = [
    nPVC ? `<span style="color:#ff8800">PVC <b>${nPVC}</b></span>` : "",
    nPAC ? `<span style="color:#2299ff">PAC <b>${nPAC}</b></span>` : "",
  ].filter(Boolean).join("  ");

  hdr.innerHTML = `
    <span class="seg-label">Seg ${seg.seg_idx}</span>
    <span class="rhythm-label" style="color:${color}">${label}${seg.label ? " (" + Math.round(seg.conf*100) + "%)" : ""}</span>
    <div class="feats">
      <span>HR <b>${f.hr}</b> bpm</span>
      <span>QRS <b>${f.qrs_ms}</b> ms</span>
      <span>PR <b>${f.pr_ms}</b> ms</span>
      <span>rr_cv <b>${f.rr_cv}</b></span>
      ${f.p_absent !== null ? `<span>P-absent <b>${f.p_absent}</b></span>` : ""}
      <span>SQI <b>${f.sqi}</b></span>
      <span>N=<b>${f.n_beats}</b></span>
      ${ectopyStr}
      <span style="color:#888">${f.method}</span>
    </div>`;
  card.appendChild(hdr);

  if (seg.label) {
    const sub = document.createElement("div");
    sub.style.cssText = "padding:2px 12px;background:#16213e;font-size:10px;color:#aaa;font-style:italic;border-bottom:1px solid #0f3460";
    sub.textContent = seg.reason;
    card.appendChild(sub);
  }

  // Canvas
  const canvas = document.createElement("canvas");
  canvas.className = "ecg-canvas";
  canvas.style.height = STRIP_H_PX + "px";
  card.appendChild(canvas);

  // Draw after append (need layout width)
  requestAnimationFrame(() => drawStrip(canvas, seg));
  window.addEventListener("resize", () => drawStrip(canvas, seg));

  // Tooltip on hover
  canvas.addEventListener("mousemove", e => onHover(e, canvas, seg));
  canvas.addEventListener("mouseleave", () => {
    document.getElementById("tooltip").style.display = "none";
  });

  return card;
}

// ── Canvas drawing ────────────────────────────────────────────────────────────
function drawStrip(canvas, seg) {
  // HiDPI
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.offsetWidth;
  const cssH = STRIP_H_PX;
  canvas.width  = cssW * dpr;
  canvas.height = cssH * dpr;
  canvas.style.height = cssH + "px";

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const W = cssW;
  const H = cssH;
  const plotW = W - PAD_LEFT - PAD_RIGHT;
  const plotH = H - PAD_TOP  - PAD_BOTTOM;

  const sig = seg.signal;
  const fs  = seg.fs;
  const n   = sig.length;

  // Y range
  const yMin = -3.0;
  const yMax =  3.0;

  // Coordinate helpers
  function tx(sampleIdx) {
    return PAD_LEFT + (sampleIdx / (n - 1)) * plotW;
  }
  function ty(mv) {
    return PAD_TOP + plotH - ((mv - yMin) / (yMax - yMin)) * plotH;
  }

  // ── Background ───────────────────────────────────────────────────────────
  ctx.fillStyle = PAPER_BG;
  ctx.fillRect(0, 0, W, H);

  // ── ECG paper grid ────────────────────────────────────────────────────────
  // Minor: 0.04 s × 0.1 mV
  const smallT  = 0.04;
  const smallMV = 0.1;
  const largT   = 0.20;
  const largMV  = 0.5;

  ctx.lineWidth = 0.4;
  ctx.strokeStyle = GRID_MINOR;
  // vertical minor
  for (let t = 0; t <= 10.0 + 1e-6; t += smallT) {
    const x = PAD_LEFT + (t / 10.0) * plotW;
    ctx.beginPath(); ctx.moveTo(x, PAD_TOP); ctx.lineTo(x, PAD_TOP + plotH); ctx.stroke();
  }
  // horizontal minor
  for (let mv = Math.ceil(yMin / smallMV) * smallMV; mv <= yMax + 1e-6; mv += smallMV) {
    const y = ty(mv);
    ctx.beginPath(); ctx.moveTo(PAD_LEFT, y); ctx.lineTo(PAD_LEFT + plotW, y); ctx.stroke();
  }

  ctx.lineWidth = 0.8;
  ctx.strokeStyle = GRID_MAJOR;
  // vertical major
  for (let t = 0; t <= 10.0 + 1e-6; t += largT) {
    const x = PAD_LEFT + (t / 10.0) * plotW;
    ctx.beginPath(); ctx.moveTo(x, PAD_TOP); ctx.lineTo(x, PAD_TOP + plotH); ctx.stroke();
  }
  // horizontal major
  for (let mv = Math.ceil(yMin / largMV) * largMV; mv <= yMax + 1e-6; mv += largMV) {
    const y = ty(mv);
    ctx.beginPath(); ctx.moveTo(PAD_LEFT, y); ctx.lineTo(PAD_LEFT + plotW, y); ctx.stroke();
  }

  // ── Y-axis labels ─────────────────────────────────────────────────────────
  ctx.fillStyle = "#888";
  ctx.font = "9px monospace";
  ctx.textAlign = "right";
  for (let mv = Math.ceil(yMin / largMV) * largMV; mv <= yMax + 1e-6; mv += largMV) {
    ctx.fillText(mv.toFixed(1), PAD_LEFT - 4, ty(mv) + 3);
  }
  // X-axis labels
  ctx.textAlign = "center";
  for (let t = 0; t <= 10.0; t += 1.0) {
    const x = PAD_LEFT + (t / 10.0) * plotW;
    ctx.fillText(t + "s", x, PAD_TOP + plotH + 14);
  }

  // ── QRS shading ───────────────────────────────────────────────────────────
  ctx.fillStyle = "rgba(255,224,50,0.18)";
  for (const [on, off] of (seg.qrs_regions || [])) {
    const x1 = tx(on);
    const x2 = tx(off);
    ctx.fillRect(x1, PAD_TOP, x2 - x1, plotH);
  }

  // ── ECG trace ─────────────────────────────────────────────────────────────
  ctx.beginPath();
  ctx.strokeStyle = "#111";
  ctx.lineWidth   = 1.1;
  ctx.lineJoin    = "round";
  ctx.moveTo(tx(0), ty(sig[0]));
  for (let i = 1; i < n; i++) {
    ctx.lineTo(tx(i), ty(sig[i]));
  }
  ctx.stroke();

  // ── Marker helpers ────────────────────────────────────────────────────────
  function dot(idx, color, r=3) {
    if (idx < 0 || idx >= n) return;
    ctx.beginPath();
    ctx.arc(tx(idx), ty(sig[idx]), r, 0, Math.PI*2);
    ctx.fillStyle = color;
    ctx.fill();
  }
  function tick(idx, color) {
    if (idx < 0 || idx >= n) return;
    const x = tx(idx);
    const yc = ty(sig[idx]);
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    ctx.moveTo(x, yc - 5);
    ctx.lineTo(x, yc + 5);
    ctx.stroke();
  }
  function triangle(idx, color, up=true) {
    if (idx < 0 || idx >= n) return;
    const x  = tx(idx);
    const yc = ty(sig[idx]);
    const sz = 5;
    ctx.beginPath();
    if (up) {
      ctx.moveTo(x, yc - sz);
      ctx.lineTo(x - sz, yc + sz);
      ctx.lineTo(x + sz, yc + sz);
    } else {
      ctx.moveTo(x, yc + sz);
      ctx.lineTo(x - sz, yc - sz);
      ctx.lineTo(x + sz, yc - sz);
    }
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  }
  function diamond(idx, color) {
    if (idx < 0 || idx >= n) return;
    const x  = tx(idx);
    const yc = ty(sig[idx]);
    const sz = 4;
    ctx.beginPath();
    ctx.moveTo(x,       yc - sz);
    ctx.lineTo(x + sz,  yc);
    ctx.lineTo(x,       yc + sz);
    ctx.lineTo(x - sz,  yc);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  }

  // ── Draw markers ──────────────────────────────────────────────────────────
  // P wave
  for (const i of seg.p_onsets)  tick(i, "#006600");
  for (const i of seg.p_offsets) tick(i, "#006600");
  for (const i of seg.p_peaks)   dot(i, "#00aa00", 3.5);

  // Q and S
  for (const i of seg.q_points)  triangle(i, "#0055cc", false);
  for (const i of seg.s_points)  triangle(i, "#cc5500", false);

  // T wave
  for (const i of seg.t_offsets) tick(i, "#7700cc");
  for (const i of seg.t_peaks)   diamond(i, "#9933dd");

  // R-peaks — colour by beat label: PVC=orange, PAC=blue, Normal=red
  const beatLabelMap = {};  // peak_sample -> label
  for (const bl of (seg.beat_labels || [])) {
    beatLabelMap[bl.peak_sample] = bl.label;
  }

  for (const i of seg.r_peaks) {
    const lbl = beatLabelMap[i] || "Normal";
    const col = lbl === "PVC" ? "#ff8800"
              : lbl === "PAC" ? "#2299ff"
              : "#dd0000";
    triangle(i, col, true);
  }

  // Beat labels (PVC / PAC text above the R-peak marker)
  ctx.font      = "bold 9px sans-serif";
  ctx.textAlign = "center";
  for (const bl of (seg.beat_labels || [])) {
    if (bl.label === "Normal") continue;
    const i   = bl.peak_sample;
    if (i < 0 || i >= n) continue;
    const x   = tx(i);
    const y   = ty(sig[i]) - 14;
    const col = bl.label === "PVC" ? "#ff6600" : "#0077ee";
    ctx.fillStyle = col;
    ctx.fillText(bl.label, x, y);
    // Small score hint
    ctx.font      = "7px monospace";
    ctx.fillStyle = "#888";
    const scoreStr = bl.label === "PVC"
      ? `p${bl.pvc_score}`
      : `p${bl.pac_score}`;
    ctx.fillText(scoreStr, x, y - 8);
    ctx.font      = "bold 9px sans-serif";
  }
}

// ── Tooltip on hover ──────────────────────────────────────────────────────────
function onHover(e, canvas, seg) {
  const rect  = canvas.getBoundingClientRect();
  const cssX  = e.clientX - rect.left;
  const cssW  = canvas.offsetWidth;
  const plotW = cssW - PAD_LEFT - PAD_RIGHT;

  if (cssX < PAD_LEFT || cssX > PAD_LEFT + plotW) {
    document.getElementById("tooltip").style.display = "none";
    return;
  }

  const tFrac     = (cssX - PAD_LEFT) / plotW;
  const sampleIdx = Math.round(tFrac * (seg.signal.length - 1));
  const timeSec   = (sampleIdx / FS).toFixed(3);
  const mv        = seg.signal[sampleIdx].toFixed(4);

  // Nearest R-peak + beat label
  let nearR = null, minD = 9999, nearBL = null;
  for (const r of seg.r_peaks) {
    const d = Math.abs(r - sampleIdx);
    if (d < minD) { minD = d; nearR = r; }
  }
  for (const bl of (seg.beat_labels || [])) {
    const d = Math.abs(bl.peak_sample - sampleIdx);
    if (d < 30) { nearBL = bl; break; }
  }

  let rInfo = "";
  if (nearR !== null && minD < 30) {
    rInfo = `  |  R@${(nearR/FS).toFixed(3)}s (Δ${((sampleIdx-nearR)/FS*1000).toFixed(0)}ms)`;
  }
  let beatInfo = "";
  if (nearBL) {
    beatInfo = `  |  ${nearBL.label}  PVC:${nearBL.pvc_score} PAC:${nearBL.pac_score}  corr:${nearBL.corr ?? "—"}  coup:${nearBL.coupling_ratio}`;
  }

  const tip = document.getElementById("tooltip");
  tip.textContent = `t=${timeSec}s  amp=${mv}mV${rInfo}${beatInfo}`;
  tip.style.display = "block";
  tip.style.left = (e.clientX + 14) + "px";
  tip.style.top  = (e.clientY - 10) + "px";
}

// ── Allow Enter key ───────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("fileinput").addEventListener("keydown", e => {
    if (e.key === "Enter") loadFile();
  });
});
</script>
</body>
</html>
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    print(f"\n  ECG Diagnostic Viewer -> {url}\n")

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
