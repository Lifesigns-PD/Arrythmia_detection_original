from flask import Flask, render_template_string, request, jsonify
from lifesigns_engine import analyze_json
import json
import uuid
import threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import hashlib
import glob
import time
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


app       = Flask(__name__)
CACHE_DIR = "batch_cache"
MANIFEST  = os.path.join(CACHE_DIR, "_manifest.json")
SENTINEL  = os.path.join(CACHE_DIR, "_processing.sentinel")
os.makedirs(CACHE_DIR, exist_ok=True)
batch_jobs = {}


# =============================================================================
#  CACHE HELPERS
# =============================================================================

def _write_manifest(job_id, manifest_files):
    """Atomically write manifest.json so other sessions can load results."""
    payload = {
        "version":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "job_id":   job_id,
        "status":   "complete",
        "files":    manifest_files,   # {display_name: {cache_file, flags}}
    }
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, MANIFEST)         # atomic rename — no partial reads
    if os.path.exists(SENTINEL):
        os.remove(SENTINEL)


def _write_sentinel():
    """Mark that a processing job is running (other sessions show a spinner)."""
    with open(SENTINEL, "w") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S"))


def _clear_cache():
    """Delete every result file, manifest, and sentinel in CACHE_DIR."""
    for path in glob.glob(os.path.join(CACHE_DIR, "*.json")) + \
                glob.glob(os.path.join(CACHE_DIR, "*.tmp")):
        try:
            os.remove(path)
        except OSError:
            pass
    for special in (MANIFEST, SENTINEL):
        if os.path.exists(special):
            try:
                os.remove(special)
            except OSError:
                pass


def _read_manifest():
    if not os.path.exists(MANIFEST):
        return None
    try:
        with open(MANIFEST, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# =============================================================================
#  HTML TEMPLATE
# =============================================================================

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Lifesigns BLE | ECG Batch Analysis</title>
    <style>
        *, *::before, *::after { box-sizing: border-box; }
        body {
            background: #121212; color: #E0E0E0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0; padding: 0; display: flex; height: 100vh; overflow: hidden;
        }
        h1, h2, h3 { color: #64B5F6; font-weight: 600; margin-top: 0; }

        /* ── Sidebar ─────────────────────────────────────────── */
        .sidebar {
            flex: 0 0 340px; background: #1a1a1a;
            border-right: 1px solid #333;
            display: flex; flex-direction: column; height: 100%;
            overflow: hidden; z-index: 10;
        }
        .sidebar-header {
            padding: 14px 16px; border-bottom: 1px solid #333;
            background: #151515; flex-shrink: 0;
            display: flex; flex-direction: column; gap: 8px;
        }
        .sidebar-header-top {
            display: flex; align-items: center;
            justify-content: space-between; gap: 8px;
        }
        .sidebar-header h2 { margin: 0; font-size: 1.05em; white-space: nowrap; }

        #new-analysis-btn {
            background: #E65100; color: #fff; border: none;
            padding: 5px 11px; border-radius: 4px; font-size: 0.76em;
            font-weight: 700; cursor: pointer; white-space: nowrap;
            transition: background 0.2s; flex-shrink: 0;
        }
        #new-analysis-btn:hover     { background: #FF6D00; }
        #new-analysis-btn:disabled  { background: #444; color: #666; cursor: not-allowed; }

        #sidebar-status { font-size: 0.75em; color: #888; }
        .cache-tag {
            font-size: 0.71em; color: #546E7A; background: #1E272C;
            padding: 3px 8px; border-radius: 4px; border: 1px solid #263238;
        }
        .cache-tag.live { color: #26C6DA; background: #00251A; border-color: #004D40; }

        .file-list { flex: 1; overflow-y: auto; padding: 8px; }

        .sidebar-item {
            padding: 10px 13px; margin-bottom: 4px; background: #222;
            border-radius: 6px; cursor: pointer; transition: all 0.2s;
            border-left: 4px solid transparent; font-size: 0.82em;
            display: flex; justify-content: space-between; align-items: center; gap: 8px;
        }
        .sidebar-item:hover  { background: #2a2a2a; border-left-color: #64B5F6; }
        .sidebar-item.active { background: #2a2a2a; border-left-color: #64B5F6;
                               color: #64B5F6; font-weight: bold; }
        .sname { flex: 1; min-width: 0; overflow: hidden;
                 text-overflow: ellipsis; white-space: nowrap; }

        /* ── Badges ─────────────────────────────────────────── */
        .badge {
            font-size: 0.72em; padding: 2px 7px; border-radius: 10px;
            white-space: nowrap; font-weight: 600; flex-shrink: 0;
        }
        .b-critical { background: #FF5252; color: #fff; }
        .b-svt      { background: #FF9800; color: #121212; }
        .b-warn     { background: #FFB74D; color: #121212; }
        .b-brady    { background: #26C6DA; color: #121212; }
        .b-noisy    { background: #607D8B; color: #fff; }
        .b-dead     { background: #455A64; color: #fff; }
        .b-ok       { background: #2E7D32; color: #fff; }
        .b-seg      { background: #333; color: #aaa; }

        /* ── Main ────────────────────────────────────────────── */
        .main-content {
            flex: 1; overflow-y: auto; padding: 30px;
            background: #121212; position: relative;
        }
        .container { max-width: 1400px; margin: 0 auto; }

        /* ── Upload overlay ─────────────────────────────────── */
        .upload-overlay {
            display: none; position: absolute; inset: 0;
            background: rgba(18,18,18,0.97); z-index: 50;
            justify-content: center; align-items: center;
        }
        .upload-overlay.show { display: flex; }
        .upload-box {
            background: #1E1E1E; padding: 40px; border-radius: 8px;
            text-align: center; border: 2px dashed #444;
            width: 100%; max-width: 600px;
        }
        .upload-box p { color: #888; margin-bottom: 22px; font-size: 0.92em; }
        .upload-box input[type="file"] {
            color: #E0E0E0; font-size: 1.05em; margin-bottom: 22px; width: 100%;
        }
        .btn-primary {
            background: #64B5F6; color: #121212; border: none;
            padding: 12px 30px; font-size: 1.05em; font-weight: bold;
            border-radius: 4px; cursor: pointer; transition: 0.2s;
        }
        .btn-primary:hover    { background: #42A5F5; }
        .btn-primary:disabled { background: #444; color: #888; cursor: not-allowed; }
        .btn-cancel {
            display: none; background: transparent; border: 1px solid #555;
            color: #aaa; padding: 7px 20px; border-radius: 4px; cursor: pointer;
            font-size: 0.88em; margin: 10px auto 0; transition: 0.2s;
        }
        .btn-cancel:hover { border-color: #aaa; color: #fff; }

        .progress-wrap { display: none; width: 100%; margin-top: 22px; text-align: left; }
        .progress-bar  { width: 100%; background: #333; border-radius: 4px;
                         overflow: hidden; height: 11px; margin-bottom: 7px; }
        .progress-fill { height: 100%; background: #64B5F6; width: 0%;
                         transition: width 0.3s ease; }
        .progress-lbl  { font-size: 0.88em; color: #B0BEC5;
                         display: flex; justify-content: space-between; }

        /* ── Toast ─────────────────────────────────────────── */
        #toast {
            display: none; position: fixed; bottom: 22px; right: 22px;
            background: #1E3A5F; color: #90CAF9;
            border: 1px solid #1565C0; border-radius: 6px;
            padding: 13px 16px; z-index: 9999; font-size: 0.88em;
            box-shadow: 0 4px 20px rgba(0,0,0,0.55); max-width: 360px;
        }
        #toast button {
            margin-left: 10px; background: #1565C0; color: #fff; border: none;
            padding: 3px 10px; border-radius: 4px; cursor: pointer; font-size: 0.82em;
        }
        #toast .t-dismiss { background: #263238; }

        /* ── Info panels ────────────────────────────────────── */
        .info-center {
            display: flex; flex-direction: column; align-items: center;
            justify-content: center; height: 80vh; text-align: center;
        }
        .info-center p { color: #888; max-width: 400px; line-height: 1.7;
                         margin: 0 0 26px; }
        .info-center button {
            background: #64B5F6; color: #121212; border: none;
            padding: 12px 32px; font-size: 1em; font-weight: bold;
            border-radius: 5px; cursor: pointer; transition: 0.2s;
        }
        .info-center button:hover { background: #42A5F5; }

        .remote-banner {
            background: #1A237E; color: #90CAF9; text-align: center;
            padding: 22px; border-radius: 8px; margin: 40px auto;
            max-width: 600px; border: 1px solid #283593;
        }
        .spinner {
            display: inline-block; width: 16px; height: 16px;
            border: 3px solid #283593; border-top-color: #64B5F6;
            border-radius: 50%; animation: spin 0.9s linear infinite;
            vertical-align: middle; margin-right: 8px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* ── Segment cards ─────────────────────────────────── */
        .seg-card { background: #1a1a1a; padding: 20px; border-radius: 8px;
                    margin-bottom: 38px; border: 1px solid #333; }
        .alert-box   { background: #FF5252; color: #fff; padding: 14px;
                       border-radius: 5px; font-weight: bold; margin-bottom: 18px;
                       text-align: center; font-size: 1.15em; }
        .svt-box     { background: #E65100; color: #fff; padding: 11px;
                       border-radius: 5px; font-weight: bold; margin-bottom: 18px;
                       text-align: center; }
        .warn-box    { background: #FFB74D; color: #121212; padding: 11px;
                       border-radius: 5px; font-weight: bold; margin-bottom: 18px;
                       text-align: center; }
        .brady-box   { background: #00838F; color: #fff; padding: 11px;
                       border-radius: 5px; font-weight: bold; margin-bottom: 18px;
                       text-align: center; }
        .sqi-box     { padding: 9px 13px; border-radius: 5px; margin-bottom: 18px;
                       font-size: 0.88em; font-weight: 500; }
        .sqi-warn    { background: #4A3000; color: #FFB74D; border: 1px solid #FF8F00; }
        .sqi-bad     { background: #3E0000; color: #FF8A80; border: 1px solid #FF5252; }
        .dead-box    { background: #37474F; color: #CFD8DC; padding: 18px;
                       border-radius: 5px; text-align: center; margin-bottom: 18px; }
        .hr-box      { background: #1A237E; color: #90CAF9; padding: 10px;
                       border-radius: 5px; font-size: 0.88em; margin-bottom: 18px; }
        .err-box     { background: #7f8c8d; color: #fff; padding: 14px;
                       border-radius: 5px; font-weight: bold; margin-bottom: 18px;
                       text-align: center; }

        /* ── Plot ──────────────────────────────────────────── */
        .plot-vp { background: #121212; border-radius: 8px; margin-bottom: 18px;
                   border: 1px solid #333; overflow: hidden; position: relative;
                   cursor: grab; display: flex; align-items: center; justify-content: center; }
        .plot-vp:active { cursor: grabbing; }
        .z-img { width: 100%; height: auto; transform-origin: center center;
                 transition: transform 0.1s ease-out; user-select: none; }
        .zc { position: absolute; top: 8px; right: 8px; z-index: 10;
              background: rgba(0,0,0,0.6); padding: 4px; border-radius: 4px;
              display: flex; gap: 4px; }
        .zb { background: #333; color: #fff; border: 1px solid #555;
              border-radius: 3px; cursor: pointer; padding: 1px 7px; font-weight: bold; }
        .zb:hover { background: #555; }

        /* ── Metrics ───────────────────────────────────────── */
        .m-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr));
                  gap: 11px; margin-top: 14px; }
        .m-card { background: #1E1E1E; padding: 13px; border-radius: 8px;
                  text-align: center; border-left: 4px solid #64B5F6; }
        .m-val  { font-size: 1.65em; font-weight: bold; margin: 7px 0; color: #fff; }
        .m-lbl  { font-size: 0.77em; color: #B0BEC5; text-transform: uppercase; letter-spacing: 1px; }
        .m-note { font-size: 0.72em; color: #888; margin-top: 3px; }
        .c-abn  { border-left-color: #FF5252; }  .c-abn  .m-val { color: #FF5252; }
        .c-act  { border-left-color: #FF9800; }  .c-act  .m-val { color: #FF9800; }
        .c-bra  { border-left-color: #26C6DA; }  .c-bra  .m-val { color: #26C6DA; }
        .c-ok   { border-left-color: #66BB6A; }  .c-ok   .m-val { color: #66BB6A; }

        .fv { display: none; }
        .fv.on { display: block; animation: fi 0.25s; }
        @keyframes fi { from { opacity: 0; } to { opacity: 1; } }
        #vloader { display: none; text-align: center; padding: 50px;
                   color: #64B5F6; font-weight: bold; font-size: 1.1em; }
        #uerr { display: none; margin-top: 16px; }
    </style>
</head>
<body>

<!-- ════ Sidebar ══════════════════════════════════════════════════════════ -->
<div class="sidebar">
    <div class="sidebar-header">
        <div class="sidebar-header-top">
            <h2>Lifesigns Explorer</h2>
            <button id="new-analysis-btn" onclick="startNewAnalysis()">+ New Analysis</button>
        </div>
        <div id="sb-status">Checking cache…</div>
        <div id="cache-tag" class="cache-tag" style="display:none;"></div>
    </div>
    <div class="file-list" id="file-list"></div>
</div>

<!-- ════ Main ═════════════════════════════════════════════════════════════ -->
<div class="main-content">
    <div id="vloader">Loading from cache…</div>
    <div class="container" id="vc">

        <!-- Remote session is processing -->
        <div id="remote-banner" class="remote-banner" style="display:none;">
            <div><span class="spinner"></span>
                <strong>A processing job is running on another session.</strong></div>
            <div style="margin-top:8px; font-size:0.88em;">
                Results will appear here automatically when complete.
            </div>
        </div>

        <!-- No cache, nothing running -->
        <div id="welcome" class="info-center" style="display:none;">
            <h2 style="margin-bottom:10px;">No Analysis Data</h2>
            <p>No results found in the cache.<br>
               Upload ECG JSON files to run the analysis pipeline.</p>
            <button onclick="showUpload()">Upload &amp; Analyse Files</button>
        </div>

    </div>

    <!-- Upload overlay -->
    <div class="upload-overlay" id="upload-overlay">
        <div class="upload-box">
            <h1>Lifesigns Batch Processor</h1>
            <p>Multi-dimensional SQI gating · Template PAC/PVC correlation ·
               SVT / Flutter / Brady detection</p>
            <input type="file" id="file-input" accept=".json"
                   webkitdirectory directory multiple>
            <button class="btn-primary" id="submit-btn" onclick="doUpload()">
                Initialize Processing Pipeline
            </button>
            <div class="progress-wrap" id="progress-wrap">
                <div class="progress-bar">
                    <div class="progress-fill" id="pfill"></div>
                </div>
                <div class="progress-lbl">
                    <span id="pcnt">0 / 0</span>
                    <span id="ppct">0%</span>
                </div>
            </div>
            <div id="uerr" class="err-box"></div>
            <button class="btn-cancel" id="cancel-btn" onclick="cancelUpload()">Cancel</button>
        </div>
    </div>
</div>

<!-- ════ Toast ════════════════════════════════════════════════════════════ -->
<div id="toast">
    <span id="toast-msg"></span>
    <button onclick="reloadCache()">Reload</button>
    <button class="t-dismiss" onclick="hideToast()">✕</button>
</div>

<script>
// ═══════════════════════════════════════════════════════════════
//  STATE
// ═══════════════════════════════════════════════════════════════
let jobId          = null;   // active upload job (this tab only)
let cacheMode      = false;
let cacheVersion   = null;
let jobStatus      = '';
let pollTimer      = null;
let versionTimer   = null;
let knownFiles     = new Set();
const ts           = new WeakMap();  // zoom transform state

// ═══════════════════════════════════════════════════════════════
//  STARTUP
// ═══════════════════════════════════════════════════════════════
window.addEventListener('DOMContentLoaded', async () => {
    await checkCache();
    versionTimer = setInterval(pollVersion, 6000);
});

async function checkCache() {
    try {
        const r = await fetch('/manifest');
        const d = await r.json();
        if (d.processing) { showRemoteBanner(); return; }
        if (!d.exists)    { showWelcome();      return; }
        cacheVersion = d.version;
        inflateFromManifest(d);
    } catch(_) { showWelcome(); }
}

// ═══════════════════════════════════════════════════════════════
//  INFLATE SIDEBAR FROM MANIFEST
// ═══════════════════════════════════════════════════════════════
function inflateFromManifest(m) {
    cacheMode = true;
    knownFiles.clear();
    document.getElementById('file-list').innerHTML = '';

    const n = Object.keys(m.files || {}).length;
    setSt(`${n} file${n!==1?'s':''} — from cache`);

    const ct = document.getElementById('cache-tag');
    ct.style.display = 'block';
    ct.className     = 'cache-tag';
    ct.textContent   = `Cached: ${(m.version||'').replace('T',' ')}`;

    hide('remote-banner'); hide('welcome');

    for (const [fname, meta] of Object.entries(m.files || {})) {
        knownFiles.add(fname);
        addSidebarItem(fname, meta.flags || {});
    }
}

// ═══════════════════════════════════════════════════════════════
//  REMOTE VERSION POLLING (detect changes from other sessions)
// ═══════════════════════════════════════════════════════════════
async function pollVersion() {
    if (jobId && jobStatus !== 'complete') return;  // we're the active session
    try {
        const r = await fetch('/cache_version');
        const d = await r.json();

        if (d.processing && !jobId) {
            showToast('A new analysis is being processed on another session.');
            return;
        }
        if (d.version && d.version !== cacheVersion && !jobId) {
            cacheVersion = d.version;
            showToast('New analysis results are available.');
        }
    } catch(_) {}
}

// ═══════════════════════════════════════════════════════════════
//  UI HELPERS
// ═══════════════════════════════════════════════════════════════
function setSt(msg) { document.getElementById('sb-status').textContent = msg; }
function show(id)   { document.getElementById(id).style.display = 'block'; }
function hide(id)   { document.getElementById(id).style.display = 'none'; }

function showWelcome()      { show('welcome');       hide('remote-banner'); setSt('No cached data'); hide('cache-tag'); }
function showRemoteBanner() { show('remote-banner'); hide('welcome');       setSt('Processing on another session…'); }

function showUpload() {
    hide('welcome'); hide('remote-banner');
    document.getElementById('upload-overlay').classList.add('show');
    document.getElementById('cancel-btn').style.display = cacheMode ? 'block' : 'none';
}

function cancelUpload() {
    document.getElementById('upload-overlay').classList.remove('show');
    if (!cacheMode) showWelcome();
}

// ═══════════════════════════════════════════════════════════════
//  NEW ANALYSIS
// ═══════════════════════════════════════════════════════════════
async function startNewAnalysis() {
    const btn = document.getElementById('new-analysis-btn');
    btn.disabled    = true;
    btn.textContent = 'Clearing…';
    try { await fetch('/clear_cache', { method: 'POST' }); } catch(_) {}

    // Reset state
    cacheMode = false; cacheVersion = null; jobId = null; jobStatus = '';
    knownFiles.clear();
    document.getElementById('file-list').innerHTML = '';
    hide('remote-banner'); hide('welcome'); hide('cache-tag');
    document.getElementById('progress-wrap').style.display  = 'none';
    document.getElementById('pfill').style.width            = '0%';
    document.getElementById('uerr').style.display           = 'none';
    document.getElementById('submit-btn').disabled          = false;
    document.getElementById('submit-btn').textContent       = 'Initialize Processing Pipeline';
    document.getElementById('file-input').value             = '';
    document.getElementById('vc').innerHTML                 = '';

    // Restore structural elements
    const vc = document.getElementById('vc');
    vc.innerHTML =
        '<div id="remote-banner" class="remote-banner" style="display:none;">'
      + '<div><span class="spinner"></span><strong>A processing job is running on another session.</strong></div>'
      + '<div style="margin-top:8px;font-size:.88em;">Results will appear here automatically when complete.</div>'
      + '</div>'
      + '<div id="welcome" class="info-center" style="display:none;">'
      + '<h2 style="margin-bottom:10px;">No Analysis Data</h2>'
      + '<p>No results found in the cache.<br>Upload ECG JSON files to run the analysis pipeline.</p>'
      + '<button onclick="showUpload()">Upload &amp; Analyse Files</button>'
      + '</div>';

    btn.disabled    = false;
    btn.textContent = '+ New Analysis';
    showUpload();
}

// ═══════════════════════════════════════════════════════════════
//  UPLOAD
// ═══════════════════════════════════════════════════════════════
async function doUpload() {
    const fi   = document.getElementById('file-input');
    const sbtn = document.getElementById('submit-btn');
    const pw   = document.getElementById('progress-wrap');
    const uerr = document.getElementById('uerr');

    if (!fi.files.length) {
        uerr.style.display = 'block';
        uerr.textContent   = 'Please select a folder containing .json files.';
        return;
    }
    const fd = new FormData();
    let n = 0;
    for (const f of fi.files) { if (f.name.endsWith('.json')) { fd.append('ecg_files', f); n++; } }
    if (!n) {
        uerr.style.display = 'block';
        uerr.textContent   = 'No .json files found in the selected folder.';
        return;
    }
    sbtn.disabled = true; sbtn.textContent = 'Transmitting…';
    uerr.style.display = 'none'; pw.style.display = 'block';

    try {
        const r = await fetch('/upload', { method: 'POST', body: fd });
        const d = await r.json();
        if (d.error) throw new Error(d.error);

        jobId = d.job_id; cacheMode = false;
        sbtn.textContent = 'Pipeline Active…';
        document.getElementById('upload-overlay').classList.remove('show');
        document.getElementById('cancel-btn').style.display = 'none';
        hide('welcome'); hide('remote-banner');

        const ct = document.getElementById('cache-tag');
        ct.style.display = 'block'; ct.className = 'cache-tag live';
        ct.textContent   = '● Processing live…';

        pollTimer = setInterval(pollJob, 1200);
    } catch(err) {
        uerr.style.display = 'block'; uerr.textContent = `Error: ${err.message}`;
        sbtn.disabled = false; sbtn.textContent = 'Initialize Processing Pipeline';
    }
}

// ═══════════════════════════════════════════════════════════════
//  JOB POLLING
// ═══════════════════════════════════════════════════════════════
async function pollJob() {
    if (!jobId) return;
    try {
        const r = await fetch(`/status/${jobId}`);
        const d = await r.json();
        if (d.error) return;

        const total = d.total||0, done = d.processed||0;
        const pct = total > 0 ? Math.round(done/total*100) : 0;
        document.getElementById('pfill').style.width  = pct + '%';
        document.getElementById('pcnt').textContent   = `${done} / ${total} Processed`;
        document.getElementById('ppct').textContent   = `${pct}%`;
        jobStatus = d.status;
        setSt(d.status==='complete' ? `Complete (${total} files)` : `Processing… ${done}/${total}`);

        for (const fn of (d.completed_files||[])) {
            if (!knownFiles.has(fn)) { knownFiles.add(fn); addSidebarItem(fn, d.file_flags?.[fn]||{}); }
        }
        for (const fn of (d.errors||[])) {
            if (!knownFiles.has(fn)) { knownFiles.add(fn); addSidebarItem(fn, { is_error: true }); }
        }

        if (d.status === 'complete') {
            clearInterval(pollTimer); pollTimer = null;
            hide('progress-wrap');
            const ct = document.getElementById('cache-tag');
            ct.className = 'cache-tag';
            ct.textContent = `Cached: ${new Date().toLocaleString()}`;
            cacheMode = true;
            try {
                const vr = await fetch('/cache_version');
                const vd = await vr.json();
                cacheVersion = vd.version || cacheVersion;
            } catch(_) {}
        }
    } catch(_) {}
}

// ═══════════════════════════════════════════════════════════════
//  SIDEBAR
// ═══════════════════════════════════════════════════════════════
function addSidebarItem(fname, flags) {
    const list = document.getElementById('file-list');
    const sid  = `sb-${sid_(fname)}`;
    if (document.getElementById(sid)) return;

    const div = document.createElement('div');
    div.className = 'sidebar-item'; div.id = sid;
    div.onclick = () => loadFile(fname);

    const ns = document.createElement('span');
    ns.className = 'sname'; ns.title = fname;
    ns.textContent = fname.replace(/^.*[\\/]/, '');
    div.appendChild(ns);

    const b = document.createElement('span');
    b.className = 'badge ';
    if (flags.is_error)    { b.className+='b-dead';     b.textContent='ERR'; }
    else if (flags.is_lethal)  { b.className+='b-critical'; b.textContent='CRITICAL'; }
    else if (flags.is_svt)     { b.className+='b-svt';      b.textContent='SVT'; }
    else if (flags.is_warn)    { b.className+='b-warn';      b.textContent='WARNING'; }
    else if (flags.is_brady)   { b.className+='b-brady';     b.textContent='BRADY'; }
    else if (flags.is_unusable){ b.className+='b-dead';      b.textContent='NO SIGNAL'; }
    else if (flags.is_noisy)   { b.className+='b-noisy';     b.textContent='NOISY'; }
    else { b.className+='b-seg'; b.textContent=`${flags.segments||'?'} Seg`; }
    div.appendChild(b);
    list.appendChild(div);
}

function sid_(s) { return s.replace(/[^a-zA-Z0-9_-]/g,'_'); }

// ═══════════════════════════════════════════════════════════════
//  FILE VIEW
// ═══════════════════════════════════════════════════════════════
async function loadFile(fname) {
    document.querySelectorAll('.sidebar-item').forEach(el => el.classList.remove('active'));
    const sb = document.getElementById(`sb-${sid_(fname)}`);
    if (sb) sb.classList.add('active');

    const ldr = document.getElementById('vloader');
    const vc  = document.getElementById('vc');
    vc.innerHTML = ''; ldr.style.display = 'block';

    // Prefer cache route; fall back to live route
    const url = cacheMode
        ? `/cache_result/${encodeURIComponent(fname)}`
        : `/result/${jobId}/${encodeURIComponent(fname)}`;

    try {
        const r = await fetch(url);
        const d = await r.json();
        ldr.style.display = 'none';
        if (d.error) { vc.innerHTML = `<div class="err-box">Error: ${d.error}</div>`; return; }

        const v = document.createElement('div');
        v.className = 'fv on';
        v.innerHTML = `<h2 style="margin-bottom:22px;">${fname.replace(/^.*[\\/]/,'')}</h2>`;
        (d.segments||[]).forEach((seg,i) => v.appendChild(renderCard(seg,i)));
        vc.appendChild(v);
    } catch(err) {
        ldr.style.display = 'none';
        vc.innerHTML = `<div class="err-box">Fetch failed: ${err.message}</div>`;
    }
}

// ═══════════════════════════════════════════════════════════════
//  SEGMENT CARD
// ═══════════════════════════════════════════════════════════════
function renderCard(seg, idx) {
    const c = document.createElement('div');
    c.className = 'seg-card';
    c.innerHTML = `<h3>${seg.label||'Segment '+(idx+1)}</h3>`;

    if (seg.unusable) {
        c.innerHTML += `<div class="dead-box">
            <div style="font-size:1.25em;margin-bottom:6px;">⚠ Segment Discarded — No Usable Signal</div>
            <div>SQI: <strong>${seg.sqi_verdict||'UNKNOWN'}</strong></div>
            <div style="color:#90A4AE;font-size:.88em;margin-top:6px;">
                Check electrode placement, skin contact and motion.</div>
        </div>`;
        return c;
    }

    const m = seg.metrics||{};

    if (seg.lethal)
        c.innerHTML += `<div class="alert-box">🚨 CRITICAL: ${seg.lethal}${seg.rate?' &nbsp;|&nbsp; '+(+seg.rate).toFixed(0)+' BPM':''}</div>`;

    if (!seg.lethal && m.svt_flag)
        c.innerHTML += `<div class="svt-box">⚡ SVT${m.flutter_hint?' — Possible Atrial Flutter (2:1)':''}
            &nbsp;|&nbsp; HR ${(m.hr||0).toFixed(1)} BPM &nbsp;|&nbsp; QRS ${(m.qrs_dur||0).toFixed(0)} ms (Narrow)</div>`;

    if (!seg.lethal && !m.svt_flag) {
        if (m.vtach_kinetic_flag) c.innerHTML += `<div class="alert-box">⚠ KINETIC VT — Wide Complex Tachycardia + AV Dissociation</div>`;
        if (m.afib_flag)          c.innerHTML += `<div class="warn-box">⚠ Possible Atrial Fibrillation</div>`;
        else if (m.tachy_flag)    c.innerHTML += `<div class="warn-box">⚠ Tachycardia — HR ${(m.hr||0).toFixed(1)} BPM</div>`;
        if (m.brady_flag)         c.innerHTML += `<div class="brady-box">⚠ Bradycardia — HR ${(m.hr||0).toFixed(1)} BPM</div>`;
        if (m.sinus_arrest_flag)  c.innerHTML += `<div class="warn-box">⚠ Long Pause Detected — Possible Sinus Arrest / AV Block</div>`;
        if (m.torsades_risk)      c.innerHTML += `<div class="warn-box">⚠ Prolonged QTc ${(m.qtc||0).toFixed(0)} ms — Torsades de Pointes Risk</div>`;
    }
    if (m.high_rate_mode)
        c.innerHTML += `<div class="hr-box">ℹ️ <strong>High-Rate Mode:</strong> PQRST delineation suspended (HR &gt; 140 BPM).</div>`;

    if (seg.sqi_verdict && seg.sqi_verdict !== 'GOOD') {
        const pct = ((seg.sqi_score||0)*100).toFixed(0);
        const cls = (seg.sqi_score||0) < 0.25 ? 'sqi-bad' : 'sqi-warn';
        c.innerHTML += `<div class="sqi-box ${cls}">${(seg.sqi_score||0)<0.25?'🔴':'🟡'}
            <strong>Signal Quality Warning:</strong> ${seg.sqi_verdict} (${pct}%) — Check electrode contact.</div>`;
    }

    if (seg.plot) {
        const t_  = Date.now()+idx, iid='i'+t_, vid='v'+t_;
        c.innerHTML += `<div class="plot-vp" id="${vid}">
            <div class="zc">
                <button class="zb" onclick="zi('${iid}')">+</button>
                <button class="zb" onclick="zo('${iid}')">−</button>
                <button class="zb" onclick="zr('${iid}')">↺</button>
            </div>
            <img class="z-img" id="${iid}" src="data:image/png;base64,${seg.plot}" draggable="false">
        </div>`;
        setTimeout(() => initZoom(iid, vid), 50);
    }

    if (!seg.unusable) c.innerHTML += buildMetrics(m, seg.lethal);
    return c;
}

function buildMetrics(m, isLethal) {
    const mc = (l,v,c,n) =>
        `<div class="m-card ${c}"><div class="m-lbl">${l}</div>`+
        `<div class="m-val">${v}</div><div class="m-note">${n||''}</div></div>`;

    const hr   = (m.hr||0).toFixed(1),   qrs  = (m.qrs_dur||0).toFixed(0);
    const pr   = (m.pr_int||0).toFixed(0), qtc = (m.qtc||0).toFixed(1);
    const sdnn = (m.hrv_sdnn||0).toFixed(1), cv = ((m.hrv_cv||0)*100).toFixed(1);
    const sqip = ((m.sqi_score||0)*100).toFixed(0);

    const hrc  = (m.hr>100||m.hr<50)                   ? 'c-abn' : '';
    const qrsc = m.qrs_dur>120                          ? 'c-abn' : '';
    const prc  = (m.pr_int>200||(m.pr_int>0&&m.pr_int<120)) ? 'c-abn' : '';
    const qtcc = m.qtc>500 ? 'c-abn' : (m.qtc>450 ? 'c-act' : '');
    const sqic = (m.sqi_score||0)<0.4 ? 'c-act' : ((m.sqi_score||0)<0.7 ? '' : 'c-ok');

    let f = '';
    if (m.svt_flag)          f += mc('Rhythm','SVT',   'c-act','Narrow complex tachycardia');
    else if (m.afib_flag)    f += mc('Rhythm','AFib?', 'c-act','Irregular, absent P waves');
    else if (m.tachy_flag)   f += mc('Rhythm','Tachy', 'c-act','HR > 100 BPM');
    else if (m.brady_flag)   f += mc('Rhythm','Brady', 'c-bra','HR < 50 BPM');
    else if (!isLethal)      f += mc('Rhythm','NSR',   'c-ok', 'Normal sinus rhythm');
    if (m.flutter_hint)      f += mc('Atrial','Flutter?','c-act','HR ≈ 150, no P waves');
    if (m.sinus_arrest_flag) f += mc('Pause','Detected','c-act','RR gap > 2.5× median');
    if (m.torsades_risk)     f += mc('TdP Risk','HIGH','c-abn',`QTc ${qtc} ms`);
    if (m.high_rate_mode)    f += mc('PQRST','Suspended','c-act','HR > 140 BPM');

    return `<div class="m-grid">
        ${mc('Heart Rate',   hr+' BPM', hrc,  'Normal: 60–100')}
        ${mc('QRS Duration', qrs+' ms', qrsc, 'Normal: 80–120')}
        ${mc('PR Interval',  pr+' ms',  prc,  'Normal: 120–200')}
        ${mc('QTc (Bazett)', qtc+' ms', qtcc, 'M&lt;440 F&lt;460')}
        ${mc('HRV (SDNN)',   sdnn+' ms','',   'CV: '+cv+'%')}
        ${mc('Signal Quality',sqip+'%', sqic, m.sqi_verdict||'GOOD')}
        ${f}
    </div>`;
}

// ═══════════════════════════════════════════════════════════════
//  TOAST
// ═══════════════════════════════════════════════════════════════
function showToast(msg) {
    document.getElementById('toast-msg').textContent = msg;
    document.getElementById('toast').style.display   = 'block';
}
function hideToast()   { document.getElementById('toast').style.display = 'none'; }
async function reloadCache() { hideToast(); await checkCache(); }

// ═══════════════════════════════════════════════════════════════
//  ZOOM
// ═══════════════════════════════════════════════════════════════
function initZoom(iid, vid) {
    const img=document.getElementById(iid), vp=document.getElementById(vid);
    if(!img||!vp) return;
    const s={sc:1,px:0,py:0,dr:false,sx:0,sy:0};
    ts.set(img,s);
    const u=()=>{img.style.transform=`translate(${s.px}px,${s.py}px) scale(${s.sc})`;};
    vp.addEventListener('wheel',e=>{
        e.preventDefault();
        s.sc=Math.min(6,Math.max(1,s.sc*(e.deltaY<0?1.12:0.89)));
        if(s.sc<=1){s.px=0;s.py=0;} u();
    },{passive:false});
    vp.addEventListener('mousedown',e=>{if(s.sc>1){s.dr=true;s.sx=e.clientX-s.px;s.sy=e.clientY-s.py;}});
    window.addEventListener('mousemove',e=>{if(s.dr){s.px=e.clientX-s.sx;s.py=e.clientY-s.sy;u();}});
    window.addEventListener('mouseup',()=>{s.dr=false;});
    vp.addEventListener('mouseleave',()=>{s.dr=false;});
}
function zi(id){const s=ts.get(document.getElementById(id));if(!s)return;s.sc=Math.min(6,s.sc*1.3);document.getElementById(id).style.transform=`translate(${s.px}px,${s.py}px) scale(${s.sc})`;}
function zo(id){const s=ts.get(document.getElementById(id));if(!s)return;s.sc=Math.max(1,s.sc*0.77);if(s.sc<=1){s.px=0;s.py=0;}document.getElementById(id).style.transform=`translate(${s.px}px,${s.py}px) scale(${s.sc})`;}
function zr(id){const s=ts.get(document.getElementById(id));if(!s)return;s.sc=1;s.px=0;s.py=0;document.getElementById(id).style.transform=`translate(0,0) scale(1)`;}
</script>
</body>
</html>
"""


# =============================================================================
#  WORKER
# =============================================================================

def analyze_single_file(payload):
    job_id, display_name, raw_bytes = payload
    try:
        data_json = json.loads(raw_bytes)
        segments  = analyze_json(data_json)

        is_lethal   = any(s.get('lethal')                             for s in segments)
        is_unusable = all(s.get('unusable')                           for s in segments)
        is_noisy    = any(s.get('sqi_verdict') not in ('GOOD', None)  for s in segments)
        is_svt      = any(s.get('metrics', {}).get('svt_flag')        for s in segments)
        is_brady    = any(s.get('metrics', {}).get('brady_flag')      for s in segments)
        is_warn     = any(
            s.get('metrics', {}).get('afib_flag')         or
            s.get('metrics', {}).get('vtach_kinetic_flag') or
            s.get('metrics', {}).get('tachy_flag')         or
            s.get('metrics', {}).get('torsades_risk')
            for s in segments
        )

        safe_name  = hashlib.md5(display_name.encode()).hexdigest() + ".json"
        cache_file = f"{job_id}_{safe_name}"
        cache_path = os.path.join(CACHE_DIR, cache_file)
        with open(cache_path, "w") as f:
            json.dump({"segments": segments}, f, cls=NumpyEncoder)

        flags = {
            "segments":    len(segments),
            "is_lethal":   is_lethal,
            "is_unusable": is_unusable,
            "is_noisy":    is_noisy and not is_lethal,
            "is_svt":      is_svt,
            "is_brady":    is_brady,
            "is_warn":     is_warn and not is_lethal and not is_svt,
        }
        return display_name, {"success": True, "cache_file": cache_file, "flags": flags}
    except Exception as e:
        return display_name, {"success": False, "error": str(e)}


def process_queue(job_id, files_data):
    ctx = multiprocessing.get_context('spawn')
    manifest_files = {}

    with ProcessPoolExecutor(mp_context=ctx) as executor:
        futures = {executor.submit(analyze_single_file, p): p[1] for p in files_data}

        for future in as_completed(futures):
            try:
                display_name, result = future.result()
                if result["success"]:
                    batch_jobs[job_id]['file_flags'][display_name] = result["flags"]
                    batch_jobs[job_id]['results'][display_name]    = \
                        os.path.join(CACHE_DIR, result["cache_file"])
                    batch_jobs[job_id]['completed_files'].append(display_name)
                    manifest_files[display_name] = {
                        "cache_file": result["cache_file"],
                        "flags":      result["flags"],
                    }
                else:
                    batch_jobs[job_id]['results'][display_name] = {"error": result["error"]}
                    batch_jobs[job_id]['errors'].append(display_name)
            except Exception as e:
                dn = futures[future]
                batch_jobs[job_id]['results'][dn] = {"error": str(e)}
                batch_jobs[job_id]['errors'].append(dn)

            batch_jobs[job_id]['processed'] += 1

    batch_jobs[job_id]['status'] = 'complete'
    _write_manifest(job_id, manifest_files)   # persist for other sessions


# =============================================================================
#  FLASK ROUTES
# =============================================================================

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist('ecg_files')
    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400

    _clear_cache()      # wipe previous results
    _write_sentinel()   # signal to other sessions that processing started

    job_id     = str(uuid.uuid4())
    files_data = []
    for file in files:
        if file.filename.endswith('.json'):
            files_data.append((job_id, file.filename, file.read()))

    if not files_data:
        return jsonify({"error": "No valid JSON files found"}), 400

    batch_jobs[job_id] = {
        'total':           len(files_data),
        'processed':       0,
        'results':         {},
        'completed_files': [],
        'errors':          [],
        'file_flags':      {},
        'status':          'running',
    }
    t = threading.Thread(target=process_queue, args=(job_id, files_data))
    t.daemon = True
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    if job_id not in batch_jobs:
        return jsonify({"error": "Job not found"}), 404
    job = batch_jobs[job_id]
    return jsonify({
        'total':           job['total'],
        'processed':       job['processed'],
        'status':          job['status'],
        'completed_files': job['completed_files'],
        'errors':          job['errors'],
        'file_flags':      job['file_flags'],
    })


@app.route("/result/<job_id>/<path:filename>", methods=["GET"])
def get_result(job_id, filename):
    """Live result lookup by job_id (current session that ran the upload)."""
    if job_id not in batch_jobs or filename not in batch_jobs[job_id]['results']:
        return jsonify({"error": "Result not found"}), 404
    r = batch_jobs[job_id]['results'][filename]
    if isinstance(r, dict) and "error" in r:
        return jsonify(r)
    try:
        with open(r, "r") as f:
            return f.read()
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/manifest", methods=["GET"])
def get_manifest():
    """
    Session-agnostic cache state endpoint.
    Returns one of:
      { "processing": true }
      { "exists": false }
      { "exists": true, "version": "...", "files": { ... } }
    """
    if os.path.exists(SENTINEL):
        return jsonify({"processing": True})
    m = _read_manifest()
    if m is None:
        return jsonify({"exists": False})
    m["exists"] = True
    return jsonify(m)


@app.route("/cache_result/<path:filename>", methods=["GET"])
def cache_result(filename):
    """
    Serve a result JSON by display-name, looked up via the manifest.
    Used by any session that didn't run the upload (no job_id required).
    """
    m = _read_manifest()
    if m is None:
        return jsonify({"error": "No manifest found"}), 404
    entry = (m.get("files") or {}).get(filename)
    if entry is None:
        return jsonify({"error": "File not in manifest"}), 404
    path = os.path.join(CACHE_DIR, entry["cache_file"])
    if not os.path.exists(path):
        return jsonify({"error": "Cache file missing from disk"}), 404
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/clear_cache", methods=["POST"])
def clear_cache_route():
    _clear_cache()
    return jsonify({"ok": True})


@app.route("/cache_version", methods=["GET"])
def cache_version():
    """
    Polled by all open sessions every 6 s to detect remote changes.
    Returns the manifest version timestamp (ISO string) or None.
    """
    if os.path.exists(SENTINEL):
        return jsonify({"processing": True,  "version": None})
    m = _read_manifest()
    if m is None:
        return jsonify({"processing": False, "version": None})
    return jsonify({"processing": False, "version": m.get("version")})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)