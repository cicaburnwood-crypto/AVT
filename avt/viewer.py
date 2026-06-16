from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .io import link_or_copy
from .schema import FrameRecord


def _round_point(point: np.ndarray) -> list[float]:
    return [round(float(point[0]), 1), round(float(point[1]), 1)]


def _window_sort_key(path: Path) -> tuple[int, int]:
    meta = json.loads((path / "window.json").read_text(encoding="utf-8"))
    return int(meta["seq_start"]), int(meta["seq_end"])


def _materialize_source_frames(
    source_root: Path,
    frame_records: list[FrameRecord],
    viewer_dir: Path,
    copy_frames: bool,
) -> list[dict[str, Any]]:
    out_dir = viewer_dir / "data" / "source_frames"
    if out_dir.exists() or out_dir.is_symlink():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, Any]] = []
    for rec in frame_records:
        src = rec.resolved_path(source_root)
        dst = out_dir / f"frame_{rec.index:06d}{src.suffix or '.jpg'}"
        link_or_copy(src, dst, copy=copy_frames)
        frames.append(
            {
                "index": rec.index,
                "rel_time_sec": rec.rel_time_sec if rec.rel_time_sec is not None else float(rec.index),
                "source_path": rec.path,
                "image": f"data/source_frames/{dst.name}",
            }
        )
    return frames


def _copy_mask(window_dir: Path, viewer_dir: Path, window_id: str) -> str | None:
    src = window_dir / "path_mask_reference.png"
    if not src.exists():
        return None
    mask_dir = viewer_dir / "data" / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    dst = mask_dir / f"{window_id}.png"
    shutil.copy2(src, dst)
    return f"data/masks/{dst.name}"


def _window_payload(
    window_dir: Path,
    viewer_dir: Path,
    frame_count: int,
    max_points_per_frame: int,
    video_base_url: str,
    tracking_root: Path,
) -> dict[str, Any]:
    meta = json.loads((window_dir / "window.json").read_text(encoding="utf-8"))
    arrays = np.load(window_dir / meta["files"]["tracks"])
    tracks = arrays["tracks_reverse"]
    visibility = arrays["visibility_reverse"].astype(bool)
    queries = arrays["queries"]
    query_records = _query_records(arrays)

    seq_start = int(meta["seq_start"])
    seq_end = int(meta["seq_end"])
    t_len, query_count = tracks.shape[:2]

    crumbs: list[dict[str, Any]] = []
    for row in queries:
        idx = int(row[0])
        record = query_records.get(idx, {})
        source = record.get("source", _query_source_from_row(row))
        if source == "sift_anchor":
            continue
        reverse_time = max(0, min(t_len - 1, int(round(float(row[1])))))
        source_frame = max(0, min(frame_count - 1, seq_end - 1 - reverse_time))
        crumbs.append(
            {
                "id": idx,
                "side": int(row[4]),
                "source": source,
                "response": record.get("response"),
                "size": record.get("size"),
                "reverse_time": reverse_time,
                "source_frame": source_frame,
                "source_xy": _round_point(tracks[reverse_time, idx]),
            }
        )

    frames_out: list[dict[str, Any]] = []
    for frame_idx in range(max(0, seq_start), min(frame_count, seq_end)):
        reverse_t = seq_end - 1 - frame_idx
        if reverse_t < 0 or reverse_t >= t_len:
            continue
        ids = np.flatnonzero(visibility[reverse_t])
        ids = np.array(
            [
                idx
                for idx in ids
                if query_records.get(int(idx), {}).get("source", _query_source_from_row(queries[int(idx)]))
                != "sift_anchor"
            ],
            dtype=np.int64,
        )
        if max_points_per_frame > 0 and ids.size > max_points_per_frame:
            ids = ids[:max_points_per_frame]
        points = [
            [int(idx), round(float(tracks[reverse_t, idx, 0]), 1), round(float(tracks[reverse_t, idx, 1]), 1)]
            for idx in ids
            if np.isfinite(tracks[reverse_t, idx]).all()
        ]
        frames_out.append({"frame": frame_idx, "reverse_time": int(reverse_t), "points": points})

    video_url = None
    reverse_video = meta["files"].get("reverse_video")
    if video_base_url and reverse_video:
        rel = (window_dir / reverse_video).resolve().relative_to(tracking_root.resolve()).as_posix()
        video_url = video_base_url.rstrip("/") + "/" + rel

    return {
        "id": meta["id"],
        "seq_start": seq_start,
        "seq_end": seq_end,
        "t_len": int(t_len),
        "query_count": int(query_count),
        "query_capture": meta.get("query_capture", {}),
        "tracker": meta.get("tracker", {}),
        "mask": {
            "image": _copy_mask(window_dir, viewer_dir, meta["id"]),
            "reference_frame": seq_start,
            "valid_only_on_reference_frame": True,
        },
        "video_url": video_url,
        "crumbs": crumbs,
        "frames": frames_out,
    }


def _query_records(arrays: np.lib.npyio.NpzFile) -> dict[int, dict[str, Any]]:
    if "query_records_json" not in arrays:
        return {}
    raw = arrays["query_records_json"]
    text = str(raw.item() if raw.shape == () else raw.tolist())
    records = json.loads(text)
    return {int(record["id"]): record for record in records}


def _query_source_from_row(row: np.ndarray) -> str:
    if len(row) < 6 or not np.isfinite(row[5]):
        return "avt"
    return {0: "avt", 1: "sift_robot", 2: "sift_anchor"}.get(int(row[5]), "unknown")


def build_payload(
    source_root: Path,
    frame_records: list[FrameRecord],
    tracking_root: Path,
    viewer_dir: Path,
    max_points_per_frame: int = 0,
    copy_frames: bool = False,
    video_base_url: str = "",
) -> dict[str, Any]:
    frames = _materialize_source_frames(source_root, frame_records, viewer_dir, copy_frames)
    window_dirs = sorted(
        [path for path in (tracking_root / "windows").glob("seq_*") if (path / "window.json").exists()],
        key=_window_sort_key,
    )
    if not window_dirs:
        raise FileNotFoundError(f"No AVT windows found under {tracking_root / 'windows'}")

    windows = [
        _window_payload(
            path,
            viewer_dir,
            len(frames),
            max_points_per_frame=max_points_per_frame,
            video_base_url=video_base_url,
            tracking_root=tracking_root,
        )
        for path in window_dirs
    ]
    trackers = sorted({win["tracker"].get("name", "unknown") for win in windows})
    return {
        "metadata": {
            "title": "AVT Inverse Tracking Viewer",
            "source_root": str(source_root.resolve()),
            "tracking_root": str(tracking_root.resolve()),
            "frame_count": len(frames),
            "successful_windows": len(windows),
            "trackers": trackers,
            "point_semantics": (
                "Points are backend-neutral inverse-video predictions. "
                "Click a point to show the source frame where that query was inserted."
            ),
            "mask_semantics": (
                "Cyan overlay is AVT's reference-frame mask built from visible inverse-tracked points. "
                "It is visualization data, not a tracker-specific requirement."
            ),
        },
        "frames": frames,
        "windows": windows,
    }


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AVT Inverse Tracking Viewer</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="topbar">
    <div>
      <h1>AVT Inverse Tracking Viewer</h1>
      <p id="statusText">Loading inverse tracks...</p>
    </div>
    <div class="legend">
      <span><i class="path"></i>Mask</span>
      <span><i class="left"></i>Left queries</span>
      <span><i class="right"></i>Right queries</span>
      <span><i class="selected"></i>Selected</span>
    </div>
  </header>
  <main class="layout">
    <section class="stage">
      <canvas id="sourceCanvas" width="1024" height="576"></canvas>
      <div class="toolbar">
        <button id="prevBtn" type="button">Prev</button>
        <button id="playBtn" type="button">Play</button>
        <button id="pauseBtn" type="button">Pause</button>
        <button id="nextBtn" type="button">Next</button>
        <label>Speed
          <select id="speedSelect">
            <option value="0.25">0.25x</option>
            <option value="0.5">0.5x</option>
            <option value="1" selected>1x</option>
            <option value="2">2x</option>
            <option value="4">4x</option>
          </select>
        </label>
      </div>
      <input id="frameSlider" type="range" min="0" max="0" value="0">
      <div class="toolbar">
        <label>Frame <input id="frameInput" type="number" min="0" value="0"></label>
        <label>Window <select id="windowSelect"></select></label>
        <label class="check"><input id="autoWindow" type="checkbox" checked> Auto window</label>
        <label class="check"><input id="showMask" type="checkbox" checked> Mask</label>
        <label class="check"><input id="showPoints" type="checkbox" checked> Points</label>
      </div>
    </section>
    <aside class="inspector">
      <div class="metrics">
        <div><span>Current frame</span><strong id="frameText">-</strong></div>
        <div><span>Time</span><strong id="timeText">-</strong></div>
        <div><span>Window</span><strong id="windowText">-</strong></div>
        <div><span>Visible points</span><strong id="pointsText">-</strong></div>
      </div>
      <section class="panel">
        <h2>Window Video</h2>
        <video id="windowVideo" controls muted playsinline></video>
        <p id="videoText">No video selected.</p>
      </section>
      <section class="panel">
        <h2>Selected Point</h2>
        <p id="selectedText">Click any rendered point.</p>
        <canvas id="pointCanvas" width="1024" height="576"></canvas>
        <button id="jumpPointBtn" type="button">Jump To Query Source Frame</button>
      </section>
      <section class="panel colorPanel">
        <h2>Colors</h2>
        <p><span class="swatch path"></span>Cyan: reference-frame mask from tracked inverse points.</p>
        <p><span class="swatch left"></span>Blue: left-side query points.</p>
        <p><span class="swatch right"></span>Gold: right-side query points.</p>
        <p><span class="swatch selected"></span>Red: selected point.</p>
      </section>
    </aside>
  </main>
  <script src="app.js?v=2"></script>
</body>
</html>
"""


CSS = """:root {
  color-scheme: light;
  --bg: #f5f7f8;
  --ink: #182024;
  --muted: #647178;
  --line: #d6dee2;
  --panel: #ffffff;
  --left: #3b82f6;
  --right: #f5b84b;
  --selected: #ef4444;
  --path: #20c7e6;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: var(--bg);
}
.topbar {
  min-height: 78px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  padding: 16px 24px;
  border-bottom: 1px solid var(--line);
  background: #fbfcfd;
}
h1, h2 { margin: 0; letter-spacing: 0; }
h1 { font-size: 23px; line-height: 1.15; }
h2 { font-size: 16px; }
p { margin: 6px 0 0; color: var(--muted); font-size: 14px; }
.legend { display: flex; gap: 16px; color: var(--muted); font-size: 14px; white-space: nowrap; }
.legend span { display: inline-flex; align-items: center; gap: 7px; }
.legend i, .swatch { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
.left { background: var(--left); }
.right { background: var(--right); }
.selected { background: var(--selected); }
.path { background: var(--path); }
.layout { display: grid; grid-template-columns: minmax(0, 1fr) 390px; min-height: calc(100vh - 78px); }
.stage { padding: 16px; min-width: 0; }
#sourceCanvas {
  display: block;
  width: 100%;
  max-height: calc(100vh - 210px);
  aspect-ratio: 16 / 9;
  background: #111820;
  border: 1px solid var(--line);
}
.toolbar {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 12px;
  flex-wrap: wrap;
}
button, select, input[type="number"] {
  height: 36px;
  border: 1px solid #bac7cd;
  border-radius: 6px;
  background: #fff;
  color: var(--ink);
  font: inherit;
}
button { padding: 0 13px; font-weight: 650; cursor: pointer; }
button:hover { border-color: #778890; }
select { padding: 0 8px; }
input[type="number"] { width: 86px; padding: 0 8px; }
label { display: inline-flex; align-items: center; gap: 7px; color: var(--muted); font-size: 14px; }
.check { color: var(--ink); }
#frameSlider { width: 100%; margin-top: 13px; accent-color: var(--left); }
.inspector { border-left: 1px solid var(--line); background: #fbfcfd; padding: 16px; min-width: 0; }
.metrics {
  display: grid;
  grid-template-columns: 1fr 1fr;
  border-top: 1px solid var(--line);
  border-left: 1px solid var(--line);
}
.metrics div {
  min-width: 0;
  padding: 11px;
  background: var(--panel);
  border-right: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
}
.metrics span { display: block; color: var(--muted); font-size: 12px; }
.metrics strong { display: block; margin-top: 5px; font-size: 17px; line-height: 1.1; overflow-wrap: anywhere; }
.panel { margin-top: 18px; display: flex; flex-direction: column; gap: 12px; }
.panel video, #pointCanvas {
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #111820;
  border: 1px solid var(--line);
}
.colorPanel { gap: 8px; }
.colorPanel p { margin: 0; line-height: 1.35; }
.swatch { margin-right: 7px; vertical-align: -1px; }
#jumpPointBtn { width: 100%; }
@media (max-width: 980px) {
  .topbar { align-items: flex-start; flex-direction: column; }
  .layout { grid-template-columns: 1fr; }
  .inspector { border-left: 0; border-top: 1px solid var(--line); }
  #sourceCanvas { max-height: none; }
}
"""


JS = """const sourceCanvas = document.getElementById("sourceCanvas");
const sourceCtx = sourceCanvas.getContext("2d");
const pointCanvas = document.getElementById("pointCanvas");
const pointCtx = pointCanvas.getContext("2d");
const statusText = document.getElementById("statusText");
const frameSlider = document.getElementById("frameSlider");
const frameInput = document.getElementById("frameInput");
const speedSelect = document.getElementById("speedSelect");
const windowSelect = document.getElementById("windowSelect");
const autoWindow = document.getElementById("autoWindow");
const showPoints = document.getElementById("showPoints");
const showMask = document.getElementById("showMask");
const frameText = document.getElementById("frameText");
const timeText = document.getElementById("timeText");
const windowText = document.getElementById("windowText");
const pointsText = document.getElementById("pointsText");
const selectedText = document.getElementById("selectedText");
const jumpPointBtn = document.getElementById("jumpPointBtn");
const windowVideo = document.getElementById("windowVideo");
const videoText = document.getElementById("videoText");

let data = null;
let frameIndex = 0;
let selectedWindowId = null;
let selectedPoint = null;
let playing = false;
let timer = null;
let drawPoints = [];
const imageCache = new Map();
const maskCache = new Map();
let pendingMainFrame = null;
let pendingSelectedFrame = null;

function frameImage(index) {
  const frame = data.frames[index];
  if (!frame) return null;
  if (!imageCache.has(frame.image)) {
    const img = new Image();
    img.src = frame.image;
    imageCache.set(frame.image, img);
  }
  return imageCache.get(frame.image);
}

function preloadFrames(center, radius = 8) {
  if (!data) return;
  const start = Math.max(0, center - 2);
  const end = Math.min(data.frames.length - 1, center + radius);
  for (let index = start; index <= end; index += 1) {
    frameImage(index);
  }
}

function maskImage(win) {
  const path = win && win.mask ? win.mask.image : null;
  if (!path) return null;
  if (!maskCache.has(path)) {
    const img = new Image();
    img.src = path;
    maskCache.set(path, img);
  }
  return maskCache.get(path);
}

function windowCovers(win, index) {
  return index >= win.seq_start && index < win.seq_end;
}

function coveringWindows(index) {
  return data.windows.filter((win) => windowCovers(win, index));
}

function frameTracks(win, index) {
  if (!win._frameMap) {
    win._frameMap = new Map(win.frames.map((frame) => [frame.frame, frame]));
    win._crumbMap = new Map(win.crumbs.map((crumb) => [crumb.id, crumb]));
  }
  return win._frameMap.get(index);
}

function activeWindow() {
  const covers = coveringWindows(frameIndex);
  if (!covers.length) return null;
  if (autoWindow.checked) {
    const exact = covers.find((win) => win.mask && win.mask.reference_frame === frameIndex);
    const latest = covers.reduce((best, win) => (win.seq_start > best.seq_start ? win : best), covers[0]);
    selectedWindowId = (exact || latest).id;
    return exact || latest;
  }
  if (!selectedWindowId || !covers.some((win) => win.id === selectedWindowId)) {
    selectedWindowId = covers[0].id;
  }
  return covers.find((win) => win.id === selectedWindowId) || covers[0];
}

function resizeCanvas(canvas, ctx) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const nextWidth = Math.max(1, Math.round(rect.width * dpr));
  const nextHeight = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
    canvas.width = nextWidth;
    canvas.height = nextHeight;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawImageFit(ctx, canvas, img) {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!img || !img.complete || !img.naturalWidth) return null;
  const scale = Math.min(rect.width / img.naturalWidth, rect.height / img.naturalHeight);
  const w = img.naturalWidth * scale;
  const h = img.naturalHeight * scale;
  const x = (rect.width - w) / 2;
  const y = (rect.height - h) / 2;
  ctx.drawImage(img, x, y, w, h);
  return { x, y, w, h, scale };
}

function imageToCanvas(pt, fit) {
  return [fit.x + pt[0] * fit.scale, fit.y + pt[1] * fit.scale];
}

function drawMask(win, fit) {
  if (!win.mask || frameIndex !== win.mask.reference_frame) return;
  const img = maskImage(win);
  if (!img) return;
  if (!img.complete) {
    img.onload = () => window.requestAnimationFrame(drawMain);
    return;
  }
  sourceCtx.drawImage(img, fit.x, fit.y, fit.w, fit.h);
}

function drawMain() {
  if (!data) return;
  const img = frameImage(frameIndex);
  if (!img) return;
  if (!img.complete) {
    pendingMainFrame = frameIndex;
    img.onload = () => {
      if (pendingMainFrame === frameIndex) window.requestAnimationFrame(drawMain);
    };
    return;
  }
  resizeCanvas(sourceCanvas, sourceCtx);
  const fit = drawImageFit(sourceCtx, sourceCanvas, img);
  const win = activeWindow();
  const tracks = win ? frameTracks(win, frameIndex) : null;
  if (fit && win && showMask.checked) drawMask(win, fit);
  drawPoints = [];
  if (fit && win && tracks && showPoints.checked) {
    tracks.points.forEach((point) => {
      const crumb = win._crumbMap.get(point[0]);
      if (!crumb) return;
      const [cx, cy] = imageToCanvas([point[1], point[2]], fit);
      const selected = selectedPoint && selectedPoint.windowId === win.id && selectedPoint.pointId === point[0];
      sourceCtx.fillStyle = selected ? "#ef4444" : (crumb.side < 0 ? "#3b82f6" : "#f5b84b");
      sourceCtx.strokeStyle = "rgba(0,0,0,0.55)";
      sourceCtx.lineWidth = selected ? 2.5 : 1.5;
      sourceCtx.beginPath();
      sourceCtx.arc(cx, cy, selected ? 6 : 4, 0, Math.PI * 2);
      sourceCtx.fill();
      sourceCtx.stroke();
      drawPoints.push({ cx, cy, pointId: point[0], x: point[1], y: point[2], crumb, window: win });
    });
  }
  updateText(win, tracks);
}

function drawSelectedSource() {
  if (!selectedPoint) {
    resizeCanvas(pointCanvas, pointCtx);
    pointCtx.clearRect(0, 0, pointCanvas.width, pointCanvas.height);
    return;
  }
  const img = frameImage(selectedPoint.crumb.source_frame);
  if (!img.complete) {
    pendingSelectedFrame = selectedPoint.crumb.source_frame;
    img.onload = () => {
      if (selectedPoint && pendingSelectedFrame === selectedPoint.crumb.source_frame) {
        window.requestAnimationFrame(drawSelectedSource);
      }
    };
    return;
  }
  resizeCanvas(pointCanvas, pointCtx);
  const fit = drawImageFit(pointCtx, pointCanvas, img);
  if (!fit) return;
  const [x, y] = imageToCanvas(selectedPoint.crumb.source_xy, fit);
  pointCtx.strokeStyle = "#ef4444";
  pointCtx.lineWidth = 3;
  pointCtx.beginPath();
  pointCtx.arc(x, y, 8, 0, Math.PI * 2);
  pointCtx.stroke();
  pointCtx.beginPath();
  pointCtx.moveTo(x - 14, y);
  pointCtx.lineTo(x + 14, y);
  pointCtx.moveTo(x, y - 14);
  pointCtx.lineTo(x, y + 14);
  pointCtx.stroke();
}

function updateText(win, tracks) {
  const frame = data.frames[frameIndex];
  frameText.textContent = `${frameIndex} / ${data.frames.length - 1}`;
  timeText.textContent = `${Number(frame.rel_time_sec || 0).toFixed(2)} s`;
  if (win) {
    const maskRef = win.mask ? win.mask.reference_frame : win.seq_start;
    const maskNote = frameIndex === maskRef ? "mask visible" : `mask reference ${maskRef}`;
    const tracker = win.tracker && win.tracker.name ? win.tracker.name : "tracker";
    windowText.textContent = `${win.id} / ${tracker} / ${maskNote}`;
    if (win.video_url) {
      if (windowVideo.src !== win.video_url) windowVideo.src = win.video_url;
      videoText.textContent = `${win.id} reverse_video.mp4`;
    } else {
      windowVideo.removeAttribute("src");
      videoText.textContent = "No video URL was embedded for this window.";
    }
  } else {
    windowText.textContent = "none";
    windowVideo.removeAttribute("src");
    videoText.textContent = "No active window for this frame.";
  }
  pointsText.textContent = tracks ? tracks.points.length : 0;
  frameSlider.value = frameIndex;
  frameInput.value = frameIndex;
  windowSelect.value = selectedWindowId || "";
  if (!selectedPoint) selectedText.textContent = "Click any rendered point.";
}

function setFrame(index) {
  frameIndex = Math.max(0, Math.min(data.frames.length - 1, Number(index) || 0));
  preloadFrames(frameIndex);
  drawMain();
}

function tick() {
  if (!playing) return;
  if (frameIndex >= data.frames.length - 1) {
    playing = false;
    return;
  }
  frameIndex += 1;
  preloadFrames(frameIndex);
  drawMain();
  timer = window.setTimeout(tick, 1000 / (10 * Number(speedSelect.value)));
}

function play() {
  if (playing) return;
  playing = true;
  tick();
}

function pause() {
  playing = false;
  if (timer) window.clearTimeout(timer);
  timer = null;
}

sourceCanvas.addEventListener("click", (event) => {
  const rect = sourceCanvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  let best = null;
  let bestDist = Infinity;
  drawPoints.forEach((point) => {
    const dist = Math.hypot(point.cx - x, point.cy - y);
    if (dist < bestDist) {
      best = point;
      bestDist = dist;
    }
  });
  if (!best || bestDist > 12) return;
  selectedPoint = { windowId: best.window.id, pointId: best.pointId, crumb: best.crumb };
  selectedText.textContent =
    `Window ${best.window.id}, point ${best.pointId}. ` +
    `Current frame ${frameIndex}, query source frame ${best.crumb.source_frame}, ` +
    `source ${best.crumb.source || "avt"}, side ${best.crumb.side < 0 ? "left" : "right"}.`;
  drawMain();
  drawSelectedSource();
});

document.getElementById("playBtn").addEventListener("click", play);
document.getElementById("pauseBtn").addEventListener("click", pause);
document.getElementById("prevBtn").addEventListener("click", () => { pause(); setFrame(frameIndex - 1); });
document.getElementById("nextBtn").addEventListener("click", () => { pause(); setFrame(frameIndex + 1); });
frameSlider.addEventListener("input", () => { pause(); setFrame(frameSlider.value); });
frameInput.addEventListener("change", () => { pause(); setFrame(frameInput.value); });
showPoints.addEventListener("change", drawMain);
showMask.addEventListener("change", drawMain);
autoWindow.addEventListener("change", drawMain);
windowSelect.addEventListener("change", () => {
  autoWindow.checked = false;
  selectedWindowId = windowSelect.value;
  drawMain();
});
jumpPointBtn.addEventListener("click", () => {
  if (!selectedPoint) return;
  pause();
  setFrame(selectedPoint.crumb.source_frame);
});
window.addEventListener("resize", () => { drawMain(); drawSelectedSource(); });

fetch("data/prediction_tracks.json")
  .then((response) => response.json())
  .then((json) => {
    data = json;
    frameSlider.max = data.frames.length - 1;
    frameInput.max = data.frames.length - 1;
    data.windows.forEach((win) => {
      const option = document.createElement("option");
      option.value = win.id;
      option.textContent = `${win.id} (${win.seq_start}-${win.seq_end})`;
      windowSelect.appendChild(option);
    });
    selectedWindowId = data.windows[0] ? data.windows[0].id : null;
    statusText.textContent =
      `${data.metadata.frame_count} frames, ${data.metadata.successful_windows} windows, ` +
      `trackers: ${data.metadata.trackers.join(", ")}.`;
    setFrame(0);
  });
"""


def write_viewer(viewer_dir: Path, payload: dict[str, Any]) -> None:
    data_dir = viewer_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "prediction_tracks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (viewer_dir / "index.html").write_text(HTML, encoding="utf-8")
    (viewer_dir / "styles.css").write_text(CSS, encoding="utf-8")
    (viewer_dir / "app.js").write_text(JS, encoding="utf-8")


def build_viewer(
    source_root: Path,
    frame_records: list[FrameRecord],
    tracking_root: Path,
    viewer_dir: Path,
    max_points_per_frame: int = 0,
    copy_frames: bool = False,
    video_base_url: str = "",
) -> dict[str, Any]:
    if viewer_dir.exists():
        shutil.rmtree(viewer_dir)
    viewer_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(
        source_root=source_root,
        frame_records=frame_records,
        tracking_root=tracking_root,
        viewer_dir=viewer_dir,
        max_points_per_frame=max_points_per_frame,
        copy_frames=copy_frames,
        video_base_url=video_base_url,
    )
    write_viewer(viewer_dir, payload)
    return payload
