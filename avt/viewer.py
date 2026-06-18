from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .io import link_or_copy
from .reliability import detect_stationary_sift_frames, frame_reliability, reliability_metadata
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


def _window_tracks_from_cache(
    window_dir: Path,
    meta: dict[str, Any],
    arrays: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    cache_reference = meta.get("cache_reference") or {}
    cache_path_value = cache_reference.get("path") or cache_reference.get("metadata")
    if not cache_path_value:
        raise KeyError("ID-only window artifact is missing cache_reference.path")

    from .tracking.cotracker_cache import CoTrackerCache

    cache_path = Path(cache_path_value)
    if not cache_path.exists() and not cache_path.is_absolute():
        cache_path = (window_dir / cache_path).resolve()
    cache = CoTrackerCache(cache_path)
    seq_start = int(meta["seq_start"])
    seq_end = int(meta["seq_end"])
    t_len = seq_end - seq_start
    cache_indices = np.arange(
        cache.reverse_index_for_source_frame(seq_end - 1),
        cache.reverse_index_for_source_frame(seq_end - 1) + t_len,
        dtype=np.int64,
    )
    point_ids = arrays["cache_point_ids"].astype(np.int64)
    point_indices = (
        arrays["cache_point_indices"].astype(np.int64)
        if "cache_point_indices" in arrays
        else point_ids
    )
    tracks = np.full((t_len, len(point_indices), 2), np.nan, dtype=np.float32)
    visibility = np.zeros((t_len, len(point_indices)), dtype=bool)
    confidence = (
        np.zeros((t_len, len(point_indices)), dtype=np.float32)
        if cache.confidence is not None
        else None
    )
    valid = np.flatnonzero(point_indices >= 0)
    if valid.size:
        cache_point_indices = point_indices[valid]
        tracks[:, valid] = np.asarray(
            cache.tracks[np.ix_(cache_indices, cache_point_indices)],
            dtype=np.float32,
        )
        visibility[:, valid] = np.asarray(
            cache.visibility[np.ix_(cache_indices, cache_point_indices)],
            dtype=bool,
        )
        if confidence is not None and cache.confidence is not None:
            confidence[:, valid] = np.asarray(
                cache.confidence[np.ix_(cache_indices, cache_point_indices)],
                dtype=np.float32,
            )

    if "queries" in arrays:
        for query_index, row in enumerate(arrays["queries"]):
            reverse_time = max(0, min(t_len, int(round(float(row[1])))))
            if reverse_time > 0:
                visibility[:reverse_time, query_index] = False
                tracks[:reverse_time, query_index] = np.nan
                if confidence is not None:
                    confidence[:reverse_time, query_index] = 0.0
    return tracks, visibility, confidence


def _window_tracks(
    window_dir: Path,
    meta: dict[str, Any],
    arrays: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if "tracks_reverse" in arrays:
        tracks = arrays["tracks_reverse"]
        visibility = arrays["visibility_reverse"].astype(bool)
        confidence = arrays["confidence_reverse"].astype(np.float32) if "confidence_reverse" in arrays else None
        if confidence is not None and confidence.shape != visibility.shape:
            raise ValueError(
                f"confidence_reverse shape {confidence.shape} does not match visibility shape {visibility.shape}"
            )
        return tracks, visibility, confidence
    if "cache_point_ids" in arrays:
        return _window_tracks_from_cache(window_dir, meta, arrays)
    raise KeyError("Window tracks.npz must contain tracks_reverse or cache_point_ids")


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
    tracks, visibility, confidence = _window_tracks(window_dir, meta, arrays)
    queries = arrays["queries"]
    query_records = _query_records(arrays)

    seq_start = int(meta["seq_start"])
    seq_end = int(meta["seq_end"])
    t_len, query_count = tracks.shape[:2]
    query_sources = {
        int(row[0]): str(query_records.get(int(row[0]), {}).get("source") or _query_source_from_row(row))
        for row in queries
    }
    sift_point_ids = [idx for idx, source in query_sources.items() if source.startswith("sift_")]
    frame_reasons = detect_stationary_sift_frames(
        tracks,
        visibility,
        seq_start=seq_start,
        seq_end=seq_end,
        sift_point_ids=sift_point_ids,
    )
    unreliable_frame_indices = sorted(frame_reasons)

    crumbs: list[dict[str, Any]] = []
    crumb_count = 0
    for row in queries:
        if max_points_per_frame > 0 and crumb_count >= max_points_per_frame:
            break
        idx = int(row[0])
        record = query_records.get(idx, {})
        source = query_sources.get(idx, _query_source_from_row(row))
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
        crumb_count += 1

    frames_out: list[dict[str, Any]] = []
    for frame_idx in range(max(0, seq_start), min(frame_count, seq_end)):
        reverse_t = seq_end - 1 - frame_idx
        if reverse_t < 0 or reverse_t >= t_len:
            continue
        ids = np.flatnonzero(visibility[reverse_t])
        if max_points_per_frame > 0 and ids.size > max_points_per_frame:
            ids = ids[:max_points_per_frame]
        points = []
        for idx in ids:
            if not np.isfinite(tracks[reverse_t, idx]).all():
                continue
            point = [
                int(idx),
                round(float(tracks[reverse_t, idx, 0]), 1),
                round(float(tracks[reverse_t, idx, 1]), 1),
            ]
            if confidence is not None:
                value = float(confidence[reverse_t, idx])
                point.append(round(max(0.0, min(1.0, value)), 4) if np.isfinite(value) else None)
            points.append(point)
        frames_out.append(
            {
                "frame": frame_idx,
                "reverse_time": int(reverse_t),
                "points": points,
                "reliability": frame_reliability(
                    frame_idx,
                    points,
                    unreliable_frame_indices=unreliable_frame_indices,
                    frame_reasons=frame_reasons,
                ),
            }
        )

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
        "point_columns": ["id", "x", "y"] + (["confidence"] if confidence is not None else []),
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

    window_payload_dir = viewer_dir / "data" / "windows"
    window_payload_dir.mkdir(parents=True, exist_ok=True)
    windows = []
    for path in window_dirs:
        win = _window_payload(
            path,
            viewer_dir,
            len(frames),
            max_points_per_frame=max_points_per_frame,
            video_base_url=video_base_url,
            tracking_root=tracking_root,
        )
        segment_path = window_payload_dir / f"{win['id']}.json"
        segment_rel = f"data/windows/{segment_path.name}"
        segment_path.write_text(json.dumps(win, separators=(",", ":")), encoding="utf-8")
        windows.append(
            {
                "id": win["id"],
                "seq_start": win["seq_start"],
                "seq_end": win["seq_end"],
                "t_len": win["t_len"],
                "query_count": win["query_count"],
                "query_capture": win["query_capture"],
                "tracker": win["tracker"],
                "mask": win["mask"],
                "video_url": win["video_url"],
                "point_columns": win["point_columns"],
                "segment": segment_rel,
            }
        )
    trackers = sorted({win["tracker"].get("name", "unknown") for win in windows})
    return {
        "metadata": {
            "title": "AVT Inverse Tracking Viewer",
            "source_root": str(source_root.resolve()),
            "tracking_root": str(tracking_root.resolve()),
            "frame_count": len(frames),
            "successful_windows": len(windows),
            "segmented": True,
            "segment_note": "Window point tracks are loaded on demand from data/windows/*.json.",
            "reliability_filter": reliability_metadata(),
            "point_columns": ["id", "x", "y", "confidence"],
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
      <span><i class="path"></i>Path</span>
      <span><i class="left"></i>Left queries</span>
      <span><i class="right"></i>Right queries</span>
      <span><i class="anchor"></i>Anchor queries</span>
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
        <label class="reliability-control">Frames <input id="reliabilitySpanInput" type="number" min="2" max="80" step="1" value="10"></label>
        <label class="reliability-control">Block <input id="reliabilityBlockInput" type="number" min="0.5" max="100" step="0.5" value="6"></label>
        <label class="reliability-control">Points <input id="reliabilityPointsInput" type="number" min="1" max="100" step="1" value="3"></label>
        <label class="reliability-control">Segment <input id="reliabilitySegmentInput" type="number" min="1" max="1000" step="1" value="40"></label>
      </div>
      <input id="frameSlider" type="range" min="0" max="0" value="0">
      <div class="toolbar">
        <label>Frame <input id="frameInput" type="number" min="0" value="0"></label>
        <label>Window <select id="windowSelect"></select></label>
        <label class="check"><input id="autoWindow" type="checkbox" checked> All active windows</label>
        <label class="check"><input id="showMask" type="checkbox" checked> Path</label>
        <label class="check"><input id="showPoints" type="checkbox" checked> Points</label>
      </div>
    </section>
    <aside class="inspector">
      <div class="metrics">
        <div><span>Current frame</span><strong id="frameText">-</strong></div>
        <div><span>Time</span><strong id="timeText">-</strong></div>
        <div><span>Window</span><strong id="windowText">-</strong></div>
        <div><span>Visible points</span><strong id="pointsText">-</strong></div>
        <div><span>Confidence</span><strong id="confidenceText">-</strong></div>
        <div class="reliability-metric"><span>Reliability</span><strong id="reliabilityText">-</strong></div>
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
        <p><span class="swatch path"></span>Cyan: filtered and smoothed path corridor from tracked robot points.</p>
        <p><span class="swatch left"></span>Blue: left-side query points.</p>
        <p><span class="swatch right"></span>Gold: right-side query points.</p>
        <p><span class="swatch anchor"></span>Green: anchor/support query points.</p>
        <p><span class="swatch selected"></span>Red: selected point.</p>
      </section>
    </aside>
  </main>
  <script src="app.js?v=12"></script>
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
  --anchor: #10b981;
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
.anchor { background: var(--anchor); }
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
.reliability-control input[type="number"] { width: 72px; }
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
.metrics .reliability-metric { grid-column: 1 / -1; }
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
const confidenceText = document.getElementById("confidenceText");
const reliabilityText = document.getElementById("reliabilityText");
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

function pointColor(crumb) {
  if (crumb.source === "sift_anchor") return "#10b981";
  return crumb.side < 0 ? "#3b82f6" : "#f5b84b";
}

function pointConfidence(point) {
  if (!point || point.length < 4) return null;
  const confidence = Number(point[3]);
  if (!Number.isFinite(confidence)) return null;
  return Math.max(0, Math.min(1, confidence));
}

function formatConfidence(confidence) {
  return confidence === null ? "unknown" : `${Math.round(confidence * 100)}%`;
}

function pointAlpha(confidence) {
  return confidence === null ? 1 : 0.25 + confidence * 0.75;
}

function pointRadius(confidence, selected) {
  if (selected) return 6;
  return confidence === null ? 4 : 2.5 + confidence * 3;
}

function confidenceSummary(trackSets, loading = false) {
  if (loading) return "loading";
  const values = [];
  trackSets.forEach(({ tracks }) => {
    if (!tracks) return;
    tracks.points.forEach((point) => {
      const confidence = pointConfidence(point);
      if (confidence !== null) values.push(confidence);
    });
  });
  if (!values.length) return "unknown";
  const avg = values.reduce((sum, value) => sum + value, 0) / values.length;
  return `avg ${formatConfidence(avg)} / min ${formatConfidence(Math.min(...values))} / max ${formatConfidence(Math.max(...values))}`;
}

function median(values) {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function quantile(values, q) {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  return sorted[base + 1] === undefined ? sorted[base] : sorted[base] + rest * (sorted[base + 1] - sorted[base]);
}

function mad(values, center) {
  return median(values.map((value) => Math.abs(value - center)));
}

function broadFilter(points, fit, useMad = true) {
  const margin = 36;
  const inside = points.filter((point) =>
    point.cx >= fit.x - margin &&
    point.cx <= fit.x + fit.w + margin &&
    point.cy >= fit.y - margin &&
    point.cy <= fit.y + fit.h + margin
  );
  if (inside.length < 12) return inside;
  const xs = inside.map((point) => point.cx);
  const ys = inside.map((point) => point.cy);
  const xLow = quantile(xs, 0.02);
  const xHigh = quantile(xs, 0.98);
  const yLow = quantile(ys, 0.02);
  const yHigh = quantile(ys, 0.98);
  const clipped = inside.filter((point) =>
    point.cx >= xLow - 12 &&
    point.cx <= xHigh + 12 &&
    point.cy >= yLow - 12 &&
    point.cy <= yHigh + 12
  );
  if (!useMad || clipped.length < 12) return clipped;
  const cx = median(clipped.map((point) => point.cx));
  const cy = median(clipped.map((point) => point.cy));
  const xLimit = Math.max(42, 5.0 * 1.4826 * mad(clipped.map((point) => point.cx), cx));
  const yLimit = Math.max(42, 5.0 * 1.4826 * mad(clipped.map((point) => point.cy), cy));
  const filtered = clipped.filter((point) => Math.abs(point.cx - cx) <= xLimit && Math.abs(point.cy - cy) <= yLimit);
  return filtered.length >= Math.max(6, clipped.length * 0.45) ? filtered : clipped;
}

function robustBoundary(points, bandHeight = 30) {
  const bins = new Map();
  points.forEach((point) => {
    const key = Math.round(point.cy / bandHeight);
    if (!bins.has(key)) bins.set(key, []);
    bins.get(key).push(point);
  });
  return Array.from(bins.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([, group]) => {
      const xs = group.map((point) => point.cx);
      const xMed = median(xs);
      const spread = 1.4826 * mad(xs, xMed);
      const limit = Math.max(10, Math.min(70, 3.0 * spread || 10));
      const filtered = group.filter((point) => Math.abs(point.cx - xMed) <= limit);
      const kept = filtered.length >= Math.max(2, group.length * 0.35) ? filtered : group;
      return {
        cx: median(kept.map((point) => point.cx)),
        cy: median(kept.map((point) => point.cy)),
        support: kept.length,
      };
    });
}

function smoothCurve(nodes, radius = 86) {
  if (nodes.length <= 2) return nodes;
  return nodes.map((target) => {
    let sw = 0;
    let sd = 0;
    let sx = 0;
    let sdd = 0;
    let sdx = 0;
    nodes.forEach((node) => {
      const dist = Math.abs(node.cy - target.cy);
      if (dist > radius) return;
      const scaled = dist / radius;
      const kernel = Math.pow(1 - Math.pow(scaled, 3), 3);
      const weight = kernel * Math.sqrt(Math.max(1, node.support || 1));
      const d = node.cy - target.cy;
      sw += weight;
      sd += weight * d;
      sx += weight * node.cx;
      sdd += weight * d * d;
      sdx += weight * d * node.cx;
    });
    const denom = sw * sdd - sd * sd;
    const cx = Math.abs(denom) > 1e-6 ? (sx * sdd - sd * sdx) / denom : sx / Math.max(sw, 1e-6);
    return { ...target, cx };
  });
}

function interpolateCurve(nodes, cy) {
  if (!nodes.length) return null;
  if (cy <= nodes[0].cy) return nodes[0].cx;
  if (cy >= nodes[nodes.length - 1].cy) return nodes[nodes.length - 1].cx;
  for (let index = 1; index < nodes.length; index += 1) {
    const prev = nodes[index - 1];
    const next = nodes[index];
    if (cy <= next.cy) {
      const span = Math.max(1e-6, next.cy - prev.cy);
      const t = (cy - prev.cy) / span;
      return prev.cx + t * (next.cx - prev.cx);
    }
  }
  return nodes[nodes.length - 1].cx;
}

function smoothSeries(sections, key, radius = 82) {
  return sections.map((target) => {
    let total = 0;
    let weightSum = 0;
    sections.forEach((section) => {
      const dist = Math.abs(section.cy - target.cy);
      if (dist > radius) return;
      const scaled = dist / radius;
      const weight = Math.pow(1 - Math.pow(scaled, 3), 3);
      total += weight * section[key];
      weightSum += weight;
    });
    return weightSum ? total / weightSum : target[key];
  });
}

function fitCorridor(leftPoints, rightPoints, fit) {
  const leftNodes = smoothCurve(robustBoundary(leftPoints));
  const rightNodes = smoothCurve(robustBoundary(rightPoints));
  if (leftNodes.length < 2 || rightNodes.length < 2) return null;
  const minY = Math.max(leftNodes[0].cy, rightNodes[0].cy);
  const maxY = Math.min(leftNodes[leftNodes.length - 1].cy, rightNodes[rightNodes.length - 1].cy);
  if (maxY - minY < 22) return null;
  const sectionCount = Math.max(3, Math.min(28, Math.round((maxY - minY) / 28) + 1));
  let sections = [];
  for (let index = 0; index < sectionCount; index += 1) {
    const t = sectionCount === 1 ? 0 : index / (sectionCount - 1);
    const cy = minY + t * (maxY - minY);
    let lx = interpolateCurve(leftNodes, cy);
    let rx = interpolateCurve(rightNodes, cy);
    if (!Number.isFinite(lx) || !Number.isFinite(rx)) continue;
    if (rx < lx) [lx, rx] = [rx, lx];
    const width = rx - lx;
    if (width >= 10 && width <= fit.w * 0.78) sections.push({ cy, center: (lx + rx) / 2, width });
  }
  if (sections.length < 3) return null;
  const widths = sections.map((section) => section.width);
  const lowWidth = Math.max(12, quantile(widths, 0.1) * 0.45);
  const highWidth = Math.min(fit.w * 0.72, quantile(widths, 0.9) * 1.75);
  sections = sections.filter((section) => section.width >= lowWidth && section.width <= highWidth);
  if (sections.length < 3) return null;
  const centers = smoothSeries(sections, "center");
  const smoothedWidths = smoothSeries(sections, "width");
  return sections.map((section, index) => {
    const width = Math.max(lowWidth, Math.min(highWidth, smoothedWidths[index]));
    const center = centers[index];
    return {
      cy: section.cy,
      center,
      width,
      lx: center - width / 2,
      rx: center + width / 2,
    };
  });
}

function cross(o, a, b) {
  return (a.cx - o.cx) * (b.cy - o.cy) - (a.cy - o.cy) * (b.cx - o.cx);
}

function convexHull(points) {
  const unique = Array.from(
    new Map(points.map((point) => [`${Math.round(point.cx * 2)},${Math.round(point.cy * 2)}`, point])).values()
  ).sort((a, b) => (a.cx === b.cx ? a.cy - b.cy : a.cx - b.cx));
  if (unique.length <= 2) return unique;
  const lower = [];
  unique.forEach((point) => {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) lower.pop();
    lower.push(point);
  });
  const upper = [];
  unique.slice().reverse().forEach((point) => {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) upper.pop();
    upper.push(point);
  });
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

function drawPathPolygon(points) {
  if (points.length < 3) return false;
  sourceCtx.save();
  sourceCtx.fillStyle = "rgba(32, 199, 230, 0.23)";
  sourceCtx.strokeStyle = "rgba(32, 199, 230, 0.9)";
  sourceCtx.lineWidth = 2;
  sourceCtx.beginPath();
  sourceCtx.moveTo(points[0].cx, points[0].cy);
  points.slice(1).forEach((point) => sourceCtx.lineTo(point.cx, point.cy));
  sourceCtx.closePath();
  sourceCtx.fill();
  sourceCtx.stroke();
  sourceCtx.restore();
  return true;
}

function drawPathCorridor(sections) {
  const polygon = sections
    .map((section) => ({ cx: section.lx, cy: section.cy }))
    .concat(sections.slice().reverse().map((section) => ({ cx: section.rx, cy: section.cy })));
  if (!drawPathPolygon(polygon)) return false;
  sourceCtx.save();
  sourceCtx.strokeStyle = "rgba(32, 199, 230, 0.95)";
  sourceCtx.lineWidth = 2;
  sourceCtx.setLineDash([8, 7]);
  sourceCtx.beginPath();
  sourceCtx.moveTo(sections[0].center, sections[0].cy);
  sections.slice(1).forEach((section) => sourceCtx.lineTo(section.center, section.cy));
  sourceCtx.stroke();
  sourceCtx.restore();
  return true;
}

function collectReliability(trackSets) {
  const records = trackSets
    .map(({ win, tracks }) => (win ? frameReliability(win, tracks ? tracks.frame : frameIndex) : null))
    .filter(Boolean);
  const disabled = records.filter((item) => item.segment_disabled || item.segment_status === "disabled");
  const reasonCounts = {};
  const triggerFrames = new Set();
  disabled.forEach((item) => {
    (item.trigger_frame_indices || []).forEach((frame) => triggerFrames.add(frame));
    Object.entries(item.reason_counts || {}).forEach(([reason, count]) => {
      reasonCounts[reason] = (reasonCounts[reason] || 0) + Number(count || 0);
    });
  });
  return {
    has: records.length > 0,
    disabled: disabled.length > 0,
    frameDisabled: records.some((item) => item.frame_disabled),
    disabledWindows: disabled.length,
    totalWindows: records.length,
    triggerFrames: Array.from(triggerFrames).sort((a, b) => a - b),
    reasonCounts,
  };
}

function reliabilitySummary(stats, loading = false) {
  if (loading) return "loading";
  if (!stats || !stats.has) return "unknown";
  if (!stats.disabled) return "enabled";
  const reasons = Object.keys(stats.reasonCounts);
  const reason = reasons.length ? reasons.join(", ") : "disabled";
  const frames = stats.triggerFrames.slice(0, 3).join(", ");
  const suffix = frames ? ` / trigger ${frames}` : "";
  return `DISABLED / ${reason}${suffix}`;
}

function drawDisabledOverlay(stats) {
  if (!stats || !stats.disabled) return;
  const rect = sourceCanvas.getBoundingClientRect();
  const width = Math.min(rect.width - 28, 620);
  sourceCtx.save();
  sourceCtx.fillStyle = "rgba(185, 28, 28, 0.82)";
  sourceCtx.fillRect(14, 14, width, 72);
  sourceCtx.strokeStyle = "rgba(255, 255, 255, 0.72)";
  sourceCtx.lineWidth = 1.5;
  sourceCtx.strokeRect(14.75, 14.75, width - 1.5, 70.5);
  sourceCtx.fillStyle = "#ffffff";
  sourceCtx.font = "700 18px system-ui, sans-serif";
  sourceCtx.fillText("DISABLED 40-FRAME SEGMENT", 30, 41);
  sourceCtx.font = "500 13px system-ui, sans-serif";
  const reason = Object.keys(stats.reasonCounts).join(", ") || "reliability rule";
  const triggers = stats.triggerFrames.slice(0, 3).join(", ");
  sourceCtx.fillText(triggers ? `${reason} / trigger ${triggers}` : reason, 30, 66);
  sourceCtx.restore();
}

function drawPathOverlay(trackSets, fit) {
  const left = [];
  const right = [];
  const robotPoints = [];
  trackSets.forEach(({ win, tracks }) => {
    if (!tracks) return;
    tracks.points.forEach((point) => {
      const crumb = win._crumbMap.get(point[0]);
      if (!crumb || crumb.source === "sift_anchor") return;
      const [cx, cy] = imageToCanvas([point[1], point[2]], fit);
      const entry = { cx, cy };
      robotPoints.push(entry);
      if (crumb.side < 0) left.push(entry);
      else right.push(entry);
    });
  });
  const filteredLeft = broadFilter(left, fit, true);
  const filteredRight = broadFilter(right, fit, true);
  const usedPoints = filteredLeft.length + filteredRight.length;
  if (filteredLeft.length >= 6 && filteredRight.length >= 6) {
    const corridor = fitCorridor(filteredLeft, filteredRight, fit);
    if (corridor && drawPathCorridor(corridor)) {
      return { mode: "smoothed", points: robotPoints.length, used: usedPoints, sections: corridor.length };
    }
  }
  const filteredRobot = broadFilter(robotPoints, fit, false);
  let drawn = false;
  if (filteredRobot.length >= 3) {
    drawn = drawPathPolygon(convexHull(filteredRobot));
  } else if (robotPoints.length >= 3) {
    drawn = drawPathPolygon(convexHull(robotPoints));
  }
  return { mode: drawn ? "filtered hull" : "none", points: robotPoints.length, used: filteredRobot.length, sections: 0 };
}

function drawMask(win, fit) {
  if (!win.mask || !win.mask.image || frameIndex !== win.mask.reference_frame) return false;
  const img = maskImage(win);
  if (!img) return false;
  if (!img.complete) {
    img.onload = () => window.requestAnimationFrame(drawMain);
    return false;
  }
  sourceCtx.drawImage(img, fit.x, fit.y, fit.w, fit.h);
  return true;
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
    `confidence ${formatConfidence(best.confidence)}, ` +
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

function updateReliabilitySettingsFromControls() {
  reliabilitySettings = readReliabilitySettingsFromControls();
  applyReliabilitySettingsToControls(reliabilitySettings);
  saveReliabilitySettings(reliabilitySettings);
  if (data) drawMain();
}

reliabilitySettings = loadStoredReliabilitySettings();
applyReliabilitySettingsToControls(reliabilitySettings);
[
  reliabilitySpanInput,
  reliabilityBlockInput,
  reliabilityPointsInput,
  reliabilitySegmentInput,
].forEach((input) => {
  input.addEventListener("change", updateReliabilitySettingsFromControls);
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


LAZY_JS = """const sourceCanvas = document.getElementById("sourceCanvas");
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
const confidenceText = document.getElementById("confidenceText");
const reliabilityText = document.getElementById("reliabilityText");
const reliabilitySpanInput = document.getElementById("reliabilitySpanInput");
const reliabilityBlockInput = document.getElementById("reliabilityBlockInput");
const reliabilityPointsInput = document.getElementById("reliabilityPointsInput");
const reliabilitySegmentInput = document.getElementById("reliabilitySegmentInput");
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
const windowCache = new Map();
const windowPromises = new Map();
let pendingMainFrame = null;
let pendingSelectedFrame = null;
const RELIABILITY_STORAGE_KEY = "avt.viewer.reliability.thresholds.v1";
const RELIABILITY_REASON = "stop_extreme_slow_motion";
const RELIABILITY_DEFAULTS = {
  spanFrames: 10,
  blockSizePx: 6,
  minPoints: 3,
  segmentSizeFrames: 40,
};
let reliabilitySettings = { ...RELIABILITY_DEFAULTS };

function normalizedReliabilitySettings(settings) {
  const spanFrames = Math.max(2, Math.round(Number(settings.spanFrames) || RELIABILITY_DEFAULTS.spanFrames));
  const blockSizePx = Math.max(0.5, Number(settings.blockSizePx) || RELIABILITY_DEFAULTS.blockSizePx);
  const minPoints = Math.max(1, Math.round(Number(settings.minPoints) || RELIABILITY_DEFAULTS.minPoints));
  const segmentSizeFrames = Math.max(
    1,
    Math.round(Number(settings.segmentSizeFrames) || RELIABILITY_DEFAULTS.segmentSizeFrames)
  );
  return { spanFrames, blockSizePx, minPoints, segmentSizeFrames };
}

function loadStoredReliabilitySettings() {
  let stored = {};
  try {
    stored = JSON.parse(window.localStorage.getItem(RELIABILITY_STORAGE_KEY) || "{}");
  } catch (_error) {
    stored = {};
  }
  const params = new URLSearchParams(window.location.search);
  return normalizedReliabilitySettings({
    ...RELIABILITY_DEFAULTS,
    ...stored,
    spanFrames: params.get("span") || params.get("reliability_span") || stored.spanFrames,
    blockSizePx: params.get("block") || params.get("reliability_block") || stored.blockSizePx,
    minPoints: params.get("points") || params.get("reliability_points") || stored.minPoints,
    segmentSizeFrames: params.get("segment") || params.get("reliability_segment") || stored.segmentSizeFrames,
  });
}

function applyReliabilitySettingsToControls(settings) {
  reliabilitySpanInput.value = settings.spanFrames;
  reliabilityBlockInput.value = settings.blockSizePx;
  reliabilityPointsInput.value = settings.minPoints;
  reliabilitySegmentInput.value = settings.segmentSizeFrames;
}

function readReliabilitySettingsFromControls() {
  return normalizedReliabilitySettings({
    spanFrames: reliabilitySpanInput.value,
    blockSizePx: reliabilityBlockInput.value,
    minPoints: reliabilityPointsInput.value,
    segmentSizeFrames: reliabilitySegmentInput.value,
  });
}

function saveReliabilitySettings(settings) {
  try {
    window.localStorage.setItem(RELIABILITY_STORAGE_KEY, JSON.stringify(settings));
  } catch (_error) {
    // Local storage is optional; the controls still work for this session.
  }
}

function reliabilitySettingsKey(settings) {
  return [
    settings.spanFrames,
    settings.blockSizePx,
    settings.minPoints,
    settings.segmentSizeFrames,
  ].join(":");
}

function segmentIdForFrame(index, segmentSizeFrames) {
  return Math.floor(Number(index) / segmentSizeFrames);
}

function segmentBoundsForFrame(index, segmentSizeFrames) {
  const start = segmentIdForFrame(index, segmentSizeFrames) * segmentSizeFrames;
  return { start, end: start + segmentSizeFrames };
}

function ensurePointMaps(win) {
  if (win._pointMaps) return win._pointMaps;
  const maps = new Map();
  win.frames.forEach((frame) => {
    const pointMap = new Map();
    (frame.points || []).forEach((point) => {
      const x = Number(point[1]);
      const y = Number(point[2]);
      if (Number.isFinite(x) && Number.isFinite(y)) pointMap.set(Number(point[0]), point);
    });
    maps.set(Number(frame.frame), pointMap);
  });
  win._pointMaps = maps;
  return maps;
}

function computeWindowReliability(win, settings = reliabilitySettings) {
  prepareWindow(win);
  const key = reliabilitySettingsKey(settings);
  if (win._reliabilityCache && win._reliabilityCache.key === key) return win._reliabilityCache;

  const pointMaps = ensurePointMaps(win);
  const siftIds = win.crumbs
    .filter((crumb) => String(crumb.source || "").startsWith("sift_"))
    .map((crumb) => Number(crumb.id));
  const frameReasons = new Map();
  const disabledSegments = new Map();
  if (siftIds.length) {
    const firstFrame = Number(win.seq_start) + settings.spanFrames - 1;
    const lastFrame = Number(win.seq_end) - 1;
    for (let frame = firstFrame; frame <= lastFrame; frame += 1) {
      let stationaryCount = 0;
      for (const id of siftIds) {
        let minX = Infinity;
        let minY = Infinity;
        let maxX = -Infinity;
        let maxY = -Infinity;
        let valid = true;
        for (let spanFrame = frame - settings.spanFrames + 1; spanFrame <= frame; spanFrame += 1) {
          const pointMap = pointMaps.get(spanFrame);
          const point = pointMap ? pointMap.get(id) : null;
          if (!point) {
            valid = false;
            break;
          }
          const x = Number(point[1]);
          const y = Number(point[2]);
          if (!Number.isFinite(x) || !Number.isFinite(y)) {
            valid = false;
            break;
          }
          minX = Math.min(minX, x);
          maxX = Math.max(maxX, x);
          minY = Math.min(minY, y);
          maxY = Math.max(maxY, y);
        }
        if (valid && maxX - minX <= settings.blockSizePx && maxY - minY <= settings.blockSizePx) {
          stationaryCount += 1;
          if (stationaryCount >= settings.minPoints) {
            frameReasons.set(frame, [RELIABILITY_REASON]);
            const segmentId = segmentIdForFrame(frame, settings.segmentSizeFrames);
            if (!disabledSegments.has(segmentId)) disabledSegments.set(segmentId, []);
            disabledSegments.get(segmentId).push(frame);
            break;
          }
        }
      }
    }
  }

  const cache = { key, frameReasons, disabledSegments };
  win._reliabilityCache = cache;
  return cache;
}

function frameReliability(win, index, settings = reliabilitySettings) {
  const cache = computeWindowReliability(win, settings);
  const frame = Number(index);
  const segmentId = segmentIdForFrame(frame, settings.segmentSizeFrames);
  const bounds = segmentBoundsForFrame(frame, settings.segmentSizeFrames);
  const triggerFrames = (cache.disabledSegments.get(segmentId) || []).slice();
  const frameReasons = cache.frameReasons.get(frame) || [];
  const reasonCounts = {};
  triggerFrames.forEach((triggerFrame) => {
    (cache.frameReasons.get(triggerFrame) || []).forEach((reason) => {
      reasonCounts[reason] = (reasonCounts[reason] || 0) + 1;
    });
  });
  return {
    schema: "avt_frame_segment_reliability_v1",
    filter: "stationary_sift_segment_filter",
    action: "client_live",
    segment_size_frames: settings.segmentSizeFrames,
    segment_id: segmentId,
    segment_start_frame: bounds.start,
    segment_end_frame: bounds.end,
    frame_unreliable: frameReasons.length > 0,
    segment_unreliable: triggerFrames.length > 0,
    frame_disabled: frameReasons.length > 0,
    segment_disabled: triggerFrames.length > 0,
    segment_status: triggerFrames.length > 0 ? "disabled" : "enabled",
    trigger_frame_indices: triggerFrames,
    unreliable_point_ids: [],
    reason_counts: reasonCounts,
  };
}

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

function prepareWindow(win) {
  if (!win._frameMap) {
    win._frameMap = new Map(win.frames.map((frame) => [frame.frame, frame]));
    win._crumbMap = new Map(win.crumbs.map((crumb) => [crumb.id, crumb]));
  }
  return win;
}

function windowSegmentUrl(summary) {
  return summary.segment || `data/windows/${summary.id}.json`;
}

function ensureWindow(summary) {
  if (!summary) return Promise.resolve(null);
  if (windowCache.has(summary.id)) {
    return Promise.resolve(windowCache.get(summary.id));
  }
  if (windowPromises.has(summary.id)) {
    return windowPromises.get(summary.id);
  }
  const promise = fetch(windowSegmentUrl(summary))
    .then((response) => {
      if (!response.ok) throw new Error(`Could not load ${summary.id}: ${response.status}`);
      return response.json();
    })
    .then((win) => {
      prepareWindow(win);
      windowCache.set(win.id, win);
      windowPromises.delete(win.id);
      return win;
    })
    .catch((error) => {
      windowPromises.delete(summary.id);
      throw error;
    });
  windowPromises.set(summary.id, promise);
  return promise;
}

function frameTracks(win, index) {
  if (!win) return null;
  prepareWindow(win);
  return win._frameMap.get(index);
}

function activeWindowSummaries() {
  const covers = coveringWindows(frameIndex);
  if (!covers.length) return [];
  if (autoWindow.checked) {
    const latest = covers.reduce((best, win) => (win.seq_start > best.seq_start ? win : best), covers[0]);
    selectedWindowId = latest.id;
    return covers;
  }
  if (!selectedWindowId || !covers.some((win) => win.id === selectedWindowId)) {
    selectedWindowId = covers[0].id;
  }
  const selected = covers.find((win) => win.id === selectedWindowId) || covers[0];
  return [selected];
}

function focusSummary(summaries) {
  if (!summaries.length) return null;
  return summaries.find((win) => win.id === selectedWindowId) || summaries[summaries.length - 1];
}

function preloadNearbyWindows(summaries) {
  const seen = new Set();
  summaries.forEach((summary) => {
    if (!summary || seen.has(summary.id)) return;
    seen.add(summary.id);
    ensureWindow(summary).catch((error) => console.warn(error));
    const start = data.windows.findIndex((win) => win.id === summary.id);
    if (start < 0) return;
    for (let offset = 1; offset <= 2; offset += 1) {
      const next = data.windows[start + offset];
      if (next && !seen.has(next.id)) {
        seen.add(next.id);
        ensureWindow(next).catch((error) => console.warn(error));
      }
    }
  });
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

function pointColor(crumb) {
  if (crumb.source === "sift_anchor") return "#10b981";
  return crumb.side < 0 ? "#3b82f6" : "#f5b84b";
}

function pointConfidence(point) {
  if (!point || point.length < 4) return null;
  const confidence = Number(point[3]);
  if (!Number.isFinite(confidence)) return null;
  return Math.max(0, Math.min(1, confidence));
}

function formatConfidence(confidence) {
  return confidence === null ? "unknown" : `${Math.round(confidence * 100)}%`;
}

function pointAlpha(confidence) {
  return confidence === null ? 1 : 0.25 + confidence * 0.75;
}

function pointRadius(confidence, selected) {
  if (selected) return 6;
  return confidence === null ? 4 : 2.5 + confidence * 3;
}

function confidenceSummary(trackSets, loading = false) {
  if (loading) return "loading";
  const values = [];
  trackSets.forEach(({ tracks }) => {
    if (!tracks) return;
    tracks.points.forEach((point) => {
      const confidence = pointConfidence(point);
      if (confidence !== null) values.push(confidence);
    });
  });
  if (!values.length) return "unknown";
  const avg = values.reduce((sum, value) => sum + value, 0) / values.length;
  return `avg ${formatConfidence(avg)} / min ${formatConfidence(Math.min(...values))} / max ${formatConfidence(Math.max(...values))}`;
}

function median(values) {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function quantile(values, q) {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  return sorted[base + 1] === undefined ? sorted[base] : sorted[base] + rest * (sorted[base + 1] - sorted[base]);
}

function mad(values, center) {
  return median(values.map((value) => Math.abs(value - center)));
}

function broadFilter(points, fit, useMad = true) {
  const margin = 36;
  const inside = points.filter((point) =>
    point.cx >= fit.x - margin &&
    point.cx <= fit.x + fit.w + margin &&
    point.cy >= fit.y - margin &&
    point.cy <= fit.y + fit.h + margin
  );
  if (inside.length < 12) return inside;
  const xs = inside.map((point) => point.cx);
  const ys = inside.map((point) => point.cy);
  const xLow = quantile(xs, 0.02);
  const xHigh = quantile(xs, 0.98);
  const yLow = quantile(ys, 0.02);
  const yHigh = quantile(ys, 0.98);
  const clipped = inside.filter((point) =>
    point.cx >= xLow - 12 &&
    point.cx <= xHigh + 12 &&
    point.cy >= yLow - 12 &&
    point.cy <= yHigh + 12
  );
  if (!useMad || clipped.length < 12) return clipped;
  const cx = median(clipped.map((point) => point.cx));
  const cy = median(clipped.map((point) => point.cy));
  const xLimit = Math.max(42, 5.0 * 1.4826 * mad(clipped.map((point) => point.cx), cx));
  const yLimit = Math.max(42, 5.0 * 1.4826 * mad(clipped.map((point) => point.cy), cy));
  const filtered = clipped.filter((point) => Math.abs(point.cx - cx) <= xLimit && Math.abs(point.cy - cy) <= yLimit);
  return filtered.length >= Math.max(6, clipped.length * 0.45) ? filtered : clipped;
}

function robustBoundary(points, bandHeight = 30) {
  const bins = new Map();
  points.forEach((point) => {
    const key = Math.round(point.cy / bandHeight);
    if (!bins.has(key)) bins.set(key, []);
    bins.get(key).push(point);
  });
  return Array.from(bins.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([, group]) => {
      const xs = group.map((point) => point.cx);
      const xMed = median(xs);
      const spread = 1.4826 * mad(xs, xMed);
      const limit = Math.max(10, Math.min(70, 3.0 * spread || 10));
      const filtered = group.filter((point) => Math.abs(point.cx - xMed) <= limit);
      const kept = filtered.length >= Math.max(2, group.length * 0.35) ? filtered : group;
      return {
        cx: median(kept.map((point) => point.cx)),
        cy: median(kept.map((point) => point.cy)),
        support: kept.length,
      };
    });
}

function smoothCurve(nodes, radius = 86) {
  if (nodes.length <= 2) return nodes;
  return nodes.map((target) => {
    let sw = 0;
    let sd = 0;
    let sx = 0;
    let sdd = 0;
    let sdx = 0;
    nodes.forEach((node) => {
      const dist = Math.abs(node.cy - target.cy);
      if (dist > radius) return;
      const scaled = dist / radius;
      const kernel = Math.pow(1 - Math.pow(scaled, 3), 3);
      const weight = kernel * Math.sqrt(Math.max(1, node.support || 1));
      const d = node.cy - target.cy;
      sw += weight;
      sd += weight * d;
      sx += weight * node.cx;
      sdd += weight * d * d;
      sdx += weight * d * node.cx;
    });
    const denom = sw * sdd - sd * sd;
    const cx = Math.abs(denom) > 1e-6 ? (sx * sdd - sd * sdx) / denom : sx / Math.max(sw, 1e-6);
    return { ...target, cx };
  });
}

function interpolateCurve(nodes, cy) {
  if (!nodes.length) return null;
  if (cy <= nodes[0].cy) return nodes[0].cx;
  if (cy >= nodes[nodes.length - 1].cy) return nodes[nodes.length - 1].cx;
  for (let index = 1; index < nodes.length; index += 1) {
    const prev = nodes[index - 1];
    const next = nodes[index];
    if (cy <= next.cy) {
      const span = Math.max(1e-6, next.cy - prev.cy);
      const t = (cy - prev.cy) / span;
      return prev.cx + t * (next.cx - prev.cx);
    }
  }
  return nodes[nodes.length - 1].cx;
}

function smoothSeries(sections, key, radius = 82) {
  return sections.map((target) => {
    let total = 0;
    let weightSum = 0;
    sections.forEach((section) => {
      const dist = Math.abs(section.cy - target.cy);
      if (dist > radius) return;
      const scaled = dist / radius;
      const weight = Math.pow(1 - Math.pow(scaled, 3), 3);
      total += weight * section[key];
      weightSum += weight;
    });
    return weightSum ? total / weightSum : target[key];
  });
}

function fitCorridor(leftPoints, rightPoints, fit) {
  const leftNodes = smoothCurve(robustBoundary(leftPoints));
  const rightNodes = smoothCurve(robustBoundary(rightPoints));
  if (leftNodes.length < 2 || rightNodes.length < 2) return null;
  const minY = Math.max(leftNodes[0].cy, rightNodes[0].cy);
  const maxY = Math.min(leftNodes[leftNodes.length - 1].cy, rightNodes[rightNodes.length - 1].cy);
  if (maxY - minY < 22) return null;
  const sectionCount = Math.max(3, Math.min(28, Math.round((maxY - minY) / 28) + 1));
  let sections = [];
  for (let index = 0; index < sectionCount; index += 1) {
    const t = sectionCount === 1 ? 0 : index / (sectionCount - 1);
    const cy = minY + t * (maxY - minY);
    let lx = interpolateCurve(leftNodes, cy);
    let rx = interpolateCurve(rightNodes, cy);
    if (!Number.isFinite(lx) || !Number.isFinite(rx)) continue;
    if (rx < lx) [lx, rx] = [rx, lx];
    const width = rx - lx;
    if (width >= 10 && width <= fit.w * 0.78) sections.push({ cy, center: (lx + rx) / 2, width });
  }
  if (sections.length < 3) return null;
  const widths = sections.map((section) => section.width);
  const lowWidth = Math.max(12, quantile(widths, 0.1) * 0.45);
  const highWidth = Math.min(fit.w * 0.72, quantile(widths, 0.9) * 1.75);
  sections = sections.filter((section) => section.width >= lowWidth && section.width <= highWidth);
  if (sections.length < 3) return null;
  const centers = smoothSeries(sections, "center");
  const smoothedWidths = smoothSeries(sections, "width");
  return sections.map((section, index) => {
    const width = Math.max(lowWidth, Math.min(highWidth, smoothedWidths[index]));
    const center = centers[index];
    return {
      cy: section.cy,
      center,
      width,
      lx: center - width / 2,
      rx: center + width / 2,
    };
  });
}

function cross(o, a, b) {
  return (a.cx - o.cx) * (b.cy - o.cy) - (a.cy - o.cy) * (b.cx - o.cx);
}

function convexHull(points) {
  const unique = Array.from(
    new Map(points.map((point) => [`${Math.round(point.cx * 2)},${Math.round(point.cy * 2)}`, point])).values()
  ).sort((a, b) => (a.cx === b.cx ? a.cy - b.cy : a.cx - b.cx));
  if (unique.length <= 2) return unique;
  const lower = [];
  unique.forEach((point) => {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) lower.pop();
    lower.push(point);
  });
  const upper = [];
  unique.slice().reverse().forEach((point) => {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) upper.pop();
    upper.push(point);
  });
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

function drawPathPolygon(points) {
  if (points.length < 3) return false;
  sourceCtx.save();
  sourceCtx.fillStyle = "rgba(32, 199, 230, 0.23)";
  sourceCtx.strokeStyle = "rgba(32, 199, 230, 0.9)";
  sourceCtx.lineWidth = 2;
  sourceCtx.beginPath();
  sourceCtx.moveTo(points[0].cx, points[0].cy);
  points.slice(1).forEach((point) => sourceCtx.lineTo(point.cx, point.cy));
  sourceCtx.closePath();
  sourceCtx.fill();
  sourceCtx.stroke();
  sourceCtx.restore();
  return true;
}

function drawPathCorridor(sections) {
  const polygon = sections
    .map((section) => ({ cx: section.lx, cy: section.cy }))
    .concat(sections.slice().reverse().map((section) => ({ cx: section.rx, cy: section.cy })));
  if (!drawPathPolygon(polygon)) return false;
  sourceCtx.save();
  sourceCtx.strokeStyle = "rgba(32, 199, 230, 0.95)";
  sourceCtx.lineWidth = 2;
  sourceCtx.setLineDash([8, 7]);
  sourceCtx.beginPath();
  sourceCtx.moveTo(sections[0].center, sections[0].cy);
  sections.slice(1).forEach((section) => sourceCtx.lineTo(section.center, section.cy));
  sourceCtx.stroke();
  sourceCtx.restore();
  return true;
}

function collectReliability(trackSets) {
  const records = trackSets
    .map(({ win, tracks }) => (win ? frameReliability(win, tracks ? tracks.frame : frameIndex) : null))
    .filter(Boolean);
  const disabled = records.filter((item) => item.segment_disabled || item.segment_status === "disabled");
  const reasonCounts = {};
  const triggerFrames = new Set();
  disabled.forEach((item) => {
    (item.trigger_frame_indices || []).forEach((frame) => triggerFrames.add(frame));
    Object.entries(item.reason_counts || {}).forEach(([reason, count]) => {
      reasonCounts[reason] = (reasonCounts[reason] || 0) + Number(count || 0);
    });
  });
  return {
    has: records.length > 0,
    disabled: disabled.length > 0,
    frameDisabled: records.some((item) => item.frame_disabled),
    disabledWindows: disabled.length,
    totalWindows: records.length,
    triggerFrames: Array.from(triggerFrames).sort((a, b) => a - b),
    reasonCounts,
  };
}

function reliabilitySummary(stats, loading = false) {
  if (loading) return "loading";
  if (!stats || !stats.has) return "unknown";
  if (!stats.disabled) return "enabled";
  const reasons = Object.keys(stats.reasonCounts);
  const reason = reasons.length ? reasons.join(", ") : "disabled";
  const frames = stats.triggerFrames.slice(0, 3).join(", ");
  const suffix = frames ? ` / trigger ${frames}` : "";
  return `DISABLED / ${reason}${suffix}`;
}

function drawDisabledOverlay(stats) {
  if (!stats || !stats.disabled) return;
  const rect = sourceCanvas.getBoundingClientRect();
  const width = Math.min(rect.width - 28, 620);
  sourceCtx.save();
  sourceCtx.fillStyle = "rgba(185, 28, 28, 0.82)";
  sourceCtx.fillRect(14, 14, width, 72);
  sourceCtx.strokeStyle = "rgba(255, 255, 255, 0.72)";
  sourceCtx.lineWidth = 1.5;
  sourceCtx.strokeRect(14.75, 14.75, width - 1.5, 70.5);
  sourceCtx.fillStyle = "#ffffff";
  sourceCtx.font = "700 18px system-ui, sans-serif";
  sourceCtx.fillText("DISABLED 40-FRAME SEGMENT", 30, 41);
  sourceCtx.font = "500 13px system-ui, sans-serif";
  const reason = Object.keys(stats.reasonCounts).join(", ") || "reliability rule";
  const triggers = stats.triggerFrames.slice(0, 3).join(", ");
  sourceCtx.fillText(triggers ? `${reason} / trigger ${triggers}` : reason, 30, 66);
  sourceCtx.restore();
}

function drawPathOverlay(trackSets, fit) {
  const left = [];
  const right = [];
  const robotPoints = [];
  trackSets.forEach(({ win, tracks }) => {
    if (!tracks) return;
    tracks.points.forEach((point) => {
      const crumb = win._crumbMap.get(point[0]);
      if (!crumb || crumb.source === "sift_anchor") return;
      const [cx, cy] = imageToCanvas([point[1], point[2]], fit);
      const entry = { cx, cy };
      robotPoints.push(entry);
      if (crumb.side < 0) left.push(entry);
      else right.push(entry);
    });
  });
  const filteredLeft = broadFilter(left, fit, true);
  const filteredRight = broadFilter(right, fit, true);
  const usedPoints = filteredLeft.length + filteredRight.length;
  if (filteredLeft.length >= 6 && filteredRight.length >= 6) {
    const corridor = fitCorridor(filteredLeft, filteredRight, fit);
    if (corridor && drawPathCorridor(corridor)) {
      return { mode: "smoothed", points: robotPoints.length, used: usedPoints, sections: corridor.length };
    }
  }
  const filteredRobot = broadFilter(robotPoints, fit, false);
  let drawn = false;
  if (filteredRobot.length >= 3) {
    drawn = drawPathPolygon(convexHull(filteredRobot));
  } else if (robotPoints.length >= 3) {
    drawn = drawPathPolygon(convexHull(robotPoints));
  }
  return { mode: drawn ? "filtered hull" : "none", points: robotPoints.length, used: filteredRobot.length, sections: 0 };
}

function drawMask(win, fit) {
  if (!win.mask || !win.mask.image || frameIndex !== win.mask.reference_frame) return false;
  const img = maskImage(win);
  if (!img) return false;
  if (!img.complete) {
    img.onload = () => window.requestAnimationFrame(drawMain);
    return false;
  }
  sourceCtx.drawImage(img, fit.x, fit.y, fit.w, fit.h);
  return true;
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
  const summaries = activeWindowSummaries();
  const missing = summaries.filter((summary) => !windowCache.has(summary.id));
  if (missing.length) {
    updateText(summaries, [], true);
    Promise.all(summaries.map((summary) => ensureWindow(summary)))
      .then((loaded) => {
        const activeIds = new Set(activeWindowSummaries().map((summary) => summary.id));
        if (loaded.some((win) => activeIds.has(win.id))) drawMain();
      })
      .catch((error) => {
        statusText.textContent = error.message;
      });
    return;
  }
  const trackSets = summaries
    .map((summary) => windowCache.get(summary.id))
    .filter(Boolean)
    .map((win) => ({ win, tracks: frameTracks(win, frameIndex) }));
  const reliabilityStats = collectReliability(trackSets);
  let pathStats = { mode: "off", points: 0 };
  if (fit && showMask.checked) {
    const savedMaskDrawn = trackSets.some(({ win }) => drawMask(win, fit));
    pathStats = savedMaskDrawn ? { mode: "saved", points: 0 } : drawPathOverlay(trackSets, fit);
  }
  drawPoints = [];
  if (fit && showPoints.checked) {
    trackSets.forEach(({ win, tracks }) => {
      if (!tracks) return;
      tracks.points.forEach((point) => {
        const crumb = win._crumbMap.get(point[0]);
        if (!crumb) return;
        const [cx, cy] = imageToCanvas([point[1], point[2]], fit);
        const confidence = pointConfidence(point);
        const selected = selectedPoint && selectedPoint.windowId === win.id && selectedPoint.pointId === point[0];
        sourceCtx.save();
        sourceCtx.globalAlpha = selected ? 1 : pointAlpha(confidence);
        sourceCtx.fillStyle = selected ? "#ef4444" : pointColor(crumb);
        sourceCtx.strokeStyle = "rgba(0,0,0,0.55)";
        sourceCtx.lineWidth = selected ? 2.5 : 1.5;
        sourceCtx.beginPath();
        sourceCtx.arc(cx, cy, pointRadius(confidence, selected), 0, Math.PI * 2);
        sourceCtx.fill();
        sourceCtx.stroke();
        sourceCtx.restore();
        drawPoints.push({ cx, cy, pointId: point[0], x: point[1], y: point[2], confidence, crumb, window: win });
      });
    });
  }
  if (fit) drawDisabledOverlay(reliabilityStats);
  updateText(summaries, trackSets, false, pathStats, reliabilityStats);
  preloadNearbyWindows(summaries);
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

function updateText(
  summaries,
  trackSets,
  loading,
  pathStats = { mode: "none", points: 0 },
  reliabilityStats = collectReliability(trackSets)
) {
  const frame = data.frames[frameIndex];
  frameText.textContent = `${frameIndex} / ${data.frames.length - 1}`;
  timeText.textContent = `${Number(frame.rel_time_sec || 0).toFixed(2)} s`;
  if (summaries.length) {
    const focus = focusSummary(summaries);
    const loadedFocus = focus ? windowCache.get(focus.id) : null;
    const trackerNames = Array.from(
      new Set(summaries.map((win) => (win.tracker && win.tracker.name) || "tracker"))
    ).join(", ");
    const savedMaskVisible = summaries.some((win) => win.mask && win.mask.image && win.mask.reference_frame === frameIndex);
    const hasSavedMasks = summaries.some((win) => win.mask && win.mask.image);
    const maskRange = `${summaries[0].seq_start}-${summaries[summaries.length - 1].seq_start}`;
    let maskNote = `path refs ${maskRange}`;
    if (pathStats.mode === "smoothed") {
      maskNote = `smoothed path ${pathStats.used}/${pathStats.points} pts, ${pathStats.sections} bands`;
    } else if (pathStats.mode === "filtered hull") {
      maskNote = `filtered hull ${pathStats.used}/${pathStats.points} pts`;
    }
    else if (pathStats.mode === "saved" || savedMaskVisible) maskNote = "saved mask visible";
    else if (!hasSavedMasks) maskNote = "no saved mask; path from tracks";
    const windowNote = autoWindow.checked ? `${summaries.length} windows` : focus.id;
    windowText.textContent = `${windowNote} / ${trackerNames} / ${loading ? "loading raw segments" : maskNote}`;
    if (loadedFocus && loadedFocus.video_url) {
      if (windowVideo.src !== loadedFocus.video_url) windowVideo.src = loadedFocus.video_url;
      videoText.textContent = `${loadedFocus.id} reverse_video.mp4`;
    } else {
      windowVideo.removeAttribute("src");
      videoText.textContent = loading ? "Loading raw point segments." : "No video URL was embedded for this window.";
    }
  } else {
    windowText.textContent = "none";
    windowVideo.removeAttribute("src");
    videoText.textContent = "No active window for this frame.";
  }
  const pointCount = trackSets.reduce((total, item) => total + (item.tracks ? item.tracks.points.length : 0), 0);
  pointsText.textContent = loading ? "loading" : pointCount;
  confidenceText.textContent = confidenceSummary(trackSets, loading);
  reliabilityText.textContent = reliabilitySummary(reliabilityStats, loading);
  frameSlider.value = frameIndex;
  frameInput.value = frameIndex;
  windowSelect.value = selectedWindowId || "";
  if (!selectedPoint) selectedText.textContent = "Click any rendered point.";
}

function setFrame(index) {
  frameIndex = Math.max(0, Math.min(data.frames.length - 1, Number(index) || 0));
  preloadFrames(frameIndex);
  const summaries = activeWindowSummaries();
  if (summaries.length) {
    Promise.all(summaries.map((summary) => ensureWindow(summary)))
      .then((loaded) => {
        const activeIds = new Set(activeWindowSummaries().map((summary) => summary.id));
        if (loaded.some((win) => activeIds.has(win.id))) drawMain();
      })
      .catch((error) => {
        statusText.textContent = error.message;
      });
  }
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
    `confidence ${formatConfidence(best.confidence)}, ` +
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

function updateReliabilitySettingsFromControls() {
  reliabilitySettings = readReliabilitySettingsFromControls();
  applyReliabilitySettingsToControls(reliabilitySettings);
  saveReliabilitySettings(reliabilitySettings);
  if (data) drawMain();
}

reliabilitySettings = loadStoredReliabilitySettings();
applyReliabilitySettingsToControls(reliabilitySettings);
[
  reliabilitySpanInput,
  reliabilityBlockInput,
  reliabilityPointsInput,
  reliabilitySegmentInput,
].forEach((input) => {
  input.addEventListener("change", updateReliabilitySettingsFromControls);
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
      `${data.metadata.frame_count} frames, ${data.metadata.successful_windows} windows. ` +
      `Raw tracks load per window segment; trackers: ${data.metadata.trackers.join(", ")}.`;
    const requestedFrame = Number(new URLSearchParams(window.location.search).get("frame") || 0);
    setFrame(Number.isFinite(requestedFrame) ? requestedFrame : 0);
  })
  .catch((error) => {
    statusText.textContent = error.message;
  });
"""


def write_viewer(viewer_dir: Path, payload: dict[str, Any]) -> None:
    data_dir = viewer_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "prediction_tracks.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    (viewer_dir / "index.html").write_text(HTML, encoding="utf-8")
    (viewer_dir / "styles.css").write_text(CSS, encoding="utf-8")
    (viewer_dir / "app.js").write_text(LAZY_JS, encoding="utf-8")


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
