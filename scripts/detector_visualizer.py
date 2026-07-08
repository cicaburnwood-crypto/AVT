#!/usr/bin/env python3
"""Interactive Web UI for comparing AVT keypoint detector outputs."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TORCH_HOME", "/home/wolfie/.cache/torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from avt.detectors.base import filter_keypoints_by_mask, make_keypoint
from avt.detectors.config import XFeatConfig
from avt.detectors.orb import OrbDetector
from avt.detectors.sift import SiftDetector
from avt.detectors.superpoint import SuperPointSuperGlueDetector
from avt.querying import QueryConfig


DEFAULT_FRAMES_ROOT = (
    REPO_ROOT
    / "data/prepared_top10_cache/01_ride_16933_20240124133319_front_uid1000_frames"
)
XFEAT_LOCAL_REPO = Path("/home/wolfie/.cache/torch/hub/verlab_accelerated_features_main")

METHODS = {
    "sift": {"label": "SIFT", "color": "#22d3ee"},
    "orb": {"label": "ORB", "color": "#f59e0b"},
    "superpoint": {"label": "SuperPoint", "color": "#ec4899"},
    "xfeat": {"label": "XFeat", "color": "#22c55e"},
}

PARAM_SPECS = {
    "sift_contrast_threshold": {"type": "float", "default": 0.018},
    "sift_edge_threshold": {"type": "float", "default": 20.0},
    "orb_fast_threshold": {"type": "int", "default": 20},
    "orb_edge_threshold": {"type": "int", "default": 31},
    "superpoint_keypoint_threshold": {"type": "float", "default": 0.005},
    "superpoint_max_keypoints": {"type": "int", "default": 1024},
    "xfeat_detection_threshold": {"type": "float", "default": 0.05},
    "xfeat_top_k": {"type": "int", "default": 4096},
}


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AVT Point Extractors</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111513;
      --panel: #18201d;
      --panel-2: #202a26;
      --text: #eef6f1;
      --muted: #9cafaa;
      --line: #33413c;
      --accent: #66d9a6;
      --danger: #ff8a8a;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      height: 100vh;
      min-height: 680px;
      background: var(--bg);
      color: var(--text);
      letter-spacing: 0;
      overflow: hidden;
    }
    button, input {
      font: inherit;
    }
    button {
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      background: #121916;
      color: var(--text);
      cursor: pointer;
      white-space: nowrap;
    }
    button:hover { border-color: var(--accent); }
    input[type="range"] { width: 100%; }
    .app {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      height: 100vh;
      min-height: 680px;
    }
    header {
      display: grid;
      gap: 10px;
      padding: 12px 14px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }
    .topbar {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) auto;
      gap: 14px;
      align-items: center;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
      font-weight: 760;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      text-align: right;
    }
    .controls {
      display: grid;
      grid-template-columns: auto auto minmax(180px, 1fr) 90px auto auto;
      gap: 9px;
      align-items: center;
    }
    .frame-label, .small-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .frame-number {
      width: 90px;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #101613;
      color: var(--text);
      padding: 0 8px;
    }
    .params {
      display: grid;
      grid-template-columns: repeat(4, minmax(170px, 1fr));
      gap: 8px;
      align-items: stretch;
    }
    .param-group {
      display: grid;
      gap: 6px;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #121916;
    }
    .param-title {
      display: flex;
      align-items: center;
      gap: 7px;
      font-size: 12px;
      font-weight: 760;
    }
    .param-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 86px;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
      min-width: 0;
    }
    .param-row input {
      width: 86px;
      height: 28px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #0e1512;
      color: var(--text);
      padding: 0 7px;
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }
    .param-row input:focus {
      outline: 2px solid var(--accent);
      outline-offset: 1px;
    }
    main {
      position: relative;
      min-height: 0;
      overflow: hidden;
      background: #050706;
    }
    canvas {
      display: block;
      width: 100%;
      height: 100%;
    }
    footer {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
      padding: 10px 14px;
      background: var(--panel);
      border-top: 1px solid var(--line);
    }
    .legend {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      min-width: 0;
    }
    .method {
      display: inline-grid;
      grid-template-columns: auto auto auto auto;
      align-items: center;
      gap: 6px;
      min-height: 30px;
      padding: 0 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
      color: var(--text);
      font-size: 13px;
      user-select: none;
      cursor: pointer;
      transition: border-color .12s ease, opacity .12s ease, background .12s ease;
    }
    .method:hover { border-color: var(--accent); }
    .method.is-off {
      opacity: .58;
      background: #141b18;
    }
    .switch-input {
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }
    .switch {
      position: relative;
      width: 34px;
      height: 18px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #0e1512;
      transition: border-color .12s ease, background .12s ease;
    }
    .switch::after {
      content: "";
      position: absolute;
      top: 2px;
      left: 2px;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: var(--muted);
      transition: transform .12s ease, background .12s ease;
    }
    .switch-input:checked + .switch {
      border-color: var(--method-color);
      background: rgba(102,217,166,.14);
    }
    .switch-input:checked + .switch::after {
      transform: translateX(16px);
      background: var(--method-color);
    }
    .switch-input:focus-visible + .switch {
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }
    .swatch {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      box-shadow: 0 0 0 2px rgba(255,255,255,.12);
    }
    .count {
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }
    .stats {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    @media (max-width: 860px) {
      body, .app { min-height: 780px; }
      .topbar, .controls, .params, footer { grid-template-columns: 1fr; }
      .status { text-align: left; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="topbar">
        <h1>AVT Point Extractors</h1>
        <div id="status" class="status">Loading</div>
      </div>
      <div class="controls">
        <button id="prevBtn">&lt;</button>
        <button id="nextBtn">&gt;</button>
        <input id="frameSlider" type="range" min="0" max="0" value="0">
        <input id="frameNumber" class="frame-number" type="number" min="0" value="0">
        <span id="frameName" class="frame-label">frame</span>
        <label class="small-label">Point Size <input id="pointSize" type="range" min="1" max="8" value="3"></label>
      </div>
      <div id="params" class="params"></div>
    </header>
    <main>
      <canvas id="canvas"></canvas>
    </main>
    <footer>
      <div id="legend" class="legend"></div>
      <div id="stats" class="stats"></div>
    </footer>
  </div>
  <script>
    const COLORS = {
      sift: "#22d3ee",
      orb: "#f59e0b",
      superpoint: "#ec4899",
      xfeat: "#22c55e",
    };
    const LABELS = {
      sift: "SIFT",
      orb: "ORB",
      superpoint: "SuperPoint",
      xfeat: "XFeat",
    };
    const PARAM_GROUPS = [
      {
        key: "sift",
        title: "SIFT",
        params: [
          { key: "sift_contrast_threshold", label: "contrastThreshold", type: "float", value: 0.018, min: 0, step: 0.001 },
          { key: "sift_edge_threshold", label: "edgeThreshold", type: "float", value: 20.0, min: 0, step: 0.5 },
        ],
      },
      {
        key: "orb",
        title: "ORB",
        params: [
          { key: "orb_fast_threshold", label: "fastThreshold", type: "int", value: 20, min: 0, step: 1 },
          { key: "orb_edge_threshold", label: "edgeThreshold", type: "int", value: 31, min: 0, step: 1 },
        ],
      },
      {
        key: "superpoint",
        title: "SuperPoint",
        params: [
          { key: "superpoint_keypoint_threshold", label: "keypoint_threshold", type: "float", value: 0.005, min: 0, step: 0.001 },
          { key: "superpoint_max_keypoints", label: "max_keypoints", type: "int", value: 1024, min: -1, step: 1 },
        ],
      },
      {
        key: "xfeat",
        title: "XFeat",
        params: [
          { key: "xfeat_detection_threshold", label: "detection_threshold", type: "float", value: 0.05, min: 0, step: 0.01 },
          { key: "xfeat_top_k", label: "top_k", type: "int", value: 4096, min: 1, step: 1 },
        ],
      },
    ];
    const state = {
      frameCount: 0,
      index: 0,
      image: null,
      data: null,
      enabled: { sift: true, orb: true, superpoint: true, xfeat: true },
      params: {},
      pointSize: 3,
      fit: { x: 0, y: 0, scale: 1 },
      pending: null,
      reloadTimer: null,
    };
    const $ = (id) => document.getElementById(id);
    const canvas = $("canvas");
    const ctx = canvas.getContext("2d");

    function setStatus(text) {
      $("status").textContent = text;
    }

    async function init() {
      const res = await fetch("/api/frames");
      const data = await res.json();
      if (!data.ok) throw new Error(data.error);
      state.frameCount = data.count;
      $("frameSlider").max = String(Math.max(0, state.frameCount - 1));
      buildLegend();
      buildParams(data.params || {});
      wire();
      await loadFrame(0);
    }

    function buildLegend() {
      const wrap = $("legend");
      wrap.innerHTML = "";
      for (const key of Object.keys(COLORS)) {
        const label = document.createElement("label");
        label.className = "method is-on";
        label.style.setProperty("--method-color", COLORS[key]);
        label.title = LABELS[key];
        label.innerHTML = `<input class="switch-input" type="checkbox" role="switch" aria-label="${LABELS[key]}" aria-checked="true" checked data-method="${key}"><span class="switch"></span><span class="swatch" style="background:${COLORS[key]}"></span><span>${LABELS[key]}</span><span id="count-${key}" class="count">0</span>`;
        label.querySelector("input").addEventListener("change", (event) => {
          state.enabled[key] = event.target.checked;
          event.target.setAttribute("aria-checked", String(event.target.checked));
          label.classList.toggle("is-on", event.target.checked);
          label.classList.toggle("is-off", !event.target.checked);
          draw();
        });
        wrap.appendChild(label);
      }
    }

    function buildParams(defaults) {
      const wrap = $("params");
      wrap.innerHTML = "";
      for (const group of PARAM_GROUPS) {
        const section = document.createElement("section");
        section.className = "param-group";
        section.style.setProperty("--method-color", COLORS[group.key]);
        const title = document.createElement("div");
        title.className = "param-title";
        title.innerHTML = `<span class="swatch" style="background:${COLORS[group.key]}"></span><span>${group.title}</span>`;
        section.appendChild(title);
        for (const param of group.params) {
          const value = defaults[param.key] ?? param.value;
          state.params[param.key] = value;
          const row = document.createElement("label");
          row.className = "param-row";
          row.innerHTML = `<span>${param.label}</span><input id="param-${param.key}" type="number" value="${value}" min="${param.min}" step="${param.step}" data-key="${param.key}" data-type="${param.type}">`;
          const input = row.querySelector("input");
          input.addEventListener("input", () => {
            const next = param.type === "int" ? Math.trunc(Number(input.value)) : Number(input.value);
            if (!Number.isFinite(next)) return;
            state.params[param.key] = next;
            scheduleReload();
          });
          section.appendChild(row);
        }
        wrap.appendChild(section);
      }
    }

    function wire() {
      $("prevBtn").addEventListener("click", () => loadFrame(Math.max(0, state.index - 1)));
      $("nextBtn").addEventListener("click", () => loadFrame(Math.min(state.frameCount - 1, state.index + 1)));
      $("frameSlider").addEventListener("input", (e) => loadFrame(Number(e.target.value)));
      $("frameNumber").addEventListener("change", (e) => {
        const value = Math.max(0, Math.min(state.frameCount - 1, Number(e.target.value) || 0));
        loadFrame(value);
      });
      $("pointSize").addEventListener("input", (e) => {
        state.pointSize = Number(e.target.value);
        draw();
      });
      window.addEventListener("resize", draw);
    }

    async function loadFrame(index) {
      if (state.reloadTimer) {
        clearTimeout(state.reloadTimer);
        state.reloadTimer = null;
      }
      state.index = index;
      $("frameSlider").value = String(index);
      $("frameNumber").value = String(index);
      $("frameName").textContent = `frame ${index}`;
      state.data = null;
      for (const key of Object.keys(COLORS)) $("count-" + key).textContent = "…";
      setStatus("Detecting");
      const token = Symbol();
      state.pending = token;
      const imagePromise = loadImage(`/frame?index=${index}&t=${Date.now()}`);
      const detectPromise = fetch(`/api/detect?${detectQuery(index)}`).then((r) => r.json());
      const [img, data] = await Promise.all([imagePromise, detectPromise]);
      if (state.pending !== token) return;
      if (!data.ok) throw new Error(data.error);
      state.image = img;
      state.data = data;
      $("frameName").textContent = data.name;
      updateCounts(data);
      setStatus("Ready");
      draw();
    }

    function scheduleReload() {
      setStatus("Detecting");
      if (state.reloadTimer) clearTimeout(state.reloadTimer);
      state.reloadTimer = setTimeout(() => loadFrame(state.index), 250);
    }

    function detectQuery(index) {
      const params = new URLSearchParams();
      params.set("index", String(index));
      params.set("params", JSON.stringify(state.params));
      return params.toString();
    }

    function loadImage(src) {
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
      });
    }

    function updateCounts(data) {
      for (const key of Object.keys(COLORS)) {
        const item = data.methods[key];
        $("count-" + key).textContent = item.error ? "err" : String(item.count ?? item.points.length);
      }
      $("stats").textContent = `${data.width}x${data.height} · ${data.elapsed_ms.toFixed(1)} ms`;
    }

    function draw() {
      const rect = canvas.parentElement.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);
      if (!state.image) return;
      const scale = Math.min(rect.width / state.image.width, rect.height / state.image.height);
      const drawW = state.image.width * scale;
      const drawH = state.image.height * scale;
      const x0 = (rect.width - drawW) / 2;
      const y0 = (rect.height - drawH) / 2;
      state.fit = { x: x0, y: y0, scale };
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(state.image, x0, y0, drawW, drawH);
      if (!state.data) return;
      for (const key of Object.keys(COLORS)) {
        if (!state.enabled[key]) continue;
        const item = state.data.methods[key];
        if (!item || item.error) continue;
        drawPoints(item.points, COLORS[key], x0, y0, scale);
      }
    }

    function drawPoints(points, color, x0, y0, scale) {
      const r = state.pointSize;
      ctx.save();
      ctx.globalAlpha = 0.92;
      ctx.fillStyle = color;
      ctx.strokeStyle = "rgba(0,0,0,.65)";
      ctx.lineWidth = 1;
      for (const p of points) {
        const x = x0 + p.x * scale;
        const y = y0 + p.y * scale;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      ctx.restore();
    }

    init().catch((error) => setStatus(error.message || String(error)));
  </script>
</body>
</html>
"""


@dataclass
class DetectorServer:
    frames_root: Path
    frames: list[Path]
    query_config: QueryConfig = field(default_factory=QueryConfig)
    cache: dict[tuple[int, tuple[tuple[str, float | int], ...]], dict[str, Any]] = field(
        default_factory=dict
    )
    lock: threading.Lock = field(default_factory=threading.Lock)
    sift: Any = None
    orb: Any = None
    superpoint: Any = None
    xfeat_model: Any = None
    xfeat_device: str | None = None

    def detect(self, index: int, params: dict[str, float | int]) -> dict[str, Any]:
        cache_key = (index, canonical_params(params))
        with self.lock:
            if cache_key in self.cache:
                return self.cache[cache_key]
            result = self._detect_uncached(index, params)
            self.cache[cache_key] = result
            return result

    def _detect_uncached(self, index: int, params: dict[str, float | int]) -> dict[str, Any]:
        frame_path = self.frames[index]
        bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"could not read frame: {frame_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames_rgb = rgb[None]
        query_config = self._query_config_for(params)
        start = time.perf_counter()
        methods = {
            "sift": self._run_detector("sift", frames_rgb, query_config),
            "orb": self._run_detector("orb", frames_rgb, query_config),
            "superpoint": self._run_detector("superpoint", frames_rgb, query_config),
            "xfeat": self._run_xfeat(rgb, query_config),
        }
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "ok": True,
            "index": index,
            "name": frame_path.name,
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "elapsed_ms": elapsed_ms,
            "params": params,
            "methods": methods,
        }

    def _query_config_for(self, params: dict[str, float | int]) -> QueryConfig:
        base = self.query_config
        return replace(
            base,
            sampling=replace(
                base.sampling,
                contrast_threshold=float(params["sift_contrast_threshold"]),
                edge_threshold=float(params["sift_edge_threshold"]),
            ),
            orb=replace(
                base.orb,
                fast_threshold=int(params["orb_fast_threshold"]),
                edge_threshold=int(params["orb_edge_threshold"]),
            ),
            superpoint=replace(
                base.superpoint,
                keypoint_threshold=float(params["superpoint_keypoint_threshold"]),
                max_keypoints=int(params["superpoint_max_keypoints"]),
            ),
            xfeat=replace(
                base.xfeat,
                detection_threshold=float(params["xfeat_detection_threshold"]),
                top_k=int(params["xfeat_top_k"]),
            ),
        )

    def _run_detector(
        self, name: str, frames_rgb: np.ndarray, query_config: QueryConfig
    ) -> dict[str, Any]:
        try:
            start = time.perf_counter()
            detector = self._make_detector(name, query_config)
            keypoints = detector.detect(frames_rgb, 0, None)
            points = keypoints_to_json(keypoints)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return {
                "points": points,
                "count": len(points),
                "elapsed_ms": elapsed_ms,
                "error": "",
            }
        except Exception as exc:
            return {
                "points": [],
                "count": 0,
                "elapsed_ms": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _make_detector(self, name: str, query_config: QueryConfig):
        if name == "sift":
            return SiftDetector(query_config.sampling, query_config.sampling)
        if name == "orb":
            return OrbDetector(query_config.orb, query_config.sampling)
        if name == "superpoint":
            return SuperPointSuperGlueDetector(query_config.superpoint)
        raise ValueError(name)

    def _run_xfeat(self, rgb: np.ndarray, query_config: QueryConfig) -> dict[str, Any]:
        try:
            start = time.perf_counter()
            model, device = self._get_xfeat_model()
            cfg = query_config.xfeat
            output = model.detectAndCompute(
                np.ascontiguousarray(rgb),
                top_k=int(cfg.top_k),
                detection_threshold=float(cfg.detection_threshold),
            )[0]
            kpts = output["keypoints"].detach().cpu().numpy()
            scores = output.get("scores")
            scores = (
                scores.detach().cpu().numpy()
                if scores is not None
                else np.ones(len(kpts), dtype=np.float32)
            )
            keypoints = [
                make_keypoint(float(x), float(y), float(score))
                for (x, y), score in zip(kpts, scores)
            ]
            points = keypoints_to_json(filter_keypoints_by_mask(keypoints, None))
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return {
                "points": points,
                "count": len(points),
                "elapsed_ms": elapsed_ms,
                "error": "",
            }
        except Exception as exc:
            return {
                "points": [],
                "count": 0,
                "elapsed_ms": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _get_xfeat_model(self):
        if self.xfeat_model is not None:
            return self.xfeat_model, self.xfeat_device
        import torch

        cfg: XFeatConfig = self.query_config.xfeat
        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if XFEAT_LOCAL_REPO.exists():
            model = torch.hub.load(
                str(XFEAT_LOCAL_REPO),
                cfg.model,
                source="local",
                pretrained=(cfg.checkpoint is None),
                top_k=int(cfg.top_k),
            )
        else:
            model = torch.hub.load(
                cfg.hub_repo,
                cfg.model,
                pretrained=(cfg.checkpoint is None),
                top_k=int(cfg.top_k),
                skip_validation=True,
            )
        if cfg.checkpoint:
            state = torch.load(cfg.checkpoint, map_location="cpu")
            model.net.load_state_dict(state)
        try:
            model = model.to(device)
        except Exception:
            pass
        self.xfeat_model = model
        self.xfeat_device = device
        return model, device


STATE: DetectorServer | None = None


def keypoints_to_json(keypoints: list[cv2.KeyPoint]) -> list[dict[str, float]]:
    return [
        {
            "x": float(kp.pt[0]),
            "y": float(kp.pt[1]),
            "response": float(kp.response),
            "size": float(kp.size),
        }
        for kp in keypoints
    ]


def list_frames(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    frames = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]
    frames.sort(key=lambda p: p.name)
    if not frames:
        raise RuntimeError(f"no image frames found in {root}")
    return frames


def default_params(config: QueryConfig) -> dict[str, float | int]:
    return {
        "sift_contrast_threshold": float(config.sampling.contrast_threshold),
        "sift_edge_threshold": float(config.sampling.edge_threshold),
        "orb_fast_threshold": int(config.orb.fast_threshold),
        "orb_edge_threshold": int(config.orb.edge_threshold),
        "superpoint_keypoint_threshold": float(config.superpoint.keypoint_threshold),
        "superpoint_max_keypoints": int(config.superpoint.max_keypoints),
        "xfeat_detection_threshold": float(config.xfeat.detection_threshold),
        "xfeat_top_k": int(config.xfeat.top_k),
    }


def parse_params(query: str, config: QueryConfig) -> dict[str, float | int]:
    raw = parse_qs(query).get("params", ["{}"])[0]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid params JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("params must be a JSON object")

    defaults = default_params(config)
    parsed: dict[str, float | int] = {}
    for key, spec in PARAM_SPECS.items():
        value = data.get(key, defaults[key])
        if spec["type"] == "int":
            number = int(value)
            if key == "xfeat_top_k" and number < 1:
                raise ValueError("xfeat_top_k must be >= 1")
            if key == "superpoint_max_keypoints" and number < -1:
                raise ValueError("superpoint_max_keypoints must be >= -1")
            if key not in {"superpoint_max_keypoints"} and number < 0:
                raise ValueError(f"{key} must be >= 0")
        else:
            number = float(value)
            if not np.isfinite(number):
                raise ValueError(f"{key} must be finite")
            if number < 0:
                raise ValueError(f"{key} must be >= 0")
        parsed[key] = number
    return parsed


def canonical_params(params: dict[str, float | int]) -> tuple[tuple[str, float | int], ...]:
    return tuple((key, params[key]) for key in PARAM_SPECS)


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def bytes_response(handler: BaseHTTPRequestHandler, content_type: str, data: bytes) -> None:
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class Handler(BaseHTTPRequestHandler):
    server_version = "AVTDetectorVisualizer/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:  # noqa: N802
        assert STATE is not None
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                bytes_response(self, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
                return
            if parsed.path == "/api/frames":
                json_response(
                    self,
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "count": len(STATE.frames),
                        "first": STATE.frames[0].name,
                        "last": STATE.frames[-1].name,
                        "params": default_params(STATE.query_config),
                    },
                )
                return
            if parsed.path == "/api/detect":
                index = parse_index(parsed.query, len(STATE.frames))
                params = parse_params(parsed.query, STATE.query_config)
                json_response(self, HTTPStatus.OK, STATE.detect(index, params))
                return
            if parsed.path == "/frame":
                index = parse_index(parsed.query, len(STATE.frames))
                frame = STATE.frames[index]
                content_type = mimetypes.guess_type(frame.name)[0] or "application/octet-stream"
                bytes_response(self, content_type, frame.read_bytes())
                return
            json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
        except Exception as exc:
            json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def parse_index(query: str, frame_count: int) -> int:
    raw = parse_qs(query).get("index", ["0"])[0]
    index = int(raw)
    if not 0 <= index < frame_count:
        raise ValueError(f"frame index out of range: {index}")
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare AVT detector keypoints in a browser.")
    parser.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8891)
    return parser.parse_args()


def main() -> None:
    global STATE
    args = parse_args()
    frames_root = args.frames_root.expanduser().resolve()
    frames = list_frames(frames_root)
    STATE = DetectorServer(frames_root=frames_root, frames=frames)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"frames={len(frames)} root={frames_root}")
    print(f"listening=http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
