from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .io import load_frame_window, write_mp4
from .schema import FrameRecord, QueryPoint, WindowSpec
from .tracking.base import PointTracker, TrackingBundle


@dataclass
class InverseTrackConfig:
    window_size: int = 250
    window_step: int = 100
    fps: float = 10.0
    query_stride: int = 10
    seed_count: int = 17
    seed_y_ratio: float = 0.92
    seed_x_min_ratio: float = 0.40
    seed_x_max_ratio: float = 0.60
    max_windows: int | None = None
    save_reverse_video: bool = True


def build_windows(frame_count: int, config: InverseTrackConfig) -> list[WindowSpec]:
    if config.window_size <= 1:
        raise ValueError("window_size must be greater than 1")
    if config.window_step <= 0:
        raise ValueError("window_step must be positive")
    windows = [
        WindowSpec(start=start, end=min(start + config.window_size, frame_count))
        for start in range(0, frame_count, config.window_step)
        if start + 2 <= frame_count
    ]
    windows = [win for win in windows if win.frame_count >= 2]
    if config.max_windows is not None:
        windows = windows[: config.max_windows]
    return windows


def build_queries(width: int, height: int, frame_count: int, config: InverseTrackConfig) -> list[QueryPoint]:
    if config.query_stride <= 0:
        raise ValueError("query_stride must be positive")
    if config.seed_count <= 0:
        raise ValueError("seed_count must be positive")
    xs = np.linspace(
        config.seed_x_min_ratio * (width - 1),
        config.seed_x_max_ratio * (width - 1),
        config.seed_count,
        dtype=np.float32,
    )
    y = float(config.seed_y_ratio * (height - 1))
    queries: list[QueryPoint] = []
    middle = (config.seed_count - 1) / 2.0
    for reverse_time in range(0, frame_count, config.query_stride):
        for i, x in enumerate(xs):
            side = -1 if i <= middle else 1
            queries.append(
                QueryPoint(
                    id=len(queries),
                    reverse_time=int(reverse_time),
                    x=float(x),
                    y=y,
                    side=side,
                )
            )
    return queries


def reference_mask(bundle: TrackingBundle, height: int, width: int) -> np.ndarray:
    """Cyan RGBA mask on the original reference frame for a window."""

    points = bundle.tracks[-1, bundle.visibility[-1]]
    points = points[np.isfinite(points).all(axis=1)]
    alpha = np.zeros((height, width), dtype=np.uint8)
    if len(points) >= 3:
        hull = cv2.convexHull(np.round(points).astype(np.int32))
        cv2.fillConvexPoly(alpha, hull, 92)
    elif len(points):
        for x, y in points:
            cv2.circle(alpha, (int(round(x)), int(round(y))), 8, 92, -1)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = 32
    rgba[..., 1] = 199
    rgba[..., 2] = 230
    rgba[..., 3] = alpha
    return rgba


def write_window_artifacts(
    output_dir: Path,
    window: WindowSpec,
    frames_rgb: np.ndarray,
    queries: list[QueryPoint],
    bundle: TrackingBundle,
    config: InverseTrackConfig,
) -> None:
    window_dir = output_dir / "windows" / window.id
    window_dir.mkdir(parents=True, exist_ok=True)
    h, w = frames_rgb.shape[1:3]

    query_array = np.array(
        [[q.id, q.reverse_time, q.x, q.y, q.side] for q in queries],
        dtype=np.float32,
    )
    np.savez_compressed(
        window_dir / "tracks.npz",
        tracks_reverse=bundle.tracks.astype(np.float32),
        visibility_reverse=bundle.visibility.astype(bool),
        queries=query_array,
    )

    mask_rgba = reference_mask(bundle, h, w)
    cv2.imwrite(str(window_dir / "path_mask_reference.png"), cv2.cvtColor(mask_rgba, cv2.COLOR_RGBA2BGRA))

    if config.save_reverse_video:
        write_mp4(window_dir / "reverse_video.mp4", frames_rgb[::-1].copy(), config.fps)

    metadata = {
        "id": window.id,
        "seq_start": window.start,
        "seq_end": window.end,
        "frame_count": window.frame_count,
        "width": int(w),
        "height": int(h),
        "fps": float(config.fps),
        "query_count": len(queries),
        "tracker": bundle.tracker.to_json(),
        "config": asdict(config),
        "files": {
            "tracks": "tracks.npz",
            "mask": "path_mask_reference.png",
            "reverse_video": "reverse_video.mp4" if config.save_reverse_video else None,
        },
        "time_order": {
            "tracks_reverse": "frame 0 is original seq_end - 1",
            "viewer_frames": "source RGB frames are original chronological order",
        },
    }
    (window_dir / "window.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_inverse_tracking(
    source_root: Path,
    frame_records: list[FrameRecord],
    output_dir: Path,
    tracker: PointTracker,
    config: InverseTrackConfig,
) -> list[WindowSpec]:
    source_root = source_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    windows = build_windows(len(frame_records), config)
    run_metadata = {
        "source_root": str(source_root),
        "frame_count": len(frame_records),
        "config": asdict(config),
        "windows": [asdict(win) | {"id": win.id} for win in windows],
    }
    (output_dir / "run.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    for ordinal, window in enumerate(windows, start=1):
        print(f"[{ordinal}/{len(windows)}] tracking {window.id}", flush=True)
        frames = load_frame_window(source_root, frame_records, window.start, window.end)
        frames_reverse = frames[::-1].copy()
        h, w = frames.shape[1:3]
        queries = build_queries(w, h, len(frames), config)
        bundle = tracker.track(frames_reverse, queries)
        bundle.validate(len(frames), len(queries))
        write_window_artifacts(output_dir, window, frames, queries, bundle, config)
    return windows
