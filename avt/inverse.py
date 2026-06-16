from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .io import load_frame_window, write_mp4
from .querying import (
    QueryConfig,
    align_virtual_robot_to_image,
    build_avt_queries,
    build_sift_queries,
    query_artifact_arrays,
    query_capture_metadata,
)
from .schema import FrameRecord, QueryPoint, WindowSpec
from .tracking.base import PointTracker, TrackingBundle


@dataclass
class InverseTrackConfig:
    window_size: int = 250
    window_step: int = 100
    fps: float = 10.0
    query_stride: int = 10
    seed_count: int = 17
    seed_y_ratio: float | None = None
    seed_x_min_ratio: float | None = None
    seed_x_max_ratio: float | None = None
    query_config: QueryConfig = field(default_factory=QueryConfig)
    max_windows: int | None = None
    save_reverse_video: bool = True


def _avt_seed_ratios(config: InverseTrackConfig, width: int, height: int) -> tuple[float, float, float]:
    alignment = align_virtual_robot_to_image(
        height=height,
        width=width,
        robot=config.query_config.robot,
    )
    return (
        alignment.seed_y_ratio if config.seed_y_ratio is None else config.seed_y_ratio,
        alignment.seed_x_min_ratio if config.seed_x_min_ratio is None else config.seed_x_min_ratio,
        alignment.seed_x_max_ratio if config.seed_x_max_ratio is None else config.seed_x_max_ratio,
    )


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


def build_queries(
    width: int,
    height: int,
    frame_count: int,
    config: InverseTrackConfig,
    frames_rgb: np.ndarray | None = None,
) -> list[QueryPoint]:
    queries: list[QueryPoint] = []
    mode = config.query_config.mode
    seed_y_ratio, seed_x_min_ratio, seed_x_max_ratio = _avt_seed_ratios(config, width, height)
    want_sift = mode in {"sift", "avt+sift"} or config.query_config.sift.enabled

    if want_sift:
        if frames_rgb is None:
            raise ValueError("frames_rgb is required for SIFT query capture")
        queries.extend(
            build_sift_queries(
                frames_rgb=frames_rgb,
                query_config=config.query_config,
                start_id=len(queries),
            )
        )

    if mode == "avt":
        queries.extend(
            build_avt_queries(
                width=width,
                height=height,
                frame_count=frame_count,
                query_stride=config.query_stride,
                seed_count=config.seed_count,
                seed_y_ratio=seed_y_ratio,
                seed_x_min_ratio=seed_x_min_ratio,
                seed_x_max_ratio=seed_x_max_ratio,
                start_id=len(queries),
            )
        )
    elif mode == "avt+sift":
        shortage = config.query_config.sift.max_query_points - len(queries)
        if shortage > 0:
            queries.extend(
                build_avt_queries(
                    width=width,
                    height=height,
                    frame_count=frame_count,
                    query_stride=config.query_stride,
                    seed_count=config.seed_count,
                    seed_y_ratio=seed_y_ratio,
                    seed_x_min_ratio=seed_x_min_ratio,
                    seed_x_max_ratio=seed_x_max_ratio,
                    start_id=len(queries),
                    max_points=shortage,
                )
            )

    if not queries and mode == "sift":
        raise ValueError("No SIFT query points were generated")

    if not queries:
        raise ValueError("No query points were generated")
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

    query_arrays = query_artifact_arrays(queries)
    np.savez_compressed(
        window_dir / "tracks.npz",
        tracks_reverse=bundle.tracks.astype(np.float32),
        visibility_reverse=bundle.visibility.astype(bool),
        **query_arrays,
    )

    mask_rgba = reference_mask(bundle, h, w)
    cv2.imwrite(str(window_dir / "path_mask_reference.png"), cv2.cvtColor(mask_rgba, cv2.COLOR_RGBA2BGRA))

    if config.save_reverse_video:
        write_mp4(window_dir / "reverse_video.mp4", frames_rgb[::-1].copy(), config.fps)

    seed_y_ratio, seed_x_min_ratio, seed_x_max_ratio = _avt_seed_ratios(config, w, h)
    metadata = {
        "id": window.id,
        "seq_start": window.start,
        "seq_end": window.end,
        "frame_count": window.frame_count,
        "width": int(w),
        "height": int(h),
        "fps": float(config.fps),
        "query_count": len(queries),
        "query_capture": query_capture_metadata(config.query_config, width=w, height=h),
        "resolved_avt_seed": {
            "y_ratio": seed_y_ratio,
            "x_min_ratio": seed_x_min_ratio,
            "x_max_ratio": seed_x_max_ratio,
        },
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
        queries = build_queries(w, h, len(frames), config, frames_rgb=frames_reverse)
        bundle = tracker.track(frames_reverse, queries)
        bundle.validate(len(frames), len(queries))
        write_window_artifacts(output_dir, window, frames, queries, bundle, config)
    return windows
