"""Stage 4 - combine.

Combines tracked trajectories into the per-window outputs: the convex-hull
reference mask, optional SIFT support points, and the serialized ``tracks.npz``
/ ``window.json`` artifacts. Behavior and file formats are unchanged.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from ..config import InverseTrackConfig
from ..io import write_mp4
from ..querying import (
    query_artifact_arrays,
    query_capture_metadata,
    robot_sift_mask,
)
from ..schema import QueryPoint, WindowSpec
from ..tracking.base import TrackingBundle
from .preprocess import _avt_seed_ratios


def reference_mask(
    bundle: TrackingBundle,
    height: int,
    width: int,
    queries: list[QueryPoint] | None = None,
    support_points: np.ndarray | None = None,
) -> np.ndarray:
    """Cyan RGBA mask on the original reference frame for a window."""

    if queries is None:
        point_indices = np.arange(bundle.tracks.shape[1])
    else:
        point_indices = np.array(
            [query.id for query in queries if query.source != "sift_anchor"],
            dtype=np.int64,
        )
    if point_indices.size:
        visible = bundle.visibility[-1, point_indices]
        points = bundle.tracks[-1, point_indices][visible]
    else:
        points = np.empty((0, 2), dtype=np.float32)
    if support_points is not None and support_points.size:
        support = np.asarray(support_points, dtype=np.float32).reshape(-1, 2)
        points = np.concatenate([points, support], axis=0)
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


def _reference_support_points(frame_rgb: np.ndarray, config: InverseTrackConfig) -> np.ndarray:
    if not config.path_support_enabled:
        return np.empty((0, 2), dtype=np.float32)
    if config.query_config.mode not in {"ventura", "sift", "avt+sift"}:
        return np.empty((0, 2), dtype=np.float32)
    if config.path_support_min_points <= 0:
        return np.empty((0, 2), dtype=np.float32)

    h, w = frame_rgb.shape[:2]
    max_query_points = max(1, int(config.query_config.sift.max_query_points))
    fraction = max(1, int(config.path_support_fraction))
    count = max(int(config.path_support_min_points), max_query_points // fraction)
    mask = robot_sift_mask(h, w, config.query_config.robot, config.query_config.sift)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    if config.query_config.sift.use_clahe:
        tile = max(1, int(config.query_config.sift.clahe_tile_grid_size))
        clahe = cv2.createCLAHE(
            clipLimit=float(config.query_config.sift.clahe_clip_limit),
            tileGridSize=(tile, tile),
        )
        gray = clahe.apply(gray)

    sift = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=4,
        contrastThreshold=0.006,
        edgeThreshold=24,
        sigma=1.2,
    )
    keypoints, _ = sift.detectAndCompute(gray, mask)
    if not keypoints:
        return np.empty((0, 2), dtype=np.float32)
    picked = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:count]
    return np.array([kp.pt for kp in picked], dtype=np.float32)


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
    track_arrays = {
        "tracks_reverse": bundle.tracks.astype(np.float32),
        "visibility_reverse": bundle.visibility.astype(bool),
    }
    if bundle.confidence is not None:
        track_arrays["confidence_reverse"] = bundle.confidence.astype(np.float32)
    for name, component in bundle.confidence_components.items():
        track_arrays[f"{name}_reverse"] = component.astype(np.float32)
    np.savez_compressed(
        window_dir / "tracks.npz",
        **track_arrays,
        **query_arrays,
    )

    support_points = np.empty((0, 2), dtype=np.float32)
    if config.save_path_mask:
        support_points = _reference_support_points(frames_rgb[0], config)
        mask_rgba = reference_mask(bundle, h, w, queries, support_points=support_points)
        cv2.imwrite(
            str(window_dir / "path_mask_reference.png"),
            cv2.cvtColor(mask_rgba, cv2.COLOR_RGBA2BGRA),
        )

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
        "resolved_pct_seed": {
            "y_ratio": seed_y_ratio,
            "x_min_ratio": seed_x_min_ratio,
            "x_max_ratio": seed_x_max_ratio,
        },
        "tracker": bundle.tracker.to_json(),
        "config": asdict(config),
        "path_support": {
            "enabled": bool(config.path_support_enabled),
            "point_count": int(len(support_points)),
            "min_points": int(config.path_support_min_points),
            "fraction": int(config.path_support_fraction),
            "sift": {
                "contrast_threshold": 0.006,
                "edge_threshold": 24.0,
                "n_octave_layers": 4,
                "sigma": 1.2,
            },
        },
        "files": {
            "tracks": "tracks.npz",
            "mask": "path_mask_reference.png" if config.save_path_mask else None,
            "reverse_video": "reverse_video.mp4" if config.save_reverse_video else None,
        },
        "time_order": {
            "tracks_reverse": "frame 0 is original seq_end - 1",
            "viewer_frames": "source RGB frames are original chronological order",
        },
    }
    (window_dir / "window.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
