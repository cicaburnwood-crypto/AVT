from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...schema import QueryPoint, TrackerInfo, WindowSpec
from ..base import TrackingBundle
from .download import ensure_foundationpose_weights


@dataclass
class FoundationPoseBackend:
    """FoundationPose adapter for AVT point tracks.

    FoundationPose is a 6D object pose tracker, not an RGB-only point tracker.
    This backend therefore consumes FoundationPose-derived per-frame image
    transforms or direct point tracks and exposes them through AVT's
    `PointTracker` protocol.

    Supported transform files:
      - `.npz` with `tracks_reverse` and optional `visibility_reverse`
      - `.npz` with `homographies_reverse` or `homographies`, shape [T,3,3]
      - `.json` with `homographies_reverse` or `homographies`

    Homographies are absolute transforms from reverse frame 0 to each reverse
    frame. For a query inserted at time q, AVT applies H[t] @ inv(H[q]).
    """

    weights_dir: Path | None = None
    transforms_path: Path | None = None
    download_weights: bool = False
    device: str = "auto"
    max_reprojection_px: float | None = None
    _window: WindowSpec | None = field(default=None, init=False, repr=False)

    def set_window_context(self, *, window: WindowSpec, output_dir: Path) -> None:
        self._window = window

    def track(self, frames_rgb: np.ndarray, queries: list[QueryPoint]) -> TrackingBundle:
        if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
            raise ValueError("frames_rgb must have shape [T,H,W,3]")
        if not queries:
            raise ValueError("No query points were provided")

        weights_dir = ensure_foundationpose_weights(
            self.weights_dir,
            download=self.download_weights,
        )
        if self.transforms_path is None:
            raise RuntimeError(
                "FoundationPose backend needs pose-derived transforms for AVT point tracking. "
                "Provide --foundationpose-transforms with an .npz/.json containing "
                "`homographies_reverse` or direct `tracks_reverse`. The official "
                "FoundationPose model itself also requires RGB-D, masks, camera intrinsics, "
                "and CAD/reference object data before those transforms can be produced."
            )

        frame_count, height, width = frames_rgb.shape[:3]
        transforms_path = self._resolve_transforms_path()
        direct = _load_direct_tracks(transforms_path, frame_count, len(queries))
        if direct is not None:
            tracks, visibility = direct
        else:
            homographies = _load_homographies(transforms_path, frame_count)
            tracks, visibility = _track_queries_with_homographies(
                queries,
                homographies,
                frame_count=frame_count,
                width=width,
                height=height,
            )

        bundle = TrackingBundle(
            tracks=tracks.astype(np.float32),
            visibility=visibility.astype(bool),
            tracker=TrackerInfo(
                name="foundationpose",
                parameters={
                    "device": self.device,
                    "weights_dir": str(weights_dir),
                    "transforms_path": str(transforms_path),
                    "window": self._window.id if self._window else None,
                    "mode": "pose_transform_adapter",
                },
            ),
        )
        bundle.validate(frame_count, len(queries))
        return bundle

    def _resolve_transforms_path(self) -> Path:
        if self.transforms_path is None:  # pragma: no cover - guarded by track
            raise ValueError("transforms_path is required")
        path = self.transforms_path.expanduser().resolve()
        if path.is_file():
            return path
        if not path.is_dir():
            raise FileNotFoundError(path)
        if self._window is None:
            raise ValueError(
                "--foundationpose-transforms points to a directory, but no AVT window "
                "context is available. Use a single transform file or run through "
                "AVT's inverse pipeline."
            )
        candidates = (
            path / f"{self._window.id}.npz",
            path / f"{self._window.id}.json",
            path / self._window.id / "foundationpose_transforms.npz",
            path / self._window.id / "foundationpose_transforms.json",
            path / self._window.id / "transforms.npz",
            path / self._window.id / "transforms.json",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        names = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            f"No FoundationPose transform file found for {self._window.id}. Tried:\n{names}"
        )


def _load_direct_tracks(
    path: Path,
    frame_count: int,
    query_count: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if path.suffix.lower() != ".npz":
        return None
    with np.load(path) as arrays:
        if "tracks_reverse" not in arrays:
            return None
        tracks = arrays["tracks_reverse"].astype(np.float32)
        if tracks.shape != (frame_count, query_count, 2):
            raise ValueError(
                f"tracks_reverse shape {tracks.shape} does not match "
                f"({frame_count}, {query_count}, 2)"
            )
        if "visibility_reverse" in arrays:
            visibility = arrays["visibility_reverse"].astype(bool)
        else:
            visibility = np.isfinite(tracks).all(axis=2)
        return tracks, visibility


def _load_homographies(path: Path, frame_count: int) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as arrays:
            for key in ("homographies_reverse", "homographies"):
                if key in arrays:
                    return _validate_homographies(arrays[key], frame_count)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("FoundationPose transform JSON must be a mapping")
        for key in ("homographies_reverse", "homographies"):
            if key in data:
                return _validate_homographies(np.array(data[key], dtype=np.float32), frame_count)
    raise ValueError(
        f"{path} does not contain supported FoundationPose transforms. "
        "Expected `tracks_reverse`, `homographies_reverse`, or `homographies`."
    )


def _validate_homographies(homographies: np.ndarray, frame_count: int) -> np.ndarray:
    homographies = np.asarray(homographies, dtype=np.float32)
    if homographies.shape != (frame_count, 3, 3):
        raise ValueError(
            f"homography shape {homographies.shape} does not match ({frame_count}, 3, 3)"
        )
    return homographies


def _track_queries_with_homographies(
    queries: list[QueryPoint],
    homographies: np.ndarray,
    *,
    frame_count: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    tracks = np.full((frame_count, len(queries), 2), np.nan, dtype=np.float32)
    visibility = np.zeros((frame_count, len(queries)), dtype=bool)
    inverses = np.linalg.inv(homographies.astype(np.float64))

    for query in queries:
        if not 0 <= query.reverse_time < frame_count:
            raise ValueError(f"query reverse_time outside window: {query}")
        start_inv = inverses[query.reverse_time]
        point = np.array([query.x, query.y, 1.0], dtype=np.float64)
        for t in range(query.reverse_time, frame_count):
            mapped = homographies[t].astype(np.float64) @ start_inv @ point
            if abs(mapped[2]) < 1e-8:
                continue
            x = float(mapped[0] / mapped[2])
            y = float(mapped[1] / mapped[2])
            if 0 <= x < width and 0 <= y < height:
                tracks[t, query.id] = [x, y]
                visibility[t, query.id] = True
    return tracks, visibility
