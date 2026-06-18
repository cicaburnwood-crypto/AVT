from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..io import load_frame_window
from ..schema import FrameRecord, QueryPoint, TrackerInfo, WindowSpec
from .base import TrackingBundle
from .cotracker import CoTrackerBackend


REGION_CHOICES = ("full", "bottom-third", "bottom-half")
CACHE_SCHEMA = "avt_cotracker_cache_v1"


@dataclass
class CoTrackerCacheConfig:
    frame_start: int = 0
    frame_count: int | None = None
    grid_stride: int = 8
    region: str = "full"
    query_mode: str = "last-frame"
    query_frame_stride: int = 1
    max_query_points: int = 0
    device: str = "auto"
    batch_size: int = 256
    torch_home: str | None = None
    hub_repo: str = "facebookresearch/co-tracker"
    hub_model: str = "cotracker3_offline"
    visibility_threshold: float = 0.9


def _region_bounds(height: int, width: int, region: str) -> tuple[int, int, int, int]:
    if region == "full":
        return 0, height, 0, width
    if region == "bottom-third":
        return height * 2 // 3, height, 0, width
    if region == "bottom-half":
        return height // 2, height, 0, width
    raise ValueError(f"Unknown cache region {region!r}; expected one of {REGION_CHOICES}")


def _grid_xy(height: int, width: int, *, stride: int, region: str) -> np.ndarray:
    if stride <= 0:
        raise ValueError("grid_stride must be positive")
    y0, y1, x0, x1 = _region_bounds(height, width, region)
    xs = np.arange(x0, x1, stride, dtype=np.float32)
    ys = np.arange(y0, y1, stride, dtype=np.float32)
    if xs.size == 0 or ys.size == 0:
        raise ValueError(f"Region {region!r} produced an empty grid")
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1).astype(np.float32)


def _cache_queries(
    *,
    frame_count: int,
    height: int,
    width: int,
    config: CoTrackerCacheConfig,
) -> list[QueryPoint]:
    grid = _grid_xy(height, width, stride=config.grid_stride, region=config.region)
    if config.query_frame_stride <= 0:
        raise ValueError("query_frame_stride must be positive")

    if config.query_mode == "last-frame":
        reverse_times = [0]
    elif config.query_mode == "every-frame":
        reverse_times = list(range(0, frame_count, config.query_frame_stride))
    else:
        raise ValueError("query_mode must be 'last-frame' or 'every-frame'")

    queries: list[QueryPoint] = []
    for reverse_time in reverse_times:
        for x, y in grid:
            queries.append(
                QueryPoint(
                    id=len(queries),
                    reverse_time=int(reverse_time),
                    x=float(x),
                    y=float(y),
                    side=0,
                    source="cotracker_cache_grid",
                )
            )
            if config.max_query_points > 0 and len(queries) >= config.max_query_points:
                return queries
    return queries


def _write_array(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return path.name


def build_cotracker_cache(
    *,
    source_root: Path,
    frame_records: list[FrameRecord],
    output_dir: Path,
    config: CoTrackerCacheConfig,
) -> dict[str, Any]:
    """Run CoTracker once and write a reusable dense/global track cache."""

    source_root = source_root.resolve()
    if config.frame_start < 0:
        raise ValueError("frame_start must be non-negative")
    frame_end = (
        min(len(frame_records), config.frame_start + config.frame_count)
        if config.frame_count is not None
        else len(frame_records)
    )
    if frame_end - config.frame_start < 2:
        raise ValueError("CoTracker cache needs at least two frames")

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_frame_window(source_root, frame_records, config.frame_start, frame_end)
    frames_reverse = frames[::-1].copy()
    height, width = frames.shape[1:3]
    queries = _cache_queries(
        frame_count=len(frames),
        height=height,
        width=width,
        config=config,
    )
    if not queries:
        raise ValueError("No cache query points were generated")

    tracker = CoTrackerBackend(
        device=config.device,
        batch_size=config.batch_size,
        torch_home=config.torch_home,
        hub_repo=config.hub_repo,
        hub_model=config.hub_model,
        visibility_threshold=config.visibility_threshold,
    )
    bundle = tracker.track(frames_reverse, queries)
    bundle.validate(len(frames), len(queries))

    point_ids = np.arange(len(queries), dtype=np.int64)
    birth_reverse_times = np.array([query.reverse_time for query in queries], dtype=np.int32)
    seed_xy = np.array([[query.x, query.y] for query in queries], dtype=np.float32)
    source_birth_frames = np.array(
        [frame_end - 1 - query.reverse_time for query in queries],
        dtype=np.int32,
    )

    arrays = {
        "tracks": _write_array(output_dir / "tracks_reverse.npy", bundle.tracks.astype(np.float32)),
        "visibility": _write_array(output_dir / "visibility_reverse.npy", bundle.visibility.astype(bool)),
        "point_ids": _write_array(output_dir / "point_ids.npy", point_ids),
        "birth_reverse_times": _write_array(output_dir / "birth_reverse_times.npy", birth_reverse_times),
        "source_birth_frames": _write_array(output_dir / "source_birth_frames.npy", source_birth_frames),
        "seed_xy": _write_array(output_dir / "seed_xy.npy", seed_xy),
    }
    if bundle.confidence is not None:
        arrays["confidence"] = _write_array(
            output_dir / "confidence_reverse.npy",
            bundle.confidence.astype(np.float32),
        )
    for name, component in bundle.confidence_components.items():
        arrays[f"confidence_component_{name}"] = _write_array(
            output_dir / f"{name}_reverse.npy",
            component.astype(np.float32),
        )

    metadata = {
        "schema": CACHE_SCHEMA,
        "source_root": str(source_root),
        "frame_start": int(config.frame_start),
        "frame_end": int(frame_end),
        "frame_count": int(frame_end - config.frame_start),
        "width": int(width),
        "height": int(height),
        "point_count": int(len(queries)),
        "config": asdict(config),
        "tracker": bundle.tracker.to_json(),
        "arrays": arrays,
        "time_order": {
            "tracks_reverse": "cache frame 0 is source frame frame_end - 1",
            "source_frame_for_reverse_t": "source_frame = frame_end - 1 - reverse_t",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


class CoTrackerCache:
    def __init__(self, path: Path, *, mmap_mode: str | None = "r") -> None:
        root = path if path.is_dir() else path.parent
        metadata_path = root / "metadata.json" if path.is_dir() else path
        self.root = root.resolve()
        self.metadata_path = metadata_path.resolve()
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema") != CACHE_SCHEMA:
            raise ValueError(f"Unsupported CoTracker cache schema: {self.metadata.get('schema')}")
        arrays = self.metadata["arrays"]
        self.tracks = np.load(self.root / arrays["tracks"], mmap_mode=mmap_mode)
        self.visibility = np.load(self.root / arrays["visibility"], mmap_mode=mmap_mode)
        self.point_ids = np.load(self.root / arrays["point_ids"], mmap_mode=mmap_mode)
        self.birth_reverse_times = np.load(
            self.root / arrays["birth_reverse_times"],
            mmap_mode=mmap_mode,
        )
        self.source_birth_frames = np.load(
            self.root / arrays["source_birth_frames"],
            mmap_mode=mmap_mode,
        )
        self.seed_xy = np.load(self.root / arrays["seed_xy"], mmap_mode=mmap_mode)
        self.confidence = (
            np.load(self.root / arrays["confidence"], mmap_mode=mmap_mode)
            if "confidence" in arrays
            else None
        )
        self.confidence_components = {
            key.removeprefix("confidence_component_"): np.load(self.root / filename, mmap_mode=mmap_mode)
            for key, filename in arrays.items()
            if key.startswith("confidence_component_")
        }
        self.frame_start = int(self.metadata["frame_start"])
        self.frame_end = int(self.metadata["frame_end"])
        self.frame_count = int(self.metadata["frame_count"])

    def reverse_index_for_source_frame(self, source_frame: int) -> int:
        reverse_t = self.frame_end - 1 - int(source_frame)
        if reverse_t < 0 or reverse_t >= self.frame_count:
            raise IndexError(
                f"source frame {source_frame} is outside cache range "
                f"[{self.frame_start}, {self.frame_end})"
            )
        return reverse_t


@dataclass
class CachedCoTrackerBackend:
    cache_path: Path
    max_match_distance: float = 12.0
    mmap_mode: str | None = "r"

    def __post_init__(self) -> None:
        self.cache = CoTrackerCache(self.cache_path, mmap_mode=self.mmap_mode)
        self.window: WindowSpec | None = None
        self.output_dir: Path | None = None

    def set_window_context(self, window: WindowSpec, output_dir: Path) -> None:
        self.window = window
        self.output_dir = output_dir

    def _visible_candidates(self, cache_reverse_t: int) -> tuple[np.ndarray, np.ndarray]:
        visible = np.asarray(self.cache.visibility[cache_reverse_t]).astype(bool)
        if not visible.any():
            return np.empty((0,), dtype=np.int64), np.empty((0, 2), dtype=np.float32)
        points = np.asarray(self.cache.tracks[cache_reverse_t], dtype=np.float32)
        finite = np.isfinite(points).all(axis=1)
        candidates = np.flatnonzero(visible & finite)
        if candidates.size == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0, 2), dtype=np.float32)
        return candidates, points[candidates]

    def _match_query_group(
        self,
        queries: list[QueryPoint],
        query_indices: np.ndarray,
        cache_reverse_t: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        matched = np.full(query_indices.size, -1, dtype=np.int64)
        distances = np.full(query_indices.size, np.inf, dtype=np.float32)
        candidates, points = self._visible_candidates(cache_reverse_t)
        if candidates.size == 0:
            return matched, distances

        query_xy = np.array(
            [[queries[int(idx)].x, queries[int(idx)].y] for idx in query_indices],
            dtype=np.float32,
        )
        try:
            from scipy.spatial import cKDTree
        except Exception:
            for out_idx, xy in enumerate(query_xy):
                delta = points - xy
                distances_sq = np.einsum("ij,ij->i", delta, delta)
                best_pos = int(np.argmin(distances_sq))
                distance = float(np.sqrt(distances_sq[best_pos]))
                if distance <= float(self.max_match_distance):
                    matched[out_idx] = int(candidates[best_pos])
                distances[out_idx] = distance
            return matched, distances

        tree = cKDTree(points)
        distance_values, positions = tree.query(
            query_xy,
            k=1,
            distance_upper_bound=float(self.max_match_distance),
            workers=-1,
        )
        ok = np.isfinite(distance_values) & (positions < candidates.size)
        matched[ok] = candidates[positions[ok]].astype(np.int64)
        distances[ok] = distance_values[ok].astype(np.float32)
        return matched, distances

    def track(self, frames_rgb: np.ndarray, queries: list[QueryPoint]) -> TrackingBundle:
        if self.window is None:
            raise RuntimeError("CachedCoTrackerBackend needs set_window_context before track().")
        if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
            raise ValueError("frames_rgb must have shape [T,H,W,3]")
        if not queries:
            raise ValueError("No query points were provided")

        frame_count = len(frames_rgb)
        cache_start_reverse = self.cache.reverse_index_for_source_frame(self.window.end - 1)
        cache_indices = np.arange(cache_start_reverse, cache_start_reverse + frame_count, dtype=np.int64)
        if cache_indices[-1] >= self.cache.frame_count:
            raise IndexError(
                f"Window {self.window.id} extends outside cache reverse range "
                f"at {int(cache_indices[-1])} >= {self.cache.frame_count}"
            )

        query_cache_indices = np.array(
            [
                self.cache.reverse_index_for_source_frame(self.window.end - 1 - query.reverse_time)
                for query in queries
            ],
            dtype=np.int32,
        )
        source_query_frames = np.array(
            [self.window.end - 1 - query.reverse_time for query in queries],
            dtype=np.int32,
        )
        matched = np.full(len(queries), -1, dtype=np.int64)
        distances = np.full(len(queries), np.inf, dtype=np.float32)
        for cache_reverse_t in sorted(set(int(value) for value in query_cache_indices)):
            query_indices = np.flatnonzero(query_cache_indices == cache_reverse_t)
            group_matches, group_distances = self._match_query_group(
                queries,
                query_indices,
                cache_reverse_t,
            )
            matched[query_indices] = group_matches
            distances[query_indices] = group_distances

        tracks = np.full((frame_count, len(queries), 2), np.nan, dtype=np.float32)
        visibility = np.zeros((frame_count, len(queries)), dtype=bool)
        confidence = (
            np.zeros((frame_count, len(queries)), dtype=np.float32)
            if self.cache.confidence is not None
            else None
        )
        confidence_components = {
            name: np.zeros((frame_count, len(queries)), dtype=np.float32)
            for name in self.cache.confidence_components
        }

        valid_query_indices = np.flatnonzero(matched >= 0)
        if valid_query_indices.size:
            cache_point_indices = matched[valid_query_indices]
            tracks[:, valid_query_indices] = np.asarray(
                self.cache.tracks[np.ix_(cache_indices, cache_point_indices)],
                dtype=np.float32,
            )
            visibility[:, valid_query_indices] = np.asarray(
                self.cache.visibility[np.ix_(cache_indices, cache_point_indices)],
                dtype=bool,
            )
            if confidence is not None and self.cache.confidence is not None:
                confidence[:, valid_query_indices] = np.asarray(
                    self.cache.confidence[np.ix_(cache_indices, cache_point_indices)],
                    dtype=np.float32,
                )
            for name, component in self.cache.confidence_components.items():
                confidence_components[name][:, valid_query_indices] = np.asarray(
                    component[np.ix_(cache_indices, cache_point_indices)],
                    dtype=np.float32,
                )

        for idx, query in enumerate(queries):
            if query.reverse_time > 0:
                visibility[: query.reverse_time, idx] = False
                tracks[: query.reverse_time, idx] = np.nan
                if confidence is not None:
                    confidence[: query.reverse_time, idx] = 0.0
                for component in confidence_components.values():
                    component[: query.reverse_time, idx] = 0.0

        birth_times = np.full(len(queries), -1, dtype=np.int32)
        source_birth_frames = np.full(len(queries), -1, dtype=np.int32)
        ok = matched >= 0
        if ok.any():
            birth_times[ok] = np.asarray(self.cache.birth_reverse_times[matched[ok]], dtype=np.int32)
            source_birth_frames[ok] = np.asarray(self.cache.source_birth_frames[matched[ok]], dtype=np.int32)

        bundle = TrackingBundle(
            tracks=tracks,
            visibility=visibility,
            confidence=confidence,
            confidence_components=confidence_components,
            extra_arrays={
                "cache_point_ids": matched.astype(np.int64),
                "cache_query_reverse_times": query_cache_indices,
                "cache_query_source_frames": source_query_frames,
                "cache_birth_reverse_times": birth_times,
                "cache_birth_source_frames": source_birth_frames,
                "cache_match_distances": distances.astype(np.float32),
            },
            extra_metadata={
                "cache_reference": {
                    "schema": CACHE_SCHEMA,
                    "path": str(self.cache.root),
                    "metadata": str(self.cache.metadata_path),
                    "frame_start": self.cache.frame_start,
                    "frame_end": self.cache.frame_end,
                    "max_match_distance": float(self.max_match_distance),
                    "matched_queries": int(ok.sum()),
                    "query_count": int(len(queries)),
                }
            },
            tracker=TrackerInfo(
                name="cotracker_cache",
                parameters={
                    "cache_path": str(self.cache.root),
                    "max_match_distance": float(self.max_match_distance),
                    "source_tracker": self.cache.metadata.get("tracker"),
                },
            ),
        )
        bundle.validate(frame_count, len(queries))
        return bundle
