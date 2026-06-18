from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from ..io import load_frame_window
from ..schema import FrameRecord, QueryPoint, TrackerInfo, WindowSpec
from .base import TrackingBundle
from .cotracker import CoTrackerBackend


REGION_CHOICES = ("full", "bottom-third", "bottom-half")
CACHE_SCHEMA = "avt_cotracker_cache_v1"
CHUNKED_CACHE_SCHEMA = "avt_cotracker_chunked_cache_v1"
DEFAULT_CACHE_CHUNK_SIZE = 480
DEFAULT_CACHE_WINDOW_SIZE = 80


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


@dataclass
class CoTrackerChunkCacheConfig:
    chunk_size: int = DEFAULT_CACHE_CHUNK_SIZE
    window_size: int = DEFAULT_CACHE_WINDOW_SIZE
    chunk_step: int | None = None
    cache: CoTrackerCacheConfig = field(default_factory=CoTrackerCacheConfig)

    @property
    def resolved_chunk_step(self) -> int:
        if self.chunk_step is not None:
            return int(self.chunk_step)
        return max(1, int(self.chunk_size) - int(self.window_size))


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


def _chunk_ranges(frame_count: int, config: CoTrackerChunkCacheConfig) -> list[tuple[int, int]]:
    chunk_size = int(config.chunk_size)
    window_size = int(config.window_size)
    chunk_step = int(config.resolved_chunk_step)
    if chunk_size < 2:
        raise ValueError("cache chunk_size must be at least 2")
    if window_size < 2:
        raise ValueError("cache window_size must be at least 2")
    if chunk_size < window_size:
        raise ValueError("cache chunk_size must be greater than or equal to window_size")
    if chunk_step <= 0:
        raise ValueError("cache chunk_step must be positive")
    if chunk_step > chunk_size - window_size + 1:
        raise ValueError(
            "cache chunk_step is too large to guarantee every extraction window "
            "fits inside a chunk"
        )

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < frame_count:
        end = min(frame_count, start + chunk_size)
        if end - start >= 2:
            ranges.append((start, end))
        if end >= frame_count:
            break
        start += chunk_step
    return ranges


def _chunk_id(start: int, end: int) -> str:
    return f"chunk_{start:06d}_{end - 1:06d}"


def _cache_metadata_matches(path: Path, *, frame_start: int, frame_end: int, config: CoTrackerCacheConfig) -> bool:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if metadata.get("schema") != CACHE_SCHEMA:
        return False
    if int(metadata.get("frame_start", -1)) != int(frame_start):
        return False
    if int(metadata.get("frame_end", -1)) != int(frame_end):
        return False
    existing = metadata.get("config", {})
    for key in ("grid_stride", "region", "query_mode", "query_frame_stride", "max_query_points"):
        if existing.get(key) != getattr(config, key):
            return False
    arrays = metadata.get("arrays", {})
    return all((path / arrays[name]).exists() for name in ("tracks", "visibility", "point_ids"))


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def build_cotracker_cache_chunks(
    *,
    source_root: Path,
    frame_records: list[FrameRecord],
    output_dir: Path,
    config: CoTrackerChunkCacheConfig,
    resume: bool = True,
) -> dict[str, Any]:
    """Build independent overlapping cache chunks and write a selector manifest."""

    source_root = source_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ranges = _chunk_ranges(len(frame_records), config)
    if not ranges:
        raise ValueError("No cache chunks were generated")

    entries: list[dict[str, Any]] = []
    started_at = time.monotonic()
    for ordinal, (start, end) in enumerate(ranges, start=1):
        chunk_id = _chunk_id(start, end)
        chunk_root = output_dir / chunk_id
        cache_config = CoTrackerCacheConfig(**asdict(config.cache))
        cache_config.frame_start = start
        cache_config.frame_count = end - start
        if resume and _cache_metadata_matches(
            chunk_root,
            frame_start=start,
            frame_end=end,
            config=cache_config,
        ):
            metadata = json.loads((chunk_root / "metadata.json").read_text(encoding="utf-8"))
            print(f"[{ordinal}/{len(ranges)}] reusing {chunk_id}", flush=True)
        else:
            print(f"[{ordinal}/{len(ranges)}] caching {chunk_id}", flush=True)
            metadata = build_cotracker_cache(
                source_root=source_root,
                frame_records=frame_records,
                output_dir=chunk_root,
                config=cache_config,
            )
        entries.append(
            {
                "chunk_id": chunk_id,
                "chunk_index": ordinal - 1,
                "path": str(chunk_root.resolve()),
                "metadata": str((chunk_root / "metadata.json").resolve()),
                "frame_start": int(start),
                "frame_end": int(end),
                "frame_count": int(end - start),
                "point_count": int(metadata["point_count"]),
            }
        )
        elapsed = time.monotonic() - started_at
        average = elapsed / ordinal
        remaining = average * (len(ranges) - ordinal)
        print(
            f"[{ordinal}/{len(ranges)}] done {chunk_id} "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(remaining)}",
            flush=True,
        )

    manifest = {
        "schema": CHUNKED_CACHE_SCHEMA,
        "source_root": str(source_root),
        "frame_count": int(len(frame_records)),
        "chunk_size": int(config.chunk_size),
        "window_size": int(config.window_size),
        "chunk_step": int(config.resolved_chunk_step),
        "overlap": int(config.chunk_size - config.resolved_chunk_step),
        "selection": {
            "mode": "full_window_single_chunk",
            "score": "mean_confidence_with_unmatched_as_zero",
            "tie_breakers": ["matched_queries", "mean_visibility", "boundary_margin"],
        },
        "cache_config": asdict(config.cache),
        "chunks": entries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return manifest


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


@dataclass(frozen=True)
class CoTrackerCacheChunk:
    chunk_id: str
    chunk_index: int
    path: Path
    metadata: Path
    frame_start: int
    frame_end: int
    point_count: int

    @property
    def frame_count(self) -> int:
        return self.frame_end - self.frame_start


class CoTrackerChunkedCacheIndex:
    def __init__(self, path: Path) -> None:
        root = path if path.is_dir() else path.parent
        manifest_path = root / "manifest.json" if path.is_dir() else path
        self.root = root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != CHUNKED_CACHE_SCHEMA:
            raise ValueError(f"Unsupported chunked cache schema: {self.manifest.get('schema')}")
        self.source_root = Path(self.manifest["source_root"]).resolve()
        self.chunks = [
            CoTrackerCacheChunk(
                chunk_id=str(entry["chunk_id"]),
                chunk_index=int(entry["chunk_index"]),
                path=Path(entry["path"]).resolve(),
                metadata=Path(entry["metadata"]).resolve(),
                frame_start=int(entry["frame_start"]),
                frame_end=int(entry["frame_end"]),
                point_count=int(entry["point_count"]),
            )
            for entry in self.manifest["chunks"]
        ]

    @classmethod
    def maybe_load(cls, path: Path) -> "CoTrackerChunkedCacheIndex | None":
        root = path if path.is_dir() else path.parent
        manifest_path = root / "manifest.json" if path.is_dir() else path
        if not manifest_path.exists():
            return None
        try:
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if metadata.get("schema") != CHUNKED_CACHE_SCHEMA:
            return None
        return cls(manifest_path)

    def candidates_for_window(self, window: WindowSpec) -> list[CoTrackerCacheChunk]:
        return [
            chunk
            for chunk in self.chunks
            if chunk.frame_start <= window.start and chunk.frame_end >= window.end
        ]


@dataclass
class CachedCoTrackerBackend:
    cache_path: Path
    max_match_distance: float = 12.0
    source_root: Path | None = None
    mmap_mode: str | None = "r"

    def __post_init__(self) -> None:
        self.chunk_index = CoTrackerChunkedCacheIndex.maybe_load(self.cache_path)
        self.cache = (
            None
            if self.chunk_index is not None
            else CoTrackerCache(self.cache_path, mmap_mode=self.mmap_mode)
        )
        self.window: WindowSpec | None = None
        self.output_dir: Path | None = None

    def set_window_context(self, window: WindowSpec, output_dir: Path) -> None:
        self.window = window
        self.output_dir = output_dir

    def _visible_candidates(self, cache: CoTrackerCache, cache_reverse_t: int) -> tuple[np.ndarray, np.ndarray]:
        visible = np.asarray(cache.visibility[cache_reverse_t]).astype(bool)
        if not visible.any():
            return np.empty((0,), dtype=np.int64), np.empty((0, 2), dtype=np.float32)
        points = np.asarray(cache.tracks[cache_reverse_t], dtype=np.float32)
        finite = np.isfinite(points).all(axis=1)
        candidates = np.flatnonzero(visible & finite)
        if candidates.size == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0, 2), dtype=np.float32)
        return candidates, points[candidates]

    def _match_query_group(
        self,
        cache: CoTrackerCache,
        queries: list[QueryPoint],
        query_indices: np.ndarray,
        cache_reverse_t: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        matched = np.full(query_indices.size, -1, dtype=np.int64)
        distances = np.full(query_indices.size, np.inf, dtype=np.float32)
        candidates, points = self._visible_candidates(cache, cache_reverse_t)
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

    def _track_with_cache(
        self,
        cache: CoTrackerCache,
        frames_rgb: np.ndarray,
        queries: list[QueryPoint],
        *,
        chunk: CoTrackerCacheChunk | None = None,
        candidate_count: int = 1,
    ) -> tuple[TrackingBundle, tuple[float, int, float, int]]:
        if self.window is None:
            raise RuntimeError("CachedCoTrackerBackend needs set_window_context before track().")

        frame_count = len(frames_rgb)
        cache_start_reverse = cache.reverse_index_for_source_frame(self.window.end - 1)
        cache_indices = np.arange(cache_start_reverse, cache_start_reverse + frame_count, dtype=np.int64)
        if cache_indices[-1] >= cache.frame_count:
            raise IndexError(
                f"Window {self.window.id} extends outside cache reverse range "
                f"at {int(cache_indices[-1])} >= {cache.frame_count}"
            )

        query_cache_indices = np.array(
            [
                cache.reverse_index_for_source_frame(self.window.end - 1 - query.reverse_time)
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
                cache,
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
            if cache.confidence is not None
            else None
        )
        confidence_components = {
            name: np.zeros((frame_count, len(queries)), dtype=np.float32)
            for name in cache.confidence_components
        }

        valid_query_indices = np.flatnonzero(matched >= 0)
        if valid_query_indices.size:
            cache_point_indices = matched[valid_query_indices]
            tracks[:, valid_query_indices] = np.asarray(
                cache.tracks[np.ix_(cache_indices, cache_point_indices)],
                dtype=np.float32,
            )
            visibility[:, valid_query_indices] = np.asarray(
                cache.visibility[np.ix_(cache_indices, cache_point_indices)],
                dtype=bool,
            )
            if confidence is not None and cache.confidence is not None:
                confidence[:, valid_query_indices] = np.asarray(
                    cache.confidence[np.ix_(cache_indices, cache_point_indices)],
                    dtype=np.float32,
                )
            for name, component in cache.confidence_components.items():
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
            birth_times[ok] = np.asarray(cache.birth_reverse_times[matched[ok]], dtype=np.int32)
            source_birth_frames[ok] = np.asarray(cache.source_birth_frames[matched[ok]], dtype=np.int32)

        cache_point_ids = np.full(len(queries), -1, dtype=np.int64)
        if ok.any():
            cache_point_ids[ok] = np.asarray(cache.point_ids[matched[ok]], dtype=np.int64)

        chunk_id = chunk.chunk_id if chunk is not None else cache.root.name
        chunk_index = chunk.chunk_index if chunk is not None else 0
        cache_chunk_indices = np.full(len(queries), -1, dtype=np.int32)
        cache_chunk_indices[ok] = chunk_index
        cache_chunk_ids = np.array([chunk_id if matched_idx >= 0 else "" for matched_idx in matched], dtype=f"<U{max(1, len(chunk_id))}")
        unique_ids = np.array(
            [
                f"{chunk_id}:{int(point_id)}" if point_id >= 0 else ""
                for point_id in cache_point_ids
            ],
            dtype=f"<U{max(1, len(chunk_id) + 32)}",
        )

        if confidence is not None:
            selection_values = confidence
        else:
            selection_values = visibility.astype(np.float32)
        selection_score = float(np.mean(selection_values)) if selection_values.size else 0.0
        mean_visibility = float(np.mean(visibility)) if visibility.size else 0.0
        boundary_margin = min(self.window.start - cache.frame_start, cache.frame_end - self.window.end)

        bundle = TrackingBundle(
            tracks=tracks,
            visibility=visibility,
            confidence=confidence,
            confidence_components=confidence_components,
            extra_arrays={
                "cache_point_ids": cache_point_ids,
                "cache_point_indices": matched.astype(np.int64),
                "cache_chunk_indices": cache_chunk_indices,
                "cache_chunk_ids": cache_chunk_ids,
                "cache_unique_point_ids": unique_ids,
                "cache_query_reverse_times": query_cache_indices,
                "cache_query_source_frames": source_query_frames,
                "cache_birth_reverse_times": birth_times,
                "cache_birth_source_frames": source_birth_frames,
                "cache_match_distances": distances.astype(np.float32),
            },
            extra_metadata={
                "cache_reference": {
                    "schema": CACHE_SCHEMA,
                    "path": str(cache.root),
                    "metadata": str(cache.metadata_path),
                    "frame_start": cache.frame_start,
                    "frame_end": cache.frame_end,
                    "chunk_id": chunk_id,
                    "chunk_index": int(chunk_index),
                    "max_match_distance": float(self.max_match_distance),
                    "matched_queries": int(ok.sum()),
                    "query_count": int(len(queries)),
                    "selection_score": selection_score,
                    "selection_metric": "mean_confidence_with_unmatched_as_zero"
                    if confidence is not None
                    else "mean_visibility_with_unmatched_as_zero",
                    "candidate_chunks": int(candidate_count),
                    "boundary_margin_frames": int(boundary_margin),
                }
            },
            tracker=TrackerInfo(
                name="cotracker_cache",
                parameters={
                    "cache_path": str(cache.root),
                    "max_match_distance": float(self.max_match_distance),
                    "source_tracker": cache.metadata.get("tracker"),
                },
            ),
        )
        bundle.validate(frame_count, len(queries))
        return bundle, (selection_score, int(ok.sum()), mean_visibility, int(boundary_margin))

    def track(self, frames_rgb: np.ndarray, queries: list[QueryPoint]) -> TrackingBundle:
        if self.window is None:
            raise RuntimeError("CachedCoTrackerBackend needs set_window_context before track().")
        if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
            raise ValueError("frames_rgb must have shape [T,H,W,3]")
        if not queries:
            raise ValueError("No query points were provided")

        if self.chunk_index is None:
            if self.cache is None:
                raise RuntimeError("Cache backend was not initialized")
            bundle, _ = self._track_with_cache(self.cache, frames_rgb, queries)
            return bundle

        candidates = self.chunk_index.candidates_for_window(self.window)
        if not candidates:
            raise IndexError(
                f"No CoTracker cache chunk fully covers window {self.window.id}; "
                f"build chunks with overlap at least the extraction window size."
            )

        best_bundle: TrackingBundle | None = None
        best_score: tuple[float, int, float, int] | None = None
        best_chunk: CoTrackerCacheChunk | None = None
        for chunk in candidates:
            cache = CoTrackerCache(chunk.path, mmap_mode=self.mmap_mode)
            bundle, score = self._track_with_cache(
                cache,
                frames_rgb,
                queries,
                chunk=chunk,
                candidate_count=len(candidates),
            )
            if best_score is None or score > best_score:
                best_bundle = bundle
                best_score = score
                best_chunk = chunk

        if best_bundle is None or best_chunk is None:
            raise RuntimeError(f"Could not select a cache chunk for window {self.window.id}")
        best_bundle.extra_metadata["cache_reference"]["chunked_schema"] = CHUNKED_CACHE_SCHEMA
        best_bundle.extra_metadata["cache_reference"]["chunked_cache_root"] = str(self.chunk_index.root)
        best_bundle.extra_metadata["cache_reference"]["chunked_manifest"] = str(self.chunk_index.manifest_path)
        best_bundle.tracker = TrackerInfo(
            name=best_bundle.tracker.name,
            version=best_bundle.tracker.version,
            parameters={
                **(best_bundle.tracker.parameters or {}),
                "chunked_cache_root": str(self.chunk_index.root),
                "chunk_selection": "highest_confidence_full_window",
            },
        )
        return best_bundle
