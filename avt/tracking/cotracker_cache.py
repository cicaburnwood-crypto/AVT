from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
import time
from typing import Any
import gc

import numpy as np

from ..io import load_frame_window
from ..schema import FrameRecord, QueryPoint, TrackerInfo, WindowSpec
from .base import TrackingBundle
from .cotracker import CoTrackerBackend


REGION_CHOICES = ("full", "bottom-third", "bottom-half")
CACHE_SCHEMA = "avt_cotracker_cache_v1"
SEGMENTED_CACHE_SCHEMA = "avt_cotracker_segmented_cache_v1"
CHUNKED_CACHE_SCHEMA = "avt_cotracker_chunked_cache_v1"
DEFAULT_CACHE_CHUNK_SIZE = 480
DEFAULT_CACHE_WINDOW_SIZE = 80


@dataclass
class CoTrackerCacheConfig:
    frame_start: int = 0
    frame_count: int | None = None
    grid_stride: int = 1
    region: str = "bottom-third"
    query_mode: str = "confidence-refresh"
    query_frame_stride: int = 1
    max_query_points: int = 0
    device: str = "auto"
    batch_size: int = 256
    torch_home: str | None = None
    hub_repo: str = "facebookresearch/co-tracker"
    hub_model: str = "cotracker3_offline"
    visibility_threshold: float = 0.9
    abort_confidence_threshold: float | None = None
    max_track_frames: int = 0
    refresh_birth_delay_frames: int = 0


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


def load_cotracker_cache_config_yaml(path: Path) -> CoTrackerChunkCacheConfig:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("CoTracker cache YAML configs require PyYAML.") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"CoTracker cache config must be a YAML mapping: {path}")
    return cotracker_cache_config_from_mapping(data)


def cotracker_cache_config_from_mapping(data: dict[str, Any]) -> CoTrackerChunkCacheConfig:
    root = data.get("cotracker_cache", data.get("cache_config", data))
    if not isinstance(root, dict):
        raise ValueError("cotracker_cache must be a mapping")

    chunk_data = _optional_mapping(root.get("chunk"), "chunk")
    cache_data = _optional_mapping(root.get("cache"), "cache")
    tracker_data = _optional_mapping(root.get("tracker", root.get("cotracker")), "tracker")

    chunk_keys = {field.name for field in fields(CoTrackerChunkCacheConfig)} - {"cache"}
    cache_keys = {field.name for field in fields(CoTrackerCacheConfig)}
    top_level_allowed = {
        "cotracker_cache",
        "cache_config",
        "chunk",
        "cache",
        "tracker",
        "cotracker",
        *chunk_keys,
        *cache_keys,
        "bad_track_confidence_threshold",
    }
    unknown_top = sorted(set(root) - top_level_allowed)
    if unknown_top:
        raise ValueError(f"Unknown CoTracker cache config keys: {', '.join(unknown_top)}")

    chunk_values = {
        field.name: getattr(CoTrackerChunkCacheConfig(), field.name)
        for field in fields(CoTrackerChunkCacheConfig)
        if field.name != "cache"
    }
    cache_values = {
        field.name: getattr(CoTrackerCacheConfig(), field.name)
        for field in fields(CoTrackerCacheConfig)
    }

    _apply_values(chunk_values, root, chunk_keys)
    _apply_values(cache_values, root, cache_keys)
    _apply_bad_track_alias(cache_values, root)

    if chunk_data is not None:
        chunk_aliases = {"size": "chunk_size", "step": "chunk_step"}
        _apply_values(chunk_values, chunk_data, chunk_keys, aliases=chunk_aliases)
        unknown = sorted(set(chunk_data) - set(chunk_aliases) - chunk_keys)
        if unknown:
            raise ValueError(f"Unknown CoTracker cache chunk keys: {', '.join(unknown)}")

    if cache_data is not None:
        _apply_values(cache_values, cache_data, cache_keys)
        _apply_bad_track_alias(cache_values, cache_data)
        unknown = sorted(set(cache_data) - cache_keys - {"bad_track_confidence_threshold"})
        if unknown:
            raise ValueError(f"Unknown CoTracker cache keys: {', '.join(unknown)}")

    if tracker_data is not None:
        tracker_keys = {
            "device",
            "batch_size",
            "torch_home",
            "hub_repo",
            "hub_model",
            "visibility_threshold",
        }
        tracker_aliases = {"cotracker_visibility_threshold": "visibility_threshold"}
        _apply_values(cache_values, tracker_data, tracker_keys, aliases=tracker_aliases)
        unknown = sorted(set(tracker_data) - tracker_keys - set(tracker_aliases))
        if unknown:
            raise ValueError(f"Unknown CoTracker tracker keys: {', '.join(unknown)}")

    cache = CoTrackerCacheConfig(**cache_values)
    return CoTrackerChunkCacheConfig(**chunk_values, cache=cache)


def _optional_mapping(value: Any, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _apply_values(
    target: dict[str, Any],
    source: dict[str, Any],
    allowed: set[str],
    *,
    aliases: dict[str, str] | None = None,
) -> None:
    aliases = aliases or {}
    for source_key, target_key in aliases.items():
        if source_key in source:
            target[target_key] = source[source_key]
    for key in allowed:
        if key in source:
            target[key] = source[key]


def _apply_bad_track_alias(cache_values: dict[str, Any], source: dict[str, Any]) -> None:
    if "bad_track_confidence_threshold" in source:
        cache_values["abort_confidence_threshold"] = source["bad_track_confidence_threshold"]


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
    if stride != 1:
        raise ValueError("CoTracker cache uses raw dense pixels only; grid_stride must be 1")
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
    if config.query_frame_stride != 1:
        raise ValueError("confidence-refresh cache checks every frame; query_frame_stride must be 1")
    if config.query_mode != "confidence-refresh":
        raise ValueError("cache query_mode must be 'confidence-refresh'")
    if frame_count < 1:
        raise ValueError("frame_count must be positive")

    queries: list[QueryPoint] = []
    for x, y in grid:
        queries.append(
            QueryPoint(
                id=len(queries),
                reverse_time=0,
                x=float(x),
                y=float(y),
                side=0,
                source="cotracker_cache_dense_seed",
            )
        )
        if config.max_query_points > 0 and len(queries) >= config.max_query_points:
            return queries
    return queries


def _abort_threshold(config: CoTrackerCacheConfig) -> float:
    if config.abort_confidence_threshold is not None:
        return float(config.abort_confidence_threshold)
    return float(config.visibility_threshold)


def _reliable_mask(bundle: TrackingBundle, *, threshold: float) -> np.ndarray:
    reliable = np.isfinite(bundle.tracks).all(axis=2) & bundle.visibility.astype(bool)
    if bundle.confidence is not None:
        confidence = np.asarray(bundle.confidence, dtype=np.float32)
        reliable &= np.isfinite(confidence) & (confidence >= float(threshold))
    return reliable


def _first_abort_times(
    bundle: TrackingBundle,
    queries: list[QueryPoint],
    *,
    threshold: float,
    max_track_frames: int = 0,
) -> np.ndarray:
    if max_track_frames < 0:
        raise ValueError("max_track_frames must be non-negative")
    reliable = _reliable_mask(bundle, threshold=threshold)
    abort_times = np.full(len(queries), -1, dtype=np.int32)
    for idx, query in enumerate(queries):
        birth_t = int(query.reverse_time)
        start = birth_t + 1
        if start >= reliable.shape[0]:
            continue
        check_end = reliable.shape[0]
        if max_track_frames > 0:
            check_end = min(check_end, birth_t + int(max_track_frames))
        if start < check_end:
            failed = np.flatnonzero(~reliable[start:check_end, idx])
        else:
            failed = np.array([], dtype=np.int64)
        if failed.size:
            abort_times[idx] = int(start + failed[0])
        elif max_track_frames > 0 and check_end < reliable.shape[0]:
            abort_times[idx] = int(check_end)
    return abort_times


def _refresh_offset_summary(queries: list[QueryPoint], refresh_times: np.ndarray) -> str:
    offsets = [
        int(refresh_t) - int(query.reverse_time)
        for query, refresh_t in zip(queries, refresh_times)
        if int(refresh_t) >= 0
    ]
    if not offsets:
        return "refresh_offsets=none"
    values = np.array(offsets, dtype=np.float32)
    p50, p90, p99 = np.percentile(values, [50, 90, 99])
    return (
        f"refresh_offsets=min:{int(values.min())} "
        f"p50:{p50:.0f} p90:{p90:.0f} p99:{p99:.0f} max:{int(values.max())}"
    )


def _child_birth_reverse_time(abort_t: int, frame_count: int, delay_frames: int) -> int | None:
    if delay_frames < 0:
        raise ValueError("refresh_birth_delay_frames must be non-negative")
    birth_t = int(abort_t) + int(delay_frames)
    if birth_t >= int(frame_count):
        return None
    return birth_t


def _clamp_xy(xy: np.ndarray, *, height: int, width: int) -> tuple[float, float]:
    x = float(np.clip(float(xy[0]), 0.0, max(0.0, float(width - 1))))
    y = float(np.clip(float(xy[1]), 0.0, max(0.0, float(height - 1))))
    return x, y


def _refresh_birth_xy(
    bundle: TrackingBundle,
    query: QueryPoint,
    query_index: int,
    abort_t: int,
    *,
    height: int,
    width: int,
) -> tuple[float, float]:
    candidates = []
    if 0 <= abort_t < bundle.tracks.shape[0]:
        candidates.append(np.asarray(bundle.tracks[abort_t, query_index], dtype=np.float32))
    previous_t = int(abort_t) - 1
    if 0 <= previous_t < bundle.tracks.shape[0]:
        candidates.append(np.asarray(bundle.tracks[previous_t, query_index], dtype=np.float32))
    candidates.append(np.array([query.x, query.y], dtype=np.float32))

    for xy in candidates:
        if np.isfinite(xy).all():
            return _clamp_xy(xy, height=height, width=width)
    return _clamp_xy(np.array([query.x, query.y], dtype=np.float32), height=height, width=width)


def _truncate_bundle_at_aborts(bundle: TrackingBundle, abort_times: np.ndarray) -> None:
    for idx, abort_t in enumerate(abort_times):
        if abort_t < 0:
            continue
        bundle.tracks[int(abort_t) :, idx] = np.nan
        bundle.visibility[int(abort_t) :, idx] = False
        if bundle.confidence is not None:
            bundle.confidence[int(abort_t) :, idx] = 0.0
        for component in bundle.confidence_components.values():
            component[int(abort_t) :, idx] = 0.0


def _concat_confidence_components(bundles: list[TrackingBundle]) -> dict[str, np.ndarray]:
    names = sorted({name for bundle in bundles for name in bundle.confidence_components})
    components: dict[str, np.ndarray] = {}
    for name in names:
        pieces = []
        for bundle in bundles:
            if name in bundle.confidence_components:
                pieces.append(bundle.confidence_components[name].astype(np.float32))
            else:
                pieces.append(np.zeros(bundle.visibility.shape, dtype=np.float32))
        components[name] = np.concatenate(pieces, axis=1).astype(np.float32)
    return components


def _write_array(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return path.name


def _write_relative_array(root: Path, path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return str(path.relative_to(root))


def _write_segmented_metadata(
    output_dir: Path,
    *,
    source_root: Path,
    frame_start: int,
    frame_end: int,
    width: int,
    height: int,
    initial_point_count: int,
    generation_entries: list[dict[str, Any]],
    threshold: float,
    config: CoTrackerCacheConfig,
    tracker: TrackerInfo,
    complete: bool,
    error: str | None = None,
) -> dict[str, Any]:
    metadata = {
        "schema": SEGMENTED_CACHE_SCHEMA,
        "coverage_model": "streamed_confidence_refresh_segments",
        "complete": bool(complete),
        "source_root": str(source_root),
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "frame_count": int(frame_end - frame_start),
        "width": int(width),
        "height": int(height),
        "point_count": int(sum(int(entry["point_count"]) for entry in generation_entries)),
        "initial_point_count": int(initial_point_count),
        "refresh_birth_count": int(
            max(0, sum(int(entry["point_count"]) for entry in generation_entries) - int(initial_point_count))
        ),
        "refresh_generation_count": int(len(generation_entries)),
        "abort_confidence_threshold": float(threshold),
        "max_track_frames": int(config.max_track_frames),
        "refresh_birth_delay_frames": int(config.refresh_birth_delay_frames),
        "config": asdict(config),
        "tracker": tracker.to_json(),
        "generations": generation_entries,
        "time_order": {
            "tracks_reverse": "cache frame 0 is source frame frame_end - 1",
            "source_frame_for_reverse_t": "source_frame = frame_end - 1 - reverse_t",
            "birth_rule": (
                "each generation is written immediately; child IDs are born when "
                "their parent first falls below the confidence/visibility threshold "
                "or reaches max_track_frames when that value is positive; child "
                "birth can be delayed by refresh_birth_delay_frames"
            ),
            "refresh_birth_xy": "abort-frame track coordinate, falling back to previous finite coordinate then seed",
        },
    }
    if error:
        metadata["error"] = str(error)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _write_generation_record(
    output_dir: Path,
    *,
    generation_index: int,
    bundle: TrackingBundle,
    queries: list[QueryPoint],
    parent_ids: list[int],
    abort_times: np.ndarray,
    frame_end: int,
) -> dict[str, Any]:
    generation_dir = output_dir / f"generation_{generation_index:04d}"
    point_ids = np.array([query.id for query in queries], dtype=np.int64)
    birth_reverse_times = np.array([query.reverse_time for query in queries], dtype=np.int32)
    seed_xy = np.array([[query.x, query.y] for query in queries], dtype=np.float32)
    source_birth_frames = np.array(
        [frame_end - 1 - query.reverse_time for query in queries],
        dtype=np.int32,
    )
    parent_point_ids = np.array(parent_ids, dtype=np.int64)
    generation_indices = np.full(len(queries), int(generation_index), dtype=np.int16)
    abort_reverse_times = np.asarray(abort_times, dtype=np.int32)

    arrays = {
        "tracks": _write_relative_array(output_dir, generation_dir / "tracks_reverse.npy", bundle.tracks.astype(np.float32)),
        "visibility": _write_relative_array(
            output_dir,
            generation_dir / "visibility_reverse.npy",
            bundle.visibility.astype(bool),
        ),
        "point_ids": _write_relative_array(output_dir, generation_dir / "point_ids.npy", point_ids),
        "birth_reverse_times": _write_relative_array(
            output_dir,
            generation_dir / "birth_reverse_times.npy",
            birth_reverse_times,
        ),
        "source_birth_frames": _write_relative_array(
            output_dir,
            generation_dir / "source_birth_frames.npy",
            source_birth_frames,
        ),
        "seed_xy": _write_relative_array(output_dir, generation_dir / "seed_xy.npy", seed_xy),
        "parent_point_ids": _write_relative_array(
            output_dir,
            generation_dir / "parent_point_ids.npy",
            parent_point_ids,
        ),
        "generation_indices": _write_relative_array(
            output_dir,
            generation_dir / "generation_indices.npy",
            generation_indices,
        ),
        "abort_reverse_times": _write_relative_array(
            output_dir,
            generation_dir / "abort_reverse_times.npy",
            abort_reverse_times,
        ),
    }
    if bundle.confidence is not None:
        arrays["confidence"] = _write_relative_array(
            output_dir,
            generation_dir / "confidence_reverse.npy",
            bundle.confidence.astype(np.float32),
        )
    for name, component in bundle.confidence_components.items():
        arrays[f"confidence_component_{name}"] = _write_relative_array(
            output_dir,
            generation_dir / f"{name}_reverse.npy",
            component.astype(np.float32),
        )

    return {
        "generation_index": int(generation_index),
        "path": generation_dir.name,
        "point_count": int(len(queries)),
        "point_id_start": int(point_ids.min()) if point_ids.size else -1,
        "point_id_end": int(point_ids.max()) + 1 if point_ids.size else -1,
        "abort_count": int(np.count_nonzero(abort_reverse_times >= 0)),
        "arrays": arrays,
    }


def build_cotracker_cache(
    *,
    source_root: Path,
    frame_records: list[FrameRecord],
    output_dir: Path,
    config: CoTrackerCacheConfig,
) -> dict[str, Any]:
    """Write a reusable dense cache with confidence-gated ID refresh.

    The cache seeds the selected region densely on reverse frame 0. Each point is
    checked on every later reverse frame; at the first low-confidence/invisible
    frame, that parent ID is aborted and exactly one child ID is born. If
    max_track_frames is set, reliable IDs are also refreshed after that many
    useful frames instead of being forced to survive the full chunk. Child IDs
    can be delayed after parent abort by refresh_birth_delay_frames.
    """

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
    initial_queries = _cache_queries(
        frame_count=len(frames),
        height=height,
        width=width,
        config=config,
    )
    if not initial_queries:
        raise ValueError("No cache query points were generated")

    tracker = CoTrackerBackend(
        device=config.device,
        batch_size=config.batch_size,
        torch_home=config.torch_home,
        hub_repo=config.hub_repo,
        hub_model=config.hub_model,
        visibility_threshold=config.visibility_threshold,
    )
    threshold = _abort_threshold(config)
    generation_entries: list[dict[str, Any]] = []
    tracker_info: TrackerInfo | None = None

    current_queries = initial_queries
    current_parent_ids = [-1] * len(current_queries)
    generation_index = 0
    next_point_id = len(current_queries)
    while current_queries:
        generation_started = time.monotonic()
        print(
            f"confidence-refresh generation {generation_index}: "
            f"tracking {len(current_queries)} point(s)",
            flush=True,
        )
        try:
            bundle = tracker.track(frames_reverse, current_queries)
        except Exception as exc:
            if generation_entries and tracker_info is not None:
                _write_segmented_metadata(
                    output_dir,
                    source_root=source_root,
                    frame_start=config.frame_start,
                    frame_end=frame_end,
                    width=width,
                    height=height,
                    initial_point_count=len(initial_queries),
                    generation_entries=generation_entries,
                    threshold=threshold,
                    config=config,
                    tracker=tracker_info,
                    complete=False,
                    error=f"{exc.__class__.__name__}: {exc}",
                )
            raise
        tracker_info = bundle.tracker
        bundle.validate(len(frames), len(current_queries))
        abort_times = _first_abort_times(
            bundle,
            current_queries,
            threshold=threshold,
            max_track_frames=int(config.max_track_frames),
        )
        _truncate_bundle_at_aborts(bundle, abort_times)
        generation_entry = _write_generation_record(
            output_dir,
            generation_index=generation_index,
            bundle=bundle,
            queries=current_queries,
            parent_ids=current_parent_ids,
            abort_times=abort_times,
            frame_end=frame_end,
        )
        generation_entries.append(generation_entry)
        _write_segmented_metadata(
            output_dir,
            source_root=source_root,
            frame_start=config.frame_start,
            frame_end=frame_end,
            width=width,
            height=height,
            initial_point_count=len(initial_queries),
            generation_entries=generation_entries,
            threshold=threshold,
            config=config,
            tracker=tracker_info,
            complete=False,
        )

        next_queries: list[QueryPoint] = []
        next_parent_ids: list[int] = []
        for query_index, abort_t in enumerate(abort_times):
            if abort_t < 0:
                continue
            child_birth_t = _child_birth_reverse_time(
                int(abort_t),
                len(frames),
                int(config.refresh_birth_delay_frames),
            )
            if child_birth_t is None:
                continue
            if config.max_query_points > 0 and next_point_id + len(next_queries) >= config.max_query_points:
                break
            x, y = _refresh_birth_xy(
                bundle,
                current_queries[query_index],
                query_index,
                int(abort_t),
                height=height,
                width=width,
            )
            next_queries.append(
                QueryPoint(
                    id=next_point_id,
                    reverse_time=int(child_birth_t),
                    x=x,
                    y=y,
                    side=0,
                    source="cotracker_cache_confidence_refresh",
                )
            )
            next_parent_ids.append(int(current_queries[query_index].id))
            next_point_id += 1

        elapsed = time.monotonic() - generation_started
        print(
            f"confidence-refresh generation {generation_index}: "
            f"aborted={int(np.count_nonzero(abort_times >= 0))} "
            f"created={len(next_queries)} "
            f"threshold={threshold:.3f} "
            f"max_track_frames={int(config.max_track_frames)} "
            f"refresh_birth_delay_frames={int(config.refresh_birth_delay_frames)} "
            f"{_refresh_offset_summary(current_queries, abort_times)} "
            f"elapsed={_format_duration(elapsed)}",
            flush=True,
        )
        del bundle
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        current_queries = next_queries
        current_parent_ids = next_parent_ids
        generation_index += 1

    if tracker_info is None:
        raise RuntimeError("No CoTracker generations were written")
    return _write_segmented_metadata(
        output_dir,
        source_root=source_root,
        frame_start=config.frame_start,
        frame_end=frame_end,
        width=width,
        height=height,
        initial_point_count=len(initial_queries),
        generation_entries=generation_entries,
        threshold=threshold,
        config=config,
        tracker=tracker_info,
        complete=True,
    )


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
    schema = metadata.get("schema")
    if schema not in (CACHE_SCHEMA, SEGMENTED_CACHE_SCHEMA):
        return False
    if schema == SEGMENTED_CACHE_SCHEMA and not bool(metadata.get("complete", False)):
        return False
    if int(metadata.get("frame_start", -1)) != int(frame_start):
        return False
    if int(metadata.get("frame_end", -1)) != int(frame_end):
        return False
    existing = metadata.get("config", {})
    for key in (
        "grid_stride",
        "region",
        "query_mode",
        "query_frame_stride",
        "max_query_points",
        "visibility_threshold",
        "abort_confidence_threshold",
        "max_track_frames",
        "refresh_birth_delay_frames",
    ):
        if existing.get(key) != getattr(config, key):
            return False
    if schema == CACHE_SCHEMA:
        arrays = metadata.get("arrays", {})
        return all((path / arrays[name]).exists() for name in ("tracks", "visibility", "point_ids"))
    generations = metadata.get("generations", [])
    if not generations:
        return False
    for generation in generations:
        arrays = generation.get("arrays", {})
        if not all((path / arrays[name]).exists() for name in ("tracks", "visibility", "point_ids")):
            return False
    return True


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


class _SegmentedArrayProxy:
    def __init__(self, arrays: list[np.ndarray], offsets: list[int], point_count: int) -> None:
        if not arrays:
            raise ValueError("Segmented array proxy needs at least one array")
        self.arrays = arrays
        self.offsets = offsets
        self.point_count = int(point_count)
        self.shape = (arrays[0].shape[0], self.point_count, *arrays[0].shape[2:])
        self.dtype = arrays[0].dtype

    def _slice_points(self, frames: np.ndarray, points: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames, dtype=np.int64).reshape(-1)
        points = np.asarray(points, dtype=np.int64).reshape(-1)
        out = np.zeros((len(frames), len(points), *self.shape[2:]), dtype=self.dtype)
        if np.issubdtype(self.dtype, np.floating):
            out[...] = np.nan
        for array, offset in zip(self.arrays, self.offsets):
            end = offset + array.shape[1]
            mask = (points >= offset) & (points < end)
            if not mask.any():
                continue
            local_points = points[mask] - offset
            out[:, mask] = array[np.ix_(frames, local_points)]
        return out

    def __getitem__(self, key):
        if isinstance(key, tuple):
            frames, points = key
            return self._slice_points(np.asarray(frames).reshape(-1), np.asarray(points).reshape(-1))
        if isinstance(key, (int, np.integer)):
            frame = int(key)
            return np.concatenate([array[frame] for array in self.arrays], axis=0)
        frames = np.arange(self.shape[0], dtype=np.int64)[key]
        points = np.arange(self.point_count, dtype=np.int64)
        return self._slice_points(np.asarray(frames), points)


class CoTrackerCache:
    def __init__(self, path: Path, *, mmap_mode: str | None = "r") -> None:
        root = path if path.is_dir() else path.parent
        metadata_path = root / "metadata.json" if path.is_dir() else path
        self.root = root.resolve()
        self.metadata_path = metadata_path.resolve()
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        schema = self.metadata.get("schema")
        if schema not in (CACHE_SCHEMA, SEGMENTED_CACHE_SCHEMA):
            raise ValueError(f"Unsupported CoTracker cache schema: {self.metadata.get('schema')}")
        self.frame_start = int(self.metadata["frame_start"])
        self.frame_end = int(self.metadata["frame_end"])
        self.frame_count = int(self.metadata["frame_count"])
        if schema == SEGMENTED_CACHE_SCHEMA:
            self._load_segmented(mmap_mode=mmap_mode)
            return

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
        self.parent_point_ids = (
            np.load(self.root / arrays["parent_point_ids"], mmap_mode=mmap_mode)
            if "parent_point_ids" in arrays
            else np.full(self.point_ids.shape, -1, dtype=np.int64)
        )
        self.generation_indices = (
            np.load(self.root / arrays["generation_indices"], mmap_mode=mmap_mode)
            if "generation_indices" in arrays
            else np.zeros(self.point_ids.shape, dtype=np.int16)
        )
        self.abort_reverse_times = (
            np.load(self.root / arrays["abort_reverse_times"], mmap_mode=mmap_mode)
            if "abort_reverse_times" in arrays
            else np.full(self.point_ids.shape, -1, dtype=np.int32)
        )
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

    def _load_segmented(self, *, mmap_mode: str | None) -> None:
        generations = self.metadata.get("generations", [])
        if not generations:
            raise ValueError(f"Segmented CoTracker cache has no generations: {self.metadata_path}")

        tracks_arrays: list[np.ndarray] = []
        visibility_arrays: list[np.ndarray] = []
        confidence_arrays: list[np.ndarray] = []
        component_arrays: dict[str, list[np.ndarray]] = {}
        offsets: list[int] = []
        point_ids = []
        birth_reverse_times = []
        source_birth_frames = []
        seed_xy = []
        parent_point_ids = []
        generation_indices = []
        abort_reverse_times = []
        point_offset = 0

        for generation in generations:
            arrays = generation["arrays"]
            count = int(generation["point_count"])
            offsets.append(point_offset)
            point_offset += count
            tracks_arrays.append(np.load(self.root / arrays["tracks"], mmap_mode=mmap_mode))
            visibility_arrays.append(np.load(self.root / arrays["visibility"], mmap_mode=mmap_mode))
            if "confidence" in arrays:
                confidence_arrays.append(np.load(self.root / arrays["confidence"], mmap_mode=mmap_mode))
            for key, filename in arrays.items():
                if key.startswith("confidence_component_"):
                    component_arrays.setdefault(key.removeprefix("confidence_component_"), []).append(
                        np.load(self.root / filename, mmap_mode=mmap_mode)
                    )
            point_ids.append(np.load(self.root / arrays["point_ids"], mmap_mode=mmap_mode))
            birth_reverse_times.append(np.load(self.root / arrays["birth_reverse_times"], mmap_mode=mmap_mode))
            source_birth_frames.append(np.load(self.root / arrays["source_birth_frames"], mmap_mode=mmap_mode))
            seed_xy.append(np.load(self.root / arrays["seed_xy"], mmap_mode=mmap_mode))
            parent_point_ids.append(np.load(self.root / arrays["parent_point_ids"], mmap_mode=mmap_mode))
            generation_indices.append(np.load(self.root / arrays["generation_indices"], mmap_mode=mmap_mode))
            abort_reverse_times.append(np.load(self.root / arrays["abort_reverse_times"], mmap_mode=mmap_mode))

        self.tracks = _SegmentedArrayProxy(tracks_arrays, offsets, point_offset)
        self.visibility = _SegmentedArrayProxy(visibility_arrays, offsets, point_offset)
        self.confidence = (
            _SegmentedArrayProxy(confidence_arrays, offsets, point_offset)
            if len(confidence_arrays) == len(generations)
            else None
        )
        self.confidence_components = {
            name: _SegmentedArrayProxy(arrays, offsets, point_offset)
            for name, arrays in component_arrays.items()
            if len(arrays) == len(generations)
        }
        self.point_ids = np.concatenate(point_ids).astype(np.int64)
        self.birth_reverse_times = np.concatenate(birth_reverse_times).astype(np.int32)
        self.source_birth_frames = np.concatenate(source_birth_frames).astype(np.int32)
        self.seed_xy = np.concatenate(seed_xy).astype(np.float32)
        self.parent_point_ids = np.concatenate(parent_point_ids).astype(np.int64)
        self.generation_indices = np.concatenate(generation_indices).astype(np.int16)
        self.abort_reverse_times = np.concatenate(abort_reverse_times).astype(np.int32)

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
        cache_parent_point_ids = np.full(len(queries), -1, dtype=np.int64)
        cache_generation_indices = np.full(len(queries), -1, dtype=np.int16)
        cache_abort_reverse_times = np.full(len(queries), -1, dtype=np.int32)
        if ok.any():
            cache_parent_point_ids[ok] = np.asarray(cache.parent_point_ids[matched[ok]], dtype=np.int64)
            cache_generation_indices[ok] = np.asarray(cache.generation_indices[matched[ok]], dtype=np.int16)
            cache_abort_reverse_times[ok] = np.asarray(cache.abort_reverse_times[matched[ok]], dtype=np.int32)

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
                "cache_parent_point_ids": cache_parent_point_ids,
                "cache_generation_indices": cache_generation_indices,
                "cache_abort_reverse_times": cache_abort_reverse_times,
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
