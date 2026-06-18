from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

RELIABILITY_SCHEMA = "avt_frame_segment_reliability_v1"
DEFAULT_FILTER_NAME = "stationary_sift_segment_filter"
DEFAULT_SEGMENT_SIZE_FRAMES = 40
STOP_EXTREME_SLOW_REASON = "stop_extreme_slow_motion"
STATIONARY_BLOCK_SIZE_PX = 6.0
STATIONARY_SPAN_FRAMES = 10
MIN_STATIONARY_SIFT_POINTS = 3


def reliability_metadata(segment_size_frames: int = DEFAULT_SEGMENT_SIZE_FRAMES) -> dict[str, Any]:
    return {
        "schema": RELIABILITY_SCHEMA,
        "filter": DEFAULT_FILTER_NAME,
        "action": "mark_only",
        "segment_size_frames": int(segment_size_frames),
        "propagation": "if any frame is unreliable, mark its whole frame segment unreliable",
        "rules": [
            {
                "reason": STOP_EXTREME_SLOW_REASON,
                "source": "sift",
                "span_frames": STATIONARY_SPAN_FRAMES,
                "block_size_px": STATIONARY_BLOCK_SIZE_PX,
                "min_points": MIN_STATIONARY_SIFT_POINTS,
                "segment_mark": "disabled",
            }
        ],
    }


def segment_bounds(frame_index: int, segment_size_frames: int = DEFAULT_SEGMENT_SIZE_FRAMES) -> tuple[int, int]:
    if segment_size_frames <= 0:
        raise ValueError("segment_size_frames must be positive")
    start = (int(frame_index) // int(segment_size_frames)) * int(segment_size_frames)
    return start, start + int(segment_size_frames)


def segment_id(frame_index: int, segment_size_frames: int = DEFAULT_SEGMENT_SIZE_FRAMES) -> int:
    start, _ = segment_bounds(frame_index, segment_size_frames)
    return start // int(segment_size_frames)


def unreliable_segments(
    unreliable_frame_indices: Sequence[int],
    segment_size_frames: int = DEFAULT_SEGMENT_SIZE_FRAMES,
) -> set[int]:
    return {segment_id(frame, segment_size_frames) for frame in unreliable_frame_indices}


def detect_stationary_sift_frames(
    tracks: np.ndarray,
    visibility: np.ndarray,
    *,
    seq_start: int,
    seq_end: int,
    sift_point_ids: Sequence[int],
    block_size_px: float = STATIONARY_BLOCK_SIZE_PX,
    span_frames: int = STATIONARY_SPAN_FRAMES,
    min_points: int = MIN_STATIONARY_SIFT_POINTS,
) -> dict[int, list[str]]:
    if span_frames <= 1:
        raise ValueError("span_frames must be greater than 1")
    if block_size_px <= 0:
        raise ValueError("block_size_px must be positive")
    if min_points <= 0:
        raise ValueError("min_points must be positive")

    ids = np.array(sorted({int(idx) for idx in sift_point_ids}), dtype=np.int64)
    if ids.size == 0:
        return {}
    ids = ids[(ids >= 0) & (ids < tracks.shape[1])]
    if ids.size == 0:
        return {}

    frame_reasons: dict[int, list[str]] = {}
    first_frame = int(seq_start) + int(span_frames) - 1
    for frame_idx in range(first_frame, int(seq_end)):
        frame_span = range(frame_idx - int(span_frames) + 1, frame_idx + 1)
        reverse_times = [int(seq_end) - 1 - frame for frame in frame_span]
        if min(reverse_times) < 0 or max(reverse_times) >= tracks.shape[0]:
            continue

        span_visibility = visibility[reverse_times][:, ids].astype(bool)
        span_tracks = tracks[reverse_times][:, ids]
        valid = span_visibility.all(axis=0) & np.isfinite(span_tracks).all(axis=(0, 2))
        if not valid.any():
            continue

        coords = span_tracks[:, valid, :]
        extents = coords.max(axis=0) - coords.min(axis=0)
        stationary = (extents[:, 0] <= block_size_px) & (extents[:, 1] <= block_size_px)
        if int(stationary.sum()) >= min_points:
            frame_reasons[int(frame_idx)] = [STOP_EXTREME_SLOW_REASON]
    return frame_reasons


def frame_reliability(
    frame_index: int,
    points: Sequence[Sequence[float | int]],
    *,
    unreliable_frame_indices: Sequence[int] = (),
    frame_reasons: Mapping[int, Sequence[str]] | None = None,
    segment_size_frames: int = DEFAULT_SEGMENT_SIZE_FRAMES,
) -> dict[str, Any]:
    _ = points
    unreliable_frames = {int(frame) for frame in unreliable_frame_indices}
    current_segment = segment_id(frame_index, segment_size_frames)
    start, end = segment_bounds(frame_index, segment_size_frames)
    trigger_frames = sorted(
        frame for frame in unreliable_frames if segment_id(frame, segment_size_frames) == current_segment
    )
    reason_counts: dict[str, int] = {}
    if frame_reasons:
        for frame in trigger_frames:
            for reason in frame_reasons.get(frame, ()):
                reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    return {
        "schema": RELIABILITY_SCHEMA,
        "filter": DEFAULT_FILTER_NAME,
        "action": "mark_only",
        "segment_size_frames": int(segment_size_frames),
        "segment_id": current_segment,
        "segment_start_frame": start,
        "segment_end_frame": end,
        "frame_unreliable": int(frame_index) in unreliable_frames,
        "segment_unreliable": bool(trigger_frames),
        "frame_disabled": int(frame_index) in unreliable_frames,
        "segment_disabled": bool(trigger_frames),
        "segment_status": "disabled" if trigger_frames else "enabled",
        "trigger_frame_indices": trigger_frames,
        "unreliable_point_ids": [],
        "reason_counts": reason_counts,
    }
