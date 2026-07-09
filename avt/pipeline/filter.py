"""Stage 3.5 - track filtering.

This stage is deliberately backend-neutral. A tracker still only needs to
return ``TrackingBundle``; a filter can then refine visibility/confidence
without knowing whether the tracks came from LK, CoTracker, BootsTAPIR, or a
custom backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np

from ..config import AnchorMotionFilterConfig, InverseTrackConfig
from ..schema import QueryPoint
from ..tracking.base import TrackingBundle
from .preprocess import PreparedWindow

__all__ = [
    "PointTrackFilter",
    "NoOpPointTrackFilter",
    "AnchorMotionFootprintFilter",
    "run_track_filter",
    "apply_anchor_motion_filter",
]


class PointTrackFilter(Protocol):
    """Interface for optional post-tracking visibility/confidence filters."""

    def filter(
        self,
        prepared: PreparedWindow,
        queries: list[QueryPoint],
        bundle: TrackingBundle,
        config: InverseTrackConfig,
    ) -> TrackingBundle:
        """Return a filtered ``TrackingBundle`` for the same query ids."""


class NoOpPointTrackFilter:
    """Pass tracks through unchanged."""

    def filter(
        self,
        prepared: PreparedWindow,
        queries: list[QueryPoint],
        bundle: TrackingBundle,
        config: InverseTrackConfig,
    ) -> TrackingBundle:
        return bundle


class AnchorMotionFootprintFilter:
    """Filter footprint tracks by their consistency with anchor-estimated motion."""

    def filter(
        self,
        prepared: PreparedWindow,
        queries: list[QueryPoint],
        bundle: TrackingBundle,
        config: InverseTrackConfig,
    ) -> TrackingBundle:
        del prepared
        filter_config = config.anchor_motion_filter
        if not filter_config.enabled:
            return bundle
        return apply_anchor_motion_filter(queries, bundle, filter_config)


def run_track_filter(
    prepared: PreparedWindow,
    queries: list[QueryPoint],
    bundle: TrackingBundle,
    config: InverseTrackConfig,
    filterer: PointTrackFilter,
) -> TrackingBundle:
    filtered = filterer.filter(prepared, queries, bundle, config)
    filtered.validate(len(prepared.frames_reverse), len(queries))
    return filtered


@dataclass(frozen=True)
class _MotionEstimate:
    transform: np.ndarray | None
    valid: bool
    method: str
    reason: str
    match_count: int
    inlier_count: int
    inlier_ratio: float | None
    reprojection_error_mean_px: float | None
    reprojection_error_median_px: float | None


def apply_anchor_motion_filter(
    queries: list[QueryPoint],
    bundle: TrackingBundle,
    config: AnchorMotionFilterConfig,
) -> TrackingBundle:
    """Combine tracker confidence with anchor-motion consistency.

    Anchors estimate adjacent-frame image motion. Footprint points are retained
    when their tracked displacement agrees with that motion.
    """

    frame_count, query_count = bundle.visibility.shape
    anchor_ids = _query_ids(queries, query_count, {"anchor"})
    target_ids = _query_ids(queries, query_count, set(config.target_sources))
    if anchor_ids.size == 0 or target_ids.size == 0:
        return bundle

    base_confidence = _base_confidence(bundle)
    motion_confidence = np.ones((frame_count, query_count), dtype=np.float32)
    residual_px = np.full((frame_count, query_count), np.nan, dtype=np.float32)
    frame_stats: list[dict[str, Any]] = []

    for reverse_time in range(1, frame_count):
        estimate = _estimate_adjacent_anchor_motion(
            bundle,
            anchor_ids,
            src_time=reverse_time - 1,
            dst_time=reverse_time,
            config=config,
        )
        stats = _motion_estimate_json(reverse_time, estimate)
        if estimate.valid and estimate.transform is not None:
            visible_targets = target_ids[
                bundle.visibility[reverse_time - 1, target_ids]
                & bundle.visibility[reverse_time, target_ids]
            ]
            if visible_targets.size:
                predicted = _apply_affine(
                    bundle.tracks[reverse_time - 1, visible_targets],
                    estimate.transform,
                )
                actual = bundle.tracks[reverse_time, visible_targets]
                finite = np.isfinite(predicted).all(axis=1) & np.isfinite(actual).all(axis=1)
                finite_ids = visible_targets[finite]
                if finite_ids.size:
                    residual = np.linalg.norm(actual[finite] - predicted[finite], axis=1)
                    confidence = _residual_confidence(residual, config.residual_scale_px)
                    residual_px[reverse_time, finite_ids] = residual.astype(np.float32)
                    motion_confidence[reverse_time, finite_ids] = confidence
        else:
            visible_targets = target_ids[bundle.visibility[reverse_time, target_ids]]
            if visible_targets.size:
                motion_confidence[reverse_time, visible_targets] = np.clip(
                    float(config.fallback_confidence),
                    0.0,
                    1.0,
                )
        frame_stats.append(stats)

    final_confidence = np.clip(base_confidence * motion_confidence, 0.0, 1.0).astype(np.float32)
    filtered_visibility = bundle.visibility.copy()
    target_visible = bundle.visibility[:, target_ids]
    rejected_targets = target_visible & (
        (motion_confidence[:, target_ids] < float(config.min_motion_confidence))
        | (final_confidence[:, target_ids] < float(config.min_final_confidence))
    )
    filtered_visibility[:, target_ids] &= ~rejected_targets
    final_confidence[~filtered_visibility] = 0.0

    rejected_by_time = rejected_targets.sum(axis=1).astype(int)
    visible_before_by_time = target_visible.sum(axis=1).astype(int)
    visible_after_by_time = filtered_visibility[:, target_ids].sum(axis=1).astype(int)
    for stats in frame_stats:
        t = int(stats["reverse_time"])
        stats["visible_target_points_before"] = int(visible_before_by_time[t])
        stats["visible_target_points_after"] = int(visible_after_by_time[t])
        stats["filtered_target_points"] = int(rejected_by_time[t])

    confidence_components = dict(bundle.confidence_components)
    if config.store_debug_arrays:
        confidence_components["tracker_confidence_input"] = base_confidence
        confidence_components["anchor_motion_confidence"] = motion_confidence
        confidence_components["anchor_motion_residual_px"] = residual_px

    metadata = dict(bundle.metadata)
    metadata["anchor_motion_filter"] = {
        "schema": "avt_anchor_motion_filter_v1",
        "enabled": True,
        "target_sources": list(config.target_sources),
        "anchor_count": int(anchor_ids.size),
        "target_count": int(target_ids.size),
        "thresholds": {
            "min_anchor_matches": int(config.min_anchor_matches),
            "min_anchor_inliers": int(config.min_anchor_inliers),
            "min_inlier_ratio": float(config.min_inlier_ratio),
            "ransac_reproj_threshold_px": float(config.ransac_reproj_threshold_px),
            "residual_scale_px": float(config.residual_scale_px),
            "min_motion_confidence": float(config.min_motion_confidence),
            "min_final_confidence": float(config.min_final_confidence),
            "fallback_confidence": float(config.fallback_confidence),
        },
        "summary": {
            "processed_transition_count": max(0, frame_count - 1),
            "valid_transition_count": int(sum(1 for item in frame_stats if item["model_valid"])),
            "visible_target_points_before": int(target_visible.sum()),
            "visible_target_points_after": int(filtered_visibility[:, target_ids].sum()),
            "filtered_target_points": int(rejected_targets.sum()),
        },
        "frames": frame_stats,
    }

    return TrackingBundle(
        tracks=bundle.tracks,
        visibility=filtered_visibility,
        tracker=bundle.tracker,
        confidence=final_confidence,
        confidence_components=confidence_components,
        metadata=metadata,
    )


def _query_ids(
    queries: list[QueryPoint],
    query_count: int,
    sources: set[str],
) -> np.ndarray:
    return np.array(
        [
            query.id
            for query in queries
            if query.source in sources and 0 <= int(query.id) < query_count
        ],
        dtype=np.int64,
    )


def _base_confidence(bundle: TrackingBundle) -> np.ndarray:
    if bundle.confidence is None:
        confidence = bundle.visibility.astype(np.float32)
    else:
        confidence = np.asarray(bundle.confidence, dtype=np.float32).copy()
    confidence = np.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0)
    confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)
    confidence[~bundle.visibility] = 0.0
    return confidence


def _estimate_adjacent_anchor_motion(
    bundle: TrackingBundle,
    anchor_ids: np.ndarray,
    *,
    src_time: int,
    dst_time: int,
    config: AnchorMotionFilterConfig,
) -> _MotionEstimate:
    visible = bundle.visibility[src_time, anchor_ids] & bundle.visibility[dst_time, anchor_ids]
    ids = anchor_ids[visible]
    if ids.size:
        src = bundle.tracks[src_time, ids].astype(np.float32)
        dst = bundle.tracks[dst_time, ids].astype(np.float32)
        finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
        src = src[finite]
        dst = dst[finite]
    else:
        src = np.empty((0, 2), dtype=np.float32)
        dst = np.empty((0, 2), dtype=np.float32)

    match_count = int(len(src))
    if match_count >= int(config.min_anchor_matches):
        estimate = _estimate_affine_partial(src, dst, config)
        if estimate.valid:
            return estimate
        if not config.translation_fallback_enabled:
            return estimate

    if (
        config.translation_fallback_enabled
        and match_count >= int(config.min_translation_matches)
    ):
        return _estimate_median_translation(src, dst)

    return _MotionEstimate(
        transform=None,
        valid=False,
        method="none",
        reason="too_few_anchor_matches",
        match_count=match_count,
        inlier_count=0,
        inlier_ratio=None,
        reprojection_error_mean_px=None,
        reprojection_error_median_px=None,
    )


def _estimate_affine_partial(
    src: np.ndarray,
    dst: np.ndarray,
    config: AnchorMotionFilterConfig,
) -> _MotionEstimate:
    matrix, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(config.ransac_reproj_threshold_px),
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None or matrix.shape != (2, 3):
        return _MotionEstimate(
            transform=None,
            valid=False,
            method="affine_partial_ransac",
            reason="ransac_failed",
            match_count=int(len(src)),
            inlier_count=0,
            inlier_ratio=0.0,
            reprojection_error_mean_px=None,
            reprojection_error_median_px=None,
        )

    inlier_mask = (
        inliers.reshape(-1).astype(bool)
        if inliers is not None
        else np.ones(len(src), dtype=bool)
    )
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = float(inlier_count / max(1, len(src)))
    projected = _apply_affine(src, matrix)
    errors = np.linalg.norm(dst - projected, axis=1)
    inlier_errors = errors[inlier_mask]
    error_mean = _finite_mean(inlier_errors)
    error_median = _finite_median(inlier_errors)

    if inlier_count < int(config.min_anchor_inliers):
        reason = "too_few_anchor_inliers"
    elif inlier_ratio < float(config.min_inlier_ratio):
        reason = "low_anchor_inlier_ratio"
    else:
        reason = "ok"

    return _MotionEstimate(
        transform=matrix.astype(np.float32),
        valid=reason == "ok",
        method="affine_partial_ransac",
        reason=reason,
        match_count=int(len(src)),
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        reprojection_error_mean_px=error_mean,
        reprojection_error_median_px=error_median,
    )


def _estimate_median_translation(src: np.ndarray, dst: np.ndarray) -> _MotionEstimate:
    delta = dst - src
    translation = np.median(delta, axis=0).astype(np.float32)
    matrix = np.array(
        [[1.0, 0.0, translation[0]], [0.0, 1.0, translation[1]]],
        dtype=np.float32,
    )
    projected = _apply_affine(src, matrix)
    errors = np.linalg.norm(dst - projected, axis=1)
    return _MotionEstimate(
        transform=matrix,
        valid=True,
        method="median_translation",
        reason="ok",
        match_count=int(len(src)),
        inlier_count=int(len(src)),
        inlier_ratio=1.0,
        reprojection_error_mean_px=_finite_mean(errors),
        reprojection_error_median_px=_finite_median(errors),
    )


def _apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return points @ matrix[:, :2].T + matrix[:, 2]


def _residual_confidence(residual: np.ndarray, scale_px: float) -> np.ndarray:
    scale = max(float(scale_px), 1.0e-6)
    confidence = np.exp(-0.5 * np.square(residual.astype(np.float32) / scale))
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def _motion_estimate_json(reverse_time: int, estimate: _MotionEstimate) -> dict[str, Any]:
    return {
        "reverse_time": int(reverse_time),
        "source_reverse_time": int(reverse_time - 1),
        "model_valid": bool(estimate.valid),
        "method": estimate.method,
        "reason": estimate.reason,
        "anchor_matches": int(estimate.match_count),
        "anchor_inliers": int(estimate.inlier_count),
        "anchor_inlier_ratio": _json_float(estimate.inlier_ratio),
        "anchor_reprojection_error_mean_px": _json_float(
            estimate.reprojection_error_mean_px
        ),
        "anchor_reprojection_error_median_px": _json_float(
            estimate.reprojection_error_median_px
        ),
    }


def _finite_mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.mean(values))


def _finite_median(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.median(values))


def _json_float(value: float | None) -> float | None:
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)
