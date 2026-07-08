from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import AnchorMotionConfig
from .schema import QueryPoint
from .tracking.base import TrackingBundle


@dataclass(frozen=True)
class AnchorMotionProjection:
    mother_point: np.ndarray
    projected_points_reverse: np.ndarray
    transforms_reverse: np.ndarray
    valid_reverse: np.ndarray
    adjacent_valid_reverse: np.ndarray
    adjacent_inlier_count_reverse: np.ndarray
    adjacent_inlier_ratio_reverse: np.ndarray
    adjacent_reprojection_error_mean_reverse: np.ndarray
    adjacent_reprojection_error_median_reverse: np.ndarray


def estimate_anchor_motion_projection(
    bundle: TrackingBundle,
    queries: list[QueryPoint],
    width: int,
    height: int,
    config: AnchorMotionConfig,
) -> AnchorMotionProjection:
    """Estimate reverse-time frame motion from anchors and project the mother point.

    The mother point lives in reverse frame 0, which is the original window's
    last frame. Each adjacent affine maps reverse frame t-1 to reverse frame t.
    """

    frame_count = int(bundle.tracks.shape[0])
    mother = np.array(
        [
            _ratio_to_pixel(config.mother_x_ratio, width),
            _ratio_to_pixel(config.mother_y_ratio, height),
        ],
        dtype=np.float32,
    )

    transforms = np.full((frame_count, 2, 3), np.nan, dtype=np.float32)
    projected = np.full((frame_count, 2), np.nan, dtype=np.float32)
    valid = np.zeros(frame_count, dtype=bool)
    adjacent_valid = np.zeros(frame_count, dtype=bool)
    inlier_count = np.zeros(frame_count, dtype=np.int32)
    inlier_ratio = np.zeros(frame_count, dtype=np.float32)
    error_mean = np.full(frame_count, np.nan, dtype=np.float32)
    error_median = np.full(frame_count, np.nan, dtype=np.float32)

    identity = np.eye(3, dtype=np.float64)
    cumulative = identity.copy()
    transforms[0] = identity[:2]
    projected[0] = mother
    valid[0] = True
    adjacent_valid[0] = True

    anchor_ids = np.array([q.id for q in queries if q.source == "anchor"], dtype=np.int64)
    min_matches = max(3, int(config.min_matches))

    cumulative_valid = True
    for t in range(1, frame_count):
        estimate = _estimate_adjacent_affine(
            bundle,
            anchor_ids,
            t - 1,
            t,
            min_matches=min_matches,
            ransac_reproj_threshold=float(config.ransac_reproj_threshold),
        )
        if estimate is None:
            cumulative_valid = False
            continue

        affine, mask, errors = estimate
        adjacent_valid[t] = True
        inliers = mask.astype(bool)
        inlier_count[t] = int(np.count_nonzero(inliers))
        inlier_ratio[t] = float(np.count_nonzero(inliers) / len(inliers))
        if errors.size:
            error_mean[t] = float(np.mean(errors))
            error_median[t] = float(np.median(errors))

        cumulative = _to_homogeneous(affine) @ cumulative
        if cumulative_valid:
            transforms[t] = cumulative[:2].astype(np.float32)
            projected[t] = _project_point(cumulative, mother)
            valid[t] = True

    return AnchorMotionProjection(
        mother_point=mother,
        projected_points_reverse=projected,
        transforms_reverse=transforms,
        valid_reverse=valid,
        adjacent_valid_reverse=adjacent_valid,
        adjacent_inlier_count_reverse=inlier_count,
        adjacent_inlier_ratio_reverse=inlier_ratio,
        adjacent_reprojection_error_mean_reverse=error_mean,
        adjacent_reprojection_error_median_reverse=error_median,
    )


def projection_mask(
    projection: AnchorMotionProjection,
    height: int,
    width: int,
    radius_px: int,
) -> np.ndarray:
    points = projection.projected_points_reverse[
        projection.valid_reverse & np.isfinite(projection.projected_points_reverse).all(axis=1)
    ]
    alpha = np.zeros((height, width), dtype=np.uint8)
    if points.size:
        rounded = np.round(points).astype(np.int32)
        rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
        radius = max(1, int(radius_px))
        if len(rounded) >= 2:
            cv2.polylines(alpha, [rounded.reshape(-1, 1, 2)], False, 92, radius * 2)
        for x, y in rounded:
            cv2.circle(alpha, (int(x), int(y)), radius, 128, -1)

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = 32
    rgba[..., 1] = 199
    rgba[..., 2] = 230
    rgba[..., 3] = alpha
    return rgba


def projection_metadata(projection: AnchorMotionProjection) -> dict[str, object]:
    valid_count = int(np.count_nonzero(projection.valid_reverse))
    adjacent_valid_count = int(np.count_nonzero(projection.adjacent_valid_reverse))
    errors = projection.adjacent_reprojection_error_mean_reverse
    finite_errors = errors[np.isfinite(errors)]
    return {
        "method": "anchor_affine_motion_projected_mother_point",
        "mother_point_reverse_time": 0,
        "mother_point_xy": projection.mother_point.astype(float).tolist(),
        "valid_projected_frame_count": valid_count,
        "adjacent_valid_motion_count": adjacent_valid_count,
        "adjacent_motion_error_mean": (
            float(np.mean(finite_errors)) if finite_errors.size else None
        ),
        "adjacent_motion_error_median": (
            float(np.median(finite_errors)) if finite_errors.size else None
        ),
    }


def _estimate_adjacent_affine(
    bundle: TrackingBundle,
    anchor_ids: np.ndarray,
    src_time: int,
    dst_time: int,
    *,
    min_matches: int,
    ransac_reproj_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if anchor_ids.size < min_matches:
        return None
    visible = bundle.visibility[src_time, anchor_ids] & bundle.visibility[dst_time, anchor_ids]
    ids = anchor_ids[visible]
    if ids.size < min_matches:
        return None
    src = bundle.tracks[src_time, ids].astype(np.float32)
    dst = bundle.tracks[dst_time, ids].astype(np.float32)
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[finite]
    dst = dst[finite]
    if len(src) < min_matches:
        return None

    affine, mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_reproj_threshold),
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if affine is None or mask is None:
        return None
    mask = mask.reshape(-1).astype(bool)
    if int(np.count_nonzero(mask)) < min_matches:
        return None
    projected = _apply_affine(affine, src[mask])
    errors = np.linalg.norm(projected - dst[mask], axis=1).astype(np.float32)
    return affine.astype(np.float64), mask, errors


def _apply_affine(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    ones = np.ones((len(points), 1), dtype=np.float32)
    homogeneous = np.concatenate([points.astype(np.float32), ones], axis=1)
    return (affine.astype(np.float32) @ homogeneous.T).T


def _project_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogeneous = np.array([float(point[0]), float(point[1]), 1.0], dtype=np.float64)
    projected = transform @ homogeneous
    return projected[:2].astype(np.float32)


def _to_homogeneous(affine: np.ndarray) -> np.ndarray:
    transform = np.eye(3, dtype=np.float64)
    transform[:2] = affine
    return transform


def _ratio_to_pixel(ratio: float, size: int) -> float:
    if size <= 0:
        raise ValueError("image dimensions must be positive")
    return float(np.clip(float(ratio), 0.0, 1.0) * (size - 1))
