from __future__ import annotations

from dataclasses import dataclass, field

from .querying import QueryConfig


@dataclass
class AnchorMotionFilterConfig:
    """Use full-frame anchors as a frame-motion prior for footprint tracks."""

    enabled: bool = True
    target_sources: tuple[str, ...] = ("footprint",)
    min_anchor_matches: int = 8
    min_anchor_inliers: int = 6
    min_inlier_ratio: float = 0.35
    ransac_reproj_threshold_px: float = 4.0
    residual_scale_px: float = 8.0
    min_motion_confidence: float = 0.20
    min_final_confidence: float = 0.25
    fallback_confidence: float = 1.0
    translation_fallback_enabled: bool = True
    min_translation_matches: int = 3
    store_debug_arrays: bool = True


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
    save_reverse_video: bool = False
    save_path_mask: bool = False
    path_support_enabled: bool = True
    path_support_min_points: int = 32
    path_support_fraction: int = 6
    anchor_motion_filter: AnchorMotionFilterConfig = field(
        default_factory=AnchorMotionFilterConfig
    )
