from __future__ import annotations

from dataclasses import dataclass, field

from .querying import QueryConfig


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
