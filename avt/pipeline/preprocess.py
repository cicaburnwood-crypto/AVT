"""Stage 1 - preprocess.

Frame-domain preparation: load a window's frames, reverse them for inverse
tracking, and resolve the window/seed geometry shared by later stages. This
stage is deliberately detector-agnostic; per-detector image enhancement
(grayscale + CLAHE) lives inside the point extractor (Stage 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import InverseTrackConfig
from ..io import load_frame_window
from ..querying import align_virtual_robot_to_image
from ..schema import FrameRecord, WindowSpec


@dataclass
class PreparedWindow:
    """Detector-agnostic frame bundle for a single tracking window."""

    window: WindowSpec
    frames_rgb: np.ndarray       # chronological [T, H, W, 3]
    frames_reverse: np.ndarray   # reversed [T, H, W, 3]
    width: int
    height: int


def build_windows(frame_count: int, config: InverseTrackConfig) -> list[WindowSpec]:
    if config.window_size <= 1:
        raise ValueError("window_size must be greater than 1")
    if config.window_step <= 0:
        raise ValueError("window_step must be positive")
    windows = [
        WindowSpec(start=start, end=min(start + config.window_size, frame_count))
        for start in range(0, frame_count, config.window_step)
        if start + 2 <= frame_count
    ]
    windows = [win for win in windows if win.frame_count >= 2]
    if config.max_windows is not None:
        windows = windows[: config.max_windows]
    return windows


def _avt_seed_ratios(
    config: InverseTrackConfig, width: int, height: int
) -> tuple[float, float, float]:
    alignment = align_virtual_robot_to_image(
        height=height,
        width=width,
        robot=config.query_config.robot,
    )
    return (
        alignment.seed_y_ratio if config.seed_y_ratio is None else config.seed_y_ratio,
        alignment.seed_x_min_ratio if config.seed_x_min_ratio is None else config.seed_x_min_ratio,
        alignment.seed_x_max_ratio if config.seed_x_max_ratio is None else config.seed_x_max_ratio,
    )


def prepare_window(
    source_root: Path,
    frame_records: list[FrameRecord],
    window: WindowSpec,
    config: InverseTrackConfig,
) -> PreparedWindow:
    """Load a window's frames and produce the reversed-time bundle."""

    frames = load_frame_window(source_root, frame_records, window.start, window.end)
    frames_reverse = frames[::-1].copy()
    h, w = frames.shape[1:3]
    return PreparedWindow(
        window=window,
        frames_rgb=frames,
        frames_reverse=frames_reverse,
        width=int(w),
        height=int(h),
    )
