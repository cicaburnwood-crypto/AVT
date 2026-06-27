"""SIFT detector - wraps the exact logic AVT used before detectors were pluggable.

Output is byte-identical to the previous inline ``cv2.SIFT_create`` +
``detectAndCompute`` path so the default pipeline behavior does not change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ..querying import SiftAnchorConfig, SiftCaptureConfig


class SiftDetector:
    """``params`` carries the per-tier SIFT tuning (anchors vs crumbs); CLAHE
    settings always come from the crumb-level ``sift_config`` (unchanged)."""

    def __init__(
        self,
        params: "SiftCaptureConfig | SiftAnchorConfig",
        sift_config: "SiftCaptureConfig",
    ) -> None:
        self._sift_config = sift_config
        self._detector = cv2.SIFT_create(
            nfeatures=0,
            nOctaveLayers=params.n_octave_layers,
            contrastThreshold=params.contrast_threshold,
            edgeThreshold=params.edge_threshold,
            sigma=params.sigma,
        )

    def detect(
        self, frames_rgb: np.ndarray, reverse_time: int, mask: np.ndarray | None
    ) -> list[cv2.KeyPoint]:
        gray = cv2.cvtColor(frames_rgb[reverse_time], cv2.COLOR_RGB2GRAY)
        gray = _local_equalize(gray, self._sift_config)
        keypoints, _ = self._detector.detectAndCompute(gray, mask)
        return list(keypoints) if keypoints else []


def _local_equalize(gray: np.ndarray, sift_config: "SiftCaptureConfig") -> np.ndarray:
    if not sift_config.use_clahe:
        return gray
    tile = max(1, int(sift_config.clahe_tile_grid_size))
    clahe = cv2.createCLAHE(
        clipLimit=float(sift_config.clahe_clip_limit),
        tileGridSize=(tile, tile),
    )
    return clahe.apply(gray)
