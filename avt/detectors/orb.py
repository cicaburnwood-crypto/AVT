"""ORB detector (OpenCV) - fast, license-free corner features.

Like SIFT it takes an OpenCV mask directly and returns cv2.KeyPoints with a
``response`` (Harris/FAST score) used for top-N selection. Descriptors are not
computed (only locations are needed downstream).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from .config import OrbDetectorConfig

if TYPE_CHECKING:
    from ..querying import SiftCaptureConfig

_SCORE_TYPES = {
    "harris": cv2.ORB_HARRIS_SCORE,
    "fast": cv2.ORB_FAST_SCORE,
}


class OrbDetector:
    def __init__(
        self,
        config: OrbDetectorConfig,
        sift_config: "SiftCaptureConfig | None" = None,
    ) -> None:
        self._config = config
        self._sift_config = sift_config
        self._detector = cv2.ORB_create(
            nfeatures=int(config.nfeatures),
            scaleFactor=float(config.scale_factor),
            nlevels=int(config.n_levels),
            edgeThreshold=int(config.edge_threshold),
            firstLevel=int(config.first_level),
            WTA_K=int(config.wta_k),
            scoreType=_SCORE_TYPES.get(config.score_type, cv2.ORB_HARRIS_SCORE),
            patchSize=int(config.patch_size),
            fastThreshold=int(config.fast_threshold),
        )

    def detect(
        self, frames_rgb: np.ndarray, reverse_time: int, mask: np.ndarray | None
    ) -> list[cv2.KeyPoint]:
        gray = cv2.cvtColor(frames_rgb[reverse_time], cv2.COLOR_RGB2GRAY)
        gray = self._maybe_equalize(gray)
        keypoints = self._detector.detect(gray, mask)
        return list(keypoints) if keypoints else []

    def _maybe_equalize(self, gray: np.ndarray) -> np.ndarray:
        sc = self._sift_config
        if not self._config.use_clahe or sc is None or not sc.use_clahe:
            return gray
        tile = max(1, int(sc.clahe_tile_grid_size))
        clahe = cv2.createCLAHE(
            clipLimit=float(sc.clahe_clip_limit),
            tileGridSize=(tile, tile),
        )
        return clahe.apply(gray)
