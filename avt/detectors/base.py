"""Keypoint detector abstraction shared by the AVT point extractor (Stage 2).

Every detector proposes keypoint *locations* (descriptors are discarded - the
tracker does the tracking). To maximize reuse, all detectors return
``list[cv2.KeyPoint]`` so the existing top-N picking (`_pick_sift_keypoints`)
and ``QueryPoint`` construction in ``querying.py`` work unchanged.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np


class KeypointDetector(Protocol):
    """Proposes keypoints for one frame of a reversed-video window.

    ``frames_rgb`` is the full ``[T, H, W, 3]`` reversed clip; ``reverse_time``
    selects the frame to detect on (detectors that match across frames - e.g.
    SuperPoint+SuperGlue - may also read neighbors). ``mask`` is the footprint
    or anchor region (uint8, nonzero = keep); ``None`` means full frame.
    """

    def detect(
        self, frames_rgb: np.ndarray, reverse_time: int, mask: np.ndarray | None
    ) -> list[cv2.KeyPoint]:
        ...


def make_keypoint(
    x: float,
    y: float,
    response: float,
    *,
    size: float = 1.0,
    angle: float = -1.0,
    octave: int = 0,
    class_id: int = -1,
) -> cv2.KeyPoint:
    """Build a cv2.KeyPoint for detectors that lack SIFT-style attributes.

    Attributes are set after construction for cross-version OpenCV safety.
    """

    kp = cv2.KeyPoint(float(x), float(y), float(size))
    kp.response = float(response)
    kp.angle = float(angle)
    kp.octave = int(octave)
    kp.class_id = int(class_id)
    return kp


def filter_keypoints_by_mask(
    keypoints: list[cv2.KeyPoint], mask: np.ndarray | None
) -> list[cv2.KeyPoint]:
    """Keep only keypoints whose rounded pixel falls inside ``mask`` (>0).

    For detectors (learned ones) that cannot take an OpenCV mask directly.
    """

    if mask is None:
        return list(keypoints)
    height, width = mask.shape[:2]
    kept: list[cv2.KeyPoint] = []
    for kp in keypoints:
        xi = int(round(kp.pt[0]))
        yi = int(round(kp.pt[1]))
        if 0 <= yi < height and 0 <= xi < width and mask[yi, xi] > 0:
            kept.append(kp)
    return kept
