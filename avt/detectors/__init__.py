"""Pluggable keypoint detectors for the AVT point extractor (Stage 2).

``build_detector`` selects one of ``sift | orb | superpoint | xfeat`` from a
``QueryConfig`` and lazy-imports only the chosen implementation, so SIFT/ORB
work with no torch installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import KeypointDetector, filter_keypoints_by_mask, make_keypoint
from .config import OrbDetectorConfig, SuperPointConfig, XFeatConfig

if TYPE_CHECKING:
    from ..querying import QueryConfig, SiftAnchorConfig, SiftCaptureConfig

DETECTORS = ("sift", "orb", "superpoint", "xfeat")

__all__ = [
    "KeypointDetector",
    "make_keypoint",
    "filter_keypoints_by_mask",
    "OrbDetectorConfig",
    "SuperPointConfig",
    "XFeatConfig",
    "DETECTORS",
    "build_detector",
]


def build_detector(
    config: "QueryConfig",
    sift_params: "SiftCaptureConfig | SiftAnchorConfig | None" = None,
) -> KeypointDetector:
    """Construct the detector named by ``config.detector``.

    ``sift_params`` carries the per-tier SIFT tuning (anchors vs crumbs) and is
    ignored by the non-SIFT detectors.
    """

    name = getattr(config, "detector", "sift")
    if name == "sift":
        from .sift import SiftDetector

        return SiftDetector(sift_params if sift_params is not None else config.sift, config.sift)
    if name == "orb":
        from .orb import OrbDetector

        return OrbDetector(config.orb, config.sift)
    if name == "superpoint":
        from .superpoint import SuperPointSuperGlueDetector

        return SuperPointSuperGlueDetector(config.superpoint)
    if name == "xfeat":
        from .xfeat import XFeatDetector

        return XFeatDetector(config.xfeat)
    raise ValueError(f"Unknown detector: {name!r}. Choose from {DETECTORS}.")
