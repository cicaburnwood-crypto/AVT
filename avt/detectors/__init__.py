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
    from ..querying import AnchorSamplingConfig, QueryConfig, QuerySamplingConfig

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
    sampling_params: "QuerySamplingConfig | AnchorSamplingConfig | None" = None,
) -> KeypointDetector:
    """Construct the detector named by ``config.detector``.

    ``sampling_params`` carries per-tier tuning for anchor vs footprint sampling.
    Only the SIFT detector consumes those detector-specific thresholds.
    """

    name = getattr(config, "detector", "sift")
    if name == "sift":
        from .sift import SiftDetector

        return SiftDetector(
            sampling_params if sampling_params is not None else config.sampling,
            config.sampling,
        )
    if name == "orb":
        from .orb import OrbDetector

        return OrbDetector(config.orb, config.sampling)
    if name == "superpoint":
        from .superpoint import SuperPointSuperGlueDetector

        return SuperPointSuperGlueDetector(config.superpoint)
    if name == "xfeat":
        from .xfeat import XFeatDetector

        return XFeatDetector(config.xfeat)
    raise ValueError(f"Unknown detector: {name!r}. Choose from {DETECTORS}.")
