"""Stage 2 - query-point extractor.

Turns a PreparedWindow into the list of query points that the tracker will
follow. The ``PointExtractor`` Protocol mirrors ``PointTracker`` so alternative
detectors (e.g. ORB / XFeat) can be dropped in the same way tracking backends
are, without touching the orchestrator. ``QueryPointExtractor`` is the default
and preserves today's anchor/footprint/AVT behavior.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..config import InverseTrackConfig
from ..querying import (
    DETECTOR_SAMPLING_MODES,
    build_anchor_footprint_queries,
    build_avt_queries,
    build_footprint_queries,
)
from ..schema import QueryPoint
from .preprocess import PreparedWindow, _avt_seed_ratios


def build_queries(
    width: int,
    height: int,
    frame_count: int,
    config: InverseTrackConfig,
    frames_rgb: np.ndarray | None = None,
) -> list[QueryPoint]:
    queries: list[QueryPoint] = []
    mode = config.query_config.mode
    seed_y_ratio, seed_x_min_ratio, seed_x_max_ratio = _avt_seed_ratios(config, width, height)
    want_anchor_footprint = mode in {"anchor_footprint", "avt+footprint"}
    want_footprint = mode == "footprint"

    if want_anchor_footprint:
        if frames_rgb is None:
            raise ValueError("frames_rgb is required for anchor/footprint query capture")
        queries.extend(
            build_anchor_footprint_queries(
                frames_rgb=frames_rgb,
                query_config=config.query_config,
                start_id=len(queries),
            )
        )
    elif want_footprint:
        if frames_rgb is None:
            raise ValueError("frames_rgb is required for footprint query capture")
        queries.extend(
            build_footprint_queries(
                frames_rgb=frames_rgb,
                query_config=config.query_config,
                start_id=len(queries),
            )
        )

    if mode == "avt":
        queries.extend(
            build_avt_queries(
                width=width,
                height=height,
                frame_count=frame_count,
                query_stride=config.query_stride,
                seed_count=config.seed_count,
                seed_y_ratio=seed_y_ratio,
                seed_x_min_ratio=seed_x_min_ratio,
                seed_x_max_ratio=seed_x_max_ratio,
                start_id=len(queries),
            )
        )

    if not queries and mode in DETECTOR_SAMPLING_MODES:
        raise ValueError("No detector-sampled query points were generated")

    if not queries:
        raise ValueError("No query points were generated")
    return queries


class PointExtractor(Protocol):
    """Interface for any query-point extractor used by AVT inverse tracking."""

    def extract(
        self, prepared: PreparedWindow, config: InverseTrackConfig
    ) -> list[QueryPoint]:
        """Return query points for ``prepared`` in reversed-video time."""


class QueryPointExtractor:
    """Default extractor: detector-sampled or AVT fixed query capture."""

    def extract(
        self, prepared: PreparedWindow, config: InverseTrackConfig
    ) -> list[QueryPoint]:
        return build_queries(
            prepared.width,
            prepared.height,
            len(prepared.frames_reverse),
            config,
            frames_rgb=prepared.frames_reverse,
        )
