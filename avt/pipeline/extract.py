"""Stage 2 - point extractor.

Turns a PreparedWindow into the list of query points that the tracker will
follow. The ``PointExtractor`` Protocol mirrors ``PointTracker`` so alternative
detectors (e.g. ORB / XFeat) can be dropped in the same way tracking backends
are, without touching the orchestrator. ``SiftQueryExtractor`` is the default
and preserves today's SIFT/VENTURA/AVT behavior.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..config import InverseTrackConfig
from ..querying import (
    build_avt_queries,
    build_sift_queries,
    build_ventura_queries,
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
    want_ventura = mode in {"ventura", "avt+sift"}
    want_sift = mode == "sift"

    if want_ventura:
        if frames_rgb is None:
            raise ValueError("frames_rgb is required for VENTURA query capture")
        queries.extend(
            build_ventura_queries(
                frames_rgb=frames_rgb,
                query_config=config.query_config,
                start_id=len(queries),
            )
        )
    elif want_sift:
        if frames_rgb is None:
            raise ValueError("frames_rgb is required for SIFT query capture")
        queries.extend(
            build_sift_queries(
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

    if not queries and mode in {"ventura", "sift", "avt+sift"}:
        raise ValueError("No VENTURA/SIFT query points were generated")

    if not queries:
        raise ValueError("No query points were generated")
    return queries


class PointExtractor(Protocol):
    """Interface for any query-point extractor used by AVT inverse tracking."""

    def extract(
        self, prepared: PreparedWindow, config: InverseTrackConfig
    ) -> list[QueryPoint]:
        """Return query points for ``prepared`` in reversed-video time."""


class SiftQueryExtractor:
    """Default extractor: SIFT/VENTURA/AVT query capture (current behavior)."""

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
