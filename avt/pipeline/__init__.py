"""AVT inverse-tracking pipeline, split into four standalone stages.

Stage 1 preprocess  -> PreparedWindow / prepare_window / build_windows
Stage 2 extract     -> PointExtractor / QueryPointExtractor / build_queries
Stage 3 track       -> PointTracker / run_tracking
Stage 3.5 filter    -> PointTrackFilter / run_track_filter
Stage 4 combine     -> reference_mask / write_window_artifacts

The orchestrator that wires these together lives in ``avt.inverse`` as
``run_inverse_tracking`` (kept there for backward compatibility).
"""

from __future__ import annotations

from .combine import reference_mask, write_window_artifacts
from .extract import PointExtractor, QueryPointExtractor, build_queries
from .filter import AnchorMotionFootprintFilter, PointTrackFilter, run_track_filter
from .preprocess import PreparedWindow, build_windows, prepare_window
from .track import PointTracker, TrackingBundle, run_tracking

__all__ = [
    "PreparedWindow",
    "prepare_window",
    "build_windows",
    "PointExtractor",
    "QueryPointExtractor",
    "build_queries",
    "PointTracker",
    "TrackingBundle",
    "run_tracking",
    "PointTrackFilter",
    "AnchorMotionFootprintFilter",
    "run_track_filter",
    "reference_mask",
    "write_window_artifacts",
]
