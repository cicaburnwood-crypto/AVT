"""Stage 3 - track.

Stage 3 is already a clean standalone: any backend implementing the
``PointTracker`` Protocol (lk / cotracker / foundationpose / bootstap) tracks
the query points through the reversed frames. This module re-exports that
interface and provides ``run_tracking`` as the uniform stage entrypoint.
"""

from __future__ import annotations

from pathlib import Path

from ..schema import QueryPoint
from ..tracking.base import PointTracker, TrackingBundle
from .preprocess import PreparedWindow

__all__ = ["PointTracker", "TrackingBundle", "run_tracking"]


def run_tracking(
    prepared: PreparedWindow,
    queries: list[QueryPoint],
    tracker: PointTracker,
    output_dir: Path | None = None,
) -> TrackingBundle:
    """Track ``queries`` through ``prepared.frames_reverse`` and validate."""

    if hasattr(tracker, "set_window_context"):
        tracker.set_window_context(window=prepared.window, output_dir=output_dir)
    bundle = tracker.track(prepared.frames_reverse, queries)
    bundle.validate(len(prepared.frames_reverse), len(queries))
    return bundle
