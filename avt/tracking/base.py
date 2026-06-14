from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..schema import QueryPoint, TrackerInfo


@dataclass
class TrackingBundle:
    """Backend-neutral point tracks in reversed-video time."""

    tracks: np.ndarray
    visibility: np.ndarray
    tracker: TrackerInfo

    def validate(self, frame_count: int, query_count: int) -> None:
        if self.tracks.shape != (frame_count, query_count, 2):
            raise ValueError(
                f"tracks shape {self.tracks.shape} does not match "
                f"({frame_count}, {query_count}, 2)"
            )
        if self.visibility.shape != (frame_count, query_count):
            raise ValueError(
                f"visibility shape {self.visibility.shape} does not match "
                f"({frame_count}, {query_count})"
            )


class PointTracker(Protocol):
    """Interface for any point tracker used by AVT inverse tracking."""

    def track(self, frames_rgb: np.ndarray, queries: list[QueryPoint]) -> TrackingBundle:
        """Track query points through frames_rgb.

        frames_rgb is ordered in reversed-video time. QueryPoint.reverse_time is
        the frame index where that point first exists in this reversed clip.
        """
