from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..schema import QueryPoint, TrackerInfo


@dataclass
class TrackingBundle:
    """Backend-neutral point tracks in reversed-video time."""

    tracks: np.ndarray
    visibility: np.ndarray
    tracker: TrackerInfo
    confidence: np.ndarray | None = None
    confidence_components: dict[str, np.ndarray] = field(default_factory=dict)

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
        if self.confidence is not None and self.confidence.shape != (frame_count, query_count):
            raise ValueError(
                f"confidence shape {self.confidence.shape} does not match "
                f"({frame_count}, {query_count})"
            )
        for name, component in self.confidence_components.items():
            if component.shape != (frame_count, query_count):
                raise ValueError(
                    f"confidence component {name!r} shape {component.shape} does not match "
                    f"({frame_count}, {query_count})"
                )


class PointTracker(Protocol):
    """Interface for any point tracker used by AVT inverse tracking."""

    def track(self, frames_rgb: np.ndarray, queries: list[QueryPoint]) -> TrackingBundle:
        """Track query points through frames_rgb.

        frames_rgb is ordered in reversed-video time. QueryPoint.reverse_time is
        the frame index where that point first exists in this reversed clip.
        """
