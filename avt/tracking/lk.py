from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..schema import QueryPoint, TrackerInfo
from .base import TrackingBundle


@dataclass
class LKTracker:
    """OpenCV Lucas-Kanade tracker implementing the AVT backend protocol."""

    win_size: int = 21
    max_level: int = 3

    def track(self, frames_rgb: np.ndarray, queries: list[QueryPoint]) -> TrackingBundle:
        if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
            raise ValueError("frames_rgb must have shape [T,H,W,3]")
        frame_count, height, width = frames_rgb.shape[:3]
        query_count = len(queries)
        tracks = np.full((frame_count, query_count, 2), np.nan, dtype=np.float32)
        visibility = np.zeros((frame_count, query_count), dtype=bool)

        by_time: dict[int, list[QueryPoint]] = {}
        for query in queries:
            if not 0 <= query.reverse_time < frame_count:
                raise ValueError(f"query reverse_time outside window: {query}")
            by_time.setdefault(query.reverse_time, []).append(query)

        lk_params = dict(
            winSize=(self.win_size, self.win_size),
            maxLevel=self.max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames_rgb]
        for t in range(frame_count):
            for query in by_time.get(t, []):
                tracks[t, query.id] = [query.x, query.y]
                visibility[t, query.id] = True

            if t == frame_count - 1:
                break
            active = np.flatnonzero(visibility[t])
            if active.size == 0:
                continue
            prev_points = tracks[t, active].reshape(-1, 1, 2).astype(np.float32)
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                gray[t], gray[t + 1], prev_points, None, **lk_params
            )
            if next_points is None or status is None:
                continue
            status = status.reshape(-1).astype(bool)
            for idx, point, ok in zip(active, next_points.reshape(-1, 2), status):
                x, y = float(point[0]), float(point[1])
                if ok and 0 <= x < width and 0 <= y < height:
                    tracks[t + 1, idx] = [x, y]
                    visibility[t + 1, idx] = True

        bundle = TrackingBundle(
            tracks=tracks,
            visibility=visibility,
            tracker=TrackerInfo(
                name="opencv-lk",
                parameters={"win_size": self.win_size, "max_level": self.max_level},
            ),
        )
        bundle.validate(frame_count, query_count)
        return bundle
