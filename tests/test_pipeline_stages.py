"""Tests for the staged pipeline split.

These cover the stage boundaries and the pluggable extractor/filter seams. The
existing end-to-end contracts live in test_inverse_viewer.py and remain
unchanged.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from avt.config import AnchorMotionFilterConfig, InverseTrackConfig
from avt.inverse import build_queries, run_inverse_tracking
from avt.io import read_frame_records
from avt.pipeline import AnchorMotionFootprintFilter, PreparedWindow, QueryPointExtractor
from avt.pipeline.preprocess import build_windows, prepare_window
from avt.querying import QueryConfig, QuerySamplingConfig
from avt.schema import QueryPoint, TrackerInfo, WindowSpec
from avt.tracking.base import TrackingBundle


class FakeTracker:
    def track(self, frames_rgb: np.ndarray, queries: list[QueryPoint]) -> TrackingBundle:
        tracks = np.full((len(frames_rgb), len(queries), 2), np.nan, dtype=np.float32)
        visibility = np.zeros((len(frames_rgb), len(queries)), dtype=bool)
        for query in queries:
            tracks[query.reverse_time :, query.id] = [query.x, query.y]
            visibility[query.reverse_time :, query.id] = True
        return TrackingBundle(
            tracks=tracks,
            visibility=visibility,
            tracker=TrackerInfo(name="fake"),
        )


class FakeExtractor:
    """Minimal custom extractor to prove the Stage-2 seam end-to-end."""

    def __init__(self) -> None:
        self.calls = 0

    def extract(self, prepared: PreparedWindow, config: InverseTrackConfig) -> list[QueryPoint]:
        self.calls += 1
        return [QueryPoint(id=0, reverse_time=0, x=3.0, y=4.0, side=-1, source="avt")]


class FakeFilter:
    """Minimal custom filter to prove the Stage-3.5 seam end-to-end."""

    def __init__(self) -> None:
        self.calls = 0

    def filter(
        self,
        prepared: PreparedWindow,
        queries: list[QueryPoint],
        bundle: TrackingBundle,
        config: InverseTrackConfig,
    ) -> TrackingBundle:
        del prepared, queries, config
        self.calls += 1
        return bundle


def write_frames(root: Path, count: int = 6) -> None:
    root.mkdir()
    for idx in range(count):
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        cv2.circle(img, (12 + idx, 24), 4, (255, 255, 255), -1)
        cv2.imwrite(str(root / f"{idx:04d}.png"), img)


def _avt_config() -> InverseTrackConfig:
    return InverseTrackConfig(
        window_size=4,
        window_step=3,
        query_stride=2,
        seed_count=3,
        max_windows=1,
        query_config=QueryConfig(mode="avt", sampling=QuerySamplingConfig(enabled=False)),
        anchor_motion_filter=AnchorMotionFilterConfig(enabled=False),
    )


def test_prepare_window_shapes_and_reversal(tmp_path: Path) -> None:
    frames_root = tmp_path / "frames"
    write_frames(frames_root)
    records = read_frame_records(frames_root, "image_dir")
    prepared = prepare_window(frames_root.resolve(), records, WindowSpec(start=0, end=4), _avt_config())

    assert isinstance(prepared, PreparedWindow)
    assert prepared.frames_rgb.shape == (4, 48, 64, 3)
    assert prepared.frames_reverse.shape == (4, 48, 64, 3)
    assert (prepared.width, prepared.height) == (64, 48)
    # Reversed-time ordering: first reversed frame is the last chronological frame.
    assert np.array_equal(prepared.frames_reverse, prepared.frames_rgb[::-1])
    assert np.array_equal(prepared.frames_reverse[0], prepared.frames_rgb[-1])


def test_build_windows_matches_config(tmp_path: Path) -> None:
    windows = build_windows(6, _avt_config())
    assert len(windows) == 1
    assert (windows[0].start, windows[0].end) == (0, 4)


def test_query_point_extractor_matches_build_queries(tmp_path: Path) -> None:
    frames_root = tmp_path / "frames"
    write_frames(frames_root)
    records = read_frame_records(frames_root, "image_dir")
    config = _avt_config()
    prepared = prepare_window(frames_root.resolve(), records, WindowSpec(start=0, end=4), config)

    via_extractor = QueryPointExtractor().extract(prepared, config)
    via_function = build_queries(
        prepared.width,
        prepared.height,
        len(prepared.frames_reverse),
        config,
        frames_rgb=prepared.frames_reverse,
    )
    # QueryPoint is a frozen dataclass, so list equality is value equality.
    assert via_extractor == via_function


def test_orchestrator_accepts_custom_extractor(tmp_path: Path) -> None:
    frames_root = tmp_path / "frames"
    write_frames(frames_root)
    records = read_frame_records(frames_root, "image_dir")
    output_root = tmp_path / "out"
    extractor = FakeExtractor()

    windows = run_inverse_tracking(
        frames_root, records, output_root, FakeTracker(), _avt_config(), extractor=extractor
    )

    assert extractor.calls == 1
    assert len(windows) == 1
    arrays = np.load(output_root / "windows" / "seq_0_4" / "tracks.npz")
    # Exactly the single query the FakeExtractor emitted flowed through tracking.
    assert arrays["tracks_reverse"].shape[1] == 1


def test_orchestrator_accepts_custom_filter(tmp_path: Path) -> None:
    frames_root = tmp_path / "frames"
    write_frames(frames_root)
    records = read_frame_records(frames_root, "image_dir")
    output_root = tmp_path / "out"
    filterer = FakeFilter()

    windows = run_inverse_tracking(
        frames_root,
        records,
        output_root,
        FakeTracker(),
        _avt_config(),
        filterer=filterer,
    )

    assert filterer.calls == 1
    assert len(windows) == 1


def test_anchor_motion_filter_rejects_drifting_footprint() -> None:
    frame_count = 4
    anchor_points = [
        (10.0, 10.0),
        (25.0, 10.0),
        (40.0, 10.0),
        (55.0, 10.0),
        (10.0, 24.0),
        (25.0, 24.0),
        (40.0, 24.0),
        (55.0, 24.0),
    ]
    anchor_count = len(anchor_points)
    query_count = anchor_count + 2
    queries = [
        QueryPoint(id=i, reverse_time=0, x=x, y=y, side=-1, source="anchor")
        for i, (x, y) in enumerate(anchor_points)
    ]
    queries.extend(
        [
            QueryPoint(id=8, reverse_time=0, x=30.0, y=35.0, side=-1, source="footprint"),
            QueryPoint(id=9, reverse_time=0, x=35.0, y=35.0, side=1, source="footprint"),
        ]
    )
    tracks = np.zeros((frame_count, query_count, 2), dtype=np.float32)
    visibility = np.ones((frame_count, query_count), dtype=bool)
    confidence = np.full((frame_count, query_count), 0.9, dtype=np.float32)

    for t in range(frame_count):
        dx, dy = 2.0 * t, 1.0 * t
        for i, (x, y) in enumerate(anchor_points):
            tracks[t, i] = [x + dx, y + dy]
        tracks[t, 8] = [30.0 + dx, 35.0 + dy]
        tracks[t, 9] = [35.0 + dx + 25.0 * t, 35.0 + dy]

    prepared = PreparedWindow(
        window=WindowSpec(start=0, end=frame_count),
        frames_rgb=np.zeros((frame_count, 48, 64, 3), dtype=np.uint8),
        frames_reverse=np.zeros((frame_count, 48, 64, 3), dtype=np.uint8),
        width=64,
        height=48,
    )
    bundle = TrackingBundle(
        tracks=tracks,
        visibility=visibility,
        confidence=confidence,
        tracker=TrackerInfo(name="fake"),
    )
    config = InverseTrackConfig(
        anchor_motion_filter=AnchorMotionFilterConfig(
            enabled=True,
            min_motion_confidence=0.25,
            min_final_confidence=0.25,
            residual_scale_px=4.0,
        )
    )

    filtered = AnchorMotionFootprintFilter().filter(prepared, queries, bundle, config)

    assert filtered.visibility[:, :anchor_count].all()
    assert filtered.visibility[:, 8].tolist() == [True, True, True, True]
    assert filtered.visibility[:, 9].tolist() == [True, False, False, False]
    assert filtered.confidence is not None
    assert filtered.confidence[1, 9] == 0.0
    assert filtered.confidence_components["anchor_motion_confidence"][1, 8] > 0.8
    assert filtered.confidence_components["anchor_motion_confidence"][1, 9] < 0.25
    assert filtered.confidence_components["anchor_motion_residual_px"][1, 9] > 20.0
    assert (
        filtered.metadata["anchor_motion_filter"]["summary"]["filtered_target_points"]
        == 3
    )


def test_backward_compat_reexports() -> None:
    import avt.config as config_mod
    import avt.inverse as inverse_mod
    import avt.pipeline.combine as combine_mod
    import avt.pipeline.extract as extract_mod
    import avt.pipeline.filter as filter_mod

    # Names re-exported from inverse are the very same objects as their new homes.
    assert inverse_mod.InverseTrackConfig is config_mod.InverseTrackConfig
    assert inverse_mod.build_queries is extract_mod.build_queries
    assert inverse_mod.PointTrackFilter is filter_mod.PointTrackFilter
    assert inverse_mod.reference_mask is combine_mod.reference_mask
    assert inverse_mod.write_window_artifacts is combine_mod.write_window_artifacts
