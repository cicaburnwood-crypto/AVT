"""Tests for the 4-stage pipeline split (preprocess / extract / track / combine).

These cover the new stage boundaries and the PointExtractor seam. The existing
end-to-end contracts live in test_inverse_viewer.py and remain unchanged.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from avt.config import InverseTrackConfig
from avt.inverse import build_queries, run_inverse_tracking
from avt.io import read_frame_records
from avt.pipeline import PreparedWindow, SiftQueryExtractor
from avt.pipeline.preprocess import build_windows, prepare_window
from avt.querying import QueryConfig, SiftCaptureConfig
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
        query_config=QueryConfig(mode="avt", sift=SiftCaptureConfig(enabled=False)),
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


def test_sift_extractor_matches_build_queries(tmp_path: Path) -> None:
    frames_root = tmp_path / "frames"
    write_frames(frames_root)
    records = read_frame_records(frames_root, "image_dir")
    config = _avt_config()
    prepared = prepare_window(frames_root.resolve(), records, WindowSpec(start=0, end=4), config)

    via_extractor = SiftQueryExtractor().extract(prepared, config)
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


def test_backward_compat_reexports() -> None:
    import avt.config as config_mod
    import avt.inverse as inverse_mod
    import avt.pipeline.combine as combine_mod
    import avt.pipeline.extract as extract_mod

    # Names re-exported from inverse are the very same objects as their new homes.
    assert inverse_mod.InverseTrackConfig is config_mod.InverseTrackConfig
    assert inverse_mod.build_queries is extract_mod.build_queries
    assert inverse_mod.reference_mask is combine_mod.reference_mask
    assert inverse_mod.write_window_artifacts is combine_mod.write_window_artifacts
