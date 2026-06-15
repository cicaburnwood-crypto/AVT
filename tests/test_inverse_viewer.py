from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from avt.cli import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_VIEWER_ROOT,
    _create_unique_run_dir,
    build_parser,
)
from avt.inverse import InverseTrackConfig, build_queries, run_inverse_tracking
from avt.io import read_frame_records
from avt.schema import QueryPoint, TrackerInfo
from avt.tracking.base import TrackingBundle
from avt.viewer import build_viewer


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


def write_frames(root: Path, count: int = 6) -> None:
    root.mkdir()
    for idx in range(count):
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        cv2.circle(img, (12 + idx, 24), 4, (255, 255, 255), -1)
        cv2.imwrite(str(root / f"{idx:04d}.png"), img)


def test_build_queries() -> None:
    config = InverseTrackConfig(query_stride=2, seed_count=3)
    queries = build_queries(width=100, height=80, frame_count=5, config=config)
    assert [q.reverse_time for q in queries] == [0, 0, 0, 2, 2, 2, 4, 4, 4]
    assert [q.id for q in queries] == list(range(9))


def test_create_unique_run_dir(tmp_path: Path) -> None:
    base = tmp_path / "outputs"
    first = _create_unique_run_dir(base, preferred_name="run_test")
    second = _create_unique_run_dir(base, preferred_name="run_test")

    assert first == base / "run_test"
    assert second == base / "run_test_01"
    assert first.exists()
    assert second.exists()


def test_cli_output_defaults() -> None:
    parser = build_parser()

    track_args = parser.parse_args(["track", "--frames-root", "/tmp/frames"])
    all_args = parser.parse_args(["all", "--frames-root", "/tmp/frames"])
    viewer_args = parser.parse_args(
        ["viewer", "--frames-root", "/tmp/frames", "--tracking-root", "/tmp/tracks"]
    )

    assert track_args.output_root == DEFAULT_OUTPUT_ROOT
    assert all_args.output_root == DEFAULT_OUTPUT_ROOT
    assert viewer_args.viewer_dir == DEFAULT_VIEWER_ROOT


def test_inverse_tracking_and_viewer(tmp_path: Path) -> None:
    frames_root = tmp_path / "frames"
    write_frames(frames_root)
    records = read_frame_records(frames_root, "image_dir")
    output_root = tmp_path / "out"
    config = InverseTrackConfig(
        window_size=4,
        window_step=3,
        query_stride=2,
        seed_count=3,
        max_windows=1,
        save_reverse_video=False,
    )
    windows = run_inverse_tracking(frames_root, records, output_root, FakeTracker(), config)
    assert len(windows) == 1
    assert (output_root / "windows" / "seq_0_4" / "tracks.npz").exists()
    assert (output_root / "windows" / "seq_0_4" / "window.json").exists()

    viewer_dir = tmp_path / "viewer"
    payload = build_viewer(frames_root, records, output_root, viewer_dir)
    assert payload["metadata"]["successful_windows"] == 1
    assert (viewer_dir / "index.html").exists()
    assert (viewer_dir / "data" / "prediction_tracks.json").exists()
