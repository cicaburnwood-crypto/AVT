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
from avt.querying import (
    QueryConfig,
    SiftCaptureConfig,
    VirtualRobotConfig,
    align_virtual_robot_to_image,
    query_artifact_arrays,
    query_config_from_mapping,
)
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


def test_build_sift_first_avt_fallback_queries() -> None:
    frames = np.zeros((6, 96, 128, 3), dtype=np.uint8)
    for t in range(len(frames)):
        cv2.rectangle(frames[t], (42, 78), (46, 82), (255, 255, 255), -1)

    config = InverseTrackConfig(
        window_size=6,
        query_stride=1,
        seed_count=17,
        query_config=QueryConfig(
            mode="avt+sift",
            robot=VirtualRobotConfig(
                width_m=0.40,
                length_m=0.60,
                camera_height_m=0.18,
                footprint_width_ratio=0.45,
                footprint_length_ratio=0.30,
            ),
            sift=SiftCaptureConfig(
                enabled=True,
                max_query_points=80,
                temporal_stride=2,
                edge_offset_ratio=0.35,
            ),
        ),
    )

    queries = build_queries(128, 96, len(frames), config, frames_rgb=frames)
    sources = {query.source for query in queries}
    arrays = query_artifact_arrays(queries)

    assert len(queries) == 80
    assert {"avt", "sift_robot"}.issubset(sources)
    assert [query.id for query in queries] == list(range(len(queries)))
    first_avt = next(idx for idx, query in enumerate(queries) if query.source == "avt")
    assert all(query.source == "sift_robot" for query in queries[:first_avt])
    assert all(query.source == "avt" for query in queries[first_avt:])
    assert arrays["queries"].shape[1] == 11
    assert arrays["queries_cotracker"].shape == (len(queries), 3)


def test_robot_aligned_avt_fallback_ratios() -> None:
    frames = np.zeros((4, 96, 128, 3), dtype=np.uint8)
    config = InverseTrackConfig(
        query_stride=2,
        seed_count=3,
        query_config=QueryConfig(
            mode="avt+sift",
            robot=VirtualRobotConfig(
                footprint_width_ratio=0.50,
                footprint_length_ratio=0.20,
            ),
            sift=SiftCaptureConfig(enabled=True, max_query_points=5, temporal_stride=2),
        ),
    )

    queries = build_queries(128, 96, len(frames), config, frames_rgb=frames)

    assert len(queries) == 5
    assert {query.source for query in queries} == {"avt"}
    assert queries[0].x == np.float32(0.25 * (128 - 1))
    assert queries[1].x == np.float32(0.50 * (128 - 1))
    assert queries[2].x == np.float32(0.75 * (128 - 1))
    assert queries[0].y == 0.90 * (96 - 1)


def test_virtual_robot_alignment_auto_detects_resolution() -> None:
    robot = VirtualRobotConfig(width_m=0.40, length_m=0.60, camera_height_m=None)

    low_res = align_virtual_robot_to_image(height=720, width=1280, robot=robot)
    wide_res = align_virtual_robot_to_image(height=1080, width=1920, robot=robot)

    assert low_res.method == "image_normalized_no_intrinsics"
    assert low_res.frame_width == 1280
    assert low_res.frame_height == 720
    assert 0 <= low_res.left < low_res.right <= 1280
    assert 0 <= low_res.top < low_res.bottom <= 720
    assert wide_res.frame_width == 1920
    assert wide_res.frame_height == 1080
    assert low_res.seed_x_min_ratio < 0.5 < low_res.seed_x_max_ratio
    assert low_res.seed_y_ratio > 0.5


def test_query_config_from_yaml_mapping() -> None:
    config = query_config_from_mapping(
        {
            "query_mode": "avt+sift",
            "virtual_robot": {
                "width_cm": 40,
                "length_cm": 60,
                "camera_height_cm": 18,
            },
            "sift": {
                "enabled": True,
                "max_query_points": 384,
                "temporal_stride": 3,
            },
        }
    )

    assert config.mode == "avt+sift"
    assert config.robot.width_m == 0.40
    assert config.robot.length_m == 0.60
    assert config.robot.camera_height_m == 0.18
    assert config.sift.max_query_points == 384


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
