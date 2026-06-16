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
    SiftAnchorConfig,
    SiftCaptureConfig,
    VirtualRobotConfig,
    align_virtual_robot_to_image,
    query_artifact_arrays,
    query_config_from_mapping,
    robot_sift_mask,
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
    config = InverseTrackConfig(
        query_stride=2,
        seed_count=3,
        query_config=QueryConfig(mode="avt", sift=SiftCaptureConfig(enabled=False)),
    )
    queries = build_queries(width=100, height=80, frame_count=5, config=config)
    assert [q.reverse_time for q in queries] == [0, 0, 0, 2, 2, 2, 4, 4, 4]
    assert [q.id for q in queries] == list(range(9))


def test_build_ventura_anchor_and_crumb_queries() -> None:
    rng = np.random.default_rng(7)
    frames = rng.integers(0, 255, size=(8, 96, 128, 3), dtype=np.uint8)

    config = InverseTrackConfig(
        query_config=QueryConfig(
            mode="ventura",
            robot=VirtualRobotConfig(width_ratio=0.60, height_ratio=0.35),
            sift=SiftCaptureConfig(
                enabled=True,
                max_query_points=16,
                window_size=4,
                edge_offset_ratio=0.25,
                contrast_threshold=0.001,
                anchors=SiftAnchorConfig(
                    enabled=True,
                    max_query_points=16,
                    window_size=4,
                    contrast_threshold=0.001,
                ),
            ),
        ),
    )

    queries = build_queries(128, 96, len(frames), config, frames_rgb=frames)
    arrays = query_artifact_arrays(queries)

    assert {query.source for query in queries} == {"sift_anchor", "sift_robot"}
    assert "avt" not in {query.source for query in queries}
    assert [query.id for query in queries] == list(range(len(queries)))
    first_robot = next(idx for idx, query in enumerate(queries) if query.source == "sift_robot")
    assert all(query.source == "sift_anchor" for query in queries[:first_robot])
    assert all(query.source == "sift_robot" for query in queries[first_robot:])
    assert arrays["queries"].shape[1] == 11
    assert arrays["queries_cotracker"].shape == (len(queries), 3)
    assert set(arrays["query_source_codes"].tolist()) == {1, 2}


def test_ventura_pct_mask_edges() -> None:
    mask = robot_sift_mask(
        100,
        200,
        VirtualRobotConfig(width_ratio=0.20, height_ratio=0.20),
        SiftCaptureConfig(edge_offset_ratio=0.25),
    )

    assert mask[90, 85] == 255
    assert mask[90, 100] == 0
    assert mask[90, 115] == 255
    assert mask[79, 85] == 0


def test_virtual_robot_alignment_auto_detects_resolution() -> None:
    robot = VirtualRobotConfig(width_ratio=0.20, height_ratio=0.15)

    low_res = align_virtual_robot_to_image(height=720, width=1280, robot=robot)
    wide_res = align_virtual_robot_to_image(height=1080, width=1920, robot=robot)

    assert low_res.method == "ventura_pct_bottom_center"
    assert low_res.frame_width == 1280
    assert low_res.frame_height == 720
    assert 0 <= low_res.left < low_res.right <= 1280
    assert 0 <= low_res.top < low_res.bottom <= 720
    assert wide_res.frame_width == 1920
    assert wide_res.frame_height == 1080
    assert low_res.seed_x_min_ratio < 0.5 < low_res.seed_x_max_ratio
    assert low_res.width_ratio == 0.20
    assert low_res.length_ratio == 0.15
    assert low_res.seed_y_ratio == 0.925


def test_query_config_from_yaml_mapping() -> None:
    config = query_config_from_mapping(
        {
            "query_mode": "ventura",
            "footprint": {
                "width_ratio": 0.25,
                "height_ratio": 0.20,
            },
            "sift": {
                "enabled": True,
                "max_query_points": 384,
                "window_size": 20,
                "anchors": {
                    "enabled": True,
                    "max_query_points": 192,
                },
            },
        }
    )

    assert config.mode == "ventura"
    assert config.robot.width_ratio == 0.25
    assert config.robot.height_ratio == 0.20
    assert config.sift.max_query_points == 384
    assert config.sift.window_size == 20
    assert config.sift.anchors.max_query_points == 192


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
        query_config=QueryConfig(mode="avt", sift=SiftCaptureConfig(enabled=False)),
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
