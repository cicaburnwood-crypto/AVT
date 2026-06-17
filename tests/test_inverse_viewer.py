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
from avt.inverse import InverseTrackConfig, build_queries, reference_mask, run_inverse_tracking
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
from avt.schema import QueryPoint, TrackerInfo, WindowSpec
from avt.tracking.base import TrackingBundle
from avt.tracking.foundationpose import FoundationPoseBackend
from avt.tracking.foundationpose.download import FOUNDATIONPOSE_WEIGHT_FILES
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
    assert config.sift.min_points_per_frame == 8
    assert config.sift.max_points_per_frame == 20
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
    assert track_args.save_reverse_video is False
    assert track_args.save_path_mask is False
    assert all_args.build_viewer is False

    debug_args = parser.parse_args(
        [
            "all",
            "--frames-root",
            "/tmp/frames",
            "--save-reverse-video",
            "--save-path-mask",
            "--build-viewer",
        ]
    )
    assert debug_args.save_reverse_video is True
    assert debug_args.save_path_mask is True
    assert debug_args.build_viewer is True


def test_cli_accepts_foundationpose_backend() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "track",
            "--frames-root",
            "/tmp/frames",
            "--backend",
            "foundationpose",
            "--foundationpose-transforms",
            "/tmp/fp_transforms.npz",
        ]
    )

    assert args.backend == "foundationpose"
    assert args.foundationpose_transforms == Path("/tmp/fp_transforms.npz")


def test_foundationpose_homography_adapter(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    for rel in FOUNDATIONPOSE_WEIGHT_FILES:
        path = weights / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")

    homographies = np.repeat(np.eye(3, dtype=np.float32)[None], 4, axis=0)
    homographies[:, 0, 2] = [0, 2, 4, 6]
    transforms = tmp_path / "transforms.npz"
    np.savez_compressed(transforms, homographies_reverse=homographies)

    frames = np.zeros((4, 20, 30, 3), dtype=np.uint8)
    queries = [
        QueryPoint(id=0, reverse_time=0, x=5, y=6, side=-1),
        QueryPoint(id=1, reverse_time=2, x=10, y=8, side=1),
    ]
    tracker = FoundationPoseBackend(weights_dir=weights, transforms_path=transforms)

    bundle = tracker.track(frames, queries)

    assert bundle.tracker.name == "foundationpose"
    assert bundle.visibility[:, 0].tolist() == [True, True, True, True]
    assert bundle.visibility[:, 1].tolist() == [False, False, True, True]
    assert bundle.tracks[:, 0, 0].tolist() == [5, 7, 9, 11]
    assert np.isnan(bundle.tracks[:2, 1]).all()
    assert bundle.tracks[2:, 1, 0].tolist() == [10, 12]


def test_foundationpose_transform_directory_uses_window_context(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    for rel in FOUNDATIONPOSE_WEIGHT_FILES:
        path = weights / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")

    transforms_dir = tmp_path / "transforms"
    transforms_dir.mkdir()
    homographies = np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0)
    homographies[:, 1, 2] = [0, 1, 2]
    np.savez_compressed(transforms_dir / "seq_0_3.npz", homographies_reverse=homographies)

    tracker = FoundationPoseBackend(weights_dir=weights, transforms_path=transforms_dir)
    tracker.set_window_context(window=WindowSpec(start=0, end=3), output_dir=tmp_path)
    frames = np.zeros((3, 20, 30, 3), dtype=np.uint8)
    queries = [QueryPoint(id=0, reverse_time=0, x=5, y=6, side=-1)]

    bundle = tracker.track(frames, queries)

    assert bundle.tracks[:, 0, 1].tolist() == [6, 7, 8]


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
    assert not (output_root / "windows" / "seq_0_4" / "reverse_video.mp4").exists()
    assert not (output_root / "windows" / "seq_0_4" / "path_mask_reference.png").exists()

    viewer_dir = tmp_path / "viewer"
    payload = build_viewer(frames_root, records, output_root, viewer_dir)
    assert payload["metadata"]["successful_windows"] == 1
    assert (viewer_dir / "index.html").exists()
    assert (viewer_dir / "data" / "prediction_tracks.json").exists()


def test_reference_mask_uses_support_points() -> None:
    bundle = TrackingBundle(
        tracks=np.empty((2, 0, 2), dtype=np.float32),
        visibility=np.empty((2, 0), dtype=bool),
        tracker=TrackerInfo(name="fake"),
    )
    support = np.array([[10, 30], [30, 30], [20, 15]], dtype=np.float32)

    mask = reference_mask(bundle, height=40, width=50, queries=[], support_points=support)

    assert mask[..., 3].sum() > 0
