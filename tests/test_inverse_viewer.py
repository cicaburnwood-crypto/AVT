from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from avt.anchor_motion import estimate_anchor_motion_projection, estimate_rolling_mother_projection
from avt.cli import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_VIEWER_ROOT,
    _create_unique_run_dir,
    build_parser,
)
from avt.config import AnchorMotionConfig
from avt.inverse import InverseTrackConfig, build_queries, reference_mask, run_inverse_tracking
from avt.io import read_frame_records
from avt.querying import (
    AnchorSamplingConfig,
    FootprintConfig,
    QueryConfig,
    QuerySamplingConfig,
    align_footprint_to_image,
    query_artifact_arrays,
    query_config_from_mapping,
    robot_footprint_mask,
)
from avt.reliability import (
    STATIONARY_BLOCK_SIZE_PX,
    STATIONARY_SPAN_FRAMES,
    STOP_EXTREME_SLOW_REASON,
    detect_stationary_query_frames,
    frame_reliability,
    segment_bounds,
    unreliable_segments,
)
from avt.schema import QueryPoint, TrackerInfo, WindowSpec
from avt.tracking.base import TrackingBundle
from avt.tracking.bootstap.backend import _query_points_for_resize, _tapnet_outputs_to_avt
from avt.tracking.bootstap.config import bootstap_config_from_mapping
from avt.tracking.foundationpose import FoundationPoseBackend
from avt.tracking.foundationpose.download import FOUNDATIONPOSE_WEIGHT_FILES
from avt.viewer import build_viewer


class FakeTracker:
    def track(self, frames_rgb: np.ndarray, queries: list[QueryPoint]) -> TrackingBundle:
        tracks = np.full((len(frames_rgb), len(queries), 2), np.nan, dtype=np.float32)
        visibility = np.zeros((len(frames_rgb), len(queries)), dtype=bool)
        confidence = np.zeros((len(frames_rgb), len(queries)), dtype=np.float32)
        for query in queries:
            tracks[query.reverse_time :, query.id] = [query.x, query.y]
            visibility[query.reverse_time :, query.id] = True
            confidence[query.reverse_time :, query.id] = 0.25 + query.id * 0.01
        return TrackingBundle(
            tracks=tracks,
            visibility=visibility,
            confidence=confidence,
            tracker=TrackerInfo(name="fake"),
        )


def test_anchor_motion_projects_mother_point_from_affine_tracks() -> None:
    base = np.array(
        [
            [10.0, 10.0],
            [30.0, 10.0],
            [10.0, 30.0],
            [30.0, 30.0],
            [20.0, 18.0],
            [35.0, 25.0],
        ],
        dtype=np.float32,
    )
    frame_count = 4
    tracks = np.zeros((frame_count, len(base), 2), dtype=np.float32)
    for t in range(frame_count):
        tracks[t] = base + np.array([2.0 * t, 3.0 * t], dtype=np.float32)
    bundle = TrackingBundle(
        tracks=tracks,
        visibility=np.ones((frame_count, len(base)), dtype=bool),
        tracker=TrackerInfo(name="fake"),
    )
    queries = [
        QueryPoint(id=i, reverse_time=0, x=float(x), y=float(y), side=-1, source="anchor")
        for i, (x, y) in enumerate(base)
    ]

    projection = estimate_anchor_motion_projection(
        bundle,
        queries,
        width=100,
        height=80,
        config=AnchorMotionConfig(min_matches=4),
    )

    mother = np.array([49.5, 79.0], dtype=np.float32)
    expected = np.stack(
        [mother + np.array([2.0 * t, 3.0 * t], dtype=np.float32) for t in range(frame_count)]
    )
    assert projection.valid_reverse.tolist() == [True, True, True, True]
    assert np.allclose(projection.projected_points_reverse, expected, atol=1e-3)


def test_rolling_mother_projection_casts_each_source_into_current_frame() -> None:
    base = np.array(
        [
            [10.0, 10.0],
            [30.0, 10.0],
            [10.0, 30.0],
            [30.0, 30.0],
            [20.0, 18.0],
            [35.0, 25.0],
        ],
        dtype=np.float32,
    )
    frame_count = 4
    tracks = np.zeros((frame_count, len(base), 2), dtype=np.float32)
    for t in range(frame_count):
        tracks[t] = base + np.array([2.0 * t, -3.0 * t], dtype=np.float32)
    bundle = TrackingBundle(
        tracks=tracks,
        visibility=np.ones((frame_count, len(base)), dtype=bool),
        tracker=TrackerInfo(name="fake"),
    )
    queries = [
        QueryPoint(id=i, reverse_time=0, x=float(x), y=float(y), side=-1, source="anchor")
        for i, (x, y) in enumerate(base)
    ]

    projection = estimate_anchor_motion_projection(
        bundle,
        queries,
        width=100,
        height=100,
        config=AnchorMotionConfig(min_matches=4),
    )
    rolling = estimate_rolling_mother_projection(
        projection,
        width=100,
        height=100,
        scale_radius_px=10.0,
    )

    mother = np.array([49.5, 99.0], dtype=np.float32)
    assert rolling.valid_reverse[3, 0]
    assert rolling.valid_reverse[3, 3]
    assert not rolling.valid_reverse[1, 2]
    assert np.allclose(rolling.points_reverse[3, 0], mother + [6.0, -9.0], atol=1e-3)
    assert np.allclose(rolling.points_reverse[3, 3], mother, atol=1e-3)
    assert np.isclose(rolling.scale_reverse[3, 0], 1.0, atol=1e-3)


def write_frames(root: Path, count: int = 6) -> None:
    root.mkdir()
    rng = np.random.default_rng(17)
    for idx in range(count):
        img = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
        cv2.circle(img, (12 + idx, 24), 4, (255, 255, 255), -1)
        cv2.imwrite(str(root / f"{idx:04d}.png"), img)


def test_build_queries() -> None:
    rng = np.random.default_rng(3)
    frames = rng.integers(0, 255, size=(5, 80, 100, 3), dtype=np.uint8)
    config = InverseTrackConfig(
        query_config=QueryConfig(
            mode="anchor_motion",
            sampling=QuerySamplingConfig(
                enabled=True,
                anchors=AnchorSamplingConfig(
                    enabled=True,
                    max_query_points=12,
                    min_points_per_frame=4,
                    max_points_per_frame=12,
                    contrast_threshold=0.001,
                ),
            ),
        ),
    )
    queries = build_queries(width=100, height=80, frame_count=5, config=config, frames_rgb=frames)
    assert queries
    assert {q.source for q in queries} == {"anchor"}
    assert {q.reverse_time for q in queries} == {0}
    assert [q.id for q in queries] == list(range(len(queries)))


def test_build_anchor_footprint_queries_are_anchor_only_alias() -> None:
    rng = np.random.default_rng(7)
    frames = rng.integers(0, 255, size=(8, 96, 128, 3), dtype=np.uint8)

    config = InverseTrackConfig(
        query_config=QueryConfig(
            mode="anchor_footprint",
            footprint=FootprintConfig(width_ratio=0.60, height_ratio=0.35),
            sampling=QuerySamplingConfig(
                enabled=True,
                max_query_points=16,
                window_size=4,
                edge_offset_ratio=0.25,
                contrast_threshold=0.001,
                anchors=AnchorSamplingConfig(
                    enabled=True,
                    max_query_points=16,
                    min_points_per_frame=4,
                    max_points_per_frame=16,
                    window_size=4,
                    contrast_threshold=0.001,
                ),
            ),
        ),
    )

    queries = build_queries(128, 96, len(frames), config, frames_rgb=frames)
    arrays = query_artifact_arrays(queries)

    assert {query.source for query in queries} == {"anchor"}
    assert "avt" not in {query.source for query in queries}
    assert "footprint" not in {query.source for query in queries}
    assert {query.reverse_time for query in queries} == {0}
    assert [query.id for query in queries] == list(range(len(queries)))
    assert arrays["queries"].shape[1] == 11
    assert arrays["queries_cotracker"].shape == (len(queries), 3)
    assert set(arrays["query_source_codes"].tolist()) == {2}


def test_bottom_center_footprint_mask_edges() -> None:
    mask = robot_footprint_mask(
        100,
        200,
        FootprintConfig(width_ratio=0.20, height_ratio=0.20),
        QuerySamplingConfig(edge_offset_ratio=0.25),
    )

    assert mask[90, 85] == 255
    assert mask[90, 100] == 0
    assert mask[90, 115] == 255
    assert mask[79, 85] == 0


def test_footprint_alignment_auto_detects_resolution() -> None:
    robot = FootprintConfig(width_ratio=0.20, height_ratio=0.15)

    low_res = align_footprint_to_image(height=720, width=1280, robot=robot)
    wide_res = align_footprint_to_image(height=1080, width=1920, robot=robot)

    assert low_res.method == "bottom_center_footprint"
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
            "query_mode": "anchor_footprint",
            "footprint": {
                "width_ratio": 0.25,
                "height_ratio": 0.20,
            },
            "sampling": {
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

    assert config.mode == "anchor_footprint"
    assert config.footprint.width_ratio == 0.25
    assert config.footprint.height_ratio == 0.20
    assert config.sampling.max_query_points == 384
    assert config.sampling.window_size == 20
    assert config.sampling.min_points_per_frame == 8
    assert config.sampling.max_points_per_frame == 20
    assert config.sampling.anchors.max_query_points == 192
    assert config.sampling.anchors.min_points_per_frame == 32
    assert config.sampling.anchors.max_points_per_frame == 128


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


def test_cli_accepts_bootstap_backend() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "track",
            "--frames-root",
            "/tmp/frames",
            "--backend",
            "bootstap",
            "--bootstap-config",
            "configs/bootstap.yaml",
            "--bootstap-checkpoint",
            "/tmp/bootstapir_checkpoint_v2.pt",
            "--bootstap-resize-height",
            "256",
            "--bootstap-resize-width",
            "320",
        ]
    )

    assert args.backend == "bootstap"
    assert args.bootstap_config == Path("configs/bootstap.yaml")
    assert args.bootstap_checkpoint == Path("/tmp/bootstapir_checkpoint_v2.pt")
    assert args.bootstap_resize_height == 256
    assert args.bootstap_resize_width == 320


def test_bootstap_config_from_mapping() -> None:
    config = bootstap_config_from_mapping(
        {
            "device": "cpu",
            "checkpoint_path": "/tmp/bootstapir_checkpoint_v2.pt",
            "resize_height": 256,
            "resize_width": 320,
            "query_chunk_size": 32,
        }
    )

    assert config.device == "cpu"
    assert config.checkpoint_path == Path("/tmp/bootstapir_checkpoint_v2.pt")
    assert config.resize_height == 256
    assert config.resize_width == 320
    assert config.query_chunk_size == 32


def test_bootstap_coordinate_adapter() -> None:
    queries = [
        QueryPoint(id=0, reverse_time=0, x=20, y=10, side=-1),
        QueryPoint(id=1, reverse_time=1, x=40, y=30, side=1),
    ]

    query_points = _query_points_for_resize(
        queries,
        source_height=100,
        source_width=200,
        resize_height=50,
        resize_width=100,
    )
    assert query_points.tolist() == [[0.0, 5.0, 10.0], [1.0, 15.0, 20.0]]

    tracks = np.array(
        [
            [[10, 5], [11, 6], [12, 7]],
            [[20, 15], [21, 16], [22, 17]],
        ],
        dtype=np.float32,
    )
    visibility = np.ones((2, 3), dtype=bool)

    tracks_avt, visibility_avt = _tapnet_outputs_to_avt(
        tracks,
        visibility,
        queries,
        source_height=100,
        source_width=200,
        resize_height=50,
        resize_width=100,
    )

    assert tracks_avt.shape == (3, 2, 2)
    assert visibility_avt[:, 0].tolist() == [True, True, True]
    assert visibility_avt[:, 1].tolist() == [False, True, True]
    assert tracks_avt[0, 0].tolist() == [20, 10]
    assert np.isnan(tracks_avt[0, 1]).all()
    assert tracks_avt[1, 1].tolist() == [42, 32]


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
        query_config=QueryConfig(
            mode="anchor_motion",
            sampling=QuerySamplingConfig(
                enabled=True,
                anchors=AnchorSamplingConfig(
                    enabled=True,
                    max_query_points=8,
                    min_points_per_frame=4,
                    max_points_per_frame=8,
                    contrast_threshold=0.001,
                ),
            ),
        ),
    )
    windows = run_inverse_tracking(frames_root, records, output_root, FakeTracker(), config)
    assert len(windows) == 1
    assert (output_root / "windows" / "seq_0_4" / "tracks.npz").exists()
    assert (output_root / "windows" / "seq_0_4" / "window.json").exists()
    assert not (output_root / "windows" / "seq_0_4" / "reverse_video.mp4").exists()
    assert not (output_root / "windows" / "seq_0_4" / "path_mask_reference.png").exists()
    arrays = np.load(output_root / "windows" / "seq_0_4" / "tracks.npz")
    assert arrays["confidence_reverse"].shape == arrays["visibility_reverse"].shape
    assert arrays["confidence_reverse"].dtype == np.float32
    assert set(arrays["query_source_codes"].tolist()) == {2}
    assert "anchor_motion_projected_mother_reverse" in arrays
    assert "anchor_motion_rolling_mother_reverse" in arrays
    assert "anchor_motion_rolling_mother_scale_reverse" in arrays
    assert "anchor_motion_rolling_mother_valid_reverse" in arrays

    viewer_dir = tmp_path / "viewer"
    payload = build_viewer(frames_root, records, output_root, viewer_dir)
    assert payload["metadata"]["successful_windows"] == 1
    assert payload["metadata"]["reliability_filter"]["schema"] == "avt_frame_segment_reliability_v1"
    assert payload["metadata"]["reliability_filter"]["segment_size_frames"] == 40
    assert payload["metadata"]["reliability_filter"]["rules"][0]["span_frames"] == 10
    assert payload["metadata"]["reliability_filter"]["rules"][0]["block_size_px"] == 6.0
    assert (viewer_dir / "index.html").exists()
    assert (viewer_dir / "data" / "prediction_tracks.json").exists()
    segment = json.loads((viewer_dir / "data" / "windows" / "seq_0_4.json").read_text())
    assert segment["point_columns"] == ["id", "x", "y", "confidence"]
    assert len(segment["frames"][0]["points"][0]) == 4
    assert 0.0 <= segment["frames"][0]["points"][0][3] <= 1.0
    assert "mother_points" in segment["frames"][0]
    assert len(segment["frames"][0]["mother_points"][0]) == 4
    assert segment["frames"][0]["reliability"]["segment_id"] == 0
    assert segment["frames"][0]["reliability"]["segment_unreliable"] is False
    assert segment["frames"][0]["reliability"]["segment_disabled"] is False
    assert segment["frames"][0]["reliability"]["unreliable_point_ids"] == []


def test_reliability_marks_40_frame_segment() -> None:
    assert segment_bounds(41) == (40, 80)
    assert unreliable_segments([41]) == {1}

    before = frame_reliability(39, [], unreliable_frame_indices=[41])
    triggered = frame_reliability(41, [], unreliable_frame_indices=[41], frame_reasons={41: ["test"]})
    same_segment = frame_reliability(79, [], unreliable_frame_indices=[41])
    next_segment = frame_reliability(80, [], unreliable_frame_indices=[41])

    assert before["segment_unreliable"] is False
    assert triggered["frame_unreliable"] is True
    assert triggered["segment_unreliable"] is True
    assert triggered["segment_disabled"] is True
    assert triggered["segment_status"] == "disabled"
    assert triggered["trigger_frame_indices"] == [41]
    assert triggered["reason_counts"] == {"test": 1}
    assert same_segment["segment_unreliable"] is True
    assert same_segment["frame_unreliable"] is False
    assert next_segment["segment_unreliable"] is False


def test_stationary_sampled_points_disable_segment() -> None:
    tracks = np.zeros((12, 4, 2), dtype=np.float32)
    visibility = np.ones((12, 4), dtype=bool)
    for reverse_t in range(12):
        tracks[reverse_t, :, 0] = 100 + reverse_t * 30 + np.arange(4)
        tracks[reverse_t, :, 1] = 200 + reverse_t * 30 + np.arange(4)

    for reverse_t in range(2, 12):
        offset = (reverse_t - 2) * 0.5
        tracks[reverse_t, :3] = [
            [10 + offset, 10],
            [20, 20 + offset],
            [15 + offset, 15 + offset],
        ]

    reasons = detect_stationary_query_frames(
        tracks,
        visibility,
        seq_start=0,
        seq_end=12,
        query_point_ids=[0, 1, 2],
    )
    assert STATIONARY_SPAN_FRAMES == 10
    assert STATIONARY_BLOCK_SIZE_PX == 6.0
    assert reasons == {9: [STOP_EXTREME_SLOW_REASON]}

    marked = frame_reliability(0, [], unreliable_frame_indices=reasons.keys(), frame_reasons=reasons)
    assert marked["segment_disabled"] is True
    assert marked["reason_counts"] == {STOP_EXTREME_SLOW_REASON: 1}


def test_reference_mask_uses_support_points() -> None:
    bundle = TrackingBundle(
        tracks=np.empty((2, 0, 2), dtype=np.float32),
        visibility=np.empty((2, 0), dtype=bool),
        tracker=TrackerInfo(name="fake"),
    )
    support = np.array([[10, 30], [30, 30], [20, 15]], dtype=np.float32)

    mask = reference_mask(bundle, height=40, width=50, queries=[], support_points=support)

    assert mask[..., 3].sum() > 0
