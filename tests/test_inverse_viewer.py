from __future__ import annotations

import json
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
from avt.reliability import (
    STATIONARY_BLOCK_SIZE_PX,
    STATIONARY_SPAN_FRAMES,
    STOP_EXTREME_SLOW_REASON,
    detect_stationary_sift_frames,
    frame_reliability,
    segment_bounds,
    unreliable_segments,
)
from avt.schema import QueryPoint, TrackerInfo, WindowSpec
from avt.tracking.base import TrackingBundle
from avt.tracking.bootstap.backend import _query_points_for_resize, _tapnet_outputs_to_avt
from avt.tracking.bootstap.config import bootstap_config_from_mapping
from avt.tracking.cotracker_cache import (
    CHUNKED_CACHE_SCHEMA,
    CachedCoTrackerBackend,
    CACHE_SCHEMA,
    CoTrackerCache,
    CoTrackerCacheConfig,
    SEGMENTED_CACHE_SCHEMA,
    cotracker_cache_config_from_mapping,
    _cache_queries,
    _first_abort_times,
)
from avt.tracking.foundationpose import FoundationPoseBackend
from avt.tracking.foundationpose.download import FOUNDATIONPOSE_WEIGHT_FILES
from avt.viewer import build_viewer, write_viewer


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


def write_frames(root: Path, count: int = 6) -> None:
    root.mkdir(parents=True)
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
    assert track_args.window_size == 80
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


def test_cli_accepts_cotracker_cache_backend() -> None:
    parser = build_parser()

    cache_args = parser.parse_args(
        [
            "cache",
            "--frames-root",
            "/tmp/frames",
            "--cache-frame-count",
            "100",
            "--cache-query-mode",
            "confidence-refresh",
            "--cache-config",
            "configs/cotracker_cache.yaml",
        ]
    )
    chunk_args = parser.parse_args(
        [
            "cache-chunks",
            "--frames-root",
            "/tmp/frames",
            "--cache-config",
            "configs/cotracker_cache.yaml",
        ]
    )
    track_args = parser.parse_args(
        [
            "track",
            "--frames-root",
            "/tmp/frames",
            "--backend",
            "cotracker_cache",
            "--cotracker-cache",
            "/tmp/cache",
            "--cache-record-ids-only",
        ]
    )

    assert cache_args.cache_frame_count == 100
    assert cache_args.cache_config == Path("configs/cotracker_cache.yaml")
    assert cache_args.cache_grid_stride == 1
    assert cache_args.cache_query_mode == "confidence-refresh"
    assert chunk_args.cache_chunk_size == 480
    assert chunk_args.cache_window_size == 80
    assert chunk_args.cache_chunk_step is None
    assert chunk_args.cache_grid_stride == 1
    assert chunk_args.cache_region == "bottom-third"
    assert chunk_args.cache_query_mode == "confidence-refresh"
    assert chunk_args.cache_config == Path("configs/cotracker_cache.yaml")
    assert track_args.backend == "cotracker_cache"
    assert track_args.cotracker_cache == Path("/tmp/cache")
    assert track_args.cache_record_ids_only is True


def test_cotracker_cache_queries_seed_initial_dense_frame_by_default() -> None:
    config = CoTrackerCacheConfig(region="bottom-half")

    queries = _cache_queries(frame_count=3, height=4, width=3, config=config)

    assert config.query_mode == "confidence-refresh"
    assert [query.reverse_time for query in queries] == [0, 0, 0, 0, 0, 0]
    assert [query.id for query in queries] == list(range(6))


def test_cotracker_cache_horizon_refreshes_reliable_tracks() -> None:
    query = QueryPoint(id=0, reverse_time=0, x=1, y=2, side=0)
    bundle = TrackingBundle(
        tracks=np.zeros((10, 1, 2), dtype=np.float32),
        visibility=np.ones((10, 1), dtype=bool),
        confidence=np.ones((10, 1), dtype=np.float32),
        tracker=TrackerInfo(name="synthetic"),
    )

    assert _first_abort_times(bundle, [query], threshold=0.85).tolist() == [-1]
    assert _first_abort_times(bundle, [query], threshold=0.85, max_track_frames=4).tolist() == [4]


def test_segmented_cotracker_cache_indexes_generations(tmp_path: Path) -> None:
    generations = []
    for generation_index, point_ids in enumerate(([10, 11], [20])):
        gen_dir = tmp_path / f"generation_{generation_index:04d}"
        gen_dir.mkdir()
        point_ids_array = np.array(point_ids, dtype=np.int64)
        tracks = np.zeros((4, len(point_ids), 2), dtype=np.float32)
        for local_index, point_id in enumerate(point_ids):
            tracks[:, local_index, 0] = point_id
            tracks[:, local_index, 1] = np.arange(4, dtype=np.float32)
        visibility = np.ones((4, len(point_ids)), dtype=bool)
        confidence = np.ones((4, len(point_ids)), dtype=np.float32) * (0.5 + generation_index)
        arrays = {}
        for name, array in {
            "tracks": tracks,
            "visibility": visibility,
            "point_ids": point_ids_array,
            "birth_reverse_times": np.zeros(len(point_ids), dtype=np.int32),
            "source_birth_frames": np.full(len(point_ids), 3, dtype=np.int32),
            "seed_xy": np.zeros((len(point_ids), 2), dtype=np.float32),
            "parent_point_ids": np.full(len(point_ids), -1, dtype=np.int64),
            "generation_indices": np.full(len(point_ids), generation_index, dtype=np.int16),
            "abort_reverse_times": np.full(len(point_ids), -1, dtype=np.int32),
            "confidence": confidence,
        }.items():
            filename = f"{name}.npy"
            np.save(gen_dir / filename, array)
            arrays[name] = f"{gen_dir.name}/{filename}"
        generations.append(
            {
                "generation_index": generation_index,
                "path": gen_dir.name,
                "point_count": len(point_ids),
                "arrays": arrays,
            }
        )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "schema": SEGMENTED_CACHE_SCHEMA,
                "complete": False,
                "frame_start": 0,
                "frame_end": 4,
                "frame_count": 4,
                "source_root": str(tmp_path),
                "point_count": 3,
                "generations": generations,
            }
        ),
        encoding="utf-8",
    )

    cache = CoTrackerCache(tmp_path)

    assert cache.point_ids.tolist() == [10, 11, 20]
    assert cache.tracks[2].shape == (3, 2)
    assert cache.tracks[2][:, 0].tolist() == [10.0, 11.0, 20.0]
    selected = cache.tracks[np.ix_(np.array([1, 3]), np.array([0, 2]))]
    assert selected.shape == (2, 2, 2)
    assert selected[:, :, 0].tolist() == [[10.0, 20.0], [10.0, 20.0]]
    assert cache.confidence[np.ix_(np.array([0]), np.array([0, 2]))].tolist() == [[0.5, 1.5]]


def test_cotracker_cache_config_from_yaml_mapping() -> None:
    config = cotracker_cache_config_from_mapping(
        {
            "chunk": {
                "size": 120,
                "window_size": 80,
                "step": 40,
            },
            "cache": {
                "region": "bottom-half",
                "bad_track_confidence_threshold": 0.72,
                "max_track_frames": 100,
            },
            "tracker": {
                "device": "cpu",
                "batch_size": 32,
                "visibility_threshold": 0.80,
            },
        }
    )

    assert config.chunk_size == 120
    assert config.window_size == 80
    assert config.chunk_step == 40
    assert config.cache.grid_stride == 1
    assert config.cache.region == "bottom-half"
    assert config.cache.query_mode == "confidence-refresh"
    assert config.cache.abort_confidence_threshold == 0.72
    assert config.cache.max_track_frames == 100
    assert config.cache.visibility_threshold == 0.80
    assert config.cache.device == "cpu"
    assert config.cache.batch_size == 32


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


def write_synthetic_cotracker_cache(
    root: Path,
    frame_count: int = 6,
    *,
    frame_start: int = 0,
    confidence_value: float = 0.75,
) -> Path:
    root.mkdir(parents=True)
    point_ids = np.array([0, 1, 2], dtype=np.int64)
    birth_reverse_times = np.array([0, 0, 2], dtype=np.int32)
    frame_end = frame_start + frame_count
    source_birth_frames = np.array([frame_end - 1, frame_end - 1, frame_end - 3], dtype=np.int32)
    seed_xy = np.array([[10, 10], [30, 20], [50, 30]], dtype=np.float32)
    tracks = np.zeros((frame_count, len(point_ids), 2), dtype=np.float32)
    visibility = np.ones((frame_count, len(point_ids)), dtype=bool)
    confidence = np.ones((frame_count, len(point_ids)), dtype=np.float32) * float(confidence_value)
    for t in range(frame_count):
        tracks[t, :, 0] = seed_xy[:, 0] + t
        tracks[t, :, 1] = seed_xy[:, 1] + t * 2
    visibility[:2, 2] = False
    tracks[:2, 2] = np.nan
    confidence[:2, 2] = 0

    np.save(root / "tracks_reverse.npy", tracks)
    np.save(root / "visibility_reverse.npy", visibility)
    np.save(root / "confidence_reverse.npy", confidence)
    np.save(root / "point_ids.npy", point_ids)
    np.save(root / "birth_reverse_times.npy", birth_reverse_times)
    np.save(root / "source_birth_frames.npy", source_birth_frames)
    np.save(root / "seed_xy.npy", seed_xy)
    metadata = {
        "schema": CACHE_SCHEMA,
        "source_root": "/tmp/source",
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_count": frame_count,
        "width": 64,
        "height": 48,
        "point_count": len(point_ids),
        "config": {"grid_stride": 1, "region": "full", "query_mode": "confidence-refresh"},
        "tracker": {"name": "cotracker"},
        "arrays": {
            "tracks": "tracks_reverse.npy",
            "visibility": "visibility_reverse.npy",
            "confidence": "confidence_reverse.npy",
            "point_ids": "point_ids.npy",
            "birth_reverse_times": "birth_reverse_times.npy",
            "source_birth_frames": "source_birth_frames.npy",
            "seed_xy": "seed_xy.npy",
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


def test_cached_cotracker_backend_slices_cache(tmp_path: Path) -> None:
    cache_root = write_synthetic_cotracker_cache(tmp_path / "cache")
    tracker = CachedCoTrackerBackend(cache_root, max_match_distance=5)
    tracker.set_window_context(WindowSpec(start=1, end=5), tmp_path)
    frames = np.zeros((4, 48, 64, 3), dtype=np.uint8)
    queries = [
        QueryPoint(id=0, reverse_time=0, x=11, y=12, side=-1),
        QueryPoint(id=1, reverse_time=2, x=13, y=16, side=1),
        QueryPoint(id=2, reverse_time=0, x=63, y=47, side=0),
    ]

    bundle = tracker.track(frames, queries)

    assert bundle.tracker.name == "cotracker_cache"
    assert bundle.extra_arrays["cache_point_ids"].tolist() == [0, 0, -1]
    assert bundle.extra_arrays["cache_query_source_frames"].tolist() == [4, 2, 4]
    assert bundle.visibility[:, 0].tolist() == [True, True, True, True]
    assert bundle.visibility[:2, 1].tolist() == [False, False]
    assert bundle.visibility[2:, 1].tolist() == [True, True]
    assert not bundle.visibility[:, 2].any()
    assert bundle.confidence is not None
    assert bundle.confidence.shape == (4, 3)


def test_chunked_cache_backend_selects_highest_confidence_chunk(tmp_path: Path) -> None:
    chunked_root = tmp_path / "chunked"
    low_chunk = write_synthetic_cotracker_cache(
        chunked_root / "chunk_000000_000005",
        frame_count=6,
        frame_start=0,
        confidence_value=0.25,
    )
    high_chunk = write_synthetic_cotracker_cache(
        chunked_root / "chunk_000002_000007",
        frame_count=6,
        frame_start=2,
        confidence_value=0.95,
    )
    manifest = {
        "schema": CHUNKED_CACHE_SCHEMA,
        "source_root": "/tmp/source",
        "frame_count": 8,
        "chunk_size": 6,
        "window_size": 4,
        "chunk_step": 2,
        "overlap": 4,
        "selection": {"mode": "full_window_single_chunk"},
        "chunks": [
            {
                "chunk_id": "chunk_000000_000005",
                "chunk_index": 0,
                "path": str(low_chunk),
                "metadata": str(low_chunk / "metadata.json"),
                "frame_start": 0,
                "frame_end": 6,
                "frame_count": 6,
                "point_count": 3,
            },
            {
                "chunk_id": "chunk_000002_000007",
                "chunk_index": 1,
                "path": str(high_chunk),
                "metadata": str(high_chunk / "metadata.json"),
                "frame_start": 2,
                "frame_end": 8,
                "frame_count": 6,
                "point_count": 3,
            },
        ],
    }
    (chunked_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    tracker = CachedCoTrackerBackend(chunked_root, max_match_distance=5)
    tracker.set_window_context(WindowSpec(start=2, end=6), tmp_path)
    frames = np.zeros((4, 48, 64, 3), dtype=np.uint8)
    queries = [QueryPoint(id=0, reverse_time=0, x=12, y=14, side=0)]

    bundle = tracker.track(frames, queries)

    assert bundle.extra_metadata["cache_reference"]["chunk_id"] == "chunk_000002_000007"
    assert bundle.extra_metadata["cache_reference"]["candidate_chunks"] == 2
    assert bundle.extra_arrays["cache_point_ids"].tolist() == [0]
    assert bundle.extra_arrays["cache_point_indices"].tolist() == [0]
    assert bundle.extra_arrays["cache_chunk_indices"].tolist() == [1]
    assert bundle.extra_arrays["cache_chunk_ids"].tolist() == ["chunk_000002_000007"]
    assert bundle.extra_arrays["cache_unique_point_ids"].tolist() == ["chunk_000002_000007:0"]
    assert bundle.confidence is not None
    assert np.isclose(float(bundle.confidence.mean()), 0.95)


def test_cache_record_ids_only_artifacts(tmp_path: Path) -> None:
    frames = np.zeros((4, 48, 64, 3), dtype=np.uint8)
    queries = [QueryPoint(id=0, reverse_time=0, x=10, y=10, side=0)]
    bundle = TrackingBundle(
        tracks=np.zeros((4, 1, 2), dtype=np.float32),
        visibility=np.ones((4, 1), dtype=bool),
        tracker=TrackerInfo(name="cotracker_cache"),
        extra_arrays={
            "cache_point_ids": np.array([7], dtype=np.int64),
            "cache_query_source_frames": np.array([3], dtype=np.int32),
        },
        extra_metadata={
            "cache_reference": {
                "schema": CACHE_SCHEMA,
                "path": "/tmp/cache",
                "matched_queries": 1,
                "query_count": 1,
            }
        },
    )
    config = InverseTrackConfig(cache_record_ids_only=True)
    from avt.inverse import write_window_artifacts

    write_window_artifacts(tmp_path, WindowSpec(0, 4), frames, queries, bundle, config)
    arrays = np.load(tmp_path / "windows" / "seq_0_4" / "tracks.npz")
    metadata = json.loads((tmp_path / "windows" / "seq_0_4" / "window.json").read_text())

    assert "tracks_reverse" not in arrays
    assert arrays["cache_point_ids"].tolist() == [7]
    assert arrays["cache_query_source_frames"].tolist() == [3]
    assert metadata["artifact_mode"] == "cache_ids_only"
    assert metadata["cache_reference"]["matched_queries"] == 1


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


def test_write_viewer_serializes_nonfinite_numbers_as_null(tmp_path: Path) -> None:
    write_viewer(
        tmp_path,
        {
            "metadata": {"bad_float": float("nan")},
            "frames": [],
            "windows": [{"bad_numpy_float": np.float32(np.inf)}],
        },
    )

    text = (tmp_path / "data" / "prediction_tracks.json").read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    payload = json.loads(text)
    assert payload["metadata"]["bad_float"] is None
    assert payload["windows"][0]["bad_numpy_float"] is None


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
    arrays = np.load(output_root / "windows" / "seq_0_4" / "tracks.npz")
    assert arrays["confidence_reverse"].shape == arrays["visibility_reverse"].shape
    assert arrays["confidence_reverse"].dtype == np.float32

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


def test_stationary_sift_points_disable_segment() -> None:
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

    reasons = detect_stationary_sift_frames(
        tracks,
        visibility,
        seq_start=0,
        seq_end=12,
        sift_point_ids=[0, 1, 2],
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
