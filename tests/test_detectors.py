"""Tests for the pluggable Stage-2 detectors (sift / orb / superpoint / xfeat).

SIFT + ORB are exercised for real (OpenCV only). The torch-based learned
detectors are checked for construction/dispatch and their install-hint error
path when torch is absent; their actual model runs need a torch environment.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from avt.config import InverseTrackConfig
from avt.detectors import build_detector
from avt.detectors.config import OrbDetectorConfig, SuperPointConfig, XFeatConfig
from avt.detectors.orb import OrbDetector
from avt.detectors.sift import SiftDetector
from avt.detectors.superpoint import SuperPointSuperGlueDetector
from avt.detectors.xfeat import XFeatDetector
from avt.inverse import build_queries, run_inverse_tracking
from avt.io import read_frame_records
from avt.querying import QueryConfig, SiftCaptureConfig, VirtualRobotConfig, robot_sift_mask
from avt.schema import QueryPoint, TrackerInfo
from avt.tracking.base import TrackingBundle


class FakeTracker:
    def track(self, frames_rgb: np.ndarray, queries: list[QueryPoint]) -> TrackingBundle:
        tracks = np.full((len(frames_rgb), len(queries), 2), np.nan, dtype=np.float32)
        visibility = np.zeros((len(frames_rgb), len(queries)), dtype=bool)
        for q in queries:
            tracks[q.reverse_time :, q.id] = [q.x, q.y]
            visibility[q.reverse_time :, q.id] = True
        return TrackingBundle(tracks=tracks, visibility=visibility, tracker=TrackerInfo(name="fake"))


def _textured_frames(count: int = 6, h: int = 120, w: int = 160) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.stack([rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(count)])


def _write_textured_frames(root: Path, count: int = 6, h: int = 120, w: int = 160) -> None:
    root.mkdir()
    frames = _textured_frames(count, h, w)
    for idx, frame in enumerate(frames):
        cv2.imwrite(str(root / f"{idx:04d}.png"), frame)


def _assert_raises(exc_type, needle, fn, *args):
    try:
        fn(*args)
    except exc_type as exc:
        assert needle in str(exc), f"expected {needle!r} in {exc!r}"
        return
    raise AssertionError(f"expected {exc_type.__name__} mentioning {needle!r}")


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def test_sift_detector_matches_raw_sift() -> None:
    """SiftDetector must reproduce the exact prior inline SIFT path."""
    frames = _textured_frames(2)
    cfg = SiftCaptureConfig()
    kps = SiftDetector(cfg, cfg).detect(frames, 0, None)

    gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=float(cfg.clahe_clip_limit),
        tileGridSize=(cfg.clahe_tile_grid_size, cfg.clahe_tile_grid_size),
    )
    gray = clahe.apply(gray)
    raw = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=cfg.n_octave_layers,
        contrastThreshold=cfg.contrast_threshold,
        edgeThreshold=cfg.edge_threshold,
        sigma=cfg.sigma,
    )
    raw_kps, _ = raw.detectAndCompute(gray, None)
    assert len(kps) == len(raw_kps) > 0
    assert [round(k.pt[0], 3) for k in kps] == [round(k.pt[0], 3) for k in raw_kps]
    assert [round(k.response, 6) for k in kps] == [round(k.response, 6) for k in raw_kps]


def test_orb_detector_respects_mask() -> None:
    frames = _textured_frames(2)
    h, w = frames.shape[1:3]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // 2 :, :] = 255  # bottom half only
    kps = OrbDetector(OrbDetectorConfig()).detect(frames, 0, mask)
    assert kps, "ORB found no keypoints on textured frame"
    assert all(kp.pt[1] >= h // 2 for kp in kps)


def test_orb_build_queries_within_footprint() -> None:
    frames = _textured_frames(6)
    h, w = frames.shape[1:3]
    # Roomy footprint without edge-carving so ORB's 31px patch fits inside the mask.
    config = InverseTrackConfig(
        query_config=QueryConfig(
            mode="sift",
            detector="orb",
            robot=VirtualRobotConfig(width_ratio=0.6, height_ratio=0.5),
            sift=SiftCaptureConfig(sample_at_edges=False),
        )
    )
    queries = build_queries(w, h, len(frames), config, frames_rgb=frames)
    assert queries
    assert all(q.source == "sift_robot" for q in queries)
    mask = robot_sift_mask(h, w, config.query_config.robot, config.query_config.sift)
    for q in queries:
        assert mask[int(round(q.y)), int(round(q.x))] > 0
        assert q.response is not None  # ORB Harris score recorded


def test_build_detector_dispatch() -> None:
    base = QueryConfig()
    expected = {
        "sift": SiftDetector,
        "orb": OrbDetector,
        "superpoint": SuperPointSuperGlueDetector,  # construction only; no torch needed
        "xfeat": XFeatDetector,
    }
    for name, cls in expected.items():
        cfg = replace(base, detector=name)
        assert isinstance(build_detector(cfg, base.sift), cls)
    _assert_raises(ValueError, "Unknown detector", build_detector, replace(base, detector="nope"))


def test_learned_detectors_install_hint_without_torch() -> None:
    if _torch_available():
        return  # torch present: the lazy-load path won't raise; nothing to assert
    frames = _textured_frames(2)
    _assert_raises(ImportError, "xfeat", XFeatDetector(XFeatConfig()).detect, frames, 0, None)
    _assert_raises(
        ImportError, "superpoint", SuperPointSuperGlueDetector(SuperPointConfig()).detect, frames, 0, None
    )


def test_orb_end_to_end_run(tmp_path: Path) -> None:
    frames_root = tmp_path / "frames"
    _write_textured_frames(frames_root, 6)
    records = read_frame_records(frames_root, "image_dir")
    config = InverseTrackConfig(
        window_size=6,
        window_step=6,
        max_windows=1,
        query_config=QueryConfig(mode="ventura", detector="orb"),
    )
    windows = run_inverse_tracking(frames_root, records, tmp_path / "out", FakeTracker(), config)
    window_dir = tmp_path / "out" / "windows" / windows[0].id
    arrays = np.load(window_dir / "tracks.npz")
    assert arrays["tracks_reverse"].shape[1] > 0  # ORB produced query points
    meta = json.loads((window_dir / "window.json").read_text())
    assert meta["config"]["query_config"]["detector"] == "orb"
