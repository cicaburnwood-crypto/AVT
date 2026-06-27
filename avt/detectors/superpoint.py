"""SuperPoint detector (HuggingFace Transformers), with an optional SuperGlue
cross-frame match prefilter.

Default path (``use_superglue=False``): pure SuperPoint keypoint detection -
keypoints inside the mask ranked by detector score.

Optional path (``use_superglue=True``, kept for later use): SuperGlue (which uses
SuperPoint as its front-end) matches keypoints between the target frame and a
neighbor frame; only confidently-matched keypoints are kept (response = match
score), biasing the query set toward repeatable / trackable points.

Requires torch + transformers. Models are loaded lazily and cached at module
level. NOTE: magic-leap-community/superglue_outdoor weights are
research/non-commercial.
"""

from __future__ import annotations

import numpy as np

from .base import filter_keypoints_by_mask, make_keypoint
from .config import SuperPointConfig

_MODEL_CACHE: dict = {}


def _resolve_device(device: str | None) -> str:
    import torch

    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_superpoint(config: SuperPointConfig):
    try:
        import torch  # noqa: F401
        from transformers import AutoImageProcessor, SuperPointForKeypointDetection
    except ImportError as exc:  # pragma: no cover - env guard
        raise ImportError(
            "The 'superpoint' detector requires torch + transformers. "
            "Install with: pip install 'avt[superpoint]'"
        ) from exc

    device = _resolve_device(config.device)
    key = ("superpoint", config.model, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    processor = AutoImageProcessor.from_pretrained(config.model)
    model = SuperPointForKeypointDetection.from_pretrained(config.model).to(device).eval()
    _MODEL_CACHE[key] = (processor, model, device)
    return _MODEL_CACHE[key]


def _load_superglue(config: SuperPointConfig):
    try:
        import torch  # noqa: F401
        from transformers import AutoImageProcessor, AutoModelForKeypointMatching
    except ImportError as exc:  # pragma: no cover - env guard
        raise ImportError(
            "The 'superpoint' detector requires torch + transformers. "
            "Install with: pip install 'avt[superpoint]'"
        ) from exc

    device = _resolve_device(config.device)
    key = ("superglue", config.superglue_model, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    processor = AutoImageProcessor.from_pretrained(config.superglue_model)
    model = AutoModelForKeypointMatching.from_pretrained(config.superglue_model).to(device).eval()
    _MODEL_CACHE[key] = (processor, model, device)
    return _MODEL_CACHE[key]


class SuperPointSuperGlueDetector:
    def __init__(self, config: SuperPointConfig) -> None:
        self._config = config
        self._sp_bundle = None
        self._sg_bundle = None

    def detect(self, frames_rgb: np.ndarray, reverse_time: int, mask: np.ndarray | None):
        if self._config.use_superglue:
            return self._detect_with_superglue(frames_rgb, reverse_time, mask)
        return self._detect_superpoint_only(frames_rgb, reverse_time, mask)

    # --- default: pure SuperPoint detection ---------------------------------
    def _detect_superpoint_only(
        self, frames_rgb: np.ndarray, reverse_time: int, mask: np.ndarray | None
    ):
        if self._sp_bundle is None:
            self._sp_bundle = _load_superpoint(self._config)
        processor, model, device = self._sp_bundle

        import torch

        img = np.ascontiguousarray(frames_rgb[reverse_time])
        inputs = processor([img], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        h, w = img.shape[:2]
        results = processor.post_process_keypoint_detection(outputs, [(h, w)])
        result = results[0]
        kpts = result["keypoints"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy()
        keypoints = [
            make_keypoint(float(x), float(y), float(s))
            for (x, y), s in zip(kpts, scores)
        ]
        return filter_keypoints_by_mask(keypoints, mask)

    # --- optional: SuperGlue cross-frame match prefilter (kept for later) ----
    def _neighbor_index(self, reverse_time: int, frame_count: int) -> int:
        off = max(1, int(self._config.neighbor_offset))
        if reverse_time + off < frame_count:
            return reverse_time + off
        if reverse_time - off >= 0:
            return reverse_time - off
        return -1

    def _detect_with_superglue(
        self, frames_rgb: np.ndarray, reverse_time: int, mask: np.ndarray | None
    ):
        frame_count = int(frames_rgb.shape[0])
        neighbor = self._neighbor_index(reverse_time, frame_count)
        if neighbor < 0:  # single-frame window: no pair to match
            return []
        if self._sg_bundle is None:
            self._sg_bundle = _load_superglue(self._config)
        processor, model, device = self._sg_bundle

        import torch

        img0 = np.ascontiguousarray(frames_rgb[reverse_time])
        img1 = np.ascontiguousarray(frames_rgb[neighbor])
        inputs = processor([[img0, img1]], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        h, w = img0.shape[:2]
        target_sizes = [[(h, w), (h, w)]]
        results = processor.post_process_keypoint_matching(
            outputs, target_sizes, threshold=float(self._config.match_threshold)
        )
        result = results[0]
        kpts0 = result["keypoints0"].detach().cpu().numpy()
        scores = result["matching_scores"].detach().cpu().numpy()
        keypoints = [
            make_keypoint(float(x), float(y), float(s))
            for (x, y), s in zip(kpts0, scores)
        ]
        return filter_keypoints_by_mask(keypoints, mask)
