"""XFeat detector (torch.hub ``verlab/accelerated_features``).

Lightweight learned features. Requires torch; the model is loaded lazily and
cached at module level so repeated calls (anchors + crumbs, many windows) reuse
one instance. Only keypoint locations + scores are used; descriptors discarded.
"""

from __future__ import annotations

import numpy as np

from .base import filter_keypoints_by_mask, make_keypoint
from .config import XFeatConfig

_MODEL_CACHE: dict = {}


def _resolve_device(device: str | None) -> str:
    import torch

    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_xfeat(config: XFeatConfig):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - env guard
        raise ImportError(
            "The 'xfeat' detector requires torch. Install with: pip install 'avt[xfeat]'"
        ) from exc

    device = _resolve_device(config.device)
    key = (config.hub_repo, config.model, int(config.top_k), device, str(config.checkpoint))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    model = torch.hub.load(
        config.hub_repo,
        config.model,
        pretrained=(config.checkpoint is None),
        top_k=int(config.top_k),
    )
    if config.checkpoint:
        state = torch.load(config.checkpoint, map_location="cpu")
        model.net.load_state_dict(state)
    try:
        model = model.to(device)
    except Exception:  # pragma: no cover - some hub wrappers manage device internally
        pass
    _MODEL_CACHE[key] = model
    return model


class XFeatDetector:
    def __init__(self, config: XFeatConfig) -> None:
        self._config = config
        self._model = None

    def detect(self, frames_rgb: np.ndarray, reverse_time: int, mask: np.ndarray | None):
        if self._model is None:
            self._model = _load_xfeat(self._config)
        frame = np.ascontiguousarray(frames_rgb[reverse_time])
        output = self._model.detectAndCompute(frame, top_k=int(self._config.top_k))[0]
        kpts = output["keypoints"].detach().cpu().numpy()
        scores = output.get("scores")
        scores = (
            scores.detach().cpu().numpy()
            if scores is not None
            else np.ones(len(kpts), dtype=np.float32)
        )
        keypoints = [
            make_keypoint(float(x), float(y), float(s))
            for (x, y), s in zip(kpts, scores)
        ]
        return filter_keypoints_by_mask(keypoints, mask)
