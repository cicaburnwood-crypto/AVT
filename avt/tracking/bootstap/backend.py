from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import types

import cv2
import numpy as np

from ...schema import QueryPoint, TrackerInfo
from ..base import TrackingBundle
from .download import DEFAULT_BOOTSTAP_CHECKPOINT_URL, ensure_bootstap_checkpoint


@dataclass
class BootstapBackend:
    """BootsTAPIR adapter behind AVT's generic point-tracker interface.

    The adapter uses the official `tapnet.torch.tapir_model.TAPIR` implementation
    and the PyTorch BootsTAPIR checkpoint. Frames and queries are converted into
    TAPNet's `[t, y, x]` resized-raster convention, while outputs are returned in
    AVT's backend-neutral reversed-video `[T, N, 2]` `[x, y]` convention.
    """

    device: str = "auto"
    checkpoint_path: Path | None = None
    checkpoint_url: str | None = DEFAULT_BOOTSTAP_CHECKPOINT_URL
    download_checkpoint: bool = False
    resize_height: int = 512
    resize_width: int = 512
    query_chunk_size: int = 64
    pyramid_level: int = 1
    visibility_threshold: float = 0.5
    strict_checkpoint: bool = True

    def track(self, frames_rgb: np.ndarray, queries: list[QueryPoint]) -> TrackingBundle:
        try:
            import torch
            import torch.nn.functional as F
            tapir_model = _import_tapir_model()
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "BootsTAPIR backend requires PyTorch and the official TAPNet package. "
                "Install with `python -m pip install -e .[bootstap]` or install "
                "https://github.com/google-deepmind/tapnet separately."
            ) from exc

        if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
            raise ValueError("frames_rgb must have shape [T,H,W,3]")
        if not queries:
            raise ValueError("No query points were provided")
        if self.resize_height <= 0 or self.resize_width <= 0:
            raise ValueError("resize_height and resize_width must be positive")
        if self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")

        device = _resolve_device(self.device, torch)
        checkpoint = ensure_bootstap_checkpoint(
            self.checkpoint_path,
            download=self.download_checkpoint,
            url=self.checkpoint_url,
        )

        model = tapir_model.TAPIR(pyramid_level=int(self.pyramid_level))
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state, strict=bool(self.strict_checkpoint))
        model = model.to(device).eval()

        resized = _resize_frames(frames_rgb, self.resize_height, self.resize_width)
        query_points = _query_points_for_resize(
            queries,
            source_height=frames_rgb.shape[1],
            source_width=frames_rgb.shape[2],
            resize_height=self.resize_height,
            resize_width=self.resize_width,
        )

        video = torch.from_numpy(resized).to(device=device, dtype=torch.float32)
        video = video / 255.0 * 2.0 - 1.0
        query_tensor = torch.from_numpy(query_points).to(device=device, dtype=torch.float32)

        with torch.inference_mode():
            outputs = model(
                video[None],
                query_tensor[None],
                query_chunk_size=int(self.query_chunk_size),
            )
            tracks = outputs["tracks"][0].detach().cpu().numpy()
            occlusions = outputs["occlusion"][0]
            expected_dist = outputs["expected_dist"][0]
            visibility = (
                (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist))
                > float(self.visibility_threshold)
            )
            visibility_np = visibility.detach().cpu().numpy().astype(bool)

        tracks_avt, visibility_avt = _tapnet_outputs_to_avt(
            tracks,
            visibility_np,
            queries,
            source_height=frames_rgb.shape[1],
            source_width=frames_rgb.shape[2],
            resize_height=self.resize_height,
            resize_width=self.resize_width,
        )

        bundle = TrackingBundle(
            tracks=tracks_avt,
            visibility=visibility_avt,
            tracker=TrackerInfo(
                name="bootstapir",
                parameters={
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_url": self.checkpoint_url,
                    "device": device,
                    "resize_height": self.resize_height,
                    "resize_width": self.resize_width,
                    "query_chunk_size": self.query_chunk_size,
                    "pyramid_level": self.pyramid_level,
                    "visibility_threshold": self.visibility_threshold,
                    "strict_checkpoint": self.strict_checkpoint,
                    "implementation": "google-deepmind/tapnet.torch.tapir_model.TAPIR",
                },
            ),
        )
        bundle.validate(frames_rgb.shape[0], len(queries))
        return bundle


def _import_tapir_model():
    try:
        from tapnet.torch import tapir_model

        return tapir_model
    except ModuleNotFoundError as exc:
        if exc.name not in {"tensorflow", "tensorflow_datasets"}:
            raise
        return _import_tapir_model_without_legacy_init()


def _import_tapir_model_without_legacy_init():
    """Import TAPNet's PyTorch model without loading legacy TF/JAX exports."""

    for name in list(sys.modules):
        if name == "tapnet" or name.startswith("tapnet."):
            sys.modules.pop(name, None)

    spec = importlib.util.find_spec("tapnet")
    if spec is None or spec.submodule_search_locations is None:
        raise ModuleNotFoundError("No module named 'tapnet'")

    root = Path(next(iter(spec.submodule_search_locations)))
    package_spec = importlib.machinery.ModuleSpec("tapnet", loader=None, is_package=True)
    package = types.ModuleType("tapnet")
    package.__path__ = [str(root)]
    package.__package__ = "tapnet"
    package.__spec__ = package_spec
    sys.modules["tapnet"] = package

    torch_spec = importlib.machinery.ModuleSpec("tapnet.torch", loader=None, is_package=True)
    torch_package = types.ModuleType("tapnet.torch")
    torch_package.__path__ = [str(root / "torch")]
    torch_package.__package__ = "tapnet.torch"
    torch_package.__spec__ = torch_spec
    sys.modules["tapnet.torch"] = torch_package

    return importlib.import_module("tapnet.torch.tapir_model")


def _resolve_device(device: str, torch_module) -> str:
    if device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    return device


def _resize_frames(frames_rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    if frames_rgb.shape[1] == height and frames_rgb.shape[2] == width:
        return np.ascontiguousarray(frames_rgb)
    resized = [
        cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        for frame in frames_rgb
    ]
    return np.ascontiguousarray(np.stack(resized, axis=0))


def _query_points_for_resize(
    queries: list[QueryPoint],
    *,
    source_height: int,
    source_width: int,
    resize_height: int,
    resize_width: int,
) -> np.ndarray:
    scale_x = resize_width / float(source_width)
    scale_y = resize_height / float(source_height)
    rows = [
        [query.reverse_time, query.y * scale_y, query.x * scale_x]
        for query in queries
    ]
    return np.asarray(rows, dtype=np.float32)


def _tapnet_outputs_to_avt(
    tracks: np.ndarray,
    visibility: np.ndarray,
    queries: list[QueryPoint],
    *,
    source_height: int,
    source_width: int,
    resize_height: int,
    resize_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"BootsTAPIR tracks must have shape [N,T,2], got {tracks.shape}")
    if visibility.shape != tracks.shape[:2]:
        raise ValueError(
            f"BootsTAPIR visibility shape {visibility.shape} does not match {tracks.shape[:2]}"
        )

    scale_x = source_width / float(resize_width)
    scale_y = source_height / float(resize_height)
    tracks_avt = np.transpose(tracks, (1, 0, 2)).astype(np.float32)
    tracks_avt[..., 0] *= scale_x
    tracks_avt[..., 1] *= scale_y
    visibility_avt = np.transpose(visibility, (1, 0)).astype(bool)

    for idx, query in enumerate(queries):
        if query.reverse_time > 0:
            visibility_avt[: query.reverse_time, idx] = False
            tracks_avt[: query.reverse_time, idx] = np.nan

    out_of_bounds = (
        (tracks_avt[..., 0] < 0)
        | (tracks_avt[..., 0] >= source_width)
        | (tracks_avt[..., 1] < 0)
        | (tracks_avt[..., 1] >= source_height)
        | ~np.isfinite(tracks_avt).all(axis=2)
    )
    visibility_avt[out_of_bounds] = False
    tracks_avt[~visibility_avt] = np.nan
    return tracks_avt, visibility_avt
