from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..schema import QueryPoint, TrackerInfo
from .base import TrackingBundle


@dataclass
class CoTrackerBackend:
    """CoTracker backend behind the generic AVT point-tracker interface.

    This backend has no dependency on VENTURA paths or artifacts. It expects
    PyTorch plus a CoTracker hub-capable environment at runtime.
    """

    device: str = "auto"
    batch_size: int = 256
    torch_home: str | None = None
    hub_repo: str = "facebookresearch/co-tracker"
    hub_model: str = "cotracker3_offline"

    def track(self, frames_rgb: np.ndarray, queries: list[QueryPoint]) -> TrackingBundle:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("CoTracker backend requires PyTorch.") from exc

        if not queries:
            raise ValueError("No query points were provided")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
            raise ValueError("frames_rgb must have shape [T,H,W,3]")

        if self.torch_home:
            os.environ.setdefault("TORCH_HOME", self.torch_home)
            Path(self.torch_home).mkdir(parents=True, exist_ok=True)

        device = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        if device == "auto":
            device = "cpu"

        video = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2)[None].float().to(device)
        query_rows = [[q.reverse_time, q.x, q.y] for q in queries]
        query_tensor = torch.tensor(query_rows, dtype=torch.float32, device=device)

        try:
            model = torch.hub.load(
                self.hub_repo,
                self.hub_model,
                trust_repo=True,
            ).to(device)
        except Exception as exc:  # pragma: no cover - depends on local cache/network
            raise RuntimeError(
                "Could not load CoTracker. Install/cache CoTracker or use "
                "--backend lk / a custom PointTracker."
            ) from exc
        model.eval()

        track_batches: list[np.ndarray] = []
        visibility_batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(queries), self.batch_size):
                batch = query_tensor[start : start + self.batch_size][None]
                pred_tracks, pred_visibility = model(video, queries=batch)
                if pred_visibility.ndim == 4:
                    pred_visibility = pred_visibility[..., 0]
                track_batches.append(pred_tracks[0].detach().cpu().numpy())
                visibility_batches.append(
                    pred_visibility[0].detach().cpu().numpy().astype(bool)
                )

        tracks = np.concatenate(track_batches, axis=1).astype(np.float32)
        visibility = np.concatenate(visibility_batches, axis=1).astype(bool)

        for idx, query in enumerate(queries):
            if query.reverse_time > 0:
                visibility[: query.reverse_time, idx] = False
                tracks[: query.reverse_time, idx] = np.nan

        bundle = TrackingBundle(
            tracks=tracks,
            visibility=visibility,
            tracker=TrackerInfo(
                name="cotracker",
                parameters={
                    "hub_repo": self.hub_repo,
                    "hub_model": self.hub_model,
                    "device": device,
                    "batch_size": self.batch_size,
                },
            ),
        )
        bundle.validate(frames_rgb.shape[0], len(queries))
        return bundle
