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
    visibility_threshold: float = 0.9

    def _track_batch_with_public_api(self, model, video, batch, torch) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        outputs = model(video, queries=batch)
        pred_tracks, pred_visibility = outputs[:2]
        if pred_visibility.ndim == 4:
            pred_visibility = pred_visibility[..., 0]
        confidence = pred_visibility[0].detach().cpu().numpy().astype(np.float32)
        visibility = (
            pred_visibility[0].detach().cpu().numpy().astype(bool)
            if pred_visibility.dtype == torch.bool
            else confidence > float(self.visibility_threshold)
        )
        return (
            pred_tracks[0].detach().cpu().numpy().astype(np.float32),
            visibility.astype(bool),
            confidence,
            {"visibility_confidence": confidence},
        )

    def _track_batch_with_raw_confidence(
        self, model, video, batch, torch
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]] | None:
        if not all(hasattr(model, attr) for attr in ("model", "interp_shape", "support_grid_size")):
            return None

        try:
            from cotracker.models.core.model_utils import get_points_on_a_grid
        except Exception:
            return None

        batch_size, frame_count, channels, height, width = video.shape
        if batch_size != 1 or channels != 3:
            return None

        interp_height, interp_width = [int(value) for value in model.interp_shape]
        video_model = video.reshape(batch_size * frame_count, channels, height, width)
        video_model = torch.nn.functional.interpolate(
            video_model,
            (interp_height, interp_width),
            mode="bilinear",
            align_corners=True,
        )
        video_model = video_model.reshape(batch_size, frame_count, channels, interp_height, interp_width)

        queries = batch.clone()
        queries[:, :, 1:] *= queries.new_tensor(
            [
                (interp_width - 1) / (width - 1),
                (interp_height - 1) / (height - 1),
            ]
        )
        requested_count = queries.shape[1]
        model_queries = queries

        support_grid_size = int(getattr(model, "support_grid_size", 0) or 0)
        if support_grid_size > 0:
            support_points = get_points_on_a_grid(
                support_grid_size,
                model.interp_shape,
                device=video.device,
            )
            support_queries = torch.cat(
                [torch.zeros_like(support_points[:, :, :1]), support_points],
                dim=2,
            ).repeat(batch_size, 1, 1)
            model_queries = torch.cat([queries, support_queries], dim=1)

        outputs = model.model.forward(video=video_model, queries=model_queries, iters=6)
        pred_tracks = outputs[0]
        visibility_confidence = outputs[1]
        tracking_confidence = None
        if len(outputs) > 2 and hasattr(outputs[2], "shape") and outputs[2].shape[:3] == visibility_confidence.shape[:3]:
            tracking_confidence = outputs[2]

        if visibility_confidence.ndim == 4:
            visibility_confidence = visibility_confidence[..., 0]
        if tracking_confidence is not None and tracking_confidence.ndim == 4:
            tracking_confidence = tracking_confidence[..., 0]

        pred_tracks = pred_tracks[:, :, :requested_count]
        visibility_confidence = visibility_confidence[:, :, :requested_count]
        if tracking_confidence is not None:
            tracking_confidence = tracking_confidence[:, :, :requested_count]

        for batch_index in range(batch_size):
            query_times = queries[batch_index, :requested_count, 0].to(torch.int64).clamp(0, frame_count - 1)
            query_ids = torch.arange(0, requested_count, device=video.device)
            pred_tracks[batch_index, query_times, query_ids] = queries[batch_index, :requested_count, 1:]
            visibility_confidence[batch_index, query_times, query_ids] = 1.0
            if tracking_confidence is not None:
                tracking_confidence[batch_index, query_times, query_ids] = 1.0

        pred_tracks *= pred_tracks.new_tensor(
            [
                (width - 1) / (interp_width - 1),
                (height - 1) / (interp_height - 1),
            ]
        )

        visibility_np = (visibility_confidence[0] > float(self.visibility_threshold)).detach().cpu().numpy().astype(bool)
        visibility_confidence_np = visibility_confidence[0].detach().cpu().numpy().astype(np.float32)
        components = {"visibility_confidence": visibility_confidence_np}
        if tracking_confidence is not None:
            components["tracking_confidence"] = tracking_confidence[0].detach().cpu().numpy().astype(np.float32)

        return (
            pred_tracks[0].detach().cpu().numpy().astype(np.float32),
            visibility_np,
            visibility_confidence_np,
            components,
        )

    def _track_batch(self, model, video, batch, torch) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        try:
            raw = self._track_batch_with_raw_confidence(model, video, batch, torch)
        except Exception:
            raw = None
        if raw is not None:
            return raw
        return self._track_batch_with_public_api(model, video, batch, torch)

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
        confidence_batches: list[np.ndarray] = []
        confidence_component_batches: dict[str, list[np.ndarray]] = {}
        with torch.inference_mode():
            for start in range(0, len(queries), self.batch_size):
                batch = query_tensor[start : start + self.batch_size][None]
                batch_tracks, batch_visibility, batch_confidence, batch_components = self._track_batch(
                    model,
                    video,
                    batch,
                    torch,
                )
                track_batches.append(batch_tracks)
                visibility_batches.append(batch_visibility)
                confidence_batches.append(batch_confidence)
                for name, component in batch_components.items():
                    confidence_component_batches.setdefault(name, []).append(component)

        tracks = np.concatenate(track_batches, axis=1).astype(np.float32)
        visibility = np.concatenate(visibility_batches, axis=1).astype(bool)
        confidence = np.concatenate(confidence_batches, axis=1).astype(np.float32)
        confidence_components = {
            name: np.concatenate(batches, axis=1).astype(np.float32)
            for name, batches in confidence_component_batches.items()
        }

        for idx, query in enumerate(queries):
            if query.reverse_time > 0:
                visibility[: query.reverse_time, idx] = False
                tracks[: query.reverse_time, idx] = np.nan
                confidence[: query.reverse_time, idx] = 0.0
                for component in confidence_components.values():
                    component[: query.reverse_time, idx] = 0.0

        bundle = TrackingBundle(
            tracks=tracks,
            visibility=visibility,
            confidence=confidence,
            confidence_components=confidence_components,
            tracker=TrackerInfo(
                name="cotracker",
                parameters={
                    "hub_repo": self.hub_repo,
                    "hub_model": self.hub_model,
                    "device": device,
                    "batch_size": self.batch_size,
                    "visibility_threshold": self.visibility_threshold,
                    "confidence": "raw_cotracker_visibility_probability",
                },
            ),
        )
        bundle.validate(frames_rgb.shape[0], len(queries))
        return bundle
