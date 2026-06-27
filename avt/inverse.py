"""AVT inverse tracking orchestrator.

The pipeline is implemented as four standalone stages under ``avt.pipeline``
(preprocess -> extract -> track -> combine). This module wires them together in
``run_inverse_tracking`` and re-exports the moved public names so existing
imports (``from avt.inverse import ...``) keep working unchanged.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .config import InverseTrackConfig
from .pipeline.combine import (
    _reference_support_points,
    reference_mask,
    write_window_artifacts,
)
from .pipeline.extract import PointExtractor, SiftQueryExtractor, build_queries
from .pipeline.preprocess import (
    PreparedWindow,
    _avt_seed_ratios,
    build_windows,
    prepare_window,
)
from .pipeline.track import run_tracking
from .schema import FrameRecord, WindowSpec
from .tracking.base import PointTracker, TrackingBundle

__all__ = [
    "InverseTrackConfig",
    "PreparedWindow",
    "PointExtractor",
    "SiftQueryExtractor",
    "build_windows",
    "prepare_window",
    "build_queries",
    "reference_mask",
    "write_window_artifacts",
    "run_inverse_tracking",
]


def run_inverse_tracking(
    source_root: Path,
    frame_records: list[FrameRecord],
    output_dir: Path,
    tracker: PointTracker,
    config: InverseTrackConfig,
    *,
    extractor: PointExtractor | None = None,
) -> list[WindowSpec]:
    extractor = extractor or SiftQueryExtractor()
    source_root = source_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    windows = build_windows(len(frame_records), config)
    run_metadata = {
        "source_root": str(source_root),
        "frame_count": len(frame_records),
        "config": asdict(config),
        "windows": [asdict(win) | {"id": win.id} for win in windows],
    }
    (output_dir / "run.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    for ordinal, window in enumerate(windows, start=1):
        print(f"[{ordinal}/{len(windows)}] tracking {window.id}", flush=True)
        prepared = prepare_window(source_root, frame_records, window, config)   # Stage 1
        queries = extractor.extract(prepared, config)                          # Stage 2
        bundle = run_tracking(prepared, queries, tracker, output_dir)          # Stage 3
        write_window_artifacts(                                                # Stage 4
            output_dir, prepared.window, prepared.frames_rgb, queries, bundle, config
        )
    return windows
