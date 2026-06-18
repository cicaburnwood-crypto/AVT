from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from .inverse import InverseTrackConfig, run_inverse_tracking
from .io import read_frame_records
from .querying import QueryConfig, load_query_config_yaml, merge_query_config
from .tracking import LKTracker
from .viewer import build_viewer


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"
DEFAULT_VIEWER_ROOT = DEFAULT_OUTPUT_ROOT / "viewer_runs"
DEFAULT_CACHE_ROOT = DEFAULT_OUTPUT_ROOT / "cotracker_caches"
DEFAULT_WINDOW_SIZE = 80
DEFAULT_CACHE_CHUNK_SIZE = 480
DEFAULT_CACHE_WINDOW_SIZE = 80


def _create_unique_run_dir(base: Path, preferred_name: str | None = None) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    stem = preferred_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    for attempt in range(100):
        name = stem if attempt == 0 else f"{stem}_{attempt:02d}"
        candidate = base / name
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"Could not create a unique run directory under {base}")


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frames-root", type=Path, required=True, help="Frame source root.")
    parser.add_argument(
        "--source-type",
        choices=("auto", "earth_rover", "color_txt", "image_dir"),
        default="auto",
        help="Frame source format.",
    )


def _add_inverse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--window-step", type=int, default=100)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--query-stride", type=int, default=10)
    parser.add_argument("--seed-count", type=int, default=17)
    parser.add_argument(
        "--seed-y-ratio",
        type=float,
        default=None,
        help="Override AVT seed-line y ratio for manual avt mode.",
    )
    parser.add_argument(
        "--seed-x-min-ratio",
        type=float,
        default=None,
        help="Override AVT seed-line left x ratio for manual avt mode.",
    )
    parser.add_argument(
        "--seed-x-max-ratio",
        type=float,
        default=None,
        help="Override AVT seed-line right x ratio for manual avt mode.",
    )
    parser.add_argument(
        "--query-mode",
        choices=("ventura", "avt", "sift", "avt+sift"),
        default=None,
        help="Query source mode. Defaults to YAML query_mode or ventura.",
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=None,
        help="YAML file with VENTURA footprint and SIFT query-capture settings.",
    )
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument(
        "--save-reverse-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write per-window reverse_video.mp4 files. Disabled by default to keep outputs small.",
    )
    parser.add_argument(
        "--no-reverse-video",
        action="store_true",
        help="Deprecated compatibility flag; reverse videos are already disabled by default.",
    )
    parser.add_argument(
        "--save-path-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write per-window path_mask_reference.png files. Disabled by default.",
    )
    parser.add_argument(
        "--path-support",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use VENTURA-style relaxed SIFT support points when drawing the reference path mask.",
    )
    parser.add_argument("--path-support-min-points", type=int, default=32)
    parser.add_argument("--path-support-fraction", type=int, default=6)


def _config_from_args(args: argparse.Namespace) -> InverseTrackConfig:
    query_config = (
        load_query_config_yaml(args.robot_config)
        if args.robot_config
        else QueryConfig()
    )
    query_config = merge_query_config(query_config, mode=args.query_mode)
    return InverseTrackConfig(
        window_size=args.window_size,
        window_step=args.window_step,
        fps=args.fps,
        query_stride=args.query_stride,
        seed_count=args.seed_count,
        seed_y_ratio=args.seed_y_ratio,
        seed_x_min_ratio=args.seed_x_min_ratio,
        seed_x_max_ratio=args.seed_x_max_ratio,
        query_config=query_config,
        max_windows=args.max_windows,
        save_reverse_video=bool(args.save_reverse_video and not args.no_reverse_video),
        save_path_mask=bool(args.save_path_mask),
        path_support_enabled=args.path_support,
        path_support_min_points=args.path_support_min_points,
        path_support_fraction=args.path_support_fraction,
        cache_record_ids_only=bool(getattr(args, "cache_record_ids_only", False)),
    )


def _tracker_from_args(args: argparse.Namespace):
    if args.backend == "lk":
        return LKTracker()
    if args.backend == "cotracker":
        from .tracking.cotracker import CoTrackerBackend

        return CoTrackerBackend(
            device=args.cotracker_device,
            batch_size=args.cotracker_batch_size,
            torch_home=args.torch_home,
            hub_repo=args.cotracker_hub_repo,
            hub_model=args.cotracker_hub_model,
            visibility_threshold=args.cotracker_visibility_threshold,
        )
    if args.backend == "cotracker_cache":
        if args.cotracker_cache is None:
            raise ValueError("--cotracker-cache is required for --backend cotracker_cache")
        from .tracking.cotracker_cache import CachedCoTrackerBackend

        return CachedCoTrackerBackend(
            cache_path=args.cotracker_cache,
            max_match_distance=args.cache_match_distance,
            source_root=args.frames_root,
        )
    if args.backend == "foundationpose":
        from .tracking.foundationpose import FoundationPoseBackend

        return FoundationPoseBackend(
            weights_dir=args.foundationpose_weights_dir,
            transforms_path=args.foundationpose_transforms,
            download_weights=args.foundationpose_download_weights,
            device=args.foundationpose_device,
        )
    if args.backend == "bootstap":
        from .tracking.bootstap import (
            BootstapBackend,
            BootstapConfig,
            load_bootstap_config_yaml,
        )
        from .tracking.bootstap.config import merge_bootstap_config

        config = (
            load_bootstap_config_yaml(args.bootstap_config)
            if args.bootstap_config
            else BootstapConfig()
        )
        config = merge_bootstap_config(
            config,
            device=args.bootstap_device,
            checkpoint_path=args.bootstap_checkpoint,
            checkpoint_url=args.bootstap_checkpoint_url,
            download_checkpoint=args.bootstap_download_checkpoint,
            resize_height=args.bootstap_resize_height,
            resize_width=args.bootstap_resize_width,
            query_chunk_size=args.bootstap_query_chunk_size,
            pyramid_level=args.bootstap_pyramid_level,
            visibility_threshold=args.bootstap_visibility_threshold,
            strict_checkpoint=args.bootstap_strict_checkpoint,
        )
        return BootstapBackend(**config.__dict__)
    raise ValueError(f"Unknown backend: {args.backend}")


def cmd_cache(args: argparse.Namespace) -> int:
    from .tracking.cotracker_cache import CoTrackerCacheConfig, build_cotracker_cache

    records = read_frame_records(args.frames_root, args.source_type)
    output_root = _create_unique_run_dir(args.output_root)
    metadata = build_cotracker_cache(
        source_root=args.frames_root,
        frame_records=records,
        output_dir=output_root,
        config=CoTrackerCacheConfig(
            frame_start=args.cache_frame_start,
            frame_count=args.cache_frame_count,
            grid_stride=args.cache_grid_stride,
            region=args.cache_region,
            query_mode=args.cache_query_mode,
            query_frame_stride=args.cache_query_frame_stride,
            max_query_points=args.cache_max_query_points,
            device=args.cotracker_device,
            batch_size=args.cotracker_batch_size,
            torch_home=args.torch_home,
            hub_repo=args.cotracker_hub_repo,
            hub_model=args.cotracker_hub_model,
            visibility_threshold=args.cotracker_visibility_threshold,
        ),
    )
    print(
        json.dumps(
            {
                "cache_base": str(args.output_root),
                "cache_root": str(output_root),
                "metadata": str(output_root / "metadata.json"),
                "frames": metadata["frame_count"],
                "points": metadata["point_count"],
                "region": metadata["config"]["region"],
                "query_mode": metadata["config"]["query_mode"],
            },
            indent=2,
        )
    )
    return 0


def cmd_cache_chunks(args: argparse.Namespace) -> int:
    from .tracking.cotracker_cache import (
        CoTrackerCacheConfig,
        CoTrackerChunkCacheConfig,
        build_cotracker_cache_chunks,
    )

    records = read_frame_records(args.frames_root, args.source_type)
    output_root = _create_unique_run_dir(args.output_root)
    manifest = build_cotracker_cache_chunks(
        source_root=args.frames_root,
        frame_records=records,
        output_dir=output_root,
        resume=bool(args.cache_resume),
        config=CoTrackerChunkCacheConfig(
            chunk_size=args.cache_chunk_size,
            window_size=args.cache_window_size,
            chunk_step=args.cache_chunk_step,
            cache=CoTrackerCacheConfig(
                frame_start=0,
                frame_count=args.cache_chunk_size,
                grid_stride=args.cache_grid_stride,
                region=args.cache_region,
                query_mode=args.cache_query_mode,
                query_frame_stride=args.cache_query_frame_stride,
                max_query_points=args.cache_max_query_points,
                device=args.cotracker_device,
                batch_size=args.cotracker_batch_size,
                torch_home=args.torch_home,
                hub_repo=args.cotracker_hub_repo,
                hub_model=args.cotracker_hub_model,
                visibility_threshold=args.cotracker_visibility_threshold,
            ),
        ),
    )
    print(
        json.dumps(
            {
                "cache_base": str(args.output_root),
                "cache_root": str(output_root),
                "manifest": str(output_root / "manifest.json"),
                "frames": manifest["frame_count"],
                "chunks": len(manifest["chunks"]),
                "chunk_size": manifest["chunk_size"],
                "chunk_step": manifest["chunk_step"],
                "window_size": manifest["window_size"],
            },
            indent=2,
        )
    )
    return 0


def cmd_track(args: argparse.Namespace) -> int:
    records = read_frame_records(args.frames_root, args.source_type)
    tracker = _tracker_from_args(args)
    output_root = _create_unique_run_dir(args.output_root)
    windows = run_inverse_tracking(
        source_root=args.frames_root,
        frame_records=records,
        output_dir=output_root,
        tracker=tracker,
        config=_config_from_args(args),
    )
    print(
        json.dumps(
            {
                "windows": len(windows),
                "output_base": str(args.output_root),
                "output_root": str(output_root),
            },
            indent=2,
        )
    )
    return 0


def cmd_viewer(args: argparse.Namespace) -> int:
    records = read_frame_records(args.frames_root, args.source_type)
    viewer_dir = _create_unique_run_dir(args.viewer_dir)
    payload = build_viewer(
        source_root=args.frames_root,
        frame_records=records,
        tracking_root=args.tracking_root,
        viewer_dir=viewer_dir,
        max_points_per_frame=args.max_points_per_frame,
        copy_frames=args.copy_frames,
        video_base_url=args.video_base_url,
    )
    print(
        json.dumps(
            {
                "viewer_base": str(args.viewer_dir),
                "viewer": str(viewer_dir / "index.html"),
                "frames": payload["metadata"]["frame_count"],
                "windows": payload["metadata"]["successful_windows"],
            },
            indent=2,
        )
    )
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    records = read_frame_records(args.frames_root, args.source_type)
    tracker = _tracker_from_args(args)
    output_root = _create_unique_run_dir(args.output_root)
    windows = run_inverse_tracking(
        source_root=args.frames_root,
        frame_records=records,
        output_dir=output_root,
        tracker=tracker,
        config=_config_from_args(args),
    )
    if not args.build_viewer:
        print(
            json.dumps(
                {
                    "output_base": str(args.output_root),
                    "output_root": str(output_root),
                    "viewer_base": str(args.viewer_dir) if args.viewer_dir else None,
                    "viewer": None,
                    "frames": len(records),
                    "windows": len(windows),
                    "viewer_note": "Viewer was not built. Run `avt viewer` or pass --build-viewer when needed.",
                },
                indent=2,
            )
        )
        return 0

    viewer_dir = (
        _create_unique_run_dir(args.viewer_dir, preferred_name=output_root.name)
        if args.viewer_dir
        else output_root / "viewer"
    )
    payload = build_viewer(
        source_root=args.frames_root,
        frame_records=records,
        tracking_root=output_root,
        viewer_dir=viewer_dir,
        max_points_per_frame=args.max_points_per_frame,
        copy_frames=args.copy_frames,
        video_base_url=args.video_base_url,
    )
    print(
        json.dumps(
            {
                "output_base": str(args.output_root),
                "output_root": str(output_root),
                "viewer_base": str(args.viewer_dir) if args.viewer_dir else None,
                "viewer": str(viewer_dir / "index.html"),
                "frames": payload["metadata"]["frame_count"],
                "windows": payload["metadata"]["successful_windows"],
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="avt",
        description="Decoupled inverse-video point tracking and WebUI visualization.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_cotracker_args(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("--cotracker-device", default="auto")
        cmd.add_argument("--cotracker-batch-size", type=int, default=256)
        cmd.add_argument("--cotracker-hub-repo", default="facebookresearch/co-tracker")
        cmd.add_argument("--cotracker-hub-model", default="cotracker3_offline")
        cmd.add_argument("--cotracker-visibility-threshold", type=float, default=0.9)
        cmd.add_argument("--torch-home", default=None)

    def add_cache_backend_args(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument(
            "--cotracker-cache",
            type=Path,
            default=None,
            help="Reusable CoTracker cache directory or metadata.json for cotracker_cache backend.",
        )
        cmd.add_argument("--cache-match-distance", type=float, default=12.0)
        cmd.add_argument(
            "--cache-record-ids-only",
            action="store_true",
            help="For cotracker_cache, write cache point IDs/frame references instead of full tracks.",
        )

    def add_tracker_args(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument(
            "--backend",
            choices=("lk", "cotracker", "cotracker_cache", "foundationpose", "bootstap"),
            default="lk",
        )
        add_cotracker_args(cmd)
        add_cache_backend_args(cmd)
        cmd.add_argument("--foundationpose-device", default="auto")
        cmd.add_argument(
            "--foundationpose-weights-dir",
            type=Path,
            default=None,
            help="FoundationPose weights root. Defaults to checkpoints/foundationpose.",
        )
        cmd.add_argument(
            "--foundationpose-download-weights",
            action="store_true",
            help="Download missing FoundationPose refiner/scorer weights before tracking.",
        )
        cmd.add_argument(
            "--foundationpose-transforms",
            type=Path,
            default=None,
            help=(
                "FoundationPose-derived .npz/.json containing homographies_reverse "
                "or direct tracks_reverse for this AVT window."
            ),
        )
        cmd.add_argument(
            "--bootstap-config",
            type=Path,
            default=None,
            help="Optional YAML config for the BootsTAPIR backend.",
        )
        cmd.add_argument("--bootstap-device", default=None)
        cmd.add_argument(
            "--bootstap-checkpoint",
            type=Path,
            default=None,
            help="Path to bootstapir_checkpoint_v2.pt.",
        )
        cmd.add_argument("--bootstap-checkpoint-url", default=None)
        cmd.add_argument(
            "--bootstap-download-checkpoint",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Download the BootsTAPIR checkpoint if it is missing.",
        )
        cmd.add_argument("--bootstap-resize-height", type=int, default=None)
        cmd.add_argument("--bootstap-resize-width", type=int, default=None)
        cmd.add_argument("--bootstap-query-chunk-size", type=int, default=None)
        cmd.add_argument("--bootstap-pyramid-level", type=int, default=None)
        cmd.add_argument("--bootstap-visibility-threshold", type=float, default=None)
        cmd.add_argument(
            "--bootstap-strict-checkpoint",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Use strict checkpoint loading for the BootsTAPIR model.",
        )

    cache = sub.add_parser("cache", help="Precompute a reusable CoTracker all-points cache.")
    _add_source_args(cache)
    cache.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help=f"Base directory for unique CoTracker caches. Default: {DEFAULT_CACHE_ROOT}",
    )
    cache.add_argument("--cache-frame-start", type=int, default=0)
    cache.add_argument("--cache-frame-count", type=int, default=None)
    cache.add_argument("--cache-grid-stride", type=int, default=8)
    cache.add_argument(
        "--cache-region",
        choices=("full", "bottom-third", "bottom-half"),
        default="full",
    )
    cache.add_argument(
        "--cache-query-mode",
        choices=("last-frame", "every-frame"),
        default="last-frame",
        help="Seed grid points once at reverse frame 0 or at every reverse frame.",
    )
    cache.add_argument("--cache-query-frame-stride", type=int, default=1)
    cache.add_argument(
        "--cache-max-query-points",
        type=int,
        default=0,
        help="Optional cap for smoke tests. 0 means no cap.",
    )
    add_cotracker_args(cache)
    cache.set_defaults(func=cmd_cache)

    cache_chunks = sub.add_parser(
        "cache-chunks",
        help="Precompute overlapping independent CoTracker cache chunks.",
    )
    _add_source_args(cache_chunks)
    cache_chunks.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help=f"Base directory for unique chunked cache runs. Default: {DEFAULT_CACHE_ROOT}",
    )
    cache_chunks.add_argument("--cache-chunk-size", type=int, default=DEFAULT_CACHE_CHUNK_SIZE)
    cache_chunks.add_argument(
        "--cache-window-size",
        type=int,
        default=DEFAULT_CACHE_WINDOW_SIZE,
        help="Extraction window size to guarantee full chunk coverage. Default: 80.",
    )
    cache_chunks.add_argument(
        "--cache-chunk-step",
        type=int,
        default=None,
        help="Frame step between chunks. Defaults to chunk_size - window_size.",
    )
    cache_chunks.add_argument("--cache-grid-stride", type=int, default=8)
    cache_chunks.add_argument(
        "--cache-region",
        choices=("full", "bottom-third", "bottom-half"),
        default="full",
    )
    cache_chunks.add_argument(
        "--cache-query-mode",
        choices=("last-frame", "every-frame"),
        default="last-frame",
    )
    cache_chunks.add_argument("--cache-query-frame-stride", type=int, default=1)
    cache_chunks.add_argument(
        "--cache-max-query-points",
        type=int,
        default=0,
        help="Optional cap for smoke tests. 0 means no cap.",
    )
    cache_chunks.add_argument(
        "--cache-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse complete chunk directories when rerunning in an existing root.",
    )
    add_cotracker_args(cache_chunks)
    cache_chunks.set_defaults(func=cmd_cache_chunks)

    track = sub.add_parser("track", help="Run inverse-video point tracking.")
    _add_source_args(track)
    _add_inverse_args(track)
    track.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Base directory for unique run outputs. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    add_tracker_args(track)
    track.set_defaults(func=cmd_track)

    viewer = sub.add_parser("viewer", help="Build the static WebUI from AVT artifacts.")
    _add_source_args(viewer)
    viewer.add_argument("--tracking-root", type=Path, required=True)
    viewer.add_argument(
        "--viewer-dir",
        type=Path,
        default=DEFAULT_VIEWER_ROOT,
        help=f"Base directory for unique viewer outputs. Default: {DEFAULT_VIEWER_ROOT}",
    )
    viewer.add_argument("--max-points-per-frame", type=int, default=0)
    viewer.add_argument("--copy-frames", action="store_true")
    viewer.add_argument("--video-base-url", default="")
    viewer.set_defaults(func=cmd_viewer)

    all_cmd = sub.add_parser("all", help="Run tracking and build the WebUI.")
    _add_source_args(all_cmd)
    _add_inverse_args(all_cmd)
    all_cmd.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Base directory for unique run outputs. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    all_cmd.add_argument(
        "--viewer-dir",
        type=Path,
        default=None,
        help="Optional base directory for unique viewer outputs.",
    )
    add_tracker_args(all_cmd)
    all_cmd.add_argument(
        "--build-viewer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build the static WebUI after tracking. Disabled by default to keep runs lightweight.",
    )
    all_cmd.add_argument("--max-points-per-frame", type=int, default=0)
    all_cmd.add_argument("--copy-frames", action="store_true")
    all_cmd.add_argument("--video-base-url", default="")
    all_cmd.set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
