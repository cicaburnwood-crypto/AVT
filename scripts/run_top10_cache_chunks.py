#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CHUNKED_CACHE_SCHEMA = "avt_cotracker_chunked_cache_v1"
DEFAULT_CACHE_CONFIG = REPO_ROOT / "configs" / "cotracker_cache.yaml"


@dataclass(frozen=True)
class TopRide:
    rank: int
    dataset: str
    asset: str
    front_frames: int
    rear_frames: int


TOP_RIDES = [
    TopRide(1, "output_rides_0", "ride_16933_20240124133319", 71901, 71898),
    TopRide(2, "output_rides_0", "ride_16987_20240125031213", 71881, 71884),
    TopRide(3, "output_rides_0", "ride_17652_20240130115819", 55855, 55833),
    TopRide(4, "output_rides_2", "ride_20010_20240305062350", 54598, 54605),
    TopRide(5, "output_rides_2", "ride_20333_20240307083945", 53642, 53616),
    TopRide(6, "output_rides_0", "ride_17198_20240126014721", 51806, 51808),
    TopRide(7, "output_rides_0", "ride_16557_20240117023647", 49705, 49714),
    TopRide(8, "output_rides_2", "ride_20501_20240308151308", 45434, 45431),
    TopRide(9, "output_rides_0", "ride_17620_20240130041859", 45262, 45239),
    TopRide(10, "output_rides_0", "ride_17686_20240131064048", 43590, 43591),
]


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _camera_uid(camera: str) -> int:
    if camera == "front":
        return 1000
    if camera == "rear":
        return 1001
    raise ValueError(f"Unsupported camera: {camera}")


def _expected_frames(ride: TopRide, camera: str) -> int:
    return ride.front_frames if camera == "front" else ride.rear_frames


def _ride_dir(data_root: Path, ride: TopRide) -> Path:
    candidates = [
        data_root / ride.dataset / ride.dataset / ride.asset,
        data_root / ride.dataset / ride.asset,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {ride.asset}; checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _playlist_for(ride_dir: Path, camera: str) -> Path:
    uid = _camera_uid(camera)
    recordings = ride_dir / "recordings"
    patterns = [
        f"*uid_s_{uid}*uid_e_video.m3u8",
        f"*uid_s_{uid}*video.m3u8",
    ]
    for pattern in patterns:
        matches = sorted(recordings.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No {camera} uid_s_{uid} video playlist found in {recordings}")


def _frame_number(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("frame_"):
        return None
    try:
        return int(stem.removeprefix("frame_"))
    except ValueError:
        return None


def _contiguous_frame_count(frames_dir: Path) -> int:
    if not frames_dir.exists():
        return 0
    indices = sorted(
        idx
        for idx in (_frame_number(path) for path in frames_dir.glob("frame_*.jpg"))
        if idx is not None
    )
    expected = 0
    for idx in indices:
        if idx < expected:
            continue
        if idx != expected:
            break
        expected += 1
    return expected


def _load_prepare_metadata(frames_dir: Path) -> dict[str, Any] | None:
    path = frames_dir / "prepare_metadata.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_frames(
    *,
    playlist: Path,
    frames_dir: Path,
    expected_frames: int,
    force: bool,
    max_frames: int | None,
) -> int:
    import cv2

    if force and frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_prepare_metadata(frames_dir)
    existing = _contiguous_frame_count(frames_dir)
    target = max_frames if max_frames is not None else None
    if metadata and metadata.get("complete") and existing >= int(metadata.get("frame_count", 0)):
        if target is None or existing >= target:
            print(f"prepared frames ready: {frames_dir} ({existing} frames)", flush=True)
            return existing

    if target is not None and existing >= target:
        _write_json(
            frames_dir / "prepare_metadata.json",
            {
                "complete": True,
                "limited": True,
                "frame_count": existing,
                "expected_frames": expected_frames,
                "playlist": str(playlist),
            },
        )
        print(f"prepared frames ready: {frames_dir} ({existing} limited frames)", flush=True)
        return existing

    cap = cv2.VideoCapture(str(playlist))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open playlist: {playlist}")
    if existing:
        cap.set(cv2.CAP_PROP_POS_FRAMES, existing)

    start_time = time.monotonic()
    written = existing
    try:
        while True:
            if target is not None and written >= target:
                break
            ok, frame = cap.read()
            if not ok:
                break
            out_path = frames_dir / f"frame_{written:06d}.jpg"
            if not cv2.imwrite(str(out_path), frame):
                raise RuntimeError(f"Could not write frame: {out_path}")
            written += 1
            if written % 1000 == 0:
                elapsed = time.monotonic() - start_time
                rate = (written - existing) / max(elapsed, 1e-6)
                remaining_count = (
                    max(0, target - written)
                    if target is not None
                    else max(0, expected_frames - written)
                )
                eta = remaining_count / rate if rate > 0 else 0.0
                print(
                    f"prepared {written} frames for {frames_dir.name} "
                    f"rate={rate:.1f}/s eta={_duration(eta)}",
                    flush=True,
                )
    finally:
        cap.release()

    complete = target is None or written >= target
    _write_json(
        frames_dir / "prepare_metadata.json",
        {
            "complete": complete,
            "limited": target is not None,
            "frame_count": written,
            "expected_frames": expected_frames,
            "playlist": str(playlist),
        },
    )
    print(f"prepared frames done: {frames_dir} ({written} frames)", flush=True)
    return written


def _manifest_ready(cache_dir: Path, *, cache_config) -> bool:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        manifest.get("schema") == CHUNKED_CACHE_SCHEMA
        and int(manifest.get("chunk_size", -1)) == int(cache_config.chunk_size)
        and int(manifest.get("window_size", -1)) == int(cache_config.window_size)
        and int(manifest.get("chunk_step", -1)) == int(cache_config.resolved_chunk_step)
        and manifest.get("cache_config") == asdict(cache_config.cache)
        and bool(manifest.get("chunks"))
    )


def build_cache(
    *,
    frames_dir: Path,
    cache_dir: Path,
    args: argparse.Namespace,
    cache_config,
) -> dict[str, Any]:
    from avt.io import read_frame_records
    from avt.tracking.cotracker_cache import build_cotracker_cache_chunks

    if args.force_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if _manifest_ready(cache_dir, cache_config=cache_config):
        manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
        print(f"chunk cache ready: {cache_dir} ({len(manifest['chunks'])} chunks)", flush=True)
        return manifest

    records = read_frame_records(frames_dir, "image_dir")
    return build_cotracker_cache_chunks(
        source_root=frames_dir,
        frame_records=records,
        output_dir=cache_dir,
        resume=True,
        config=cache_config,
    )


def _tasks(camera: str) -> list[tuple[TopRide, str]]:
    cameras = ("front", "rear") if camera == "both" else (camera,)
    return [(ride, selected_camera) for ride in TOP_RIDES for selected_camera in cameras]


def run(args: argparse.Namespace) -> int:
    from avt.tracking.cotracker_cache import load_cotracker_cache_config_yaml

    data_root = args.data_root.resolve()
    prepared_root = args.prepared_root.resolve()
    output_root = args.output_root.resolve()
    cache_config_path = args.cache_config.resolve()
    cache_config = load_cotracker_cache_config_yaml(cache_config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    args_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    args_payload["cache_config_values"] = asdict(cache_config)
    selected = _tasks(args.camera)
    if args.limit_videos is not None:
        selected = selected[: args.limit_videos]

    status_path = output_root / "top10_cache_status.json"
    run_started = time.monotonic()
    completed: list[dict[str, Any]] = []
    print(
        f"starting {len(selected)} cache task(s), chunk={cache_config.chunk_size}, "
        f"window={cache_config.window_size}, region={cache_config.cache.region}, "
        f"bad_track_confidence_threshold={cache_config.cache.abort_confidence_threshold}",
        flush=True,
    )

    for ordinal, (ride, camera) in enumerate(selected, start=1):
        task_started = time.monotonic()
        ride_dir = _ride_dir(data_root, ride)
        playlist = _playlist_for(ride_dir, camera)
        uid = _camera_uid(camera)
        stem = f"{ride.rank:02d}_{ride.asset}_{camera}_uid{uid}"
        frames_dir = prepared_root / f"{stem}_frames"
        cache_dir = output_root / f"{stem}_chunks{cache_config.chunk_size}_w{cache_config.window_size}"

        print(f"[{ordinal}/{len(selected)}] {ride.asset} {camera}: preparing frames", flush=True)
        frame_count = prepare_frames(
            playlist=playlist,
            frames_dir=frames_dir,
            expected_frames=_expected_frames(ride, camera),
            force=args.force_prepare,
            max_frames=args.max_frames,
        )

        print(f"[{ordinal}/{len(selected)}] {ride.asset} {camera}: building cache", flush=True)
        manifest = build_cache(
            frames_dir=frames_dir,
            cache_dir=cache_dir,
            args=args,
            cache_config=cache_config,
        )
        task_elapsed = time.monotonic() - task_started
        completed.append(
            {
                "ride": asdict(ride),
                "camera": camera,
                "frames_dir": str(frames_dir),
                "frame_count": frame_count,
                "cache_dir": str(cache_dir),
                "manifest": str(cache_dir / "manifest.json"),
                "chunks": len(manifest["chunks"]),
                "elapsed_sec": task_elapsed,
            }
        )
        elapsed = time.monotonic() - run_started
        average = elapsed / ordinal
        eta = average * (len(selected) - ordinal)
        _write_json(
            status_path,
            {
                "complete": ordinal == len(selected),
                "completed_tasks": ordinal,
                "total_tasks": len(selected),
                "elapsed_sec": elapsed,
                "eta_sec": eta,
                "args": args_payload,
                "completed": completed,
            },
        )
        print(
            f"[{ordinal}/{len(selected)}] done {ride.asset} {camera} "
            f"elapsed={_duration(task_elapsed)} run_eta={_duration(eta)}",
            flush=True,
        )

    print(f"all cache tasks complete; status={status_path}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and cache the AVT top-10 ride videos.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--prepared-root", type=Path, default=Path("data/prepared_top10_cache"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/top10_cotracker_cache_chunks"))
    parser.add_argument("--camera", choices=("front", "rear", "both"), default="front")
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Debug limit for each prepared video.")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument(
        "--cache-config",
        type=Path,
        default=DEFAULT_CACHE_CONFIG,
        help=f"YAML file for CoTracker cache settings. Default: {DEFAULT_CACHE_CONFIG}",
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
