from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from .schema import FrameRecord

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_frame_records(root: Path, source_type: str = "auto") -> list[FrameRecord]:
    root = root.resolve()
    if source_type == "auto":
        if (root / "frames.jsonl").exists():
            source_type = "earth_rover"
        elif (root / "color.txt").exists():
            source_type = "color_txt"
        else:
            source_type = "image_dir"

    if source_type == "earth_rover":
        return _read_earth_rover_frames(root / "frames.jsonl")
    if source_type == "color_txt":
        return _read_color_txt(root / "color.txt")
    if source_type == "image_dir":
        return _read_image_dir(root)
    raise ValueError(f"Unknown source_type: {source_type}")


def _read_earth_rover_frames(path: Path) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            front = row.get("files", {}).get("front", {})
            rel = front.get("path")
            if not rel:
                continue
            timestamp = row.get("sdk_timestamp")
            elapsed = row.get("elapsed_seconds")
            records.append(
                FrameRecord(
                    index=len(records),
                    path=str(rel),
                    timestamp=float(timestamp) if timestamp is not None else None,
                    rel_time_sec=float(elapsed) if elapsed is not None else None,
                )
            )
    return _normalize_rel_time(records)


def _read_color_txt(path: Path) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        timestamp, rel = line.split()[:2]
        records.append(
            FrameRecord(index=len(records), path=rel, timestamp=float(timestamp))
        )
    return _normalize_rel_time(records)


def _read_image_dir(root: Path) -> list[FrameRecord]:
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    return [
        FrameRecord(index=i, path=str(path.relative_to(root)), rel_time_sec=float(i))
        for i, path in enumerate(paths)
    ]


def _normalize_rel_time(records: list[FrameRecord]) -> list[FrameRecord]:
    if not records:
        raise FileNotFoundError("No RGB frames were found")
    values = [
        rec.rel_time_sec if rec.rel_time_sec is not None else rec.timestamp
        for rec in records
    ]
    origin = values[0] if values[0] is not None else 0.0
    return [
        FrameRecord(
            index=i,
            path=rec.path,
            timestamp=rec.timestamp,
            rel_time_sec=float((values[i] if values[i] is not None else i) - origin),
        )
        for i, rec in enumerate(records)
    ]


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_frame_window(root: Path, records: list[FrameRecord], start: int, end: int) -> np.ndarray:
    frames = [read_rgb(records[i].resolved_path(root)) for i in range(start, end)]
    if not frames:
        raise ValueError(f"Empty frame window {start}:{end}")
    return np.stack(frames, axis=0)


def write_mp4(path: Path, frames_rgb: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_rgb.shape[1:3]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(w), int(h)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    try:
        for frame in frames_rgb:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def link_or_copy(src: Path, dst: Path, copy: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())
