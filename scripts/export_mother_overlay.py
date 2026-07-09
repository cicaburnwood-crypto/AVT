from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _round_point(point: np.ndarray) -> list[float]:
    return [round(float(point[0]), 1), round(float(point[1]), 1)]


def _round_matrix(matrix: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in matrix.reshape(-1)]


def _affine_scale(matrix: np.ndarray) -> float | None:
    linear = matrix[:, :2].astype(np.float64)
    det = float(np.linalg.det(linear))
    if not np.isfinite(det):
        return None
    return round(float(np.sqrt(abs(det))), 6)


def _to_homogeneous(matrix: np.ndarray) -> np.ndarray:
    out = np.eye(3, dtype=np.float64)
    out[:2] = matrix.astype(np.float64)
    return out


def _project(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogeneous = np.array([float(point[0]), float(point[1]), 1.0], dtype=np.float64)
    return (matrix @ homogeneous)[:2]


def _rolling_mother_frames(
    *,
    seq_end: int,
    width: int,
    height: int,
    mother: np.ndarray,
    transforms: np.ndarray,
    valid: np.ndarray,
    radius_px: float,
) -> tuple[list[dict[str, object]], int]:
    frame_count = int(transforms.shape[0])
    radius = max(1.0, float(radius_px))
    center = mother.astype(np.float64)
    left = center + np.array([-radius, 0.0], dtype=np.float64)
    right = center + np.array([radius, 0.0], dtype=np.float64)
    left[0] = np.clip(left[0], 0.0, max(0, width - 1))
    right[0] = np.clip(right[0], 0.0, max(0, width - 1))
    packet_width = float(np.linalg.norm(right - left)) or radius * 2.0

    cumulative: list[np.ndarray | None] = []
    inverses: list[np.ndarray | None] = []
    for t, matrix in enumerate(transforms.astype(np.float32)):
        if not bool(valid[t]) or not np.isfinite(matrix).all():
            cumulative.append(None)
            inverses.append(None)
            continue
        homogeneous = _to_homogeneous(matrix)
        try:
            inverse = np.linalg.inv(homogeneous)
        except np.linalg.LinAlgError:
            cumulative.append(None)
            inverses.append(None)
            continue
        cumulative.append(homogeneous)
        inverses.append(inverse)

    frames: list[dict[str, object]] = []
    point_count = 0
    for current_t in range(frame_count):
        current = cumulative[current_t]
        if current is None:
            continue
        frame = seq_end - 1 - current_t
        points: list[list[float | int | None]] = []
        for source_t in range(current_t + 1):
            source_inv = inverses[source_t]
            if source_inv is None:
                continue
            source_to_current = current @ source_inv
            xy = _project(source_to_current, center)
            if not np.isfinite(xy).all():
                continue
            x, y = float(xy[0]), float(xy[1])
            if x < 0.0 or x > width - 1 or y < 0.0 or y > height - 1:
                continue
            left_xy = _project(source_to_current, left)
            right_xy = _project(source_to_current, right)
            scale = float(np.linalg.norm(right_xy - left_xy) / packet_width)
            if not np.isfinite(scale):
                continue
            source_frame = seq_end - 1 - source_t
            points.append([int(source_frame), round(x, 1), round(y, 1), round(scale, 4)])
        points.sort(key=lambda item: int(item[0]))
        point_count += len(points)
        frames.append({"frame": int(frame), "reverse_time": int(current_t), "points": points})
    frames.sort(key=lambda item: int(item["frame"]))
    return frames, point_count


def _window_sort_key(path: Path) -> tuple[int, str]:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        return int(meta.get("seq_start", 0)), path.parent.name
    except Exception:
        return 0, path.parent.name


def export_run(run_dir: Path) -> dict[str, object]:
    windows_root = run_dir / "windows"
    out_path = run_dir / "viewer" / "data" / "mother_overlay.json"
    windows = []
    total_points = 0
    for meta_path in sorted(windows_root.glob("*/window.json"), key=_window_sort_key):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tracks_name = str(meta.get("files", {}).get("tracks") or "tracks.npz")
        arrays_path = meta_path.parent / tracks_name
        if not arrays_path.exists():
            continue
        arrays = np.load(arrays_path)
        if "anchor_motion_projected_mother_reverse" not in arrays:
            continue
        transforms = arrays.get("anchor_motion_transforms_reverse")
        valid = arrays.get("anchor_motion_valid_reverse")
        mother = arrays.get("anchor_motion_mother_point")
        if transforms is None or mother is None:
            continue
        if valid is None:
            valid = np.isfinite(transforms).all(axis=(1, 2))
        else:
            valid = valid.astype(bool) & np.isfinite(transforms).all(axis=(1, 2))

        seq_start = int(meta["seq_start"])
        seq_end = int(meta["seq_end"])
        radius_arr = arrays.get("anchor_motion_mother_scale_radius_px")
        radius_px = float(radius_arr.item()) if radius_arr is not None and radius_arr.shape == () else 16.0
        frames, count = _rolling_mother_frames(
            seq_end=seq_end,
            width=int(meta.get("width", 0)),
            height=int(meta.get("height", 0)),
            mother=mother,
            transforms=transforms,
            valid=valid,
            radius_px=radius_px,
        )
        total_points += count

        windows.append(
            {
                "id": str(meta["id"]),
                "seq_start": seq_start,
                "seq_end": seq_end,
                "frame_count": int(meta.get("frame_count", seq_end - seq_start)),
                "width": int(meta.get("width", 0)),
                "height": int(meta.get("height", 0)),
                "mother_point": _round_point(mother) if mother is not None else None,
                "scale_radius_px": radius_px,
                "frames": frames,
            }
        )

    payload = {
        "schema": "avt_rolling_mother_projection_overlay_v1",
        "run": run_dir.name,
        "time_order": {
            "points": "[source_frame, x, y, relative_scale] in each current frame",
            "reverse_time": "current reverse time; source reverse time is derived from source_frame",
        },
        "window_count": len(windows),
        "point_count": total_points,
        "windows": windows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return {"run_dir": str(run_dir), "output": str(out_path), "windows": len(windows), "points": total_points}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export anchor-motion mother points for the local multi-viewer.")
    parser.add_argument("viewer_root", type=Path, help="Directory containing viewer_manifest.json.")
    args = parser.parse_args()

    viewer_root = args.viewer_root.resolve()
    manifest = json.loads((viewer_root / "viewer_manifest.json").read_text(encoding="utf-8"))
    seen: set[Path] = set()
    summaries = []
    for entry in manifest.get("entries", []):
        run_dir = (viewer_root / entry["runDir"]).resolve()
        if run_dir in seen:
            continue
        seen.add(run_dir)
        summaries.append(export_run(run_dir))

    print(json.dumps({"runs": summaries}, indent=2))


if __name__ == "__main__":
    main()
