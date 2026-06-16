from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .schema import QueryPoint


QUERY_SOURCE_CODES = {
    "avt": 0,
    "sift_robot": 1,
}

QUERY_NUMERIC_COLUMNS = [
    "id",
    "reverse_time",
    "x",
    "y",
    "side",
    "source_code",
    "response",
    "size",
    "angle",
    "octave",
    "class_id",
]


@dataclass
class VirtualRobotConfig:
    """Approximate virtual robot footprint used to place AVT/SIFT query regions."""

    width_m: float = 0.40
    length_m: float = 0.60
    camera_height_m: float | None = 0.18
    footprint_width_ratio: float | None = None
    footprint_length_ratio: float | None = None

    def effective_camera_height_m(self) -> float:
        if self.camera_height_m is None:
            return 0.18
        return _positive(self.camera_height_m, "virtual robot camera_height_m")

    def derived_footprint_ratios(self) -> tuple[float, float]:
        width = _positive(self.width_m, "virtual robot width_m")
        length = _positive(self.length_m, "virtual robot length_m")
        camera_height = self.effective_camera_height_m()
        span = length + camera_height
        width_ratio = self.footprint_width_ratio
        length_ratio = self.footprint_length_ratio
        if width_ratio is None:
            width_ratio = width / (2.0 * span)
        if length_ratio is None:
            length_ratio = length / (4.0 * span)
        return _clamp(width_ratio, 0.03, 0.80), _clamp(length_ratio, 0.03, 0.60)

    def avt_seed_x_ratios(self) -> tuple[float, float]:
        width_ratio, _ = self.derived_footprint_ratios()
        return 0.5 - width_ratio / 2.0, 0.5 + width_ratio / 2.0

    def avt_seed_y_ratio(self) -> float:
        _, length_ratio = self.derived_footprint_ratios()
        return _clamp(1.0 - length_ratio / 2.0, 0.0, 1.0)

    def avt_seed_ratios(self) -> tuple[float, float, float]:
        x_min, x_max = self.avt_seed_x_ratios()
        return self.avt_seed_y_ratio(), x_min, x_max


@dataclass(frozen=True)
class RobotImageAlignment:
    """Resolution-aware, calibration-free robot footprint alignment."""

    frame_width: int
    frame_height: int
    method: str
    width_ratio: float
    length_ratio: float
    left: int
    right: int
    top: int
    bottom: int
    seed_y_ratio: float
    seed_x_min_ratio: float
    seed_x_max_ratio: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SiftCaptureConfig:
    """VENTURA-style SIFT query capture controls."""

    enabled: bool = False
    max_query_points: int = 384
    temporal_stride: int = 3
    sample_at_edges: bool = True
    edge_offset_ratio: float = 0.15
    n_octave_layers: int = 7
    contrast_threshold: float = 0.02
    edge_threshold: float = 18.0
    sigma: float = 1.6


@dataclass
class QueryConfig:
    """Top-level query source configuration."""

    mode: str = "avt"
    robot: VirtualRobotConfig = field(default_factory=VirtualRobotConfig)
    sift: SiftCaptureConfig = field(default_factory=SiftCaptureConfig)


def load_query_config_yaml(path: Path) -> QueryConfig:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("YAML robot configs require PyYAML.") from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Robot config must be a YAML mapping: {path}")
    return query_config_from_mapping(data)


def query_config_from_mapping(data: dict[str, Any]) -> QueryConfig:
    mode = str(data.get("query_mode", data.get("mode", "avt")))
    robot_data = data.get("virtual_robot", data.get("robot", {})) or {}
    sift_data = data.get("sift", data.get("sift_capture", {})) or {}
    if not isinstance(robot_data, dict):
        raise ValueError("virtual_robot must be a mapping")
    if not isinstance(sift_data, dict):
        raise ValueError("sift must be a mapping")

    robot = VirtualRobotConfig(
        width_m=_meters(robot_data, "width", 0.40),
        length_m=_meters(robot_data, "length", 0.60),
        camera_height_m=_optional_meters(robot_data, "camera_height", 0.18),
        footprint_width_ratio=_optional_float(robot_data, "footprint_width_ratio"),
        footprint_length_ratio=_optional_float(robot_data, "footprint_length_ratio"),
    )
    sift = SiftCaptureConfig(
        enabled=_as_bool(sift_data.get("enabled", mode in {"sift", "avt+sift"})),
        max_query_points=int(sift_data.get("max_query_points", 384)),
        temporal_stride=int(sift_data.get("temporal_stride", sift_data.get("window_size", 3))),
        sample_at_edges=_as_bool(sift_data.get("sample_at_edges", True)),
        edge_offset_ratio=float(sift_data.get("edge_offset_ratio", 0.15)),
        n_octave_layers=int(sift_data.get("n_octave_layers", 7)),
        contrast_threshold=float(sift_data.get("contrast_threshold", 0.02)),
        edge_threshold=float(sift_data.get("edge_threshold", 18.0)),
        sigma=float(sift_data.get("sigma", 1.6)),
    )
    mode = _validate_mode(mode)
    if mode in {"sift", "avt+sift"} and not sift.enabled:
        sift = replace(sift, enabled=True)
    if sift.enabled and mode == "avt":
        mode = "avt+sift"
    return QueryConfig(mode=mode, robot=robot, sift=sift)


def merge_query_config(
    base: QueryConfig,
    *,
    mode: str | None = None,
    enable_sift: bool | None = None,
) -> QueryConfig:
    next_mode = _validate_mode(mode) if mode else base.mode
    next_sift = base.sift
    if mode in {"sift", "avt+sift"} and not next_sift.enabled:
        next_sift = replace(next_sift, enabled=True)
    if enable_sift is not None:
        next_sift = replace(next_sift, enabled=enable_sift)
        if enable_sift and next_mode == "avt":
            next_mode = "avt+sift"
    return replace(base, mode=next_mode, sift=next_sift)


def build_avt_queries(
    width: int,
    height: int,
    frame_count: int,
    query_stride: int,
    seed_count: int,
    seed_y_ratio: float,
    seed_x_min_ratio: float,
    seed_x_max_ratio: float,
    *,
    start_id: int = 0,
    max_points: int | None = None,
) -> list[QueryPoint]:
    if query_stride <= 0:
        raise ValueError("query_stride must be positive")
    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    if max_points is not None and max_points <= 0:
        return []
    xs = np.linspace(
        seed_x_min_ratio * (width - 1),
        seed_x_max_ratio * (width - 1),
        seed_count,
        dtype=np.float32,
    )
    y = float(seed_y_ratio * (height - 1))
    queries: list[QueryPoint] = []
    middle = (seed_count - 1) / 2.0
    for reverse_time in range(0, frame_count, query_stride):
        for i, x in enumerate(xs):
            side = -1 if i <= middle else 1
            queries.append(
                QueryPoint(
                    id=start_id + len(queries),
                    reverse_time=int(reverse_time),
                    x=float(x),
                    y=y,
                    side=side,
                    source="avt",
                )
            )
            if max_points is not None and len(queries) >= max_points:
                return queries
    return queries


def build_sift_queries(
    frames_rgb: np.ndarray,
    query_config: QueryConfig,
    *,
    start_id: int = 0,
) -> list[QueryPoint]:
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError("frames_rgb must have shape [T,H,W,3]")
    if query_config.sift.max_query_points <= 0:
        raise ValueError("sift max_query_points must be positive")
    if query_config.sift.temporal_stride <= 0:
        raise ValueError("sift temporal_stride must be positive")

    frame_count, height, width = frames_rgb.shape[:3]
    mask = robot_sift_mask(height, width, query_config.robot, query_config.sift)
    sift = cv2.SIFT_create(
        nOctaveLayers=query_config.sift.n_octave_layers,
        contrastThreshold=query_config.sift.contrast_threshold,
        edgeThreshold=query_config.sift.edge_threshold,
        sigma=query_config.sift.sigma,
    )
    times = _sift_times(frame_count, query_config.sift.temporal_stride)
    samples_per_time = max(1, int(np.ceil(query_config.sift.max_query_points / len(times))))

    captured: list[tuple[float, QueryPoint]] = []
    for reverse_time in times:
        gray = cv2.cvtColor(frames_rgb[reverse_time], cv2.COLOR_RGB2GRAY)
        keypoints = sorted(sift.detect(gray, mask), key=lambda kp: kp.response, reverse=True)
        for keypoint in keypoints[:samples_per_time]:
            x, y = keypoint.pt
            side = -1 if x < width / 2.0 else 1
            captured.append(
                (
                    float(keypoint.response),
                    QueryPoint(
                        id=-1,
                        reverse_time=int(reverse_time),
                        x=float(x),
                        y=float(y),
                        side=side,
                        source="sift_robot",
                        response=float(keypoint.response),
                        size=float(keypoint.size),
                        angle=float(keypoint.angle),
                        octave=int(keypoint.octave),
                        class_id=int(keypoint.class_id),
                    ),
                )
            )

    captured.sort(key=lambda item: item[0], reverse=True)
    captured = captured[: query_config.sift.max_query_points]
    captured.sort(key=lambda item: (item[1].reverse_time, item[1].y, item[1].x))
    return [
        replace(query, id=start_id + idx)
        for idx, (_, query) in enumerate(captured)
    ]


def robot_sift_mask(
    height: int,
    width: int,
    robot: VirtualRobotConfig,
    sift: SiftCaptureConfig,
) -> np.ndarray:
    alignment = align_virtual_robot_to_image(height=height, width=width, robot=robot)
    left, right = alignment.left, alignment.right
    top, bottom = alignment.top, alignment.bottom

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:bottom, left:right] = 255

    if sift.sample_at_edges:
        rect_width = max(1, right - left)
        edge_width = max(1, int(round(rect_width * sift.edge_offset_ratio)))
        inner_left = min(right, left + edge_width)
        inner_right = max(left, right - edge_width)
        if inner_right > inner_left:
            mask[top:bottom, inner_left:inner_right] = 0
    return mask


def align_virtual_robot_to_image(
    *,
    height: int,
    width: int,
    robot: VirtualRobotConfig,
) -> RobotImageAlignment:
    """Align an approximate robot footprint to an image without camera intrinsics.

    This deliberately uses normalized image geometry. It is meant for internet
    videos where focal length, pitch, and camera height are usually unknown.
    """

    if width <= 0 or height <= 0:
        raise ValueError("frame width and height must be positive")

    width_ratio, length_ratio = robot.derived_footprint_ratios()
    frame_aspect = width / float(height)
    reference_aspect = 16.0 / 9.0

    if robot.footprint_width_ratio is None:
        aspect_scale = _clamp((frame_aspect / reference_aspect) ** 0.25, 0.85, 1.15)
        width_ratio = _clamp(width_ratio / aspect_scale, 0.03, 0.80)
    if robot.footprint_length_ratio is None:
        aspect_scale = _clamp((reference_aspect / frame_aspect) ** 0.10, 0.92, 1.08)
        length_ratio = _clamp(length_ratio * aspect_scale, 0.03, 0.60)

    rect_width = max(1, int(round(width * width_ratio)))
    rect_height = max(1, int(round(height * length_ratio)))
    left = max(0, width // 2 - rect_width // 2)
    right = min(width, left + rect_width)
    top = max(0, height - rect_height)
    bottom = height

    seed_x_min = _clamp(0.5 - width_ratio / 2.0, 0.0, 1.0)
    seed_x_max = _clamp(0.5 + width_ratio / 2.0, 0.0, 1.0)
    seed_y = _clamp(1.0 - length_ratio / 2.0, 0.0, 1.0)

    return RobotImageAlignment(
        frame_width=int(width),
        frame_height=int(height),
        method="image_normalized_no_intrinsics",
        width_ratio=float(width_ratio),
        length_ratio=float(length_ratio),
        left=int(left),
        right=int(right),
        top=int(top),
        bottom=int(bottom),
        seed_y_ratio=float(seed_y),
        seed_x_min_ratio=float(seed_x_min),
        seed_x_max_ratio=float(seed_x_max),
    )


def query_artifact_arrays(queries: list[QueryPoint]) -> dict[str, np.ndarray]:
    query_rows = []
    cotracker_rows = []
    sides = []
    source_codes = []
    for query in queries:
        source_code = QUERY_SOURCE_CODES.get(query.source, -1)
        row = [
            query.id,
            query.reverse_time,
            query.x,
            query.y,
            query.side,
            source_code,
            _nan_if_none(query.response),
            _nan_if_none(query.size),
            _nan_if_none(query.angle),
            _nan_if_none(query.octave),
            _nan_if_none(query.class_id),
        ]
        query_rows.append(row)
        cotracker_rows.append([query.reverse_time, query.x, query.y])
        sides.append(query.side)
        source_codes.append(source_code)
    return {
        "queries": np.array(query_rows, dtype=np.float32),
        "queries_cotracker": np.array(cotracker_rows, dtype=np.float32),
        "query_sides": np.array(sides, dtype=np.int8),
        "query_source_codes": np.array(source_codes, dtype=np.int16),
        "query_records_json": np.array(
            json.dumps([query.to_json() for query in queries]),
            dtype=np.str_,
        ),
    }


def query_capture_metadata(
    config: QueryConfig,
    *,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    width_ratio, length_ratio = config.robot.derived_footprint_ratios()
    avt_seed_y, avt_seed_x_min, avt_seed_x_max = config.robot.avt_seed_ratios()
    alignment = (
        align_virtual_robot_to_image(height=height, width=width, robot=config.robot)
        if width is not None and height is not None
        else None
    )
    return {
        "schema": "avt_query_capture_v2",
        "numeric_columns": QUERY_NUMERIC_COLUMNS,
        "cotracker_columns": ["reverse_time", "x", "y"],
        "source_codes": QUERY_SOURCE_CODES,
        "mode": config.mode,
        "virtual_robot": asdict(config.robot)
        | {
            "derived_footprint_width_ratio": width_ratio,
            "derived_footprint_length_ratio": length_ratio,
            "derived_avt_seed_y_ratio": avt_seed_y,
            "derived_avt_seed_x_min_ratio": avt_seed_x_min,
            "derived_avt_seed_x_max_ratio": avt_seed_x_max,
        },
        "image_alignment": alignment.to_json() if alignment else None,
        "sift": asdict(config.sift),
    }


def _sift_times(frame_count: int, temporal_stride: int) -> list[int]:
    if frame_count <= 1:
        return [0]
    num_windows = max(1, frame_count // temporal_stride)
    idxs = np.linspace(0, frame_count - 1, num=num_windows, dtype=int)
    idxs = np.concatenate(([0], idxs, [max(0, frame_count - 2)]))
    return sorted({int(idx) for idx in idxs if 0 <= idx < frame_count})


def _validate_mode(mode: str) -> str:
    if mode not in {"avt", "sift", "avt+sift"}:
        raise ValueError("query_mode must be one of: avt, sift, avt+sift")
    return mode


def _meters(data: dict[str, Any], base: str, default: float) -> float:
    if f"{base}_m" in data:
        return float(data[f"{base}_m"])
    if f"{base}_cm" in data:
        return float(data[f"{base}_cm"]) / 100.0
    return float(data.get(base, default))


def _optional_meters(data: dict[str, Any], base: str, default: float | None) -> float | None:
    if f"{base}_m" in data:
        value = data[f"{base}_m"]
    elif f"{base}_cm" in data:
        value = data[f"{base}_cm"]
        if value is None:
            return None
        return float(value) / 100.0
    else:
        value = data.get(base, default)
    if value is None:
        return None
    return float(value)


def _optional_float(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    return float(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _positive(value: float, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _nan_if_none(value: float | int | None) -> float:
    return float("nan") if value is None else float(value)
