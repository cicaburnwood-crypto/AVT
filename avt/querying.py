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
    "sift_anchor": 2,
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
    """VENTURA-style bottom-center robot footprint in normalized image units."""

    width_ratio: float = 0.20
    height_ratio: float = 0.15

    def derived_footprint_ratios(self) -> tuple[float, float]:
        return _clamp(self.width_ratio, 0.03, 0.80), _clamp(self.height_ratio, 0.03, 0.60)

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
class SiftAnchorConfig:
    """VENTURA full-frame SIFT anchors used to stabilize tracking."""

    enabled: bool = True
    max_query_points: int = 384
    window_size: int | None = None
    n_octave_layers: int = 3
    contrast_threshold: float = 0.008
    edge_threshold: float = 15.0
    sigma: float = 1.2


@dataclass
class SiftCaptureConfig:
    """VENTURA SIFT query capture controls."""

    enabled: bool = True
    max_query_points: int = 384
    window_size: int = 20
    sample_at_edges: bool = True
    edge_offset_ratio: float = 0.10
    n_octave_layers: int = 7
    contrast_threshold: float = 0.02
    edge_threshold: float = 18.0
    sigma: float = 1.6
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    anchors: SiftAnchorConfig = field(default_factory=SiftAnchorConfig)

    @property
    def temporal_stride(self) -> int:
        """Compatibility alias for older AVT configs; VENTURA calls this window_size."""

        return self.window_size


@dataclass
class QueryConfig:
    """Top-level query source configuration."""

    mode: str = "ventura"
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
    mode = str(data.get("query_mode", data.get("mode", "ventura")))
    robot_data = (
        data.get("footprint")
        or data.get("ventura_footprint")
        or data.get("virtual_robot")
        or data.get("robot")
        or {}
    )
    sift_data = data.get("sift", data.get("sift_capture", {})) or {}
    if not isinstance(robot_data, dict):
        raise ValueError("footprint must be a mapping")
    if not isinstance(sift_data, dict):
        raise ValueError("sift must be a mapping")
    anchor_data = sift_data.get("anchors", sift_data.get("anchor", {})) or {}
    if not isinstance(anchor_data, dict):
        raise ValueError("sift.anchors must be a mapping")

    robot = VirtualRobotConfig(
        width_ratio=_ratio_value(
            robot_data,
            ("width_ratio", "robot_width_pct", "footprint_width_ratio"),
            0.20,
        ),
        height_ratio=_ratio_value(
            robot_data,
            ("height_ratio", "robot_height_pct", "footprint_height_ratio", "footprint_length_ratio"),
            0.15,
        ),
    )
    anchors = SiftAnchorConfig(
        enabled=_as_bool(anchor_data.get("enabled", True)),
        max_query_points=int(anchor_data.get("max_query_points", 384)),
        window_size=(
            int(anchor_data["window_size"])
            if anchor_data.get("window_size") is not None
            else None
        ),
        n_octave_layers=int(anchor_data.get("n_octave_layers", 3)),
        contrast_threshold=float(anchor_data.get("contrast_threshold", 0.008)),
        edge_threshold=float(anchor_data.get("edge_threshold", 15.0)),
        sigma=float(anchor_data.get("sigma", 1.2)),
    )
    sift = SiftCaptureConfig(
        enabled=_as_bool(sift_data.get("enabled", mode in {"ventura", "sift", "avt+sift"})),
        max_query_points=int(sift_data.get("max_query_points", 384)),
        window_size=int(sift_data.get("window_size", sift_data.get("temporal_stride", 20))),
        sample_at_edges=_as_bool(sift_data.get("sample_at_edges", True)),
        edge_offset_ratio=float(sift_data.get("edge_offset_ratio", 0.10)),
        n_octave_layers=int(sift_data.get("n_octave_layers", 7)),
        contrast_threshold=float(sift_data.get("contrast_threshold", 0.02)),
        edge_threshold=float(sift_data.get("edge_threshold", 18.0)),
        sigma=float(sift_data.get("sigma", 1.6)),
        use_clahe=_as_bool(sift_data.get("use_clahe", True)),
        clahe_clip_limit=float(sift_data.get("clahe_clip_limit", 2.0)),
        clahe_tile_grid_size=int(sift_data.get("clahe_tile_grid_size", 8)),
        anchors=anchors,
    )
    mode = _validate_mode(mode)
    if mode in {"ventura", "sift", "avt+sift"} and not sift.enabled:
        sift = replace(sift, enabled=True)
    if sift.enabled and mode == "avt":
        mode = "ventura"
    return QueryConfig(mode=mode, robot=robot, sift=sift)


def merge_query_config(
    base: QueryConfig,
    *,
    mode: str | None = None,
    enable_sift: bool | None = None,
) -> QueryConfig:
    next_mode = _validate_mode(mode) if mode else base.mode
    next_sift = base.sift
    if mode in {"ventura", "sift", "avt+sift"} and not next_sift.enabled:
        next_sift = replace(next_sift, enabled=True)
    if enable_sift is not None:
        next_sift = replace(next_sift, enabled=enable_sift)
        if enable_sift and next_mode == "avt":
            next_mode = "ventura"
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
    if query_config.sift.window_size <= 0:
        raise ValueError("sift window_size must be positive")

    frame_count, height, width = frames_rgb.shape[:3]
    mask = robot_sift_mask(height, width, query_config.robot, query_config.sift)
    times = _sift_times(frame_count, query_config.sift.window_size)
    return _sample_sift_queries(
        frames_rgb=frames_rgb,
        times=times,
        max_query_points=query_config.sift.max_query_points,
        mask=mask,
        sift_config=query_config.sift,
        params=query_config.sift,
        source="sift_robot",
        start_id=start_id,
        balance_full_mask=False,
    )


def build_ventura_queries(
    frames_rgb: np.ndarray,
    query_config: QueryConfig,
    *,
    start_id: int = 0,
) -> list[QueryPoint]:
    """Build VENTURA-equivalent anchor + robot-footprint SIFT queries."""

    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError("frames_rgb must have shape [T,H,W,3]")
    if not query_config.sift.enabled:
        return []
    if query_config.sift.window_size <= 0:
        raise ValueError("sift window_size must be positive")

    frame_count, height, width = frames_rgb.shape[:3]
    queries: list[QueryPoint] = []
    anchors = query_config.sift.anchors
    if anchors.enabled:
        anchor_window = anchors.window_size or query_config.sift.window_size
        if anchor_window <= 0:
            raise ValueError("sift anchor window_size must be positive")
        queries.extend(
            _sample_sift_queries(
                frames_rgb=frames_rgb,
                times=_sift_times(frame_count, anchor_window),
                max_query_points=anchors.max_query_points,
                mask=None,
                sift_config=query_config.sift,
                params=anchors,
                source="sift_anchor",
                start_id=start_id + len(queries),
                balance_full_mask=True,
            )
        )
    queries.extend(
        _sample_sift_queries(
            frames_rgb=frames_rgb,
            times=_sift_times(frame_count, query_config.sift.window_size),
            max_query_points=query_config.sift.max_query_points,
            mask=robot_sift_mask(height, width, query_config.robot, query_config.sift),
            sift_config=query_config.sift,
            params=query_config.sift,
            source="sift_robot",
            start_id=start_id + len(queries),
            balance_full_mask=False,
        )
    )
    return queries


def _sample_sift_queries(
    *,
    frames_rgb: np.ndarray,
    times: list[int],
    max_query_points: int,
    mask: np.ndarray | None,
    sift_config: SiftCaptureConfig,
    params: SiftCaptureConfig | SiftAnchorConfig,
    source: str,
    start_id: int,
    balance_full_mask: bool,
) -> list[QueryPoint]:
    if max_query_points <= 0:
        return []
    if not times:
        return []
    _, height, width = frames_rgb.shape[:3]
    sift_detector = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=params.n_octave_layers,
        contrastThreshold=params.contrast_threshold,
        edgeThreshold=params.edge_threshold,
        sigma=params.sigma,
    )
    samples_per_time = max(1, int(max_query_points / len(times)))
    queries: list[QueryPoint] = []
    for reverse_time in times:
        gray = cv2.cvtColor(frames_rgb[reverse_time], cv2.COLOR_RGB2GRAY)
        gray = _local_equalize(gray, sift_config=sift_config)
        keypoints, _ = sift_detector.detectAndCompute(gray, mask)
        if not keypoints:
            continue
        picked = _pick_sift_keypoints(
            keypoints,
            width=width,
            count=samples_per_time,
            balance_halves=balance_full_mask,
        )
        for keypoint in picked:
            x, y = keypoint.pt
            side = -1 if x < width / 2.0 else 1
            queries.append(
                QueryPoint(
                    id=start_id + len(queries),
                    reverse_time=int(reverse_time),
                    x=float(x),
                    y=float(y),
                    side=side,
                    source=source,
                    response=float(keypoint.response),
                    size=float(keypoint.size),
                    angle=float(keypoint.angle),
                    octave=int(keypoint.octave),
                    class_id=int(keypoint.class_id),
                )
            )
    return queries


def _pick_sift_keypoints(
    keypoints: tuple[cv2.KeyPoint, ...] | list[cv2.KeyPoint],
    *,
    width: int,
    count: int,
    balance_halves: bool,
) -> list[cv2.KeyPoint]:
    ordered = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
    if not balance_halves:
        return ordered[:count]

    left = [kp for kp in ordered if kp.pt[0] < width / 2.0]
    right = [kp for kp in ordered if kp.pt[0] >= width / 2.0]
    n_left = count // 2
    n_right = count - n_left
    picked = left[:n_left] + right[:n_right]
    if len(picked) < count:
        picked_ids = {id(kp) for kp in picked}
        picked.extend([kp for kp in ordered if id(kp) not in picked_ids][: count - len(picked)])
    return picked


def _local_equalize(gray: np.ndarray, *, sift_config: SiftCaptureConfig) -> np.ndarray:
    if not sift_config.use_clahe:
        return gray
    tile = max(1, int(sift_config.clahe_tile_grid_size))
    clahe = cv2.createCLAHE(
        clipLimit=float(sift_config.clahe_clip_limit),
        tileGridSize=(tile, tile),
    )
    return clahe.apply(gray)


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
        edge_width = int(rect_width * sift.edge_offset_ratio)
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
    """Align VENTURA's normalized bottom robot footprint to an image."""

    if width <= 0 or height <= 0:
        raise ValueError("frame width and height must be positive")

    width_ratio, height_ratio = robot.derived_footprint_ratios()
    grid_width = width * width_ratio
    grid_height = height * height_ratio
    left = max(0, int((width // 2) - (grid_width // 2)))
    right = min(width, int((width // 2) + (grid_width // 2)))
    top = max(0, int(height - grid_height))
    bottom = height

    seed_x_min = _clamp(0.5 - width_ratio / 2.0, 0.0, 1.0)
    seed_x_max = _clamp(0.5 + width_ratio / 2.0, 0.0, 1.0)
    seed_y = _clamp(1.0 - height_ratio / 2.0, 0.0, 1.0)

    return RobotImageAlignment(
        frame_width=int(width),
        frame_height=int(height),
        method="ventura_pct_bottom_center",
        width_ratio=float(width_ratio),
        length_ratio=float(height_ratio),
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
    width_ratio, height_ratio = config.robot.derived_footprint_ratios()
    avt_seed_y, avt_seed_x_min, avt_seed_x_max = config.robot.avt_seed_ratios()
    alignment = (
        align_virtual_robot_to_image(height=height, width=width, robot=config.robot)
        if width is not None and height is not None
        else None
    )
    return {
        "schema": "avt_ventura_query_capture_v1",
        "numeric_columns": QUERY_NUMERIC_COLUMNS,
        "cotracker_columns": ["reverse_time", "x", "y"],
        "source_codes": QUERY_SOURCE_CODES,
        "mode": config.mode,
        "ventura_footprint": asdict(config.robot)
        | {
            "robot_width_pct": width_ratio,
            "robot_height_pct": height_ratio,
            "derived_avt_seed_y_ratio": avt_seed_y,
            "derived_avt_seed_x_min_ratio": avt_seed_x_min,
            "derived_avt_seed_x_max_ratio": avt_seed_x_max,
        },
        "image_alignment": alignment.to_json() if alignment else None,
        "sift": asdict(config.sift),
    }


def _sift_times(frame_count: int, window_size: int) -> list[int]:
    if frame_count <= 1:
        return [0]
    num_windows = max(1, frame_count // window_size)
    idxs = np.linspace(0, frame_count - 1, num=num_windows, dtype=int)
    idxs = np.concatenate(([0], idxs, [max(0, frame_count - 2)]))
    return sorted({int(idx) for idx in idxs if 0 <= idx < frame_count})


def _validate_mode(mode: str) -> str:
    if mode not in {"ventura", "avt", "sift", "avt+sift"}:
        raise ValueError("query_mode must be one of: ventura, avt, sift, avt+sift")
    return mode


def _ratio_value(data: dict[str, Any], keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        if data.get(key) is not None:
            return float(data[key])
    return float(default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _nan_if_none(value: float | int | None) -> float:
    return float("nan") if value is None else float(value)
