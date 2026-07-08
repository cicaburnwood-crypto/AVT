from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detectors import build_detector
from .detectors.config import OrbDetectorConfig, SuperPointConfig, XFeatConfig
from .schema import QueryPoint


PRIMARY_QUERY_SOURCE_CODES = {
    "avt": 0,
    "footprint": 1,
    "anchor": 2,
}

QUERY_SOURCE_CODES = PRIMARY_QUERY_SOURCE_CODES

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
class FootprintConfig:
    """Bottom-center robot footprint in normalized image units."""

    width_ratio: float = 0.25
    height_ratio: float = 0.20

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
class AnchorSamplingConfig:
    """Full-frame anchor sampling controls used to stabilize tracking."""

    enabled: bool = True
    max_query_points: int = 384
    window_size: int | None = None
    min_points_per_frame: int = 32
    max_points_per_frame: int = 128
    n_octave_layers: int = 3
    contrast_threshold: float = 0.008
    edge_threshold: float = 15.0
    sigma: float = 1.2


@dataclass
class QuerySamplingConfig:
    """Detector-agnostic query sampling controls."""

    enabled: bool = True
    max_query_points: int = 384
    window_size: int = 20
    min_points_per_frame: int = 8
    max_points_per_frame: int = 20
    sample_at_edges: bool = True
    edge_offset_ratio: float = 0.10
    n_octave_layers: int = 5
    contrast_threshold: float = 0.018
    edge_threshold: float = 20.0
    sigma: float = 1.5
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    anchors: AnchorSamplingConfig = field(default_factory=AnchorSamplingConfig)

@dataclass(init=False)
class QueryConfig:
    """Top-level query source configuration."""

    mode: str = "anchor_motion"
    detector: str = "sift"
    footprint: FootprintConfig = field(default_factory=FootprintConfig)
    sampling: QuerySamplingConfig = field(default_factory=QuerySamplingConfig)
    orb: OrbDetectorConfig = field(default_factory=OrbDetectorConfig)
    superpoint: SuperPointConfig = field(default_factory=SuperPointConfig)
    xfeat: XFeatConfig = field(default_factory=XFeatConfig)

    def __init__(
        self,
        mode: str = "anchor_motion",
        detector: str = "sift",
        footprint: FootprintConfig | None = None,
        sampling: QuerySamplingConfig | None = None,
        orb: OrbDetectorConfig | None = None,
        superpoint: SuperPointConfig | None = None,
        xfeat: XFeatConfig | None = None,
    ) -> None:
        self.mode = _validate_mode(mode)
        self.detector = _validate_detector(detector)
        self.footprint = footprint if footprint is not None else FootprintConfig()
        self.sampling = sampling if sampling is not None else QuerySamplingConfig()
        self.orb = orb if orb is not None else OrbDetectorConfig()
        self.superpoint = superpoint if superpoint is not None else SuperPointConfig()
        self.xfeat = xfeat if xfeat is not None else XFeatConfig()


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


_DETECTORS = ("sift", "orb", "superpoint", "xfeat")
QUERY_MODES = ("anchor_motion", "anchor_footprint", "footprint", "avt", "avt+footprint")
DETECTOR_SAMPLING_MODES = ("anchor_motion", "anchor_footprint", "footprint", "avt+footprint")


def _validate_detector(name: str) -> str:
    if name not in _DETECTORS:
        raise ValueError(f"detector must be one of {_DETECTORS}, got {name!r}")
    return name


def _orb_config_from_mapping(data: dict[str, Any]) -> OrbDetectorConfig:
    if not isinstance(data, dict):
        raise ValueError("orb must be a mapping")
    base = OrbDetectorConfig()
    return OrbDetectorConfig(
        nfeatures=int(data.get("nfeatures", base.nfeatures)),
        scale_factor=float(data.get("scale_factor", base.scale_factor)),
        n_levels=int(data.get("n_levels", base.n_levels)),
        edge_threshold=int(data.get("edge_threshold", base.edge_threshold)),
        first_level=int(data.get("first_level", base.first_level)),
        wta_k=int(data.get("wta_k", base.wta_k)),
        score_type=str(data.get("score_type", base.score_type)),
        patch_size=int(data.get("patch_size", base.patch_size)),
        fast_threshold=int(data.get("fast_threshold", base.fast_threshold)),
        use_clahe=_as_bool(data.get("use_clahe", base.use_clahe)),
    )


def _superpoint_config_from_mapping(data: dict[str, Any]) -> SuperPointConfig:
    if not isinstance(data, dict):
        raise ValueError("superpoint must be a mapping")
    base = SuperPointConfig()
    return SuperPointConfig(
        model=str(data.get("model", base.model)),
        use_superglue=_as_bool(data.get("use_superglue", base.use_superglue)),
        superglue_model=str(data.get("superglue_model", base.superglue_model)),
        keypoint_threshold=float(data.get("keypoint_threshold", base.keypoint_threshold)),
        max_keypoints=int(data.get("max_keypoints", base.max_keypoints)),
        match_threshold=float(data.get("match_threshold", base.match_threshold)),
        neighbor_offset=int(data.get("neighbor_offset", base.neighbor_offset)),
        device=data.get("device", base.device),
    )


def _xfeat_config_from_mapping(data: dict[str, Any]) -> XFeatConfig:
    if not isinstance(data, dict):
        raise ValueError("xfeat must be a mapping")
    base = XFeatConfig()
    return XFeatConfig(
        hub_repo=str(data.get("hub_repo", base.hub_repo)),
        model=str(data.get("model", base.model)),
        top_k=int(data.get("top_k", base.top_k)),
        detection_threshold=float(data.get("detection_threshold", base.detection_threshold)),
        checkpoint=data.get("checkpoint", base.checkpoint),
        device=data.get("device", base.device),
    )


def query_config_from_mapping(data: dict[str, Any]) -> QueryConfig:
    mode = _validate_mode(str(data.get("query_mode", data.get("mode", "anchor_motion"))))
    robot_data = data.get("footprint", {}) or {}
    sampling_data = data.get("sampling", {}) or {}
    if not isinstance(robot_data, dict):
        raise ValueError("footprint must be a mapping")
    if not isinstance(sampling_data, dict):
        raise ValueError("sampling must be a mapping")
    anchor_data = sampling_data.get("anchors", sampling_data.get("anchor", {})) or {}
    if not isinstance(anchor_data, dict):
        raise ValueError("sampling.anchors must be a mapping")

    footprint = FootprintConfig(
        width_ratio=_ratio_value(
            robot_data,
            ("width_ratio",),
            0.25,
        ),
        height_ratio=_ratio_value(
            robot_data,
            ("height_ratio",),
            0.20,
        ),
    )
    anchors = AnchorSamplingConfig(
        enabled=_as_bool(anchor_data.get("enabled", True)),
        max_query_points=int(anchor_data.get("max_query_points", 384)),
        window_size=(
            int(anchor_data["window_size"])
            if anchor_data.get("window_size") is not None
            else None
        ),
        min_points_per_frame=int(anchor_data.get("min_points_per_frame", 8)),
        max_points_per_frame=int(anchor_data.get("max_points_per_frame", 20)),
        n_octave_layers=int(anchor_data.get("n_octave_layers", 3)),
        contrast_threshold=float(anchor_data.get("contrast_threshold", 0.008)),
        edge_threshold=float(anchor_data.get("edge_threshold", 15.0)),
        sigma=float(anchor_data.get("sigma", 1.2)),
    )
    sampling = QuerySamplingConfig(
        enabled=_as_bool(sampling_data.get("enabled", mode in DETECTOR_SAMPLING_MODES)),
        max_query_points=int(sampling_data.get("max_query_points", 384)),
        window_size=int(sampling_data.get("window_size", 20)),
        min_points_per_frame=int(sampling_data.get("min_points_per_frame", 8)),
        max_points_per_frame=int(sampling_data.get("max_points_per_frame", 20)),
        sample_at_edges=_as_bool(sampling_data.get("sample_at_edges", True)),
        edge_offset_ratio=float(sampling_data.get("edge_offset_ratio", 0.10)),
        n_octave_layers=int(sampling_data.get("n_octave_layers", 5)),
        contrast_threshold=float(sampling_data.get("contrast_threshold", 0.018)),
        edge_threshold=float(sampling_data.get("edge_threshold", 20.0)),
        sigma=float(sampling_data.get("sigma", 1.5)),
        use_clahe=_as_bool(sampling_data.get("use_clahe", True)),
        clahe_clip_limit=float(sampling_data.get("clahe_clip_limit", 2.0)),
        clahe_tile_grid_size=int(sampling_data.get("clahe_tile_grid_size", 8)),
        anchors=anchors,
    )
    if mode in DETECTOR_SAMPLING_MODES and not sampling.enabled:
        sampling = replace(sampling, enabled=True)
    if sampling.enabled and mode == "avt":
        mode = "anchor_motion"
    detector = _validate_detector(str(data.get("detector", "sift")))
    orb = _orb_config_from_mapping(data.get("orb", {}) or {})
    superpoint = _superpoint_config_from_mapping(data.get("superpoint", {}) or {})
    xfeat = _xfeat_config_from_mapping(data.get("xfeat", {}) or {})
    return QueryConfig(
        mode=mode,
        detector=detector,
        footprint=footprint,
        sampling=sampling,
        orb=orb,
        superpoint=superpoint,
        xfeat=xfeat,
    )


def merge_query_config(
    base: QueryConfig,
    *,
    mode: str | None = None,
    enable_sampling: bool | None = None,
    detector: str | None = None,
    orb: OrbDetectorConfig | None = None,
    superpoint: SuperPointConfig | None = None,
    xfeat: XFeatConfig | None = None,
) -> QueryConfig:
    next_mode = _validate_mode(mode) if mode else base.mode
    next_sampling = base.sampling
    if mode and next_mode in DETECTOR_SAMPLING_MODES and not next_sampling.enabled:
        next_sampling = replace(next_sampling, enabled=True)
    if enable_sampling is not None:
        next_sampling = replace(next_sampling, enabled=enable_sampling)
        if enable_sampling and next_mode == "avt":
            next_mode = "anchor_motion"
    next_detector = _validate_detector(detector) if detector else base.detector
    return replace(
        base,
        mode=next_mode,
        detector=next_detector,
        sampling=next_sampling,
        orb=orb if orb is not None else base.orb,
        superpoint=superpoint if superpoint is not None else base.superpoint,
        xfeat=xfeat if xfeat is not None else base.xfeat,
    )


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


def build_footprint_queries(
    frames_rgb: np.ndarray,
    query_config: QueryConfig,
    *,
    start_id: int = 0,
) -> list[QueryPoint]:
    raise ValueError("footprint query points are disabled in the anchor-motion repo")


def build_anchor_motion_queries(
    frames_rgb: np.ndarray,
    query_config: QueryConfig,
    *,
    start_id: int = 0,
) -> list[QueryPoint]:
    """Build last-frame full-image anchor queries for anchor-motion projection."""

    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError("frames_rgb must have shape [T,H,W,3]")
    if not query_config.sampling.enabled:
        return []

    anchors = query_config.sampling.anchors
    if not anchors.enabled:
        return []
    return _sample_detector_queries(
        frames_rgb=frames_rgb,
        query_config=query_config,
        times=[0],
        max_query_points=anchors.max_query_points,
        mask=None,
        params=anchors,
        source="anchor",
        start_id=start_id,
        balance_full_mask=True,
    )


def build_anchor_footprint_queries(
    frames_rgb: np.ndarray,
    query_config: QueryConfig,
    *,
    start_id: int = 0,
) -> list[QueryPoint]:
    """Compatibility alias: footprint points are disabled in this repo."""

    return build_anchor_motion_queries(frames_rgb, query_config, start_id=start_id)


def _sample_detector_queries(
    *,
    frames_rgb: np.ndarray,
    query_config: QueryConfig,
    times: list[int],
    max_query_points: int,
    mask: np.ndarray | None,
    params: QuerySamplingConfig | AnchorSamplingConfig,
    source: str,
    start_id: int,
    balance_full_mask: bool,
) -> list[QueryPoint]:
    if max_query_points <= 0:
        return []
    if not times:
        return []
    _, height, width = frames_rgb.shape[:3]
    detector = build_detector(query_config, params)
    samples_per_time = _samples_per_time(max_query_points, len(times), params)
    queries: list[QueryPoint] = []
    for reverse_time in times:
        keypoints = detector.detect(frames_rgb, reverse_time, mask)
        if not keypoints:
            continue
        picked = _pick_top_keypoints(
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


def _pick_top_keypoints(
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


def robot_footprint_mask(
    height: int,
    width: int,
    robot: FootprintConfig,
    sampling: QuerySamplingConfig,
) -> np.ndarray:
    alignment = align_footprint_to_image(height=height, width=width, robot=robot)
    left, right = alignment.left, alignment.right
    top, bottom = alignment.top, alignment.bottom

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:bottom, left:right] = 255

    if sampling.sample_at_edges:
        rect_width = max(1, right - left)
        edge_width = int(rect_width * sampling.edge_offset_ratio)
        inner_left = min(right, left + edge_width)
        inner_right = max(left, right - edge_width)
        if inner_right > inner_left:
            mask[top:bottom, inner_left:inner_right] = 0
    return mask


def align_footprint_to_image(
    *,
    height: int,
    width: int,
    robot: FootprintConfig,
) -> RobotImageAlignment:
    """Align a normalized bottom-center robot footprint to an image."""

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
        method="bottom_center_footprint",
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
        source_code = QUERY_SOURCE_CODES.get(canonical_query_source(query.source), -1)
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
            json.dumps([
                query.to_json() | {"source": canonical_query_source(query.source)}
                for query in queries
            ]),
            dtype=np.str_,
        ),
    }


def query_capture_metadata(
    config: QueryConfig,
    *,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    width_ratio, height_ratio = config.footprint.derived_footprint_ratios()
    avt_seed_y, avt_seed_x_min, avt_seed_x_max = config.footprint.avt_seed_ratios()
    alignment = (
        align_footprint_to_image(height=height, width=width, robot=config.footprint)
        if width is not None and height is not None
        else None
    )
    return {
        "schema": "avt_query_capture_v2",
        "numeric_columns": QUERY_NUMERIC_COLUMNS,
        "cotracker_columns": ["reverse_time", "x", "y"],
        "source_codes": PRIMARY_QUERY_SOURCE_CODES,
        "mode": config.mode,
        "footprint": asdict(config.footprint)
        | {
            "footprint_width_ratio": width_ratio,
            "footprint_height_ratio": height_ratio,
            "derived_avt_seed_y_ratio": avt_seed_y,
            "derived_avt_seed_x_min_ratio": avt_seed_x_min,
            "derived_avt_seed_x_max_ratio": avt_seed_x_max,
        },
        "image_alignment": alignment.to_json() if alignment else None,
        "sampling": asdict(config.sampling),
    }


def _sampling_times(frame_count: int, window_size: int) -> list[int]:
    if frame_count <= 1:
        return [0]
    num_windows = max(1, frame_count // window_size)
    idxs = np.linspace(0, frame_count - 1, num=num_windows, dtype=int)
    idxs = np.concatenate(([0], idxs, [max(0, frame_count - 2)]))
    return sorted({int(idx) for idx in idxs if 0 <= idx < frame_count})


def _samples_per_time(
    max_query_points: int,
    time_count: int,
    params: QuerySamplingConfig | AnchorSamplingConfig,
) -> int:
    if time_count <= 0:
        return 0
    base = int(max_query_points / time_count)
    min_points = max(1, int(params.min_points_per_frame))
    max_points = max(min_points, int(params.max_points_per_frame))
    return min(max_query_points, max(min_points, min(max_points, base)))


def _validate_mode(mode: str) -> str:
    if mode not in QUERY_MODES:
        raise ValueError(
            "query_mode must be one of: anchor_motion, anchor_footprint, footprint, avt, avt+footprint"
        )
    return mode


def canonical_query_source(source: str) -> str:
    return str(source)


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
