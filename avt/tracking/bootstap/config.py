from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .download import DEFAULT_BOOTSTAP_CHECKPOINT_URL


@dataclass
class BootstapConfig:
    """Runtime settings for the AVT BootsTAPIR adapter."""

    device: str = "auto"
    checkpoint_path: Path | None = None
    checkpoint_url: str | None = DEFAULT_BOOTSTAP_CHECKPOINT_URL
    download_checkpoint: bool = False
    resize_height: int = 512
    resize_width: int = 512
    query_chunk_size: int = 64
    pyramid_level: int = 1
    visibility_threshold: float = 0.5
    strict_checkpoint: bool = True


def load_bootstap_config_yaml(path: Path) -> BootstapConfig:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency is in pyproject
        raise RuntimeError("BootsTAPIR YAML configs require PyYAML.") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"BootsTAPIR config must be a YAML mapping: {path}")
    return bootstap_config_from_mapping(data)


def bootstap_config_from_mapping(data: dict[str, Any]) -> BootstapConfig:
    allowed = {field.name for field in fields(BootstapConfig)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown BootsTAPIR config keys: {', '.join(unknown)}")

    values = dict(data)
    if values.get("checkpoint_path") is not None:
        values["checkpoint_path"] = Path(values["checkpoint_path"])
    return BootstapConfig(**values)


def merge_bootstap_config(base: BootstapConfig, **overrides: Any) -> BootstapConfig:
    values = {field.name: getattr(base, field.name) for field in fields(BootstapConfig)}
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return BootstapConfig(**values)
