from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrameRecord:
    """One RGB source frame in original chronological order."""

    index: int
    path: str
    timestamp: float | None = None
    rel_time_sec: float | None = None

    def resolved_path(self, root: Path) -> Path:
        return (root / self.path).resolve()

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QueryPoint:
    """A point inserted into the reversed video at reverse_time."""

    id: int
    reverse_time: int
    x: float
    y: float
    side: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WindowSpec:
    start: int
    end: int

    @property
    def id(self) -> str:
        return f"seq_{self.start}_{self.end}"

    @property
    def frame_count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class TrackerInfo:
    name: str
    version: str | None = None
    parameters: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
