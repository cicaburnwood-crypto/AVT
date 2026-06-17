from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


DEFAULT_BOOTSTAP_CHECKPOINT_URL = (
    "https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt"
)
DEFAULT_BOOTSTAP_CHECKPOINT = (
    Path(__file__).resolve().parents[3]
    / "checkpoints"
    / "bootstap"
    / "bootstapir_checkpoint_v2.pt"
)


def ensure_bootstap_checkpoint(
    checkpoint_path: Path | None = None,
    *,
    download: bool = False,
    url: str | None = None,
) -> Path:
    path = (checkpoint_path or DEFAULT_BOOTSTAP_CHECKPOINT).expanduser().resolve()
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(
            f"BootsTAPIR checkpoint not found at {path}. "
            "Download it first or pass --bootstap-download-checkpoint."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url or DEFAULT_BOOTSTAP_CHECKPOINT_URL, path)
    return path


if __name__ == "__main__":  # pragma: no cover - helper entrypoint
    print(ensure_bootstap_checkpoint(download=True))
