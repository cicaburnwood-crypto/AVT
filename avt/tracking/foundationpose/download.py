from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


HF_REPO_ID = "gpue/foundationpose-weights"
HF_BASE_URL = f"https://huggingface.co/{HF_REPO_ID}/resolve/main"

FOUNDATIONPOSE_WEIGHT_FILES = (
    "2023-10-28-18-33-37/config.yml",
    "2023-10-28-18-33-37/model_best.pth",
    "2024-01-11-20-02-45/config.yml",
    "2024-01-11-20-02-45/model_best.pth",
)


def default_weights_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "checkpoints" / "foundationpose"


def missing_foundationpose_weights(weights_dir: Path) -> list[str]:
    return [rel for rel in FOUNDATIONPOSE_WEIGHT_FILES if not (weights_dir / rel).exists()]


def ensure_foundationpose_weights(
    weights_dir: Path | None = None,
    *,
    download: bool = False,
) -> Path:
    """Validate or download FoundationPose refiner/scorer checkpoints.

    The files are not part of the AVT git repository. They are downloaded from
    a public Hugging Face mirror of the official FoundationPose weights.
    """

    root = (weights_dir or default_weights_dir()).expanduser().resolve()
    missing = missing_foundationpose_weights(root)
    if not missing:
        return root
    if not download:
        missing_text = "\n".join(f"  - {root / rel}" for rel in missing)
        raise FileNotFoundError(
            "FoundationPose weights are missing. Run "
            "`python -m avt.tracking.foundationpose.download` or pass "
            "--foundationpose-download-weights.\n"
            f"Missing files:\n{missing_text}"
        )

    root.mkdir(parents=True, exist_ok=True)
    for rel in missing:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(f"{HF_BASE_URL}/{rel}", dst)
    return root


def main() -> int:
    root = ensure_foundationpose_weights(download=True)
    print(root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
