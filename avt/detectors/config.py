"""Per-detector configuration dataclasses.

Pure dataclasses (no cv2/torch imports) so ``QueryConfig`` in ``querying.py`` can
embed them without creating an import cycle with the detector implementations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OrbDetectorConfig:
    """OpenCV ORB keypoint detector (FAST corners + Harris/FAST scoring)."""

    nfeatures: int = 2000
    scale_factor: float = 1.2
    n_levels: int = 8
    edge_threshold: int = 31
    first_level: int = 0
    wta_k: int = 2
    score_type: str = "harris"  # "harris" or "fast"
    patch_size: int = 31
    fast_threshold: int = 20
    use_clahe: bool = True  # reuses the SIFT CLAHE clip/tile settings


@dataclass
class SuperPointConfig:
    """SuperPoint detector (HF Transformers), optionally prefiltered by SuperGlue.

    Default (``use_superglue=False``): pure SuperPoint keypoints inside the mask,
    ranked by detector score. When ``use_superglue=True``: keep only keypoints
    SuperGlue confidently matches against a neighbor frame (response = match
    score). SuperGlue's magic-leap-community/superglue_outdoor weights carry a
    research/non-commercial license.
    """

    model: str = "magic-leap-community/superpoint"
    use_superglue: bool = False  # default: pure SuperPoint detection
    superglue_model: str = "magic-leap-community/superglue_outdoor"
    keypoint_threshold: float = 0.005
    max_keypoints: int = 1024
    match_threshold: float = 0.2
    neighbor_offset: int = 1  # frames away (in reversed clip) to match against
    device: str | None = None  # None -> cuda if available else cpu


@dataclass
class XFeatConfig:
    """XFeat accelerated features (torch.hub verlab/accelerated_features)."""

    hub_repo: str = "verlab/accelerated_features"
    model: str = "XFeat"
    top_k: int = 4096
    detection_threshold: float = 0.05
    checkpoint: str | None = None  # optional local weights override
    device: str | None = None  # None -> cuda if available else cpu
