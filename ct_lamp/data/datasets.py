"""Dataset utilities for Mayo sparse-view CT experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def _load_ct_slice(path: Path, image_size: int, num_channels: int) -> torch.Tensor:
    """Load one CT slice into a tensor in [-1, 1] with shape (1, C, H, W)."""
    img = Image.open(path).convert("L").resize((image_size, image_size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # [0,255] -> [-1,1]
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    if num_channels == 3:
        tensor = tensor.repeat(1, 3, 1, 1)
    return tensor


class MayoDataset:
    """Mayo low-dose CT dataset loader.

    Expected folder layout:
        root/split/case_id/slice.png

    Example:
        ../data/Mayo/test/C081/1.png
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "test",
        image_size: int = 256,
        num_images: int | None = None,
        start_idx: int = 0,
        case_ids: list[str] | None = None,
        num_channels: int = 1,
    ) -> None:
        if num_channels not in (1, 3):
            raise ValueError(f"num_channels must be 1 or 3, got {num_channels}.")

        root = Path(root_dir)
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Mayo split directory not found: {split_dir}")

        wanted_cases = set(case_ids or [])
        case_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        if wanted_cases:
            case_dirs = [p for p in case_dirs if p.name in wanted_cases]
            if not case_dirs:
                raise ValueError(
                    f"No matching case_ids found in {split_dir}: {sorted(wanted_cases)}"
                )

        paths: list[Path] = []
        for case_dir in case_dirs:
            for image_path in sorted(case_dir.iterdir()):
                if image_path.suffix.lower() in _IMAGE_EXTENSIONS:
                    paths.append(image_path)

        if start_idx < 0:
            raise ValueError(f"start_idx must be >= 0, got {start_idx}.")
        paths = paths[start_idx:]
        if num_images is not None:
            paths = paths[:num_images]
        if not paths:
            raise ValueError(
                f"No images found for Mayo dataset at {split_dir} "
                f"(start_idx={start_idx}, num_images={num_images})."
            )

        self.paths = paths
        self.image_size = image_size
        self.num_channels = num_channels
        logger.info(
            "MayoDataset: %d slices from %s (split=%s, channels=%d)",
            len(paths),
            root,
            split,
            num_channels,
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return _load_ct_slice(self.paths[idx], self.image_size, self.num_channels)

    def __iter__(self) -> Iterator[torch.Tensor]:
        for i in range(len(self)):
            yield self[i]


def build_dataset(cfg: dict) -> MayoDataset:
    """Build the Mayo dataset from config."""
    data_cfg = cfg.get("data", {})
    source = data_cfg.get("source", "mayo")
    if source != "mayo":
        raise ValueError(f"Only data.source='mayo' is supported, got {source!r}.")

    return MayoDataset(
        root_dir=data_cfg.get("root_dir", "../data/Mayo"),
        split=data_cfg.get("split", "test"),
        image_size=data_cfg.get("image_size", 256),
        num_images=data_cfg.get("num_images", 10),
        start_idx=data_cfg.get("start_idx", 0),
        case_ids=data_cfg.get("case_ids", []),
        num_channels=data_cfg.get("num_channels", 1),
    )
