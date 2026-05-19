"""Image I/O utilities for tensors in [-1, 1]."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_image(path: str | Path, size: int | None = None, num_channels: int = 3) -> torch.Tensor:
    """Load an image as (1, C, H, W) float32 tensor in [-1, 1].

    Args:
        path: Path to image file.
        size: If given, resize to (size, size) before conversion.

    Returns:
        Tensor of shape (1, 3, H, W) in [-1, 1].
    """
    if num_channels not in (1, 3):
        raise ValueError(f"num_channels must be 1 or 3, got {num_channels}.")
    img = Image.open(path).convert("L" if num_channels == 1 else "RGB")
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    if num_channels == 1:
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    else:
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a (1,C,H,W) or (C,H,W) float32 tensor in [-1,1] as PNG.

    Args:
        tensor: Image tensor in [-1, 1].
        path: Output file path (parent directory must exist).
    """
    path = Path(path)
    t = tensor.detach().cpu()
    if t.dim() == 4:
        t = t.squeeze(0)  # (3, H, W)
    if t.dim() != 3:
        raise ValueError(f"Expected 3D tensor after squeeze, got shape {tuple(t.shape)}")
    if t.shape[0] == 1:
        arr = t.squeeze(0).numpy()  # (H, W)
        arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(path)
        return
    if t.shape[0] == 3:
        arr = t.permute(1, 2, 0).numpy()  # (H, W, 3)
        arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
        Image.fromarray(arr, mode="RGB").save(path)
        return
    raise ValueError(f"Unsupported channel count for save_image: {t.shape[0]}")


def save_measurement(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a nonnegative sinogram-like tensor with per-image min-max normalization."""
    path = Path(path)
    t = tensor.detach().cpu()
    if t.dim() == 4:
        t = t.squeeze(0)
    if t.dim() != 3:
        raise ValueError(f"Expected 3D tensor after squeeze, got shape {tuple(t.shape)}")
    if t.shape[0] != 1:
        raise ValueError(f"Expected 1 channel measurement, got {t.shape[0]}")
    arr = t.squeeze(0).numpy()
    arr = np.maximum(arr, 0.0)
    vmax = float(arr.max())
    if vmax > 0:
        arr = arr / vmax
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a (1,C,H,W) or (C,H,W) tensor in [-1,1] to PIL."""
    t = tensor.detach().cpu()
    if t.dim() == 4:
        t = t.squeeze(0)
    if t.shape[0] == 1:
        arr = t.squeeze(0).numpy()
        arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")
    if t.shape[0] == 3:
        arr = t.permute(1, 2, 0).numpy()
        arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    raise ValueError(f"Unsupported channel count for tensor_to_pil: {t.shape[0]}")


def measurement_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a nonnegative sinogram-like tensor to PIL using min-max normalization."""
    t = tensor.detach().cpu()
    if t.dim() == 4:
        t = t.squeeze(0)
    if t.shape[0] != 1:
        raise ValueError(f"Expected 1 channel measurement, got {t.shape[0]}")
    arr = t.squeeze(0).numpy()
    arr = np.maximum(arr, 0.0)
    vmax = float(arr.max())
    if vmax > 0:
        arr = arr / vmax
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def pil_to_tensor(img: Image.Image, num_channels: int = 3) -> torch.Tensor:
    """Convert PIL image to (1,C,H,W) in [-1,1]."""
    if num_channels == 1:
        arr = np.array(img.convert("L"), dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
