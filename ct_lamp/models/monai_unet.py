"""MONAI UNet loader adapted from RD-DGP."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import torch
import yaml

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.autocast\(args\.\.\.\) is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*cuda\.cudart module is deprecated.*",
    category=FutureWarning,
)

try:
    # Exact class used by RD-DGP.
    from generative.networks.nets import DiffusionModelUNet as RDGPDiffusionModelUNet
    _HAS_RDGP_GENERATIVE = True
except Exception:
    RDGPDiffusionModelUNet = None
    _HAS_RDGP_GENERATIVE = False

from monai.networks.nets import DiffusionModelUNet as MonaiDiffusionModelUNet


def load_monai_config(config_path: str | Path) -> dict[str, Any]:
    """Load MONAI UNet YAML config."""
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"MONAI config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML dict in {path}, got {type(data)}")
    return data


def create_monai_unet(config: dict[str, Any]):
    """Instantiate UNet using RD-DGP schema.

    If monai-generative is available, we use the exact RD-DGP class.
    """
    common_kwargs = dict(
        spatial_dims=2,
        in_channels=int(config.get("in_channels", 1)),
        out_channels=int(config.get("out_channels", 1)),
        attention_levels=tuple(config.get("attention_levels", (False, False, True, True))),
        num_res_blocks=int(config.get("layers_per_block", 2)),
        num_head_channels=int(config.get("num_head_channels", 32)),
    )
    block_channels = tuple(config.get("block_out_channels", (64, 128, 256, 512)))

    if _HAS_RDGP_GENERATIVE:
        logger.info("Using generative.networks.nets.DiffusionModelUNet backend.")
        return RDGPDiffusionModelUNet(num_channels=block_channels, **common_kwargs)

    logger.warning(
        "monai-generative is not available; falling back to monai.networks.nets. "
        "This may produce different sampling quality than RD-DGP."
    )
    return MonaiDiffusionModelUNet(channels=block_channels, **common_kwargs)


def _strip_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        cleaned[nk] = v
    return cleaned


def _remap_monai_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map legacy RD-DGP MONAI key names to current MONAI names."""
    remapped: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        nk = nk.replace(".to_q.", ".attn.to_q.")
        nk = nk.replace(".to_k.", ".attn.to_k.")
        nk = nk.replace(".to_v.", ".attn.to_v.")
        nk = nk.replace(".to_out.0.", ".attn.out_proj.")
        nk = nk.replace(".proj_attn.", ".attn.out_proj.")
        nk = nk.replace(".upsampler.conv.conv.", ".upsampler.postconv.conv.")
        remapped[nk] = v
    return remapped


def load_monai_unet(
    weights_path: str | Path,
    config_path: str | Path,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Any, dict[str, Any]]:
    """Load RD-DGP MONAI UNet and weights."""
    cfg = load_monai_config(config_path)
    model = create_monai_unet(cfg)

    w_path = Path(weights_path).expanduser()
    if not w_path.is_file():
        raise FileNotFoundError(f"MONAI weights not found: {w_path}")
    raw = torch.load(w_path, map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError(f"Expected state_dict in {w_path}, got {type(raw)}")

    state = _strip_prefixes(raw)
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        # For monai fallback only, try legacy key remap.
        if _HAS_RDGP_GENERATIVE:
            raise
        remapped = _remap_monai_keys(state)
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "MONAI checkpoint did not load cleanly after key remapping: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

    model = model.to(device=device, dtype=dtype).eval()
    logger.info(
        "Loaded MONAI UNet from %s (sample_size=%s, in_channels=%s)",
        w_path,
        cfg.get("sample_size", "unknown"),
        cfg.get("in_channels", "unknown"),
    )
    return model, cfg
