"""Local diffusion model loading and schedule utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from ct_lamp.diffusion.models import UNet2DModel
from ct_lamp.diffusion.schedulers import DDPMScheduler
from ct_lamp.models.monai_unet import load_monai_unet

logger = logging.getLogger(__name__)


class NoiseSchedule:
    """Pre-computed alpha, sigma, and lambda_t arrays for DPM-Solver++ updates."""

    def __init__(self, alphas_cumprod: torch.Tensor, device: str) -> None:
        self.device = device
        self.alphas_cumprod = alphas_cumprod.to(device=device, dtype=torch.float32)
        self.alpha = torch.sqrt(self.alphas_cumprod).clamp(min=1e-12)
        self.sigma = torch.sqrt(1.0 - self.alphas_cumprod).clamp(min=1e-12)
        self.log_snr = torch.log(self.alpha / self.sigma)
        self.T = int(self.alphas_cumprod.shape[0])

    def get_timestep_sequence(self, num_steps: int) -> list[int]:
        """Return timesteps approximately uniform in lambda_t = log(alpha_t / sigma_t)."""
        if num_steps <= 1:
            return [self.T - 1]
        lambda_min = self.log_snr[-1].item()
        # skip t=0 to avoid nearly infinite lambda
        lambda_max = self.log_snr[1].item() if self.T > 1 else self.log_snr[0].item()
        targets = torch.linspace(lambda_min, lambda_max, num_steps, device=self.device)
        diff = (self.log_snr.unsqueeze(0) - targets.unsqueeze(1)).abs()
        timesteps = diff.argmin(dim=1).tolist()
        # Keep descending unique timesteps. Duplicates can appear for short schedules.
        return sorted({int(t) for t in timesteps}, reverse=True)

    def get_alpha(self, t: int) -> torch.Tensor:
        return self.alpha[t]

    def get_sigma(self, t: int) -> torch.Tensor:
        return self.sigma[t]

    def get_lambda(self, t: int) -> torch.Tensor:
        return self.log_snr[t]


def _extract_state_dict(raw: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract a model state dict from common checkpoint wrappers."""
    for key in ("state_dict", "model_state_dict", "model", "unet", "ema_state_dict"):
        candidate = raw.get(key)
        if isinstance(candidate, dict):
            raw = candidate
            break

    state: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if not isinstance(v, torch.Tensor):
            continue
        # Strip common wrappers from training scripts.
        for prefix in ("module.", "model.", "unet."):
            if k.startswith(prefix):
                k = k[len(prefix) :]
        state[k] = v
    return state


def _load_local_unet(model_cfg: dict[str, Any]) -> UNet2DModel:
    """Load UNet2DModel from local checkpoint directory or file."""
    checkpoint_path = Path(model_cfg["checkpoint_path"]).expanduser()
    if checkpoint_path.is_dir():
        logger.info("Loading local UNet directory: %s", checkpoint_path)
        return UNet2DModel.from_pretrained(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"model.checkpoint_path not found: {checkpoint_path}")

    config_path = model_cfg.get("model_config_path")
    if config_path is None:
        raise ValueError(
            "When checkpoint_path points to a .pt/.bin file, "
            "model.model_config_path must be provided."
        )

    config_path = Path(config_path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"model.model_config_path not found: {config_path}")
    if config_path.suffix.lower() == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
    else:
        config_dict = torch.load(config_path, map_location="cpu")
    if not isinstance(config_dict, dict):
        raise ValueError(
            f"model_config_path must contain a config dict, got {type(config_dict)}"
        )

    unet = UNet2DModel(**config_dict)
    raw = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected checkpoint content in {checkpoint_path}: {type(raw)}")
    state = _extract_state_dict(raw)
    missing, unexpected = unet.load_state_dict(state, strict=False)
    if missing:
        logger.warning("UNet missing keys (%d): %s", len(missing), missing[:8])
    if unexpected:
        logger.warning("UNet unexpected keys (%d): %s", len(unexpected), unexpected[:8])
    return unet


class LocalDiffusionModel:
    """Diffusion wrapper around local UNet2DModel and local DDPM scheduler."""

    def __init__(self, cfg: dict, device: str, dtype: torch.dtype = torch.float32) -> None:
        model_cfg = cfg.get("model", {})
        scheduler_cfg = cfg.get("scheduler", {})

        self.device = device
        self.dtype = dtype
        self.backend = str(model_cfg.get("backend", "local_unet2d"))
        self.model_config: dict[str, Any] = {}
        if self.backend == "monai":
            config_path = model_cfg.get("model_config_path")
            if config_path is None:
                raise ValueError("model.model_config_path is required when model.backend='monai'.")
            self.unet, monai_cfg = load_monai_unet(
                weights_path=model_cfg["checkpoint_path"],
                config_path=config_path,
                device=device,
                dtype=dtype,
            )
            self.model_config = monai_cfg
        else:
            self.unet = _load_local_unet(model_cfg).to(device=device, dtype=dtype).eval()

        self.scheduler = DDPMScheduler(
            num_train_timesteps=int(scheduler_cfg.get("num_train_timesteps", 1000)),
            beta_start=float(scheduler_cfg.get("beta_start", 0.0001)),
            beta_end=float(scheduler_cfg.get("beta_end", 0.02)),
            beta_schedule=str(scheduler_cfg.get("beta_schedule", "linear")),
            clip_sample=bool(scheduler_cfg.get("clip_sample", False)),
        ).to(torch.device(device))

        alphas_cumprod = self.scheduler.alphas_cumprod.detach().clone()
        self.noise_schedule = NoiseSchedule(alphas_cumprod, device=device)
        if self.backend == "monai":
            self.in_channels = int(self.model_config.get("in_channels", 1))
            self.sample_size = int(self.model_config.get("sample_size", 256))
        else:
            self.in_channels = int(self.unet.in_channels)
            self.sample_size = int(self.unet.sample_size)
        logger.info(
            "Loaded diffusion model backend=%s: sample_size=%d, in_channels=%d, T=%d",
            self.backend,
            self.sample_size,
            self.in_channels,
            self.noise_schedule.T,
        )

    def unet_forward(
        self,
        x_t: torch.Tensor,
        t_tensor: torch.Tensor,
        grad: bool = False,
    ) -> torch.Tensor:
        if grad:
            out = self.unet(x_t.to(self.dtype), t_tensor)
            return out.sample if hasattr(out, "sample") else out
        with torch.no_grad():
            out = self.unet(x_t.to(self.dtype), t_tensor)
            return out.sample if hasattr(out, "sample") else out


def build_model(cfg: dict, device: str, dtype: torch.dtype) -> LocalDiffusionModel:
    """Build local diffusion model from config."""
    return LocalDiffusionModel(cfg=cfg, device=device, dtype=dtype)
