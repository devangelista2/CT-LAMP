#!/usr/bin/env python
"""Generate a grid of unconditional samples from the CT-LAMP diffusion prior."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import torch
import torchvision
from diffusers import DDIMScheduler as DiffusersDDIMScheduler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from ct_lamp.diffusion.schedulers import DDPMScheduler
from ct_lamp.models.loader import build_model
from ct_lamp.utils.config import apply_overrides, load_config, resolve_device
from ct_lamp.utils.logging import setup_logging
from ct_lamp.utils.random import set_seed

logger = logging.getLogger(__name__)


def _deep_update(base: dict, override: dict) -> None:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate image grid from CT-LAMP prior.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--num_images", type=int, default=16, help="Number of images to sample.")
    parser.add_argument("--nrow", type=int, default=0, help="Grid columns. 0 selects sqrt(num_images).")
    parser.add_argument("--num_steps", type=int, default=100, help="Reverse diffusion steps.")
    parser.add_argument(
        "--sampler",
        type=str,
        default="ddim",
        choices=("ddim", "ddpm"),
        help="Scheduler used for unconditional generation.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="DDIM eta. Ignored for ddpm sampler.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/sample_grid.png",
        help="Output PNG path for grid.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("overrides", nargs="*", help="Dotted config overrides.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    base_cfg = load_config(Path(__file__).parent.parent / "configs" / "base.yaml")
    _deep_update(base_cfg, cfg)
    cfg = apply_overrides(base_cfg, args.overrides)

    setup_logging(cfg.get("output", {}).get("log_level", "INFO"))
    device = resolve_device(cfg)
    logger.info("Generating %d samples on %s", args.num_images, device)
    logger.info("Seed: %d", args.seed)

    set_seed(args.seed)

    model_cfg = cfg.get("model", {})
    dtype = torch.float16 if model_cfg.get("dtype") == "float16" else torch.float32
    model = build_model(cfg, device=device, dtype=dtype)

    in_channels = int(model.in_channels)
    sample_size = int(model.sample_size)
    shape = (args.num_images, in_channels, sample_size, sample_size)
    latents = torch.randn(shape, device=device, dtype=dtype)

    scheduler_cfg = cfg.get("scheduler", {})
    if args.sampler == "ddim":
        scheduler = DiffusersDDIMScheduler(
            num_train_timesteps=int(scheduler_cfg.get("num_train_timesteps", 1000)),
            beta_start=float(scheduler_cfg.get("beta_start", 0.0001)),
            beta_end=float(scheduler_cfg.get("beta_end", 0.02)),
            beta_schedule=str(scheduler_cfg.get("beta_schedule", "linear")),
            clip_sample=True,
            prediction_type="epsilon",
        )
    else:
        scheduler = DDPMScheduler(
            num_train_timesteps=int(scheduler_cfg.get("num_train_timesteps", 1000)),
            beta_start=float(scheduler_cfg.get("beta_start", 0.0001)),
            beta_end=float(scheduler_cfg.get("beta_end", 0.02)),
            beta_schedule=str(scheduler_cfg.get("beta_schedule", "linear")),
            clip_sample=True,
        ).to(torch.device(device))
    scheduler.set_timesteps(args.num_steps, device=torch.device(device))
    timesteps = scheduler.timesteps

    with torch.no_grad():
        for t in tqdm(timesteps, desc="Sampling"):
            t_int = int(t.item()) if torch.is_tensor(t) else int(t)
            t_input = torch.full((latents.shape[0],), t_int, device=device, dtype=torch.long)
            model_output = model.unet_forward(latents, t_input)
            if args.sampler == "ddim":
                step_out = scheduler.step(
                    model_output=model_output,
                    timestep=t_int,
                    sample=latents,
                    eta=float(args.eta),
                )
            else:
                step_out = scheduler.step(
                    model_output=model_output,
                    timestep=t_int,
                    sample=latents,
                )
            latents = step_out.prev_sample

    nrow = args.nrow if args.nrow > 0 else max(1, int(math.sqrt(args.num_images)))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(
        latents.float().cpu(),
        output_path,
        nrow=nrow,
        padding=2,
        normalize=True,
        value_range=(-1, 1),
    )
    logger.info("Saved sample grid to %s", output_path)


if __name__ == "__main__":
    main()
