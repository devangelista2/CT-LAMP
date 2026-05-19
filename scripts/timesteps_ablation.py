#!/usr/bin/env python
"""Ablate reconstruction quality as a function of the number of timesteps."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ct_lamp.data.datasets import build_dataset
from ct_lamp.metrics.metrics import compute_lpips, compute_psnr, compute_ssim
from ct_lamp.models.loader import build_model
from ct_lamp.operators.sparse_view_ct import SparseViewCTProjector
from ct_lamp.utils.config import apply_overrides, load_config, resolve_device
from ct_lamp.utils.logging import setup_logging
from ct_lamp.utils.output import OutputManager
from ct_lamp.utils.random import set_seed

logger = logging.getLogger(__name__)

SAMPLERS = {
    "fbp": "ct_lamp.samplers.fbp.FBPSampler",
    "sirt": "ct_lamp.samplers.sirt.SIRTSampler",
    "ddnm_plus": "ct_lamp.samplers.ddnm_plus.DDNMPlusSampler",
    "ct_lamp": "ct_lamp.samplers.ct_lamp.CTLAMPSampler",
    "ct_lamp_3m": "ct_lamp.samplers.ct_lamp_3m.CTLAMP3MSampler",
    "score_sde": "ct_lamp.samplers.score_sde.ScoreSDESampler",
    "mcg": "ct_lamp.samplers.mcg.MCGSampler",
    "dps": "ct_lamp.samplers.dps.DPSSampler",
    "ps_plus": "ct_lamp.samplers.ps_plus.PSPlusSampler",
    "diffpir": "ct_lamp.samplers.diffpir.DiffPIRSampler",
}


def import_sampler(dotted_path: str):
    module_path, _, class_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _deep_update(base: dict, override: dict) -> None:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablate a method over different numbers of sampling timesteps."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--method",
        required=True,
        choices=list(SAMPLERS.keys()),
        help="Method to ablate.",
    )
    parser.add_argument(
        "--timesteps",
        required=True,
        type=int,
        nargs="+",
        help="List of timestep counts to evaluate, e.g. --timesteps 10 20 50.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Override the number of test images to evaluate.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="Override the dataset start index for the test subset.",
    )
    parser.add_argument("overrides", nargs="*", help="Extra dotted config overrides.")
    args = parser.parse_args()
    args.timesteps = sorted(dict.fromkeys(args.timesteps))
    return args


def build_trial_cfg(base_cfg: dict, method: str, num_steps: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault(method, {})
    cfg[method]["num_steps"] = int(num_steps)
    return cfg


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_ssim_plot(out_dir: Path, summary_rows: list[dict]) -> None:
    if not summary_rows:
        return

    OutputManager._apply_paper_rcparams()

    summary_rows = sorted(summary_rows, key=lambda row: int(row["num_steps"]))
    xs = np.asarray([int(row["num_steps"]) for row in summary_rows], dtype=np.int64)
    ys = np.asarray([float(row["ssim_mean"]) for row in summary_rows], dtype=np.float64)
    stds = np.asarray([float(row["ssim_std"]) for row in summary_rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(xs, ys, color="#1b3a57", marker="o", label="Mean SSIM")
    ax.fill_between(
        xs,
        np.clip(ys - stds, 0.0, 1.0),
        np.clip(ys + stds, 0.0, 1.0),
        color="#1b3a57",
        alpha=0.18,
        linewidth=0.0,
        label="Mean +/- Std",
    )
    ax.set_xlabel("Number of Timesteps")
    ax.set_ylabel("SSIM")
    ax.set_title("Timesteps Ablation")
    ax.set_ylim(0.0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "ssim_vs_timesteps.pdf", bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    base_cfg = load_config(Path(__file__).parent.parent / "configs" / "base.yaml")
    _deep_update(base_cfg, cfg)
    cfg = apply_overrides(base_cfg, args.overrides)

    if args.num_images is not None:
        cfg.setdefault("data", {})
        cfg["data"]["num_images"] = int(args.num_images)
    if args.start_idx is not None:
        cfg.setdefault("data", {})
        cfg["data"]["start_idx"] = int(args.start_idx)

    setup_logging(cfg.get("output", {}).get("log_level", "INFO"))
    logger.info(
        "Starting timesteps ablation for method=%s with timesteps=%s",
        args.method,
        args.timesteps,
    )

    device = resolve_device(cfg)
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    set_seed(seed)
    logger.info("Device: %s", device)
    logger.info("Seed: %d", seed)

    model_cfg = cfg.get("model", {})
    dtype = torch.float16 if model_cfg.get("dtype") == "float16" else torch.float32
    model = build_model(cfg, device=device, dtype=dtype)

    data_cfg = cfg.get("data", {})
    op_cfg = cfg.get("operator", {})
    image_size = int(data_cfg.get("image_size", 256))
    operator = SparseViewCTProjector(
        image_size=image_size,
        num_angles=int(op_cfg.get("num_angles", 64)),
        det_count=int(op_cfg.get("det_count", image_size)),
        start_angle=float(op_cfg.get("start_angle", 0.0)),
        end_angle=float(op_cfg.get("end_angle", 3.141592653589793)),
        device=device,
    )

    dataset = list(build_dataset(cfg))
    out = OutputManager(cfg.get("output", {}).get("base_dir", "outputs"))
    cfg_to_save = copy.deepcopy(cfg)
    cfg_to_save.setdefault("timesteps_ablation", {})
    cfg_to_save["timesteps_ablation"]["method"] = args.method
    cfg_to_save["timesteps_ablation"]["timesteps"] = args.timesteps
    out.save_config(cfg_to_save)

    noise_sigma = float(op_cfg.get("noise_sigma", 0.01))
    start_idx = int(data_cfg.get("start_idx", 0))
    prepared: list[tuple[torch.Tensor, torch.Tensor]] = []
    for img_idx, x_true in enumerate(dataset):
        x_true = x_true.to(device)
        with torch.no_grad():
            y = operator.forward(x_true)
            noise_rng = torch.Generator(device=device)
            noise_rng.manual_seed(seed + start_idx + img_idx)
            noise = torch.randn(y.shape, generator=noise_rng, device=device) * noise_sigma
            y_noisy = (y + noise).clamp(min=0.0)
        prepared.append((x_true, y_noisy))

    sampler_class = import_sampler(SAMPLERS[args.method])
    per_image_rows: list[dict] = []
    summary_rows: list[dict] = []

    for num_steps in args.timesteps:
        logger.info("=== %s with %d timesteps ===", args.method, num_steps)
        trial_cfg = build_trial_cfg(cfg, args.method, num_steps)
        sampler = sampler_class(model, operator, trial_cfg)

        trial_psnr: list[float] = []
        trial_ssim: list[float] = []
        trial_lpips: list[float] = []
        trial_times: list[float] = []

        for img_idx, (x_true, measurement) in enumerate(prepared):
            set_seed(seed + start_idx + img_idx)
            b, c, h, w = x_true.shape
            t0 = time.time()
            x_hat, _ = sampler.sample(
                measurement=measurement,
                shape=(b, c, h, w),
                x_true=None,
            )
            elapsed = time.time() - t0

            psnr = compute_psnr(x_hat, x_true)
            ssim = compute_ssim(x_hat, x_true)
            lpips_val = compute_lpips(x_hat, x_true)

            trial_psnr.append(psnr)
            trial_ssim.append(ssim)
            trial_lpips.append(lpips_val)
            trial_times.append(elapsed)

            per_image_rows.append(
                {
                    "num_steps": num_steps,
                    "image": img_idx,
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lpips_val,
                    "time_s": elapsed,
                }
            )
            logger.info(
                "timesteps=%d image=%d: PSNR=%.2f SSIM=%.4f LPIPS=%.4f (%.1fs)",
                num_steps,
                img_idx,
                psnr,
                ssim,
                lpips_val,
                elapsed,
            )

        summary_row = {
            "num_steps": num_steps,
            "psnr_mean": float(np.nanmean(trial_psnr)),
            "psnr_std": float(np.nanstd(trial_psnr)),
            "ssim_mean": float(np.nanmean(trial_ssim)),
            "ssim_std": float(np.nanstd(trial_ssim)),
            "lpips_mean": float(np.nanmean(trial_lpips)),
            "lpips_std": float(np.nanstd(trial_lpips)),
            "time_mean_s": float(np.nanmean(trial_times)),
            "time_std_s": float(np.nanstd(trial_times)),
        }
        summary_rows.append(summary_row)
        logger.info(
            "timesteps=%d summary: PSNR=%.2f +/- %.2f, SSIM=%.4f +/- %.4f, LPIPS=%.4f +/- %.4f",
            num_steps,
            summary_row["psnr_mean"],
            summary_row["psnr_std"],
            summary_row["ssim_mean"],
            summary_row["ssim_std"],
            summary_row["lpips_mean"],
            summary_row["lpips_std"],
        )

    save_csv(out.run_dir / "per_image_metrics.csv", per_image_rows)
    save_csv(out.run_dir / "summary_metrics.csv", summary_rows)
    save_ssim_plot(out.run_dir, summary_rows)
    logger.info("Timesteps ablation complete. Results in: %s", out.run_dir)


if __name__ == "__main__":
    main()
