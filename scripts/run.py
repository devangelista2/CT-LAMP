#!/usr/bin/env python
"""Run one CT-LAMP method on Mayo sparse-view CT data."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Run CT-LAMP on Mayo sparse-view CT.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--method",
        default="ct_lamp",
        choices=list(SAMPLERS.keys()),
        help="Method to run.",
    )
    parser.add_argument("overrides", nargs="*", help="Dotted config overrides.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    base_cfg = load_config(Path(__file__).parent.parent / "configs" / "base.yaml")
    _deep_update(base_cfg, cfg)
    cfg = apply_overrides(base_cfg, args.overrides)

    setup_logging(cfg.get("output", {}).get("log_level", "INFO"))
    logger.info("Starting CT-LAMP run with method=%s", args.method)

    device = resolve_device(cfg)
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    set_seed(seed)
    logger.info("Device: %s", device)
    logger.info("Seed: %d", seed)

    model_cfg = cfg.get("model", {})
    dtype = torch.float16 if model_cfg.get("dtype") == "float16" else torch.float32
    model = build_model(cfg, device=device, dtype=dtype)

    data_cfg = cfg.get("data", {})
    image_size = int(data_cfg.get("image_size", 256))
    if hasattr(model, "sample_size") and model.sample_size != image_size:
        logger.warning(
            "data.image_size=%d differs from UNet sample_size=%d.",
            image_size,
            model.sample_size,
        )

    op_cfg = cfg.get("operator", {})
    operator = SparseViewCTProjector(
        image_size=image_size,
        num_angles=int(op_cfg.get("num_angles", 64)),
        det_count=int(op_cfg.get("det_count", image_size)),
        start_angle=float(op_cfg.get("start_angle", 0.0)),
        end_angle=float(op_cfg.get("end_angle", 3.141592653589793)),
        device=device,
    )

    dataset = build_dataset(cfg)
    out = OutputManager(cfg.get("output", {}).get("base_dir", "outputs"))
    out.save_config(cfg)

    sampler_class = import_sampler(SAMPLERS[args.method])
    sampler = sampler_class(model, operator, cfg)

    all_metrics: list[dict] = []
    method_outputs_last: dict[str, torch.Tensor] = {}
    last_measurement: torch.Tensor | None = None
    last_gt: torch.Tensor | None = None

    noise_sigma = float(op_cfg.get("noise_sigma", 0.01))
    for img_idx, x_true in enumerate(dataset):
        x_true = x_true.to(device)
        logger.info("Image %d/%d", img_idx + 1, len(dataset))

        with torch.no_grad():
            y = operator.forward(x_true)
            noise = torch.randn_like(y) * noise_sigma
            y_noisy = (y + noise).clamp(min=0.0)
        logger.info(
            "Measurement range: raw=[%.3f, %.3f] noisy=[%.3f, %.3f]",
            y.min().item(),
            y.max().item(),
            y_noisy.min().item(),
            y_noisy.max().item(),
        )

        method_dir = out.method_dir(args.method)
        out.save_image(x_true, method_dir / f"img_{img_idx:04d}_ground_truth.png")
        out.save_measurement(y_noisy, method_dir / f"img_{img_idx:04d}_measurement.png")

        b, c, h, w = x_true.shape
        t0 = time.time()
        x_hat, history = sampler.sample(
            measurement=y_noisy,
            shape=(b, c, h, w),
            x_true=x_true,
        )
        elapsed = time.time() - t0

        out.save_image(x_hat, method_dir / f"img_{img_idx:04d}_solution.png")
        out.save_step_metrics_csv(history, args.method)

        psnr = compute_psnr(x_hat, x_true)
        ssim = compute_ssim(x_hat, x_true)
        lpips_val = compute_lpips(x_hat, x_true)
        metrics = {"psnr": psnr, "ssim": ssim, "lpips": lpips_val, "time_s": elapsed}
        all_metrics.append(metrics)
        out.save_metrics_json(metrics, args.method, img_idx)
        logger.info(
            "Image %d: PSNR=%.2f SSIM=%.4f LPIPS=%.4f (%.1fs)",
            img_idx,
            psnr,
            ssim,
            lpips_val,
            elapsed,
        )

        method_outputs_last = {args.method: x_hat}
        last_measurement = y_noisy
        last_gt = x_true

    if method_outputs_last and last_measurement is not None and last_gt is not None:
        out.save_comparison_figure(last_measurement, last_gt, method_outputs_last, img_idx=img_idx)

    avg = {
        "method": args.method,
        "psnr": sum(m["psnr"] for m in all_metrics) / len(all_metrics),
        "ssim": sum(m["ssim"] for m in all_metrics) / len(all_metrics),
        "lpips": sum(m["lpips"] for m in all_metrics) / len(all_metrics),
        "time_s": sum(m["time_s"] for m in all_metrics) / len(all_metrics),
    }
    out.save_comparison_csv([avg])
    logger.info("Done. Results in: %s", out.run_dir)


if __name__ == "__main__":
    main()
