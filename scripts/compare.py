#!/usr/bin/env python
"""Compare CT-LAMP family methods on Mayo sparse-view CT."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from collections import defaultdict
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

DEFAULT_METHODS = [
    "fbp",
    "sirt",
    "ddnm_plus",
    "ct_lamp",
    "ct_lamp_3m",
    "score_sde",
    "mcg",
    "dps",
    "ps_plus",
    "diffpir",
]


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
    parser = argparse.ArgumentParser(description="Compare CT-LAMP methods.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated method list (e.g. ddnm_plus,ct_lamp,ct_lamp_3m).",
    )
    parser.add_argument("overrides", nargs="*", help="Dotted config overrides.")
    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in SAMPLERS]
    if unknown:
        parser.error(f"Unknown methods: {unknown}. Allowed: {list(SAMPLERS)}")
    args.methods = methods
    return args


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    base_cfg = load_config(Path(__file__).parent.parent / "configs" / "base.yaml")
    _deep_update(base_cfg, cfg)
    cfg = apply_overrides(base_cfg, args.overrides)

    setup_logging(cfg.get("output", {}).get("log_level", "INFO"))
    logger.info("Starting CT-LAMP comparison.")

    device = resolve_device(cfg)
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    set_seed(seed)
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

    dataset = build_dataset(cfg)
    out = OutputManager(cfg.get("output", {}).get("base_dir", "outputs"))
    out.save_config(cfg)

    comparison_rows: list[dict] = []
    method_histories: dict[str, list[list[dict]]] = defaultdict(list)
    last_outputs: dict[str, torch.Tensor] = {}
    last_measurement: torch.Tensor | None = None
    last_gt: torch.Tensor | None = None

    noise_sigma = float(op_cfg.get("noise_sigma", 0.01))
    start_idx = int(data_cfg.get("start_idx", 0))

    for img_idx, x_true in enumerate(dataset):
        x_true = x_true.to(device)
        logger.info("=== Image %d/%d ===", img_idx + 1, len(dataset))

        with torch.no_grad():
            y = operator.forward(x_true)
            noise_rng = torch.Generator(device=device)
            noise_rng.manual_seed(seed + start_idx + img_idx)
            noise = (
                torch.randn(y.shape, generator=noise_rng, device=device) * noise_sigma
            )
            y_noisy = (y + noise).clamp(min=0.0)
        logger.info(
            "Measurement range: raw=[%.3f, %.3f] noisy=[%.3f, %.3f]",
            y.min().item(),
            y.max().item(),
            y_noisy.min().item(),
            y_noisy.max().item(),
        )

        b, c, h, w = x_true.shape
        image_outputs: dict[str, torch.Tensor] = {}
        for method_name in args.methods:
            set_seed(seed)
            logger.info("--- Method: %s ---", method_name)

            method_dir = out.method_dir(method_name)
            out.save_image(x_true, method_dir / f"img_{img_idx:04d}_ground_truth.png")
            out.save_measurement(
                y_noisy, method_dir / f"img_{img_idx:04d}_measurement.png"
            )

            sampler_class = import_sampler(SAMPLERS[method_name])
            sampler = sampler_class(model, operator, cfg)

            t0 = time.time()
            x_hat, history = sampler.sample(
                measurement=y_noisy,
                shape=(b, c, h, w),
                x_true=x_true,
            )
            elapsed = time.time() - t0

            out.save_image(x_hat, method_dir / f"img_{img_idx:04d}_solution.png")
            out.save_step_metrics_csv(history, method_name)
            method_histories[method_name].append(history)

            psnr = compute_psnr(x_hat, x_true)
            ssim = compute_ssim(x_hat, x_true)
            lpips_val = compute_lpips(x_hat, x_true)
            comparison_rows.append(
                {
                    "image": img_idx,
                    "method": method_name,
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lpips_val,
                    "time_s": elapsed,
                }
            )
            logger.info(
                "%s: PSNR=%.2f SSIM=%.4f LPIPS=%.4f (%.1fs)",
                method_name,
                psnr,
                ssim,
                lpips_val,
                elapsed,
            )
            image_outputs[method_name] = x_hat

        last_outputs = image_outputs
        last_measurement = y_noisy
        last_gt = x_true

    if last_measurement is not None and last_gt is not None and last_outputs:
        out.save_comparison_figure(
            last_measurement, last_gt, last_outputs, img_idx=img_idx
        )
    out.save_comparison_csv(comparison_rows)
    out.save_metrics_comparison_plots(dict(method_histories))
    logger.info("Comparison complete. Results in: %s", out.run_dir)


if __name__ == "__main__":
    main()
