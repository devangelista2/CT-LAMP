#!/usr/bin/env python
"""Lightweight tuner for DDNM+ and CT-LAMP on Mayo sparse-view CT."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ct_lamp.data.datasets import build_dataset
from ct_lamp.metrics.metrics import compute_psnr, compute_ssim
from ct_lamp.models.loader import build_model
from ct_lamp.operators.sparse_view_ct import SparseViewCTProjector
from ct_lamp.utils.config import apply_overrides, load_config, resolve_device
from ct_lamp.utils.logging import setup_logging
from ct_lamp.utils.random import set_seed

logger = logging.getLogger(__name__)


SAMPLERS = {
    "ddnm_plus": "ct_lamp.samplers.ddnm_plus.DDNMPlusSampler",
    "ct_lamp": "ct_lamp.samplers.ct_lamp.CTLAMPSampler",
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


def _set_dotted(cfg: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    cur = cfg
    for part in parts[:-1]:
        next_value = cur.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cur[part] = next_value
        cur = next_value
    cur[parts[-1]] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune DDNM+ and CT-LAMP.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--method",
        required=True,
        choices=list(SAMPLERS),
        help="Method to tune.",
    )
    parser.add_argument(
        "--grid-json",
        required=True,
        help="JSON dict of parameter name -> list of values.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=2,
        help="Number of images from the dataset subset.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Dataset start index for the tuning subset.",
    )
    parser.add_argument(
        "--steps-override",
        type=int,
        default=None,
        help="If set, override method num_steps for fast coarse search.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many best rows to print at the end.",
    )
    parser.add_argument("overrides", nargs="*", help="Extra dotted config overrides.")
    return parser.parse_args()


def build_trial_cfg(base_cfg: dict, method: str, params: dict, steps_override: int | None) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for key, value in params.items():
        if "." in key:
            _set_dotted(cfg, key, value)
        else:
            cfg.setdefault(method, {})
            cfg[method][key] = value
    if steps_override is not None:
        cfg.setdefault(method, {})
        cfg[method]["num_steps"] = int(steps_override)
    return cfg


def objective(row: dict[str, float]) -> float:
    """Prefer SSIM first, then PSNR."""
    return 100.0 * row["ssim_mean"] + row["psnr_mean"]


def main() -> None:
    args = parse_args()
    grid = json.loads(args.grid_json)
    if not isinstance(grid, dict) or not grid:
        raise ValueError("--grid-json must be a non-empty JSON dict.")

    cfg = load_config(args.config)
    base_cfg = load_config(Path(__file__).parent.parent / "configs" / "base.yaml")
    _deep_update(base_cfg, cfg)
    cfg = apply_overrides(base_cfg, args.overrides)

    setup_logging(cfg.get("output", {}).get("log_level", "INFO"))
    device = resolve_device(cfg)
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    set_seed(seed)
    logger.info("Tuning method=%s on device=%s with seed=%d", args.method, device, seed)

    model_cfg = cfg.get("model", {})
    dtype = torch.float16 if model_cfg.get("dtype") == "float16" else torch.float32
    model = build_model(cfg, device=device, dtype=dtype)

    data_cfg = copy.deepcopy(cfg.get("data", {}))
    data_cfg["num_images"] = int(args.num_images)
    data_cfg["start_idx"] = int(args.start_idx)
    tune_cfg = copy.deepcopy(cfg)
    tune_cfg["data"] = data_cfg
    dataset = list(build_dataset(tune_cfg))

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

    noise_sigma = float(op_cfg.get("noise_sigma", 0.01))
    prepared = []
    for img_idx, x_true in enumerate(dataset):
        x_true = x_true.to(device)
        with torch.no_grad():
            y = operator.forward(x_true)
            noise_rng = torch.Generator(device=device)
            noise_rng.manual_seed(seed + int(args.start_idx) + img_idx)
            noise = torch.randn(y.shape, generator=noise_rng, device=device) * noise_sigma
            y_noisy = (y + noise).clamp(min=0.0)
        prepared.append((x_true, y_noisy))

    sampler_class = import_sampler(SAMPLERS[args.method])
    method_grid_keys = list(grid.keys())
    method_grid_values = [grid[k] for k in method_grid_keys]
    combos = list(itertools.product(*method_grid_values))
    logger.info("Testing %d parameter combinations.", len(combos))

    rows: list[dict[str, float | str]] = []
    for trial_idx, combo in enumerate(combos, start=1):
        params = dict(zip(method_grid_keys, combo))
        trial_cfg = build_trial_cfg(cfg, args.method, params, args.steps_override)
        sampler = sampler_class(model, operator, trial_cfg)

        psnrs: list[float] = []
        ssims: list[float] = []
        t0 = time.time()
        for x_true, measurement in prepared:
            b, c, h, w = x_true.shape
            set_seed(seed)
            x_hat, _ = sampler.sample(
                measurement=measurement,
                shape=(b, c, h, w),
                x_true=None,
            )
            psnrs.append(compute_psnr(x_hat, x_true))
            ssims.append(compute_ssim(x_hat, x_true))
        elapsed = time.time() - t0

        row: dict[str, float | str] = {
            "method": args.method,
            "trial": trial_idx,
            "psnr_mean": sum(psnrs) / len(psnrs),
            "ssim_mean": sum(ssims) / len(ssims),
            "time_s": elapsed,
            "score": 0.0,
        }
        for k, v in params.items():
            row[k] = v
        row["score"] = objective(row)  # type: ignore[arg-type]
        rows.append(row)
        logger.info(
            "[%d/%d] %s -> PSNR=%.3f SSIM=%.4f time=%.1fs",
            trial_idx,
            len(combos),
            params,
            row["psnr_mean"],
            row["ssim_mean"],
            elapsed,
        )

    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    out_path = Path("outputs") / f"tune_{args.method}_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Top %d results:", args.top_k)
    for row in rows[: args.top_k]:
        logger.info("%s", row)
    logger.info("Wrote summary CSV to %s", out_path)


if __name__ == "__main__":
    main()
