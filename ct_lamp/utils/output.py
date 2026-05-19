"""Output directory management, CSV logging, and comparison figures."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml

logger = logging.getLogger(__name__)


PAPER_COLORS = [
    "#1b3a57",
    "#b33c2e",
    "#2f6b3b",
    "#c07a00",
    "#5b4b8a",
    "#7a3e65",
    "#007c91",
    "#6b6b6b",
    "#7a8f24",
    "#8a5a44",
]
PAPER_LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]
PAPER_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p"]


class OutputManager:
    """Manages the timestamped output directory for an experiment run.

    Directory structure::

        outputs/{TIMESTAMP}/
            config.yaml
            comparison.csv
            comparison_figure.png
            {method}/
                img_{i:04d}_ground_truth.png
                img_{i:04d}_measurement.png
                img_{i:04d}_solution.png
                img_{i:04d}_metrics.json
                metrics.csv
                metrics_curve.png
    """

    def __init__(self, base_dir: str | Path = "outputs") -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / ts
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.paper_dpi = 400
        logger.info("Output directory: %s", self.run_dir)

    @staticmethod
    def _apply_paper_rcparams() -> None:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.titlesize": 15,
            "axes.titleweight": "semibold",
            "axes.linewidth": 1.1,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "legend.fontsize": 11,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.fancybox": False,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "lines.linewidth": 2.4,
            "lines.markersize": 6.0,
            "savefig.pad_inches": 0.05,
        })

    def save_config(self, cfg: dict[str, Any]) -> None:
        """Write the full resolved config to config.yaml."""
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

    def method_dir(self, method: str) -> Path:
        """Return (and create) the per-method subdirectory."""
        d = self.run_dir / method
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_image(self, tensor, path: Path) -> None:
        """Save a tensor image using ct_lamp.utils.io."""
        from ct_lamp.utils.io import save_image as _save
        path.parent.mkdir(parents=True, exist_ok=True)
        _save(tensor, path)

    def save_measurement(self, tensor, path: Path) -> None:
        """Save a sinogram-like tensor with its physical range visible."""
        import numpy as np
        path.parent.mkdir(parents=True, exist_ok=True)

        arr = tensor.detach().cpu()
        if arr.dim() == 4:
            arr = arr.squeeze(0)
        if arr.dim() != 3 or arr.shape[0] != 1:
            raise ValueError(f"Expected measurement with shape (1,1,A,D), got {tuple(tensor.shape)}")
        arr = np.maximum(arr.squeeze(0).numpy(), 0.0)

        vmax = float(arr.max()) if arr.size else 1.0
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        im = ax.imshow(arr, cmap="gray", aspect="auto", vmin=0.0, vmax=max(vmax, 1e-8))
        ax.set_title(f"Measurement range: [{arr.min():.2f}, {arr.max():.2f}]")
        ax.set_xlabel("Detector")
        ax.set_ylabel("Angle")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(path, dpi=self.paper_dpi, bbox_inches="tight")
        plt.close(fig)

    def save_metrics_json(self, metrics: dict[str, Any], method: str, img_idx: int) -> None:
        """Save per-image metrics as JSON."""
        p = self.method_dir(method) / f"img_{img_idx:04d}_metrics.json"
        with open(p, "w") as f:
            json.dump(metrics, f, indent=2)

    def save_step_metrics_csv(self, history: list[dict], method: str) -> None:
        """Save step-by-step metrics to metrics.csv and plot metrics_curve.png."""
        if not history:
            return
        method_d = self.method_dir(method)
        csv_path = method_d / "metrics.csv"
        keys = list(history[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(history)

        # Plot PSNR and SSIM curves if present
        self._apply_paper_rcparams()
        steps = [row.get("step", i) for i, row in enumerate(history)]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
        if "psnr" in history[0]:
            axes[0].plot(steps, [r["psnr"] for r in history], color=PAPER_COLORS[0])
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("PSNR (dB)")
            axes[0].set_title("PSNR vs Step")
        if "ssim" in history[0]:
            axes[1].plot(steps, [r["ssim"] for r in history], color=PAPER_COLORS[1])
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("SSIM")
            axes[1].set_title("SSIM vs Step")
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(method_d / "metrics_curve.png", dpi=self.paper_dpi, bbox_inches="tight")
        plt.close(fig)
        plt.rcParams.update(plt.rcParamsDefault)

    def save_comparison_csv(self, results: list[dict[str, Any]]) -> None:
        """Save comparison CSV with rows: method, psnr, ssim, lpips, time_s."""
        csv_path = self.run_dir / "comparison.csv"
        if not results:
            return
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        logger.info("Comparison CSV written to %s", csv_path)

    def save_metrics_comparison_plots(
        self,
        method_histories: dict[str, list[list[dict]]],
        metrics: list[str] | None = None,
    ) -> None:
        """Save publication-quality per-metric plots comparing all methods over iterations.

        Creates one figure per metric (PSNR, SSIM, LPIPS), each with one curve per method.
        Curves are averaged across images when multiple images are present.

        Args:
            method_histories: dict mapping method name -> list of per-image histories.
                              Each per-image history is a list[dict] with keys
                              ``step``, ``psnr``, ``ssim``, ``lpips``.
            metrics: Metrics to plot. Defaults to ["psnr", "ssim", "lpips"].
        """
        import numpy as np

        if metrics is None:
            metrics = ["psnr", "ssim", "lpips"]

        self._apply_paper_rcparams()

        metric_labels = {
            "psnr": "PSNR (dB)",
            "ssim": "SSIM",
            "lpips": "LPIPS",
        }
        metric_titles = {
            "psnr": "PSNR vs. Iteration",
            "ssim": "SSIM vs. Iteration",
            "lpips": "LPIPS vs. Iteration",
        }

        for metric in metrics:
            # Skip this metric entirely if no method recorded it
            any_has_metric = any(
                any(h and metric in h[0] for h in img_histories)
                for img_histories in method_histories.values()
            )
            if not any_has_metric:
                logger.info("Skipping %s comparison plot: metric not in step history.", metric)
                continue

            fig, ax = plt.subplots(figsize=(7.0, 5.0))

            for idx, (method_name, img_histories) in enumerate(method_histories.items()):
                # Filter to images that actually recorded this metric
                valid = [h for h in img_histories if h and metric in h[0]]
                if not valid:
                    continue

                # Align by step index; take shortest history to avoid ragged averaging
                min_len = min(len(h) for h in valid)
                trimmed = [h[:min_len] for h in valid]

                steps = [row.get("step", i) for i, row in enumerate(trimmed[0])]
                series = np.asarray(
                    [[row[metric] for row in h] for h in trimmed],
                    dtype=np.float64,
                )
                values = np.nanmean(series, axis=0)
                std = np.nanstd(series, axis=0)

                color = PAPER_COLORS[idx % len(PAPER_COLORS)]
                ls = PAPER_LINESTYLES[idx % len(PAPER_LINESTYLES)]
                marker = PAPER_MARKERS[idx % len(PAPER_MARKERS)]
                # Use markers only every N steps to avoid clutter
                markevery = max(1, len(steps) // 7)
                lower = values - std
                upper = values + std
                if metric == "ssim":
                    lower = np.clip(lower, 0.0, 1.0)
                    upper = np.clip(upper, 0.0, 1.0)
                elif metric == "lpips":
                    lower = np.clip(lower, 0.0, None)
                ax.fill_between(
                    steps,
                    lower,
                    upper,
                    color=color,
                    alpha=0.18,
                    linewidth=0.0,
                )
                ax.plot(
                    steps, values,
                    color=color,
                    linestyle=ls,
                    marker=marker,
                    markevery=markevery,
                    markersize=5.5,
                    label=method_name,
                )

            ax.set_xlabel("Diffusion Step")
            ax.set_ylabel(metric_labels.get(metric, metric.upper()))
            ax.set_title(metric_titles.get(metric, metric.upper()))
            if metric == "ssim":
                ax.set_ylim(0.0, 1.0)
            elif metric == "lpips":
                ax.set_ylim(bottom=0.0)
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=min(4, max(1, len(method_histories))),
                edgecolor="#666666",
                columnspacing=1.2,
                handlelength=2.4,
            )
            ax.grid(True, which="major")
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))

            png_path = self.run_dir / f"metrics_comparison_{metric}.png"
            pdf_path = self.run_dir / f"metrics_comparison_{metric}.pdf"
            plt.savefig(png_path, dpi=self.paper_dpi, bbox_inches="tight")
            plt.savefig(pdf_path, bbox_inches="tight")
            plt.close(fig)
            logger.info("Metrics comparison plot saved: %s", png_path)

        # Reset rcParams to avoid bleeding into other plots
        plt.rcParams.update(plt.rcParamsDefault)

    def save_comparison_figure(
        self,
        measurement,
        ground_truth,
        method_outputs: dict[str, Any],
        img_idx: int = 0,
    ) -> None:
        """Save side-by-side comparison figure.

        Args:
            measurement: (1,3,H,W) tensor in [-1,1].
            ground_truth: (1,3,H,W) tensor in [-1,1] or None.
            method_outputs: dict mapping method name -> (1,3,H,W) tensor.
            img_idx: Image index for filename.
        """
        from ct_lamp.utils.io import measurement_to_pil, tensor_to_pil
        panels = []
        labels = []
        if measurement is not None:
            panels.append(measurement_to_pil(measurement))
            m = measurement.detach().cpu()
            labels.append(
                f"Measurement [{m.min().item():.1f}, {m.max().item():.1f}]"
            )
        if ground_truth is not None:
            panels.append(tensor_to_pil(ground_truth))
            labels.append("Ground Truth")
        for name, tensor in method_outputs.items():
            if tensor is not None:
                panels.append(tensor_to_pil(tensor))
                labels.append(name)

        n = len(panels)
        if n == 0:
            return
        self._apply_paper_rcparams()
        fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.2))
        if n == 1:
            axes = [axes]
        for ax, img, label in zip(axes, panels, labels):
            ax.imshow(img)
            ax.set_title(label, fontsize=12, fontweight="semibold")
            ax.axis("off")
        plt.tight_layout()
        out_path = self.run_dir / f"comparison_figure_img{img_idx:04d}.png"
        pdf_path = self.run_dir / f"comparison_figure_img{img_idx:04d}.pdf"
        plt.savefig(out_path, dpi=self.paper_dpi, bbox_inches="tight")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        plt.rcParams.update(plt.rcParamsDefault)
        logger.info("Comparison figure saved to %s", out_path)
