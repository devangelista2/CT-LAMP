"""Metric wrappers: PSNR, SSIM (RGB-aware), lazy LPIPS."""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def compute_psnr(x_hat: torch.Tensor, x_true: torch.Tensor) -> float:
    """Compute PSNR between two (B,C,H,W) tensors in [-1,1].

    Delegates to IPPy's PSNR which averages over all elements.
    Falls back to a manual computation if IPPy is unavailable.
    """
    try:
        from IPPy.utilities.metrics import PSNR as ippyPSNR
        # IPPy PSNR expects numpy arrays in [0,1]; convert
        x_hat_np = ((x_hat.detach().cpu().clamp(-1, 1) + 1) / 2).numpy()
        x_true_np = ((x_true.detach().cpu().clamp(-1, 1) + 1) / 2).numpy()
        return float(ippyPSNR(x_hat_np, x_true_np))
    except Exception:
        mse = torch.mean((x_hat - x_true) ** 2).item()
        if mse == 0:
            return float("inf")
        # data range is 2.0 (from -1 to 1)
        return float(10 * torch.log10(torch.tensor(4.0 / mse)).item())


def compute_ssim(x_hat: torch.Tensor, x_true: torch.Tensor) -> float:
    """Compute mean SSIM over channels for (B,C,H,W) tensors in [-1,1].

    Calls IPPy's SSIM per channel and averages.
    Falls back to skimage.metrics.structural_similarity if IPPy is unavailable.
    """
    try:
        from IPPy.utilities.metrics import SSIM as ippySSIM
        # IPPy SSIM expects numpy arrays in [0,1]
        x_hat_np = ((x_hat.detach().cpu().clamp(-1, 1) + 1) / 2).numpy()
        x_true_np = ((x_true.detach().cpu().clamp(-1, 1) + 1) / 2).numpy()
        # Average over batch and channel
        return float(ippySSIM(x_hat_np, x_true_np))
    except Exception:
        from skimage.metrics import structural_similarity
        x_h = ((x_hat.detach().cpu().clamp(-1, 1) + 1) / 2).numpy()
        x_t = ((x_true.detach().cpu().clamp(-1, 1) + 1) / 2).numpy()
        ssim_vals = []
        for b in range(x_h.shape[0]):
            for c in range(x_h.shape[1]):
                ssim_vals.append(
                    structural_similarity(x_h[b, c], x_t[b, c], data_range=1.0)
                )
        return float(sum(ssim_vals) / len(ssim_vals)) if ssim_vals else 0.0


_lpips_model: Any = None
_lpips_unavailable_logged = False


def compute_lpips(x_hat: torch.Tensor, x_true: torch.Tensor) -> float:
    """Compute LPIPS perceptual distance (lazy import).

    Args:
        x_hat: (B,3,H,W) tensor in [-1,1].
        x_true: (B,3,H,W) tensor in [-1,1].

    Returns:
        Mean LPIPS value as a float (lower is better).
    """
    global _lpips_model, _lpips_unavailable_logged
    try:
        if _lpips_model is None:
            import lpips
            _lpips_model = lpips.LPIPS(net="alex")
        device = x_hat.device
        _lpips_model = _lpips_model.to(device)
        if x_hat.shape[1] == 1:
            x_hat = x_hat.repeat(1, 3, 1, 1)
            x_true = x_true.repeat(1, 3, 1, 1)
        elif x_hat.shape[1] != 3:
            logger.warning("LPIPS supports only 1 or 3 channels, got %d", x_hat.shape[1])
            return float("nan")
        with torch.no_grad():
            val = _lpips_model(x_hat, x_true)
        return float(val.mean().item())
    except Exception as e:
        if not _lpips_unavailable_logged:
            logger.warning("LPIPS computation failed: %s", e)
            _lpips_unavailable_logged = True
        return float("nan")
