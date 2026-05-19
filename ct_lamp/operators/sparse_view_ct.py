"""Sparse-view CT operator backed by IPPy's ASTRA CTProjector."""

from __future__ import annotations

import math

import numpy as np
import torch

from IPPy.operators import CTProjector


class SparseViewCTProjector:
    """Thin adapter over IPPy's CTProjector for CT-LAMP samplers.

    Input image tensors are expected as `(B, C, H, W)` and sinograms as
    `(B, C, num_angles, det_count)`. Current CT-LAMP setup uses `C=1`.
    """

    def __init__(
        self,
        image_size: int,
        num_angles: int,
        det_count: int | None = None,
        start_angle: float = 0.0,
        end_angle: float = math.pi,
        device: str = "cpu",
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be > 0.")
        if num_angles <= 0:
            raise ValueError("num_angles must be > 0.")
        if det_count is not None and det_count <= 0:
            raise ValueError("det_count must be > 0 when provided.")

        self.image_size = int(image_size)
        self.num_angles = int(num_angles)
        self.det_count = int(det_count or image_size)
        self.start_angle = float(start_angle)
        self.end_angle = float(end_angle)
        self.device = device

        # Use endpoint=False to avoid duplicating 0 and pi views.
        self.angles = np.linspace(
            self.start_angle,
            self.end_angle,
            self.num_angles,
            endpoint=False,
            dtype=np.float32,
        )
        self._op = CTProjector(
            img_shape=(self.image_size, self.image_size),
            angles=self.angles,
            det_size=self.det_count,
            geometry="parallel",
            force_cpu=device == "cpu",
        )

    def _validate_image(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (B,C,H,W), got {tuple(x.shape)}.")
        _, c, h, w = x.shape
        if c != 1:
            raise ValueError(
                f"IPPy CTProjector in CT-LAMP currently supports C=1, got C={c}."
            )
        if h != self.image_size or w != self.image_size:
            raise ValueError(
                f"Expected image size ({self.image_size}, {self.image_size}), got ({h}, {w})."
            )

    def _validate_sinogram(self, y: torch.Tensor) -> None:
        if y.ndim != 4:
            raise ValueError(f"Expected y with shape (B,C,A,D), got {tuple(y.shape)}.")
        _, c, a, d = y.shape
        if c != 1:
            raise ValueError(
                f"IPPy CTProjector in CT-LAMP currently supports C=1, got C={c}."
            )
        if a != self.num_angles or d != self.det_count:
            raise ValueError(
                f"Expected sinogram shape (*,1,{self.num_angles},{self.det_count}), got {tuple(y.shape)}."
            )

    @staticmethod
    def to_physics(x: torch.Tensor) -> torch.Tensor:
        """Map diffusion-domain image from [-1, 1] to physical [0, 1]."""
        return ((x + 1.0) * 0.5).clamp(0.0, 1.0)

    @staticmethod
    def to_physics_unclamped(x: torch.Tensor) -> torch.Tensor:
        """Map diffusion-domain image from [-1, 1] to physical units without clamping.

        Use this only when the caller already enforced the valid image range and we
        want to preserve gradients at the boundaries.
        """
        return (x + 1.0) * 0.5

    @staticmethod
    def to_diffusion(x_phys: torch.Tensor) -> torch.Tensor:
        """Map physical-domain image from [0, 1] to diffusion [-1, 1]."""
        return x_phys.clamp(0.0, 1.0) * 2.0 - 1.0

    def forward_physics(self, x_phys: torch.Tensor) -> torch.Tensor:
        """Apply forward projection to a physical-domain image in [0, 1]."""
        self._validate_image(x_phys)
        with torch.no_grad():
            sinos = []
            for i in range(x_phys.shape[0]):
                sinos.append(self._op._matvec(x_phys[i : i + 1]))
            return torch.cat(sinos, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward projection to a diffusion-domain image in [-1, 1]."""
        x_phys = self.to_physics(x)
        return self.forward_physics(x_phys)

    def forward_autograd(self, x: torch.Tensor) -> torch.Tensor:
        """Apply differentiable forward projection to a diffusion-domain image."""
        x_phys = self.to_physics_unclamped(x)
        self._validate_image(x_phys)
        return self._op(x_phys)

    def adjoint_physics(self, y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint in physical image units."""
        self._validate_sinogram(y)
        with torch.no_grad():
            recons = []
            for i in range(y.shape[0]):
                recons.append(self._op._adjoint(y[i : i + 1]))
            return torch.cat(recons, dim=0)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint and return the result in diffusion image units."""
        return self.to_diffusion(self.adjoint_physics(y))

    def fbp_physics(self, y: torch.Tensor) -> torch.Tensor:
        """Filtered backprojection in physical image units."""
        self._validate_sinogram(y)
        with torch.no_grad():
            recon = self._op.FBP(y).clamp(0.0, 1.0)
        return recon

    def fbp_raw_physics(self, y: torch.Tensor) -> torch.Tensor:
        """Filtered backprojection without output clipping.

        This is used inside iterative samplers where FBP acts as a pseudoinverse
        or range-space operator and signed intermediate values are meaningful.
        """
        self._validate_sinogram(y)
        with torch.no_grad():
            recon = self._op.FBP(y)
        return recon

    def fbp(self, y: torch.Tensor) -> torch.Tensor:
        """Filtered backprojection returned in diffusion image units."""
        return self.to_diffusion(self.fbp_physics(y))

    def range_project_raw_physics(self, x_phys: torch.Tensor) -> torch.Tensor:
        """Approximate range projection A_dagger A x in physical image units."""
        return self.fbp_raw_physics(self.forward_physics(x_phys))

    def sirt_physics(self, y: torch.Tensor, num_iters: int = 50) -> torch.Tensor:
        """SIRT reconstruction in physical image units."""
        self._validate_sinogram(y)
        with torch.no_grad():
            recon = self._op.SIRT(y, num_iters=num_iters).clamp(0.0, 1.0)
        return recon

    def sirt(self, y: torch.Tensor, num_iters: int = 50) -> torch.Tensor:
        """SIRT reconstruction returned in diffusion image units."""
        return self.to_diffusion(self.sirt_physics(y, num_iters=num_iters))

    def normal_physics(self, x_phys: torch.Tensor) -> torch.Tensor:
        """Apply the physical normal operator K^T K to an image in [0, 1]."""
        return self.adjoint_physics(self.forward_physics(x_phys))

    def normal(self, x: torch.Tensor) -> torch.Tensor:
        """Apply K^T K after mapping a diffusion-domain image into physical space."""
        return self.normal_physics(self.to_physics(x))
