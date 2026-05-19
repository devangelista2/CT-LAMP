"""DDNM+ sampler for sparse-view CT."""

from __future__ import annotations

import torch

from ct_lamp.samplers.base import PosteriorSamplerBase


class DDNMPlusSampler(PosteriorSamplerBase):
    """First-order DDNM+ with noise-adaptive data consistency."""

    def __init__(self, model, operator, cfg: dict) -> None:
        super().__init__(model, operator, cfg)
        method_cfg = cfg.get("ddnm_plus", {})
        if "num_steps" in method_cfg:
            self.num_steps = int(method_cfg["num_steps"])

        noise_sigma = float(cfg.get("operator", {}).get("noise_sigma", 0.01))
        eta_scale = float(method_cfg.get("eta_scale", 1.0))
        self.eta2 = (eta_scale * noise_sigma) ** 2
        self.cg_iters = int(method_cfg.get("cg_iters", 12))
        self.cg_tol = float(method_cfg.get("cg_tol", 1e-5))
        self.cg_early_stop = bool(method_cfg.get("cg_early_stop", False))
        self.smart_init = bool(method_cfg.get("smart_init", True))
        self.consistency_solver = str(method_cfg.get("consistency_solver", "cg")).lower()
        if self.consistency_solver not in {"cg", "fbp"}:
            raise ValueError(
                "ddnm_plus.consistency_solver must be 'cg' or 'fbp', "
                f"got {self.consistency_solver!r}."
            )
        self.fbp_relaxation = float(method_cfg.get("fbp_relaxation", 1.0))

    def _normal_y(
        self,
        y: torch.Tensor,
        eta2_alpha2: torch.Tensor,
        sigma2: torch.Tensor,
    ) -> torch.Tensor:
        """Apply (eta^2 alpha^2 A A^T + sigma^2 I) to y."""
        return (
            eta2_alpha2 * self.operator.forward_physics(self.operator.adjoint_physics(y))
            + sigma2 * y
        )

    def _cg_solve(
        self,
        rhs: torch.Tensor,
        eta2_alpha2: torch.Tensor,
        sigma2: torch.Tensor,
    ) -> torch.Tensor:
        """Solve (eta^2 alpha^2 A A^T + sigma^2 I)u = rhs by conjugate gradient."""
        x = torch.zeros_like(rhs)
        r = rhs.clone()
        p = r.clone()
        rr = torch.sum(r * r)
        if rr.item() <= 0:
            return x

        for _ in range(self.cg_iters):
            ap = self._normal_y(p, eta2_alpha2, sigma2)
            denom = torch.sum(p * ap).clamp(min=1e-12)
            alpha = rr / denom
            x = x + alpha * p
            r = r - alpha * ap
            rr_new = torch.sum(r * r)
            if self.cg_early_stop and torch.sqrt(rr_new / rhs.numel()) < self.cg_tol:
                break
            beta = rr_new / rr.clamp(min=1e-12)
            p = r + beta * p
            rr = rr_new
        return x

    def correct_x0(
        self,
        x0_hat: torch.Tensor,
        measurement: torch.Tensor,
        alpha_s: torch.Tensor,
        sigma_s: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Apply DDNM+ correction to x0_hat.

        Solves:
            x_corr_phys = argmin_x ||K x - y||^2 / (2 sigma_s^2)
                               + ||x - x0_phys||^2 / (2 (eta alpha_s / 2)^2)

        The diffusion model predicts `x0_hat` in [-1, 1], but the CT forward model
        is only physically meaningful on [0, 1]. We therefore perform data
        consistency in physical image units and map the corrected result back to
        diffusion units afterward.
        """
        x0_phys = self.operator.to_physics(x0_hat)
        residual = measurement - self.operator.forward_physics(x0_phys)
        eta2_alpha2 = (
            0.25 * torch.as_tensor(self.eta2, device=x0_hat.device) * alpha_s * alpha_s
        )
        sigma2 = (sigma_s * sigma_s).clamp(min=1e-8)

        if self.consistency_solver == "fbp":
            # Approximate the exact DDNM+ proximal step with a pseudoinverse
            # correction. For CT, raw FBP behaves much more like A_dagger than
            # A^T because FBP(Kx) stays on the image scale while K^T Kx does not.
            gain = self.fbp_relaxation * eta2_alpha2 / (eta2_alpha2 + sigma2)
            x_corr_phys = x0_phys + gain * self.operator.fbp_raw_physics(residual)
        else:
            u = self._cg_solve(residual, eta2_alpha2=eta2_alpha2, sigma2=sigma2)
            x_corr_phys = x0_phys + eta2_alpha2 * self.operator.adjoint_physics(u)
        x_corr_phys = x_corr_phys.clamp(0.0, 1.0)
        x_corr = self.operator.to_diffusion(x_corr_phys)

        residual_after = measurement - self.operator.forward_physics(x_corr_phys)
        residual_norm = torch.sqrt(torch.mean(residual_after * residual_after)).item()
        return x_corr, residual_norm

    def compute_d_cur(
        self,
        x0_hat: torch.Tensor,
        measurement: torch.Tensor,
        t_cur: int,
        state: dict,
    ) -> tuple[torch.Tensor, float]:
        alpha_s = self.ns.get_alpha(t_cur)
        sigma_s = self.ns.get_sigma(t_cur)
        return self.correct_x0(
            x0_hat=x0_hat,
            measurement=measurement,
            alpha_s=alpha_s,
            sigma_s=sigma_s,
        )

    def sample(
        self,
        measurement: torch.Tensor,
        shape: tuple[int, int, int, int],
        x_true: torch.Tensor | None = None,
        x_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Run the full reverse process with optional FBP-based smart init."""
        with torch.inference_mode():
            if x_init is not None:
                x_t = x_init.to(self.device)
            elif self.smart_init:
                timesteps = self.ns.get_timestep_sequence(self.num_steps)
                first_t = int(timesteps[0]) if timesteps else 0
                x_fbp = self.operator.fbp(measurement)
                noise = torch.randn_like(x_fbp)
                init_timesteps = torch.full(
                    (shape[0],),
                    first_t,
                    device=self.device,
                    dtype=torch.long,
                )
                x_t = self.model.scheduler.add_noise(x_fbp, noise, init_timesteps)
            else:
                x_t = torch.randn(shape, device=self.device)

            step_pairs = self._build_step_pairs()
            history: list[dict] = []
            state: dict = {}

            from tqdm import tqdm

            progress = tqdm(
                list(enumerate(step_pairs)),
                total=len(step_pairs),
                desc="Sampling",
            )
            for step_idx, (t_cur, t_prev) in progress:
                x_t, state = self.step(x_t, t_cur, t_prev, measurement, state)
                self._record_history(progress, history, step_idx, x_t, x_true, state)

        return x_t, history
