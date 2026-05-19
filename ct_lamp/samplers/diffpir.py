"""DiffPIR sampler adapted to CT physical-space measurements."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from ct_lamp.samplers.base import PosteriorSamplerBase


class DiffPIRSampler(PosteriorSamplerBase):
    """DiffPIR with a CT-domain proximal data-fidelity step."""

    def __init__(self, model, operator, cfg: dict) -> None:
        super().__init__(model, operator, cfg)
        method_cfg = cfg.get("diffpir", {})
        if "num_steps" in method_cfg:
            self.num_steps = int(method_cfg["num_steps"])
        self.lambda_data = float(method_cfg.get("lambda_data", 7.0))
        self.zeta = float(method_cfg.get("zeta", 0.0))
        self.cg_iters = int(method_cfg.get("cg_iters", 16))
        self.cg_tol = float(method_cfg.get("cg_tol", 1.0e-5))
        self.smart_init = bool(method_cfg.get("smart_init", True))
        self.noise_sigma = float(cfg.get("operator", {}).get("noise_sigma", 0.01))

    def _build_timesteps(self) -> list[int]:
        """Build the short sqrt-spaced schedule used by DiffPIR."""
        total_steps = int(self.ns.T)
        if self.num_steps <= 1:
            return [total_steps - 1]
        grid = np.linspace(np.sqrt(total_steps - 1), 0.0, self.num_steps, dtype=np.float64)
        timesteps = np.round(grid * grid).astype(np.int64).tolist()
        return sorted({int(t) for t in timesteps}, reverse=True)

    def _apply_prox_matrix(
        self,
        x_phys: torch.Tensor,
        rho_t: torch.Tensor,
        sigma_y2: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the linear system matrix of the L2 proximal step."""
        return self.operator.normal_physics(x_phys) / sigma_y2 + rho_t * x_phys

    def _cg_solve(
        self,
        rhs: torch.Tensor,
        rho_t: torch.Tensor,
        sigma_y2: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """Solve the DiffPIR proximal system with conjugate gradient."""
        x = x0.clone()
        r = rhs - self._apply_prox_matrix(x, rho_t=rho_t, sigma_y2=sigma_y2)
        p = r.clone()
        rr = torch.sum(r * r)
        if rr.item() <= 0:
            return x

        for _ in range(self.cg_iters):
            ap = self._apply_prox_matrix(p, rho_t=rho_t, sigma_y2=sigma_y2)
            denom = torch.sum(p * ap).clamp(min=1e-12)
            alpha = rr / denom
            x = x + alpha * p
            r = r - alpha * ap
            rr_new = torch.sum(r * r)
            if torch.sqrt(rr_new / rhs.numel()) < self.cg_tol:
                break
            beta = rr_new / rr.clamp(min=1e-12)
            p = r + beta * p
            rr = rr_new
        return x

    def _prox_step(
        self,
        x0_hat: torch.Tensor,
        measurement: torch.Tensor,
        t_cur: int,
    ) -> tuple[torch.Tensor, float]:
        """Apply the official DiffPIR-style L2 proximal step in CT units."""
        x0_phys = self.operator.to_physics(x0_hat)
        alpha_t = self.ns.get_alpha(t_cur)
        sigma_t = self.ns.get_sigma(t_cur)
        sigma_k = (sigma_t / alpha_t.clamp(min=1e-8)).clamp(min=1e-6)
        rho_t = torch.as_tensor(
            self.lambda_data,
            device=x0_hat.device,
            dtype=x0_hat.dtype,
        ) / (sigma_k * sigma_k)
        sigma_y2 = torch.as_tensor(
            max(self.noise_sigma * self.noise_sigma, 1e-8),
            device=x0_hat.device,
            dtype=x0_hat.dtype,
        )

        rhs = self.operator.adjoint_physics(measurement) / sigma_y2 + rho_t * x0_phys
        x_prox_phys = self._cg_solve(
            rhs=rhs,
            rho_t=rho_t,
            sigma_y2=sigma_y2,
            x0=x0_phys,
        ).clamp(0.0, 1.0)
        residual = measurement - self.operator.forward_physics(x_prox_phys)
        return self.operator.to_diffusion(x_prox_phys), torch.sqrt(
            torch.mean(residual * residual)
        ).item()

    def compute_d_cur(
        self,
        x0_hat: torch.Tensor,
        measurement: torch.Tensor,
        t_cur: int,
        state: dict,
    ) -> tuple[torch.Tensor, float]:
        return self._prox_step(
            x0_hat=x0_hat,
            measurement=measurement,
            t_cur=t_cur,
        )

    def sample(
        self,
        measurement: torch.Tensor,
        shape: tuple[int, int, int, int],
        x_true: torch.Tensor | None = None,
        x_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        timesteps = self._build_timesteps()
        first_t = timesteps[0] if timesteps else 0
        if x_init is not None:
            x_t = x_init.to(self.device)
        elif self.smart_init:
            x_fbp = self.operator.fbp(measurement)
            noise = torch.randn_like(x_fbp)
            init_timesteps = torch.full((shape[0],), first_t, device=self.device, dtype=torch.long)
            x_t = self.model.scheduler.add_noise(x_fbp, noise, init_timesteps)
        else:
            x_t = torch.randn(shape, device=self.device)

        history: list[dict] = []
        state: dict = {}
        step_pairs = list(zip(timesteps[:-1], timesteps[1:]))
        if timesteps and timesteps[-1] != 0:
            step_pairs.append((timesteps[-1], 0))
        progress = tqdm(list(enumerate(step_pairs)), total=len(step_pairs), desc="Sampling")
        x_est = x_t

        for step_idx, (t_cur, t_prev) in progress:
            with torch.no_grad():
                eps, x0_hat = self._tweedie(x_t, t_cur)
                x_est, residual_norm = self.compute_d_cur(
                    x0_hat=x0_hat,
                    measurement=measurement,
                    t_cur=t_cur,
                    state=state,
                )
                alpha_prev = self.ns.get_alpha(t_prev)
                sigma_prev = self.ns.get_sigma(t_prev)
                noise = (
                    torch.randn_like(x_t)
                    if (self.zeta > 0.0 and t_prev > 0)
                    else torch.zeros_like(x_t)
                )
                mixed_eps = (
                    torch.sqrt(torch.as_tensor(1.0 - self.zeta, device=x_t.device, dtype=x_t.dtype))
                    * eps
                    + torch.sqrt(torch.as_tensor(self.zeta, device=x_t.device, dtype=x_t.dtype))
                    * noise
                )
                x_t = alpha_prev * x_est + sigma_prev * mixed_eps
                state["residual"] = residual_norm

            self._record_history(progress, history, step_idx, x_est, x_true, state)

        return x_est, history
