"""MCG-style manifold constrained posterior sampler adapted to CT."""

from __future__ import annotations

import torch
from tqdm import tqdm

from ct_lamp.samplers.score_sde import ScoreSDESampler


class MCGSampler(ScoreSDESampler):
    """Projection-style score-SDE sampler with an MCG manifold term."""

    def __init__(self, model, operator, cfg: dict) -> None:
        super().__init__(model, operator, cfg)
        method_cfg = cfg.get("mcg", {})
        if "num_steps" in method_cfg:
            self.num_steps = int(method_cfg["num_steps"])
        self.corrector_steps = int(method_cfg.get("corrector_steps", 1))
        self.snr = float(method_cfg.get("snr", 0.15))
        self.lambda_start = float(method_cfg.get("kaczmarz_start", 1.0))
        self.lambda_end = float(method_cfg.get("kaczmarz_end", 0.6))
        self.manifold_weight = float(method_cfg.get("manifold_weight", 0.1))
        self.grad_clip = float(method_cfg.get("grad_clip", 2.0))
        self.smart_init = bool(method_cfg.get("smart_init", True))

    def step(self, x_t, t_cur, t_prev, measurement, state):
        raise RuntimeError("MCGSampler uses a custom sample() implementation.")

    def _project_x0(
        self,
        x_t: torch.Tensor,
        t_int: int,
        measurement: torch.Tensor,
        lambda_t: float,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Project with a Kaczmarz term and an MCG manifold penalty."""
        x = x_t.detach().requires_grad_(True)
        eps, x0_hat = self._tweedie(x, t_int, grad=True)
        x0_eval = x0_hat.clamp(-1.0, 1.0).requires_grad_(True)
        grad_x0, residual_rms = self._measurement_grad_x0(
            x0_eval=x0_eval,
            measurement=measurement,
        )
        grad_x0 = self._clip_batch_vector_norm(grad_x0, self.grad_clip)
        x_grad = (x0_eval - lambda_t * grad_x0).clamp(-1.0, 1.0)

        x0_phys = self.operator.to_physics(x_grad.detach())
        x_null_phys = x0_phys - self.operator.range_project_raw_physics(x0_phys)
        x_corr_phys = (
            x0_phys - self.manifold_weight * x_null_phys
        ).clamp(0.0, 1.0)
        x_corr = self.operator.to_diffusion(x_corr_phys)
        residual_after = measurement - self.operator.forward_physics(x_corr_phys)
        residual_rms = torch.sqrt(torch.mean(residual_after * residual_after)).item()
        return x_corr, eps.detach(), residual_rms

    def sample(
        self,
        measurement: torch.Tensor,
        shape: tuple[int, int, int, int],
        x_true: torch.Tensor | None = None,
        x_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        scheduler = self._build_ddim_scheduler(self.num_steps)
        timesteps = scheduler.timesteps
        first_t = int(timesteps[0].item()) if len(timesteps) > 0 else 0
        x_t = self._build_initial_sample(
            measurement=measurement,
            shape=shape,
            first_t=first_t,
            x_init=x_init,
            smart_init=self.smart_init,
        )

        history: list[dict] = []
        state: dict = {}
        progress = tqdm(list(enumerate(timesteps)), total=len(timesteps), desc="Sampling")
        x_est = x_t

        for step_idx, t_cur in progress:
            t_int = int(t_cur.item()) if torch.is_tensor(t_cur) else int(t_cur)
            frac = step_idx / max(len(timesteps) - 1, 1)
            lambda_t = self.lambda_start + frac * (self.lambda_end - self.lambda_start)

            with torch.no_grad():
                x_t = self._prior_corrector(x_t, t_int)
            with torch.enable_grad():
                x_est, eps_refined, residual_rms = self._project_x0(
                    x_t=x_t,
                    t_int=t_int,
                    measurement=measurement,
                    lambda_t=lambda_t,
                )
            with torch.no_grad():
                alpha_t = self.ns.get_alpha(t_int)
                sigma_t = self.ns.get_sigma(t_int).clamp(min=1e-8)
                eps_consistent = (x_t - alpha_t * x_est) / sigma_t
                x_t = scheduler.step(
                    model_output=eps_consistent,
                    timestep=t_int,
                    sample=x_t,
                    eta=0.0,
                ).prev_sample
                state["residual"] = residual_rms

            self._record_history(progress, history, step_idx, x_est, x_true, state)

        return x_est, history
