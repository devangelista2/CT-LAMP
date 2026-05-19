"""Projection-style score-SDE sampler adapted to CT."""

from __future__ import annotations

import torch
from tqdm import tqdm

from ct_lamp.samplers.base import CTSamplerBase


class ScoreSDESampler(CTSamplerBase):
    """Score-SDE predictor-corrector with CT projection consistency.

    This is adapted from the official score_inverse_problems projection samplers
    and the sparse-view CT score-SDE updates used in MCG_diffusion, but mapped
    to the local VP epsilon-prediction model.
    """

    def __init__(self, model, operator, cfg: dict) -> None:
        super().__init__(model, operator, cfg)
        method_cfg = cfg.get("score_sde", {})
        if "num_steps" in method_cfg:
            self.num_steps = int(method_cfg["num_steps"])
        self.corrector_steps = int(method_cfg.get("corrector_steps", 1))
        self.snr = float(method_cfg.get("snr", 0.15))
        self.guidance_scale = float(method_cfg.get("guidance_scale", 1.0))
        self.grad_clip = float(method_cfg.get("grad_clip", 2.0))
        self.smart_init = bool(method_cfg.get("smart_init", True))

    def step(self, x_t, t_cur, t_prev, measurement, state):
        raise RuntimeError("ScoreSDESampler uses a custom sample() implementation.")

    def _measurement_grad_x0(
        self,
        x0_eval: torch.Tensor,
        measurement: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Compute the CT data-fidelity gradient with respect to x0."""
        y_hat = self.operator.forward_autograd(x0_eval)
        residual = measurement - y_hat.float()
        norm = torch.linalg.norm(
            residual.reshape(residual.shape[0], -1),
            dim=1,
        ).sum()
        grad_x0 = torch.autograd.grad(outputs=norm, inputs=x0_eval)[0].float()
        residual_rms = torch.sqrt(residual.square().mean()).item()
        return grad_x0, residual_rms

    def _prior_corrector(self, x_t: torch.Tensor, t_int: int) -> torch.Tensor:
        """Annealed Langevin corrector using only the prior score."""
        sigma_t = self.ns.get_sigma(t_int).clamp(min=1e-8)
        for _ in range(self.corrector_steps):
            eps, _ = self._tweedie(x_t, t_int)
            prior_score = -eps / sigma_t
            noise = torch.randn_like(x_t)
            grad_norm = torch.linalg.norm(
                prior_score.reshape(prior_score.shape[0], -1), dim=-1
            ).mean()
            noise_norm = torch.linalg.norm(
                noise.reshape(noise.shape[0], -1), dim=-1
            ).mean()
            step_size = (
                (self.snr * noise_norm / grad_norm.clamp(min=1e-8)) ** 2 * 2.0
            )
            x_t = x_t + step_size * prior_score + torch.sqrt(2.0 * step_size) * noise
        return x_t

    def _project_x0(
        self,
        x_t: torch.Tensor,
        t_int: int,
        measurement: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Project the current x0 estimate toward the CT measurement set."""
        x = x_t.detach().requires_grad_(True)
        eps, x0_hat = self._tweedie(x, t_int, grad=True)
        x0_eval = x0_hat.clamp(-1.0, 1.0).requires_grad_(True)
        grad_x0, residual_rms = self._measurement_grad_x0(
            x0_eval=x0_eval,
            measurement=measurement,
        )
        grad_x0 = self._clip_batch_vector_norm(grad_x0, self.grad_clip)
        x_corr = (x0_eval - self.guidance_scale * grad_x0).clamp(-1.0, 1.0).detach()
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

            with torch.no_grad():
                x_t = self._prior_corrector(x_t, t_int)
            with torch.enable_grad():
                x_est, eps_refined, residual_rms = self._project_x0(
                    x_t=x_t,
                    t_int=t_int,
                    measurement=measurement,
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
