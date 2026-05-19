"""Faithful DPS sampler adapted to sparse-view CT."""

from __future__ import annotations

import torch
from tqdm import tqdm

from ct_lamp.samplers.base import CTSamplerBase


class DPSSampler(CTSamplerBase):
    """Diffusion Posterior Sampling with CT likelihood guidance.

    This follows the original DPS structure:
      1. draw a stochastic reverse sample with the diffusion posterior,
      2. predict x0,
      3. subtract a fixed-scale gradient of ||y - A(x0)|| with respect to x_t.

    The CT adaptation is only in the forward operator: we evaluate the
    measurement mismatch with the differentiable CT projector instead of the
    image degradations used in LAMP-Diff.
    """

    def __init__(self, model, operator, cfg: dict) -> None:
        super().__init__(model, operator, cfg)
        method_cfg = cfg.get("dps", {})
        if "num_steps" in method_cfg:
            self.num_steps = int(method_cfg["num_steps"])
        self.scale = float(method_cfg.get("scale", method_cfg.get("zeta", 0.3)))
        self.smart_init = bool(method_cfg.get("smart_init", False))

    def step(self, x_t, t_cur, t_prev, measurement, state):
        raise RuntimeError("DPSSampler uses a custom sample() implementation.")

    def _build_initial_sample(
        self,
        measurement: torch.Tensor,
        shape: tuple[int, int, int, int],
        first_t: int,
        x_init: torch.Tensor | None,
    ) -> torch.Tensor:
        return super()._build_initial_sample(
            measurement=measurement,
            shape=shape,
            first_t=first_t,
            x_init=x_init,
            smart_init=self.smart_init,
        )

    def sample(
        self,
        measurement: torch.Tensor,
        shape: tuple[int, int, int, int],
        x_true: torch.Tensor | None = None,
        x_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        scheduler = self._build_ddpm_scheduler(self.num_steps)
        timesteps = scheduler.timesteps
        first_t = int(timesteps[0].item()) if len(timesteps) > 0 else 0
        x_t = self._build_initial_sample(measurement, shape, first_t, x_init)

        history: list[dict] = []
        state: dict = {}
        x_est = x_t

        progress = tqdm(
            list(enumerate(timesteps)),
            total=len(timesteps),
            desc="Sampling",
        )
        for step_idx, t_cur in progress:
            t_int = int(t_cur.item()) if torch.is_tensor(t_cur) else int(t_cur)
            t_batch = torch.full(
                (shape[0],),
                t_int,
                device=self.device,
                dtype=torch.long,
            )

            with torch.enable_grad():
                x_prev = x_t.detach().requires_grad_(True)
                eps, x0_hat = self._tweedie(x_prev, t_int, grad=True)
                x0_eval = x0_hat.clamp(-1.0, 1.0)
                y_hat = self.operator.forward_autograd(x0_eval)
                residual = measurement - y_hat.float()
                norm = torch.linalg.norm(residual.reshape(residual.shape[0], -1), dim=1).sum()
                grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0].float()

            with torch.no_grad():
                step_out = scheduler.step(
                    model_output=eps.detach(),
                    timestep=t_int,
                    sample=x_prev.detach(),
                )
                x_t = step_out.prev_sample - self.scale * grad
                x_est = x0_eval.detach()
                state["residual"] = torch.sqrt(residual.square().mean()).item()

            self._record_history(progress, history, step_idx, x_est, x_true, state)

        return x_est.clamp(-1.0, 1.0), history
