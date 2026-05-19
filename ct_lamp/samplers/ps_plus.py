"""PS+ sampler from the official DPS repository, adapted to CT."""

from __future__ import annotations

import torch
from tqdm import tqdm

from ct_lamp.samplers.dps import DPSSampler


class PSPlusSampler(DPSSampler):
    """Posterior Sampling Plus with noise-averaged CT guidance."""

    def __init__(self, model, operator, cfg: dict) -> None:
        super().__init__(model, operator, cfg)
        method_cfg = cfg.get("ps_plus", {})
        if "num_steps" in method_cfg:
            self.num_steps = int(method_cfg["num_steps"])
        self.zeta = float(method_cfg.get("zeta", self.zeta))
        self.grad_clip = float(method_cfg.get("grad_clip", 1.0))
        self.smart_init = bool(method_cfg.get("smart_init", self.smart_init))
        self.num_sampling = int(method_cfg.get("num_sampling", 5))
        self.noise_std = float(method_cfg.get("noise_std", 0.05))

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
        progress = tqdm(list(enumerate(timesteps)), total=len(timesteps), desc="Sampling")
        x_est = x_t

        for step_idx, t_cur in progress:
            t_int = int(t_cur.item()) if torch.is_tensor(t_cur) else int(t_cur)
            with torch.enable_grad():
                x = x_t.detach().requires_grad_(True)
                eps, x0_hat = self._tweedie(x, t_int, grad=True)
                x0_eval = x0_hat.clamp(-1.0, 1.0)

                residual_vals: list[float] = []
                grad_norm_sum = 0.0
                grad = torch.zeros_like(x)
                for _ in range(self.num_sampling):
                    x0_noisy = (
                        x0_eval + self.noise_std * torch.randn_like(x0_eval)
                    ).clamp(-1.0, 1.0)
                    y_hat = self.operator.forward_autograd(x0_noisy)
                    residual = measurement - y_hat.float()
                    norm = torch.linalg.norm(
                        residual.reshape(residual.shape[0], -1),
                        dim=1,
                    ).sum()
                    grad_i = torch.autograd.grad(
                        outputs=norm,
                        inputs=x,
                        retain_graph=True,
                    )
                    grad = grad + grad_i[0]
                    residual_vals.append(torch.sqrt(residual.square().mean()).item())
                grad = grad / float(self.num_sampling)
                grad_norm_sum = torch.linalg.norm(grad)
                grad = self._clip_batch_vector_norm(grad, self.grad_clip)

            with torch.no_grad():
                step_out = scheduler.step(
                    model_output=eps.detach(),
                    timestep=t_int,
                    sample=x_t,
                    eta=0.0,
                )
                x_t = (step_out.prev_sample - self.zeta * grad).detach()
                x_est = x0_hat.detach().clamp(-1.0, 1.0)
                state["residual"] = sum(residual_vals) / max(len(residual_vals), 1)

            self._record_history(progress, history, step_idx, x_est, x_true, state)

        return x_est, history
