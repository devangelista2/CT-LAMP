"""Common building blocks for CT-LAMP samplers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from ct_lamp.diffusion.schedulers import DDIMScheduler, DDPMScheduler
from ct_lamp.metrics.metrics import compute_lpips, compute_psnr, compute_ssim


def exp_integrals(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact moments for DPM-Solver++ notation.

    A0 = integral_0^h exp(-u) du
    A1 = integral_0^h u exp(-u) du
    A2 = integral_0^h u^2 exp(-u) du
    """
    exp_minus_h = torch.exp(-h)
    a0 = 1.0 - exp_minus_h
    a1 = 1.0 - (1.0 + h) * exp_minus_h
    a2 = 2.0 - (2.0 + 2.0 * h + h * h) * exp_minus_h
    return a0, a1, a2


class CTSamplerBase(ABC):
    """Base class shared by DDNM+ and CT-LAMP multistep variants."""

    def __init__(self, model, operator, cfg: dict) -> None:
        self.model = model
        self.operator = operator
        self.cfg = cfg
        self.ns = model.noise_schedule
        self.device = model.device
        sampling_cfg = cfg.get("sampling", {})
        self.num_steps = int(sampling_cfg.get("num_steps", 50))
        self.clip_x0 = bool(sampling_cfg.get("clip_x0", True))

    @abstractmethod
    def step(
        self,
        x_t: torch.Tensor,
        t_cur: int,
        t_prev: int,
        measurement: torch.Tensor,
        state: dict,
    ) -> tuple[torch.Tensor, dict]:
        """Run one reverse step."""
        raise NotImplementedError

    def _tweedie(
        self,
        x_t: torch.Tensor,
        t: int,
        grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute epsilon and x0_hat from one UNet evaluation."""
        alpha_t = self.ns.get_alpha(t)
        sigma_t = self.ns.get_sigma(t)
        t_tensor = torch.full(
            (x_t.shape[0],),
            fill_value=t,
            device=self.device,
            dtype=torch.long,
        )
        eps = self.model.unet_forward(x_t, t_tensor, grad=grad)
        x0_hat = (x_t - sigma_t * eps) / alpha_t
        if self.clip_x0:
            x0_hat = x0_hat.clamp(-1.0, 1.0)
        return eps, x0_hat

    def _build_step_pairs(self) -> list[tuple[int, int]]:
        """Build reverse-time timestep pairs `(t_cur, t_prev)`."""
        timesteps = self.ns.get_timestep_sequence(self.num_steps)
        step_pairs = list(zip(timesteps[:-1], timesteps[1:]))
        if timesteps:
            step_pairs.append((timesteps[-1], 0))
        return [(int(t_cur), int(t_prev)) for t_cur, t_prev in step_pairs]

    def _record_history(
        self,
        progress: tqdm,
        history: list[dict],
        step_idx: int,
        x_est: torch.Tensor,
        x_true: torch.Tensor | None,
        state: dict,
    ) -> None:
        """Record step metrics and update the tqdm postfix."""
        if x_true is None:
            return
        psnr = compute_psnr(x_est, x_true)
        ssim = compute_ssim(x_est, x_true)
        lpips = compute_lpips(x_est, x_true)
        progress.set_postfix(
            psnr=f"{psnr:.2f}",
            ssim=f"{ssim:.4f}",
            lpips="nan" if lpips != lpips else f"{lpips:.4f}",
        )
        history.append(
            {
                "step": step_idx,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips,
                "residual": float(state.get("residual", float("nan"))),
            }
        )

    def _build_ddim_scheduler(self, num_steps: int) -> DDIMScheduler:
        """Build a DDIM scheduler aligned with the current diffusion config."""
        scheduler_cfg = self.cfg.get("scheduler", {})
        scheduler = DDIMScheduler(
            num_train_timesteps=int(scheduler_cfg.get("num_train_timesteps", 1000)),
            beta_start=float(scheduler_cfg.get("beta_start", 0.0001)),
            beta_end=float(scheduler_cfg.get("beta_end", 0.02)),
            beta_schedule=str(scheduler_cfg.get("beta_schedule", "linear")),
            clip_sample=True,
        ).to(torch.device(self.device))
        scheduler.set_timesteps(num_steps, device=torch.device(self.device))
        return scheduler

    def _build_ddpm_scheduler(self, num_steps: int) -> DDPMScheduler:
        """Build a DDPM scheduler aligned with the current diffusion config."""
        scheduler_cfg = self.cfg.get("scheduler", {})
        scheduler = DDPMScheduler(
            num_train_timesteps=int(scheduler_cfg.get("num_train_timesteps", 1000)),
            beta_start=float(scheduler_cfg.get("beta_start", 0.0001)),
            beta_end=float(scheduler_cfg.get("beta_end", 0.02)),
            beta_schedule=str(scheduler_cfg.get("beta_schedule", "linear")),
            clip_sample=bool(scheduler_cfg.get("clip_sample", False)),
        ).to(torch.device(self.device))
        scheduler.set_timesteps(num_steps, device=torch.device(self.device))
        return scheduler

    def _build_initial_sample(
        self,
        measurement: torch.Tensor,
        shape: tuple[int, int, int, int],
        first_t: int,
        x_init: torch.Tensor | None,
        smart_init: bool,
    ) -> torch.Tensor:
        """Create an initial latent, optionally by noising the FBP reconstruction."""
        if x_init is not None:
            return x_init.to(self.device)
        if not smart_init:
            return torch.randn(shape, device=self.device)

        x_fbp = self.operator.fbp(measurement)
        noise = torch.randn_like(x_fbp)
        init_timesteps = torch.full(
            (shape[0],),
            first_t,
            device=self.device,
            dtype=torch.long,
        )
        return self.model.scheduler.add_noise(x_fbp, noise, init_timesteps)

    @staticmethod
    def _clip_batch_vector_norm(x: torch.Tensor, max_norm: float) -> torch.Tensor:
        """Clip each batch element to the requested vector norm."""
        if max_norm <= 0:
            return x
        flat = x.reshape(x.shape[0], -1)
        norms = torch.linalg.norm(flat, dim=-1, keepdim=True).clamp(min=1e-8)
        scales = (max_norm / norms).clamp(max=1.0)
        return (flat * scales).reshape_as(x)

    def sample(
        self,
        measurement: torch.Tensor,
        shape: tuple[int, int, int, int],
        x_true: torch.Tensor | None = None,
        x_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Run the full reverse process."""
        with torch.inference_mode():
            x_t = (
                x_init.to(self.device)
                if x_init is not None
                else torch.randn(shape, device=self.device)
            )

            step_pairs = self._build_step_pairs()
            history: list[dict] = []
            state: dict = {}

            progress = tqdm(
                list(enumerate(step_pairs)),
                total=len(step_pairs),
                desc="Sampling",
            )
            for step_idx, (t_cur, t_prev) in progress:
                x_t, state = self.step(x_t, t_cur, t_prev, measurement, state)
                self._record_history(progress, history, step_idx, x_t, x_true, state)

        return x_t, history


class PosteriorSamplerBase(CTSamplerBase):
    """Shared reverse sampler using a corrected image estimate D_cur.

    The common reverse update is
        x_prev = alpha_prev * D_cur + sigma_prev * eps
    where `eps` is predicted by the diffusion model and each method differs in
    how it defines the data-consistent image estimate `D_cur`.
    """

    @abstractmethod
    def compute_d_cur(
        self,
        x0_hat: torch.Tensor,
        measurement: torch.Tensor,
        t_cur: int,
        state: dict,
    ) -> tuple[torch.Tensor, float]:
        """Return the corrected image estimate D_cur and a residual statistic."""
        raise NotImplementedError

    def posterior_update(
        self,
        eps: torch.Tensor,
        d_cur: torch.Tensor,
        t_prev: int,
    ) -> torch.Tensor:
        """Apply the shared posterior-form reverse update."""
        alpha_prev = self.ns.get_alpha(t_prev)
        sigma_prev = self.ns.get_sigma(t_prev)
        return alpha_prev * d_cur + sigma_prev * eps

    def step(
        self,
        x_t: torch.Tensor,
        t_cur: int,
        t_prev: int,
        measurement: torch.Tensor,
        state: dict,
    ) -> tuple[torch.Tensor, dict]:
        with torch.no_grad():
            eps, x0_hat = self._tweedie(x_t, t_cur)
            d_cur, residual_norm = self.compute_d_cur(
                x0_hat=x0_hat,
                measurement=measurement,
                t_cur=t_cur,
                state=state,
            )
            x_prev = self.posterior_update(
                eps=eps,
                d_cur=d_cur,
                t_prev=t_prev,
            )

        state["residual"] = residual_norm
        state["d_cur"] = d_cur.detach()
        return x_prev, state
