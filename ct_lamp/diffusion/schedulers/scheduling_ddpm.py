from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import torch

from .scheduling_utils import SchedulerOutput


class DDPMScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        clip_sample: bool = False,
    ):
        if beta_schedule != "linear":
            raise ValueError("Only linear beta_schedule is supported in this build.")
        self.config = SimpleNamespace(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
        )
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)

    def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None):
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be > 0")
        if num_inference_steps == self.config.num_train_timesteps:
            timesteps = torch.arange(
                self.config.num_train_timesteps - 1, -1, -1
            )
        else:
            timesteps = torch.linspace(
                self.config.num_train_timesteps - 1,
                0,
                num_inference_steps,
            ).round().to(torch.long)
        if device is not None:
            timesteps = timesteps.to(device)
        self.timesteps = timesteps
        return self

    def to(self, device: torch.device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.timesteps = self.timesteps.to(device)
        return self

    def _get_index(self, t: int | torch.Tensor) -> int:
        if torch.is_tensor(t):
            t = int(t.item())
        return int(t)

    def add_noise(
        self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        a = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return torch.sqrt(a) * original_samples + torch.sqrt(1.0 - a) * noise

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        **kwargs,
    ) -> SchedulerOutput:
        t = self._get_index(timestep)
        beta_t = self.betas[t].to(sample.device)
        alpha_t = self.alphas[t].to(sample.device)
        alpha_cumprod_t = self.alphas_cumprod[t].to(sample.device)
        if t == 0:
            alpha_cumprod_prev = torch.tensor(1.0, device=sample.device)
        else:
            alpha_cumprod_prev = self.alphas_cumprod[t - 1].to(sample.device)

        pred_x0 = (sample - torch.sqrt(1.0 - alpha_cumprod_t) * model_output) / torch.sqrt(
            alpha_cumprod_t
        )
        if self.config.clip_sample:
            pred_x0 = pred_x0.clamp(-1.0, 1.0)

        posterior_variance = beta_t * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t)
        posterior_mean = (
            torch.sqrt(alpha_cumprod_prev) * beta_t / (1.0 - alpha_cumprod_t) * pred_x0
            + torch.sqrt(alpha_t) * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t) * sample
        )

        if t > 0:
            noise = torch.randn_like(sample)
            prev_sample = posterior_mean + torch.sqrt(posterior_variance) * noise
        else:
            prev_sample = posterior_mean
        return SchedulerOutput(prev_sample=prev_sample)
