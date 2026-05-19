from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch

from .scheduling_utils import SchedulerOutput


@dataclass
class DDIMSchedulerOutput(SchedulerOutput):
    pass


class DDIMScheduler:
    """Lightweight DDIM scheduler for prior sampling."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        clip_sample: bool = True,
    ):
        if beta_schedule != "linear":
            raise ValueError("Only linear beta_schedule is supported.")
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
        self._timestep_index = {int(t.item()): i for i, t in enumerate(self.timesteps)}

    def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None):
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be > 0")
        if num_inference_steps == self.config.num_train_timesteps:
            timesteps = torch.arange(self.config.num_train_timesteps - 1, -1, -1)
        else:
            timesteps = torch.linspace(
                self.config.num_train_timesteps - 1,
                0,
                num_inference_steps,
            ).round().to(torch.long)
        if device is not None:
            timesteps = timesteps.to(device)
        self.timesteps = timesteps
        self._timestep_index = {int(t.item()): i for i, t in enumerate(timesteps)}
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

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        eta: float = 0.0,
    ) -> DDIMSchedulerOutput:
        t = self._get_index(timestep)
        alpha_t = self.alphas_cumprod[t].to(sample.device)
        idx = self._timestep_index.get(t, 0)
        if idx + 1 >= len(self.timesteps):
            alpha_prev = torch.tensor(1.0, device=sample.device)
        else:
            prev_t = int(self.timesteps[idx + 1].item())
            alpha_prev = self.alphas_cumprod[prev_t].to(sample.device)

        pred_x0 = (sample - torch.sqrt(1.0 - alpha_t) * model_output) / torch.sqrt(alpha_t)
        if self.config.clip_sample:
            pred_x0 = pred_x0.clamp(-1.0, 1.0)

        sigma = (
            eta
            * torch.sqrt((1.0 - alpha_prev) / (1.0 - alpha_t))
            * torch.sqrt(1.0 - alpha_t / alpha_prev)
        )

        noise = torch.randn_like(sample) if eta > 0 else torch.zeros_like(sample)
        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma**2, min=0.0)) * model_output
        prev_sample = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + sigma * noise
        return DDIMSchedulerOutput(prev_sample=prev_sample)
