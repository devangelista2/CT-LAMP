from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class SchedulerOutput:
    prev_sample: torch.Tensor
