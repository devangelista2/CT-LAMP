"""Simple filtered backprojection baseline."""

from __future__ import annotations


class FBPSampler:
    """One-step FBP baseline on the CT operator."""

    def __init__(self, model, operator, cfg: dict) -> None:
        self.operator = operator

    def sample(
        self,
        measurement,
        shape,
        x_true=None,
        x_init=None,
    ):
        x_hat = self.operator.fbp(measurement)
        return x_hat, []
