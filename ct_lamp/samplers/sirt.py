"""Simple SIRT baseline using ASTRA through the CT operator."""

from __future__ import annotations


class SIRTSampler:
    """One-step wrapper around ASTRA SIRT on the CT geometry."""

    def __init__(self, model, operator, cfg: dict) -> None:
        self.operator = operator
        method_cfg = cfg.get("sirt", {})
        self.num_iters = int(method_cfg.get("num_iters", 50))

    def sample(
        self,
        measurement,
        shape,
        x_true=None,
        x_init=None,
    ):
        x_hat = self.operator.sirt(measurement, num_iters=self.num_iters)
        return x_hat, []
