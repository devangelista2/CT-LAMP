"""CT-LAMP-3M: third-order multistep DDNM+ correction."""

from __future__ import annotations

import torch

from ct_lamp.samplers.base import exp_integrals
from ct_lamp.samplers.ddnm_plus import DDNMPlusSampler


class CTLAMP3MSampler(DDNMPlusSampler):
    """Third-order variant using slope and curvature terms in lambda space."""

    def __init__(self, model, operator, cfg: dict) -> None:
        super().__init__(model, operator, cfg)
        self.method_name = "ct_lamp_3m"
        method_cfg = cfg.get("ct_lamp_3m", {})
        if "num_steps" in method_cfg:
            self.num_steps = int(method_cfg["num_steps"])

        self.warmup_steps = int(method_cfg.get("warmup_steps", 3))
        self.correction_scale = float(method_cfg.get("correction_scale", -6.0))
        self.correction_scale_a2 = float(method_cfg.get("correction_scale_a2", 7.0))
        self.smart_init = bool(method_cfg.get("smart_init", self.smart_init))
        if "cg_iters" in method_cfg:
            self.cg_iters = int(method_cfg["cg_iters"])
        if "cg_tol" in method_cfg:
            self.cg_tol = float(method_cfg["cg_tol"])
        if "consistency_solver" in method_cfg:
            self.consistency_solver = str(method_cfg["consistency_solver"]).lower()
        if "fbp_relaxation" in method_cfg:
            self.fbp_relaxation = float(method_cfg["fbp_relaxation"])
        if "eta_scale" in method_cfg:
            noise_sigma = float(cfg.get("operator", {}).get("noise_sigma", 0.01))
            self.eta2 = (float(method_cfg["eta_scale"]) * noise_sigma) ** 2

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

            alpha_t = self.ns.get_alpha(t_prev)
            sigma_t = self.ns.get_sigma(t_prev)
            h = self.ns.get_lambda(t_prev) - self.ns.get_lambda(t_cur)
            _, a1, a2 = exp_integrals(h)

            d_cur, residual_norm = self.correct_x0(
                x0_hat=x0_hat,
                measurement=measurement,
                alpha_prev=alpha_t,
            )

            x_prev = alpha_t * d_cur + sigma_t * eps
            step_idx = int(state.get("step_idx", 0))
            d_prev = state.get("d_prev")
            h_prev = float(state.get("h_prev", 1.0))

            if d_prev is not None and step_idx >= self.warmup_steps:
                b = (d_cur - d_prev) / h_prev
                d_prev2 = state.get("d_prev2")
                h_prev2 = float(state.get("h_prev2", 1.0))
                if d_prev2 is not None and step_idx >= self.warmup_steps + 1:
                    b_prev = (d_prev - d_prev2) / h_prev2
                    c = (b - b_prev) / (h_prev + h_prev2)
                    x_prev = x_prev + alpha_t * (
                        self.correction_scale * a1 * b
                        + self.correction_scale_a2 * c * (a2 + h_prev * a1)
                    )
                else:
                    x_prev = x_prev + alpha_t * self.correction_scale * a1 * b

        state["d_prev2"] = state.get("d_prev")
        state["h_prev2"] = state.get("h_prev")
        state["d_prev"] = d_cur.detach()
        state["h_prev"] = float(h.item() if isinstance(h, torch.Tensor) else h)
        state["step_idx"] = int(state.get("step_idx", 0)) + 1
        state["residual"] = residual_norm
        state["x0_hat"] = x0_hat.detach()
        return x_prev, state

    def _reset_time_travel_state(self, state: dict) -> dict:
        """Discard multistep memory across a DDNM-style time-travel restart."""
        state = dict(state)
        for key in ("d_prev", "d_prev2", "h_prev", "h_prev2"):
            state.pop(key, None)
        return state
