"""CT-LAMP: second-order multistep DDNM+ with negative correction."""

from __future__ import annotations

import math

import torch

from ct_lamp.samplers.ddnm_plus import DDNMPlusSampler


class CTLAMPSampler(DDNMPlusSampler):
    """Second-order CT-LAMP update.

    The base DDNM+ update is:
        x_{t-1} = alpha_t * D_cur + sigma_t * eps

    CT-LAMP can be parameterized in two equivalent ways:
      1. fixed gamma:
           x_{t-1} += alpha_t * A1(h) * gamma * (D_cur - D_prev) / h_prev
      2. direct beta_t schedule:
           D_tilde = (1 - beta_t) * D_cur + beta_t * D_prev
           x_{t-1} = alpha_t * D_tilde + sigma_t * eps

    If a beta schedule is provided it overrides gamma.
    """

    def __init__(self, model, operator, cfg: dict) -> None:
        super().__init__(model, operator, cfg)
        method_cfg = cfg.get("ct_lamp", {})
        if "num_steps" in method_cfg:
            self.num_steps = int(method_cfg["num_steps"])

        self.warmup_steps = int(method_cfg.get("warmup_steps", 3))
        self.gamma = float(method_cfg.get("gamma", method_cfg.get("correction_scale", -6.0)))
        self.beta_schedule_cfg = method_cfg.get("beta_schedule", method_cfg.get("beta"))
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

        self.beta_schedule = self._build_beta_schedule(self.beta_schedule_cfg)

    def _build_beta_schedule(self, schedule_cfg):
        """Compile a beta(t) schedule from config."""
        if schedule_cfg is None:
            return None
        if isinstance(schedule_cfg, (int, float)):
            value = float(schedule_cfg)
            return lambda **kwargs: value
        if isinstance(schedule_cfg, str):
            expression = schedule_cfg.strip()
            if not expression:
                return None
            return self._build_expression_schedule(expression)
        if not isinstance(schedule_cfg, dict):
            raise TypeError(
                "ct_lamp.beta_schedule must be a number, string expression, or dict, "
                f"got {type(schedule_cfg)!r}."
            )

        kind = str(schedule_cfg.get("kind", schedule_cfg.get("type", "constant"))).lower()
        if kind == "constant":
            value = float(schedule_cfg.get("value", schedule_cfg.get("beta", 0.0)))
            return lambda **kwargs: value
        if kind == "linear":
            start = float(schedule_cfg.get("start", 0.0))
            end = float(schedule_cfg.get("end", start))
            return lambda **kwargs: start + (end - start) * float(kwargs["frac"])
        if kind == "cosine":
            start = float(schedule_cfg.get("start", 0.0))
            end = float(schedule_cfg.get("end", start))
            return lambda **kwargs: end + 0.5 * (start - end) * (
                1.0 + math.cos(math.pi * float(kwargs["frac"]))
            )
        if kind == "polynomial":
            start = float(schedule_cfg.get("start", 0.0))
            end = float(schedule_cfg.get("end", start))
            power = float(schedule_cfg.get("power", 1.0))
            return lambda **kwargs: start + (end - start) * (float(kwargs["frac"]) ** power)
        if kind == "exp":
            start = float(schedule_cfg.get("start", 0.0))
            end = float(schedule_cfg.get("end", start))
            rate = float(schedule_cfg.get("rate", 5.0))
            denom = 1.0 - math.exp(-rate)
            if abs(denom) < 1e-12:
                return lambda **kwargs: end
            return lambda **kwargs: start + (end - start) * (
                (1.0 - math.exp(-rate * float(kwargs["frac"]))) / denom
            )
        if kind == "expression":
            expression = str(schedule_cfg.get("expression", "")).strip()
            if not expression:
                raise ValueError("ct_lamp.beta_schedule.expression cannot be empty.")
            return self._build_expression_schedule(expression)
        raise ValueError(
            "Unsupported ct_lamp.beta_schedule kind. "
            "Use one of: constant, linear, cosine, polynomial, exp, expression."
        )

    def _build_expression_schedule(self, expression: str):
        """Build a beta(t) schedule from a restricted expression string."""
        allowed = {
            "abs": abs,
            "min": min,
            "max": max,
            "sqrt": math.sqrt,
            "exp": math.exp,
            "log": math.log,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
        }

        def schedule(**kwargs):
            scope = dict(allowed)
            scope.update(kwargs)
            return float(eval(expression, {"__builtins__": {}}, scope))

        return schedule

    def _beta_from_gamma(self, a1: float, h_prev: float) -> float:
        """Convert the gamma parameterization into the equivalent beta_t."""
        return -self.gamma * a1 / max(h_prev, 1e-8)

    def _evaluate_beta_t(
        self,
        *,
        step_idx: int,
        t_cur: int,
        t_prev: int,
        h: float,
        h_prev: float,
    ) -> float:
        """Return beta_t, either from the explicit schedule or from gamma."""
        if self.beta_schedule is None:
            a1 = 1.0 - (1.0 + h) * math.exp(-h)
            return self._beta_from_gamma(a1=a1, h_prev=h_prev)

        frac = 0.0 if self.num_steps <= 1 else step_idx / max(self.num_steps - 1, 1)
        return float(
            self.beta_schedule(
                t=float(t_cur),
                t_cur=float(t_cur),
                t_prev=float(t_prev),
                step_idx=float(step_idx),
                num_steps=float(self.num_steps),
                frac=float(frac),
                h=float(h),
                h_prev=float(h_prev),
                lambda_cur=float(self.ns.get_lambda(t_cur).item()),
                lambda_prev=float(self.ns.get_lambda(t_prev).item()),
                gamma=float(self.gamma),
            )
        )

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

            alpha_s = self.ns.get_alpha(t_cur)
            sigma_s = self.ns.get_sigma(t_cur)
            alpha_t = self.ns.get_alpha(t_prev)
            sigma_t = self.ns.get_sigma(t_prev)
            h = self.ns.get_lambda(t_prev) - self.ns.get_lambda(t_cur)

            d_cur, residual_norm = self.correct_x0(
                x0_hat=x0_hat,
                measurement=measurement,
                alpha_s=alpha_s,
                sigma_s=sigma_s,
            )

            x_prev = alpha_t * d_cur + sigma_t * eps

            step_idx = int(state.get("step_idx", 0))
            d_prev = state.get("d_prev")
            h_prev = float(state.get("h_prev", 1.0))
            if d_prev is not None and step_idx >= self.warmup_steps:
                h_value = float(h.item() if isinstance(h, torch.Tensor) else h)
                beta_t = self._evaluate_beta_t(
                    step_idx=step_idx,
                    t_cur=t_cur,
                    t_prev=t_prev,
                    h=h_value,
                    h_prev=h_prev,
                )
                d_tilde = (1.0 - beta_t) * d_cur + beta_t * d_prev
                x_prev = alpha_t * d_tilde + sigma_t * eps
                state["beta_t"] = beta_t
            else:
                state["beta_t"] = 0.0

        state["d_prev"] = d_cur.detach()
        state["h_prev"] = float(h.item() if isinstance(h, torch.Tensor) else h)
        state["step_idx"] = int(state.get("step_idx", 0)) + 1
        state["residual"] = residual_norm
        return x_prev, state
