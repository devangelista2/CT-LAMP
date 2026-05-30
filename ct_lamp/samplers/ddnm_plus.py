"""DDNM+ sampler for sparse-view CT using a CGLS pseudoinverse."""

from __future__ import annotations

import math

import torch

from ct_lamp.samplers.base import PosteriorSamplerBase


class DDNMPlusSampler(PosteriorSamplerBase):
    """CT adaptation of DDNM+ based on a numerical pseudoinverse.

    The official DDNM/DDNM+ correction is built from A^dagger and the associated
    range/null-space projections. For sparse-view CT we do not have a tractable
    SVD, so we approximate A^dagger numerically with CGLS. If the CGLS solve is
    run to exact convergence from zero, it recovers the Moore-Penrose
    pseudoinverse applied to the target sinogram.
    """

    def __init__(self, model, operator, cfg: dict) -> None:
        super().__init__(model, operator, cfg)
        self.method_name = "ddnm_plus"
        method_cfg = cfg.get("ddnm_plus", {})
        if "num_steps" in method_cfg:
            self.num_steps = int(method_cfg["num_steps"])

        noise_sigma = float(cfg.get("operator", {}).get("noise_sigma", 0.01))
        self.sigma_y = float(method_cfg.get("sigma_y", noise_sigma))
        self.eta = float(method_cfg.get("eta", 0.0))
        self.pinv_solver = str(method_cfg.get("pinv_solver", "cgls")).lower()
        if self.pinv_solver not in {"cgls", "fbp"}:
            raise ValueError(
                "ddnm_plus.pinv_solver must be 'cgls' or 'fbp', "
                f"got {self.pinv_solver!r}."
            )

        self.cgls_iters = int(method_cfg.get("cgls_iters", method_cfg.get("cg_iters", 16)))
        self.cgls_tol = float(method_cfg.get("cgls_tol", method_cfg.get("cg_tol", 1e-5)))
        self.cgls_early_stop = bool(
            method_cfg.get("cgls_early_stop", method_cfg.get("cg_early_stop", False))
        )
        self.smart_init = bool(method_cfg.get("smart_init", True))

    def _time_travel_cfg(self) -> dict:
        """Return the configured DDNM time-travel settings for this sampler."""
        method_cfg = self.cfg.get(self.method_name, {})
        return method_cfg.get("time_travel", self.cfg.get("time_travel", {}))

    def _get_schedule_jump(self, num_steps: int, travel_length: int, travel_repeat: int) -> list[int]:
        """Official DDNM/DDNM+ jump schedule over abstract reverse-step indices."""
        if num_steps <= 0:
            return [-1]

        travel_length = max(int(travel_length), 1)
        travel_repeat = max(int(travel_repeat), 1)
        jumps = {}
        for jump_start in range(0, max(num_steps - travel_length, 0), travel_length):
            jumps[jump_start] = travel_repeat - 1

        t = num_steps
        schedule: list[int] = []
        while t >= 1:
            t -= 1
            schedule.append(t)
            if jumps.get(t, 0) > 0:
                jumps[t] -= 1
                for _ in range(travel_length):
                    t += 1
                    schedule.append(t)
        schedule.append(-1)
        return schedule

    def _build_time_travel_pairs(self) -> list[tuple[int, int]]:
        """Build reverse and jump-forward pairs using the official schedule."""
        timesteps = self.ns.get_timestep_sequence(self.num_steps)
        if not timesteps:
            return []

        tt_cfg = self._time_travel_cfg()
        enabled = bool(tt_cfg.get("enabled", False))
        travel_length = int(tt_cfg.get("travel_length", 1))
        travel_repeat = int(tt_cfg.get("travel_repeat", 1))
        if not enabled or travel_repeat <= 1:
            return self._build_step_pairs()

        abstract_schedule = self._get_schedule_jump(
            num_steps=len(timesteps),
            travel_length=travel_length,
            travel_repeat=travel_repeat,
        )
        idx_to_timestep = {idx: int(timesteps[-1 - idx]) for idx in range(len(timesteps))}

        pairs: list[tuple[int, int]] = []
        for idx_cur, idx_next in zip(abstract_schedule[:-1], abstract_schedule[1:]):
            t_cur = idx_to_timestep[int(idx_cur)]
            t_next = 0 if idx_next < 0 else idx_to_timestep[int(idx_next)]
            pairs.append((t_cur, t_next))
        return pairs

    def _apply_time_travel(
        self,
        x0_hat: torch.Tensor,
        t_next: int,
    ) -> torch.Tensor:
        """Jump back to a noisier state from the latest clean prediction."""
        noise = torch.randn_like(x0_hat)
        next_timesteps = torch.full(
            (x0_hat.shape[0],),
            fill_value=t_next,
            device=x0_hat.device,
            dtype=torch.long,
        )
        return self.model.scheduler.add_noise(x0_hat, noise, next_timesteps)

    def _reset_time_travel_state(self, state: dict) -> dict:
        """Hook for subclasses that keep extra multistep history."""
        return state

    def _step_noise_params(self, alpha_prev: torch.Tensor) -> tuple[float, float, float]:
        """Return DDNM+ noise parameters at the target reverse step."""
        alpha_prev_val = float(alpha_prev.item() if torch.is_tensor(alpha_prev) else alpha_prev)
        sigma_t = math.sqrt(max(0.0, 1.0 - alpha_prev_val))
        a_prev = math.sqrt(max(alpha_prev_val, 1e-12))
        return sigma_t, a_prev, self.sigma_y

    def _lambda_t(self, alpha_prev: torch.Tensor) -> float:
        """Official DDNM+ scalar lambda for generic linear operators."""
        sigma_t, a_prev, sigma_y = self._step_noise_params(alpha_prev)
        intro_noise = a_prev * sigma_y
        if intro_noise > 1e-12 and sigma_t < intro_noise:
            return sigma_t * math.sqrt(max(0.0, 1.0 - self.eta**2)) / (intro_noise + 1e-8)
        return 1.0

    def _cgls_pinv(self, y: torch.Tensor) -> torch.Tensor:
        """Compute A^dagger y by CGLS on min_x ||Ax - y||^2 starting from x=0."""
        x = torch.zeros(
            (y.shape[0], 1, self.operator.image_size, self.operator.image_size),
            device=y.device,
            dtype=y.dtype,
        )
        r = y.clone()
        s = self.operator.adjoint_physics(r)
        p = s.clone()
        gamma = torch.sum(s * s)
        if gamma.item() <= 0:
            return x

        for _ in range(self.cgls_iters):
            q = self.operator.forward_physics(p)
            denom = torch.sum(q * q).clamp(min=1e-12)
            alpha = gamma / denom
            x = x + alpha * p
            r = r - alpha * q
            if self.cgls_early_stop and torch.sqrt(torch.mean(r * r)) < self.cgls_tol:
                break
            s = self.operator.adjoint_physics(r)
            gamma_new = torch.sum(s * s)
            if gamma_new.item() <= 0:
                break
            beta = gamma_new / gamma.clamp(min=1e-12)
            p = s + beta * p
            gamma = gamma_new
        return x

    def pinv_physics(self, y: torch.Tensor) -> torch.Tensor:
        """Approximate A^dagger y in physical image units."""
        if self.pinv_solver == "fbp":
            return self.operator.fbp_raw_physics(y)
        return self._cgls_pinv(y)

    def range_project_physics(self, x_phys: torch.Tensor) -> torch.Tensor:
        """Apply A^dagger A to a physical-domain image."""
        return self.pinv_physics(self.operator.forward_physics(x_phys))

    def null_project_physics(self, x_phys: torch.Tensor) -> torch.Tensor:
        """Apply (I - A^dagger A) to a physical-domain image."""
        return x_phys - self.range_project_physics(x_phys)

    def correct_x0(
        self,
        x0_hat: torch.Tensor,
        measurement: torch.Tensor,
        alpha_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Apply the DDNM+ pseudoinverse correction to x0_hat."""
        x0_phys = self.operator.to_physics(x0_hat)
        residual = measurement - self.operator.forward_physics(x0_phys)
        lam = self._lambda_t(alpha_prev)
        correction = self.pinv_physics(residual)
        x_corr_phys = (x0_phys + lam * correction).clamp(0.0, 1.0)
        x_corr = self.operator.to_diffusion(x_corr_phys)

        residual_after = measurement - self.operator.forward_physics(x_corr_phys)
        residual_norm = torch.sqrt(torch.mean(residual_after * residual_after)).item()
        return x_corr, residual_norm

    def compute_d_cur(
        self,
        x0_hat: torch.Tensor,
        measurement: torch.Tensor,
        t_cur: int,
        t_prev: int,
        state: dict,
    ) -> tuple[torch.Tensor, float]:
        alpha_prev = self.ns.get_alpha(t_prev)
        return self.correct_x0(
            x0_hat=x0_hat,
            measurement=measurement,
            alpha_prev=alpha_prev,
        )

    def sample(
        self,
        measurement: torch.Tensor,
        shape: tuple[int, int, int, int],
        x_true: torch.Tensor | None = None,
        x_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Run the full reverse process with optional pseudoinverse smart init."""
        with torch.inference_mode():
            if x_init is not None:
                x_t = x_init.to(self.device)
            elif self.smart_init:
                timesteps = self.ns.get_timestep_sequence(self.num_steps)
                first_t = int(timesteps[0]) if timesteps else 0
                if self.pinv_solver == "cgls":
                    x_pinv = self.operator.to_diffusion(self.pinv_physics(measurement).clamp(0.0, 1.0))
                else:
                    x_pinv = self.operator.fbp(measurement)
                noise = torch.randn_like(x_pinv)
                init_timesteps = torch.full(
                    (shape[0],),
                    first_t,
                    device=self.device,
                    dtype=torch.long,
                )
                x_t = self.model.scheduler.add_noise(x_pinv, noise, init_timesteps)
            else:
                x_t = torch.randn(shape, device=self.device)

            step_pairs = self._build_time_travel_pairs()
            history: list[dict] = []
            state: dict = {}

            from tqdm import tqdm

            progress = tqdm(
                list(enumerate(step_pairs)),
                total=len(step_pairs),
                desc="Sampling",
            )
            for step_idx, (t_cur, t_prev) in progress:
                if t_prev < t_cur:
                    x_t, state = self.step(x_t, t_cur, t_prev, measurement, state)
                else:
                    x0_hat = state.get("x0_hat")
                    if x0_hat is None:
                        raise RuntimeError(
                            "Time-travel jump requested before any x0 prediction was stored."
                        )
                    x_t = self._apply_time_travel(x0_hat=x0_hat, t_next=t_prev)
                    state = self._reset_time_travel_state(state)
                self._record_history(progress, history, step_idx, x_t, x_true, state)

        return x_t, history
