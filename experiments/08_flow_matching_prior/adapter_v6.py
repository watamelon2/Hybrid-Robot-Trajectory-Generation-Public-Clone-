"""Adapter for v_theta v6 (Conditional Flow Matching, Lipman 2023).

Public interface mirrors ``adapter_v3.FlowPriorV3`` exactly:

  - ``FlowPriorV6.from_checkpoint(path, device)``
  - ``.sample_initial_mean(s0, n_steps, seed) -> (H, A) float32``
  - ``.make_prior_score_fn(s0, K, sigma_schedule) -> (mu, k) -> (H, A) score``

What's different from v3
-------------------------

1. **Score formula**: ``∇log p_t(x) = (t · v_θ(x, t, s) - x) / (1 - t)``
   instead of v3's Karras identity ``(D(x; σ, s) - x) / σ²``.  We map
   MBD's σ at step ``k`` to physical time via ``t = 1 - σ``, so
   the bridge denominator (1 - t) equals σ — bounded below by σ_min.

2. **Sampler**: Heun 2nd-order ODE on t ∈ [0, 1 - t_eps] (forward),
   versus v3's Heun on σ ∈ [σ_max, σ_min] (reverse).  Both are
   2nd-order accurate; the difference is the parameterisation of
   the noise level.

3. **No σ-schedule baked into the checkpoint.**  v6 trained over
   ``t ~ U[0, 1 - t_eps]``, so the inference σ schedule can use any
   ``[σ_min, σ_max] ⊂ [t_eps, 1]``.  We default to v3's runtime values
   ``(0.05, 0.25)`` so that head-to-head comparisons are apples-to-apples.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch


_THIS_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _THIS_DIR.parent
for _p in (_EXPERIMENTS_DIR, _THIS_DIR):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from model_v6_cfm import (  # noqa: E402
    CFMField,
    FlowConfigV6,
    cfm_sample,
    cfm_score,
)


def _cosine_sigma_schedule(
    K: int, sigma_max: float, sigma_min: float,
) -> np.ndarray:
    """Cosine descent ``sigma_max -> sigma_min`` over ``K`` steps.

    Identical to ``adapter_v3._cosine_sigma_schedule``; duplicated here
    so the adapter has no v3 dependency at import time.
    """
    ts = np.linspace(0.0, np.pi, int(K), dtype=np.float32)
    cos = (1.0 + np.cos(ts)) * 0.5  # 1 -> 0
    return (sigma_min + (sigma_max - sigma_min) * cos).astype(np.float32)


class FlowPriorV6:
    """Loaded v_theta v6 (CFM) wrapper with obs normalisation."""

    DEFAULT_SIGMA_MIN = 0.05
    DEFAULT_SIGMA_MAX = 0.25

    def __init__(
        self,
        net: CFMField,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        device: torch.device | str = "cpu",
    ):
        self.net = net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.config: FlowConfigV6 = net.config
        self.obs_mean = obs_mean.astype(np.float32)
        self.obs_std = np.maximum(obs_std.astype(np.float32), 1e-6)
        self.device = torch.device(device)
        # CFM is trained over t ∈ [0, 1 - t_eps] uniformly, so the
        # inference σ-range is a free choice.  Default to v3's values
        # so MBD's cosine_sigma_schedule -> CFM-score works without
        # any caller-side change.
        self.sigma_min = float(
            sigma_min if sigma_min is not None else self.DEFAULT_SIGMA_MIN
        )
        self.sigma_max = float(
            sigma_max if sigma_max is not None else self.DEFAULT_SIGMA_MAX
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        device: torch.device | str = "cpu",
        sigma_min: float | None = None,
        sigma_max: float | None = None,
    ) -> "FlowPriorV6":
        blob = torch.load(path, map_location=device, weights_only=False)
        arch = blob.get("arch", "")
        if arch != "cfm_v6":
            raise ValueError(
                f"checkpoint at {path} has arch={arch!r}, expected "
                "'cfm_v6'.  Use adapter.FlowPrior.from_checkpoint for "
                "auto-dispatch."
            )
        return cls._from_blob(
            blob, device=device,
            sigma_min=sigma_min, sigma_max=sigma_max,
        )

    @classmethod
    def _from_blob(
        cls,
        blob: dict,
        device: torch.device | str = "cpu",
        sigma_min: float | None = None,
        sigma_max: float | None = None,
    ) -> "FlowPriorV6":
        config = FlowConfigV6.from_dict(blob["config"])
        net = CFMField(config)
        net.load_state_dict(blob["state_dict"])
        return cls(
            net,
            np.asarray(blob["obs_mean"], dtype=np.float32),
            np.asarray(blob["obs_std"], dtype=np.float32),
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            device=device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _normalise_s(self, s: np.ndarray) -> torch.Tensor:
        z = (s.astype(np.float32) - self.obs_mean) / self.obs_std
        return torch.from_numpy(z).to(self.device).unsqueeze(0)

    @torch.no_grad()
    def sample_initial_mean(
        self,
        s0: np.ndarray,
        n_steps: int = 32,
        seed: int | None = None,
    ) -> np.ndarray:
        """Sample one warm-start trajectory ``(H, A)`` from p(. | s0).

        Heun 2nd-order ODE from t=0 (noise) to t = 1 - t_eps (data).
        """
        s_t = self._normalise_s(np.asarray(s0))
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
        traj = cfm_sample(
            self.net, s_t,
            n_steps=int(n_steps),
            t_max=1.0 - float(self.config.t_eps),
            rng=gen,
        )
        return traj.squeeze(0).cpu().numpy().astype(np.float32)

    def make_prior_score_fn(
        self,
        s0: np.ndarray,
        K: int,
        sigma_schedule: np.ndarray | None = None,
    ) -> Callable[[np.ndarray, int], np.ndarray]:
        """Return a ``(mu, k) -> score`` closure for MBD's denoise_loop.

        Maps MBD's σ at step ``k`` (from a cosine schedule by default)
        to physical time ``t = 1 - σ``, queries v_θ(mu, t, s0), and
        returns the marginal score
            ∇log p_t(x) = (t · v_θ - x) / (1 - t).

        ``sigma_schedule`` is optional and lets the caller override the
        default cosine descent (same convention as ``adapter_v3``).
        """
        s_t = self._normalise_s(np.asarray(s0))
        H = self.config.horizon
        A = self.config.action_dim

        if sigma_schedule is not None:
            schedule = np.asarray(sigma_schedule, dtype=np.float32)
            if schedule.shape != (int(K),):
                raise ValueError(
                    f"sigma_schedule shape {schedule.shape}, expected ({K},)"
                )
        else:
            schedule = _cosine_sigma_schedule(
                int(K), self.sigma_max, self.sigma_min,
            )

        # Clip σ from above by 1 - t_eps so (1 - t) = σ stays in the
        # support of the trained network.
        sigma_clip_hi = 1.0 - float(self.config.t_eps)
        sigma_clip_lo = float(self.config.t_eps)

        @torch.no_grad()
        def score_fn(mu: np.ndarray, k: int) -> np.ndarray:
            sigma_k = float(np.clip(schedule[int(k)], sigma_clip_lo, sigma_clip_hi))
            x = torch.from_numpy(
                np.asarray(mu, dtype=np.float32).reshape(1, H * A)
            ).to(self.device)
            sigma_t = torch.full((1,), sigma_k, device=self.device)
            score = cfm_score(self.net, x, sigma_t, s_t)        # (1, H*A)
            return (
                score.cpu()
                .numpy()
                .reshape(H, A)
                .astype(np.float32)
            )

        return score_fn
