"""Adapter for v_theta v3 (Karras-EDM noise-conditional score model).

Public interface mirrors ``adapter.FlowPrior`` / ``adapter_v2.FlowPriorV2``
so the unified driver doesn't have to branch:

  - ``FlowPriorV3.from_checkpoint(path, device)``
  - ``.sample_initial_mean(s0, n_steps, seed) -> (H, A) float32``
  - ``.make_prior_score_fn(s0, K) -> (mu, k) -> (H, A) score``

What's different from v1 / v2
-----------------------------

1. **No bridge formula.**  ``make_prior_score_fn`` returns the
   *direct* Karras score ``(D(mu, σ_k, s0) - mu) / σ_k²``.  No
   ``(1-t)`` or ``(1-t)²`` conversion -- it's a one-line Tweedie
   denoising-step inversion.

2. **σ-schedule aware.**  The adapter is configured with
   ``(sigma_min, sigma_max)`` at construction time so it can map
   MBD's step index ``k`` (which is what ``denoise_loop`` passes to
   the score-fn closure) to the physically-correct σ_k.  These
   default to the values baked into the checkpoint
   (``config.sigma_min``, ``config.sigma_max``) but can be
   overridden at load time if the runtime schedule differs.

3. **Heun sampler.**  ``sample_initial_mean`` integrates a 2nd-order
   Heun ODE from σ_max down to σ_min using Karras's σ schedule
   (eq. 5 with ρ=7).  This is the same sampler used by EDM and
   Stable Diffusion at inference time -- markedly less drift than
   v2's 32-step Euler with the OT-CFM velocity.
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

from model_v3 import (  # noqa: E402
    FilmEdmField,
    FlowConfigV3,
    edm_sample,
    edm_score,
)


# Import the schedule helper once so we don't depend on mbd.py at
# adapter import time (keeps the adapter testable in isolation).
def _cosine_sigma_schedule(
    K: int, sigma_max: float, sigma_min: float,
) -> np.ndarray:
    """Cosine descent ``sigma_max -> sigma_min`` over ``K`` steps.

    Mirrors ``experiments/04_mbd_baseline/mbd.cosine_sigma_schedule``
    bit-for-bit; duplicated here so the adapter has no exp 04
    dependency.
    """
    ts = np.linspace(0.0, np.pi, int(K), dtype=np.float32)
    cos = (1.0 + np.cos(ts)) * 0.5  # 1 -> 0
    return (sigma_min + (sigma_max - sigma_min) * cos).astype(np.float32)


class FlowPriorV3:
    """Loaded v_theta v3 wrapper with obs normalisation."""

    def __init__(
        self,
        net: FilmEdmField,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        device: torch.device | str = "cpu",
    ):
        self.net = net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.config: FlowConfigV3 = net.config
        self.obs_mean = obs_mean.astype(np.float32)
        self.obs_std = np.maximum(obs_std.astype(np.float32), 1e-6)
        self.device = torch.device(device)
        # Default σ-schedule range to the trained one; allow override.
        self.sigma_min = float(
            sigma_min if sigma_min is not None else self.config.sigma_min
        )
        self.sigma_max = float(
            sigma_max if sigma_max is not None else self.config.sigma_max
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        device: torch.device | str = "cpu",
        sigma_min: float | None = None,
        sigma_max: float | None = None,
    ) -> "FlowPriorV3":
        blob = torch.load(path, map_location=device, weights_only=False)
        arch = blob.get("arch", "")
        if arch != "karras_edm_v3":
            raise ValueError(
                f"checkpoint at {path} has arch={arch!r}, expected "
                "'karras_edm_v3'.  Use adapter.FlowPrior.from_checkpoint "
                "for v1/v2 checkpoints."
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
    ) -> "FlowPriorV3":
        config = FlowConfigV3.from_dict(blob["config"])
        net = FilmEdmField(config)
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
    # Public API mirroring adapter.FlowPrior / adapter_v2.FlowPriorV2
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
        """Sample one warm-start trajectory ``(H, A)`` from ``p(. | s0)``.

        Uses Karras's 2nd-order Heun sampler from σ_max to σ_min.
        """
        s_t = self._normalise_s(np.asarray(s0))
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
        traj = edm_sample(
            self.net, s_t,
            n_steps=int(n_steps),
            sigma_init=self.sigma_max,
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

        The closure computes the **direct** score
        ``(D(mu; σ_k, s0) - mu) / σ_k²`` -- no bridge formula.  Here
        ``σ_k`` is the entry of MBD's actual schedule at step ``k``;
        defaults to ``cosine_sigma_schedule(K, sigma_max, sigma_min)``
        but can be overridden via ``sigma_schedule`` if the caller
        builds a non-cosine schedule.

        Parameters
        ----------
        s0 : np.ndarray
            The chunk's starting observation (un-normalised; we
            apply ``(s - obs_mean) / obs_std`` internally).
        K : int
            Number of MBD denoising iterations.
        sigma_schedule : optional (K,) ndarray
            If given, use this exact schedule.  Otherwise build the
            adapter's default cosine schedule with the adapter's
            ``(sigma_min, sigma_max)``.
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

        sigma_min_clip = float(max(self.config.sigma_min, 1e-4))

        @torch.no_grad()
        def score_fn(mu: np.ndarray, k: int) -> np.ndarray:
            sigma_k = float(max(schedule[int(k)], sigma_min_clip))
            x = torch.from_numpy(
                np.asarray(mu, dtype=np.float32).reshape(1, H * A)
            ).to(self.device)
            sigma_t = torch.full((1,), sigma_k, device=self.device)
            score = edm_score(self.net, x, sigma_t, s_t)        # (1, H*A)
            return (
                score.cpu()
                .numpy()
                .reshape(H, A)
                .astype(np.float32)
            )

        return score_fn
