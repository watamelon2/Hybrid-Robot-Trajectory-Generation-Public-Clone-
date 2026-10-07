"""Adapt a trained ``v_theta`` checkpoint for use inside MBD.

Provides two entry points used by exp 07's lambda-sweep runner and
by future hybrid runs that want to swap exp 03's MLP-BC for the
flow-matching prior:

1. ``sample_initial_mean(s0)`` -- one ODE-integrated trajectory
   from the prior, conditioned on the live env's initial obs.
   Drop-in replacement for exp 05's ``bc_warmstart_actions``.

2. ``make_prior_score_fn(s0)`` -- a closure with the
   ``(mu_k, k) -> score`` signature required by
   ``mbd.denoise_loop``'s new ``prior_score_fn`` argument.

The score is a finite-difference estimator using the same
linear-interpolant flow Lipman et al. trains under: at iteration
``k`` of MBD's ``K`` denoising iterations we map ``k -> t in (0, 1)``
linearly, query ``v_theta(x_flat, t)`` once, and return the
implicit-score conversion::

    grad log p_t(x) ~= (t * v_theta(x, t) - x) / max((1 - t)**2, eps)

This is a known-stable approximation; it is *not* the exact score,
but it is monotone in the direction of the data manifold which is
all the composite update needs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from model import FlowConfig, VelocityField, sample_trajectory  # noqa: E402


class FlowPrior:
    """Loaded v_theta wrapper with obs normalisation."""

    def __init__(
        self,
        net: VelocityField,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        device: torch.device | str = "cpu",
    ):
        self.net = net.to(device).eval()
        self.config: FlowConfig = net.config
        self.obs_mean = obs_mean.astype(np.float32)
        self.obs_std = np.maximum(
            obs_std.astype(np.float32), 1e-6
        )
        self.device = torch.device(device)

    @classmethod
    def from_checkpoint(
        cls, path: Path, device: torch.device | str = "cpu",
        sigma_min: float | None = None,
        sigma_max: float | None = None,
    ):
        """Load a flow-prior checkpoint and return the right wrapper.

        We auto-dispatch on the checkpoint's ``arch`` field:

          - ``arch == "karras_edm_v3"`` (or ``format_version >= 3``):
            return a ``FlowPriorV3`` instance from ``adapter_v3.py``.
            Honours ``sigma_min`` / ``sigma_max`` for the inference σ
            schedule (defaults baked into the checkpoint).

          - ``arch == "film_x1"`` (or ``format_version == 2``):
            return a ``FlowPriorV2`` instance from
            ``adapter_v2.py``.  ``sigma_*`` are ignored (v2 uses an
            OT-CFM time mapping that doesn't have a σ schedule).

          - missing / ``"velocity_concat"`` (v1, the original): load
            the v1 ``VelocityField`` here and return a ``FlowPrior``.
            ``sigma_*`` are ignored (v1 uses the same OT-CFM mapping).

        All three classes expose ``sample_initial_mean(s0, n_steps,
        seed)`` and ``make_prior_score_fn(s0, K, ...)`` with the
        same public surface, so the unified driver doesn't care
        which generation it gets.
        """
        blob = torch.load(path, map_location=device, weights_only=False)
        arch = blob.get("arch", "velocity_concat")
        format_version = int(blob.get("format_version", 1))
        if arch == "dp_unet1d" or format_version >= 7:
            # Delay-import so v1/v2/v3/v6 paths don't pull in DP deps.
            from adapter_dp import FlowPriorDP
            return FlowPriorDP._from_blob(
                blob, device=device,
                sigma_min=sigma_min, sigma_max=sigma_max,
            )
        if arch == "cfm_v6" or format_version >= 6:
            # Delay-import so v3/v2/v1 paths don't pull in v6 deps.
            from adapter_v6 import FlowPriorV6
            return FlowPriorV6._from_blob(
                blob, device=device,
                sigma_min=sigma_min, sigma_max=sigma_max,
            )
        if arch == "karras_edm_v3" or format_version >= 3:
            # Delay-import to keep v1 standalone if v3 deps not present.
            from adapter_v3 import FlowPriorV3
            return FlowPriorV3._from_blob(
                blob, device=device,
                sigma_min=sigma_min, sigma_max=sigma_max,
            )
        if arch == "film_x1" or format_version >= 2:
            from adapter_v2 import FlowPriorV2
            return FlowPriorV2._from_blob(blob, device=device)
        # v1 path (unchanged behaviour)
        config = FlowConfig.from_dict(blob["config"])
        net = VelocityField(config)
        net.load_state_dict(blob["state_dict"])
        return cls(
            net,
            np.asarray(blob["obs_mean"], dtype=np.float32),
            np.asarray(blob["obs_std"], dtype=np.float32),
            device=device,
        )

    def _normalise_s0(self, s0: np.ndarray) -> torch.Tensor:
        s = (s0.astype(np.float32) - self.obs_mean) / self.obs_std
        return torch.from_numpy(s).to(self.device).unsqueeze(0)

    @torch.no_grad()
    def sample_initial_mean(
        self,
        s0: np.ndarray,
        n_steps: int = 32,
        seed: int | None = None,
    ) -> np.ndarray:
        """One ODE-integrated trajectory ``(H, action_dim)`` from p(.|s0)."""
        s_tensor = self._normalise_s0(s0)
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
        traj = sample_trajectory(self.net, s_tensor, n_steps=n_steps, rng=gen)
        return traj.squeeze(0).cpu().numpy().astype(np.float32)

    def make_prior_score_fn(
        self,
        s0: np.ndarray,
        K: int,
        eps: float = 1e-2,
        bridge: str = "linear",
    ) -> Callable[[np.ndarray, int], np.ndarray]:
        """Return a ``(mu, k) -> score`` closure for ``denoise_loop``.

        ``bridge`` selects the score-bridge denominator (mirrors
        ``adapter_v2.FlowPriorV2.make_prior_score_fn``):

            "linear"     :  (t * v - x) / max( 1 - t,        eps)
            "quadratic"  :  (t * v - x) / max((1 - t) ** 2,  eps)

        Default is "linear" so v1 reproduces bit-for-bit.

        The mapping ``k -> t`` is linear in the MBD iteration index:
        ``t_k = (k + 1) / K`` so the prior contribution starts at
        small ``t`` (large displacement allowed) and ends near 1
        (only fine adjustments). This mirrors the noise-schedule
        annealing already used by ``cosine_sigma_schedule``.
        """
        if bridge not in {"linear", "quadratic"}:
            raise ValueError(
                f"bridge must be 'linear' or 'quadratic', got {bridge!r}"
            )
        s_tensor = self._normalise_s0(s0)
        H = self.config.horizon
        A = self.config.action_dim
        is_quad = bridge == "quadratic"

        @torch.no_grad()
        def score_fn(mu: np.ndarray, k: int) -> np.ndarray:
            t_val = float((k + 1) / max(K, 1))
            t_clip = float(min(max(t_val, eps), 0.99))
            x = torch.from_numpy(
                np.asarray(mu, dtype=np.float32).reshape(1, H * A)
            ).to(self.device)
            t = torch.full((1,), t_clip, device=self.device)
            v = self.net(x, t, s_tensor)  # (1, H*A)
            one_minus_t = 1.0 - t_clip
            if is_quad:
                denom = max(one_minus_t * one_minus_t, eps)
            else:
                denom = max(one_minus_t, eps)
            score = (t_clip * v - x) / denom
            return (
                score.cpu()
                .numpy()
                .reshape(H, A)
                .astype(np.float32)
            )

        return score_fn
