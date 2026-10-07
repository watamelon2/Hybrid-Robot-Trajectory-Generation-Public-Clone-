"""Adapter for v_theta v2 (FiLM x_1-parameterised).

Same public interface as ``adapter.FlowPrior`` (``from_checkpoint``,
``sample_initial_mean``, ``make_prior_score_fn``) so the unified
driver doesn't care which generation it gets. ``adapter.py``'s
``FlowPrior.from_checkpoint`` peeks at the checkpoint's ``arch``
field and delegates here when ``arch == "film_x1"``.

Score-bridge formulas (see ``model_v2.py`` docstring for derivation):

The conditional density of x_t = t*x1 + (1-t)*x0 with x0 ~ N(0, I)
is N(t*x1, (1-t)^2 * I), so::

    grad_xt log p(x_t | x1) = -(x_t - t * x1) / (1 - t)^2

Marginalising over x1 ~ p_data and substituting the model's
prediction x1_hat ≈ E[x1 | x_t] gives the theoretical bridge::

    bridge="quadratic":  score ~= (t * x1_hat - x_t) / max((1 - t)^2, eps)

The original (pre-2026-05-16) v_theta sweep used a linear denom::

    bridge="linear":     score ~= (t * x1_hat - x_t) /  max(1 - t, eps)

which under-weights the prior near t=1.  The CLI defaults to
``"linear"`` so all previously published runs reproduce bit-for-bit.
Phase 7 of the v_theta_v2 sweep tests ``"quadratic"`` head-to-head.
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

from model_v2 import (  # noqa: E402
    FilmX1Field,
    FlowConfigV2,
    sample_trajectory_v2,
)


class FlowPriorV2:
    """Loaded v_theta v2 wrapper with obs normalisation.

    Public interface mirrors ``adapter.FlowPrior`` exactly.
    """

    def __init__(
        self,
        net: FilmX1Field,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        device: torch.device | str = "cpu",
    ):
        self.net = net.to(device).eval()
        self.config: FlowConfigV2 = net.config
        self.obs_mean = obs_mean.astype(np.float32)
        self.obs_std = np.maximum(obs_std.astype(np.float32), 1e-6)
        self.device = torch.device(device)

    @classmethod
    def from_checkpoint(
        cls, path: Path, device: torch.device | str = "cpu"
    ) -> "FlowPriorV2":
        blob = torch.load(path, map_location=device, weights_only=False)
        return cls._from_blob(blob, device=device)

    @classmethod
    def _from_blob(
        cls, blob: dict, device: torch.device | str = "cpu"
    ) -> "FlowPriorV2":
        """Load from an in-memory checkpoint dict (used by adapter.py
        dispatch to avoid loading twice)."""
        config = FlowConfigV2.from_dict(blob["config"])
        net = FilmX1Field(config)
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
        s_tensor = self._normalise_s0(s0)
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
        traj = sample_trajectory_v2(
            self.net, s_tensor, n_steps=n_steps, rng=gen
        )
        return traj.squeeze(0).cpu().numpy().astype(np.float32)

    def make_prior_score_fn(
        self,
        s0: np.ndarray,
        K: int,
        eps: float = 1e-2,
        bridge: str = "linear",
    ) -> Callable[[np.ndarray, int], np.ndarray]:
        """Return a ``(mu, k) -> score`` closure for ``denoise_loop``.

        Score formula for x_1-parameterisation, selected by ``bridge``:

            "linear"     :  (t * x1_hat - x) / max( 1 - t,        eps)
            "quadratic"  :  (t * x1_hat - x) / max((1 - t) ** 2,  eps)

        See module docstring for the derivation; "quadratic" is the
        theoretical bridge under linear-interp + N(0,I) noising,
        "linear" is the convention used by the published v1 sweep
        and the 2026-05-16 v_theta_v2 sweep.  Default is "linear" so
        Cell phase 5 results reproduce bit-for-bit.

        We map MBD iteration index k to t linearly: ``t_k = (k+1)/K``.
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
            x1_hat = self.net(x, t, s_tensor)        # (1, H*A) -- predicted x_1
            one_minus_t = 1.0 - t_clip
            if is_quad:
                denom = max(one_minus_t * one_minus_t, eps)
            else:
                denom = max(one_minus_t, eps)
            score = (t_clip * x1_hat - x) / denom
            return (
                score.cpu()
                .numpy()
                .reshape(H, A)
                .astype(np.float32)
            )

        return score_fn
