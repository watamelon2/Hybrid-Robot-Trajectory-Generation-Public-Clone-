"""Tiny MLP velocity field v_theta(x, t) for the flow-matching prior.

Uses the *conditional* flow-matching objective from Lipman et al.
(ICLR 2023) with the optimal-transport (linear-interpolant) coupling::

    x0 ~ N(0, I),  x1 ~ p_data,  t ~ U(0, 1)
    x_t   = (1 - t) x0 + t x1
    v_t*  = x1 - x0
    L     = E_t,x0,x1 || v_theta(x_t, t) - v_t* ||^2

We model trajectories of shape ``(H, action_dim)`` flat-vectorised
to ``(H * action_dim,)`` and condition on the initial observation
``s0 ~ R^{D_obs}`` (the same flat ``low_dim_v141`` obs used by
exp 03's BC). The conditioning is concatenated to ``[x_t, t]``;
no fancy attention. Architecture is intentionally small so a
laptop-sized run can converge in a few minutes.

The score interpretation (used by ``denoise_loop``'s
``prior_score_fn`` hook in exp 04's mbd.py): for the
linear-interpolant flow the marginal score at time t is

    grad log p_t(x) ~= (t * v_theta(x, t) - x) / max((1 - t), eps)

Sampling from the prior uses the standard ODE integration
``x_{t + dt} = x_t + v_theta(x_t, t) * dt`` from t=0 to t=1.

This module is **import-only**: the trainer lives in train.py and
the MBD-side adapter (which wraps a saved checkpoint as a
``PriorScoreFn``) lives in adapter.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class FlowConfig:
    obs_dim: int            # D_obs (e.g. 46 for can/ph low_dim_v141)
    action_dim: int         # 7 for OSC_POSE
    horizon: int            # H (e.g. 120)
    hidden_sizes: tuple[int, ...] = (512, 512, 512)
    time_embed_dim: int = 32

    @property
    def x_dim(self) -> int:
        return self.horizon * self.action_dim

    def to_dict(self) -> dict[str, Any]:
        return {
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "horizon": int(self.horizon),
            "hidden_sizes": list(self.hidden_sizes),
            "time_embed_dim": int(self.time_embed_dim),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FlowConfig":
        return cls(
            obs_dim=int(d["obs_dim"]),
            action_dim=int(d["action_dim"]),
            horizon=int(d["horizon"]),
            hidden_sizes=tuple(int(h) for h in d["hidden_sizes"]),
            time_embed_dim=int(d["time_embed_dim"]),
        )


def sinusoidal_time_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer-style time embedding."""
    half = dim // 2
    freqs = torch.exp(
        -np.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class VelocityField(nn.Module):
    """v_theta(x_t, t, s0) -> (B, x_dim) velocity prediction."""

    def __init__(self, config: FlowConfig):
        super().__init__()
        self.config = config
        in_dim = config.x_dim + config.time_embed_dim + config.obs_dim
        layers: list[nn.Module] = []
        prev = in_dim
        for h in config.hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            prev = h
        layers.append(nn.Linear(prev, config.x_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, s0: torch.Tensor
    ) -> torch.Tensor:
        # x: (B, x_dim); t: (B,); s0: (B, obs_dim)
        emb = sinusoidal_time_embed(t, self.config.time_embed_dim)
        h = torch.cat([x, emb, s0], dim=-1)
        return self.net(h)


def conditional_fm_loss(
    model: VelocityField,
    x1: torch.Tensor,        # (B, x_dim) -- target trajectories
    s0: torch.Tensor,        # (B, obs_dim)
    rng: torch.Generator | None = None,
    n_t_per_traj: int = 64,
) -> torch.Tensor:
    """Conditional flow-matching loss with linear-interpolant coupling.

    For each of the ``B`` target trajectories we draw ``n_t_per_traj``
    independent ``(x0, t)`` pairs and average the squared error. This
    multiplies the effective gradient signal per epoch by
    ``n_t_per_traj`` at the cost of one extra forward pass over the
    repeated batch (which on this 6.7M-param net is the bottleneck
    anyway, so the wall-time hit is modest). Default ``64`` was
    chosen empirically as the sweet spot between throughput and
    diminishing returns; see 2026-05-15 debug entry in run_log.md.
    """
    B = x1.shape[0]
    device = x1.device
    K = max(int(n_t_per_traj), 1)
    if K > 1:
        x1_rep = x1.repeat_interleave(K, dim=0)
        s0_rep = s0.repeat_interleave(K, dim=0)
    else:
        x1_rep = x1
        s0_rep = s0
    N = x1_rep.shape[0]
    if rng is None:
        x0 = torch.randn_like(x1_rep)
        t = torch.rand(N, device=device)
    else:
        x0 = torch.randn(x1_rep.shape, generator=rng, device=device)
        t = torch.rand(N, generator=rng, device=device)
    t_b = t[:, None]
    xt = (1.0 - t_b) * x0 + t_b * x1_rep
    target_v = x1_rep - x0
    pred_v = model(xt, t, s0_rep)
    return ((pred_v - target_v) ** 2).mean()


@torch.no_grad()
def sample_trajectory(
    model: VelocityField,
    s0: torch.Tensor,         # (B, obs_dim)
    n_steps: int = 32,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Integrate the ODE x' = v_theta(x, t, s0) from t=0 (noise) to t=1."""
    B = s0.shape[0]
    device = s0.device
    cfg = model.config
    if rng is None:
        x = torch.randn(B, cfg.x_dim, device=device)
    else:
        x = torch.randn(B, cfg.x_dim, generator=rng, device=device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((B,), i * dt, device=device)
        v = model(x, t, s0)
        x = x + dt * v
    return x.view(B, cfg.horizon, cfg.action_dim)
