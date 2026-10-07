"""FiLM-conditioned, x_1-parameterised velocity field for v_theta v2.

Motivation -- see 2026-05-15 debug entries: the original ``VelocityField``
in ``model.py`` exhibited two failure modes when trained on 20 demos of
can/ph:

1.  **No s0 conditioning in practice.** s0 enters the network only as
    ~5% of the input vector after concatenation with the 840-dim x_t
    flat, so the optimizer ignores it. Empirical evidence:
    ``sample_initial_mean(s0)`` returns nearly identical trajectories
    for 5 different demos (L2 distance to each demo's actions
    converges to ~22 regardless of which demo was used to condition).

2.  **Noise-dominated training target.** With the velocity target
    ``v* = x_1 - x_0`` and ``x_0 ~ N(0, I)`` on a 840-dim vector,
    most of the target variance comes from the noise. The optimizer
    only needs to learn ``-x_0`` to explain ~60% of variance, which
    is exactly the empirical loss floor we hit (loss ~0.6 with the
    zero predictor at ~1.2).

v2 addresses both:

* **FiLM conditioning**: a small per-layer scale-and-shift produced
  from s0 modulates the trunk's hidden activations multiplicatively.
  This forces s0 to have a multiplicative (not just additive) effect
  on the output, so the optimizer cannot ignore it.

* **x_1-parameterisation**: predict the clean trajectory directly,
  not velocity. Loss is just MSE against the real demo. Velocity at
  sampling time is recovered analytically::

      v_t = (x1_hat - x_t) / max(1 - t, eps)

  This is mathematically equivalent at the optimum but has a much
  better-conditioned loss landscape (no noise term in the target).

* **Score-bridge formula** for ``denoise_loop``::

      grad log p_t(x_t) ~= (t * x1_hat - x_t) / max(1 - t, eps)

  Derivation: starting from the same linear-interpolant identity
  ``x_t = (1 - t) x_0 + t x_1`` and the prior approximation
  ``x_t | x_1 ~ N(t x_1, (1-t)^2 I)``, the score under that
  Gaussian is ``(t x_1 - x_t)/(1-t)^2``. We keep the ``(1-t)``
  denominator (not squared) for numerical consistency with
  ``model.py``'s convention so an existing lambda sweep transfers.

Checkpoint format:
    {
        "format_version": 2,
        "arch": "film_x1",          # adapter.py dispatches on this
        "config": FlowConfigV2.to_dict(),
        "state_dict": ...,
        "obs_mean": (D_obs,),
        "obs_std":  (D_obs,),
        "n_demos": int,
        "horizon": int,
        "wall_s": float,
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class FlowConfigV2:
    """Hyper-parameters for the FiLM x_1-parameterised flow model."""

    obs_dim: int                                # D_obs (46 for can/ph)
    action_dim: int                             # 7
    horizon: int                                # 120
    hidden: int = 768                           # width of trunk + FiLM net
    n_blocks: int = 4                           # number of residual FiLM blocks
    cond_hidden: int = 256                      # FiLM-conditioning MLP width
    time_embed_dim: int = 64                    # sinusoidal time embed dim
    arch: str = "film_x1"                       # used by adapter dispatch

    @property
    def x_dim(self) -> int:
        return self.horizon * self.action_dim

    def to_dict(self) -> dict[str, Any]:
        return {
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "horizon": int(self.horizon),
            "hidden": int(self.hidden),
            "n_blocks": int(self.n_blocks),
            "cond_hidden": int(self.cond_hidden),
            "time_embed_dim": int(self.time_embed_dim),
            "arch": str(self.arch),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FlowConfigV2":
        return cls(
            obs_dim=int(d["obs_dim"]),
            action_dim=int(d["action_dim"]),
            horizon=int(d["horizon"]),
            hidden=int(d["hidden"]),
            n_blocks=int(d["n_blocks"]),
            cond_hidden=int(d["cond_hidden"]),
            time_embed_dim=int(d["time_embed_dim"]),
            arch=str(d.get("arch", "film_x1")),
        )


def sinusoidal_time_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformer-style sinusoidal time embedding (same as v1)."""
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


class FiLMBlock(nn.Module):
    """Residual block with FiLM conditioning::

        h <- h + Linear_out( SiLU( scale * Linear_in(h) + shift ) )

    Both ``scale`` and ``shift`` have shape ``(B, hidden)`` and are
    produced from the conditioning vector by the parent's cond-net.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.in_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(
        self, h: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        z = self.in_proj(h)
        z = scale * z + shift
        z = torch.nn.functional.silu(z)
        z = self.out_proj(z)
        return h + z


class FilmX1Field(nn.Module):
    """Predicts the clean trajectory ``x1_hat`` given ``(x_t, t, s0)``.

    Conditioning trunk:
        s0 -> [Linear, SiLU, Linear, SiLU] -> cond_vec
        t  -> sinusoidal time embed -> Linear -> SiLU
        c  = cond_vec + time_proj
        for each block i:
            (scale_i, shift_i) <- block_film[i](c)

    Main trunk:
        x_t -> Linear (input projection) -> h
        for each block i:
            h <- FiLMBlock(h, scale_i, shift_i)
        h -> Linear (output) -> x1_hat
    """

    def __init__(self, config: FlowConfigV2):
        super().__init__()
        self.config = config
        H = config.hidden

        # Conditioning trunk: produces a single ``c`` vector per sample.
        self.s0_net = nn.Sequential(
            nn.Linear(config.obs_dim, config.cond_hidden),
            nn.SiLU(),
            nn.Linear(config.cond_hidden, H),
            nn.SiLU(),
        )
        self.time_net = nn.Sequential(
            nn.Linear(config.time_embed_dim, H),
            nn.SiLU(),
        )

        # One (scale, shift) producer per residual block.
        self.block_films = nn.ModuleList([
            nn.Linear(H, 2 * H) for _ in range(config.n_blocks)
        ])

        # Main trunk: input projection -> n_blocks of FiLMBlock -> output.
        self.in_proj = nn.Linear(config.x_dim, H)
        self.blocks = nn.ModuleList([FiLMBlock(H) for _ in range(config.n_blocks)])
        self.out_proj = nn.Linear(H, config.x_dim)

        # Init out_proj near-zero so the network starts close to the identity
        # x1_hat ~ x_t (which is reasonable since at t=1 the target is x_t).
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Init FiLM gates close to (scale=1, shift=0) so blocks act as identity
        # at init. We split each block_film's output into (scale, shift) and
        # initialise the scale-half of the bias to 1.
        for lin in self.block_films:
            nn.init.zeros_(lin.weight)
            with torch.no_grad():
                b = torch.zeros(2 * H)
                b[:H] = 1.0  # scale = 1
                lin.bias.copy_(b)

    def forward(
        self,
        x: torch.Tensor,                # (B, x_dim) -- noised trajectory x_t
        t: torch.Tensor,                # (B,)
        s0: torch.Tensor,               # (B, obs_dim)
    ) -> torch.Tensor:
        """Return the **predicted clean trajectory** ``x1_hat`` (B, x_dim).

        Note: this is x_1, not velocity. Use ``predicted_velocity()`` or
        the closures in ``adapter_v2.py`` for inference-time velocity.
        """
        t_emb = sinusoidal_time_embed(t, self.config.time_embed_dim)
        cond = self.s0_net(s0) + self.time_net(t_emb)        # (B, H)

        h = self.in_proj(x)                                  # (B, H)
        for film_layer, block in zip(self.block_films, self.blocks):
            sc_sh = film_layer(cond)                         # (B, 2H)
            scale, shift = sc_sh.chunk(2, dim=-1)
            h = block(h, scale, shift)

        # The residual branch x + delta lets the network "stay close" to x_t
        # at high t (where x_t is already near x_1), which improves training
        # stability.
        x1_hat = x + self.out_proj(h)
        return x1_hat


def predicted_velocity(
    x1_hat: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1e-2,
) -> torch.Tensor:
    """Analytical conversion: x1-prediction -> velocity at (x_t, t).

    For the OT linear-interpolant flow ``x_t = (1-t) x_0 + t x_1``::

        v_t = x_1 - x_0
            = x_1 - (x_t - t x_1) / (1 - t)
            = (x_1 - x_t) / (1 - t)
    """
    denom = torch.clamp(1.0 - t, min=eps)[:, None]
    return (x1_hat - x_t) / denom


def conditional_fm_loss_v2(
    model: FilmX1Field,
    x1: torch.Tensor,        # (B, x_dim) -- target trajectories
    s0: torch.Tensor,        # (B, obs_dim)
    rng: torch.Generator | None = None,
    n_t_per_traj: int = 64,
    weight_by_dt: bool = True,
) -> torch.Tensor:
    """x_1-parameterised flow-matching loss.

    Loss is simple MSE on the clean-trajectory prediction. Optionally
    weights each sample by ``(1 - t)`` so the loss at the optimum is
    numerically comparable to the standard velocity-MSE (since the
    velocity loss has an implicit ``1/(1-t)`` factor from the
    parameterisation conversion). We default ``weight_by_dt=True``
    because:

    1. It up-weights low-t samples (where the target ``x_1`` is far
       from ``x_t``) -- those carry more learning signal.

    2. It removes the spurious ``1/(1-t)^2`` blow-up near t=1 that
       would otherwise dominate the gradient.

    Empirically this gives smoother convergence than uniform-t MSE.
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
    x1_pred = model(xt, t, s0_rep)
    err2 = (x1_pred - x1_rep) ** 2                   # (N, x_dim)
    if weight_by_dt:
        w = (1.0 - t).clamp(min=1e-3)                # (N,)
        # mean over feature dim, weighted mean over batch:
        per_sample = err2.mean(dim=-1)               # (N,)
        return (w * per_sample).sum() / (w.sum() + 1e-8)
    return err2.mean()


@torch.no_grad()
def sample_trajectory_v2(
    model: FilmX1Field,
    s0: torch.Tensor,         # (B, obs_dim)
    n_steps: int = 32,
    rng: torch.Generator | None = None,
    eps: float = 1e-2,
) -> torch.Tensor:
    """Integrate the x_1-parameterised ODE from t=0 (noise) to t=1.

    At each step we query ``x1_hat = model(x_t, t, s0)`` and step::

        v_t   = (x1_hat - x_t) / max(1 - t, eps)
        x_{t+dt} = x_t + v_t * dt

    This is the Euler scheme on the underlying probability flow. The
    final ``x`` is the model's best estimate of the clean trajectory.
    """
    B = s0.shape[0]
    device = s0.device
    cfg = model.config
    if rng is None:
        x = torch.randn(B, cfg.x_dim, device=device)
    else:
        x = torch.randn(B, cfg.x_dim, generator=rng, device=device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_scalar = i * dt
        t = torch.full((B,), t_scalar, device=device)
        x1_hat = model(x, t, s0)
        denom = max(1.0 - t_scalar, eps)
        v = (x1_hat - x) / denom
        x = x + dt * v
    return x.view(B, cfg.horizon, cfg.action_dim)
