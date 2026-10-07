"""Conditional Flow Matching v6 — Lipman 2023 with v3 trunks.

This module implements the *exact* CFM objective from Lipman et al.,
"Flow Matching for Generative Modeling" (ICLR 2023), using the
configurable trunks (FiLM-MLP, 1D Conv U-Net, Transformer) we already
built for v3.

Compared to v3 (Karras EDM denoising-score-matching), v6 differs in:

  1. **Probability path**: linear interpolant
         ψ_t(x | x_1) = (1 - t) · x_0 + t · x_1,   x_0 ~ N(0, I)
     so x_t = (1 - t) · ε + t · x_1, with ε ~ N(0, I).

  2. **Target**: the conditional vector field is
         u_t(x_t | x_1, x_0) = x_1 - x_0          (= x_1 - ε)
     (the constant velocity along the straight-line path).

  3. **Loss**: ``E_{t, x_1, ε} || v_θ(x_t, t, s) − (x_1 − ε) ||²``
     with ``t ~ U[0, 1 − t_eps]``.

  4. **Sampling**: forward ODE
         dx/dt = v_θ(x_t, t, s),   x(0) = ε ~ N(0, I),   t ∈ [0, 1].
     Integrated with Heun's 2nd-order method (same accuracy as v3's
     EDM Heun) over an evenly-spaced t grid.

  5. **Score-bridge for MBD**: for the linear path the marginal
     score and the marginal velocity are related by
         ∇log p_t(x) = (t · v_θ(x, t) − x) / (1 − t).
     The MBD step index ``k`` is mapped to physical time via
     ``t_k = 1 − σ_k`` where ``σ_k`` comes from the runtime cosine
     σ-schedule (same as v3).  This keeps the score-bridge formula
     literally identical to v1's, but with a much better v_θ.

The trunks (``Conv1DUNetTrunk``, ``TransformerTrunk``) are imported
straight from ``model_v3``: backbone choice is the *only* axis of
variation between v3-EDM and v6-CFM, so we can attribute any
performance delta cleanly to the training objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

# Reuse trunks + helpers from v3 / v2 -- this is the whole point of
# v6: identical capacity, only the objective changes.
from model_v2 import FiLMBlock, sinusoidal_time_embed  # noqa: E402
from model_v3 import Conv1DUNetTrunk, TransformerTrunk  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlowConfigV6:
    """Hyper-parameters for the Conditional Flow Matching v6 model.

    Field layout *intentionally mirrors* ``FlowConfigV3`` so the same
    trunks (Conv1DUNetTrunk, TransformerTrunk) can be reused without
    modification.  The only fields that change semantics:

      - no ``sigma_data`` (no EDM preconditioning)
      - ``t_embed_dim`` replaces ``sigma_embed_dim`` (now embedding
        the physical time t ∈ [0, 1] rather than c_noise = 0.25 log σ).
      - ``t_eps`` is the minimum (1 - t) at training time, to keep the
        score-bridge denominator finite.
    """

    obs_dim: int                                # D_obs (e.g. 46 for can)
    action_dim: int                             # 7
    horizon: int                                # 120
    hidden: int = 768                           # width of trunk + FiLM net
    n_blocks: int = 4                           # number of residual blocks
    cond_hidden: int = 256                      # obs-encoder hidden width
    t_embed_dim: int = 64                       # sinusoidal time embed dim
    t_eps: float = 1e-3                         # min |1 - t| at training & inf
    arch: str = "cfm_v6"                        # adapter dispatch key
    backbone: str = "film_mlp"                  # 'film_mlp' | 'conv1d_unet' | 'transformer'

    @property
    def x_dim(self) -> int:
        return self.horizon * self.action_dim

    # For trunk-side compatibility: the v3 trunks read ``config.hidden``,
    # ``config.horizon``, ``config.action_dim``, ``config.n_blocks``.
    # Those fields are present.  No further shims needed.

    def to_dict(self) -> dict[str, Any]:
        return {
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "horizon": int(self.horizon),
            "hidden": int(self.hidden),
            "n_blocks": int(self.n_blocks),
            "cond_hidden": int(self.cond_hidden),
            "t_embed_dim": int(self.t_embed_dim),
            "t_eps": float(self.t_eps),
            "arch": str(self.arch),
            "backbone": str(self.backbone),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FlowConfigV6":
        return cls(
            obs_dim=int(d["obs_dim"]),
            action_dim=int(d["action_dim"]),
            horizon=int(d["horizon"]),
            hidden=int(d.get("hidden", 768)),
            n_blocks=int(d.get("n_blocks", 4)),
            cond_hidden=int(d.get("cond_hidden", 256)),
            t_embed_dim=int(d.get("t_embed_dim", 64)),
            t_eps=float(d.get("t_eps", 1e-3)),
            arch=str(d.get("arch", "cfm_v6")),
            backbone=str(d.get("backbone", "film_mlp")),
        )


# ---------------------------------------------------------------------------
# CFM velocity field
# ---------------------------------------------------------------------------


class CFMField(nn.Module):
    """Conditional flow-matching velocity field ``v_θ(x_t, t, s)``.

    The conditioning pathway is identical to v3's: obs encoder +
    time encoder are summed to form a single ``cond`` vector that
    is fed (as FiLM scales/shifts) into the trunk.  The trunk is
    one of the v3 backbones, selected by ``config.backbone``.

    Output is the velocity prediction of shape ``(B, x_dim)`` —
    same shape as the input ``x``.  There is no EDM-style preconditioning;
    the network directly outputs ``v``.
    """

    def __init__(self, config: FlowConfigV6):
        super().__init__()
        self.config = config
        H = config.hidden

        # Obs encoder (same shape / init as v3).
        self.s_net = nn.Sequential(
            nn.Linear(config.obs_dim, config.cond_hidden),
            nn.SiLU(),
            nn.Linear(config.cond_hidden, H),
            nn.SiLU(),
        )
        # Time encoder: sinusoidal embedding -> Linear -> SiLU.
        self.t_net = nn.Sequential(
            nn.Linear(config.t_embed_dim, H),
            nn.SiLU(),
        )

        # Backbone-specific trunk.  Re-use the v3 trunks where possible.
        if config.backbone == "film_mlp":
            self.block_films = nn.ModuleList([
                nn.Linear(H, 2 * H) for _ in range(config.n_blocks)
            ])
            self.in_proj = nn.Linear(config.x_dim, H)
            self.blocks = nn.ModuleList(
                [FiLMBlock(H) for _ in range(config.n_blocks)]
            )
            self.out_proj = nn.Linear(H, config.x_dim)
            # Zero-init output projection + FiLM scale=1 / shift=0 init,
            # so v_θ ≈ 0 at start and the network learns to add velocity.
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
            for lin in self.block_films:
                nn.init.zeros_(lin.weight)
                with torch.no_grad():
                    b = torch.zeros(2 * H)
                    b[:H] = 1.0
                    lin.bias.copy_(b)
        elif config.backbone == "conv1d_unet":
            # The trunk class reads config.hidden / horizon / action_dim,
            # which FlowConfigV6 exposes; no shim needed.
            self.trunk = Conv1DUNetTrunk(config)  # type: ignore[arg-type]
        elif config.backbone == "transformer":
            self.trunk = TransformerTrunk(config)  # type: ignore[arg-type]
        else:
            raise ValueError(
                f"unknown backbone: {config.backbone!r}; expected one of "
                "'film_mlp', 'conv1d_unet', 'transformer'."
            )

    def forward(
        self,
        x: torch.Tensor,         # (B, x_dim)
        t: torch.Tensor,         # (B,)    in [0, 1]
        s: torch.Tensor,         # (B, obs_dim)
    ) -> torch.Tensor:
        """Return ``v_θ(x, t, s)`` of shape ``(B, x_dim)``."""
        # Sinusoidal embedding of physical time t (in [0, 1]); the
        # embedding is invariant to scale of t, but we feed t directly
        # since t already lives on an O(1) range (no log/exp needed).
        t_emb = sinusoidal_time_embed(t, self.config.t_embed_dim)

        cond = self.s_net(s) + self.t_net(t_emb)          # (B, H)

        if self.config.backbone == "film_mlp":
            h = self.in_proj(x)                            # (B, H)
            for film_layer, block in zip(self.block_films, self.blocks):
                sc_sh = film_layer(cond)
                scale, shift = sc_sh.chunk(2, dim=-1)
                h = block(h, scale, shift)
            v = self.out_proj(h)                            # (B, x_dim)
        else:
            v = self.trunk(x, cond)                         # (B, x_dim)

        return v


# ---------------------------------------------------------------------------
# Loss / sampler / score helpers
# ---------------------------------------------------------------------------


def cfm_loss(
    model: CFMField,
    x_data: torch.Tensor,         # (B, x_dim) -- clean target trajectories
    s: torch.Tensor,              # (B, obs_dim)
    t_eps: float | None = None,
) -> torch.Tensor:
    """Lipman 2023 CFM loss with linear (OT) interpolant.

    Per-sample:
        ε ~ N(0, I_x)
        t ~ U[0, 1 - t_eps]
        x_t = (1 - t) ε + t x_data
        target = x_data - ε        (= u_t along the straight-line path)
        loss   = || v_θ(x_t, t, s) - target ||²
    """
    B = x_data.shape[0]
    device = x_data.device
    teps = float(model.config.t_eps if t_eps is None else t_eps)
    t = torch.rand(B, device=device) * (1.0 - teps)        # (B,) ∈ [0, 1-teps]
    eps = torch.randn_like(x_data)
    x_t = (1.0 - t)[:, None] * eps + t[:, None] * x_data
    target = x_data - eps                                  # (B, x_dim)
    pred = model(x_t, t, s)                                # (B, x_dim)
    # Mean over both batch and x_dim; matches the v3 EDM-loss convention
    # so the two training scripts produce comparably-scaled curves.
    return ((pred - target) ** 2).mean()


@torch.no_grad()
def cfm_velocity(
    model: CFMField,
    x: torch.Tensor,             # (B, x_dim)
    t: torch.Tensor,             # (B,) -- physical time in [0, 1]
    s: torch.Tensor,             # (B, obs_dim)
) -> torch.Tensor:
    """Return ``v_θ(x, t, s)`` -- no postprocessing."""
    return model(x, t, s)


@torch.no_grad()
def cfm_score(
    model: CFMField,
    x: torch.Tensor,             # (B, x_dim)
    sigma: torch.Tensor,         # (B,) -- MBD σ at the current step; t = 1 - σ
    s: torch.Tensor,             # (B, obs_dim)
    eps: float = 1e-3,
) -> torch.Tensor:
    """Marginal score ``∇log p_t(x)`` derived from the velocity field.

    For the Lipman linear interpolant ``x_t = (1-t) ε + t x_1`` with
    ``ε ~ N(0, I)``, the marginal score and the marginal velocity are
    related by (Lipman 2023 §4.1 + Karras 2022 unification, derivation
    inline in the module docstring)::

        ∇log p_t(x) = (t · v_θ(x, t, s) - x) / (1 - t)

    We map MBD's noise level σ to physical time via ``t = 1 - σ`` so
    that ``σ ∈ [σ_min, σ_max]`` -> ``t ∈ [1 - σ_max, 1 - σ_min]``,
    keeping the score-bridge denominator ``(1 - t) = σ`` bounded
    below by ``σ_min`` (typically ~0.05).
    """
    t = (1.0 - sigma).clamp(min=0.0, max=1.0 - eps)        # (B,)
    v = model(x, t, s)                                     # (B, x_dim)
    one_minus_t = (1.0 - t).clamp(min=eps)                 # (B,)
    return (t[:, None] * v - x) / one_minus_t[:, None]


@torch.no_grad()
def cfm_sample(
    model: CFMField,
    s: torch.Tensor,             # (B, obs_dim)
    n_steps: int = 32,
    t_max: float = 0.999,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Heun (2nd-order) ODE integration from t=0 (noise) to t=t_max.

    Returns the data-side sample reshaped to ``(B, horizon, action_dim)``.
    """
    config = model.config
    B = s.shape[0]
    x_dim = config.x_dim
    H = config.horizon
    A = config.action_dim
    device = s.device

    # Initialise x ~ N(0, I).
    if rng is not None:
        x = torch.randn(B, x_dim, device=device, generator=rng)
    else:
        x = torch.randn(B, x_dim, device=device)

    # Evenly spaced t schedule, 0 -> t_max.
    ts = torch.linspace(0.0, float(t_max), int(n_steps) + 1, device=device)

    for i in range(int(n_steps)):
        t0 = ts[i].expand(B)
        t1 = ts[i + 1].expand(B)
        dt = float(ts[i + 1] - ts[i])

        v0 = model(x, t0, s)                                # (B, x_dim)
        x_pred = x + dt * v0                                # Euler predictor
        v1 = model(x_pred, t1, s)
        x = x + dt * 0.5 * (v0 + v1)                        # Heun corrector

    return x.view(B, H, A)
