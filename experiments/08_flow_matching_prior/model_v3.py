"""v_theta v3 — Karras-EDM noise-conditional score model.

Addresses the three failure modes documented in
``results/sweep_v2_2026_05_16/MORNING_BRIEF.md`` for v2:

1. **σ-preservation mismatch.** v2 was trained as OT-CFM on
   ``x_t = (1-t) x_0 + t x_1`` with ``x_0 ~ N(0, I)``.  At MBD's
   actual iterate ``mu + sigma * eps``, the marginal is
   ``p^data * N(0, sigma^2 I)`` — a *different* distribution than
   v2's transport-path marginal.  v3 trains on ``x_0 + sigma * eps``
   *directly*, with ``sigma`` sampled log-uniform over MBD's actual
   schedule ``[sigma_min, sigma_max]``.  No bridge formula is
   needed: the network output *is* the denoised mean, and the
   score is ``(D(x; sigma) - x) / sigma^2``.

2. **t=0-only training.** v2's dataset was per-demo: ``(s_0, full
   demo)``.  Mid-trajectory states are OOD, which broke replanning
   (Phase 8: R=1 → 0/60).  v3's dataset is per-chunk
   ``(s_t, a_{t:t+H})`` for every ``t`` (see ``dataset_v3.py``),
   matching BC's per-state training distribution exactly.

3. **Wrong score object.** v2's "linear" / "quadratic" bridge
   (Phase 7) tried to convert flow velocity to a score; the quadratic
   variant — which is the textbook bridge formula for OT-CFM — gave
   4/60 at λ=0.1 (vs 10/60 at λ=0).  This file removes the
   conversion entirely: training output *is* the score (modulo
   Karras preconditioning, which is a fixed σ-dependent rescaling).

Architecture
------------
We keep v2's FiLM block from ``model_v2.FiLMBlock`` (it was the v1
→ v2 fix that closed the per-demo conditioning gap, certified by
v2's sample-norm 1.45 vs target 1.24).  The only architectural
change is **the conditioning scalar**: v2 used ``t ∈ [0, 1]``,
v3 uses ``c_noise(sigma) = 0.25 * log(sigma)`` (Karras 2022 eq. 213).

The forward pass is wrapped in Karras EDM preconditioning::

    c_in(σ)    = 1 / sqrt(σ² + σ_data²)
    c_out(σ)   = σ · σ_data / sqrt(σ² + σ_data²)
    c_skip(σ)  = σ_data² / (σ² + σ_data²)
    c_noise(σ) = 0.25 · log σ

    D(x; σ, s) = c_skip · x + c_out · F_θ(c_in · x, c_noise, s)

Loss::

    L(θ) = E_{x_0, σ, ε}[ λ(σ) || D(x_0 + σε; σ, s) - x_0 ||² ]
    λ(σ) = (σ² + σ_data²) / (σ · σ_data)²

This is the same objective that powers Karras EDM / Stable
Diffusion, specialised to our 7-DoF × 120-step trajectories.

References
----------
- Karras, Aittala, Aila, Laine.  "Elucidating the design space of
  diffusion-based generative models."  NeurIPS 2022.  (Table 1,
  the ``--EDM--`` column).
- Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole.  "Score-based
  generative modeling through SDEs."  ICLR 2021.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Reuse v2's FiLM block — proven to handle s_0 conditioning.
from model_v2 import FiLMBlock, sinusoidal_time_embed


@dataclass(frozen=True)
class FlowConfigV3:
    """Hyper-parameters for the Karras-EDM noise-conditional score model.

    The ``sigma_*`` fields are baked into the checkpoint so the
    inference-time schedule matches the training-time distribution
    of ``sigma``.  MBD's ``cosine_sigma_schedule(K, sigma_max,
    sigma_min)`` must use parameters that lie inside
    ``[sigma_min, sigma_max]`` for v3's score to be well-calibrated.
    """

    obs_dim: int                                # D_obs (e.g. 30 for can)
    action_dim: int                             # 7
    horizon: int                                # 120
    hidden: int = 768                           # width of trunk + FiLM net
    n_blocks: int = 4                           # number of residual FiLM blocks
    cond_hidden: int = 256                      # FiLM-conditioning MLP width
    sigma_embed_dim: int = 64                   # sinusoidal log-sigma embed
    sigma_min: float = 0.01                     # min noise level during training
    sigma_max: float = 1.0                      # max noise level during training
    sigma_data: float = 0.5                     # Karras EDM data-scale prior
    arch: str = "karras_edm_v3"                 # adapter dispatch key
    backbone: str = "film_mlp"                  # trunk: 'film_mlp' (v3), 'conv1d_unet' (DP-style),
                                                # or 'transformer' (BESO-style).  All three keep
                                                # the *same* EDM training objective (DSM, Karras
                                                # preconditioning, log-uniform σ sampling, Heun
                                                # sampler). The only thing that changes is how
                                                # F_θ(x_in, cond) maps an (H, A) chunk to its
                                                # raw output. The MLP variant flattens the chunk
                                                # (→ destroys temporal structure); UNet/Transformer
                                                # preserve it.

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
            "sigma_embed_dim": int(self.sigma_embed_dim),
            "sigma_min": float(self.sigma_min),
            "sigma_max": float(self.sigma_max),
            "sigma_data": float(self.sigma_data),
            "arch": str(self.arch),
            "backbone": str(self.backbone),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FlowConfigV3":
        return cls(
            obs_dim=int(d["obs_dim"]),
            action_dim=int(d["action_dim"]),
            horizon=int(d["horizon"]),
            hidden=int(d["hidden"]),
            n_blocks=int(d["n_blocks"]),
            cond_hidden=int(d["cond_hidden"]),
            sigma_embed_dim=int(d.get("sigma_embed_dim", 64)),
            sigma_min=float(d.get("sigma_min", 0.01)),
            sigma_max=float(d.get("sigma_max", 1.0)),
            sigma_data=float(d.get("sigma_data", 0.5)),
            arch=str(d.get("arch", "karras_edm_v3")),
            # Default to 'film_mlp' so checkpoints predating the v4
            # refactor (i.e., v3 trained before 2026-05-23) load with
            # the original MLP trunk and reproduce bit-for-bit.
            backbone=str(d.get("backbone", "film_mlp")),
        )


# ---------------------------------------------------------------------------
# Karras EDM preconditioning -- closed-form σ-dependent rescaling that
# (a) keeps the network's inputs O(1) regardless of σ, (b) keeps
# outputs interpretable as a denoised mean (additive σ_data offset).
# These four functions are exactly Karras 2022 Table 1, EDM column.
# ---------------------------------------------------------------------------


def edm_scalings(
    sigma: torch.Tensor, sigma_data: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(c_in, c_out, c_skip, c_noise)`` for batched sigma.

    Inputs: ``sigma`` is ``(B,)``.  Outputs are ``(B,)`` each.
    """
    s2 = sigma * sigma
    sd2 = float(sigma_data) ** 2
    denom = torch.sqrt(s2 + sd2)
    c_in = 1.0 / denom
    c_out = sigma * float(sigma_data) / denom
    c_skip = sd2 / (s2 + sd2)
    # Karras's c_noise.  ``0.25 * log(σ)`` -- shifts the very long
    # σ tails to the order of 1.
    c_noise = 0.25 * torch.log(sigma.clamp(min=1e-8))
    return c_in, c_out, c_skip, c_noise


# ---------------------------------------------------------------------------
# Backbones (v4 refactor)
# ---------------------------------------------------------------------------
#
# All three trunks expose the same interface::
#
#     trunk(x_in: (B, x_dim), cond: (B, hidden)) -> F_raw: (B, x_dim)
#
# where ``x_in`` is ``c_in(σ) * x`` (Karras's preconditioned input) and
# ``cond`` is the *fused* obs + sigma conditioning vector produced by the
# parent ``FilmEdmField`` (so cond pathway is shared across backbones).
#
# Why share the cond pathway and only swap the trunk: this isolates the
# *backbone* as the single variable in the v3 → v4 ablation.  Any
# difference between v3 (film_mlp) and v4_unet / v4_transformer is then
# attributable to *how the chunk is processed*, not to the obs encoder.


class Conv1DResBlock(nn.Module):
    """1D conv residual block with FiLM conditioning (Diffusion-Policy style).

    The block applies two 1D convolutions over the temporal axis with a
    FiLM (scale, shift) injected between them.  The FiLM parameters come
    from a per-block linear projection of ``cond``, matching the
    Diffusion Policy ``ConditionalResidualBlock1D`` from Chi 2023 § 5.2.

    Group normalisation is used in place of batch norm: stable for the
    small effective batch sizes we see (B=256 × n_sigma=8 = 2048 samples
    per gradient step, but with multiple σ per chunk the effective N is
    less independent than that, so GN is safer).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        pad = kernel_size // 2
        # Choose group count that divides both channel counts.
        groups_in = math.gcd(n_groups, in_channels) if in_channels >= n_groups else 1
        groups_out = math.gcd(n_groups, out_channels) if out_channels >= n_groups else 1
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(groups_out, out_channels)
        # FiLM projector: cond -> (2 * out_channels) -> split into scale, shift.
        self.film = nn.Linear(cond_dim, 2 * out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad)
        self.norm2 = nn.GroupNorm(groups_out, out_channels)
        # Skip connection: identity if channels match, else 1x1 conv.
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        # FiLM init: scale = 1, shift = 0 (block is identity at init).
        nn.init.zeros_(self.film.weight)
        with torch.no_grad():
            b = torch.zeros(2 * out_channels)
            b[:out_channels] = 1.0
            self.film.bias.copy_(b)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T),  cond: (B, cond_dim)
        h = self.conv1(x)
        h = self.norm1(h)
        sc_sh = self.film(cond)                                # (B, 2 C_out)
        scale, shift = sc_sh.chunk(2, dim=-1)
        h = h * scale[:, :, None] + shift[:, :, None]          # broadcast on T
        h = torch.nn.functional.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = torch.nn.functional.silu(h)
        return h + self.skip(x)


class Conv1DUNetTrunk(nn.Module):
    """Diffusion-Policy-style 1D conv U-Net trunk.

    The trunk treats the chunk as a temporal signal of shape ``(B, A, H)``
    and applies a symmetric down → mid → up pyramid with skip connections.
    For our default H=120 (divisible by 4), we use two levels of stride-2
    downsampling, two mid blocks, then two upsampling levels.

    Channel layout (driven by ``config.hidden`` so model size scales
    automatically):

        base = hidden // 4         (e.g. hidden=768  -> base=192)
        levels: [base, 2*base, 4*base]
                                   (e.g. 192, 384, 768)

    Output shape matches the input: F_raw of shape ``(B, x_dim)`` where
    ``x_dim = H * A``.  The output ``Conv1d`` is zero-initialised so
    ``F_raw == 0`` at init, which mirrors v3's identity-at-init
    behaviour (``D(x; σ, s) ≈ c_skip · x``).
    """

    def __init__(self, config: FlowConfigV3):
        super().__init__()
        H = config.horizon
        A = config.action_dim
        cond_dim = config.hidden
        self.H = H
        self.A = A

        base = max(config.hidden // 4, 32)
        ch = [base, base * 2, base * 4]

        # Input projection: (B, A, H) -> (B, base, H)
        self.in_proj = nn.Conv1d(A, ch[0], kernel_size=3, padding=1)

        # Encoder: two downsampling stages.
        self.down1 = Conv1DResBlock(ch[0], ch[0], cond_dim)
        self.pool1 = nn.AvgPool1d(2)
        self.proj1 = nn.Conv1d(ch[0], ch[1], kernel_size=1)
        self.down2 = Conv1DResBlock(ch[1], ch[1], cond_dim)
        self.pool2 = nn.AvgPool1d(2)
        self.proj2 = nn.Conv1d(ch[1], ch[2], kernel_size=1)

        # Mid: two residual blocks at the bottleneck.
        self.mid1 = Conv1DResBlock(ch[2], ch[2], cond_dim)
        self.mid2 = Conv1DResBlock(ch[2], ch[2], cond_dim)

        # Decoder: two upsampling stages with skip-concat from encoder.
        self.up1_proj = nn.Conv1d(ch[2] + ch[1], ch[1], kernel_size=1)
        self.up1 = Conv1DResBlock(ch[1], ch[1], cond_dim)
        self.up2_proj = nn.Conv1d(ch[1] + ch[0], ch[0], kernel_size=1)
        self.up2 = Conv1DResBlock(ch[0], ch[0], cond_dim)

        # Output projection: zero-init so F_raw == 0 at start (identity-at-init).
        self.out_proj = nn.Conv1d(ch[0], A, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_in: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x_in: (B, x_dim) -- already multiplied by c_in
        B = x_in.shape[0]
        # Reshape to (B, H, A) -> (B, A, H) for Conv1d's channel-first layout.
        x = x_in.view(B, self.H, self.A).transpose(1, 2)        # (B, A, H)

        h = self.in_proj(x)                                      # (B, c0, H)

        h_d1 = self.down1(h, cond)                               # (B, c0, H)
        h = self.proj1(self.pool1(h_d1))                         # (B, c1, H/2)
        h_d2 = self.down2(h, cond)                               # (B, c1, H/2)
        h = self.proj2(self.pool2(h_d2))                         # (B, c2, H/4)

        h = self.mid1(h, cond)
        h = self.mid2(h, cond)                                   # (B, c2, H/4)

        # Up 1: H/4 -> H/2
        h = torch.nn.functional.interpolate(
            h, scale_factor=2.0, mode="nearest",
        )                                                        # (B, c2, H/2)
        h = self.up1_proj(torch.cat([h, h_d2], dim=1))           # (B, c1, H/2)
        h = self.up1(h, cond)

        # Up 2: H/2 -> H
        h = torch.nn.functional.interpolate(
            h, scale_factor=2.0, mode="nearest",
        )                                                        # (B, c1, H)
        h = self.up2_proj(torch.cat([h, h_d1], dim=1))           # (B, c0, H)
        h = self.up2(h, cond)

        F = self.out_proj(h)                                     # (B, A, H)
        F = F.transpose(1, 2).reshape(B, self.H * self.A)        # (B, x_dim)
        return F


class TransformerTrunk(nn.Module):
    """BESO-style Transformer trunk (encoder with conditioning token).

    The chunk is treated as a sequence of H tokens, each of dimension A.
    Each token is projected to ``hidden_token`` dim and a learned
    positional embedding is added.  A single conditioning token, formed
    by projecting ``cond`` to the same dim, is prepended.  The whole
    sequence (length H+1) is then processed by a stack of
    ``nn.TransformerEncoderLayer`` blocks.  Finally we drop the
    conditioning token and project back to A per timestep.

    This mirrors BESO (Reuss 2023) at a structural level while staying
    well within PyTorch primitives.  The output projection is zero-init
    so ``F_raw == 0`` at start (identity-at-init, same as the U-Net).
    """

    def __init__(self, config: FlowConfigV3):
        super().__init__()
        self.H = config.horizon
        self.A = config.action_dim
        # Choose token dim and head count from config.hidden.
        hidden_token = max(config.hidden // 2, 128)
        n_heads = 8
        if hidden_token % n_heads != 0:
            # Round hidden_token down to a multiple of n_heads.
            hidden_token = (hidden_token // n_heads) * n_heads
        self.hidden_token = hidden_token

        self.token_proj = nn.Linear(self.A, hidden_token)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.H, hidden_token) * 0.02
        )
        # Conditioning token: project the parent's (B, cond_dim) vector
        # to (B, 1, hidden_token).
        self.cond_proj = nn.Linear(config.hidden, hidden_token)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_token,
            nhead=n_heads,
            dim_feedforward=4 * hidden_token,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_blocks)

        self.out_proj = nn.Linear(hidden_token, self.A)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Pos embed init -- small Gaussian (the *0.02 above is wrong
        # because zeros() returns zeros; do this here explicitly).
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x_in: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x_in: (B, x_dim) -- already multiplied by c_in
        B = x_in.shape[0]
        x = x_in.view(B, self.H, self.A)                         # (B, H, A)

        tokens = self.token_proj(x) + self.pos_embed             # (B, H, D)
        cond_token = self.cond_proj(cond).unsqueeze(1)           # (B, 1, D)
        seq = torch.cat([cond_token, tokens], dim=1)             # (B, H+1, D)

        seq = self.encoder(seq)                                   # (B, H+1, D)

        # Drop the cond token and project back to A.
        tokens_out = seq[:, 1:, :]                                # (B, H, D)
        F = self.out_proj(tokens_out)                             # (B, H, A)
        return F.reshape(B, self.H * self.A)                      # (B, x_dim)


# ---------------------------------------------------------------------------
# FilmEdmField -- dispatches to one of three trunks based on config.backbone
# ---------------------------------------------------------------------------


class FilmEdmField(nn.Module):
    """Network ``F_θ`` whose preconditioned wrapper is the denoiser ``D``.

    Conditioning pathway (shared across all backbones):
        s_t      -> [Linear, SiLU, Linear, SiLU] -> cond_vec
        c_noise  -> sinusoidal embed -> Linear -> SiLU
        cond     = cond_vec + sigma_proj                          (B, hidden)

    Backbone (selected by ``config.backbone``):
        'film_mlp'      -> v3 default; flatten chunk, FiLM-MLP residual stack
        'conv1d_unet'   -> Diffusion-Policy-style 1D conv U-Net
        'transformer'   -> BESO-style Transformer encoder with cond token

    Composition:
        D(x; σ, s) = c_skip · x + c_out · F_raw

    For ``backbone='film_mlp'`` the trunk is kept *bit-for-bit identical*
    to the original v3 implementation (same parameter names, same init)
    so checkpoints trained before this refactor reload unchanged.
    """

    def __init__(self, config: FlowConfigV3):
        super().__init__()
        self.config = config
        H = config.hidden

        # Obs conditioning: same shape as v2's s0_net.
        self.s_net = nn.Sequential(
            nn.Linear(config.obs_dim, config.cond_hidden),
            nn.SiLU(),
            nn.Linear(config.cond_hidden, H),
            nn.SiLU(),
        )
        # σ conditioning: c_noise -> sinusoidal embed -> Linear -> SiLU.
        self.sigma_net = nn.Sequential(
            nn.Linear(config.sigma_embed_dim, H),
            nn.SiLU(),
        )

        # Backbone-specific trunk.  We keep the v3 'film_mlp' layout
        # (block_films, in_proj, blocks, out_proj as direct attributes
        # of self) so old checkpoints load with the same key names.
        if config.backbone == "film_mlp":
            self.block_films = nn.ModuleList([
                nn.Linear(H, 2 * H) for _ in range(config.n_blocks)
            ])
            self.in_proj = nn.Linear(config.x_dim, H)
            self.blocks = nn.ModuleList(
                [FiLMBlock(H) for _ in range(config.n_blocks)]
            )
            self.out_proj = nn.Linear(H, config.x_dim)
            # Identity-at-init: zero out_proj, FiLM scale=1 shift=0.
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
            for lin in self.block_films:
                nn.init.zeros_(lin.weight)
                with torch.no_grad():
                    b = torch.zeros(2 * H)
                    b[:H] = 1.0
                    lin.bias.copy_(b)
        elif config.backbone == "conv1d_unet":
            self.trunk = Conv1DUNetTrunk(config)
        elif config.backbone == "transformer":
            self.trunk = TransformerTrunk(config)
        else:
            raise ValueError(
                f"unknown backbone: {config.backbone!r}; expected one of "
                "'film_mlp', 'conv1d_unet', 'transformer'."
            )

    def forward(
        self,
        x: torch.Tensor,         # (B, x_dim) -- noised trajectory x_σ
        sigma: torch.Tensor,     # (B,)
        s: torch.Tensor,         # (B, obs_dim) -- conditioning obs
    ) -> torch.Tensor:
        """Return the denoiser output ``D(x; σ, s)`` of shape ``(B, x_dim)``.

        This is *the denoised mean*: at the optimum,
        ``D(x_0 + σε; σ, s) = E[x_0 | x_0 + σε, s]``.
        """
        c_in, c_out, c_skip, c_noise = edm_scalings(sigma, self.config.sigma_data)
        sigma_emb = sinusoidal_time_embed(c_noise, self.config.sigma_embed_dim)

        cond = self.s_net(s) + self.sigma_net(sigma_emb)        # (B, H)
        x_in = c_in[:, None] * x                                 # (B, x_dim)

        if self.config.backbone == "film_mlp":
            h = self.in_proj(x_in)                               # (B, H)
            for film_layer, block in zip(self.block_films, self.blocks):
                sc_sh = film_layer(cond)                         # (B, 2H)
                scale, shift = sc_sh.chunk(2, dim=-1)
                h = block(h, scale, shift)
            F_raw = self.out_proj(h)                             # (B, x_dim)
        else:
            F_raw = self.trunk(x_in, cond)                       # (B, x_dim)

        D = c_skip[:, None] * x + c_out[:, None] * F_raw
        return D


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def sample_sigma_loguniform(
    n: int,
    sigma_min: float,
    sigma_max: float,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Log-uniform sampling over ``[sigma_min, sigma_max]``.

    Equivalent to ``exp(U(log σ_min, log σ_max))``.  This matches
    the bias of Karras's recommended training schedule and ensures
    we spend comparable training budget on low- and high-σ regions.
    """
    if generator is None:
        u = torch.rand(n, device=device)
    else:
        u = torch.rand(n, generator=generator, device=device)
    log_lo = math.log(float(sigma_min))
    log_hi = math.log(float(sigma_max))
    return torch.exp(u * (log_hi - log_lo) + log_lo)


def edm_loss(
    model: FilmEdmField,
    x0: torch.Tensor,           # (B, x_dim) -- clean target trajectories
    s: torch.Tensor,            # (B, obs_dim)
    generator: torch.Generator | None = None,
    n_sigma_per_traj: int = 8,
) -> torch.Tensor:
    """Karras EDM denoising-score-matching loss.

    For each of the ``B`` (x_0, s) pairs we draw ``n_sigma_per_traj``
    independent ``(σ, ε)`` pairs.  Each contributes a single MSE
    term weighted by ``λ(σ) = (σ² + σ_data²) / (σ σ_data)²``.

    The weighting ``λ(σ)`` makes the loss numerically comparable
    across σ; without it the σ ≈ σ_data band dominates the gradient
    and the model under-fits both tails.  See Karras 2022 § 5.1.
    """
    cfg = model.config
    B = x0.shape[0]
    K = max(int(n_sigma_per_traj), 1)
    if K > 1:
        x0_rep = x0.repeat_interleave(K, dim=0)
        s_rep = s.repeat_interleave(K, dim=0)
    else:
        x0_rep = x0
        s_rep = s
    N = x0_rep.shape[0]
    device = x0.device

    sigma = sample_sigma_loguniform(
        N, cfg.sigma_min, cfg.sigma_max, device=device, generator=generator,
    )
    if generator is None:
        eps = torch.randn_like(x0_rep)
    else:
        eps = torch.randn(x0_rep.shape, generator=generator, device=device)
    x_sigma = x0_rep + sigma[:, None] * eps

    D = model(x_sigma, sigma, s_rep)                             # (N, x_dim)
    err2 = (D - x0_rep) ** 2                                     # (N, x_dim)

    sd2 = float(cfg.sigma_data) ** 2
    weight = (sigma * sigma + sd2) / ((sigma * float(cfg.sigma_data)) ** 2)
    # Mean over feature dim, weighted mean over batch.
    per_sample = err2.mean(dim=-1)
    return (weight * per_sample).mean()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def edm_score(
    model: FilmEdmField,
    x: torch.Tensor,             # (B, x_dim)
    sigma: torch.Tensor,         # (B,)
    s: torch.Tensor,             # (B, obs_dim)
) -> torch.Tensor:
    """Score ``∇ log p^data_σ(x | s) ≈ (D(x; σ, s) - x) / σ²``.

    This is the *correct* score-bridge for the variance-exploding
    forward process — no extra ``(1-t)`` or ``(1-t)²`` factor.  It
    follows directly from Tweedie's formula applied to
    ``x = x_0 + σε`` with isotropic ``N(0, σ²I)`` noise.
    """
    D = model(x, sigma, s)
    return (D - x) / (sigma[:, None] ** 2).clamp(min=1e-8)


@torch.no_grad()
def edm_sample(
    model: FilmEdmField,
    s: torch.Tensor,             # (B, obs_dim)
    n_steps: int = 32,
    sigma_init: float | None = None,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Karras 2nd-order Heun sampler from σ_max -> σ_min.

    Returns ``(B, horizon, action_dim)`` denoised trajectories
    conditioned on ``s``.  Used for warm-start construction.
    """
    cfg = model.config
    B = s.shape[0]
    device = s.device
    sigma_max = float(sigma_init) if sigma_init is not None else float(cfg.sigma_max)
    sigma_min = float(cfg.sigma_min)

    # Karras's σ schedule (eq. 5 of the paper) with rho=7.
    rho = 7.0
    step_indices = torch.arange(n_steps, device=device, dtype=torch.float32)
    t = (
        sigma_max ** (1 / rho)
        + step_indices / max(n_steps - 1, 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    # Append 0 sentinel so the final Heun step is correct.
    t = torch.cat([t, torch.zeros(1, device=device)])

    if rng is None:
        x = torch.randn(B, cfg.x_dim, device=device) * sigma_max
    else:
        x = torch.randn(B, cfg.x_dim, generator=rng, device=device) * sigma_max

    for i in range(n_steps):
        sigma_cur = t[i].expand(B)
        sigma_next = t[i + 1].expand(B)
        # Euler step.
        d_cur = (x - model(x, sigma_cur, s)) / sigma_cur[:, None].clamp(min=1e-8)
        x_next = x + (sigma_next - sigma_cur)[:, None] * d_cur
        # Heun correction (skipped at the final step where σ_next = 0).
        if i < n_steps - 1:
            d_next = (
                (x_next - model(x_next, sigma_next, s))
                / sigma_next[:, None].clamp(min=1e-8)
            )
            x_next = x + (sigma_next - sigma_cur)[:, None] * 0.5 * (d_cur + d_next)
        x = x_next

    return x.view(B, cfg.horizon, cfg.action_dim)
