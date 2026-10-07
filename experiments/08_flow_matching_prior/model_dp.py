"""Diffusion Policy architecture + DDPM objective, trained on our data.

Reimplements ``ConditionalUnet1D`` from Chi et al., "Diffusion Policy:
Visuomotor Policy Learning via Action Diffusion" (RSS 2023), and the
matching DDPM ε-prediction training objective (Ho et al. 2020).  Both
are reproduced as faithfully as the published code in
``columbia-ai-robotics/diffusion_policy``:

  - 1D conv U-Net with FiLM conditioning  (cond_predict_scale=True)
  - GroupNorm + Mish everywhere
  - Strided conv downsample / transposed conv upsample
  - Diffusion timestep -> sinusoidal pos-emb -> MLP -> concat with obs
  - ε-prediction loss with squaredcos_cap_v2 β-schedule, T=100 steps
  - DDPM ancestral sampler at inference (also exposed via score)

Why reimplement instead of loading their pretrained checkpoint
-------------------------------------------------------------

Their released checkpoint depends on a hydra config tree, a different
obs space (lowdim feat. selection differs from our v141 obs), and a
different action normaliser ([-1, 1] min-max vs our z-score from
the demo bank).  Each mismatch is an integration bug waiting to
happen, and the value of "their exact weights" is small because what
we care about is the *architecture + objective* contribution, not
the specific training random seed they used.

By training their architecture on our data, we get a clean axis-
isolated ablation::

    v3   = our trunk(s) + Karras EDM      (one objective baseline)
    v6   = our trunk(s) + Lipman CFM      (different objective)
    v7   = DP's UNet1D  + Ho et al. DDPM  (different arch + objective)

All four trained on the same 200-demo can/ph dataset, same horizon,
same obs-dim, same z-score normalisation.  Any env-success delta is
attributable to the model design, not to data shimmer.

If, after v7 trains, we want to *also* check DP's released-checkpoint
performance, we can add a separate ``DPCheckpointAdapter`` later --
but the architecture+objective contribution is the scientifically
interesting one and that's what v7 captures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlowConfigDP:
    """Hyper-parameters for the DP-architecture model + DDPM objective.

    Field names borrow from ``FlowConfigV3`` where they overlap so the
    same data loader (``dataset_v3.load_chunks``) feeds all training
    scripts without modification.

    Defaults reproduce DP's "low-dim" recipe from their config
    ``diffusion_policy/config/task/lowdim_can_*.yaml`` and the
    "low-dim image" workspace ``train_diffusion_unet_lowdim``:

      - down_dims = (256, 512, 1024)
      - diffusion_step_embed_dim = 256
      - kernel_size = 5, n_groups = 8
      - cond_predict_scale = True   (FiLM-style modulation of conv features)
      - num_train_timesteps = 100, squaredcos_cap_v2 schedule
      - prediction_type = epsilon
    """

    obs_dim: int                                # D_obs (e.g. 46 for can/v141)
    action_dim: int                             # 7
    horizon: int                                # 120
    n_obs_steps: int = 1                        # T_obs.  DP paper uses 2; we
                                                # default to 1 to share the v3
                                                # data loader.  Set to 2 only
                                                # after we add an obs-stacking
                                                # data variant.
    diffusion_step_embed_dim: int = 256
    down_dims: tuple = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True             # FiLM scale+shift (DP default)
    # DDPM β-schedule
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    arch: str = "dp_unet1d"

    @property
    def x_dim(self) -> int:
        return self.horizon * self.action_dim

    @property
    def global_cond_dim(self) -> int:
        return self.obs_dim * self.n_obs_steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "horizon": int(self.horizon),
            "n_obs_steps": int(self.n_obs_steps),
            "diffusion_step_embed_dim": int(self.diffusion_step_embed_dim),
            "down_dims": list(self.down_dims),
            "kernel_size": int(self.kernel_size),
            "n_groups": int(self.n_groups),
            "cond_predict_scale": bool(self.cond_predict_scale),
            "num_train_timesteps": int(self.num_train_timesteps),
            "beta_schedule": str(self.beta_schedule),
            "beta_start": float(self.beta_start),
            "beta_end": float(self.beta_end),
            "arch": str(self.arch),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FlowConfigDP":
        return cls(
            obs_dim=int(d["obs_dim"]),
            action_dim=int(d["action_dim"]),
            horizon=int(d["horizon"]),
            n_obs_steps=int(d.get("n_obs_steps", 1)),
            diffusion_step_embed_dim=int(d.get("diffusion_step_embed_dim", 256)),
            down_dims=tuple(d.get("down_dims", (256, 512, 1024))),
            kernel_size=int(d.get("kernel_size", 5)),
            n_groups=int(d.get("n_groups", 8)),
            cond_predict_scale=bool(d.get("cond_predict_scale", True)),
            num_train_timesteps=int(d.get("num_train_timesteps", 100)),
            beta_schedule=str(d.get("beta_schedule", "squaredcos_cap_v2")),
            beta_start=float(d.get("beta_start", 1e-4)),
            beta_end=float(d.get("beta_end", 2e-2)),
            arch=str(d.get("arch", "dp_unet1d")),
        )


# ---------------------------------------------------------------------------
# DDPM β-schedule
# ---------------------------------------------------------------------------


def make_beta_schedule(
    num_timesteps: int,
    schedule: str = "squaredcos_cap_v2",
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> torch.Tensor:
    """Return ``β_t`` for ``t = 0..T-1``.

    ``squaredcos_cap_v2`` reproduces the schedule used by DP /
    diffusers' DDPMScheduler.  It is the "improved DDPM" cosine
    schedule of Nichol & Dhariwal 2021, with a small offset (0.008)
    to avoid β = 0 at t = 0, and clipped to β <= 0.999 to keep the
    SNR finite at the noise end.
    """
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_timesteps)
    if schedule == "squaredcos_cap_v2":
        steps = num_timesteps + 1
        t = torch.linspace(0.0, 1.0, steps)
        alphas_bar = torch.cos((t + 0.008) / 1.008 * math.pi / 2.0) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
        return betas.clamp(max=0.999)
    raise ValueError(f"unknown β-schedule: {schedule!r}")


class DDPMSchedule:
    """Pre-computed DDPM constants for a fixed β-schedule.

    Exposes:
      - ``betas``                : (T,)
      - ``alphas``               : (T,) = 1 - β
      - ``alphas_cumprod``       : (T,)
      - ``sqrt_alphas_cumprod``  : (T,) = √(α̅_t)
      - ``sqrt_one_minus_alphas_cumprod`` : (T,) = √(1 − α̅_t)  = our σ_t
      - ``posterior_variance``   : (T,) for ancestral sampling

    The ``σ_t`` notation above (= √(1 − α̅_t)) is the VP-SDE noise
    level: ``x_t = √(α̅_t) · x_0 + σ_t · ε``.  The conversion to
    Karras-EDM σ is direct, so the ε-to-score formula

        ∇log p_t(x) = −ε_θ(x, t) / σ_t

    plugs straight into the MBD score-bridge.
    """

    def __init__(
        self,
        num_timesteps: int,
        schedule: str = "squaredcos_cap_v2",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | str = "cpu",
    ):
        self.num_timesteps = int(num_timesteps)
        betas = make_beta_schedule(num_timesteps, schedule, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1), alphas_cumprod[:-1]], dim=0
        )
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        dev = torch.device(device)
        self.betas = betas.to(dev)
        self.alphas = alphas.to(dev)
        self.alphas_cumprod = alphas_cumprod.to(dev)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(dev)
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(dev)
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(dev)
        self.posterior_variance = posterior_variance.to(dev)

    def to(self, device: torch.device | str) -> "DDPMSchedule":
        dev = torch.device(device)
        for k in (
            "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
        ):
            setattr(self, k, getattr(self, k).to(dev))
        return self

    def nearest_t_for_sigma(self, sigma: float) -> int:
        """Map a continuous σ (Karras-EDM convention) to the nearest
        DDPM integer timestep so we can query the network at it.

        Definition: σ in our MBD schedule corresponds to the VP-SDE noise
        level ``σ_t = √(1 − α̅_t)``.  Return the t whose ``σ_t`` is
        closest to ``sigma``.  Used by the score-bridge to translate the
        runtime σ-schedule into a DDPM step index.
        """
        s = float(sigma)
        diffs = torch.abs(self.sqrt_one_minus_alphas_cumprod - s)
        return int(torch.argmin(diffs).item())


# ---------------------------------------------------------------------------
# Network components (1D conv U-Net building blocks)
# ---------------------------------------------------------------------------


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for the (integer) diffusion step."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,) -- float or long; cast to float for sin/cos
        device = x.device
        half_dim = self.dim // 2
        denom = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -denom)
        args = x.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish (the unit DP builds its U-Net out of)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, n_groups: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_ch),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample1d(nn.Module):
    """Strided conv downsample (factor 2)."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """Transposed conv upsample (factor 2)."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    """Residual block with FiLM-style conditioning (DP's core unit).

    Two conv blocks with a FiLM modulation between them.  If
    ``cond_predict_scale=True`` the conditioning produces both a scale
    and a shift (DP's default).  Otherwise just an additive bias.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()
        self.cond_predict_scale = cond_predict_scale
        self.out_ch = int(out_ch)
        self.block1 = Conv1dBlock(in_ch, out_ch, kernel_size, n_groups)
        self.block2 = Conv1dBlock(out_ch, out_ch, kernel_size, n_groups)

        cond_channels = 2 * out_ch if cond_predict_scale else out_ch
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )
        # Residual projection: 1x1 conv if dims differ, else identity.
        self.residual = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """``x``: (B, C_in, T)  ``cond``: (B, cond_dim)  -> (B, C_out, T)."""
        h = self.block1(x)                                      # (B, C_out, T)
        embed = self.cond_encoder(cond)                         # (B, 2*C_out) or (B, C_out)
        if self.cond_predict_scale:
            embed = embed.view(embed.shape[0], 2, self.out_ch, 1)
            scale = embed[:, 0]                                 # (B, C_out, 1)
            shift = embed[:, 1]
            h = scale * h + shift
        else:
            h = h + embed[..., None]
        h = self.block2(h)
        return h + self.residual(x)


class ConditionalUnet1D(nn.Module):
    """1D Conv U-Net used by Diffusion Policy.

    Inputs at forward time::

        sample      : (B, T, action_dim)
        timestep    : (B,) integer DDPM step in [0, num_train_timesteps)
        global_cond : (B, n_obs_steps * obs_dim)   -- T_obs frames of obs

    Output::

        (B, T, action_dim)   -- ε-prediction (noise) in normalised
                                 action space.

    ``T`` (horizon) must be divisible by 2^(len(down_dims)-1) so the
    U-Net's down/up samples line up cleanly.  For DP's default
    ``down_dims=(256, 512, 1024)`` this means ``T`` divisible by 4.
    Our default horizon of 120 satisfies this.
    """

    def __init__(self, config: FlowConfigDP):
        super().__init__()
        self.config = config
        action_dim = config.action_dim
        down_dims = list(config.down_dims)
        all_dims = [action_dim] + down_dims
        start_dim = down_dims[0]

        # Diffusion-step encoder (sinusoidal -> MLP).
        dsed = config.diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + config.global_cond_dim

        # Down-blocks: each = (2x conditional resblock) + downsample.
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=config.kernel_size,
                    n_groups=config.n_groups,
                    cond_predict_scale=config.cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=config.kernel_size,
                    n_groups=config.n_groups,
                    cond_predict_scale=config.cond_predict_scale,
                ),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        # Mid-blocks.
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=config.kernel_size,
                n_groups=config.n_groups,
                cond_predict_scale=config.cond_predict_scale,
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=config.kernel_size,
                n_groups=config.n_groups,
                cond_predict_scale=config.cond_predict_scale,
            ),
        ])

        # Up-blocks: mirror of down with skip connections.
        # The up-path iterates over ``reversed(in_out[1:])`` which has
        # ``len(in_out) - 1`` entries.  ``is_last`` here matches DP's
        # convention from ``conditional_unet1d.py`` and is True only at
        # ``ind >= len(in_out) - 1`` -- which, since the loop has
        # ``len(in_out) - 1`` iterations (ind in [0, len-2]), is *never*
        # True.  This makes every up-level apply a real Upsample1d, so
        # the count of upsamples matches the count of downsamples in the
        # encoder (= len(in_out) - 1) and the U-Net preserves T exactly.
        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=config.kernel_size,
                    n_groups=config.n_groups,
                    cond_predict_scale=config.cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=config.kernel_size,
                    n_groups=config.n_groups,
                    cond_predict_scale=config.cond_predict_scale,
                ),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        # Final 1x1 conv back to action_dim.
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim,
                        kernel_size=config.kernel_size,
                        n_groups=config.n_groups),
            nn.Conv1d(start_dim, action_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,        # (B, T, action_dim)
        timestep: torch.Tensor,      # (B,)  integer
        global_cond: torch.Tensor,   # (B, n_obs_steps * obs_dim)
    ) -> torch.Tensor:
        # Broadcast timestep to batch if needed.
        if timestep.dim() == 0:
            timestep = timestep[None].expand(sample.shape[0])
        elif timestep.dim() == 1 and timestep.shape[0] == 1 and sample.shape[0] > 1:
            timestep = timestep.expand(sample.shape[0])
        timestep = timestep.to(sample.device)

        # Build conditioning vector: [step_emb || global_cond].
        step_emb = self.diffusion_step_encoder(timestep.float())  # (B, dsed)
        cond = torch.cat([step_emb, global_cond], dim=-1)         # (B, cond_dim)

        # Conv U-Net expects (B, C, T); our sample is (B, T, A).
        x = sample.movedim(-1, -2)                                # (B, A, T)

        # Encoder.
        skips: list[torch.Tensor] = []
        for resblock1, resblock2, downsample in self.down_modules:
            x = resblock1(x, cond)
            x = resblock2(x, cond)
            skips.append(x)
            x = downsample(x)

        # Middle.
        for mid_block in self.mid_modules:
            x = mid_block(x, cond)

        # Decoder.
        for resblock1, resblock2, upsample in self.up_modules:
            x = torch.cat([x, skips.pop()], dim=1)
            x = resblock1(x, cond)
            x = resblock2(x, cond)
            x = upsample(x)

        x = self.final_conv(x)                                    # (B, A, T)
        return x.movedim(-2, -1)                                  # (B, T, A)


# ---------------------------------------------------------------------------
# Loss, sampler, score
# ---------------------------------------------------------------------------


def dp_loss(
    model: ConditionalUnet1D,
    schedule: DDPMSchedule,
    x_data: torch.Tensor,        # (B, x_dim) flat, OR (B, T, A)
    global_cond: torch.Tensor,   # (B, n_obs_steps * obs_dim)
) -> torch.Tensor:
    """DDPM ε-prediction loss (Ho et al. 2020 Eq. 14).

    Per-sample:
        t  ~ Uniform{0, 1, ..., T-1}
        ε  ~ N(0, I)
        x_t = √(α̅_t) · x_0 + √(1 − α̅_t) · ε
        loss = || ε_θ(x_t, t, cond) − ε ||²

    Accepts ``x_data`` flat ``(B, T*A)`` *or* already-shaped ``(B, T, A)``
    to be compatible with the v3/v6 training scripts' batched flat input.
    """
    B = x_data.shape[0]
    config = model.config
    T = config.horizon
    A = config.action_dim
    if x_data.dim() == 2:
        x0 = x_data.view(B, T, A)
    else:
        x0 = x_data
    t = torch.randint(0, schedule.num_timesteps, (B,), device=x0.device)
    eps = torch.randn_like(x0)
    sqrt_a = schedule.sqrt_alphas_cumprod[t].view(B, 1, 1)
    sqrt_1ma = schedule.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)
    x_t = sqrt_a * x0 + sqrt_1ma * eps
    eps_pred = model(x_t, t, global_cond)
    return ((eps_pred - eps) ** 2).mean()


@torch.no_grad()
def dp_sample_ddim(
    model: ConditionalUnet1D,
    schedule: DDPMSchedule,
    global_cond: torch.Tensor,           # (B, n_obs_steps * obs_dim)
    n_inference_steps: int,
    eta: float = 0.0,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """DDIM sampler (Song et al. 2021) — stride-correct.

    The DDPM one-step Markov posterior ``p(x_{t-1} | x_t)`` is only
    valid for the **consecutive** transition ``t -> t-1``.  Subsampling
    the DDPM schedule and then applying that same formula
    (as the previous ``dp_sample`` implementation did) gives
    severely under-denoised samples — the documented bug that this
    function replaces.

    DDIM's update is stride-correct because it routes through the
    estimated ``x_0`` at each step and then re-noises to the next
    target timestep::

        eps         = eps_theta(x_t, t, cond)
        x_0_hat     = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        x_{t'}      = sqrt(alpha_bar_{t'}) * x_0_hat
                      + sqrt(1 - alpha_bar_{t'} - sigma^2) * eps
                      + sigma * z                     where z ~ N(0, I)
        sigma       = eta * sqrt((1 - alpha_bar_{t'}) / (1 - alpha_bar_t))
                            * sqrt(1 - alpha_bar_t / alpha_bar_{t'})

    With ``eta = 0`` (default, matching DP's published config) the
    update is deterministic, and the last step uses
    ``alpha_bar_{-1} := 1`` so the final iterate is ``x_0_hat``.
    ``eta = 1`` recovers DDPM ancestral.

    This matches HuggingFace ``diffusers.DDIMScheduler`` and Cheng
    Chi's reference DP implementation.

    Parameters
    ----------
    n_inference_steps
        Number of integer DDPM timesteps to keep, uniformly subsampled
        from ``[0, num_train_timesteps - 1]``.  DP's paper-default is
        100 for ``T = 100`` (no striding); we usually run 16-32 for
        speed.
    eta
        DDIM stochasticity coefficient.  Default 0 (deterministic).
    rng
        Optional ``torch.Generator`` for reproducible warm-starts.
    """
    config = model.config
    B = global_cond.shape[0]
    T = config.horizon
    A = config.action_dim
    device = global_cond.device

    K = int(n_inference_steps)
    if K <= 0:
        raise ValueError(f"n_inference_steps must be > 0, got {K}")

    # Uniformly spaced subset of [0, num_train_timesteps - 1] in descending
    # order, deduplicated.  Identical to the index list that the old
    # strided-DDPM path produced, so existing callers see the same
    # number of network evaluations.
    idx = torch.linspace(
        0, schedule.num_timesteps - 1, K,
    ).round().long().tolist()
    steps = list(reversed(sorted(set(idx))))            # [t_K-1, ..., t_0]

    if rng is not None:
        x = torch.randn(B, T, A, device=device, generator=rng)
    else:
        x = torch.randn(B, T, A, device=device)

    for i, t_int in enumerate(steps):
        t = torch.full((B,), t_int, dtype=torch.long, device=device)
        eps_pred = model(x, t, global_cond)             # (B, T, A)

        alpha_bar_t = schedule.alphas_cumprod[t_int]
        sqrt_ab_t = torch.sqrt(alpha_bar_t)
        sqrt_1mab_t = torch.sqrt(1.0 - alpha_bar_t)

        # x_0 estimate (Eq. 9, Song 2021).
        x0_hat = (x - sqrt_1mab_t * eps_pred) / sqrt_ab_t

        # alpha_bar_prev: at the next-in-stride timestep.  For the
        # final iteration there is no next step; set alpha_bar = 1 so
        # the output is x_0_hat exactly (clean signal, no residual
        # noise).
        if i == len(steps) - 1:
            alpha_bar_prev = torch.ones_like(alpha_bar_t)
        else:
            alpha_bar_prev = schedule.alphas_cumprod[steps[i + 1]]
        sqrt_ab_prev = torch.sqrt(alpha_bar_prev)

        if float(eta) > 0.0 and i != len(steps) - 1:
            # Stochastic DDIM step (eta in (0, 1]).
            sigma = float(eta) * torch.sqrt(
                (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            ) * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
            dir_xt = torch.sqrt(
                torch.clamp(1.0 - alpha_bar_prev - sigma ** 2, min=0.0)
            ) * eps_pred
            if rng is not None:
                noise = torch.randn(B, T, A, device=device, generator=rng)
            else:
                noise = torch.randn(B, T, A, device=device)
            x = sqrt_ab_prev * x0_hat + dir_xt + sigma * noise
        else:
            # Deterministic DDIM (eta = 0), or final step.
            sqrt_1mab_prev = torch.sqrt(1.0 - alpha_bar_prev)
            x = sqrt_ab_prev * x0_hat + sqrt_1mab_prev * eps_pred

    return x


@torch.no_grad()
def dp_sample(
    model: ConditionalUnet1D,
    schedule: DDPMSchedule,
    global_cond: torch.Tensor,           # (B, n_obs_steps * obs_dim)
    n_inference_steps: int | None = None,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one (B, T, A) action chunk conditioned on ``global_cond``.

    Dispatches to the **mathematically correct** sampler for the
    requested number of inference steps:

      * ``n_inference_steps is None`` or ``>= num_train_timesteps``:
        the standard **DDPM ancestral** chain over consecutive
        ``t -> t - 1`` transitions.  This is the one-step Markov
        posterior derivation from Ho et al. 2020 Eq. 11; it is only
        valid for *consecutive* timesteps, but with the full schedule
        every transition is consecutive, so it is correct here.

      * ``n_inference_steps < num_train_timesteps``: a **DDIM** sampler
        (``dp_sample_ddim``, ``eta = 0`` by default).  DDIM is
        stride-correct because it routes through the estimated
        ``x_0`` at each step and re-noises to the next target
        timestep, so any subset of the DDPM schedule works.

    The previous strided-DDPM-ancestral path was wrong: it applied the
    consecutive-step posterior formula across multi-step jumps and
    produced massively under-denoised samples (verified empirically:
    0/30 success on Can with ``n_inference_steps=18``).  Replacing
    that path with DDIM fixes the bug; the full-schedule branch is
    unchanged so existing ``--n-steps 100`` callers keep their
    behaviour.
    """
    if (
        n_inference_steps is None
        or int(n_inference_steps) >= schedule.num_timesteps
    ):
        # Full unstrided DDPM ancestral — the original correct path.
        config = model.config
        B = global_cond.shape[0]
        T = config.horizon
        A = config.action_dim
        device = global_cond.device

        steps = list(range(schedule.num_timesteps - 1, -1, -1))

        if rng is not None:
            x = torch.randn(B, T, A, device=device, generator=rng)
        else:
            x = torch.randn(B, T, A, device=device)

        for t_int in steps:
            t = torch.full((B,), t_int, dtype=torch.long, device=device)
            eps_pred = model(x, t, global_cond)
            # DDPM posterior mean.
            alpha_t = schedule.alphas[t_int]
            beta_t = schedule.betas[t_int]
            sqrt_1ma_t = schedule.sqrt_one_minus_alphas_cumprod[t_int]
            mu = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / sqrt_1ma_t) * eps_pred
            )
            if t_int > 0:
                var = schedule.posterior_variance[t_int]
                if rng is not None:
                    noise = torch.randn(B, T, A, device=device, generator=rng)
                else:
                    noise = torch.randn(B, T, A, device=device)
                x = mu + torch.sqrt(var) * noise
            else:
                x = mu
        return x

    # Strided -> DDIM.
    return dp_sample_ddim(
        model, schedule, global_cond,
        n_inference_steps=int(n_inference_steps),
        eta=0.0,
        rng=rng,
    )


@torch.no_grad()
def dp_score(
    model: ConditionalUnet1D,
    schedule: DDPMSchedule,
    x: torch.Tensor,             # (B, T, A) or (B, T*A)
    sigma: float | torch.Tensor, # scalar or (B,) -- Karras-EDM σ
    global_cond: torch.Tensor,   # (B, n_obs_steps * obs_dim)
) -> torch.Tensor:
    """Score ``∇log p_t(x)`` derived from the ε-prediction.

    Conversion (Karras 2022 § 2, VP-SDE row of the unification
    table):

        ε_θ(x, t) ≈ (x - √(α̅_t) · D_t(x)) / σ_t      [training target]
        ∇log p_t(x) = −ε_θ(x, t) / σ_t

    So we map the runtime σ (continuous) to the nearest DDPM integer
    timestep, query the network there, and return ``-ε_pred / σ``.
    Output is reshaped to match ``x``'s input shape.
    """
    config = model.config
    B = x.shape[0]
    T = config.horizon
    A = config.action_dim
    was_flat = (x.dim() == 2)
    x_3d = x.view(B, T, A) if was_flat else x

    sigma_scalar = (
        float(sigma) if not torch.is_tensor(sigma) else float(sigma.mean())
    )
    t_int = schedule.nearest_t_for_sigma(sigma_scalar)
    t = torch.full((B,), t_int, dtype=torch.long, device=x.device)

    eps_pred = model(x_3d, t, global_cond)        # (B, T, A)
    score = -eps_pred / max(sigma_scalar, 1e-6)

    return score.reshape(B, T * A) if was_flat else score
