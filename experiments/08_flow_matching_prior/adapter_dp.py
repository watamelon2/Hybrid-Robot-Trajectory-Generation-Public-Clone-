"""Adapter for v_theta-dp (DP UNet1D + DDPM ε-prediction).

Public interface mirrors ``adapter_v3.FlowPriorV3`` /
``adapter_v6.FlowPriorV6`` exactly, plus two **optional** extensions
for ``T_obs > 1`` support:

  - ``FlowPriorDP.from_checkpoint(path, device)``
  - ``.sample_initial_mean(s0, n_steps, seed) -> (H, A) float32``
  - ``.make_prior_score_fn(s0, K, sigma_schedule) -> (mu, k) -> (H, A) score``
  - ``.reset_obs_buffer()``                  -- NEW (T_obs > 1 only)
  - ``.observe(s)``                           -- NEW (T_obs > 1 only)

What's different from v3 / v6
------------------------------

1. **Score formula**: ``∇log p_t(x) = − ε_θ(x, t) / σ_t``  (Karras 2022
   unification, VP-SDE row).  ``σ_t`` is the VP noise level
   ``√(1 − α̅_t)`` computed from the DDPM β-schedule.  We map MBD's
   continuous σ_k to the *nearest DDPM integer timestep* and query
   the network there.

2. **Sampler**: DDPM ancestral sampling over the full T=100 timesteps
   (or a subsampled schedule).  Slower than v3/v6's 32-step Heun
   ODE but stays inside the training distribution of timesteps.

3. **Conditioning is integer t, not σ**.  The translation (σ -> t)
   happens inside the adapter so the runtime caller (MBD's
   ``denoise_loop``) doesn't have to know.

4. **T_obs > 1 obs buffer**.  When the loaded checkpoint declares
   ``n_obs_steps > 1``, the adapter maintains an internal deque of
   the last ``n_obs_steps`` observations.  Two API options for the
   runtime caller:

     (a) **Stateless mode (default, backward-compatible)**.  Each call
         to ``sample_initial_mean(s0)`` or ``make_prior_score_fn(s0)``
         resets the buffer to ``[s0, s0, ..., s0]`` (replicated
         ``T_obs`` times).  This matches the boundary-replication
         convention of training-time ``stack_obs_history`` -- the
         model sees [s0, s0] at the first chunk of a rollout, which
         is exactly the same conditioning it saw on demo's first
         chunk.  After that, you lose obs *history* but the
         architecture's other gains (FiLM, deeper net) still apply.

     (b) **Stateful mode (closed-loop callers)**.  The runtime calls
         ``prior.reset_obs_buffer()`` at the start of each rollout
         and ``prior.observe(s_t)`` after each env observation.  Then
         ``sample_initial_mean`` / ``make_prior_score_fn`` use the
         buffered history.  This recovers DP's true ``T_obs=2``
         behaviour and is required for the closed-loop-short-horizon
         (Phase C) experiments.

   Stateless mode is detected automatically: if you never call
   ``observe``, the adapter falls back to "replicate current obs"
   on every call.  Stateful mode kicks in as soon as you call
   ``observe`` at least once.
"""

from __future__ import annotations

import sys
from collections import deque
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

from model_dp import (  # noqa: E402
    ConditionalUnet1D,
    DDPMSchedule,
    FlowConfigDP,
    dp_sample,
    dp_score,
)


def _cosine_sigma_schedule(
    K: int, sigma_max: float, sigma_min: float,
) -> np.ndarray:
    """Cosine descent ``sigma_max -> sigma_min`` over ``K`` steps.

    Same helper used by adapter_v3 / adapter_v6; duplicated here so the
    adapter has no v3/v6 dependency at import time.
    """
    ts = np.linspace(0.0, np.pi, int(K), dtype=np.float32)
    cos = (1.0 + np.cos(ts)) * 0.5
    return (sigma_min + (sigma_max - sigma_min) * cos).astype(np.float32)


class FlowPriorDP:
    """Loaded DP-UNet1D + DDPM wrapper with obs normalisation."""

    DEFAULT_SIGMA_MIN = 0.05
    DEFAULT_SIGMA_MAX = 0.25

    def __init__(
        self,
        net: ConditionalUnet1D,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        device: torch.device | str = "cpu",
        action_min: np.ndarray | None = None,
        action_max: np.ndarray | None = None,
        normalize_actions: bool = False,
        obs_min: np.ndarray | None = None,
        obs_max: np.ndarray | None = None,
        normalize_obs: bool = False,
        task: str = "can",
    ):
        self.net = net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.config: FlowConfigDP = net.config
        self.obs_mean = obs_mean.astype(np.float32)
        self.obs_std = np.maximum(obs_std.astype(np.float32), 1e-6)
        self.device = torch.device(device)

        # Task name recorded in the ckpt blob by ``train_dp.py``.  The
        # unified MBD pipeline reads this to select the right obs
        # schema (``can`` -> 46-D legacy, ``can_dp`` -> 23-D DP paper
        # with synthetic Can_to_eef_pos/quat) before the first rollout.
        # Backward-compat: legacy blobs without ``task`` default to
        # ``"can"`` (every pre-Phase-B ckpt was trained on the 46-D
        # schema).  Direct callers of FlowPriorDP that don't care
        # about the runtime obs builder (e.g. the standalone driver
        # which validates obs_dim separately) can ignore this field.
        self.task: str = str(task)

        # Observation normalisation mode (DP paper recipe, Finding C;
        # added 2026-05-28).  Two paths:
        #   - z-score (legacy): use ``obs_mean`` / ``obs_std``.
        #     What every checkpoint before 2026-05-28 used.
        #   - min-max [-1, 1]: use ``obs_min`` / ``obs_max``.
        #     Mirrors DP's ``LinearNormalizer.fit(mode='limits')``.
        # The choice is recorded in the ckpt blob and replayed at
        # inference so the model sees the same distribution it
        # trained on.  Backward-compat: legacy blobs without
        # ``obs_min`` / ``obs_max`` fall through to the z-score
        # branch (``normalize_obs=False``).
        D = int(self.config.obs_dim)
        if obs_min is None:
            obs_min = -np.ones(D, dtype=np.float32)
        if obs_max is None:
            obs_max = +np.ones(D, dtype=np.float32)
        self.obs_min = np.asarray(obs_min, dtype=np.float32).reshape(D)
        self.obs_max = np.asarray(obs_max, dtype=np.float32).reshape(D)
        self.normalize_obs = bool(normalize_obs)
        # Pre-compute the obs-norm affine constants so ``_normalise_s_np``
        # is a single broadcast multiply-add at call time.
        obs_rng = np.maximum(self.obs_max - self.obs_min, 1e-4).astype(np.float32)
        self._obs_norm_scale = (2.0 / obs_rng).astype(np.float32)            # (D,)
        self._obs_norm_offset = (-1.0 - self.obs_min * self._obs_norm_scale).astype(np.float32)

        # Action normalisation (DP paper recipe, added 2026-05-27).
        # When ``normalize_actions=True``, the model was trained with
        # actions rescaled per-dim to ``[-1, 1]`` via min-max, and the
        # adapter must (a) denormalise sample outputs back to env-space
        # before returning, (b) scale the score by the same Jacobian so
        # MBD callers see a score for the env-space action variable.
        #
        # Backward-compat: legacy checkpoints (Phase-1/2/2c, raw-action
        # training) pass ``normalize_actions=False`` and
        # ``action_min=-1, action_max=+1``, making the affine map the
        # identity.
        A = int(self.config.action_dim)
        if action_min is None:
            action_min = -np.ones(A, dtype=np.float32)
        if action_max is None:
            action_max = +np.ones(A, dtype=np.float32)
        self.action_min = np.asarray(action_min, dtype=np.float32).reshape(A)
        self.action_max = np.asarray(action_max, dtype=np.float32).reshape(A)
        self.normalize_actions = bool(normalize_actions)
        # Pre-compute the inverse-affine constants once.  Cached as
        # (A,) numpy arrays so ``sample_initial_mean`` can apply them
        # via broadcasting without recomputing the schedule.
        rng = np.maximum(self.action_max - self.action_min, 1e-4).astype(np.float32)
        self._denorm_scale = (rng / 2.0).astype(np.float32)          # (A,)
        self._denorm_offset = ((self.action_min + self.action_max) / 2.0).astype(np.float32)
        # Jacobian from normalised-space to env-space:
        #   x_env = scale * x_norm + offset
        #   d x_norm / d x_env = 1 / scale  (per-dim)
        # Score in env-space = score_in_norm_space * (1 / scale).
        self._jac_inv = (1.0 / self._denorm_scale).astype(np.float32)  # (A,)
        self.schedule = DDPMSchedule(
            self.config.num_train_timesteps,
            schedule=self.config.beta_schedule,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end,
            device=self.device,
        )
        # σ-range default to v3/v6 runtime defaults so MBD's cosine
        # schedule plugs in without changes.  Both bounds are clamped
        # at runtime to the actual σ_t range of the DDPM schedule.
        sig_t = self.schedule.sqrt_one_minus_alphas_cumprod
        self.sigma_train_min = float(sig_t.min().item())
        self.sigma_train_max = float(sig_t.max().item())
        self.sigma_min = float(
            sigma_min if sigma_min is not None else self.DEFAULT_SIGMA_MIN
        )
        self.sigma_max = float(
            sigma_max if sigma_max is not None else self.DEFAULT_SIGMA_MAX
        )

        # Obs-history buffer for T_obs > 1.  Empty until the caller
        # either calls ``observe(s)`` or implicitly triggers a reset
        # by calling ``sample_initial_mean`` / ``make_prior_score_fn``
        # with no prior ``observe`` (stateless mode).
        #
        # We use ``maxlen=n_obs_steps`` so old frames fall off the
        # left as new ones are pushed on the right (FIFO, the right-
        # most entry is the *most recent* obs, matching the training
        # convention from ``dataset_v3.stack_obs_history``).
        self._obs_buf: deque[np.ndarray] = deque(
            maxlen=int(self.config.n_obs_steps)
        )
        self._observe_called = False

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        device: torch.device | str = "cpu",
        sigma_min: float | None = None,
        sigma_max: float | None = None,
    ) -> "FlowPriorDP":
        blob = torch.load(path, map_location=device, weights_only=False)
        arch = blob.get("arch", "")
        if arch != "dp_unet1d":
            raise ValueError(
                f"checkpoint at {path} has arch={arch!r}, expected "
                "'dp_unet1d'.  Use adapter.FlowPrior.from_checkpoint for "
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
    ) -> "FlowPriorDP":
        config = FlowConfigDP.from_dict(blob["config"])
        net = ConditionalUnet1D(config)
        net.load_state_dict(blob["state_dict"])
        # Action norm fields (added 2026-05-27).  Older blobs that
        # predate the audit don't carry them; fall back to the
        # identity affine map so legacy ckpts evaluate unchanged.
        a_min = blob.get("action_min")
        a_max = blob.get("action_max")
        normalize_actions = bool(blob.get("normalize_actions", False))
        # Obs norm fields (added 2026-05-28, Finding C / Phase-3.5).
        # Same backward-compat shape as action norm: legacy blobs
        # without these keys default to z-score (``normalize_obs=False``)
        # and the constructor builds an identity-affine map.
        o_min = blob.get("obs_min")
        o_max = blob.get("obs_max")
        normalize_obs = bool(blob.get("normalize_obs", False))
        # Task field (added 2026-05-29, Phase-B / Phase-C wiring).
        # The ckpt blob records the obs schema name (``can`` / ``can_dp``
        # / ``square`` / ``lift``).  Unified-pipeline callers read this
        # via ``prior.task`` to pick the right runtime obs builder.
        # Legacy blobs without it default to ``"can"``.
        task = str(blob.get("task", "can"))
        return cls(
            net,
            np.asarray(blob["obs_mean"], dtype=np.float32),
            np.asarray(blob["obs_std"], dtype=np.float32),
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            device=device,
            action_min=(np.asarray(a_min, dtype=np.float32)
                        if a_min is not None else None),
            action_max=(np.asarray(a_max, dtype=np.float32)
                        if a_max is not None else None),
            normalize_actions=normalize_actions,
            obs_min=(np.asarray(o_min, dtype=np.float32)
                     if o_min is not None else None),
            obs_max=(np.asarray(o_max, dtype=np.float32)
                     if o_max is not None else None),
            normalize_obs=normalize_obs,
            task=task,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalise_s_np(self, s: np.ndarray) -> np.ndarray:
        """Return the normalised float32 vector for ``s``.

        Branches on ``self.normalize_obs``:
          - True: min-max [-1, 1] (DP paper recipe, Finding C; Phase-3.5+).
            ``y = s * scale + offset`` where ``scale = 2 / (b - a)`` and
            ``offset = -1 - a * scale`` (precomputed in ``__init__``).
          - False: z-score (legacy; every ckpt before 2026-05-28).
            ``y = (s - obs_mean) / obs_std``.

        Either way the trailing dim must equal ``config.obs_dim``;
        leading dims broadcast freely (used both for single obs and
        for the stacked-history buffer concatenation).
        """
        s_arr = np.asarray(s, dtype=np.float32)
        if self.normalize_obs:
            return (
                s_arr * self._obs_norm_scale + self._obs_norm_offset
            ).astype(np.float32)
        return ((s_arr - self.obs_mean) / self.obs_std).astype(np.float32)

    def reset_obs_buffer(self) -> None:
        """Clear the internal obs history.

        Call at the start of each rollout when running in stateful
        (closed-loop) mode.  After this, the next ``observe(s)``
        starts a fresh history.
        """
        self._obs_buf.clear()
        self._observe_called = False

    def observe(self, s: np.ndarray) -> None:
        """Push a new env observation onto the obs-history buffer.

        Stores the *normalised* (z-score) form so later concatenation
        can be done with one ``np.concatenate``.  Switches the adapter
        into stateful mode -- subsequent ``sample_initial_mean`` /
        ``make_prior_score_fn`` calls will use the buffer instead of
        replicating their ``s0`` argument.
        """
        z = self._normalise_s_np(s)
        self._obs_buf.append(z)
        self._observe_called = True

    def _global_cond_from_s0(self, s0: np.ndarray) -> torch.Tensor:
        """Build the ``(1, T_obs * obs_dim)`` conditioning tensor.

        Behaviour:

          - ``T_obs == 1``: just return the normalised ``s0`` as a
            ``(1, obs_dim)`` tensor.  Unchanged from the original
            stateless adapter.

          - ``T_obs > 1`` and the caller has *never* called
            ``observe``: replicate ``s0`` ``T_obs`` times.  This
            matches the training-time boundary-replication convention
            (``dataset_v3.stack_obs_history`` does the same for the
            first ``T_obs - 1`` chunks of every demo).

          - ``T_obs > 1`` and the buffer has been populated via
            ``observe``: pad the buffer with its left-most frame
            (oldest obs) if it is shorter than ``T_obs``, then
            concatenate in [oldest, ..., newest] order.  ``s0`` is
            *ignored* in this path -- the caller is expected to have
            already pushed the same value via ``observe(s0)``.
        """
        T_obs = int(self.config.n_obs_steps)
        z = self._normalise_s_np(s0)                          # (obs_dim,)

        if T_obs == 1:
            cond_np = z[None, :]                              # (1, obs_dim)
        elif not self._observe_called:
            # Stateless mode: replicate s0 T_obs times.
            cond_np = np.tile(z, (1, T_obs))                  # (1, T_obs * obs_dim)
        else:
            # Stateful mode: use buffer, left-pad with oldest frame.
            frames = list(self._obs_buf)                      # newest at right end
            if len(frames) == 0:
                # observe() was called, then reset_obs_buffer().  Treat as
                # stateless for safety.
                cond_np = np.tile(z, (1, T_obs))
            else:
                while len(frames) < T_obs:
                    frames.insert(0, frames[0])               # replicate left
                cond_np = np.concatenate(frames, axis=0)[None, :]  # (1, T_obs * obs_dim)

        return torch.from_numpy(cond_np.astype(np.float32)).to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_initial_mean(
        self,
        s0: np.ndarray,
        n_steps: int = 32,
        seed: int | None = None,
    ) -> np.ndarray:
        """Sample one warm-start trajectory ``(H, A)`` from p(. | s0).

        Uses DDPM ancestral sampling.  ``n_steps`` is the number of
        denoising steps (DDIM-style uniform stride if fewer than
        num_train_timesteps); default 32 to match v3/v6's warm-start
        budget.  The DP paper uses the full 100 at inference; ours
        cuts that to 32 because each step is a UNet forward pass.
        """
        cond = self._global_cond_from_s0(s0)
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
        traj = dp_sample(
            self.net, self.schedule, cond,
            n_inference_steps=int(n_steps),
            rng=gen,
        )                                                  # (1, H, A)
        out = traj.squeeze(0).cpu().numpy().astype(np.float32)   # (H, A)
        # Denormalise from [-1, 1] -> env-space if the ckpt was
        # trained with action min-max norm.  For legacy (raw-action)
        # ckpts, ``_denorm_scale = 1, _denorm_offset = 0`` so this is
        # a no-op (identity).
        if self.normalize_actions:
            out = (
                out * self._denorm_scale[None, :]
                + self._denorm_offset[None, :]
            ).astype(np.float32)
        return out

    def make_prior_score_fn(
        self,
        s0: np.ndarray,
        K: int,
        sigma_schedule: np.ndarray | None = None,
    ) -> Callable[[np.ndarray, int], np.ndarray]:
        """Return a ``(mu, k) -> score`` closure for MBD's denoise_loop.

        At each call:
          1. Clamp σ_k to the DDPM training σ range (otherwise the
             nearest-t lookup degenerates to the endpoint).
          2. Look up the nearest integer DDPM timestep t = argmin_t |σ_t − σ_k|.
          3. Query ε_θ(mu, t, cond) and return ``-ε / σ_k``.
        """
        cond = self._global_cond_from_s0(s0)
        H = self.config.horizon
        A = self.config.action_dim

        if sigma_schedule is not None:
            schedule_np = np.asarray(sigma_schedule, dtype=np.float32)
            if schedule_np.shape != (int(K),):
                raise ValueError(
                    f"sigma_schedule shape {schedule_np.shape}, "
                    f"expected ({K},)"
                )
        else:
            schedule_np = _cosine_sigma_schedule(
                int(K), self.sigma_max, self.sigma_min,
            )

        sig_lo = max(self.sigma_train_min, 1e-4)
        sig_hi = self.sigma_train_max

        # Cache action-norm constants on the closure so each score
        # call sees a consistent affine map even if the adapter is
        # later mutated (which we don't do, but the cache makes the
        # contract explicit).
        normalize_actions = bool(self.normalize_actions)
        denorm_scale = self._denorm_scale.reshape(1, 1, A).astype(np.float32)
        denorm_offset = self._denorm_offset.reshape(1, 1, A).astype(np.float32)
        jac_inv = self._jac_inv.reshape(1, 1, A).astype(np.float32)

        @torch.no_grad()
        def score_fn(mu: np.ndarray, k: int) -> np.ndarray:
            sigma_k = float(np.clip(schedule_np[int(k)], sig_lo, sig_hi))
            mu_arr = np.asarray(mu, dtype=np.float32).reshape(1, H, A)
            # If the model was trained on min-max-normalised actions,
            # the runtime ``mu`` (env-space) must be projected into
            # normalised space before querying the network.  We then
            # un-do the Jacobian on the returned score so the caller
            # sees ``∇_{x_env} log p`` instead of ``∇_{x_norm} log p``.
            # NOTE: the σ scale also differs by the per-dim affine
            # Jacobian; we approximate with the scalar mean(scale) so
            # MBD's σ schedule still indexes a sensible DDPM
            # timestep.  TODO when we wire DP-with-action-norm into
            # MBD (Phase-3): convert ``sigma_k`` to normalised σ via
            # ``sigma_k / scale.mean()`` before the nearest-t lookup.
            if normalize_actions:
                mu_norm = (mu_arr - denorm_offset) / denorm_scale
                x = torch.from_numpy(mu_norm).to(self.device)
            else:
                x = torch.from_numpy(mu_arr).to(self.device)
            score = dp_score(self.net, self.schedule, x, sigma_k, cond)
            # ``dp_score`` returns (1, H, A) when input is 3D.
            score_np = (
                score.cpu()
                .numpy()
                .reshape(1, H, A)
                .astype(np.float32)
            )
            if normalize_actions:
                # Convert score from normalised-x to env-x via Jacobian:
                #   ∇_{x_env} log p = (∇_{x_norm} log p) · (∂x_norm/∂x_env)
                #                   = score_norm · (1 / denorm_scale)
                score_np = score_np * jac_inv
            return score_np.reshape(H, A).astype(np.float32)

        return score_fn
