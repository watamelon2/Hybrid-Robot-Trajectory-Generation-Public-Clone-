"""Per-chunk trajectory dataset for the v3 prior.

Differs from ``dataset.py`` (v1 / v2) in **one** way that matters:
v1 / v2 emit one example per demo, with ``s_0`` from t=0 and the
full demo as the target trajectory.  v3 emits one example per
*chunk*: every starting timestep ``t`` produces an
``(s_t, a_{t:t+H})`` pair, with edge-padding when the demo runs
out before ``t+H``.

This mirrors the BC dataset's per-step sampling pattern (each
demo of length L_i produces L_i examples), and is the reason the
v3 prior can survive MPC replanning while v2 collapsed (see
``results/sweep_v2_2026_05_16/MORNING_BRIEF.md`` Phase 8/9 update).

Output format
-------------

``load_chunks(path, horizon, ...)`` returns

  X:       (N_chunks, H, action_dim)   float32, target action chunks
  S:       (N_chunks, D_obs)            float32, conditioning obs at chunk start
  meta:    list[dict]                    per-chunk provenance
  demo_ids: list[str]                    HDF5 group names consumed

where ``N_chunks = sum_i L_i`` (one chunk per timestep in every
demo, including tail chunks that get edge-padded).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Reuse exp 03's task-aware obs schema + ObsStats so v3's obs
# vector layout is bit-for-bit compatible with the BC pipeline
# and the zero-train retrieval bank.  This is essential: the live
# env at planning time produces the same OBS_KEYS for the matching
# ``--task``, so v3's conditioning lives in the same feature space
# as the bank.
_THIS_DIR = Path(__file__).resolve().parent
_EXP_03 = _THIS_DIR.parent / "03_bc_baseline"
for _p in (_EXP_03,):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import importlib.util as _iu
_ds_spec = _iu.spec_from_file_location("exp03_dataset", _EXP_03 / "dataset.py")
_ds = _iu.module_from_spec(_ds_spec)
_ds_spec.loader.exec_module(_ds)
ObsStats = _ds.ObsStats
_concat_obs_dict = _ds._concat_obs_dict
get_obs_keys = _ds.get_obs_keys
get_object_slices = _ds.get_object_slices
get_synthetic_keys = _ds.get_synthetic_keys
compute_synthetic_obs_keys = _ds.compute_synthetic_obs_keys
compute_obs_stats = _ds.compute_obs_stats

import h5py  # noqa: E402


@dataclass(frozen=True)
class ChunkBundle:
    """Per-chunk arrays packaged for the trainer.

    Use ``.to_tensors()`` to materialise torch tensors lazily.

    Action statistics
    -----------------

    ``action_min`` / ``action_max`` (added 2026-05-27) are the per-
    dimension min / max of the training actions, used for the DP-
    paper ``LinearNormalizer`` recipe: each action dim is rescaled
    to ``[-1, 1]`` via min-max normalisation before being fed to the
    DDPM loss.  v3/v6 trainers ignore these fields entirely; only
    the DP trainer + adapter consume them.

    Helpers ``normalize_actions`` / ``denormalize_actions`` apply
    the affine map ``x in [a, b] <-> y in [-1, 1]``.

    Observation statistics
    ----------------------

    Two parallel normalisations are pre-computed and stored:

      - z-score: ``obs_mean`` / ``obs_std`` / ``S_norm`` (legacy
        path; what every trainer before 2026-05-28 used).
      - min-max [-1, 1]: ``obs_min`` / ``obs_max`` /
        ``S_norm_minmax`` (added 2026-05-28; mirrors DP paper's
        ``LinearNormalizer.fit(mode='limits')``).

    The DP trainer picks one path based on its ``--normalize-obs``
    flag and saves the chosen mode + the corresponding stats in
    the checkpoint blob.  v3/v6 trainers always use the z-score
    path (``S_norm``) and ignore the min-max fields.

    Helpers ``normalize_obs`` / ``denormalize_obs`` apply the min-
    max affine map ``s in [a, b] <-> y in [-1, 1]`` (parallel to
    the action helpers).
    """

    X: np.ndarray              # (N_chunks, H, A) float32 -- action targets (RAW, env-space)
    S: np.ndarray              # (N_chunks, D)    float32 -- conditioning obs (un-normalised)
    S_norm: np.ndarray         # (N_chunks, D)    float32 -- z-score normalised
    S_norm_minmax: np.ndarray  # (N_chunks, D)    float32 -- min-max [-1, 1] normalised
    obs_mean: np.ndarray
    obs_std: np.ndarray
    obs_min: np.ndarray        # (D,) float32 -- per-dim min over training obs
    obs_max: np.ndarray        # (D,) float32 -- per-dim max over training obs
    action_min: np.ndarray     # (A,) float32 -- per-dim min over training actions
    action_max: np.ndarray     # (A,) float32 -- per-dim max over training actions
    demo_lens: np.ndarray      # (M,) int64, original demo lengths
    demo_ids: list[str]
    horizon: int
    action_dim: int
    obs_dim: int

    @property
    def n_chunks(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_demos(self) -> int:
        return len(self.demo_ids)

    def normalize_actions(self, x: np.ndarray) -> np.ndarray:
        """Map env-space actions to ``[-1, 1]`` per-dim min-max.

        Affine map ``y = x * (2 / (b - a)) + (-1 - a * (2 / (b - a)))``
        with ``a = action_min``, ``b = action_max``.  Broadcasts over
        any leading dims of ``x`` as long as the trailing dim equals
        ``action_dim``.
        """
        a = self.action_min.astype(np.float32)
        b = self.action_max.astype(np.float32)
        rng = (b - a).astype(np.float32)
        scale = 2.0 / np.maximum(rng, 1e-4)
        offset = -1.0 - a * scale
        return (np.asarray(x, dtype=np.float32) * scale + offset).astype(np.float32)

    def denormalize_actions(self, y: np.ndarray) -> np.ndarray:
        """Inverse of ``normalize_actions``: ``[-1, 1]`` -> env-space.

        ``x = y * ((b - a) / 2) + ((a + b) / 2)`` per-dim.  Used by
        the DP adapter at sample time to convert the model's
        normalised-space output back to env-space actions.
        """
        a = self.action_min.astype(np.float32)
        b = self.action_max.astype(np.float32)
        denorm_scale = ((b - a) / 2.0).astype(np.float32)
        denorm_offset = ((a + b) / 2.0).astype(np.float32)
        return (
            np.asarray(y, dtype=np.float32) * denorm_scale + denorm_offset
        ).astype(np.float32)

    def normalize_obs(self, s: np.ndarray) -> np.ndarray:
        """Map env-space obs to ``[-1, 1]`` per-dim min-max.

        Affine map ``y = s * (2 / (b - a)) + (-1 - a * (2 / (b - a)))``
        with ``a = obs_min``, ``b = obs_max``.  Broadcasts over any
        leading dims of ``s`` as long as the trailing dim equals
        ``obs_dim``.  Parallels ``normalize_actions`` but for the
        observation conditioning vector.

        Used by ``train_dp.py`` when ``--normalize-obs`` is set, and
        by ``adapter_dp.FlowPriorDP._normalise_s_np`` at inference
        when the checkpoint was trained in min-max mode.
        """
        a = self.obs_min.astype(np.float32)
        b = self.obs_max.astype(np.float32)
        rng = (b - a).astype(np.float32)
        scale = 2.0 / np.maximum(rng, 1e-4)
        offset = -1.0 - a * scale
        return (np.asarray(s, dtype=np.float32) * scale + offset).astype(np.float32)

    def denormalize_obs(self, y: np.ndarray) -> np.ndarray:
        """Inverse of ``normalize_obs``: ``[-1, 1]`` -> env-space.

        ``s = y * ((b - a) / 2) + ((a + b) / 2)`` per-dim.  Provided
        for symmetry with the action helpers; the DP pipeline only
        consumes the forward direction (env obs are normalised
        before being fed to the model, never denormalised back).
        """
        a = self.obs_min.astype(np.float32)
        b = self.obs_max.astype(np.float32)
        denorm_scale = ((b - a) / 2.0).astype(np.float32)
        denorm_offset = ((a + b) / 2.0).astype(np.float32)
        return (
            np.asarray(y, dtype=np.float32) * denorm_scale + denorm_offset
        ).astype(np.float32)


def _edge_pad_actions(actions: np.ndarray, horizon: int) -> np.ndarray:
    """Pad a (T, A) action trajectory to (T, horizon, A) per-chunk windows.

    For each start index ``t`` in ``[0, T)``, return ``actions[t:t+H]``
    edge-padded with the last action when ``t + H > T``.  Output has
    shape ``(T, H, A)``: one full-horizon chunk starting at every
    timestep, mirroring BC's per-step training distribution.

    Why edge-padding (rather than zero-padding):
      End-of-demo behaviour in robomimic ``ph`` data is typically a
      held grasp / hover.  Replicating the final action keeps the
      tail in-distribution with what the env actually executes after
      a demo's final step; zero-padding would inject ``release
      gripper, neutral motion`` everywhere and bias the model.
    """
    T, A = actions.shape
    H = int(horizon)
    if H <= 0:
        raise ValueError(f"horizon must be > 0, got {H}")
    # Build a padded source with H-1 trailing copies of the last action.
    if T == 0:
        raise ValueError("empty action trajectory")
    pad = np.broadcast_to(actions[-1:], (H - 1, A)).astype(np.float32, copy=True)
    padded = np.concatenate([actions, pad], axis=0)  # (T + H - 1, A)
    out = np.empty((T, H, A), dtype=np.float32)
    for t in range(T):
        out[t] = padded[t : t + H]
    return out


def load_chunks(
    dataset_path: Path,
    horizon: int,
    task: str = "can",
    n_demos: int | None = None,
    obs_keys: tuple[str, ...] | None = None,
) -> ChunkBundle:
    """Read demos and emit per-timestep chunks.

    Parameters
    ----------
    dataset_path
        Robomimic-style HDF5 (``low_dim_v141.hdf5``).
    horizon
        ``H`` -- length of each action chunk.  Must match the runtime
        planner horizon.
    task
        Task name in ``("can", "square", "lift", "tool_hang", "transport")``.
        Picks the right ``OBS_KEYS`` tuple.
    n_demos
        Optional cap on the number of demos consumed.
    obs_keys
        Override ``get_obs_keys(task)`` if you need a custom key list.

    Returns
    -------
    ChunkBundle with N_chunks = sum_i L_i (one example per demo
    timestep, edge-padded at the tail).
    """
    dataset_path = Path(dataset_path)
    if obs_keys is None:
        obs_keys = get_obs_keys(task)
    object_slices = get_object_slices(task)
    synth_keys = get_synthetic_keys(task)

    X_chunks: list[np.ndarray] = []
    S_chunks: list[np.ndarray] = []
    lens: list[int] = []
    used: list[str] = []

    with h5py.File(str(dataset_path), "r") as h5:
        all_demos = sorted(
            h5["data"].keys(), key=lambda k: int(k.split("_")[1])
        )
        if n_demos is not None and n_demos > 0:
            all_demos = all_demos[:n_demos]
        for demo_id in all_demos:
            demo = h5["data"][demo_id]
            obs_dict: dict[str, np.ndarray] = {}
            for k in obs_keys:
                if k in object_slices:
                    src, sl = object_slices[k]
                    obs_dict[k] = np.asarray(demo["obs"][src][:, sl])
                elif k in synth_keys:
                    # Synthetic key: deps must already be in obs_dict
                    # (OBS_KEYS_<TASK> orders deps before synthetics).
                    # Compute and insert in-place.
                    compute_synthetic_obs_keys(
                        obs_dict, {k: synth_keys[k]},
                    )
                else:
                    obs_dict[k] = np.asarray(demo["obs"][k][()])
            obs_flat = _concat_obs_dict(obs_dict, obs_keys).astype(np.float32)
            actions = np.asarray(demo["actions"][()], dtype=np.float32)
            T = int(actions.shape[0])
            if obs_flat.shape[0] != T:
                raise ValueError(
                    f"obs / action length mismatch in {demo_id}: "
                    f"{obs_flat.shape[0]} vs {T}"
                )
            # Per-timestep windows.
            chunks = _edge_pad_actions(actions, horizon)        # (T, H, A)
            X_chunks.append(chunks)
            S_chunks.append(obs_flat)                            # (T, D)
            lens.append(T)
            used.append(demo_id)

    X = np.concatenate(X_chunks, axis=0)
    S = np.concatenate(S_chunks, axis=0)
    stats = compute_obs_stats(S)
    S_norm = stats.normalize(S).astype(np.float32)

    # Per-dim action min/max over all training timesteps (NOT just chunk
    # starts), matching DP's ``LinearNormalizer.fit(mode='limits')`` which
    # reduces over the leading batch axis.  Any dim with collapsed range
    # (< 1e-4) gets bumped to a symmetric ±1 fallback so the eventual
    # ``2 / (b - a)`` scale stays finite.
    flat_actions = X.reshape(-1, X.shape[-1]).astype(np.float32)   # (N*H, A)
    action_min = flat_actions.min(axis=0).astype(np.float32)
    action_max = flat_actions.max(axis=0).astype(np.float32)
    collapsed = (action_max - action_min) < 1e-4
    action_min = np.where(collapsed, -1.0, action_min).astype(np.float32)
    action_max = np.where(collapsed, +1.0, action_max).astype(np.float32)

    # Per-dim observation min/max over the chunk-start S (these are the
    # actual values the model conditions on; chunk-internal S timesteps
    # never get fed to the network because conditioning is sampled at the
    # chunk start only).  Same collapse-handling pattern as actions.
    # Added 2026-05-28 to implement DP paper's ``LinearNormalizer`` for
    # obs (Phase-3.5 / Finding C).  Cost: one O(N * D) pass at load
    # time, ~0.05 s for the typical 23 k chunks × 46 dims.
    obs_min = S.min(axis=0).astype(np.float32)
    obs_max = S.max(axis=0).astype(np.float32)
    obs_collapsed = (obs_max - obs_min) < 1e-4
    obs_min = np.where(obs_collapsed, -1.0, obs_min).astype(np.float32)
    obs_max = np.where(obs_collapsed, +1.0, obs_max).astype(np.float32)
    # Pre-compute the min-max normalised version of S so the trainer
    # can pick z-score vs min-max without a second pass.
    _obs_rng = np.maximum(obs_max - obs_min, 1e-4).astype(np.float32)
    _obs_scale = (2.0 / _obs_rng).astype(np.float32)
    _obs_offset = (-1.0 - obs_min * _obs_scale).astype(np.float32)
    S_norm_minmax = (S * _obs_scale + _obs_offset).astype(np.float32)

    return ChunkBundle(
        X=X.astype(np.float32),
        S=S.astype(np.float32),
        S_norm=S_norm.astype(np.float32),
        S_norm_minmax=S_norm_minmax,
        obs_mean=stats.mean.astype(np.float32),
        obs_std=stats.std.astype(np.float32),
        obs_min=obs_min,
        obs_max=obs_max,
        action_min=action_min,
        action_max=action_max,
        demo_lens=np.asarray(lens, dtype=np.int64),
        demo_ids=used,
        horizon=int(horizon),
        action_dim=int(X.shape[-1]),
        obs_dim=int(S.shape[-1]),
    )


# ---------------------------------------------------------------------------
# Stacked-obs (T_obs > 1) post-processing
# ---------------------------------------------------------------------------


def stack_obs_history(
    S: np.ndarray,
    demo_lens: np.ndarray,
    n_obs_steps: int,
) -> np.ndarray:
    """Build a stacked-obs conditioning tensor for ``T_obs > 1``.

    Given per-timestep obs ``S`` of shape ``(N_chunks, D_obs)`` (one row
    per chunk start, in the order produced by ``load_chunks``) and the
    per-demo length array ``demo_lens``, return a new conditioning
    tensor of shape ``(N_chunks, n_obs_steps * D_obs)`` whose row ``t``
    inside demo ``i`` is the concatenation

        [s_{t - n_obs_steps + 1}, s_{t - n_obs_steps + 2}, ..., s_t]

    with boundary frames replicated when ``t - k < 0`` (i.e. earlier
    than the demo's start).  This matches Diffusion Policy's
    ``n_obs_steps`` convention: the most recent obs is the *last*
    block of the concatenated vector, and missing history at the
    beginning is filled by repeating the first available frame.

    Parameters
    ----------
    S
        Per-chunk obs (typically ``bundle.S_norm``).  Shape ``(N, D_obs)``.
    demo_lens
        Per-demo lengths, summing to ``N``.  Same array produced by
        ``load_chunks``.
    n_obs_steps
        ``T_obs``.  Returns ``S`` unchanged (cast to float32) when this
        is 1.  Must be a positive int.

    Returns
    -------
    S_stacked
        Shape ``(N, n_obs_steps * D_obs)``, float32.

    Notes
    -----
    The function is O(N * n_obs_steps) and does not allocate per-demo
    intermediate copies (it slices into the input array in place).
    """
    n_obs_steps = int(n_obs_steps)
    if n_obs_steps < 1:
        raise ValueError(f"n_obs_steps must be >= 1, got {n_obs_steps}")
    S = np.asarray(S, dtype=np.float32)
    if S.ndim != 2:
        raise ValueError(f"S must be 2D, got shape {S.shape}")
    if n_obs_steps == 1:
        return S.astype(np.float32, copy=False)

    N_total, D_obs = S.shape
    demo_lens = np.asarray(demo_lens, dtype=np.int64)
    if int(demo_lens.sum()) != N_total:
        raise ValueError(
            f"demo_lens sum ({int(demo_lens.sum())}) != N_chunks ({N_total})"
        )

    out = np.empty((N_total, n_obs_steps * D_obs), dtype=np.float32)
    cursor = 0
    for L in demo_lens.tolist():
        L = int(L)
        if L == 0:
            continue
        # Demo's slice of S: rows [cursor, cursor + L).
        demo_S = S[cursor : cursor + L]                    # (L, D_obs)
        # For each block k in [0, n_obs_steps), we want the obs at
        # offset (t - (n_obs_steps - 1 - k)) inside the demo, clamped
        # at 0.  The last block (k = n_obs_steps - 1) is the current
        # obs.  Earlier blocks are progressively older.
        for k in range(n_obs_steps):
            lag = (n_obs_steps - 1) - k                    # >= 0
            # Indices into demo_S, clamped at 0 for the first ``lag`` rows.
            idx = np.arange(L, dtype=np.int64) - lag
            idx = np.clip(idx, 0, L - 1)
            out[cursor : cursor + L, k * D_obs : (k + 1) * D_obs] = demo_S[idx]
        cursor += L

    assert cursor == N_total
    return out
