"""Per-demo trajectory dataset for the flow-matching prior.

Whereas exp 03's BC dataset emits *per-step* (obs, action) pairs,
the flow-matching prior models the *whole trajectory* conditioned
on the demo's initial obs ``s0``. We therefore re-bucket the same
HDF5 demos into fixed-length action chunks of length H starting at
each demo's t=0.

Demos that are shorter than H are tail-padded by repeating the
last action (a deliberately mild form of leakage; we only need the
prior to be reasonable, not pixel-perfect). Demos longer than H
are truncated to the first H steps so every example is the same
shape and matches the planner's horizon exactly.

Output:

  X: (N, H * action_dim) float32 -- flat-vectorised target trajectories
  S: (N, obs_dim)        float32 -- demo's flat initial obs
  obs_stats: ObsStats             -- mean/std normalisation reused at eval
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Reuse exp 03's obs-key list and HDF5 reader. Same compatibility
# contract: the live env will produce the same OBS_KEYS at planning
# time so the prior's conditioning is in-distribution.
_THIS_DIR = Path(__file__).resolve().parent
_EXP_03 = _THIS_DIR.parent / "03_bc_baseline"
for _p in (_EXP_03,):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Import from exp03's dataset.py, not from ourselves (same filename).
import importlib.util as _iu
_ds_spec = _iu.spec_from_file_location("exp03_dataset", _EXP_03 / "dataset.py")
_ds = _iu.module_from_spec(_ds_spec)
_ds_spec.loader.exec_module(_ds)
OBS_KEYS = _ds.OBS_KEYS
ObsStats = _ds.ObsStats
_DATASET_OBJECT_SLICES = _ds._DATASET_OBJECT_SLICES
# Task-aware helpers re-exported for Phase-B (2026-05-29): the
# DP-paper-faithful 23-D obs schema is selected via ``task='can_dp'``.
# These wrappers expose the same registry that ``dataset_v3.load_chunks``
# already uses internally, so train and runtime obs vectors are
# bit-identical for the synthetic-key path.
get_obs_keys = _ds.get_obs_keys
get_object_slices = _ds.get_object_slices
get_synthetic_keys = _ds.get_synthetic_keys
compute_synthetic_obs_keys = _ds.compute_synthetic_obs_keys

import h5py  # noqa: E402


def load_trajectories(
    dataset_path: Path,
    horizon: int,
    obs_keys: tuple[str, ...] = OBS_KEYS,
    n_demos: int | None = None,
    pad: str = "edge",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return ``(actions, init_obs, demo_lengths, demo_ids)``.

    actions:    (N, H, action_dim) float32, padded/truncated to H
    init_obs:   (N, D)              float32
    demo_lens:  (N,)                int32  -- original demo lengths
    """
    act_chunks: list[np.ndarray] = []
    obs0: list[np.ndarray] = []
    lens: list[int] = []
    used: list[str] = []
    with h5py.File(str(dataset_path), "r") as h5:
        keys = sorted(
            h5["data"].keys(), key=lambda k: int(k.split("_")[1])
        )
        if n_demos is not None and n_demos > 0:
            keys = keys[:n_demos]
        for demo_id in keys:
            demo = h5["data"][demo_id]
            actions = np.asarray(demo["actions"][()], dtype=np.float32)
            H_demo = int(actions.shape[0])
            A = int(actions.shape[1])
            if H_demo >= horizon:
                clip = actions[:horizon]
            else:
                pad_n = horizon - H_demo
                if pad == "edge":
                    last = actions[-1:].repeat(pad_n, axis=0)
                else:
                    last = np.zeros((pad_n, A), dtype=np.float32)
                clip = np.concatenate([actions, last], axis=0)

            # Initial obs: build it the same way exp 03 does, but
            # only at t=0 (no flatten across all timesteps).
            obs0_dict: dict[str, np.ndarray] = {}
            for k in obs_keys:
                if k in _DATASET_OBJECT_SLICES:
                    src, sl = _DATASET_OBJECT_SLICES[k]
                    obs0_dict[k] = np.asarray(demo["obs"][src][0, sl])
                else:
                    obs0_dict[k] = np.asarray(demo["obs"][k][0])
            s0 = np.concatenate(
                [obs0_dict[k].astype(np.float32) for k in obs_keys],
                axis=-1,
            )

            act_chunks.append(clip[None, :, :])
            obs0.append(s0[None, :])
            lens.append(H_demo)
            used.append(demo_id)

    actions_arr = np.concatenate(act_chunks, axis=0)
    obs0_arr = np.concatenate(obs0, axis=0)
    lens_arr = np.asarray(lens, dtype=np.int32)
    return actions_arr, obs0_arr, lens_arr, used


def compute_obs_stats(
    obs: np.ndarray, eps: float = 1e-6
) -> ObsStats:
    mean = obs.mean(axis=0).astype(np.float32)
    std = obs.std(axis=0).astype(np.float32)
    std = np.maximum(std, eps)
    return ObsStats(mean=mean, std=std)
