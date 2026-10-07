"""Demo-phase pseudo-cost (doc 09 §7.1, route 3) -- follow-up #3.

A zero-train dense reward for sparse-terminal tasks (tool_hang/transport):
score how far along the *nearest demonstration* the current state has
progressed.  Concretely, every demo timestep carries a monotone phase
``in [0, 1]`` (fraction of the way through its demo); at run time we map
the live obs to its nearest demo timestep (in the normalised obs feature
space the ``demo_1nn`` prior already uses) and emit ``scale * phase`` as
the per-step shaped reward.  This converts the *static* demo target into a
dense *progress* signal -- no value training, reuses the demo bank.

Caveat (documented in §7.1): the phase potential inherits the demo's
suboptimality, exactly like a bad prior.  It is the cheapest of the four
densification routes, not the principled one (that is potential-based
shaping / a learned value, routes 1-2).

Public API:
    build_demo_phase_shaper(dataset_path, task, obs_keys, n_demos, scale)
        -> shaping_fn(obs_dict) -> float
    wrap_env_with_demo_phase_reward(env, task, dataset_path, obs_keys,
        n_demos, scale, include_sparse) -> env  (monkey-patches env.step)

The nearest-phase core (``_nearest_phase``) is pure-numpy and unit-tested
without robosuite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

_EPS = 1e-6


def _obs_vec_from_dict(
    obs_dict: dict, obs_keys: tuple[str, ...]
) -> np.ndarray:
    """Concatenate obs_dict[k] in obs_keys order (matches the demo bank).

    Mirrors ``unified_loop._make_obs_vec_for_prior`` for the non-synthetic
    key path (tool_hang/transport declare no synthetic keys).
    """
    parts = []
    for k in obs_keys:
        if k not in obs_dict:
            raise KeyError(
                f"runtime obs missing {k!r} (available: "
                f"{sorted(obs_dict.keys())})"
            )
        parts.append(np.asarray(obs_dict[k], dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=-1)


def _nearest_phase(
    vec: np.ndarray,
    bank_norm: np.ndarray,
    phase: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> float:
    """Phase of the bank row nearest ``vec`` in normalised feature space.

    Parameters
    ----------
    vec:        (D,) raw live obs vector.
    bank_norm:  (N, D) pre-normalised demo obs rows.
    phase:      (N,) per-row phase in [0, 1].
    mean, std:  (D,) normalisation stats (std clipped >= eps upstream).
    """
    v = (np.asarray(vec, dtype=np.float32) - mean) / std
    # Squared Euclidean to every bank row; argmin -> nearest demo step.
    d2 = ((bank_norm - v[None, :]) ** 2).sum(axis=1)
    return float(phase[int(np.argmin(d2))])


def _phase_from_bank(
    demo_starts: np.ndarray, demo_lengths: np.ndarray, n_rows: int
) -> np.ndarray:
    """Per-row phase in [0, 1]: fraction of the way through each demo."""
    phase = np.zeros(n_rows, dtype=np.float32)
    for start, length in zip(demo_starts, demo_lengths):
        start = int(start)
        length = int(length)
        if length <= 1:
            phase[start:start + length] = 1.0
            continue
        phase[start:start + length] = np.linspace(
            0.0, 1.0, length, dtype=np.float32
        )
    return phase


def build_demo_phase_shaper(
    dataset_path: Path,
    task: str,
    obs_keys: tuple[str, ...],
    n_demos: int | None = None,
    scale: float = 10.0,
) -> Callable[[dict], float]:
    """Return ``shaping_fn(obs_dict) -> scale * nearest-demo phase``."""
    from priors_zerotrain import load_demo_bank

    bank = load_demo_bank(
        dataset_path, n_demos=n_demos, obs_keys=obs_keys, task=task,
        action_history_len=0, verbose=False,
    )
    obs = np.asarray(bank.obs, dtype=np.float32)
    mean = np.asarray(bank.obs_mean, dtype=np.float32)
    std = np.maximum(np.asarray(bank.obs_std, dtype=np.float32), _EPS)
    bank_norm = (obs - mean[None, :]) / std[None, :]
    phase = _phase_from_bank(
        bank.demo_starts, bank.demo_lengths, obs.shape[0]
    )
    scale_f = float(scale)

    def shaping_fn(obs_dict: dict) -> float:
        vec = _obs_vec_from_dict(obs_dict, obs_keys)
        return scale_f * _nearest_phase(vec, bank_norm, phase, mean, std)

    return shaping_fn


def wrap_env_with_demo_phase_reward(
    env: Any,
    task: str,
    dataset_path: Path,
    obs_keys: tuple[str, ...],
    *,
    n_demos: int | None = None,
    scale: float = 10.0,
    include_sparse: bool = True,
    verbose: bool = True,
) -> Any:
    """Monkey-patch ``env.step`` to add the demo-phase progress reward.

    Idempotent (won't double-wrap).  Mirrors
    ``env_dense_rewards.wrap_env_with_dense_reward`` so the success
    heuristic (>0.5, dominated by the sparse +bonus) keeps working when
    ``include_sparse=True``.
    """
    if getattr(env, "_demo_phase_wrapped", False):
        return env
    shaping_fn = build_demo_phase_shaper(
        dataset_path, task, obs_keys, n_demos=n_demos, scale=scale
    )
    original_step = env.step

    def shaped_step(action):
        obs, sparse_r, done, info = original_step(action)
        try:
            obs_dict = env._get_observations(force_update=True)
        except Exception:
            obs_dict = obs
        shaped = float(shaping_fn(obs_dict))
        total = shaped + (float(sparse_r) if include_sparse else 0.0)
        return obs, total, done, info

    env.step = shaped_step  # type: ignore[method-assign]
    env._demo_phase_wrapped = True  # type: ignore[attr-defined]
    env._demo_phase_task = task  # type: ignore[attr-defined]
    if verbose:
        print(
            f"[demo_phase_reward] wrapped env.step for task={task!r} "
            f"(scale={scale}, include_sparse={include_sparse})"
        )
    return env
