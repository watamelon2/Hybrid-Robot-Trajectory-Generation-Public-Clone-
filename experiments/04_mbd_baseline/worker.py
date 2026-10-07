"""Parallel-rollout worker functions for MBD.

Defined in a standalone module (not inside ``run_can.py``) so that
``multiprocessing.Pool`` can pickle them regardless of how the
caller imports ``run_can`` (e.g. exp 07's ``importlib``-based import).

Workers also propagate any runtime env perturbations (cost-time
goal-pose, dynamics mass/friction) from the main process to each
worker's env via the ``perturbations`` initarg.  This is the
single piece that makes the cost-time and dynamics adaptation
experiments work end-to-end -- without it, workers spawn fresh
envs from MJCF and silently use the canonical (unperturbed)
goal/dynamics, so MBD's planning would run against a different
cost / dynamics from the one the main rollout executes.  See
``docs/adaptation/01_cost_time_spec.md`` and
``docs/adaptation/02_dynamics_spec.md`` for the experiment plans
this enables.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure experiments/ is on sys.path for spawn-context workers,
# which start with a clean sys.path that does not include the
# parent process's modifications.
_THIS_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _THIS_DIR.parent
for _p in (_THIS_DIR, _EXPERIMENTS_DIR):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

_worker_env = None


def _worker_init(
    dataset_path_str: str,
    perturbations: dict | None = None,
) -> None:
    """Initialise a worker process with its own robosuite env.

    Parameters
    ----------
    dataset_path_str
        Path to the .hdf5 dataset.  Forwarded to
        ``common.make_env_for_dataset``.
    perturbations
        Optional dict of runtime env perturbations to apply after
        env creation.  Recognised keys:

        - ``task`` (str): one of ``can`` / ``lift`` / ``square`` /
          ``tool_hang`` / ``transport``.  Required if any other
          perturbation key is set.
        - ``goal_xyz`` ((float, float, float)): world-frame
          translation of the task's goal body, in metres.  Forwarded
          to ``env_perturbations.apply_goal_perturbation``.
          ``(0, 0, 0)`` is a no-op.
        - ``mass_scale`` (float): multiplicative scale on link
          masses.  Forwarded to
          ``env_perturbations.apply_dynamics_perturbation``.
          ``1.0`` (default) is a no-op.
        - ``friction_scale`` (float): multiplicative scale on
          object friction.  Same as above.  ``1.0`` is a no-op.

        ``None`` (default) means "no perturbations" and is the
        canonical (slate E / slate G / ...) behaviour.
    """
    global _worker_env
    from common import make_env_for_dataset

    _worker_env, _ = make_env_for_dataset(Path(dataset_path_str))
    if hasattr(_worker_env, "reward_shaping"):
        _worker_env.reward_shaping = True

    if perturbations:
        _apply_perturbations(_worker_env, perturbations)


def _apply_perturbations(env, perturbations: dict) -> None:
    """Apply runtime env perturbations to a worker env.

    Mirrors the corresponding apply_* call in ``run_can.main``'s
    main env, so the worker's reward / dynamics agree with the
    main process's by construction.

    The exp 15 dir is added to ``sys.path`` lazily here so that
    workers which don't perturb don't pay for the import cost or
    for any module-import side effects.
    """
    task = perturbations.get("task")
    goal_xyz = perturbations.get("goal_xyz")
    mass_scale = perturbations.get("mass_scale")
    friction_scale = perturbations.get("friction_scale")

    # env_perturbations.py lives in experiments/15_unified_hybrid;
    # the spawn-context worker's sys.path already includes
    # experiments/ (set at module-load time), but not the exp15 dir
    # itself.  Add it lazily.
    exp15_dir = str(_EXPERIMENTS_DIR / "15_unified_hybrid")
    if exp15_dir not in sys.path:
        sys.path.insert(0, exp15_dir)

    if (
        task is not None
        and goal_xyz is not None
        and tuple(goal_xyz) != (0.0, 0.0, 0.0)
    ):
        from env_perturbations import apply_goal_perturbation
        apply_goal_perturbation(env, task, tuple(goal_xyz))

    if (mass_scale is not None and mass_scale != 1.0) or (
        friction_scale is not None and friction_scale != 1.0
    ):
        from env_perturbations import apply_dynamics_perturbation
        apply_dynamics_perturbation(
            env,
            task=task,
            mass_scale=float(mass_scale) if mass_scale is not None else 1.0,
            friction_scale=float(friction_scale) if friction_scale is not None else 1.0,
        )


def _worker_rollout(args: tuple[np.ndarray, np.ndarray]) -> tuple[float, bool]:
    """Run one open-loop rollout in the worker's persistent env."""
    global _worker_env
    from common import check_success, set_sim_state

    init_state, actions = args
    env = _worker_env
    env.reset()
    set_sim_state(env, init_state)
    total_reward = 0.0
    for action in actions:
        _obs, reward, _done, _info = env.step(action)
        total_reward += float(reward)
    success = bool(check_success(env))
    return total_reward, success


def _obs_dict_to_vec_for_value(obs_dict: dict) -> np.ndarray:
    """Concat obs dict into the OBS_KEYS-ordered flat vector V_phi expects.

    Importing OBS_KEYS lazily so this module stays importable in test
    contexts that don't set up the BC sys.path tweak.
    """
    from dataset import OBS_KEYS

    parts = []
    for k in OBS_KEYS:
        if k not in obs_dict:
            raise KeyError(
                f"runtime obs missing {k!r} (available: "
                f"{sorted(obs_dict.keys())})"
            )
        parts.append(np.asarray(obs_dict[k], dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=-1)


def _worker_rollout_with_obs(
    args: tuple[np.ndarray, np.ndarray],
) -> tuple[float, bool, np.ndarray]:
    """Like ``_worker_rollout`` but also returns the per-step obs traj.

    Used by Branch A's V_phi control variate (exp 16); the obs traj
    is shape ``(H, obs_dim)`` with one observation collected *before*
    each action is applied. ``rollout_fn_with_obs`` aggregates these
    over M trajectories into ``(M, H, obs_dim)``.
    """
    global _worker_env
    from common import check_success, set_sim_state

    init_state, actions = args
    env = _worker_env
    env.reset()
    set_sim_state(env, init_state)

    obs_buf: list[np.ndarray] = []
    total_reward = 0.0
    for action in actions:
        obs_dict = env._get_observations(force_update=True)
        obs_buf.append(_obs_dict_to_vec_for_value(obs_dict))
        _obs, reward, _done, _info = env.step(action)
        total_reward += float(reward)
    success = bool(check_success(env))
    obs_traj = np.asarray(obs_buf, dtype=np.float32)
    return total_reward, success, obs_traj
