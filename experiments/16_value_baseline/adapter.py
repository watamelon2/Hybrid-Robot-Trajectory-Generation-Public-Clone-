"""Adapter wiring V_phi into the unified driver's planning loop.

The control-variate trick is: replace the per-trajectory cost
``c_i = -reward_i`` with ``c_i - sum_t V_phi(s_{i,t})``. The ``sum_t``
subtracts the *expected* portion of the cost (under V_phi's
prediction) and leaves only the *advantage* portion. Variance over
the M perturbed trajectories drops; the softmax-weighted update
isn't biased because the subtracted term shares no information with
which trajectory was sampled given the same s_0.

Contract:

  * ``make_rollout_fn_with_obs`` returns a rollout_fn that yields
    ``(rewards, obs_trajs)`` where ``rewards`` is shape ``(M,)`` and
    ``obs_trajs`` is shape ``(M, H, obs_dim)``. See exp 04's
    ``run_can.make_rollout_fn_with_obs``.
  * ``wrap_cost_fn_with_value_baseline(base, v_phi)`` returns a new
    cost_fn that takes ``(rewards, obs_trajs)`` and traj_batch and
    returns the V_phi-adjusted costs.

The unified driver (``experiments/15_unified_hybrid/run_can.py``)
dispatches to these when ``--value-baseline-ckpt`` is set.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np

# Reuse exp 16's ValuePhi loader.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from value_model import ValuePhi  # noqa: E402


# Type alias: the "rewards + per-step obs" payload returned by the
# obs-collecting rollout_fn.
RolloutResultWithObs = tuple[np.ndarray, np.ndarray]
# rewards: (M,)
# obs_trajs: (M, H, obs_dim)


def wrap_cost_fn_with_value_baseline(
    base_cost_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    v_phi: ValuePhi,
) -> Callable[[RolloutResultWithObs, np.ndarray], np.ndarray]:
    """Return a cost_fn that subtracts ``sum_t V_phi(s_t)`` per trajectory.

    Args:
        base_cost_fn: the underlying cost function. Original signature
            is ``(rewards: (M,), traj_batch: (M, H, A)) -> (M,)`` --
            it operates on the rewards alone.
        v_phi: a loaded ``ValuePhi`` instance.

    Returns:
        A new ``cost_fn`` with signature
        ``((rewards, obs_trajs), traj_batch) -> (M,)`` that subtracts
        ``sum_t V_phi(s_{i,t})`` from each per-trajectory cost.
    """
    def cost_fn(
        states_batch: RolloutResultWithObs,
        traj_batch: np.ndarray,
    ) -> np.ndarray:
        rewards, obs_trajs = states_batch
        # base_cost_fn doesn't know about obs_trajs; pass only rewards.
        base_costs = np.asarray(
            base_cost_fn(rewards, traj_batch), dtype=np.float64
        )
        if base_costs.shape != (rewards.shape[0],):
            raise ValueError(
                f"base_cost_fn returned shape {base_costs.shape}; "
                f"expected ({rewards.shape[0]},)"
            )
        # (M, H, obs_dim) -> (M, H) -> (M,)
        values = v_phi.value_batch(obs_trajs)
        if values.shape != obs_trajs.shape[:2]:
            raise ValueError(
                f"v_phi.value_batch returned shape {values.shape}; "
                f"expected {obs_trajs.shape[:2]}"
            )
        v_sum = values.sum(axis=1).astype(np.float64)
        return base_costs - v_sum

    return cost_fn


__all__ = [
    "RolloutResultWithObs",
    "ValuePhi",
    "wrap_cost_fn_with_value_baseline",
]
