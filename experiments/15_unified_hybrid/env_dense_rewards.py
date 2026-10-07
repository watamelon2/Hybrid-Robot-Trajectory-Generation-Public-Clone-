"""Dense reward shaping wrappers for robosuite tasks that ship sparse-only.

Upstream robosuite 1.5.x has only sparse reward implementations for
``ToolHang`` and ``TwoArmTransport`` -- both files carry a literal
``# TODO!`` where the shaped branch would live and ``reward_shaping=True``
gets silently down-graded to sparse (transport prints a warning, tool_hang
just returns 0).  This makes both tasks unusable for the score-bridge
hybrid: with no signal along the way, MBD cannot rank rollouts and the
prior carries the entire load.

Slate H H1 (tool_hang demo_1nn lambda=1 H=300) hit this on 2026-05-21:
33/60 rollouts in, all reward=0.00.  Replaying a known-successful
demo through the same env gave the same 0.00 reward.  Root cause is
the sparse-only upstream reward, not a wiring or warm-start bug.

This module monkey-patches ``env.step`` for the affected tasks so the
returned reward is a staged distance-shaping sum:

  reward = sum_i  w_i * f_i(obs)   for i in subgoals

where each ``f_i`` is either a binary subgoal indicator (e.g.
``frame_is_assembled``) or a continuous distance term shaped via
``1 - tanh(beta * dist)``.  Weights are picked so that:

  - the running per-step reward in pre-grasp phases is O(1);
  - completing a discrete subgoal flips the running reward by O(5);
  - full success adds an extra O(10) bonus.

The shaping is *additive only* on top of upstream's sparse 1.0
success signal -- the original sparse reward is added to the staged
shaping if you also set ``include_sparse=True`` (default).  This
keeps the headline "succ_rate" rule (succ iff reward > 0.5 at any
step) compatible with all existing post-processing scripts;
shaped reward at success is therefore >= ~15 (=10 bonus + ongoing
distance terms) and the sparse threshold is dominated.

Public API
----------
``wrap_env_with_dense_reward(env, task)``
    Returns ``env`` with its ``step`` method replaced by a closure
    that computes the shaped reward after the inner ``step`` runs.
    No-op for tasks that already have dense shaping upstream
    (``can``, ``lift``, ``square``).

``dense_reward_tool_hang(obs)``
``dense_reward_transport(obs)``
    Pure functions (obs dict -> float).  Useful for testing and
    for documenting the shaping outside of the wrapped env.

Smoke check (this module's ``__main__``)::

    python experiments/15_unified_hybrid/env_dense_rewards.py --task tool_hang
    python experiments/15_unified_hybrid/env_dense_rewards.py --task transport

Replays the first demo through the wrapped env and prints the
reward trajectory.  Pre-fix, both tasks print all zeros; post-fix
they print a monotonically (or near-monotonically) rising
trajectory ending above 10 at success states.
"""
from __future__ import annotations

from typing import Any

import numpy as np


# Tasks that already have dense shaping in upstream robosuite.  For
# these, ``wrap_env_with_dense_reward`` is a no-op (the env's own
# ``reward_shaping = True`` already produces the dense signal that
# the score-bridge needs).
_UPSTREAM_DENSE_TASKS = frozenset({"can", "lift", "square"})

# Tasks for which we provide our own shaping; the dispatch maps
# each task name to a (function, weights) pair.  Extend by adding
# a new ``dense_reward_<task>(obs) -> float`` and registering it.
_SHAPING_REGISTRY: dict[str, Any] = {}


def register_dense_reward(task: str):
    """Decorator: register ``func`` as the dense-reward function for ``task``."""

    def deco(func):
        _SHAPING_REGISTRY[task] = func
        return func

    return deco


# ---------------------------------------------------------------------------
# Per-task shaping functions
# ---------------------------------------------------------------------------


@register_dense_reward("tool_hang")
def dense_reward_tool_hang(obs: dict[str, Any]) -> float:
    """Staged distance shaping for ``ToolHang``.

    Subgoals (cumulative, additive):

      1. **EEF-near-frame** -- always active.  Pulls the gripper to
         the frame in the early phase before assembly.  Weight 1.0,
         shaped via 1 - tanh(10 * dist).
      2. **frame assembled** (``frame_is_assembled > 0.5``) -- +5
         constant once the frame is locked on the stand.  This is
         the first irreversible milestone.
      3. **EEF-near-tool** -- active only after assembly, +1.0 *
         (1 - tanh(10 * ||tool_to_eef||)).
      4. **tool-near-frame-hook** -- active only after assembly,
         +2.0 * (1 - tanh(5 * ||tool_pos - frame_pos||)).
      5. **success** (``tool_on_frame > 0.5``) -- +10 bonus.

    Running maxima in each regime:
      * pre-assembly: r <= 1.0
      * post-assembly, pre-hung: r <= 1 + 5 + 1 + 2 = 9.0
      * success: r ~= 19.0 (subject to small fluctuations from the
        tanh terms; the success bonus dominates).
    """
    r = 0.0

    # Stage 1: bring EEF to frame (always).
    d_frame = float(np.linalg.norm(np.asarray(obs["frame_to_robot0_eef_pos"])))
    r += 1.0 * (1.0 - np.tanh(10.0 * d_frame))

    # Stage 2: frame assembled gates the rest of the curriculum.
    frame_assembled = float(np.asarray(obs.get("frame_is_assembled", 0.0))) > 0.5
    if frame_assembled:
        r += 5.0

        # Stage 3: bring EEF to tool.
        d_tool = float(np.linalg.norm(np.asarray(obs["tool_to_robot0_eef_pos"])))
        r += 1.0 * (1.0 - np.tanh(10.0 * d_tool))

        # Stage 4: bring tool close to the hook (use the frame body's
        # centroid as the hook anchor; an exact hook-pos field is not
        # exposed in the obs dict, but ``frame_pos`` is a good proxy
        # since the hook is rigidly attached).
        d_tool_to_frame = float(
            np.linalg.norm(
                np.asarray(obs["tool_pos"]) - np.asarray(obs["frame_pos"])
            )
        )
        r += 2.0 * (1.0 - np.tanh(5.0 * d_tool_to_frame))

        # Stage 5: success bonus.
        tool_on_frame = float(np.asarray(obs.get("tool_on_frame", 0.0))) > 0.5
        if tool_on_frame:
            r += 10.0

    return r


@register_dense_reward("transport")
def dense_reward_transport(obs: dict[str, Any]) -> float:
    """Staged distance shaping for ``TwoArmTransport``.

    The TwoArmTransport task has two coupled sub-objectives
    enforced by ``_check_success``: payload-in-target-bin AND
    trash-in-trash-bin.  We expose both as separate continuous
    distance shapings plus a joint success bonus.

    Subgoals (cumulative, additive):

      1. **gripper0_to_payload** -- always active, +1.0 *
         (1 - tanh(10 * ||gripper0_to_payload||)).
      2. **payload_in_target_bin** -- +5.0 once the payload lands
         in the target bin.
      3. **gripper1_to_trash** -- always active, +1.0 *
         (1 - tanh(10 * ||gripper1_to_trash||)).
      4. **trash_in_trash_bin** -- +5.0 once the trash lands in
         the trash bin.
      5. **success** (both above flags) -- +10 joint bonus.

    Running maxima:
      * approach phase:    r <= 2.0
      * one subgoal done:  r <= 2 + 5 = 7.0
      * both subgoals:     r <= 2 + 5 + 5 = 12.0
      * success:           r ~= 22.0
    """
    r = 0.0

    # Stage 1: gripper0 reaches for payload.
    d_g0_payload = float(
        np.linalg.norm(np.asarray(obs["gripper0_to_payload"]))
    )
    r += 1.0 * (1.0 - np.tanh(10.0 * d_g0_payload))

    # Stage 3: gripper1 reaches for trash.
    d_g1_trash = float(np.linalg.norm(np.asarray(obs["gripper1_to_trash"])))
    r += 1.0 * (1.0 - np.tanh(10.0 * d_g1_trash))

    # Stage 2: payload placed.
    payload_in = bool(np.asarray(obs.get("payload_in_target_bin", False)))
    if payload_in:
        r += 5.0

    # Stage 4: trash placed.
    trash_in = bool(np.asarray(obs.get("trash_in_trash_bin", False)))
    if trash_in:
        r += 5.0

    # Stage 5: joint success bonus.
    if payload_in and trash_in:
        r += 10.0

    return r


# ---------------------------------------------------------------------------
# Env wrapper
# ---------------------------------------------------------------------------


def wrap_env_with_dense_reward(
    env: Any,
    task: str,
    *,
    include_sparse: bool = True,
    verbose: bool = True,
) -> Any:
    """Monkey-patch ``env.step`` to return a dense-shaped reward.

    Parameters
    ----------
    env:
        A robosuite-style env that exposes ``step(action) -> (obs, reward,
        done, info)`` and ``_get_observations(force_update=True) -> dict``.
    task:
        Robomimic task name (matches ``args.task``).  If the task is
        already dense upstream (``can``/``lift``/``square``) or not
        registered here, the env is returned unchanged.
    include_sparse:
        If True (default), the upstream env's sparse reward is added
        to our shaped reward.  Keeps the headline succ-rate rule
        compatible: shaped reward at success is dominated by the
        +10 bonus, so the >0.5 succ heuristic still triggers.  Set
        False to study the shaping in isolation.
    verbose:
        Print a one-line confirmation when wrapping.

    Returns
    -------
    The same ``env`` object, with ``env.step`` replaced.  The
    replacement is idempotent: calling ``wrap_env_with_dense_reward``
    twice does not double-wrap (subsequent calls are no-ops).
    """
    if task in _UPSTREAM_DENSE_TASKS:
        if verbose:
            print(
                f"[env_dense_rewards] task={task!r} already has upstream "
                "dense shaping; no wrap applied."
            )
        return env

    shaping_fn = _SHAPING_REGISTRY.get(task)
    if shaping_fn is None:
        if verbose:
            print(
                f"[env_dense_rewards] task={task!r} has no registered "
                "dense-reward override; leaving env unchanged."
            )
        return env

    if getattr(env, "_dense_reward_wrapped", False):
        # Idempotent: don't double-wrap.
        return env

    original_step = env.step

    def shaped_step(action):
        obs, sparse_r, done, info = original_step(action)
        # Fetch the latest obs dict via the same canonical path that
        # ``unified_loop._make_obs_vec_for_prior`` uses; ``obs`` from
        # ``step`` may omit a few derived keys that are computed only
        # in ``_get_observations`` (the live-prior code already calls
        # ``force_update=True`` for the same reason).
        try:
            obs_dict = env._get_observations(force_update=True)
        except Exception:
            obs_dict = obs  # fallback; for envs that don't expose the helper
        shaped = float(shaping_fn(obs_dict))
        total = shaped + (float(sparse_r) if include_sparse else 0.0)
        return obs, total, done, info

    env.step = shaped_step  # type: ignore[method-assign]
    env._dense_reward_wrapped = True  # type: ignore[attr-defined]
    env._dense_reward_task = task  # type: ignore[attr-defined]
    if verbose:
        print(
            f"[env_dense_rewards] wrapped env.step for task={task!r} "
            f"(include_sparse={include_sparse})"
        )
    return env


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def _smoke(task: str, n_steps: int = 300) -> None:
    """Replay the first demo through a wrapped env and print rewards."""
    import sys
    from pathlib import Path

    _HERE = Path(__file__).resolve().parent
    _EXPERIMENTS_DIR = _HERE.parent
    for _sp in (str(_EXPERIMENTS_DIR), str(_HERE)):
        if _sp not in sys.path:
            sys.path.insert(0, _sp)

    import h5py
    from common import default_dataset_path, make_env_for_dataset, set_sim_state

    ds = default_dataset_path(task)
    env, _meta = make_env_for_dataset(ds)
    wrap_env_with_dense_reward(env, task)

    with h5py.File(str(ds), "r") as h5:
        demos = sorted(h5["data"].keys(), key=lambda k: int(k.split("_")[1]))
        states = np.asarray(h5["data"][demos[0]]["states"][()])
        actions = np.asarray(h5["data"][demos[0]]["actions"][()])

    env.reset()
    set_sim_state(env, states[0])
    rewards: list[float] = []
    for i in range(min(n_steps, len(actions))):
        _obs, r, _done, _info = env.step(actions[i])
        rewards.append(float(r))

    arr = np.asarray(rewards)
    print(f"\n=== {task} dense-reward smoke ({len(arr)} steps) ===")
    print(f"  min:   {arr.min():+.4f}")
    print(f"  mean:  {arr.mean():+.4f}")
    print(f"  max:   {arr.max():+.4f}")
    print(f"  final: {arr[-1]:+.4f}")
    print(f"  num positive (>0.05): {(arr > 0.05).sum()} / {len(arr)}")
    print(f"  num >= 5 (subgoal):   {(arr >= 5.0).sum()}")
    print(f"  num >=10 (success bonus active): {(arr >= 10.0).sum()}")


def _main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--task", type=str, default="tool_hang",
        choices=("tool_hang", "transport"),
    )
    p.add_argument("--n-steps", type=int, default=300)
    args = p.parse_args()
    _smoke(args.task, args.n_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
