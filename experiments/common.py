"""Shared simulation utilities for experiments in this repo.

This module is the single source of truth for:

  - Locating robomimic-downloaded HDF5 datasets.
  - Reading the dataset's embedded ``env_args`` JSON.
  - Instantiating the matching ``robosuite`` environment.
  - The v1.4.x → robosuite 1.5 ``OSC_POSE`` controller-config shim.
  - Restoring an env's MuJoCo state from a flattened state vector.
  - Probing an env for task success across robosuite versions.
  - Running one episode and returning per-episode diagnostics.

Each experiment script under ``experiments/NN_*/`` adds the
``experiments/`` directory to ``sys.path`` and imports from this module
so we never duplicate the v1.4 → v1.5 shim or the env-construction
logic. See ``experiments/01_demo_replay/replay.py`` for the template
import block.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Dataset / metadata
# ---------------------------------------------------------------------------
def default_dataset_path(task: str = "can") -> Path:
    """Return the default robomimic-downloaded ``ph`` dataset for ``task``.

    ``robomimic.scripts.download_datasets`` places datasets inside the
    active conda env's ``site-packages``.
    """
    import robomimic

    pkg_dir = Path(robomimic.__file__).resolve().parent.parent
    dataset_task = task[:-3] if task.endswith("_dp") else task
    return pkg_dir / "datasets" / dataset_task / "ph" / "low_dim_v141.hdf5"


def read_env_meta(dataset_path: Path) -> dict:
    """Read the dataset's embedded ``env_args`` metadata JSON."""
    with h5py.File(str(dataset_path), "r") as h5:
        env_args_raw = h5["data"].attrs["env_args"]
    if isinstance(env_args_raw, bytes):
        env_args_raw = env_args_raw.decode("utf-8")
    return json.loads(env_args_raw)


# ---------------------------------------------------------------------------
# Controller config compatibility (robosuite v1.4.x  ->  v1.5+)
# ---------------------------------------------------------------------------
# Legacy (robosuite <=1.4) flat single-controller types that v1.5 expects
# nested under ``body_parts.<arm>.type`` in a BASIC composite config.
_LEGACY_PART_CONTROLLERS = {
    "OSC_POSE",
    "OSC_POSITION",
    "JOINT_POSITION",
    "JOINT_VELOCITY",
    "JOINT_TORQUE",
    "IK_POSE",
}

# Single-arm manipulators: the BASIC default registers controllers for
# left/torso/head/base/legs that these robots don't have, which produces
# a noisy "skipping <part>" warning at every reset. Trim to ``right``.
_SINGLE_ARM_ROBOTS = {"Panda", "Sawyer", "UR5e", "Kinova3", "IIWA"}


def upgrade_controller_kwargs(kwargs: dict) -> dict:
    """Translate a v1.4.x flat ``controller_configs`` to robosuite >=1.5.

    v1.4.1 datasets ship ``controller_configs={"type": "OSC_POSE", ...}``
    which robosuite 1.5 rejects with::

        AssertionError: OSC_POSE controller is specified, but not imported

    1.5 expects a *composite-controller* config such as::

        {"type": "BASIC", "body_parts": {"right": {"type": "OSC_POSE", ...}}}

    The cleanest backward-compat path is to load the BASIC default for
    the recorded robot and use that. The BASIC default for Panda happens
    to use exactly the same OSC_POSE params (kp=150, damping_ratio=1,
    output_max=[0.05]*3+[0.5]*3) as the v1.4 dataset, so the on-policy
    behaviour is preserved.
    """
    cc = kwargs.get("controller_configs")
    if not (isinstance(cc, dict) and cc.get("type") in _LEGACY_PART_CONTROLLERS):
        return kwargs

    from robosuite.controllers import load_composite_controller_config

    robots = kwargs.get("robots") or ["Panda"]
    robot_name = robots[0] if isinstance(robots, (list, tuple)) else robots

    new_cc = load_composite_controller_config(
        controller="BASIC", robot=robot_name
    )
    if isinstance(new_cc.get("body_parts"), dict) and robot_name in _SINGLE_ARM_ROBOTS:
        new_cc["body_parts"] = {
            k: v for k, v in new_cc["body_parts"].items() if k == "right"
        }

    upgraded = dict(kwargs)
    upgraded["controller_configs"] = new_cc
    return upgraded


# ---------------------------------------------------------------------------
# Env construction
# ---------------------------------------------------------------------------
def make_env_for_dataset(dataset_path: Path) -> tuple[Any, dict]:
    """Instantiate the robosuite env recorded in the dataset's metadata.

    We build the env directly through ``robosuite.make`` rather than via
    ``robomimic.utils.env_utils``: the only ``robomimic`` release on PyPI
    (0.3.0, 2022) unconditionally imports the legacy ``mujoco_py`` at
    module top, which is incompatible with the new ``mujoco`` bindings
    we standardise on. Robosuite 1.5+ already speaks the new bindings.
    """
    import robosuite  # heavy import, do it lazily

    env_meta = read_env_meta(dataset_path)
    if env_meta.get("type", 1) != 1:
        raise NotImplementedError(
            f"Only robosuite envs (type=1) are supported here; "
            f"dataset declares type={env_meta.get('type')!r}."
        )

    kwargs = dict(env_meta.get("env_kwargs", {}))
    # Force these for batched / headless replay regardless of what the
    # dataset's recording config requested.
    kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
    )
    kwargs = upgrade_controller_kwargs(kwargs)
    env = robosuite.make(env_meta["env_name"], **kwargs)
    return env, env_meta


# ---------------------------------------------------------------------------
# State / success helpers
# ---------------------------------------------------------------------------
def set_sim_state(env: Any, state_vec: np.ndarray) -> None:
    """Restore the underlying MuJoCo state of a robosuite env."""
    sim = env.sim
    if hasattr(sim, "set_state_from_flattened"):
        sim.set_state_from_flattened(state_vec)
    else:  # pragma: no cover - defensive fallback for older robosuite forks
        from robosuite.utils.binding_utils import MjSimState

        sim.set_state(MjSimState.from_flattened(state_vec, sim))
    sim.forward()


def check_success(env: Any) -> bool:
    """Probe robosuite for a task-success flag in a backwards-compat way."""
    for attr in ("_check_success", "check_success"):
        fn = getattr(env, attr, None)
        if callable(fn):
            return bool(fn())
    return False


def get_action_bounds(env: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(low, high)`` numpy arrays for the env's action space."""
    spec = env.action_spec
    low, high = spec[0], spec[1]
    return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_episode(
    env: Any,
    init_state: np.ndarray,
    actions: np.ndarray,
) -> dict:
    """Run one open-loop episode and return per-episode diagnostics.

    ``init_state`` is a flat MuJoCo state vector (as stored in robomimic
    HDF5s); ``actions`` is a ``(H, A)`` array. The env is reset, then
    its sim state is restored to ``init_state``, then the action
    sequence is stepped through.
    """
    env.reset()
    set_sim_state(env, init_state)

    t0 = time.perf_counter()
    last_done = False
    for action in actions:
        _obs, _reward, last_done, _info = env.step(action)
    elapsed = time.perf_counter() - t0

    H = int(len(actions))
    return {
        "horizon": H,
        "success": check_success(env),
        "done": bool(last_done),
        "elapsed_s": float(elapsed),
        "steps_per_s": float(H / max(elapsed, 1e-9)),
    }
