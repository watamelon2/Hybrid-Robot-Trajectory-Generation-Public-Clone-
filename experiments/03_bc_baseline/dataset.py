"""Robomimic low-dim HDF5 -> in-memory (obs, action) tensors for BC.

The proficient-human ``can/ph/low_dim_v141.hdf5`` dataset stores 200
demos under ``data/demo_<i>/{actions, obs/...}``. Every demo's
``obs/<key>`` is a ``(H_i, D_k)`` array; we concatenate a fixed list of
keys along the feature axis to get a flat per-step observation, then
flatten across all demos so the result is a single
``(sum_i H_i, D)`` matrix paired with a ``(sum_i H_i, A)`` action
matrix.

The same ``OBS_KEYS`` order is consumed at evaluation time
(``eval.py``) when building the obs vector from the live robosuite env,
so train-time and eval-time feature axes line up exactly. The
runtime-vs-dataset key naming gotcha (``object`` vs ``object-state``)
is handled in ``eval.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset


# Fixed feature order used by both training and evaluation. Sum of the
# per-key dimensions on can/ph/low_dim_v141 is 46. Keep this list
# stable across runs -- changing it invalidates saved checkpoints.
#
# Two robosuite 1.4 -> 1.5 drifts forced compromises here:
#
#   1. robosuite 1.5 does not expose ``robot0_eef_vel_lin`` /
#      ``robot0_eef_vel_ang`` as separate observables (they're rolled
#      into ``robot0_proprio-state``). We drop both keys -- this also
#      matches the standard ``low_dim`` config in ``robomimic.algo.bc``.
#
#   2. The dataset's ``obs/object`` is the concatenation
#      ``[Can_pos(3), Can_quat(4), Can_to_eef_pos(3), Can_to_eef_quat(4)]``
#      in robosuite 1.4 ordering, whereas the live env's ``object-state``
#      uses ``[Can_to_eef_*, Can_pos, Can_quat]`` AND the
#      ``Can_to_eef_*`` numerical values themselves do not agree between
#      versions (different reference frame). The portable subset is
#      ``Can_pos`` and ``Can_quat`` -- both match exactly between
#      recording and runtime. We pull them from the dataset by slicing
#      ``obs/object[:, :7]`` (see ``_DATASET_OBJECT_SLICES``) and from
#      the live env via the dedicated ``Can_pos`` / ``Can_quat`` keys.
OBS_KEYS: tuple[str, ...] = (
    "Can_pos",
    "Can_quat",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "robot0_joint_pos",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
)


# Dataset-side: where to find each "synthesized" key inside the HDF5.
# Maps an OBS_KEYS entry to ``(hdf5_obs_key, slice)``. Anything not
# listed here is read directly as ``demo["obs"][k]``.
_DATASET_OBJECT_SLICES: dict[str, tuple[str, slice]] = {
    "Can_pos":  ("object", slice(0, 3)),
    "Can_quat": ("object", slice(3, 7)),
}


# ---------------------------------------------------------------------------
# Task-aware obs schema (added 2026-05-19 for Slate F cross-task work).
#
# The Can-specific ``OBS_KEYS`` / ``_DATASET_OBJECT_SLICES`` above are
# kept verbatim because the canonical BC checkpoint
# (``experiments/03_bc_baseline/results/policy.pt``) was trained
# against them and changing the default would invalidate the saved
# obs_mean / obs_std / state_dict.
#
# For other tasks (``square/ph`` first; future cross-task additions
# go here), use ``get_obs_keys(task)`` and ``get_object_slices(task)``.
# The zero-train retrieval priors in
# ``experiments/15_unified_hybrid/priors_zerotrain.py`` honour these
# helpers; BC ``eval.py`` still uses the legacy constants.
#
# Layout assumption: the ``obs/object`` field is laid out as
# ``[obj_pos(3), obj_quat(4), obj_to_eef_pos(3), obj_to_eef_quat(4)]``
# for every robomimic ``*/ph`` dataset we've inspected (verified
# for ``can`` and ``square`` on 2026-05-19). The portable subset
# is ``[:7]`` (object pos + quat). The live env exposes
# ``<Object>_pos`` and ``<Object>_quat`` keys directly under the
# robosuite 1.5 obs dict for both tasks, so no runtime remap is
# required.
OBS_KEYS_SQUARE: tuple[str, ...] = (
    "SquareNut_pos",
    "SquareNut_quat",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "robot0_joint_pos",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
)

_DATASET_OBJECT_SLICES_SQUARE: dict[str, tuple[str, slice]] = {
    "SquareNut_pos":  ("object", slice(0, 3)),
    "SquareNut_quat": ("object", slice(3, 7)),
}


# ---------------------------------------------------------------------------
# Lift (single-object pick-and-lift, simplest robomimic task).
#
# Added 2026-05-20 as part of the Slate G cross-task expansion.
# Runtime obs keys (probed against robosuite >= 1.5):  ``cube_pos``,
# ``cube_quat``.  Dataset ``object`` blob layout is
# ``[cube_pos(3), cube_quat(4), gripper_to_cube_pos(3)]`` = 10 dims.
# Single Panda, action_dim=7, H_demo ~ 50-80 (PH).  H=120 is generous.
OBS_KEYS_LIFT: tuple[str, ...] = (
    "cube_pos",
    "cube_quat",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "robot0_joint_pos",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
)

_DATASET_OBJECT_SLICES_LIFT: dict[str, tuple[str, slice]] = {
    "cube_pos":  ("object", slice(0, 3)),
    "cube_quat": ("object", slice(3, 7)),
}


# ---------------------------------------------------------------------------
# ToolHang (precision insertion, two named objects: frame + tool).
#
# Added 2026-05-20.  Runtime keys probed:  ``frame_pos``,
# ``frame_quat``, ``tool_pos``, ``tool_quat``.  Dataset ``object``
# blob (44 dims) layout recovered by per-frame matching against a
# runtime reset (see slate_g preamble notes):
#
#   [0:3]   base_pos      (table edge; constant, not useful)
#   [3:7]   base_quat
#   [7:10]  base_to_eef_pos
#   [10:14] base_to_eef_quat
#   [14:17] frame_pos
#   [17:21] frame_quat
#   [21:24] frame_to_eef_pos
#   [24:28] frame_to_eef_quat
#   [28:31] tool_pos
#   [31:35] tool_quat
#   [35:38] tool_to_eef_pos
#   [38:42] tool_to_eef_quat
#   [42]    frame_is_assembled  (binary)
#   [43]    tool_on_frame       (binary)
#
# Caveat: PH ToolHang demo horizons average ~600 steps; the current
# H=120 planner horizon is too short to complete the task.  The obs
# map below is correct for retrieval, but a meaningful Slate G cell
# on tool_hang needs H>=300-400.  Documented in slate_g BRIEF.
OBS_KEYS_TOOL_HANG: tuple[str, ...] = (
    "frame_pos",
    "frame_quat",
    "tool_pos",
    "tool_quat",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "robot0_joint_pos",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
)

_DATASET_OBJECT_SLICES_TOOL_HANG: dict[str, tuple[str, slice]] = {
    "frame_pos":  ("object", slice(14, 17)),
    "frame_quat": ("object", slice(17, 21)),
    "tool_pos":   ("object", slice(28, 31)),
    "tool_quat":  ("object", slice(31, 35)),
}


# ---------------------------------------------------------------------------
# Transport (dual-arm: payload + lid + trash + two bins).
#
# Added 2026-05-20 but flagged EXPERIMENTAL: action_dim=14 (two
# Pandas concatenated) and obs vector includes both robot0_* and
# robot1_* proprio sub-vectors.  ``run_can.py`` already pulls
# action_dim dynamically from the env action_spec, so the planner
# is action-agnostic.  But two caveats remain before transport can
# run end-to-end:
#   (i)  PH transport demos have H ~ 700; our default H=120 will
#        not complete the task.
#   (ii) The runtime obs vocabulary doubles in size compared with
#        single-arm tasks (extra ~30 dims).
#
# Dataset ``object`` blob (41 dims) layout, recovered as for ToolHang:
#
#   [0:3]   payload_pos
#   [3:7]   payload_quat
#   [7:10]  trash_pos
#   [10:14] trash_quat
#   [14:17] lid_handle_pos
#   [17:21] lid_handle_quat
#   [21:24] target_bin_pos
#   [24:27] trash_bin_pos
#   [27:..] gripper{0,1}_to_{payload,lid_handle,trash} (relative)
#   [..]    payload_in_target_bin, trash_in_trash_bin  (binary)
OBS_KEYS_TRANSPORT: tuple[str, ...] = (
    "payload_pos",
    "payload_quat",
    "trash_pos",
    "trash_quat",
    "lid_handle_pos",
    "lid_handle_quat",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "robot0_joint_pos",
    "robot0_joint_vel",
    "robot1_eef_pos",
    "robot1_eef_quat",
    "robot1_gripper_qpos",
    "robot1_gripper_qvel",
    "robot1_joint_pos",
    "robot1_joint_vel",
)

_DATASET_OBJECT_SLICES_TRANSPORT: dict[str, tuple[str, slice]] = {
    "payload_pos":     ("object", slice(0, 3)),
    "payload_quat":    ("object", slice(3, 7)),
    "trash_pos":       ("object", slice(7, 10)),
    "trash_quat":      ("object", slice(10, 14)),
    "lid_handle_pos":  ("object", slice(14, 17)),
    "lid_handle_quat": ("object", slice(17, 21)),
}


# ---------------------------------------------------------------------------
# DP-paper-faithful Can obs schema (Phase-B, added 2026-05-29).
#
# Background: the legacy ``OBS_KEYS`` (46-D) was chosen for the BC
# baseline in 2026-05-19, and deliberately slices off the relative-pose
# tail of ``obs/object`` for cross-version safety (robosuite 1.4 -> 1.5
# changed the reference frame of ``Can_to_eef_*``).  The price is that
# the policy is forced to learn the can-to-hand relative pose from
# 28 dims of joint redundancy + 7 dims of absolute eef pose.
#
# Diffusion Policy's published Can/ph config uses the full obs/object
# blob (14-D, ``[Can_pos, Can_quat, Can_to_eef_pos, Can_to_eef_quat]``)
# plus eef_pos (3) + eef_quat (4) + gripper_qpos (2) = 23 dims.  No
# joint state.  This is the structural hypothesis for the 25 %
# standalone ceiling we observe on Phase-3 ep0399 / ep1999 -- see
# ``results/overnight_2026_05_24/BRIEF.md`` § "Open follow-ups" and
# the 2026-05-29 obs-schema audit.
#
# Cross-version safety: rather than trust the live env's
# ``Can_to_robot0_eef_*`` keys (which carry a 1.5 reference-frame
# convention that differs from the HDF5 1.4 recording), we recompute
# the relative-pose dims from ``Can_pos`` and ``robot0_eef_pos`` /
# ``robot0_eef_quat`` at BOTH training and inference time via
# ``compute_synthetic_obs_keys``.  Same function, same inputs -> same
# outputs.  Bit-equivalent train and runtime.
OBS_KEYS_CAN_DP: tuple[str, ...] = (
    "Can_pos",                       # 3, slice obs/object[:, 0:3]
    "Can_quat",                      # 4, slice obs/object[:, 3:7]
    "robot0_eef_pos",                # 3, direct
    "robot0_eef_quat",               # 4, direct
    "Can_to_eef_pos_synth",          # 3, SYNTHETIC = eef_pos - Can_pos (world frame)
    "Can_to_eef_quat_synth",         # 4, SYNTHETIC = q_can_conj * q_eef (Hamilton)
    "robot0_gripper_qpos",           # 2, direct
)  # total = 23

_DATASET_OBJECT_SLICES_CAN_DP: dict[str, tuple[str, slice]] = {
    "Can_pos":  ("object", slice(0, 3)),
    "Can_quat": ("object", slice(3, 7)),
    # NOTE: deliberately NOT slicing the obs/object[7:14] tail.  The
    # ``Can_to_eef_*_synth`` keys are recomputed from Can_pos +
    # robot0_eef_* below to guarantee train/runtime parity.
}

# Map synthetic key -> tuple of (already-loaded) keys it depends on.
# Resolved by ``compute_synthetic_obs_keys`` after the base keys are
# loaded.  Loaders MUST iterate ``OBS_KEYS_<TASK>`` in dep order, i.e.
# all entries appearing in this map must come AFTER their deps in the
# task's OBS_KEYS tuple.
_SYNTHETIC_KEYS_CAN_DP: dict[str, tuple[str, ...]] = {
    "Can_to_eef_pos_synth":  ("Can_pos", "robot0_eef_pos"),
    "Can_to_eef_quat_synth": ("Can_quat", "robot0_eef_quat"),
}

OBS_KEYS_SQUARE_DP: tuple[str, ...] = (
    "SquareNut_pos",
    "SquareNut_quat",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "SquareNut_to_eef_pos_synth",
    "SquareNut_to_eef_quat_synth",
    "robot0_gripper_qpos",
)

_DATASET_OBJECT_SLICES_SQUARE_DP: dict[str, tuple[str, slice]] = {
    "SquareNut_pos":  ("object", slice(0, 3)),
    "SquareNut_quat": ("object", slice(3, 7)),
}

_SYNTHETIC_KEYS_SQUARE_DP: dict[str, tuple[str, ...]] = {
    "SquareNut_to_eef_pos_synth":  ("SquareNut_pos", "robot0_eef_pos"),
    "SquareNut_to_eef_quat_synth": ("SquareNut_quat", "robot0_eef_quat"),
}


def _quat_conj(q: np.ndarray) -> np.ndarray:
    """Hamilton-convention quaternion conjugate.

    ``q = (x, y, z, w)`` -> ``(-x, -y, -z, w)``.  Matches robosuite's
    ``[x, y, z, w]`` storage order (verified against the live env's
    ``robot0_eef_quat`` and HDF5 obs/object[3:7] which both have
    ``w`` as the last component).
    """
    q = np.asarray(q, dtype=np.float32)
    out = q.copy()
    out[..., :3] *= -1.0
    return out


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton-convention quaternion multiplication ``q1 * q2``.

    Both operands in ``[x, y, z, w]`` order.  Vectorised over leading
    dims; the trailing dim must be 4.
    """
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    ).astype(np.float32)


def compute_synthetic_obs_keys(
    obs_dict: dict[str, np.ndarray],
    synth_keys: dict[str, tuple[str, ...]],
) -> dict[str, np.ndarray]:
    """Populate the synthetic keys in ``obs_dict`` in-place and return it.

    Synthetic-key formulas:

    * ``Can_to_eef_pos_synth = robot0_eef_pos - Can_pos`` (world frame).
      Independent of any robosuite-version reference-frame change.
    * ``Can_to_eef_quat_synth = quat_conj(Can_quat) * robot0_eef_quat``
      (Hamilton convention, robosuite ``[x, y, z, w]`` order).  Encodes
      the orientation of the eef in the can's body frame.

    Both formulas are pure functions of inputs that are bit-identical
    between the HDF5 recording (sliced from ``obs/object[:, :7]``) and
    the live env (read directly from ``Can_pos`` / ``robot0_eef_*``).
    Therefore the synthetic features are bit-identical at train and
    runtime as well, sidestepping the robosuite-1.4 -> 1.5 reference-
    frame divergence on the dataset's ``obs/object[:, 7:14]`` tail.
    """
    for k, deps in synth_keys.items():
        for d in deps:
            if d not in obs_dict:
                raise KeyError(
                    f"synthetic key {k!r} depends on {d!r}, which is "
                    f"not present in obs_dict (have: "
                    f"{sorted(obs_dict.keys())}).  Ensure deps are "
                    f"loaded BEFORE synthetic keys in OBS_KEYS."
                )
        if k == "Can_to_eef_pos_synth":
            obs_dict[k] = (
                np.asarray(obs_dict["robot0_eef_pos"], dtype=np.float32)
                - np.asarray(obs_dict["Can_pos"], dtype=np.float32)
            ).astype(np.float32)
        elif k == "Can_to_eef_quat_synth":
            q_can_conj = _quat_conj(obs_dict["Can_quat"])
            q_eef = np.asarray(obs_dict["robot0_eef_quat"], dtype=np.float32)
            obs_dict[k] = _quat_mul(q_can_conj, q_eef)
        elif k == "SquareNut_to_eef_pos_synth":
            obs_dict[k] = (
                np.asarray(obs_dict["robot0_eef_pos"], dtype=np.float32)
                - np.asarray(obs_dict["SquareNut_pos"], dtype=np.float32)
            ).astype(np.float32)
        elif k == "SquareNut_to_eef_quat_synth":
            q_nut_conj = _quat_conj(obs_dict["SquareNut_quat"])
            q_eef = np.asarray(obs_dict["robot0_eef_quat"], dtype=np.float32)
            obs_dict[k] = _quat_mul(q_nut_conj, q_eef)
        else:
            raise ValueError(
                f"compute_synthetic_obs_keys: no formula registered for "
                f"key {k!r}.  Add a branch above or remove {k!r} from "
                f"the task's synthetic-keys map."
            )
    return obs_dict


def get_synthetic_keys(task: str = "can") -> dict[str, tuple[str, ...]]:
    """Per-task ``synthetic_key -> (dep_key, ...)`` map.

    Returns an empty dict for tasks that don't use synthetic keys
    (every existing non-DP-schema task), so existing loaders
    can call this unconditionally.
    """
    return _SYNTHETIC_KEYS_BY_TASK.get(task, {})


_OBS_KEYS_BY_TASK: dict[str, tuple[str, ...]] = {
    "can": OBS_KEYS,
    "can_dp": OBS_KEYS_CAN_DP,
    "square": OBS_KEYS_SQUARE,
    "square_dp": OBS_KEYS_SQUARE_DP,
    "lift": OBS_KEYS_LIFT,
    "tool_hang": OBS_KEYS_TOOL_HANG,
    "transport": OBS_KEYS_TRANSPORT,
}

_OBJECT_SLICES_BY_TASK: dict[str, dict[str, tuple[str, slice]]] = {
    "can": _DATASET_OBJECT_SLICES,
    "can_dp": _DATASET_OBJECT_SLICES_CAN_DP,
    "square": _DATASET_OBJECT_SLICES_SQUARE,
    "square_dp": _DATASET_OBJECT_SLICES_SQUARE_DP,
    "lift": _DATASET_OBJECT_SLICES_LIFT,
    "tool_hang": _DATASET_OBJECT_SLICES_TOOL_HANG,
    "transport": _DATASET_OBJECT_SLICES_TRANSPORT,
}

# Per-task synthetic-key registry.  Tasks not present here use an
# empty dict (no synthetic keys), so existing loaders that call
# ``get_synthetic_keys`` unconditionally see a no-op for the legacy
# 'can'/'square'/'lift'/'tool_hang'/'transport' schemas.
_SYNTHETIC_KEYS_BY_TASK: dict[str, dict[str, tuple[str, ...]]] = {
    "can_dp": _SYNTHETIC_KEYS_CAN_DP,
    "square_dp": _SYNTHETIC_KEYS_SQUARE_DP,
}


def get_obs_keys(task: str = "can") -> tuple[str, ...]:
    """Return the fixed obs-vector order for ``task``.

    ``task`` matches the directory name under ``robomimic/datasets/``
    (``can``, ``square``, ...).  Defaults to ``"can"`` so existing
    callers that don't pass a task still see Can's keys.
    """
    if task not in _OBS_KEYS_BY_TASK:
        raise ValueError(
            f"Unknown task {task!r}; supported: "
            f"{sorted(_OBS_KEYS_BY_TASK)}"
        )
    return _OBS_KEYS_BY_TASK[task]


def get_object_slices(
    task: str = "can",
) -> dict[str, tuple[str, slice]]:
    """Per-task ``OBS_KEY -> (hdf5_dataset_key, slice)`` map.

    Mirrors ``get_obs_keys`` for the dataset-side object slicing
    convention.  Used by the zero-train retrieval bank loader to
    materialise the same ``obj_pos`` / ``obj_quat`` synthetic keys
    across tasks.
    """
    if task not in _OBJECT_SLICES_BY_TASK:
        raise ValueError(
            f"Unknown task {task!r}; supported: "
            f"{sorted(_OBJECT_SLICES_BY_TASK)}"
        )
    return _OBJECT_SLICES_BY_TASK[task]


@dataclass(frozen=True)
class ObsStats:
    """Per-dimension mean/std for normalising the flat obs vector."""

    mean: np.ndarray  # shape (D,), float32
    std: np.ndarray   # shape (D,), float32, clipped to >= eps

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def _concat_obs_dict(obs: dict, keys: tuple[str, ...]) -> np.ndarray:
    """Concatenate ``obs[k]`` for k in ``keys`` along the last axis.

    Each value can be ``(D_k,)`` (single-step) or ``(H, D_k)``
    (per-demo). The concatenation is along axis=-1.
    """
    parts = []
    for k in keys:
        if k not in obs:
            raise KeyError(
                f"obs missing required key {k!r}; available keys: "
                f"{sorted(obs.keys())}"
            )
        v = np.asarray(obs[k], dtype=np.float32)
        parts.append(v)
    return np.concatenate(parts, axis=-1)


def load_demos(
    dataset_path: Path,
    obs_keys: tuple[str, ...] = OBS_KEYS,
    n_demos: int | None = None,
    object_slices: dict[str, tuple[str, slice]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read ``n_demos`` demos and return ``(obs, actions, demo_ids)``.

    ``obs`` has shape ``(N, D)``, ``actions`` has shape ``(N, 7)``, and
    ``demo_ids`` is the list of HDF5 group names actually consumed.
    Demos are taken in numeric ``demo_<i>`` order (not the HDF5 group
    iteration order, which varies).

    ``object_slices`` maps each named-object key (e.g. ``"cube_pos"``)
    to the ``(hdf5_obs_key, slice)`` pair that extracts it from the
    HDF5 ``object`` blob.  Defaults to ``_DATASET_OBJECT_SLICES`` (Can)
    for backward compatibility; pass an explicit dict for cross-task
    loading.  The convenience entry point ``build_tensor_dataset``
    resolves this via ``get_object_slices(task)``.
    """
    if object_slices is None:
        object_slices = _DATASET_OBJECT_SLICES
    obs_chunks: list[np.ndarray] = []
    act_chunks: list[np.ndarray] = []
    used_demos: list[str] = []
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
                else:
                    obs_dict[k] = np.asarray(demo["obs"][k][()])
            obs_flat = _concat_obs_dict(obs_dict, obs_keys)  # (H, D)
            actions = np.asarray(demo["actions"][()], dtype=np.float32)
            obs_chunks.append(obs_flat)
            act_chunks.append(actions)
            used_demos.append(demo_id)
    obs = np.concatenate(obs_chunks, axis=0)
    actions = np.concatenate(act_chunks, axis=0)
    return obs, actions, used_demos


def compute_obs_stats(obs: np.ndarray, eps: float = 1e-6) -> ObsStats:
    """Per-dimension (mean, std) over the training obs matrix."""
    mean = obs.mean(axis=0).astype(np.float32)
    std = obs.std(axis=0).astype(np.float32)
    std = np.maximum(std, eps)
    return ObsStats(mean=mean, std=std)


def build_tensor_dataset(
    dataset_path: Path,
    obs_keys: tuple[str, ...] = OBS_KEYS,
    n_demos: int | None = None,
    object_slices: dict[str, tuple[str, slice]] | None = None,
) -> tuple[TensorDataset, ObsStats, list[str]]:
    """End-to-end loader: returns a normalised ``TensorDataset``.

    ``object_slices`` lets the caller pin the named-object slice map
    explicitly (e.g. ``get_object_slices("lift")``); defaults to Can.
    """
    obs, actions, demo_ids = load_demos(
        dataset_path, obs_keys, n_demos, object_slices=object_slices,
    )
    stats = compute_obs_stats(obs)
    obs_norm = stats.normalize(obs).astype(np.float32)
    return (
        TensorDataset(
            torch.from_numpy(obs_norm),
            torch.from_numpy(actions.astype(np.float32)),
        ),
        stats,
        demo_ids,
    )
