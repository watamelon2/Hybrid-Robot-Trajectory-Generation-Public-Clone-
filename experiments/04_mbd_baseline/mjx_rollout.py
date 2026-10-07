"""MJX-batched rollouts for MBD on PickPlaceCan.

Couples three pieces:

  * `jax_osc.delta_action_to_ctrl`    -- robosuite-compatible OSC.
  * `mjx.step`                        -- physics integration.
  * `_pickplace_step_cost`            -- a smooth, MBD-friendly cost.

The exported API mirrors the CPU rollout already used by exp 04 so that
`run_can.py` can be flipped to MJX with `--mjx` (Gate 3) without
rewriting MBD itself:

  * `build_rollout_context(xml_path, ...)` returns a `RolloutContext`
    bundling the static model, JAX-typed indices, the JIT-compiled
    single-rollout function, and the JIT-compiled batched version.
  * `single_rollout(context, qpos_init, qvel_init, action_seq)` -> cost.
  * `batched_rollout(context, qpos_init, qvel_init, action_seqs)` -> costs.

ACTION CONVENTION (matches robosuite OSC_POSE composite controller):
  `action[i, j, 0:3]` = Cartesian position delta in [-1, +1], scaled
                        through OSC by 0.05 m / dim
  `action[i, j, 3:6]` = axis-angle orientation delta in [-1, +1],
                        scaled through OSC by 0.5 rad / dim
  `action[i, j, 6]`   = gripper command in [-1, +1]; +1 closes.
where `i` indexes the batch (M rollouts) and `j` the policy step (H steps).

EACH policy step performs `n_substeps` calls to `mjx.step` with the
same OSC-derived `ctrl`, then evaluates an incremental cost. Total
trajectory cost is the sum over the H policy steps.

COST (lower = better; matches the MBD sign convention in
`docs/MuJoCo.md` Section 5):

  c_t = w_xy * ||can_xy - goal_xy||^2                # bin alignment
      + w_z  * relu(goal_z - can_z) ** 2             # lift toward goal z
      + w_ee * ||ee_pos - can_pos||^2                # reaching shaping
      + w_act * ||action||^2                         # control regulariser

This is intentionally smoother than robosuite's piecewise-staged dense
reward; the rank correlation with the robosuite reward is the Gate-2
acceptance criterion (target Spearman > 0.7 on a sample of trajectories).
We can swap in a faithful piecewise reward later if rank correlation
fails -- this version is the simplest thing that fits the MBD smoothness
assumption.
"""

from __future__ import annotations

import functools
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

from jax_osc import (
    OSCConfig,
    OSCArmState,
    compute_osc_torques,
    delta_action_to_ctrl,
    osc_state_from_mjx,
    update_goal_from_delta,
)


# ---------------------------------------------------------------------------
# Default knobs. Override via `build_rollout_context(...)` kwargs.
# ---------------------------------------------------------------------------

DEFAULT_N_SUBSTEPS = 5      # mjx.step calls per policy step
DEFAULT_W_XY = 1.0
DEFAULT_W_Z = 2.0
DEFAULT_W_EE = 0.2
DEFAULT_W_ACT = 1e-3


# ---------------------------------------------------------------------------
# Context: precomputed static info, JITed rollout functions, OSC config.
# ---------------------------------------------------------------------------

class CostWeights(NamedTuple):
    w_xy: float = DEFAULT_W_XY
    w_z: float = DEFAULT_W_Z
    w_ee: float = DEFAULT_W_EE
    w_act: float = DEFAULT_W_ACT


class RolloutContext(NamedTuple):
    """All the JAX-side state needed to roll trajectories out.

    Attributes
    ----------
    mj_model        : `mujoco.MjModel` (CPU). Kept around for inspection
                      / rendering / sanity checks. Not used in the hot path.
    mjx_model       : `mjx.Model`. Pushed onto the device.
    ee_site_id      : int site id of the EE.
    can_body_id     : int body id of the can.
    goal_site_id    : int site id of the bin-target goal marker.
    arm_dof_ids     : (7,) int32 jnp array of arm qvel/qpos indices.
    initial_qpos_arm: (7,) jnp.float32 home joint config (for the
                      nullspace term).
    goal_pos        : (3,) jnp.float32 world-frame goal position.
    n_substeps      : int -- mjx.step calls per policy step.
    cost_w          : CostWeights.
    osc_cfg         : OSCConfig.
    single_rollout  : JIT-compiled (qpos_init, qvel_init, action_seq) ->
                      (cost_total, qpos_final).
    batched_rollout : JIT-compiled vmap'd version over leading batch dim.
    """

    mj_model: mujoco.MjModel
    mjx_model: mjx.Model
    ee_site_id: int
    can_body_id: int
    goal_site_id: int
    arm_dof_ids: jax.Array
    initial_qpos_arm: jax.Array
    goal_pos: jax.Array
    n_substeps: int
    cost_w: CostWeights
    osc_cfg: OSCConfig
    single_rollout: Callable
    batched_rollout: Callable


# ---------------------------------------------------------------------------
# Cost function.
# ---------------------------------------------------------------------------

def _pickplace_step_cost(
    can_pos: jax.Array,
    ee_pos: jax.Array,
    goal_pos: jax.Array,
    action: jax.Array,
    cost_w: CostWeights,
) -> jax.Array:
    """One-step PickPlaceCan cost (lower is better)."""
    xy_err = can_pos[:2] - goal_pos[:2]
    z_gap = jax.nn.relu(goal_pos[2] - can_pos[2])
    ee_to_can = ee_pos - can_pos
    return (
        cost_w.w_xy * jnp.sum(xy_err * xy_err)
        + cost_w.w_z * z_gap * z_gap
        + cost_w.w_ee * jnp.sum(ee_to_can * ee_to_can)
        + cost_w.w_act * jnp.sum(action * action)
    )


# ---------------------------------------------------------------------------
# Inner sim loop: one policy step (an action held for n_substeps).
# ---------------------------------------------------------------------------

def _policy_step(
    data: mjx.Data,
    action: jax.Array,
    *,
    mjx_model: mjx.Model,
    ee_site_id: int,
    arm_dof_ids: jax.Array,
    initial_qpos_arm: jax.Array,
    osc_cfg: OSCConfig,
    n_substeps: int,
) -> mjx.Data:
    """Advance `data` by `n_substeps` mjx.steps under a fixed `action`.

    The OSC controller is recomputed AT EVERY substep (matching robosuite's
    `lite_physics=True` behaviour). This is critical: the EE pose changes
    inside the policy step under gravity / inertia, so the torque required
    to track the goal also changes. Recomputing the OSC ctrl each substep
    is the textbook impedance controller behaviour.
    """

    def _substep(d, _):
        # PRECONDITION: `d` has been forward()-ed (either before the
        # outer scan or by the previous substep's trailing forward).
        # That guarantees `site_xpos`, `qM`, `qfrc_bias` are
        # consistent with `qpos`, which is what OSC needs.
        state = osc_state_from_mjx(mjx_model, d, ee_site_id, arm_dof_ids)
        ctrl = delta_action_to_ctrl(action, state, initial_qpos_arm, osc_cfg)
        d = d.replace(ctrl=ctrl)
        d = mjx.step(mjx_model, d)
        # `mjx.step` integrates qpos but DOES NOT refresh kinematics
        # afterwards (verified empirically: site_xpos drifts ~1e-5 m
        # between step and a follow-up forward). Re-run forward so
        # the next substep starts from a kinematically consistent
        # state. Cost: ~one extra forward per substep -- acceptable
        # for Gate-2 work, can be tightened later by splitting step
        # into forward + euler integrate to avoid the redundant
        # forward inside mjx.step.
        d = mjx.forward(mjx_model, d)
        return d, None

    data, _ = jax.lax.scan(_substep, data, None, length=n_substeps)
    return data


# ---------------------------------------------------------------------------
# Single rollout: scan over H policy steps, accumulate cost.
# ---------------------------------------------------------------------------

def _make_single_rollout(
    mjx_model: mjx.Model,
    *,
    ee_site_id: int,
    can_body_id: int,
    goal_pos: jax.Array,
    arm_dof_ids: jax.Array,
    initial_qpos_arm: jax.Array,
    n_substeps: int,
    cost_w: CostWeights,
    osc_cfg: OSCConfig,
) -> Callable:
    """Build a JIT-compiled single-rollout function bound to the static config."""

    def _rollout(qpos_init: jax.Array, qvel_init: jax.Array, action_seq: jax.Array):
        # Build a fresh data and seed it with the requested init state.
        data = mjx.make_data(mjx_model)
        data = data.replace(qpos=qpos_init, qvel=qvel_init)
        # Forward pass so site_xpos / qM / qfrc_bias are populated
        # before the first OSC call.
        data = mjx.forward(mjx_model, data)

        def _step(carry, action):
            d, cum_cost = carry
            d_next = _policy_step(
                d,
                action,
                mjx_model=mjx_model,
                ee_site_id=ee_site_id,
                arm_dof_ids=arm_dof_ids,
                initial_qpos_arm=initial_qpos_arm,
                osc_cfg=osc_cfg,
                n_substeps=n_substeps,
            )
            can_pos = d_next.xpos[can_body_id]
            ee_pos = d_next.site_xpos[ee_site_id]
            c = _pickplace_step_cost(can_pos, ee_pos, goal_pos, action, cost_w)
            return (d_next, cum_cost + c), c

        (final_data, total_cost), per_step_cost = jax.lax.scan(
            _step, (data, jnp.zeros((), dtype=jnp.float32)), action_seq
        )
        return total_cost, final_data.qpos, per_step_cost

    return jax.jit(_rollout)


# ---------------------------------------------------------------------------
# Public builder.
# ---------------------------------------------------------------------------

def build_rollout_context(
    xml_path: str,
    *,
    n_substeps: int = DEFAULT_N_SUBSTEPS,
    cost_w: CostWeights = CostWeights(),
    osc_cfg: OSCConfig = OSCConfig(),
    ee_site_name: str = "gripper",
    can_body_name: str = "can",
    goal_site_name: str = "goal",
    arm_dof_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6),
    keyframe_name: str = "home",
) -> RolloutContext:
    """Compile and return a ready-to-use RolloutContext.

    Parameters
    ----------
    xml_path : path to the scene MJCF (e.g. `scenes/simplified_can.xml`).
    n_substeps : `mjx.step` calls per policy step. For robosuite parity,
        `policy_freq=20, sim_freq=500` -> 25. We default to 5 (faster
        Gate-2 iteration); bump to 25 for the official parity number.
    cost_w / osc_cfg : tunable cost weights and OSC params.
    ee_site_name / can_body_name / goal_site_name : MJCF entity names.
    arm_dof_indices : qvel indices of the 7 arm joints (0..6 for our
        Franka, since the panda is the first body chain).
    keyframe_name : keyframe to pull the nullspace home posture from.
    """
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mjx_model = mjx.put_model(mj_model)

    ee_site_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name))
    can_body_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, can_body_name))
    goal_site_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, goal_site_name))
    key_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name))
    if min(ee_site_id, can_body_id, goal_site_id, key_id) < 0:
        raise RuntimeError(
            f"Scene `{xml_path}` is missing required entities: "
            f"ee_site={ee_site_id}, can_body={can_body_id}, "
            f"goal_site={goal_site_id}, key={key_id}"
        )

    # Resolve the goal site's world-frame position once (it never moves
    # since the goal site lives inside the static `bin_target` body).
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    mujoco.mj_forward(mj_model, mj_data)
    goal_pos_np = np.array(mj_data.site_xpos[goal_site_id], dtype=np.float32)
    initial_qpos_full = np.array(mj_data.qpos, dtype=np.float32)
    arm_dof_ids_np = np.asarray(arm_dof_indices, dtype=np.int32)

    arm_dof_ids = jnp.asarray(arm_dof_ids_np)
    goal_pos = jnp.asarray(goal_pos_np)
    initial_qpos_arm = jnp.asarray(initial_qpos_full[arm_dof_ids_np])

    single_rollout = _make_single_rollout(
        mjx_model,
        ee_site_id=ee_site_id,
        can_body_id=can_body_id,
        goal_pos=goal_pos,
        arm_dof_ids=arm_dof_ids,
        initial_qpos_arm=initial_qpos_arm,
        n_substeps=n_substeps,
        cost_w=cost_w,
        osc_cfg=osc_cfg,
    )

    # Batched version: vmap over leading dim of action_seqs. We hold the
    # initial state fixed across the batch (one start, M action plans) --
    # the typical MBD inner loop.
    @jax.jit
    def batched_rollout(qpos_init, qvel_init, action_seqs):
        # in_axes=(None, None, 0): broadcast qpos/qvel, batch over action.
        return jax.vmap(single_rollout, in_axes=(None, None, 0))(
            qpos_init, qvel_init, action_seqs
        )

    return RolloutContext(
        mj_model=mj_model,
        mjx_model=mjx_model,
        ee_site_id=ee_site_id,
        can_body_id=can_body_id,
        goal_site_id=goal_site_id,
        arm_dof_ids=arm_dof_ids,
        initial_qpos_arm=initial_qpos_arm,
        goal_pos=goal_pos,
        n_substeps=n_substeps,
        cost_w=cost_w,
        osc_cfg=osc_cfg,
        single_rollout=single_rollout,
        batched_rollout=batched_rollout,
    )


# ---------------------------------------------------------------------------
# Convenience: read the home keyframe's qpos as an initial-state seed.
# ---------------------------------------------------------------------------

def home_state(ctx: RolloutContext) -> tuple[jax.Array, jax.Array]:
    """Return (qpos_init, qvel_init) at the home keyframe."""
    key_id = int(
        mujoco.mj_name2id(ctx.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
    )
    qpos = jnp.asarray(ctx.mj_model.key_qpos[key_id], dtype=jnp.float32)
    # No key_qvel field in older MuJoCo; just start at rest.
    qvel = jnp.zeros((ctx.mj_model.nv,), dtype=jnp.float32)
    return qpos, qvel
