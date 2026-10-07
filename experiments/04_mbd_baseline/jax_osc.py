"""JAX port of robosuite's OSC_POSE (operational space control) controller.

This module reimplements the math performed by
`robosuite.controllers.parts.arm.osc.OperationalSpaceController` so we can
run it inside a `jax.jit`-able rollout that pushes torques into MJX. We
match robosuite 1.5.x's defaults for the Franka Panda Pick-Place-Can
task verbatim:

  * `input_max =  1`, `input_min = -1`
  * `output_max = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)`
  * `kp = 150`  (uniform across the 6 task-space dims)
  * `damping_ratio = 1` -> `kd = 2 * sqrt(kp) = ~24.49`
  * `joint_kp_null = 10` -> `joint_kv_null = 2 * sqrt(10) ~= 6.32`
  * `uncouple_pos_ori = True`
  * `input_type = "delta"`,  `goal_update_mode = "achieved"`
  * `input_ref_frame = "base"` -- for our Franka mounted at the world
    origin with identity orientation, this collapses to the world frame
    and the `origin_pos / origin_ori` transforms become identities.
    `compute_osc_torques` therefore does not take an origin pose; if you
    later mount the arm on a moving base, plumb a `base_pos / base_ori`
    pair through.

The algorithm (one OSC tick):

  1. Read EE state from MJX: `ee_pos = data.site_xpos[ee]`,
     `R_ee = data.site_xmat[ee].reshape(3,3)`. Arm Jacobian
     `J_pos, J_ori = (3,7)` via `mjx.jac` filtered to the 7 arm DOFs.
  2. Compute desired wrench from pose error:
       F_des  = kp * (goal_pos - ee_pos) - kd * (J_pos @ qdot_arm)
       T_des  = kp * ori_err            - kd * (J_ori @ qdot_arm)
     where `ori_err = 0.5 * sum_i cross(R_ee[:,i], R_goal[:,i])` is
     robosuite's three-column orientation error formula.
  3. Operational-space inertia:
       Lambda_pos = pinv(J_pos @ M_inv @ J_pos.T)        (3x3)
       Lambda_ori = pinv(J_ori @ M_inv @ J_ori.T)        (3x3)
       Lambda_full = pinv(J_full @ M_inv @ J_full.T)     (6x6)
     and nullspace projector
       Jbar  = M_inv @ J_full.T @ Lambda_full            (7x6)
       N     = I - Jbar @ J_full                         (7x7)
  4. Joint torques:
       tau_task = J_pos.T @ (Lambda_pos @ F_des)
                + J_ori.T @ (Lambda_ori @ T_des)
       tau_grav = qfrc_bias[arm]                         (gravity comp)
       tau_null = N.T @ (M_arm @ (joint_kp_null * (q_init - q)
                                  - joint_kv_null * qdot))
       tau      = tau_task + tau_grav + tau_null         (clamped to
                                                          forcerange)

The set-goal half of robosuite's OSC (parse delta action -> goal_pos,
goal_ori, with goal_update_mode="achieved") is implemented in
`update_goal_from_delta`. Call it once per policy step (20 Hz in
robosuite's stack) and then call `compute_osc_torques` once per sim
substep (500 Hz) inside the inner mjx.step loop.

PARITY NOTES vs robosuite:
  * robosuite reads EE velocity via mj_data.cvel which is in
    world-frame angular+linear convention. We instead reconstruct EE
    velocity as `J @ qdot_arm`, which is the same quantity for fixed
    bases (and what robosuite *intends* to compute -- see the
    `vel_pos_error = -(self.ref_pos_vel - base_pos_vel)` line where
    `base_pos_vel` is 0 for a welded base). Matches to numerical
    precision.
  * robosuite's `opspace_matrices` uses `np.linalg.pinv` on the 3x3 /
    6x6 inertia blocks. We use the same `jnp.linalg.pinv`. For
    nondegenerate arm configurations these are full rank and pinv = inv.
  * We deliberately do NOT scale the gripper dimension through OSC;
    `delta_action_to_ctrl` maps action[6] in [-1, +1] to ctrl[7] in
    [0, 0.04] m (finger separation, robosuite convention: +1 = closed).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Config (match robosuite 1.5.x OSC_POSE defaults for Franka exactly).
# ---------------------------------------------------------------------------

class OSCConfig(NamedTuple):
    """Tunable OSC parameters. Defaults match robosuite OSC_POSE defaults.

    Attributes
    ----------
    kp_pos : positional gain (uniform across x, y, z). robosuite: 150.
    kp_ori : orientation gain (uniform across ax, ay, az). robosuite: 150.
    damping_ratio : multiplier on critical damping for the task-space PD.
      robosuite: 1.0 (critically damped) -> `kd = 2 * sqrt(kp)`.
    null_kp : proportional gain on the nullspace "stay close to
      initial joint config" objective. robosuite: 10.
    output_max_pos : per-dim scaling for action[0:3] (delta xyz, m).
      robosuite: 0.05.
    output_max_ori : per-dim scaling for action[3:6] (delta axis-angle, rad).
      robosuite: 0.5.
    arm_force_limit : torque clamp per arm joint (Nm). The Franka spec sheet
      gives 87 for joints 1-4 and 12 for joints 5-7; we use a single
      conservative bound matching the largest joint-1..4 cap, but the MJCF
      `<motor ctrlrange>` also clamps further per-joint. If you tighten this
      to 12.0 the joints 1-4 will lose some authority but joints 5-7 will
      see less saturation transient.
    gripper_open_pos : finger_joint1 position at fully open (m).
      Menagerie panda: 0.04.
    """

    kp_pos: float = 150.0
    kp_ori: float = 150.0
    damping_ratio: float = 1.0
    null_kp: float = 10.0
    output_max_pos: float = 0.05
    output_max_ori: float = 0.5
    arm_force_limit: float = 87.0
    gripper_open_pos: float = 0.04


# ---------------------------------------------------------------------------
# Small math helpers. All JIT-friendly, no Python control flow on values.
# ---------------------------------------------------------------------------

def axis_angle_to_matrix(vec: jax.Array) -> jax.Array:
    """Rodrigues' formula. `vec` is a 3-d axis*angle (in radians).

    Returns a 3x3 rotation matrix. Numerically safe at theta=0: when the
    norm is tiny we fall back to `safe_theta = 1` so `axis = vec / 1`
    stays finite; sin(0) * K and (1 - cos(0)) * K^2 both vanish, so the
    result is exactly the identity matrix regardless.
    """
    theta = jnp.linalg.norm(vec)
    safe_theta = jnp.where(theta > 1e-12, theta, 1.0)
    axis = vec / safe_theta
    K = jnp.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return jnp.eye(3) + jnp.sin(theta) * K + (1.0 - jnp.cos(theta)) * (K @ K)


def orientation_error_matrices(R_desired: jax.Array, R_current: jax.Array) -> jax.Array:
    """robosuite's three-column orientation error.

    err = 0.5 * sum_{i in {0,1,2}} cross(R_current[:, i], R_desired[:, i])

    This is the linearised so(3) error in the world frame for small
    rotations; it matches what robosuite's `control_utils.orientation_error`
    computes and what its OSC controller feeds into the task-space PD.
    """
    return 0.5 * (
        jnp.cross(R_current[:, 0], R_desired[:, 0])
        + jnp.cross(R_current[:, 1], R_desired[:, 1])
        + jnp.cross(R_current[:, 2], R_desired[:, 2])
    )


# ---------------------------------------------------------------------------
# Goal update (robosuite's `set_goal` half, mode="achieved", frame="base"
# with identity base transform).
# ---------------------------------------------------------------------------

def update_goal_from_delta(
    action6: jax.Array,
    ee_pos: jax.Array,
    R_ee: jax.Array,
    cfg: OSCConfig = OSCConfig(),
) -> tuple[jax.Array, jax.Array]:
    """Map a 6-d delta action to (goal_pos, goal_ori).

    Parameters
    ----------
    action6 : (6,) array. Values are clipped to [-1, 1] per
        robosuite's `input_max / input_min = +/- 1`, then scaled by
        `output_max_pos` (xyz) and `output_max_ori` (axis-angle).
    ee_pos  : (3,) current EE world-frame position.
    R_ee    : (3, 3) current EE world-frame rotation matrix.
    cfg     : OSC config (uses `output_max_pos / output_max_ori`).

    Returns
    -------
    goal_pos : (3,) world-frame target position (achieved + delta).
    goal_ori : (3, 3) world-frame target rotation (R_delta @ R_current).

    Notes
    -----
    Implements the goal_update_mode="achieved" branch of robosuite's
    `set_goal`. The "desired" branch (chained from the previous goal
    rather than the achieved pose) is not used by robosuite's PickPlaceCan
    BC stack and not implemented here.
    """
    clipped = jnp.clip(action6, -1.0, 1.0)
    scale = jnp.array(
        [
            cfg.output_max_pos,
            cfg.output_max_pos,
            cfg.output_max_pos,
            cfg.output_max_ori,
            cfg.output_max_ori,
            cfg.output_max_ori,
        ]
    )
    scaled = clipped * scale
    goal_pos = ee_pos + scaled[0:3]
    R_delta = axis_angle_to_matrix(scaled[3:6])
    goal_ori = R_delta @ R_ee
    return goal_pos, goal_ori


# ---------------------------------------------------------------------------
# Main controller. One call -> 7 arm torques.
# ---------------------------------------------------------------------------

class OSCArmState(NamedTuple):
    """Per-tick MJX-derived state the OSC controller needs.

    All quantities are sliced/reshaped to the arm's 7 DOFs already, so
    `compute_osc_torques` itself stays free of indexing logic.

    Attributes
    ----------
    ee_pos  : (3,)   EE site world-frame position.
    R_ee    : (3,3) EE site world-frame rotation matrix.
    J_pos   : (3,7) site translation Jacobian, columns sliced to arm DOFs.
    J_ori   : (3,7) site rotation    Jacobian, columns sliced to arm DOFs.
    M_arm   : (7,7) arm-only block of the joint-space inertia matrix.
    qpos_arm : (7,) current arm joint positions.
    qvel_arm : (7,) current arm joint velocities.
    qfrc_bias_arm : (7,) coriolis + gravity at this state, on the arm DOFs.
    """

    ee_pos: jax.Array
    R_ee: jax.Array
    J_pos: jax.Array
    J_ori: jax.Array
    M_arm: jax.Array
    qpos_arm: jax.Array
    qvel_arm: jax.Array
    qfrc_bias_arm: jax.Array


def compute_osc_torques(
    state: OSCArmState,
    goal_pos: jax.Array,
    goal_ori: jax.Array,
    initial_qpos_arm: jax.Array,
    cfg: OSCConfig = OSCConfig(),
) -> jax.Array:
    """Compute the 7-D arm torque vector from the OSC algorithm.

    Mirrors robosuite's `OperationalSpaceController.run_controller`
    with `uncoupling=True`, `lite_physics=True`. See module docstring
    for the math.

    Parameters
    ----------
    state : populated OSCArmState (call `osc_state_from_mjx` to build).
    goal_pos : (3,) world-frame target position.
    goal_ori : (3,3) world-frame target rotation matrix.
    initial_qpos_arm : (7,) reference joint config for the nullspace
        posture objective. Typically the home keyframe.
    cfg : OSC config.

    Returns
    -------
    tau_arm : (7,) joint torques, clamped to +/- cfg.arm_force_limit.
    """
    kp_pos = cfg.kp_pos
    kp_ori = cfg.kp_ori
    kd_pos = 2.0 * jnp.sqrt(kp_pos) * cfg.damping_ratio
    kd_ori = 2.0 * jnp.sqrt(kp_ori) * cfg.damping_ratio
    null_kp = cfg.null_kp
    null_kv = 2.0 * jnp.sqrt(null_kp)

    # --- EE velocity in world frame, reconstructed via J @ qdot. For a
    # welded-to-world base this is exactly what robosuite reads from
    # mj_data.cvel. ---
    ee_lin_vel = state.J_pos @ state.qvel_arm
    ee_ang_vel = state.J_ori @ state.qvel_arm

    # --- Task-space PD: desired Cartesian wrench. ---
    pos_err = goal_pos - state.ee_pos
    ori_err = orientation_error_matrices(goal_ori, state.R_ee)
    F_des = kp_pos * pos_err - kd_pos * ee_lin_vel
    T_des = kp_ori * ori_err - kd_ori * ee_ang_vel

    # --- Operational-space inertia + nullspace projector. ---
    M_inv = jnp.linalg.inv(state.M_arm)
    J_full = jnp.concatenate([state.J_pos, state.J_ori], axis=0)  # (6, 7)

    lambda_pos = jnp.linalg.pinv(state.J_pos @ M_inv @ state.J_pos.T)
    lambda_ori = jnp.linalg.pinv(state.J_ori @ M_inv @ state.J_ori.T)
    lambda_full = jnp.linalg.pinv(J_full @ M_inv @ J_full.T)

    Jbar = M_inv @ J_full.T @ lambda_full  # (7, 6)
    N = jnp.eye(7) - Jbar @ J_full         # (7, 7) nullspace projector

    # --- Decoupled task torques (uncoupling=True branch). ---
    decoupled_wrench = jnp.concatenate(
        [lambda_pos @ F_des, lambda_ori @ T_des]
    )                                       # (6,)
    tau_task = J_full.T @ decoupled_wrench  # (7,)

    # --- Gravity / Coriolis compensation. qfrc_bias = C(q,qdot) + G(q). ---
    tau_grav = state.qfrc_bias_arm

    # --- Nullspace posture objective. ---
    pose_torques = state.M_arm @ (
        null_kp * (initial_qpos_arm - state.qpos_arm)
        - null_kv * state.qvel_arm
    )
    tau_null = N.T @ pose_torques

    tau = tau_task + tau_grav + tau_null
    return jnp.clip(tau, -cfg.arm_force_limit, cfg.arm_force_limit)


# ---------------------------------------------------------------------------
# Glue: extract arm-only state from a full MJX `Data` object.
# ---------------------------------------------------------------------------

def osc_state_from_mjx(
    mjx_model,
    mjx_data,
    ee_site_id: int,
    arm_dof_ids: jax.Array,
) -> OSCArmState:
    """Build an OSCArmState from a full MJX (model, data) at the current step.

    NOTE: `mjx_data` must have had `mjx.forward(...)` run on it (or, in
    practice, must come from the output of a `mjx.step`) so that
    `site_xpos`, `site_xmat`, `qfrc_bias`, and `qM` are populated. Inside
    a rollout loop where each iteration calls `mjx.step`, this is
    automatic.

    Parameters
    ----------
    mjx_model    : `mjx.Model`. Used for full-mass-matrix densification.
    mjx_data     : `mjx.Data` snapshot to read state from.
    ee_site_id   : integer site id of the EE (the `gripper` site).
    arm_dof_ids  : (7,) int array of qvel/qpos indices for the arm.
                   For our Franka they are 0..6.

    Returns
    -------
    OSCArmState with all fields populated.
    """
    import mujoco.mjx as mjx  # local import: keeps this module importable
                              # even when mjx isn't available (helpful for
                              # static unit tests of the math helpers).

    ee_pos = mjx_data.site_xpos[ee_site_id]
    # `site_xmat` is already shaped (nsite, 3, 3) in MJX 3.8.x -- no
    # reshape needed (whereas CPU MuJoCo flattens it to length 9).
    R_ee = mjx_data.site_xmat[ee_site_id]

    # `mjx.jac` returns (jacp, jacr) each of shape (nv, 3) -- the
    # TRANSPOSE of the (3, nv) convention used by CPU MuJoCo's
    # `mj_jacSite`. We transpose back so the rest of the math reads
    # like the textbook task-space equations. Then slice the columns
    # down to the 7 arm DOFs.
    jacp, jacr = mjx.jac(mjx_model, mjx_data, ee_pos, mjx_model.site_bodyid[ee_site_id])
    J_pos = jacp.T[:, arm_dof_ids]  # (3, 7)
    J_ori = jacr.T[:, arm_dof_ids]  # (3, 7)

    # Densify the joint-space mass matrix and pull out the arm block.
    M_full = mjx.full_m(mjx_model, mjx_data)
    M_arm = M_full[arm_dof_ids][:, arm_dof_ids]

    qpos_arm = mjx_data.qpos[arm_dof_ids]
    qvel_arm = mjx_data.qvel[arm_dof_ids]
    qfrc_bias_arm = mjx_data.qfrc_bias[arm_dof_ids]

    return OSCArmState(
        ee_pos=ee_pos,
        R_ee=R_ee,
        J_pos=J_pos,
        J_ori=J_ori,
        M_arm=M_arm,
        qpos_arm=qpos_arm,
        qvel_arm=qvel_arm,
        qfrc_bias_arm=qfrc_bias_arm,
    )


# ---------------------------------------------------------------------------
# Action wrapping: 7-D robosuite action -> 8-D MJX ctrl vector.
# ---------------------------------------------------------------------------

def delta_action_to_ctrl(
    action7: jax.Array,
    state: OSCArmState,
    initial_qpos_arm: jax.Array,
    cfg: OSCConfig = OSCConfig(),
) -> jax.Array:
    """End-to-end action -> ctrl pipe.

    Maps the 7-D robosuite OSC_POSE action to the 8-D MJX ctrl vector
    expected by our `simplified_can.xml` (7 arm torques + 1 finger
    position).

    Parameters
    ----------
    action7 : (7,) array. action[0:3] = delta pos, action[3:6] = delta
        axis-angle, action[6] = gripper command in [-1, 1] where +1
        closes and -1 opens. Matches the convention used by
        robosuite's `PandaGripper` controller.
    state   : OSCArmState extracted from the current MJX data.
    initial_qpos_arm : (7,) home pose, used for the nullspace term.
    cfg     : OSC config.

    Returns
    -------
    ctrl : (8,) MJX ctrl vector. ctrl[0:7] are arm torques, ctrl[7] is
        the desired finger_joint1 position.
    """
    goal_pos, goal_ori = update_goal_from_delta(
        action7[:6], state.ee_pos, state.R_ee, cfg
    )
    tau_arm = compute_osc_torques(state, goal_pos, goal_ori, initial_qpos_arm, cfg)
    # Gripper: robosuite convention is +1 = closed, -1 = open. Map
    # action[6] in [-1, 1] linearly to finger pos in [open, closed]
    # = [cfg.gripper_open_pos, 0.0].
    grip_cmd = jnp.clip(action7[6], -1.0, 1.0)
    finger_pos = cfg.gripper_open_pos * 0.5 * (1.0 - grip_cmd)
    ctrl = jnp.concatenate([tau_arm, finger_pos[None]])
    return ctrl
