"""Gate-2b OSC math parity test: JAX OSC vs robosuite's reference numpy OSC.

This test ISOLATES the OSC controller from the physics simulator. We
draw synthetic but structurally-realistic OSC inputs (a 7-DOF
positive-definite mass matrix, a full-rank 6x7 Jacobian, a random EE
pose, a random goal pose, random q/qdot, random gravity-comp bias)
and feed them through both:

  * `jax_osc.compute_osc_torques`             (our JAX port)
  * robosuite's reference numpy functions in
    `robosuite.utils.control_utils`           (the ground truth)

If the math is faithfully ported, the two should produce identical
torque vectors up to float32 round-off (~1e-4 Nm).

This is the CLEANEST possible parity check: it avoids all the
known sources of trajectory divergence (different MJCFs, different
mesh inertias, different damping, simplified collision geometry)
and tests ONLY the OSC math.

A separate integration test would compare a full MJX rollout vs a
robosuite rollout for a fixed action sequence; that test has
~1 cm of trajectory drift even with correct math, due to the
sim-to-sim physics differences listed in docs/mjx_migration.md.

Run with::

    conda run -n hybrid-traj python experiments/04_mbd_baseline/test_osc_parity.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Suppress JAX overflow warnings on int64 -> int32 casts during import.
warnings.filterwarnings("ignore", message=".*overflow encountered in cast.*")

import jax  # noqa: E402

# Enable float64 for the parity test only. Production OSC stays in
# float32 for throughput; here we want to isolate "math is correct"
# from "float32 round-off accumulated through opspace pinv chains."
# Must be done BEFORE any jnp arrays are created.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

# Reference implementation: robosuite's actual numpy code.
from robosuite.utils.control_utils import (  # noqa: E402
    nullspace_torques as rs_nullspace_torques,
    opspace_matrices as rs_opspace_matrices,
    orientation_error as rs_orientation_error,
)
from robosuite.utils.transform_utils import (  # noqa: E402
    axisangle2quat as rs_axisangle2quat,
    quat2mat as rs_quat2mat,
)

# Our JAX port.
from jax_osc import (  # noqa: E402
    OSCConfig,
    OSCArmState,
    axis_angle_to_matrix,
    compute_osc_torques,
    orientation_error_matrices,
    update_goal_from_delta,
)

_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"


def _ok(label: str, msg: str) -> None:
    print(f"  {_GREEN}[OK   {label}]{_RESET} {msg}")


def _fail(label: str, msg: str) -> int:
    print(f"  {_RED}[FAIL {label}]{_RESET} {msg}")
    return 1


# ---------------------------------------------------------------------------
# Reference torque computation: a faithful, side-by-side reproduction of
# `OperationalSpaceController.run_controller` with `uncoupling=True`,
# extracted as a pure function so we can unit-test against it.
# ---------------------------------------------------------------------------

def reference_osc_torques(
    *,
    M_arm: np.ndarray,       # (7, 7) joint-space inertia (arm block).
    J_pos: np.ndarray,       # (3, 7) site translation Jacobian.
    J_ori: np.ndarray,       # (3, 7) site rotation Jacobian.
    R_ee: np.ndarray,        # (3, 3) current EE rotation.
    ee_pos: np.ndarray,      # (3,)   current EE position.
    qpos_arm: np.ndarray,    # (7,)
    qvel_arm: np.ndarray,    # (7,)
    qfrc_bias_arm: np.ndarray,  # (7,)
    goal_pos: np.ndarray,    # (3,)
    goal_ori: np.ndarray,    # (3, 3)
    initial_qpos_arm: np.ndarray,  # (7,)
    kp_pos: float = 150.0,
    kp_ori: float = 150.0,
    damping_ratio: float = 1.0,
    null_kp: float = 10.0,
    arm_force_limit: float = 87.0,
) -> np.ndarray:
    """Numpy reference -- structurally identical to robosuite's
    `run_controller` with all the configurable robosuite defaults
    inlined. Uses `rs_opspace_matrices`, `rs_nullspace_torques`,
    `rs_orientation_error` so that any deviation from robosuite's
    actual library code shows up as a parity failure."""
    kd_pos = 2.0 * np.sqrt(kp_pos) * damping_ratio
    kd_ori = 2.0 * np.sqrt(kp_ori) * damping_ratio

    # EE velocity = J @ qdot (matches robosuite's intent for a fixed base).
    ee_lin_vel = J_pos @ qvel_arm
    ee_ang_vel = J_ori @ qvel_arm

    # Task-space PD.
    pos_err = goal_pos - ee_pos
    ori_err = rs_orientation_error(goal_ori, R_ee)
    F_des = kp_pos * pos_err - kd_pos * ee_lin_vel
    T_des = kp_ori * ori_err - kd_ori * ee_ang_vel

    # robosuite's opspace_matrices wants J_full as well.
    J_full = np.concatenate([J_pos, J_ori], axis=0)
    lambda_full, lambda_pos, lambda_ori, nullspace_matrix = rs_opspace_matrices(
        M_arm, J_full, J_pos, J_ori
    )

    # uncoupling=True branch.
    decoupled_wrench = np.concatenate(
        [lambda_pos @ F_des, lambda_ori @ T_des]
    )
    tau_task = J_full.T @ decoupled_wrench

    tau_grav = qfrc_bias_arm

    tau_null = rs_nullspace_torques(
        M_arm, nullspace_matrix, initial_qpos_arm, qpos_arm, qvel_arm, joint_kp=null_kp
    )

    tau = tau_task + tau_grav + tau_null
    return np.clip(tau, -arm_force_limit, arm_force_limit)


# ---------------------------------------------------------------------------
# Synthetic-input generator. Builds inputs that LOOK like a Franka in a
# nondegenerate configuration: SPD mass matrix, full-rank Jacobian,
# valid rotation matrices.
# ---------------------------------------------------------------------------

def random_osc_inputs(rng: np.random.Generator) -> dict:
    # Mass matrix: build SPD via M = A @ A.T + epsilon * I. Scale roughly
    # like a Franka (~kg.m^2).
    A = rng.standard_normal((7, 7)) * 0.3
    M_arm = A @ A.T + 0.1 * np.eye(7)

    # Jacobian: full-rank random (Franka's J is full row rank in
    # non-singular poses).
    J = rng.standard_normal((6, 7)) * 0.5
    J_pos, J_ori = J[:3], J[3:]

    # EE pose: random pos + random rotation matrix via SVD trick.
    ee_pos = rng.standard_normal(3) * 0.5
    R_seed = rng.standard_normal((3, 3))
    U, _, Vt = np.linalg.svd(R_seed)
    R_ee = U @ Vt
    if np.linalg.det(R_ee) < 0:
        R_ee[:, 0] = -R_ee[:, 0]  # ensure right-handed.

    # Goal pose: also valid.
    goal_pos = ee_pos + rng.standard_normal(3) * 0.05  # small delta.
    delta_axisangle = rng.standard_normal(3) * 0.1
    R_delta_quat = rs_axisangle2quat(delta_axisangle)
    R_delta = rs_quat2mat(R_delta_quat)
    goal_ori = R_delta @ R_ee

    qpos_arm = rng.uniform(-1.0, 1.0, size=7)
    qvel_arm = rng.standard_normal(7) * 0.3
    qfrc_bias_arm = rng.standard_normal(7) * 5.0  # ~5 Nm typical bias.
    initial_qpos_arm = rng.uniform(-1.0, 1.0, size=7)

    return dict(
        M_arm=M_arm,
        J_pos=J_pos,
        J_ori=J_ori,
        R_ee=R_ee,
        ee_pos=ee_pos,
        qpos_arm=qpos_arm,
        qvel_arm=qvel_arm,
        qfrc_bias_arm=qfrc_bias_arm,
        goal_pos=goal_pos,
        goal_ori=goal_ori,
        initial_qpos_arm=initial_qpos_arm,
    )


# ---------------------------------------------------------------------------
# Component-level checks: orientation error + Rodrigues, then full
# torque parity over N random samples.
# ---------------------------------------------------------------------------

def main() -> int:
    print("== Gate-2b OSC math parity ==")
    rng = np.random.default_rng(0xC0FFEE)

    # --- Component 1: orientation_error matches robosuite ---
    max_oe_err = 0.0
    for _ in range(64):
        # Two random rotation matrices.
        Ra_seed = rng.standard_normal((3, 3))
        Rb_seed = rng.standard_normal((3, 3))
        Ua, _, Vat = np.linalg.svd(Ra_seed)
        Ub, _, Vbt = np.linalg.svd(Rb_seed)
        Ra = Ua @ Vat
        Rb = Ub @ Vbt
        if np.linalg.det(Ra) < 0:
            Ra[:, 0] = -Ra[:, 0]
        if np.linalg.det(Rb) < 0:
            Rb[:, 0] = -Rb[:, 0]
        ours = np.asarray(orientation_error_matrices(jnp.asarray(Ra), jnp.asarray(Rb)))
        ref = rs_orientation_error(Ra, Rb)
        max_oe_err = max(max_oe_err, float(np.max(np.abs(ours - ref))))
    _ok("ori-err", f"max |ours - rs_orientation_error| over 64 trials: {max_oe_err:.2e}")
    if max_oe_err > 1e-5:
        return _fail("ori-err", f"orientation_error mismatch {max_oe_err:.2e} > 1e-5")

    # --- Component 2: axis_angle_to_matrix matches axisangle2quat + quat2mat ---
    max_aa_err = 0.0
    for _ in range(64):
        # Test across a range of magnitudes including near-zero.
        scale = 10 ** rng.uniform(-6, 0)
        vec = rng.standard_normal(3) * scale
        ours = np.asarray(axis_angle_to_matrix(jnp.asarray(vec)))
        ref = rs_quat2mat(rs_axisangle2quat(vec))
        max_aa_err = max(max_aa_err, float(np.max(np.abs(ours - ref))))
    _ok("rodrigues", f"max |ours - rs(axisangle->quat->mat)| over 64 trials: {max_aa_err:.2e}")
    if max_aa_err > 1e-5:
        return _fail("rodrigues", f"Rodrigues mismatch {max_aa_err:.2e} > 1e-5")

    # --- Component 3: update_goal_from_delta matches set_goal logic ---
    # robosuite path: clip action -> scale by output_max -> compute_goal_pos
    # (achieved mode, base frame, identity base) -> compute_goal_ori (delta).
    max_gp_err = 0.0
    max_go_err = 0.0
    cfg = OSCConfig()
    out_scale = np.array(
        [cfg.output_max_pos] * 3 + [cfg.output_max_ori] * 3
    )
    for _ in range(32):
        # Action: include both in-range and out-of-range values to test
        # the clipping.
        action6 = rng.uniform(-1.5, 1.5, size=6)
        # Identity base frame -> world == base; our update_goal uses world EE pose.
        ee_pos = rng.standard_normal(3) * 0.5
        R_seed = rng.standard_normal((3, 3))
        U, _, Vt = np.linalg.svd(R_seed)
        R_ee = U @ Vt
        if np.linalg.det(R_ee) < 0:
            R_ee[:, 0] = -R_ee[:, 0]

        # Reference: robosuite's compute_goal_pos in "achieved" mode +
        # "world" ref-frame is `current_pos + scaled_delta`; in "base"
        # frame with identity origin it's the same.
        scaled = np.clip(action6, -1.0, 1.0) * out_scale
        ref_goal_pos = ee_pos + scaled[:3]
        ref_R_delta = rs_quat2mat(rs_axisangle2quat(scaled[3:]))
        ref_goal_ori = ref_R_delta @ R_ee

        ours_goal_pos, ours_goal_ori = update_goal_from_delta(
            jnp.asarray(action6), jnp.asarray(ee_pos), jnp.asarray(R_ee), cfg
        )
        max_gp_err = max(
            max_gp_err,
            float(np.max(np.abs(np.asarray(ours_goal_pos) - ref_goal_pos))),
        )
        max_go_err = max(
            max_go_err,
            float(np.max(np.abs(np.asarray(ours_goal_ori) - ref_goal_ori))),
        )
    _ok("set-goal", f"goal_pos max err: {max_gp_err:.2e}, goal_ori max err: {max_go_err:.2e}")
    if max(max_gp_err, max_go_err) > 1e-5:
        return _fail(
            "set-goal",
            f"set_goal mismatch (pos={max_gp_err:.2e}, ori={max_go_err:.2e})",
        )

    # --- Component 4: full OSC torque parity ---
    max_tau_err = 0.0
    sample_taus = []
    for trial in range(32):
        inputs = random_osc_inputs(rng)
        ref_tau = reference_osc_torques(**inputs)

        state = OSCArmState(
            ee_pos=jnp.asarray(inputs["ee_pos"]),
            R_ee=jnp.asarray(inputs["R_ee"]),
            J_pos=jnp.asarray(inputs["J_pos"]),
            J_ori=jnp.asarray(inputs["J_ori"]),
            M_arm=jnp.asarray(inputs["M_arm"]),
            qpos_arm=jnp.asarray(inputs["qpos_arm"]),
            qvel_arm=jnp.asarray(inputs["qvel_arm"]),
            qfrc_bias_arm=jnp.asarray(inputs["qfrc_bias_arm"]),
        )
        ours_tau = np.asarray(
            compute_osc_torques(
                state,
                jnp.asarray(inputs["goal_pos"]),
                jnp.asarray(inputs["goal_ori"]),
                jnp.asarray(inputs["initial_qpos_arm"]),
            )
        )
        err = float(np.max(np.abs(ours_tau - ref_tau)))
        max_tau_err = max(max_tau_err, err)
        if trial < 3:
            sample_taus.append((ours_tau.copy(), ref_tau.copy()))

    _ok(
        "torques",
        f"max |JAX - robosuite| over 32 random OSC problems: "
        f"{max_tau_err:.2e} Nm   (float64 path, target < 1e-6)",
    )
    # In float64 the OSC math should reproduce robosuite to roughly
    # numpy/scipy round-off (~1e-10 for well-conditioned problems,
    # up to ~1e-7 if a particular trial draws a borderline-singular
    # mass / Jacobian). 1e-6 Nm gives 4-5 orders of margin and still
    # catches any genuine porting bug.
    if max_tau_err > 1e-6:
        # Show the first failing trial for diagnostics.
        print("    First-trial torques (ours vs robosuite):")
        for ours, ref in sample_taus[:1]:
            for i in range(7):
                print(
                    f"      joint {i}:  ours={ours[i]: .5f}  "
                    f"rs={ref[i]: .5f}  diff={(ours[i] - ref[i]): .2e}"
                )
        return _fail(
            "torques",
            f"OSC torque parity exceeded 1e-3 Nm: {max_tau_err:.2e}",
        )

    print()
    print(f"{_GREEN}PASS{_RESET}  Gate-2b OSC math parity complete.")
    print(
        f"       JAX OSC reproduces robosuite's `run_controller` to "
        f"{max_tau_err:.2e} Nm over 32 random problems."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
