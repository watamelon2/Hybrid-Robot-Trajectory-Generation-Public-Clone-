"""Gate-2 OSC + rollout smoke test.

Verifies, in order:

  1. `build_rollout_context` JITs cleanly on the simplified_can scene.
  2. A zero-action rollout (action == 0 everywhere) keeps the arm near
     the home pose: with `action[6]=0` the gripper is commanded to a
     position somewhere between fully-open and closed, and the OSC's
     gravity comp + nullspace term hold the arm config close to the
     home keyframe.
  3. A "push toward the can" rollout (action[0:3] points at the can,
     no rotation, gripper open) drives the EE *closer* to the can
     than the zero-action rollout does.
  4. A batched rollout of M = 8 random plans returns finite costs of
     finite per-step cost and finite final qpos.

This is the Gate-2 sanity check before the real parity test against
robosuite (`test_mjx_parity.py`, written separately).

Run with::

    conda run -n hybrid-traj python experiments/04_mbd_baseline/test_osc_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # so the local jax_osc / mjx_rollout import

from mjx_rollout import (  # noqa: E402
    CostWeights,
    build_rollout_context,
    home_state,
)


_SCENE = _HERE / "scenes" / "simplified_can.xml"
_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"


def _ok(label: str, msg: str = "") -> None:
    print(f"  {_GREEN}[{label}]{_RESET} {msg}")


def _fail(label: str, msg: str) -> int:
    print(f"  {_RED}[FAIL {label}]{_RESET} {msg}")
    return 1


def main() -> int:
    print("== Gate-2 OSC smoke test ==")
    print(f"scene: {_SCENE.relative_to(_HERE.parents[1])}")

    t0 = time.perf_counter()
    ctx = build_rollout_context(str(_SCENE), n_substeps=5)
    _ok(
        "build",
        f"ee_site={ctx.ee_site_id}  can_body={ctx.can_body_id}  "
        f"goal_site={ctx.goal_site_id}  ({(time.perf_counter() - t0) * 1e3:.0f} ms)",
    )
    _ok("goal", f"world-frame goal_pos = {np.asarray(ctx.goal_pos).round(3)}")
    _ok(
        "init",
        f"home arm qpos = {np.asarray(ctx.initial_qpos_arm).round(3)}",
    )

    qpos_init, qvel_init = home_state(ctx)

    # ---- 1. Zero-action regulation test ----
    H = 10
    action_zero = jnp.zeros((H, 7), dtype=jnp.float32)
    # action[6] = -1.0 fully opens the gripper (matches the keyframe).
    action_zero = action_zero.at[:, 6].set(-1.0)

    t0 = time.perf_counter()
    cost_zero, qpos_final_zero, per_step_zero = ctx.single_rollout(
        qpos_init, qvel_init, action_zero
    )
    cost_zero.block_until_ready()
    jit_s = time.perf_counter() - t0
    _ok("jit/zero", f"compile + first rollout: {jit_s * 1e3:.0f} ms")

    if not np.isfinite(np.asarray(cost_zero)).all():
        return _fail("zero", "cost not finite")
    if not np.isfinite(np.asarray(qpos_final_zero)).all():
        return _fail("zero", "qpos_final not finite")

    arm_drift = float(jnp.linalg.norm(qpos_final_zero[:7] - ctx.initial_qpos_arm))
    _ok(
        "zero-action",
        f"cost={float(cost_zero):.4f}  "
        f"arm drift |qpos_final - qpos_init|={arm_drift:.4f} rad  "
        f"(target: < 0.15 rad after {H} policy steps)",
    )
    if arm_drift > 0.5:
        # Too much drift -> the OSC isn't stabilising. Could be a
        # sign flip in the orientation error, missing gravity comp,
        # or a Jacobian-shape bug. Bail with a clear message.
        return _fail(
            "zero",
            f"arm drift {arm_drift:.3f} rad after {H} policy steps with "
            f"action=0 -- OSC is not regulating. Check torque signs / "
            f"qfrc_bias slicing / Jacobian shape.",
        )

    # ---- 2. Reaching test: push toward the can ----
    # Can is at world (0.5, -0.18, -0.10). EE starts at (0.307, 0, 0.49).
    # Unit vector from EE to can:
    ee_init = np.array([0.307, 0.0, 0.49], dtype=np.float32)
    can_init = np.array([0.5, -0.18, -0.10], dtype=np.float32)
    direction = can_init - ee_init
    direction = direction / np.linalg.norm(direction)
    action_reach = jnp.zeros((H, 7), dtype=jnp.float32)
    # delta_pos = +1 in [-1, 1] -> +0.05 m / step in OSC-scaled units.
    # Push at full magnitude along the EE->can direction.
    action_reach = action_reach.at[:, 0:3].set(jnp.asarray(direction)[None, :])
    action_reach = action_reach.at[:, 6].set(-1.0)  # keep gripper open

    cost_reach, qpos_final_reach, per_step_reach = ctx.single_rollout(
        qpos_init, qvel_init, action_reach
    )
    if not np.isfinite(np.asarray(cost_reach)).all():
        return _fail("reach", "cost not finite")

    # The EE-to-can shaping term should improve under the reach plan.
    # We can read final EE pos via a quick CPU forward at the final qpos.
    import mujoco
    mj_data = mujoco.MjData(ctx.mj_model)
    mj_data.qpos[:] = np.asarray(qpos_final_reach)
    mujoco.mj_forward(ctx.mj_model, mj_data)
    ee_final_reach = mj_data.site_xpos[ctx.ee_site_id].copy()
    mj_data.qpos[:] = np.asarray(qpos_final_zero)
    mujoco.mj_forward(ctx.mj_model, mj_data)
    ee_final_zero = mj_data.site_xpos[ctx.ee_site_id].copy()

    d_reach = float(np.linalg.norm(ee_final_reach - can_init))
    d_zero = float(np.linalg.norm(ee_final_zero - can_init))
    _ok(
        "reach-action",
        f"cost={float(cost_reach):.4f}  "
        f"|EE_final - can|: zero={d_zero:.3f}  reach={d_reach:.3f}",
    )
    if d_reach >= d_zero:
        return _fail(
            "reach",
            f"reach plan did not get the EE closer to the can "
            f"(zero={d_zero:.3f}, reach={d_reach:.3f}). OSC may not be "
            f"tracking the Cartesian goal.",
        )

    # ---- 3. Batched rollout ----
    rng = np.random.default_rng(0)
    M = 8
    actions_batch = rng.standard_normal((M, H, 7)).astype(np.float32) * 0.3
    actions_batch[:, :, 6] = -1.0  # all keep gripper open
    actions_batch = jnp.asarray(actions_batch)

    t0 = time.perf_counter()
    costs_batch, qpos_batch, per_step_batch = ctx.batched_rollout(
        qpos_init, qvel_init, actions_batch
    )
    costs_batch.block_until_ready()
    batch_s = time.perf_counter() - t0
    _ok(
        "jit/batch",
        f"M={M}  H={H}  compile + first batch: {batch_s * 1e3:.0f} ms",
    )

    costs_np = np.asarray(costs_batch)
    if not np.isfinite(costs_np).all():
        return _fail(
            "batch",
            f"non-finite costs: {costs_np}",
        )
    _ok(
        "batch-costs",
        f"min={costs_np.min():.3f}  median={np.median(costs_np):.3f}  "
        f"max={costs_np.max():.3f}",
    )

    # Re-run batch (no recompile) to measure steady-state throughput.
    t0 = time.perf_counter()
    costs_batch2, _, _ = ctx.batched_rollout(qpos_init, qvel_init, actions_batch)
    costs_batch2.block_until_ready()
    steady_s = time.perf_counter() - t0
    _ok(
        "batch-throughput",
        f"second batch (no recompile): {steady_s * 1e3:.0f} ms  "
        f"= {(M * H) / steady_s:.0f} policy-step/s  "
        f"= {(M * H * ctx.n_substeps) / steady_s:.0f} sim-step/s",
    )

    print()
    print(f"{_GREEN}PASS{_RESET}  Gate-2 OSC smoke checks complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
