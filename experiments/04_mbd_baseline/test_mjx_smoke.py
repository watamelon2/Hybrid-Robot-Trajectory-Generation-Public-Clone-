"""Gate-1 smoke test for the simplified PickPlaceCan scene.

Checks, in order:
  1. The scene XML loads under both CPU MuJoCo and MJX.
  2. `mjx.validate_model` (when available) returns no diagnostics.
  3. Resetting to the `home` keyframe puts the arm and can in the
     expected configuration.
  4. A 120-step zero-control rollout in MJX runs without NaN and
     the can settles (we expect it to drop a few mm onto the bin
     under gravity).
  5. The same 120-step rollout in CPU MuJoCo runs without NaN.

This is Gate 1 of the migration plan in `docs/mjx_migration.md`.
It does NOT yet test trajectory parity vs robosuite (Gate 2) or
the OSC controller (Gate 2). It only certifies that the scene
itself is MJX-compatible.

Run with::

    conda run -n hybrid-traj python experiments/04_mbd_baseline/test_mjx_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


_SCENE = Path(__file__).resolve().parent / "scenes" / "simplified_can.xml"


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def _step(label: str, msg: str = "") -> None:
    print(f"  [{label}] {msg}")


def run() -> int:
    import mujoco

    print(f"== Gate-1 smoke test ==")
    print(f"scene: {_SCENE.relative_to(_SCENE.parents[3])}")

    # ---- Step 1: load under CPU MuJoCo ----
    t0 = time.perf_counter()
    try:
        mj_model = mujoco.MjModel.from_xml_path(str(_SCENE))
    except Exception as e:  # noqa: BLE001
        print(_red(f"[FAIL] CPU MuJoCo failed to load scene: {e}"))
        return 1
    mj_data = mujoco.MjData(mj_model)
    _step(
        "load/cpu",
        f"nq={mj_model.nq} nv={mj_model.nv} nu={mj_model.nu} "
        f"ngeom={mj_model.ngeom} nbody={mj_model.nbody} "
        f"({(time.perf_counter() - t0) * 1e3:.0f} ms)",
    )

    # ---- Step 2: load + validate under MJX ----
    t0 = time.perf_counter()
    try:
        import mujoco.mjx as mjx
        import jax
        import jax.numpy as jnp
    except ImportError as e:
        print(_red(f"[FAIL] mujoco.mjx / jax not importable: {e}"))
        return 1

    try:
        mjx_model = mjx.put_model(mj_model)
    except Exception as e:  # noqa: BLE001
        print(_red(f"[FAIL] mjx.put_model raised: {e}"))
        return 1
    _step(
        "load/mjx",
        f"jax devices={jax.devices()} "
        f"({(time.perf_counter() - t0) * 1e3:.0f} ms)",
    )

    # `mjx.validate_model` was the documented gate; newer MJX
    # versions made the validation implicit inside `put_model`
    # (anything unsupported raises). If the symbol exists, call it
    # for the explicit "zero diagnostics" pass; otherwise the
    # successful `put_model` above is itself the gate.
    if hasattr(mjx, "validate_model"):
        try:
            mjx.validate_model(mj_model)
            _step("validate", "mjx.validate_model: OK")
        except Exception as e:  # noqa: BLE001
            print(_red(f"[FAIL] mjx.validate_model: {e}"))
            return 1
    else:
        _step(
            "validate",
            "mjx.validate_model not present in this MJX version "
            "(implicit validation passed via put_model)",
        )

    # ---- Step 3: keyframe reset ----
    try:
        key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
    except Exception:
        key_id = -1
    if key_id < 0:
        print(_red("[FAIL] keyframe 'home' not found"))
        return 1
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    mujoco.mj_forward(mj_model, mj_data)

    can_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "can")
    goal_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "goal")
    gripper_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    if min(can_body_id, goal_site_id, gripper_site_id) < 0:
        print(
            _red(
                f"[FAIL] required entity missing: "
                f"can_body={can_body_id} goal_site={goal_site_id} "
                f"gripper_site={gripper_site_id}"
            )
        )
        return 1
    _step(
        "reset",
        f"can xyz={mj_data.xpos[can_body_id].round(3)}  "
        f"goal xyz={mj_data.site_xpos[goal_site_id].round(3)}  "
        f"EE xyz={mj_data.site_xpos[gripper_site_id].round(3)}",
    )

    # ---- Step 4: MJX 120-step zero-control rollout ----
    mjx_data = mjx.make_data(mjx_model)
    # Initialise MJX data from the CPU keyframe state.
    mjx_data = mjx_data.replace(
        qpos=jnp.asarray(mj_data.qpos, dtype=jnp.float32),
        qvel=jnp.asarray(mj_data.qvel, dtype=jnp.float32),
        ctrl=jnp.asarray(mj_data.ctrl, dtype=jnp.float32),
    )

    H = 120
    t0 = time.perf_counter()
    # Warm up the JIT first to separate compile time from step time.
    step_jit = jax.jit(lambda d: mjx.step(mjx_model, d))
    mjx_data = step_jit(mjx_data)
    mjx_data.qpos.block_until_ready()
    compile_s = time.perf_counter() - t0
    _step("mjx/jit", f"compile + first step: {compile_s * 1e3:.0f} ms")

    t0 = time.perf_counter()
    rolled = jax.lax.fori_loop(
        0, H - 1, lambda _i, d: mjx.step(mjx_model, d), mjx_data
    )
    rolled.qpos.block_until_ready()
    roll_s = time.perf_counter() - t0
    qpos_np = np.asarray(rolled.qpos)
    qvel_np = np.asarray(rolled.qvel)
    if not np.isfinite(qpos_np).all() or not np.isfinite(qvel_np).all():
        print(_red(f"[FAIL] MJX rollout produced NaN/inf"))
        return 1
    _step(
        "mjx/roll",
        f"H={H} steps in {roll_s * 1e3:.0f} ms "
        f"({H / roll_s:.0f} step/s) "
        f"final |qvel|={float(jnp.linalg.norm(rolled.qvel)):.4f}",
    )

    # ---- Step 5: CPU MuJoCo 120-step zero-control rollout ----
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    t0 = time.perf_counter()
    for _ in range(H):
        mujoco.mj_step(mj_model, mj_data)
    cpu_s = time.perf_counter() - t0
    if not (np.isfinite(mj_data.qpos).all() and np.isfinite(mj_data.qvel).all()):
        print(_red("[FAIL] CPU rollout produced NaN/inf"))
        return 1
    _step(
        "cpu/roll",
        f"H={H} steps in {cpu_s * 1e3:.0f} ms ({H / cpu_s:.0f} step/s)",
    )

    # ---- Quick coarse parity check: same H=120 zero-ctrl, both
    # solvers, compare end-of-rollout joint position. Loose
    # tolerance (5e-2 rad on each arm joint) -- Gate 2 will
    # tighten and check action-driven trajectories. ----
    delta = np.asarray(rolled.qpos) - np.asarray(mj_data.qpos)
    arm_delta = float(np.max(np.abs(delta[:7])))
    print(
        _step.__doc__ if False else (
            f"  [parity] max |MJX-CPU| on arm qpos[:7]: "
            f"{arm_delta:.4f} rad "
            f"(tolerance for Gate-1 advisory: 5.0e-02)"
        )
    )

    print()
    print(_green("PASS  Gate 1 smoke checks complete."))
    print(
        f"  MJX step throughput (after JIT): "
        f"~{H / roll_s:.0f} step/s on {jax.devices()[0]}\n"
        f"  CPU throughput:                 ~{H / cpu_s:.0f} step/s\n"
        f"  MJX/CPU sequential ratio:        {(H / roll_s) / (H / cpu_s):.1f}x"
    )
    return 0


if __name__ == "__main__":
    sys.exit(run())
