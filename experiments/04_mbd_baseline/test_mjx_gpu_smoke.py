"""Gate-4 fail-fast GPU smoke for the MJX rollout.

This script runs FIRST on every GPU VM, before the full experiment.
Its job is to crash loudly within ~60 s if anything about the GPU
setup is wrong, so we don't burn $0.70 / hr running a slow CPU
fallback for an hour.

Checks (in order, abort on first failure):

  1. ``jax.default_backend() == 'gpu'``.
  2. A small matmul actually executes on a GPU device.
  3. ``mujoco`` + ``mujoco.mjx`` import.
  4. ``build_rollout_context(scenes/simplified_can.xml)`` completes.
  5. One batched MJX rollout (M=8, H=20, n_substeps=5) compiles and
     executes WITHOUT NaN; record wall time so we can sanity-check
     the speedup claim.
  6. ``ctx.batched_rollout`` second call (post-compile) finishes in
     less than CPU_BASELINE_S seconds. CPU baseline at the same shape
     is 6-8 s on Mac; L4 should be sub-second post-compile.

The thresholds are loose -- the goal is "GPU is actually working",
not "GPU is fast." Speed measurement happens in the full experiment.

Exit codes:
   0  -- all checks passed
  20  -- JAX backend is not GPU
  21  -- a check raised an exception (see stderr)
  22  -- post-compile batched rollout slower than CPU_BASELINE_S
        (likely silent CPU fallback)

Usage:
    python experiments/04_mbd_baseline/test_mjx_gpu_smoke.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _HERE.parent
for _p in (_HERE, _EXPERIMENTS_DIR):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

warnings.filterwarnings("ignore", message=".*overflow encountered in cast.*")


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"


# Per-check timing budget. L4 post-compile batched_rollout at
# (M=8, H=20, n_substeps=5) should land well under 1 s. CPU on a
# beefy Mac takes ~6-8 s. Set the threshold at 3 s: above that, we
# almost certainly silently fell back to CPU.
POST_COMPILE_BUDGET_S = 3.0


def main() -> int:
    print("== Gate-4 MJX GPU smoke ==")

    # ---- 1. JAX backend ----
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as e:
        print(_red(f"  [FAIL jax-import] {e!r}"), file=sys.stderr)
        return 21
    backend = jax.default_backend()
    devices = jax.devices()
    print(f"  [.. backend]  jax={jax.__version__}  backend={backend!r}  "
          f"devices={[str(d) for d in devices]}")
    if backend != "gpu":
        print(_red(
            f"  [FAIL backend] jax.default_backend() = {backend!r}, "
            "expected 'gpu'. The jax[cuda12] wheel may be installed but "
            "the CUDA runtime can't see a device (NVIDIA driver missing? "
            "ldconfig stale?). See /tmp/install_linux_gpu.log."
        ), file=sys.stderr)
        return 20
    print(_green("  [OK   backend] gpu detected"))

    # ---- 2. matmul on GPU ----
    try:
        x = jnp.ones((1024, 1024), dtype=jnp.float32)
        t0 = time.perf_counter()
        y = (x @ x).sum()
        y.block_until_ready()
        matmul_s = time.perf_counter() - t0
        print(f"  [OK   matmul] 1024x1024 matmul on gpu: "
              f"{matmul_s * 1e3:.1f} ms   result={float(y):.0f}")
        # On L4 this should be sub-millisecond after warmup; we don't
        # threshold here because the first matmul also pays JIT cost.
    except Exception as e:  # noqa: BLE001
        print(_red(f"  [FAIL matmul] {e!r}"), file=sys.stderr)
        return 21

    # ---- 3. mujoco + mjx import ----
    try:
        import mujoco  # noqa: F401
        import mujoco.mjx as mjx  # noqa: F401
        print(f"  [OK   mjx-imp] mujoco={mujoco.__version__}  mjx imported")
    except ImportError as e:
        print(_red(f"  [FAIL mjx-imp] {e!r}"), file=sys.stderr)
        return 21

    # ---- 4. build_rollout_context ----
    try:
        from mjx_rollout import build_rollout_context
    except ImportError as e:
        print(_red(f"  [FAIL ctx-imp] {e!r}"), file=sys.stderr)
        return 21

    scene_xml = _HERE / "scenes" / "simplified_can.xml"
    if not scene_xml.exists():
        print(_red(f"  [FAIL ctx-xml] {scene_xml} not found"), file=sys.stderr)
        return 21
    try:
        t0 = time.perf_counter()
        ctx = build_rollout_context(str(scene_xml), n_substeps=5)
        ctx_build_s = time.perf_counter() - t0
        print(f"  [OK   ctx]     built in {ctx_build_s:.1f}s  "
              f"(nv={ctx.mj_model.nv}, nu={ctx.mj_model.nu})")
    except Exception as e:  # noqa: BLE001
        print(_red(f"  [FAIL ctx] {e!r}"), file=sys.stderr)
        return 21

    # ---- 5. one batched rollout (compile + execute) ----
    M, H, action_dim = 8, 20, 7
    rng = np.random.default_rng(0)
    plans = rng.normal(0.0, 0.3, size=(M, H, action_dim)).astype(np.float32)
    plans = np.clip(plans, -1.0, 1.0)

    key_id = int(mujoco.mj_name2id(ctx.mj_model, mujoco.mjtObj.mjOBJ_KEY,
                                    "home"))
    qpos_init = jnp.asarray(ctx.mj_model.key_qpos[key_id], dtype=jnp.float32)
    qvel_init = jnp.zeros((ctx.mj_model.nv,), dtype=jnp.float32)

    try:
        t0 = time.perf_counter()
        costs, _qf, _ps = ctx.batched_rollout(
            qpos_init, qvel_init, jnp.asarray(plans),
        )
        costs.block_until_ready()
        first_s = time.perf_counter() - t0
        if not np.isfinite(np.asarray(costs)).all():
            print(_red(
                "  [FAIL rollout-nan] batched_rollout produced NaN/inf"
            ), file=sys.stderr)
            return 21
        print(f"  [OK   rollout-1] first batched rollout (incl. compile): "
              f"{first_s:.1f}s  costs range "
              f"[{float(np.asarray(costs).min()):+.3f}, "
              f"{float(np.asarray(costs).max()):+.3f}]")
    except Exception as e:  # noqa: BLE001
        print(_red(f"  [FAIL rollout-1] {e!r}"), file=sys.stderr)
        return 21

    # ---- 6. second batched rollout (post-compile, must be fast) ----
    try:
        # Slightly different plans so the cache isn't trivially hit.
        plans2 = rng.normal(0.0, 0.3, size=(M, H, action_dim)).astype(np.float32)
        plans2 = np.clip(plans2, -1.0, 1.0)
        t0 = time.perf_counter()
        costs2, _qf, _ps = ctx.batched_rollout(
            qpos_init, qvel_init, jnp.asarray(plans2),
        )
        costs2.block_until_ready()
        second_s = time.perf_counter() - t0
        print(f"  [..  rollout-2] post-compile batched rollout: "
              f"{second_s:.2f}s  (budget: < {POST_COMPILE_BUDGET_S:.1f}s)")
        if second_s > POST_COMPILE_BUDGET_S:
            print(_yellow(
                f"  [FAIL rollout-2] post-compile time {second_s:.2f}s "
                f"> budget {POST_COMPILE_BUDGET_S:.1f}s. "
                "Likely silent CPU fallback inside JAX (jax-cuda12 "
                "plugin loaded but CUDA runtime unhappy). Check "
                "/tmp/install_linux_gpu.log for warnings."
            ), file=sys.stderr)
            return 22
        print(_green(
            f"  [OK   rollout-2] {second_s:.2f}s   "
            f"speedup vs first {first_s / max(second_s, 1e-3):.0f}x"
        ))
    except Exception as e:  # noqa: BLE001
        print(_red(f"  [FAIL rollout-2] {e!r}"), file=sys.stderr)
        return 21

    print()
    print(_green(
        "PASS  GPU smoke clean.  Safe to launch full MJX experiment."
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
