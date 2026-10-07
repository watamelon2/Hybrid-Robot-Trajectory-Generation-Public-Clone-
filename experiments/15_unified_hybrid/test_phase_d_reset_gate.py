"""Phase-D regression test: chunk-boundary `env.reset()` gate.

Verifies that ``unified_plan_and_execute`` does NOT call
``env.reset()`` between chunks when ``params.K == 0``.  This is
the smoking-gun regression for the 2026-05-30 Phase-D bug
(see ``experiments/17_dp_hybrid/results/phase_d_2026_05_30/``):

  * Pre-fix: every chunk triggered ``env.reset() +
    set_sim_state(env, current_state)`` under the
    ``_mbd_touches_main_env`` branch, wiping robosuite's
    Python-side controller state at every chunk boundary.
  * Post-fix: the chunk-boundary reset+restore is now gated on
    ``params.K > 0`` (i.e. only fires when MBD actually
    clobbered the main env).

Test strategy:

  1. Build a real robosuite env (no ckpt needed -- we don't
     evaluate a learned prior, just shape).
  2. Wrap ``env.reset`` to count invocations.
  3. Call ``unified_plan_and_execute`` with ``K = 0``,
     ``n_replans = 2`` ( -> 3 chunks of 8 steps each, total
     24 env steps), and a dummy prior that returns zeros.
  4. Assert the env was reset exactly ONCE during the call --
     the top-of-function reset at ``unified_loop.py:297``.
     Pre-fix would produce ``1 + n_chunks = 4`` resets.

Run::

    conda run -n hybrid-traj python experiments/15_unified_hybrid/test_phase_d_reset_gate.py
"""

from __future__ import annotations

import importlib.util as _importlib_util
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXP_ROOT = _HERE.parent
# Put ``common``/``mbd``/``unified_loop`` on path.  We deliberately do
# NOT add 04_mbd_baseline to sys.path because both 04 and 15 ship a
# ``run_can.py`` and a naive ``from run_can import ...`` would pick
# whichever was earlier on the path.  Instead we load exp 04's
# ``run_can.py`` by absolute file location -- mirroring how
# ``15_unified_hybrid/run_can.py:71`` does it.
sys.path.insert(0, str(_EXP_ROOT))                  # for ``common``
sys.path.insert(0, str(_EXP_ROOT / "03_bc_baseline"))  # for ``dataset`` (OBS_KEYS)
sys.path.insert(0, str(_EXP_ROOT / "04_mbd_baseline"))  # for ``mbd``
sys.path.insert(0, str(_HERE))                      # for ``unified_loop``

from common import (  # noqa: E402
    default_dataset_path,
    make_env_for_dataset,
)
from mbd import MBDParams, cosine_sigma_schedule  # noqa: E402
from unified_loop import unified_plan_and_execute  # noqa: E402

_spec4 = _importlib_util.spec_from_file_location(
    "exp04_run_can", str(_EXP_ROOT / "04_mbd_baseline" / "run_can.py")
)
_exp04 = _importlib_util.module_from_spec(_spec4)
_spec4.loader.exec_module(_exp04)
exp04_make_rollout_fn = _exp04.make_rollout_fn
exp04_cost_fn = _exp04.cost_fn


_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"


def _ok(label: str, msg: str = "") -> None:
    print(f"  {_GREEN}[{label}]{_RESET} {msg}")


def _fail(label: str, msg: str) -> int:
    print(f"  {_RED}[FAIL {label}]{_RESET} {msg}")
    return 1


class _ResetCountingEnvWrapper:
    """Forwards everything to the wrapped env but counts ``.reset()``."""

    def __init__(self, env):
        self._env = env
        self.reset_count = 0

    def __getattr__(self, name):
        # __getattr__ only called when normal lookup fails -> forwards
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        self.reset_count += 1
        return self._env.reset(*args, **kwargs)


def _make_zero_prior(action_dim: int, H: int):
    """Build a dummy ``(initial_mean_fn, score_fn_factory)`` that
    returns zeros.  We don't care about plan quality here; only the
    chunk-loop semantics being exercised."""

    def initial_mean_fn(env, init_state, s0_vec):  # noqa: ARG001
        return np.zeros((H, action_dim), dtype=np.float32)

    def score_fn_factory(s0_vec, K):  # noqa: ARG001
        # Returning None means the unified loop won't build / pass a
        # prior_score_fn to denoise_loop -- exactly what we want at
        # K=0 with no prior pull.
        return None

    return initial_mean_fn, score_fn_factory


def main() -> int:
    print("== Phase-D regression: chunk-boundary env.reset() gate ==")
    print(f"unified_loop: {_HERE / 'unified_loop.py'}")

    t0 = time.perf_counter()
    dataset_path = default_dataset_path("can")
    env, _env_meta = make_env_for_dataset(str(dataset_path))
    print(f"env built in {time.perf_counter() - t0:.1f} s")

    # Snapshot a real init state from the env.  We don't care which
    # demo it is; just that ``set_sim_state`` will accept it.
    env.reset()
    init_state = np.asarray(env.sim.get_state().flatten())

    action_dim = int(env.action_spec[0].shape[0])
    action_low = np.asarray(env.action_spec[0], dtype=np.float32)
    action_high = np.asarray(env.action_spec[1], dtype=np.float32)

    H = 16
    chunk_steps = 8
    n_replans = 2  # -> total chunks = n_replans + 1 = 3
    expected_chunks = n_replans + 1

    # MBDParams with K=0.  The schedule must have shape (K,) so for
    # K=0 it's the empty array.
    params = MBDParams(
        H=H, M=4, K=0, action_dim=action_dim,
        lambda_=1.0,
        sigma_schedule=cosine_sigma_schedule(0, sigma_max=0.25, sigma_min=0.05),
        clip_action=(action_low, action_high),
    )

    rng = np.random.default_rng(0)
    initial_mean_fn, score_fn_factory = _make_zero_prior(action_dim, H)

    counting_env = _ResetCountingEnvWrapper(env)
    counting_env.reset_count = 0  # reset the counter after env init

    # ---- Run the unified loop at K=0 ----
    t0 = time.perf_counter()
    result = unified_plan_and_execute(
        counting_env, init_state,
        prior_funcs=(initial_mean_fn, score_fn_factory),
        lambda_prior=0.0,  # no score pull
        params=params,
        rng=rng,
        n_replans=n_replans,
        chunk_steps=chunk_steps,
        action_low=action_low,
        action_high=action_high,
        no_warm_start=False,
        n_workers=1,          # the buggy path; Phase-D Fix 1 should still gate this
        dataset_path=dataset_path,
        make_rollout_fn=exp04_make_rollout_fn,
        cost_fn=exp04_cost_fn,
    )
    wall_s = time.perf_counter() - t0
    _ok("k0/run", f"completed in {wall_s:.1f} s")

    # We expect exactly 1 reset: the top-of-function init at
    # ``unified_loop.py:297``.  Pre-fix would be 1 + expected_chunks = 4.
    expected_n1 = 1
    expected_pre_fix_n1 = 1 + expected_chunks  # 4

    print(f"  observed env.reset() calls: {counting_env.reset_count}")
    print(f"  expected (post-fix):        {expected_n1}")
    print(f"  expected (pre-fix BUGGY):   {expected_pre_fix_n1}")

    if counting_env.reset_count != expected_n1:
        return _fail(
            "k0/reset-count",
            f"K=0 with n_workers=1: expected {expected_n1} env.reset() "
            f"calls (top-of-function only); got {counting_env.reset_count}.  "
            f"If got {expected_pre_fix_n1}, Phase-D Fix 1 has regressed -- "
            f"see ``experiments/17_dp_hybrid/results/phase_d_2026_05_30/"
            f"BRIEF.md``.  If got something else, the unified loop's chunk "
            f"semantics may have changed; re-examine the gate logic at "
            f"``unified_loop.py:478``.",
        )
    _ok("k0/reset-count", f"exactly {expected_n1} env.reset() call (top-of-function only)")

    # Sanity: per-chunk records present (each chunk records
    # ``steps_executed`` separately; the function returns a list of
    # them under ``chunks``).
    chunks_list = result.get("chunks", [])
    n_chunk_rec = len(chunks_list)
    if n_chunk_rec != expected_chunks:
        return _fail(
            "k0/chunks",
            f"expected {expected_chunks} chunk records; got {n_chunk_rec}",
        )
    _ok("k0/chunks", f"{n_chunk_rec} chunk records present")

    total_steps = sum(int(c.get("steps_executed", 0)) for c in chunks_list)
    expected_steps = expected_chunks * chunk_steps
    if total_steps != expected_steps:
        return _fail(
            "k0/steps",
            f"expected total steps_executed={expected_steps} "
            f"({expected_chunks} chunks * {chunk_steps} steps); "
            f"got {total_steps}",
        )
    _ok(
        "k0/steps",
        f"total steps_executed = {total_steps} = "
        f"{expected_chunks} chunks * {chunk_steps}",
    )

    print()
    print(f"{_GREEN}PASS{_RESET}  Phase-D Fix 1 (K=0 chunk-boundary reset gate) holds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
