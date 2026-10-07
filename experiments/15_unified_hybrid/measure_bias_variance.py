"""Measure per-step action (bias, variance) for T2.1 hybrid vs T2.3 BC.

**Status (2026-05-31):** scaffolded but DEFERRED to a journal version.
The immediate writeup uses the **cell-level success-rate proxy** computed
by ``proxy_bias_variance.py`` (T2 multiseed data, no fresh compute) and
the matching ``docs/results_writeups/pareto_analysis.md``.  See those
artefacts for the actual numbers verifying the bias-variance Pareto
theorem's Corollary 1 in adaptation space.

Why this script is deferred
---------------------------

This per-step decomposition requires an action-saving hook in
``unified_plan_and_execute`` (about 30 LOC) plus ~1 h of fresh CPU per
seed.  The proxy script delivers a qualitative answer (T2.1 strictly
dominates T2.3 in (bias², variance) space) using only the already-merged
multiseed slate data, which is enough for the present writeup.  The
per-step measurement would let us:

  - quantify which **action dimensions** carry the bias (action-dim 6
    = gripper, per the T2.3 mechanism diagnostic);
  - separate the dynamic-MBD-search variance from the BC-retrain
    initialisation variance;
  - check the small-beta asymptotic prediction
    ``||bias_hybrid|| = O(beta)`` derived in
    ``docs/theory/08_asymmetric_robustness_model.md`` directly.

Methodology (when run)
----------------------
1. Pick K init-states from the canonical Can eval set (default 30).
2. For each init-state, replay each method (T2.1 / T2.3) for H steps
   under N stochastic seeds (default 5) at Can mass=1.3, capturing
   the per-step executed actions.
3. The "canonical reference action" a_ref(s, t) is the canonical
   hybrid's action at that (s, t) (from the eval on main, computed
   once and cached -- if cache missing the script computes it).
4. Per (method, action_dim, t):
     bias  = mean_{seeds, init_states}(a - a_ref)
     var   = var_{seeds}(a) averaged over init_states
   Aggregate into a (bias^2, variance) scalar per method via the
   standard bias-variance decomposition of MSE.

Compute budget (when run): 30 init-states x 5 seeds x 2 methods x
~10 s/replay = ~1 h CPU.  Hybrid is more expensive; budget conservative.

To-do before this script can run
--------------------------------
- Add ``--save-executed-actions`` flag to ``run_can.py`` that writes
  the per-chunk ``actions_executed`` arrays into the per-rollout JSON.
- Implement ``replay_hybrid_t2_1`` (currently NotImplementedError) by
  wiring through ``priors_zerotrain.load_demo_bank`` +
  ``unified_plan_and_execute`` with R=1 and the action-saving hook.

Both are bounded (~30 LOC each).  Tracked as a journal-version
follow-up in the project plan.

Usage (once runnable)
---------------------
::

    python measure_bias_variance.py \\
        --t2-1-checkpoint <canonical hybrid result dir> \\
        --t2-3-checkpoint experiments/03_bc_baseline/results/policy_mass1p3_205.pt \\
        --n-init-states 30 \\
        --n-seeds 5 \\
        --output-dir experiments/15_unified_hybrid/results/pareto_analysis_2026_05_28
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_EXP_DIR = _THIS_DIR.parent
for _p in (_THIS_DIR, _EXP_DIR, _EXP_DIR / "03_bc_baseline", _EXP_DIR / "05_hybrid_method"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", type=str, default="can")
    p.add_argument("--mass-scale", type=float, default=1.3)
    p.add_argument("--n-init-states", type=int, default=30)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--H", type=int, default=120)
    p.add_argument(
        "--t2-1-bank", type=Path, required=False,
        default=None,
        help="Path to can_mass1p3_205demos.hdf5 (the T2.1 bank). "
             "Default: SHARED_DATA/can_mass1p3_205demos.hdf5 if set, "
             "else fall back to experiments/15_unified_hybrid/data.",
    )
    p.add_argument(
        "--t2-3-checkpoint", type=Path, required=True,
        help="Path to BC checkpoint retrained on the T2.1 bank "
             "(experiments/03_bc_baseline/results/policy_mass1p3_205.pt).",
    )
    p.add_argument(
        "--canonical-ref-checkpoint", type=Path, default=None,
        help="Path to a canonical hybrid result JSON used to extract "
             "a_ref(s, t).  If None, compute the canonical hybrid "
             "ourselves on the same init-states (more expensive but "
             "self-contained).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("results/pareto_analysis"),
    )
    return p.parse_args()


def replay_bc(env, init_state, policy, action_low, action_high, H, seed):
    """Replay BC closed-loop and return per-step actions (H, A)."""
    from run_can import bc_warmstart_actions  # exp 05
    actions, _info = bc_warmstart_actions(
        env, init_state, policy, H,
        action_low.astype(np.float32),
        action_high.astype(np.float32),
        reset_seed=int(seed) & 0xFFFFFFFF,
    )
    return np.asarray(actions, dtype=np.float64)


def replay_hybrid_t2_1(env, init_state, bank, lam, H, K, M, sigma_max, sigma_min, seed):
    """Replay T2.1 hybrid (demo_1nn + lambda=1) and return per-step actions (H, A).

    Reuses the prior + MPC machinery from experiments/15_unified_hybrid/run_can.py
    by calling into _load_prior + unified_plan_and_execute with K=1 replan,
    n_rollouts=1, just to capture the action sequence for ONE seed.
    """
    # NOTE: scaffolded; full implementation requires wiring through
    # priors_zerotrain.load_demo_bank with --prior-bank-dataset, then
    # calling MPC with a single-rollout-per-seed budget.  TODO when run.
    raise NotImplementedError(
        "T2.1 hybrid replay: see TODO above.  Skeleton in place; the "
        "per-step action capture needs unified_plan_and_execute to be "
        "called with R=1 + action-saving hook."
    )


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bias-var] task            = {args.task}")
    print(f"[bias-var] mass_scale      = {args.mass_scale}")
    print(f"[bias-var] n_init_states   = {args.n_init_states}")
    print(f"[bias-var] n_seeds         = {args.n_seeds}")
    print(f"[bias-var] H               = {args.H}")
    print(f"[bias-var] output_dir      = {out_dir}")

    # TODO: build env at mass=1.3, sample init_states from canonical dataset,
    #       run replay_bc + replay_hybrid_t2_1 over (init_states, seeds),
    #       compute (bias, variance) decomposition, dump JSON.

    print("[bias-var] STUB -- see docstring TODO list; not yet runnable.")
    return 2  # explicit non-zero to flag the stub state


if __name__ == "__main__":
    sys.exit(main())
