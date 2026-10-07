"""Regression test for the worker-perturbation propagation patch.

Without this patch, multiprocessing workers spawn fresh envs from
the dataset's MJCF and never see the goal / dynamics perturbations
applied to the main env, so MBD's planning runs against a silently
*different* cost from the one the executed rollout uses.  This test
checks the two halves of the fix:

  PART A.  In-process call to ``_worker_init`` with a perturbations
           dict moves the can task's ``bin2`` body by exactly the
           requested xyz on the worker-side env, and also touches
           the cached env-level attributes
           (``env.bin2_pos``, ``env.target_bin_placements``) that
           the staged-reward function reads from.

  PART B.  A multiprocessing.Pool created with the new ``_worker_init``
           initargs runs rollouts whose total rewards differ from the
           unperturbed canonical case.  The exact reward delta is
           task-dependent and stochastic across spawn-init random
           seeds, but bit-identical rewards are the unique signature
           of "perturbation didn't propagate" -- so we just check
           the rewards are not equal.

Run from the repo root:

    python experiments/04_mbd_baseline/test_worker_perturbation.py

Exit code 0 = PASS, anything else = FAIL.  Wall time: ~20s on the
M4 Max (most of it is two spawn-context env constructions in PART B).
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

# Resolve repo paths regardless of CWD.
_THIS_FILE = Path(__file__).resolve()
EXP04 = _THIS_FILE.parent
EXPERIMENTS = EXP04.parent
EXP15 = EXPERIMENTS / "15_unified_hybrid"

for p in (EXPERIMENTS, EXP04, EXP15):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# 20 cm displacement in +x.  Large enough to leave a clear reward
# delta in PART B without dragging the can off the workspace.
GOAL_XYZ_TEST = (0.20, 0.0, 0.0)


def part_a(can_ph_dataset: Path) -> None:
    print("\n=== PART A.  In-process _worker_init probe ===")
    import worker as worker_mod  # exp04/worker.py

    # Canonical (no perturbation) baseline.
    worker_mod._worker_env = None
    worker_mod._worker_init(str(can_ph_dataset), perturbations=None)
    env_canon = worker_mod._worker_env
    bin2_id = env_canon.sim.model.body_name2id("bin2")
    canon_pos = env_canon.sim.model.body_pos[bin2_id].copy()
    canon_cached = env_canon.bin2_pos.copy()
    canon_targets = env_canon.target_bin_placements.copy()
    print(f"  canonical bin2_pos          = {canon_pos}")
    print(f"  canonical env.bin2_pos      = {canon_cached}")
    print(f"  canonical target_bin_placements.shape = {canon_targets.shape}")

    # Fresh env with perturbation applied during _worker_init.
    worker_mod._worker_env = None
    worker_mod._worker_init(
        str(can_ph_dataset),
        perturbations={"task": "can", "goal_xyz": GOAL_XYZ_TEST},
    )
    env_pert = worker_mod._worker_env
    pert_pos = env_pert.sim.model.body_pos[bin2_id].copy()
    pert_cached = env_pert.bin2_pos.copy()
    pert_targets = env_pert.target_bin_placements.copy()
    print(f"  perturbed bin2_pos          = {pert_pos}")
    print(f"  perturbed env.bin2_pos      = {pert_cached}")

    delta_body_pos = pert_pos - canon_pos
    delta_cached = pert_cached - canon_cached
    expected = np.array(GOAL_XYZ_TEST)
    print(f"  delta(body_pos)             = {delta_body_pos}  (expected {expected})")
    print(f"  delta(env.bin2_pos)         = {delta_cached}    (expected {expected})")

    np.testing.assert_allclose(
        delta_body_pos, expected, atol=1e-9,
        err_msg="body_pos delta != requested goal_xyz",
    )
    np.testing.assert_allclose(
        delta_cached, expected, atol=1e-9,
        err_msg="env.bin2_pos cached delta != requested goal_xyz",
    )

    # The Can task has 4 target-bin slots (one per object); the can's
    # slot is the one that should move.  Other slots may or may not
    # move (env_perturbations.py mutates the whole target_bin_placements
    # row for the can; see its docstring).  We just assert that at
    # least one row moved.
    delta_targets_norms = np.linalg.norm(pert_targets - canon_targets, axis=1)
    moved_rows = int((delta_targets_norms > 1e-9).sum())
    print(f"  target_bin_placements: {moved_rows} of {len(canon_targets)} rows moved")
    assert moved_rows >= 1, (
        "target_bin_placements should have AT LEAST one moved row "
        "for can/ph (the can's target bin)."
    )

    # Drop the canonical in-process env so the spawn-context workers
    # in PART B start clean.
    worker_mod._worker_env = None
    print("PART A: PASS")


def part_b(can_ph_dataset: Path) -> int:
    print("\n=== PART B.  Pool rollout reward delta ===")

    import multiprocessing as mp
    from worker import _worker_init, _worker_rollout

    with h5py.File(can_ph_dataset, "r") as f:
        init_state = f["data/demo_0/states"][0].copy()
        actions = f["data/demo_0/actions"][:60].copy()
    print(f"  init_state shape = {init_state.shape}")
    print(f"  actions shape    = {actions.shape}")

    ctx = mp.get_context("spawn")

    def _make_pool(perturbations):
        return ctx.Pool(
            2,
            initializer=_worker_init,
            initargs=(str(can_ph_dataset), perturbations),
        )

    print("  Spawning canonical pool (no perturbation)...")
    pool_canon = _make_pool(None)
    canon_results = pool_canon.map(
        _worker_rollout,
        [(init_state, actions) for _ in range(2)],
    )
    canon_rewards = [r[0] for r in canon_results]
    canon_successes = [r[1] for r in canon_results]
    pool_canon.close()
    pool_canon.join()
    print(f"  canonical rewards   = {canon_rewards}  successes={canon_successes}")

    print("  Spawning perturbed pool (goal +0.20 m in +x)...")
    pool_pert = _make_pool({"task": "can", "goal_xyz": GOAL_XYZ_TEST})
    pert_results = pool_pert.map(
        _worker_rollout,
        [(init_state, actions) for _ in range(2)],
    )
    pert_rewards = [r[0] for r in pert_results]
    pert_successes = [r[1] for r in pert_results]
    pool_pert.close()
    pool_pert.join()
    print(f"  perturbed rewards   = {pert_rewards}  successes={pert_successes}")

    delta = abs(canon_rewards[0] - pert_rewards[0])
    print(f"  reward delta        = {delta:.6f}")

    # Bit-identical rewards = "perturbation didn't propagate" (the
    # regression we are guarding against).  Under spawn context with
    # fresh envs there is no reason for the rewards to be exactly
    # equal even at canon=pert except by this specific bug.
    if canon_rewards == pert_rewards:
        print("PART B: FAIL  -- rewards bit-identical, perturbation didn't propagate!")
        return 2

    print("PART B: PASS  (rewards differ, perturbation propagated to workers)")
    return 0


def main() -> int:
    from common import default_dataset_path
    can_ph_dataset = default_dataset_path("can")
    if not can_ph_dataset.exists():
        raise FileNotFoundError(
            f"Could not locate Can/ph dataset at {can_ph_dataset}."
        )
    print(f"Dataset: {can_ph_dataset}")

    part_a(can_ph_dataset)
    rc = part_b(can_ph_dataset)
    if rc == 0:
        print("\nSMOKE OVERALL: PASS")
    else:
        print("\nSMOKE OVERALL: FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
