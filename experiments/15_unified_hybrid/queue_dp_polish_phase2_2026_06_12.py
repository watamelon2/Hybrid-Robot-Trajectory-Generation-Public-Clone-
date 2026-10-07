#!/usr/bin/env python3
"""Phase-2 report-polish extension for the nwfix DP grid (2026-06-12).

Phase 1 (``requeue_dp_grid_nwfix_2026_06_12.py``) re-ran the full 140-shard
grid with the safe ``--n-workers 4`` into
``results/vast_safe_dp_nwfix_2026_06_12``.  This Phase-2 queue *adds* to
that same root so the cells combine automatically, and concentrates the
extra overnight compute where the report's "calibration of test-time
guidance" story lives: the two flagship lambda-sweeps.

What it adds (all nw=4, written into the Phase-1 root):

  T1  Flagship lambda-curve densification + N->60 (highest value):
        Can  ep0199 sigma0.10 : lambda in {0.25,0.75,1.5} at N=60
                                (seed blocks 100000-100059)
        Square ep0199         : lambda in {0.25,0.75,1.5} at N=60
      => smooth 9-point (Can) / 7-point (Square) over-anchoring curves.
  T2  N->60 for the lambda values Phase 1 already ran at N=30
        (second seed block 100030-100059 only):
        Can  ep0199 sigma0.10 : {0,0.5,1,2,5,10}
        Square ep0199         : {0,0.5,1,2}
  T3  N->60 for the Square prior-quality ladder (lowest priority, runs
      last) so the "does guidance help weaker priors more?" axis also
      gets tight CIs:
        Square ep0149/ep0099/ep0049 : {0,0.5,1,2} second block only.

Jobs are emitted in T1 -> T2 -> T3 order; ``run_queue`` pops from the
front, so an early stop still banks the most report-critical data.

Usage (remote venv):
    python queue_dp_polish_phase2_2026_06_12.py --manifest   # dry-run
    python queue_dp_polish_phase2_2026_06_12.py --run        # launch
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import queue_vast_safe_remote_2026_06_10 as base  # noqa: E402

SAFE_WORKERS = str(int(os.environ.get("HRTG_NWFIX_WORKERS", "4")))
assert int(SAFE_WORKERS) >= 2, "n_workers must be >= 2 to avoid corruption"

ROOT = base.REPO / "experiments/15_unified_hybrid/results/vast_safe_dp_nwfix_2026_06_12"
SHARD = 6                       # rollouts per shard (matches Phase 1)
BLOCK0 = range(0, 30, SHARD)    # seeds 100000..100024 (Phase-1 block)
BLOCK1 = range(30, 60, SHARD)   # seeds 100030..100054 (new second block)
SANITY = _HERE / "check_unified_sanity.py"
FEATURES = _HERE / "extract_calibration_features.py"

CAN_SMAX = "0.10"               # flagship Can sigma_max
CAN_NEW_LAMS = ("0.25", "0.75", "1.5")
CAN_OLD_LAMS = ("0", "0.5", "1", "2", "5", "10")
SQ_NEW_LAMS = ("0.25", "0.75", "1.5")
SQ_OLD_LAMS = ("0", "0.5", "1", "2")
SQ_LADDER = ("ep0149", "ep0099", "ep0049")

# Tier selector for splitting work across machines (disjoint slices).
# HRTG_PHASE2_TIERS="T1,T2" on one box, "T3" on another, etc.
_PHASE2_TIERS = {
    t for t in os.environ.get("HRTG_PHASE2_TIERS", "T1,T2,T3").replace(" ", "").split(",") if t
}


def _nw4(cmd: list[str]) -> tuple[str, ...]:
    return tuple(base.replace_flag(cmd, "--n-workers", [SAFE_WORKERS]))


def _can_job(lam: str, offset: int) -> base.Job:
    seed0 = 100000 + offset
    name = (f"can_dp_ep0199_smax{base.tag_value(CAN_SMAX)}"
            f"_lam{base.tag_value(lam)}_seed{seed0}_n{SHARD}")
    outdir = ROOT / name
    cmd = base.base_can_dp_command(outdir, name, CAN_SMAX, (lam,), SHARD, seed0)
    return base.Job(name=name, command=_nw4(cmd), outdir=outdir,
                    gpu=None)


def _sq_job(ckpt: str, lam: str, offset: int) -> base.Job:
    seed0 = 100000 + offset
    name = (f"square_dp_{ckpt}_lam{base.tag_value(lam)}"
            f"_seed{seed0}_n{SHARD}")
    outdir = ROOT / name
    cmd = base.base_square_dp_command(outdir, name, ckpt, (lam,), SHARD, seed0)
    return base.Job(name=name, command=_nw4(cmd), outdir=outdir,
                    gpu=None)


def make_phase2_jobs() -> list[base.Job]:
    jobs: list[base.Job] = []
    # --- T1: flagship dense lambda at N=60 (both blocks) ---
    if "T1" in _PHASE2_TIERS:
        for lam in CAN_NEW_LAMS:
            for off in (*BLOCK0, *BLOCK1):
                jobs.append(_can_job(lam, off))
        for lam in SQ_NEW_LAMS:
            for off in (*BLOCK0, *BLOCK1):
                jobs.append(_sq_job("ep0199", lam, off))
    # --- T2: existing flagship lambda, second block only (N 30->60) ---
    if "T2" in _PHASE2_TIERS:
        for lam in CAN_OLD_LAMS:
            for off in BLOCK1:
                jobs.append(_can_job(lam, off))
        for lam in SQ_OLD_LAMS:
            for off in BLOCK1:
                jobs.append(_sq_job("ep0199", lam, off))
    # --- T3: Square prior-quality ladder, second block only (N 30->60) ---
    if "T3" in _PHASE2_TIERS:
        for ckpt in SQ_LADDER:
            for lam in SQ_OLD_LAMS:
                for off in BLOCK1:
                    jobs.append(_sq_job(ckpt, lam, off))
    # round-robin GPUs in emission order
    return [base.replace(j, gpu=i % 2) for i, j in enumerate(jobs)]


def manifest() -> None:
    jobs = make_phase2_jobs()
    print(f"[phase2] safe n_workers = {SAFE_WORKERS}")
    print(f"[phase2] root (shared with Phase 1) = {ROOT}")
    print(f"[phase2] active tiers = {sorted(_PHASE2_TIERS)}")
    print(f"[phase2] total shards = {len(jobs)} "
          f"({len(jobs)*SHARD} rollouts)")
    bad = [j.name for j in jobs if _nw_of(j.command) != SAFE_WORKERS]
    print(f"[phase2] jobs not on n_workers={SAFE_WORKERS} = {len(bad)} (must be 0)")
    print(f"[phase2] first 3 cells: {[j.name for j in jobs[:3]]}")
    print(f"[phase2] last 3 cells:  {[j.name for j in jobs[-3:]]}")


def _nw_of(cmd: tuple[str, ...]) -> str:
    return cmd[cmd.index("--n-workers") + 1] if "--n-workers" in cmd else "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.manifest or not args.run:
        manifest()
        if not args.run:
            return 0
    ROOT.mkdir(parents=True, exist_ok=True)
    return base.run_queue(
        make_phase2_jobs(), ROOT,
        max_total=base.DP_MAX_TOTAL,
        max_per_gpu=base.DP_MAX_PER_GPU,
        launch_delay=base.DP_LAUNCH_DELAY_SECONDS,
        poll_seconds=base.DP_POLL_SECONDS,
        max_retries=base.DP_MAX_RETRIES,
    )


if __name__ == "__main__":
    raise SystemExit(main())
