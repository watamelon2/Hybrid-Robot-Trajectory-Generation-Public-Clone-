#!/usr/bin/env python3
"""Phase-3 deep report-polish extension for the nwfix DP grid (2026-06-12).

Runs AFTER Phase 1 (140 shards) + Phase 2 (170 shards) into the SAME root
``results/vast_safe_dp_nwfix_2026_06_12`` so everything pools.  This phase
exists to keep the accelerated 64-way box productive past 12h and to round
out the report's figures.  All cells are nw=4 and emitted in priority
order (run_queue pops from the front), so an early stop banks the most
valuable data first.

Tiers (value-ordered):
  T1  Can sigma0.25 -> full publication treatment (mirror sigma0.10):
        dense lambda {0.25,0.75,1.5} at N=60 + existing {0,0.5,1,2,5,10}
        second block -> a second smooth 9-point over-anchoring curve so
        the report can show the sigma_max x lambda interaction.
  T2  Square prior-quality ladder (ep0149/ep0099/ep0049) dense lambda
        {0.25,0.75,1.5} at N=60 -> completes the prior-quality x lambda
        surface ("does guidance help weaker priors more, and where is
        each prior's over-anchoring knee?").
  T3  N->90 (third seed block 100060-100084) for the two flagship curves
        (Can sigma0.10 9 lambdas + Square ep0199 7 lambdas) -> tightest
        CIs on the headline figures.
  T4  Robustness / dynamics-shift sub-study (NEW axis): Can ep0199
        sigma0.10, lambda in {0,1}, under mass-scale {1.5,2.5} and
        friction-scale {0.5,2.0} (4 shifted conditions), N=30 -> "does
        test-time cost guidance buy robustness to dynamics shift?".
  T5  N->90 third block for Can sigma0.25 (full 9 lambda) + Square ladder
        existing lambda -> depth across the rest of the surface.
  T6  Robustness sub-study for Square ep0199 (same 4 shifts, lambda {0,1}).
  T7  N->90 third block for Square ladder dense lambda.

Usage (remote venv):
    python queue_dp_polish_phase3_2026_06_12.py --manifest   # dry-run
    python queue_dp_polish_phase3_2026_06_12.py --run        # launch
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
SHARD = 6
BLOCK0 = range(0, 30, SHARD)     # seeds 100000..100024 (N first 30)
BLOCK1 = range(30, 60, SHARD)    # seeds 100030..100054 (-> N=60)
BLOCK2 = range(60, 90, SHARD)    # seeds 100060..100084 (-> N=90)

CAN_FULL_LAMS = ("0", "0.25", "0.5", "0.75", "1", "1.5", "2", "5", "10")
CAN_OLD_LAMS = ("0", "0.5", "1", "2", "5", "10")
CAN_NEW_LAMS = ("0.25", "0.75", "1.5")
SQ_FULL_LAMS = ("0", "0.25", "0.5", "0.75", "1", "1.5", "2")
SQ_OLD_LAMS = ("0", "0.5", "1", "2")
SQ_NEW_LAMS = ("0.25", "0.75", "1.5")
SQ_LADDER = ("ep0149", "ep0099", "ep0049")
# (mass_scale, friction_scale) shifted-dynamics conditions.
SHIFTS = (("1.5", "1.0"), ("2.5", "1.0"), ("1.0", "0.5"), ("1.0", "2.0"))
ROBUST_LAMS = ("0", "1")

# Tier selector for splitting work across machines (disjoint slices).
# e.g. HRTG_PHASE3_TIERS="T4,T6" runs only the robustness sub-study.
_PHASE3_TIERS = {
    t for t in os.environ.get("HRTG_PHASE3_TIERS", "T1,T2,T3,T4,T5,T6,T7").replace(" ", "").split(",") if t
}


def _nw4(cmd: list[str]) -> tuple[str, ...]:
    return tuple(base.replace_flag(cmd, "--n-workers", [SAFE_WORKERS]))


def _pert_suffix(mass: str, fric: str) -> str:
    if mass == "1.0" and fric == "1.0":
        return ""
    return f"_m{base.tag_value(mass)}_f{base.tag_value(fric)}"


def _can_job(lam: str, off: int, smax: str,
             mass: str = "1.0", fric: str = "1.0") -> base.Job:
    seed0 = 100000 + off
    name = (f"can_dp_ep0199_smax{base.tag_value(smax)}"
            f"_lam{base.tag_value(lam)}{_pert_suffix(mass, fric)}"
            f"_seed{seed0}_n{SHARD}")
    outdir = ROOT / name
    cmd = base.base_can_dp_command(outdir, name, smax, (lam,), SHARD, seed0)
    if mass != "1.0" or fric != "1.0":
        cmd = [*cmd, "--mass-scale", mass, "--friction-scale", fric]
    return base.Job(name=name, command=_nw4(cmd), outdir=outdir, gpu=None)


def _sq_job(ckpt: str, lam: str, off: int,
            mass: str = "1.0", fric: str = "1.0") -> base.Job:
    seed0 = 100000 + off
    name = (f"square_dp_{ckpt}_lam{base.tag_value(lam)}"
            f"{_pert_suffix(mass, fric)}_seed{seed0}_n{SHARD}")
    outdir = ROOT / name
    cmd = base.base_square_dp_command(outdir, name, ckpt, (lam,), SHARD, seed0)
    if mass != "1.0" or fric != "1.0":
        cmd = [*cmd, "--mass-scale", mass, "--friction-scale", fric]
    return base.Job(name=name, command=_nw4(cmd), outdir=outdir, gpu=None)


def make_phase3_jobs() -> list[base.Job]:
    j: list[base.Job] = []
    # T1: Can sigma0.25 full treatment to N=60.
    if "T1" in _PHASE3_TIERS:
        for lam in CAN_NEW_LAMS:
            for off in (*BLOCK0, *BLOCK1):
                j.append(_can_job(lam, off, "0.25"))
        for lam in CAN_OLD_LAMS:
            for off in BLOCK1:
                j.append(_can_job(lam, off, "0.25"))
    # T2: Square ladder dense lambda to N=60.
    if "T2" in _PHASE3_TIERS:
        for ckpt in SQ_LADDER:
            for lam in SQ_NEW_LAMS:
                for off in (*BLOCK0, *BLOCK1):
                    j.append(_sq_job(ckpt, lam, off))
    # T3: N->90 third block, flagship curves.
    if "T3" in _PHASE3_TIERS:
        for lam in CAN_FULL_LAMS:
            for off in BLOCK2:
                j.append(_can_job(lam, off, "0.10"))
        for lam in SQ_FULL_LAMS:
            for off in BLOCK2:
                j.append(_sq_job("ep0199", lam, off))
    # T4: robustness sub-study, Can ep0199 sigma0.10.
    if "T4" in _PHASE3_TIERS:
        for mass, fric in SHIFTS:
            for lam in ROBUST_LAMS:
                for off in BLOCK0:
                    j.append(_can_job(lam, off, "0.10", mass, fric))
    # T5: N->90 third block, Can sigma0.25 (full) + Square ladder old lambda.
    if "T5" in _PHASE3_TIERS:
        for lam in CAN_FULL_LAMS:
            for off in BLOCK2:
                j.append(_can_job(lam, off, "0.25"))
        for ckpt in SQ_LADDER:
            for lam in SQ_OLD_LAMS:
                for off in BLOCK2:
                    j.append(_sq_job(ckpt, lam, off))
    # T6: robustness sub-study, Square ep0199.
    if "T6" in _PHASE3_TIERS:
        for mass, fric in SHIFTS:
            for lam in ROBUST_LAMS:
                for off in BLOCK0:
                    j.append(_sq_job("ep0199", lam, off, mass, fric))
    # T7: N->90 third block, Square ladder dense lambda.
    if "T7" in _PHASE3_TIERS:
        for ckpt in SQ_LADDER:
            for lam in SQ_NEW_LAMS:
                for off in BLOCK2:
                    j.append(_sq_job(ckpt, lam, off))
    return [base.replace(job, gpu=i % 2) for i, job in enumerate(j)]


def _nw_of(cmd: tuple[str, ...]) -> str:
    return cmd[cmd.index("--n-workers") + 1] if "--n-workers" in cmd else "?"


def manifest() -> None:
    j = make_phase3_jobs()
    n_can = sum(1 for x in j if x.name.startswith("can_"))
    n_sq = sum(1 for x in j if x.name.startswith("square_"))
    n_pert = sum(1 for x in j if "_m" in x.name and "_f" in x.name)
    bad = [x.name for x in j if _nw_of(x.command) != SAFE_WORKERS]
    names = {x.name for x in j}
    print(f"[phase3] safe n_workers = {SAFE_WORKERS}")
    print(f"[phase3] root (shared) = {ROOT}")
    print(f"[phase3] active tiers = {sorted(_PHASE3_TIERS)}")
    print(f"[phase3] total shards = {len(j)} ({len(j)*SHARD} rollouts)"
          f"  can={n_can} square={n_sq} robustness={n_pert}")
    print(f"[phase3] unique names = {len(names)} (collisions={len(j)-len(names)})")
    print(f"[phase3] jobs not on n_workers={SAFE_WORKERS} = {len(bad)} (must be 0)")
    print(f"[phase3] first 2: {[x.name for x in j[:2]]}")
    print(f"[phase3] a robustness cell: "
          f"{[x.name for x in j if '_m' in x.name][:1]}")
    print(f"[phase3] last 2: {[x.name for x in j[-2:]]}")


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
        make_phase3_jobs(), ROOT,
        max_total=base.DP_MAX_TOTAL,
        max_per_gpu=base.DP_MAX_PER_GPU,
        launch_delay=base.DP_LAUNCH_DELAY_SECONDS,
        poll_seconds=base.DP_POLL_SECONDS,
        max_retries=base.DP_MAX_RETRIES,
    )


if __name__ == "__main__":
    raise SystemExit(main())
