#!/usr/bin/env python3
"""Phase-4 robustness deep-dive for the nwfix DP grid (2026-06-13).

Motivation (from analyze_nwfix_grid_2026_06_13.py on the pooled N<=60 grid):
the flagship over-anchoring lambda-curves WEAKENED with more seeds (Square
ep0199 lam0->lam1 went +43pp CI-separated at N=30 -> +15pp CI-overlapping at
N=60).  The cleanest remaining signal is the ROBUSTNESS sub-study: under
moderate dynamics shift, lambda=1 guidance beats lambda=0 by +10..+23pp
(larger than the +6pp nominal lift) -- but at N=30 that is not CI-separated,
and the breakdown KNEE between mass1.5 (guidance helps) and mass2.5 (total
collapse) is unmapped.

This phase, designed to run ~4-5h maximally parallel on the two 2xA40 boxes at
the proven-safe 24-way x nw8 (~216 procs < 255 cores; no oversubscription):

  Can tiers  (fast ~1.2h/cell) -> run on machine B:
    C1  power-boost the 3 best shifts {fric2.0, mass1.5, fric0.5} at
        lambda {0,1} with the 2nd seed block (N 30->60)            30 cells
    C2  lambda-response under the 2 best shifts {fric2.0, mass1.5} at
        the missing lambda {0.5, 2}, block0                        20 cells
    C3  breakdown knee: mass {1.25,1.75,2.0} at lambda {0,1}, block0 30 cells
      => 80 can cells ~= 4.0h at 24-way.

  Square tiers (slow ~2.8h/cell) -> run on machine A:
    S1  power-boost best shift {fric0.5} at lambda {0,1}, block1    10 cells
    S2  lambda-response under {fric0.5} at lambda {0.5,2}, block0   10 cells
    S3  breakdown knee: mass {1.75,2.0} at lambda {0,1}, block0     20 cells
      => 40 square cells ~= 4.7h at 24-way.

Split across machines with HRTG_PHASE4_TASKS=can (B) / square (A).  All cells
are NEW (new shifts / lambdas / seed blocks) so they never collide with the
Phase 1-3 cells already in the shared root.

Usage (remote venv):
    HRTG_PHASE4_TASKS=square python queue_dp_polish_phase4_2026_06_13.py --manifest
    HRTG_PHASE4_TASKS=square python queue_dp_polish_phase4_2026_06_13.py --run
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

SAFE_WORKERS = str(int(os.environ.get("HRTG_NWFIX_WORKERS", "8")))
assert int(SAFE_WORKERS) >= 2, "n_workers must be >= 2 to avoid corruption"

ROOT = base.REPO / "experiments/15_unified_hybrid/results/vast_safe_dp_nwfix_2026_06_12"
SHARD = 6
BLOCK0 = range(0, 30, SHARD)     # seeds 100000..100024  (N=30)
BLOCK1 = range(30, 60, SHARD)    # seeds 100030..100054  (-> N=60)
CAN_SMAX = "0.10"

# tasks selector for splitting across machines.
_TASKS = {t for t in os.environ.get("HRTG_PHASE4_TASKS", "can,square").replace(" ", "").split(",") if t}


def _nw(cmd: list[str]) -> tuple[str, ...]:
    return tuple(base.replace_flag(cmd, "--n-workers", [SAFE_WORKERS]))


def _suf(mass: str, fric: str) -> str:
    if mass == "1.0" and fric == "1.0":
        return ""
    return f"_m{base.tag_value(mass)}_f{base.tag_value(fric)}"


def _can(lam: str, off: int, mass: str = "1.0", fric: str = "1.0") -> base.Job:
    seed0 = 100000 + off
    name = (f"can_dp_ep0199_smax{base.tag_value(CAN_SMAX)}"
            f"_lam{base.tag_value(lam)}{_suf(mass, fric)}_seed{seed0}_n{SHARD}")
    outdir = ROOT / name
    cmd = base.base_can_dp_command(outdir, name, CAN_SMAX, (lam,), SHARD, seed0)
    if mass != "1.0" or fric != "1.0":
        cmd = [*cmd, "--mass-scale", mass, "--friction-scale", fric]
    return base.Job(name=name, command=_nw(cmd), outdir=outdir, gpu=None)


def _sq(lam: str, off: int, mass: str = "1.0", fric: str = "1.0") -> base.Job:
    seed0 = 100000 + off
    name = (f"square_dp_ep0199_lam{base.tag_value(lam)}{_suf(mass, fric)}"
            f"_seed{seed0}_n{SHARD}")
    outdir = ROOT / name
    cmd = base.base_square_dp_command(outdir, name, "ep0199", (lam,), SHARD, seed0)
    if mass != "1.0" or fric != "1.0":
        cmd = [*cmd, "--mass-scale", mass, "--friction-scale", fric]
    return base.Job(name=name, command=_nw(cmd), outdir=outdir, gpu=None)


def make_phase4_jobs() -> list[base.Job]:
    j: list[base.Job] = []
    if "can" in _TASKS:
        # C1: power-boost 3 best shifts to N=60 (block1), lam {0,1}
        for mass, fric in (("1.0", "2.0"), ("1.5", "1.0"), ("1.0", "0.5")):
            for lam in ("0", "1"):
                for off in BLOCK1:
                    j.append(_can(lam, off, mass, fric))
        # C2: lambda-response {0.5,2} under 2 best shifts, block0
        for mass, fric in (("1.0", "2.0"), ("1.5", "1.0")):
            for lam in ("0.5", "2"):
                for off in BLOCK0:
                    j.append(_can(lam, off, mass, fric))
        # C3: breakdown knee mass {1.25,1.75,2.0}, lam {0,1}, block0
        for mass in ("1.25", "1.75", "2.0"):
            for lam in ("0", "1"):
                for off in BLOCK0:
                    j.append(_can(lam, off, mass, "1.0"))
    if "square" in _TASKS:
        # S1: power-boost best shift fric0.5 to N=60 (block1), lam {0,1}
        for lam in ("0", "1"):
            for off in BLOCK1:
                j.append(_sq(lam, off, "1.0", "0.5"))
        # S2: lambda-response {0.5,2} under fric0.5, block0
        for lam in ("0.5", "2"):
            for off in BLOCK0:
                j.append(_sq(lam, off, "1.0", "0.5"))
        # S3: breakdown knee mass {1.75,2.0}, lam {0,1}, block0
        for mass in ("1.75", "2.0"):
            for lam in ("0", "1"):
                for off in BLOCK0:
                    j.append(_sq(lam, off, mass, "1.0"))
    return [base.replace(job, gpu=i % 2) for i, job in enumerate(j)]


def _nw_of(cmd: tuple[str, ...]) -> str:
    return cmd[cmd.index("--n-workers") + 1] if "--n-workers" in cmd else "?"


def manifest() -> None:
    j = make_phase4_jobs()
    n_can = sum(1 for x in j if x.name.startswith("can_"))
    n_sq = sum(1 for x in j if x.name.startswith("square_"))
    names = {x.name for x in j}
    bad = [x.name for x in j if _nw_of(x.command) != SAFE_WORKERS]
    est_h = (n_can * 1.2 + n_sq * 2.8) / 24
    print(f"[phase4] tasks = {sorted(_TASKS)}  n_workers = {SAFE_WORKERS}")
    print(f"[phase4] total = {len(j)}  can={n_can} square={n_sq}")
    print(f"[phase4] unique names = {len(names)} (collisions={len(j)-len(names)})")
    print(f"[phase4] not on nw={SAFE_WORKERS}: {len(bad)} (must be 0)")
    print(f"[phase4] est wall @24-way ~= {est_h:.1f}h")
    print(f"[phase4] first 2: {[x.name for x in j[:2]]}")
    print(f"[phase4] last 2:  {[x.name for x in j[-2:]]}")


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
        make_phase4_jobs(), ROOT,
        max_total=base.DP_MAX_TOTAL,
        max_per_gpu=base.DP_MAX_PER_GPU,
        launch_delay=base.DP_LAUNCH_DELAY_SECONDS,
        poll_seconds=base.DP_POLL_SECONDS,
        max_retries=base.DP_MAX_RETRIES,
    )


if __name__ == "__main__":
    raise SystemExit(main())
