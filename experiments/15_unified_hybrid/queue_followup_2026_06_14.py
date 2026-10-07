#!/usr/bin/env python3
"""Final-report follow-up cells (2026-06-14), to launch once Phase-4 frees GPUs.

Two job sets, selected by HRTG_FU_PHASE in {combo, deepen, both} and split across
machines by HRTG_FU_TASKS in {can, square, "can,square"}:

  COMBO  -- additive-vs-multiplicative combination-rule contrast (the report's
            "why product/gate/temper over-optimize" piece).  At the over-anchoring
            operating point (ep0199, nominal dynamics, lambda=1), matched seeds
            (block0, N=30), compares 4 fusion rules with the over-optimization
            proxy (--log-gate-diagnostic -> per-rollout gate_frac / ldem_minus_lmod):
              add  = additive bridge (--combine-mode score; our rule)
              prod = product / PoE   (--combine-mode tempered --temper-p 0)
              temp = generalised-mean (--combine-mode tempered --temper-p 1)
              gate = MBD-Demo max-gate(--combine-mode tempered --temper-p inf)
            -> 4 rules x 2 tasks x 5 seed-blocks = 40 cells.  Isolated in its own
            root so it cannot perturb the main nwfix grid parser.

  DEEPEN -- push the 2 best robustness cells to N=90 (block2 = seeds 100060..84):
              Can  friction x2.0   lambda {0,1}
              Square friction x0.5 lambda {0,1}
            -> 2 shifts x 2 lambda x 5 seed-blocks = 20 cells, written into the
            nwfix root so they extend the Phase 1-4 robustness cells.

Usage (remote venv, after phase4):
    HRTG_FU_PHASE=both HRTG_FU_TASKS=can    python queue_followup_2026_06_14.py --manifest
    HRTG_FU_PHASE=both HRTG_FU_TASKS=square python queue_followup_2026_06_14.py --run
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import queue_vast_safe_remote_2026_06_10 as base  # noqa: E402

SAFE_WORKERS = str(int(os.environ.get("HRTG_NWFIX_WORKERS", "8")))
assert int(SAFE_WORKERS) >= 2, "n_workers must be >= 2 to avoid corruption"

NWFIX_ROOT = base.REPO / "experiments/15_unified_hybrid/results/vast_safe_dp_nwfix_2026_06_12"
COMBO_ROOT = base.REPO / "experiments/15_unified_hybrid/results/combo_rules_2026_06_14"
SHARD = 6
BLOCK0 = range(0, 30, SHARD)      # seeds 100000..100024  (N=30)
BLOCK2 = range(60, 90, SHARD)     # seeds 100060..100084  (-> N=90)
CAN_SMAX = "0.10"

_TASKS = {t for t in os.environ.get("HRTG_FU_TASKS", "can,square").replace(" ", "").split(",") if t}
_PHASE = os.environ.get("HRTG_FU_PHASE", "both")

# rule tag -> extra flags appended to the base DP command (lambda already set to 1)
RULES = {
    "add":  ["--combine-mode", "score"],
    "prod": ["--combine-mode", "tempered", "--temper-p", "0",   "--sigma-demo", "0.25"],
    "temp": ["--combine-mode", "tempered", "--temper-p", "1",   "--sigma-demo", "0.25"],
    "gate": ["--combine-mode", "tempered", "--temper-p", "inf", "--sigma-demo", "0.25"],
}


def _nw(cmd: list[str]) -> tuple[str, ...]:
    # nw>=8 (corruption guard) + drop --save-step-trace: the gate diagnostic lives
    # in chunks.iter_curve (logged by --log-gate-diagnostic), independent of the
    # per-step obs trace, so stripping it keeps per-rollout files small for pulls.
    cmd = [c for c in cmd if c != "--save-step-trace"]
    return tuple(base.replace_flag(cmd, "--n-workers", [SAFE_WORKERS]))


def _suf(mass: str, fric: str) -> str:
    return "" if (mass == "1.0" and fric == "1.0") else f"_m{base.tag_value(mass)}_f{base.tag_value(fric)}"


def _combo_can(rule: str, off: int) -> base.Job:
    seed0 = 100000 + off
    name = f"can_dp_ep0199_smax{base.tag_value(CAN_SMAX)}_lam1_comb{rule}_seed{seed0}_n{SHARD}"
    outdir = COMBO_ROOT / name
    cmd = base.base_can_dp_command(outdir, name, CAN_SMAX, ("1",), SHARD, seed0)
    cmd = [*cmd, *RULES[rule], "--log-gate-diagnostic"]
    return base.Job(name=name, command=_nw(cmd), outdir=outdir, gpu=None)


def _combo_sq(rule: str, off: int) -> base.Job:
    seed0 = 100000 + off
    name = f"square_dp_ep0199_lam1_comb{rule}_seed{seed0}_n{SHARD}"
    outdir = COMBO_ROOT / name
    cmd = base.base_square_dp_command(outdir, name, "ep0199", ("1",), SHARD, seed0)
    cmd = [*cmd, *RULES[rule], "--log-gate-diagnostic"]
    return base.Job(name=name, command=_nw(cmd), outdir=outdir, gpu=None)


def _deepen_can(lam: str, off: int) -> base.Job:
    seed0 = 100000 + off
    name = f"can_dp_ep0199_smax{base.tag_value(CAN_SMAX)}_lam{base.tag_value(lam)}_m1p0_f2p0_seed{seed0}_n{SHARD}"
    outdir = NWFIX_ROOT / name
    cmd = base.base_can_dp_command(outdir, name, CAN_SMAX, (lam,), SHARD, seed0)
    cmd = [*cmd, "--mass-scale", "1.0", "--friction-scale", "2.0"]
    return base.Job(name=name, command=_nw(cmd), outdir=outdir, gpu=None)


def _deepen_sq(lam: str, off: int) -> base.Job:
    seed0 = 100000 + off
    name = f"square_dp_ep0199_lam{base.tag_value(lam)}_m1p0_f0p5_seed{seed0}_n{SHARD}"
    outdir = NWFIX_ROOT / name
    cmd = base.base_square_dp_command(outdir, name, "ep0199", (lam,), SHARD, seed0)
    cmd = [*cmd, "--mass-scale", "1.0", "--friction-scale", "0.5"]
    return base.Job(name=name, command=_nw(cmd), outdir=outdir, gpu=None)


def make_jobs() -> list[base.Job]:
    j: list[base.Job] = []
    if _PHASE in ("combo", "both"):
        for rule in RULES:
            for off in BLOCK0:
                if "can" in _TASKS:
                    j.append(_combo_can(rule, off))
                if "square" in _TASKS:
                    j.append(_combo_sq(rule, off))
    if _PHASE in ("deepen", "both"):
        for lam in ("0", "1"):
            for off in BLOCK2:
                if "can" in _TASKS:
                    j.append(_deepen_can(lam, off))
                if "square" in _TASKS:
                    j.append(_deepen_sq(lam, off))
    return [base.replace(job, gpu=i % 2) for i, job in enumerate(j)]


def _nw_of(cmd: tuple[str, ...]) -> str:
    return cmd[cmd.index("--n-workers") + 1] if "--n-workers" in cmd else "?"


def manifest() -> None:
    j = make_jobs()
    n_can = sum(1 for x in j if x.name.startswith("can_"))
    n_sq = sum(1 for x in j if x.name.startswith("square_"))
    names = {x.name for x in j}
    bad = [x.name for x in j if _nw_of(x.command) != SAFE_WORKERS]
    est_h = (n_can * 1.2 + n_sq * 2.8) / 24
    print(f"[followup] phase={_PHASE} tasks={sorted(_TASKS)} n_workers={SAFE_WORKERS}")
    print(f"[followup] total={len(j)} can={n_can} square={n_sq}")
    print(f"[followup] unique names={len(names)} (collisions={len(j)-len(names)})")
    print(f"[followup] not on nw={SAFE_WORKERS}: {len(bad)} (must be 0)")
    print(f"[followup] combo_root={COMBO_ROOT.name}  deepen_root={NWFIX_ROOT.name}")
    print(f"[followup] est wall @24-way ~= {est_h:.1f}h")
    for x in j[:3]:
        print("   e.g.", x.name)


def run() -> None:
    jobs = make_jobs()
    COMBO_ROOT.mkdir(parents=True, exist_ok=True)
    NWFIX_ROOT.mkdir(parents=True, exist_ok=True)
    base.run_queue(
        jobs,
        COMBO_ROOT,
        int(os.environ.get("HRTG_DP_MAX_TOTAL", "24")),
        int(os.environ.get("HRTG_DP_MAX_PER_GPU", "12")),
        base.DP_LAUNCH_DELAY_SECONDS,
        base.DP_POLL_SECONDS,
        base.DP_MAX_RETRIES,
    )


if __name__ == "__main__":
    if "--manifest" in sys.argv:
        manifest()
    elif "--run" in sys.argv:
        run()
    else:
        print("pass --manifest or --run")
