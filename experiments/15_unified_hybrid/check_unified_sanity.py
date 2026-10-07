#!/usr/bin/env python3
"""Audit a unified_hybrid results root for the n_workers=1 corruption.

Background: when a unified run is launched with ``--n-workers 1`` under
chunked MPC (``--mpc-replans>=1``) and MBD (``--K>0``), unified_loop.py:599
forces a controller-state-wiping ``env.reset()`` between chunks and
unified success silently collapses to ~0%.  This invalidated the
``vast_safe_dp_2026_06_10`` grid (see
``docs/results_writeups/vast_dp_grid_recovery_analysis_2026_06_11.md``).

``run_can.py`` now refuses that combination by default (``_guard_nworkers``),
but this script is the *post-hoc* detector for already-produced JSONs and
the CI-style gate to bolt onto any new offload.  It flags two signatures:

1. **all-zero chunked-MPC cell** (success_rate == 0 over n>=MIN_N rollouts
   with mpc_replans>=1) -- the smoking gun for the corruption (a genuinely
   unsolved task is possible, hence "SUSPECT" not "FAIL", but it always
   warrants a look);
2. **lambda0 below a standalone reference** -- if you pass ``--ref
   task=rate`` (e.g. ``--ref can=0.66 --ref square=0.53``), any lambda0
   cell whose success is more than ``--margin`` below the reference is
   flagged FAIL (lambda0 is warm-start+MBD, which should roughly match the
   prior's standalone rate).

Exit code is non-zero if any FAIL fires, so it can gate a queue script:
``python check_unified_sanity.py <root> --ref can=0.6 --ref square=0.5``.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

MIN_N = 6  # don't flag all-zero on tiny smoke shards below this


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _task_of(d: dict) -> str:
    p = d.get("params", {}) or {}
    return str(p.get("task") or d.get("task") or "?")


def _rollout_secs(d: dict) -> list[float]:
    out = []
    for r in d.get("per_rollout", []) or []:
        v = r.get("total_elapsed_s") or r.get("plan_elapsed_s")
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def audit(root: Path, refs: dict[str, float], margin: float) -> int:
    files = sorted(root.rglob("unified_*.json"))
    files = [f for f in files if "summary" not in f.name]
    if not files:
        print(f"[sanity] no unified_*.json under {root}")
        return 0

    rows = []
    fails = 0
    suspects = 0
    for f in files:
        d = _load(f)
        if d is None:
            continue
        n = int(d.get("n_total", 0) or 0)
        sr = d.get("success_rate")
        sr = float(sr) if isinstance(sr, (int, float)) else None
        lam = d.get("lambda_prior")
        r = int(d.get("mpc_replans", 0) or 0)
        params = d.get("params", {}) or {}
        K = int(params.get("K", 0) or 0)
        task = _task_of(d)
        secs = _rollout_secs(d)
        med_s = statistics.median(secs) if secs else float("nan")

        flag = ""
        if r >= 1 and K > 0 and sr == 0.0 and n >= MIN_N:
            flag = "SUSPECT all-zero chunked-MPC (possible nw=1 corruption)"
            suspects += 1
        if (
            lam is not None and float(lam) == 0.0
            and task in refs and sr is not None
            and sr < refs[task] - margin
        ):
            flag = (f"FAIL lambda0 {sr:.0%} << standalone ref "
                    f"{refs[task]:.0%} (margin {margin:.0%})")
            fails += 1
        rows.append((f.parent.name, task, lam, r, K, n, sr, med_s, flag))

    # Report.
    print(f"[sanity] root={root}  cells={len(rows)}")
    print(f"{'cell':<42} {'task':<7} {'lam':>5} {'R':>3} {'K':>3} "
          f"{'n':>3} {'succ':>6} {'med_s':>8}  flag")
    for cell, task, lam, r, K, n, sr, med_s, flag in rows:
        srs = f"{sr*100:4.0f}%" if sr is not None else "   ?"
        lams = f"{lam}" if lam is not None else "?"
        print(f"{cell[:42]:<42} {task:<7} {lams:>5} {r:>3} {K:>3} "
              f"{n:>3} {srs:>6} {med_s:8.0f}  {flag}")

    print(f"\n[sanity] FAIL={fails}  SUSPECT={suspects}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path, help="results root to audit")
    ap.add_argument(
        "--ref", action="append", default=[], metavar="task=rate",
        help="standalone reference success rate for a task, e.g. "
             "--ref can=0.66 --ref square=0.53. lambda0 cells below "
             "(ref - margin) are flagged FAIL.",
    )
    ap.add_argument("--margin", type=float, default=0.20)
    args = ap.parse_args()
    refs: dict[str, float] = {}
    for item in args.ref:
        k, _, v = item.partition("=")
        refs[k.strip()] = float(v)
    return audit(args.root, refs, float(args.margin))


if __name__ == "__main__":
    sys.exit(main())
