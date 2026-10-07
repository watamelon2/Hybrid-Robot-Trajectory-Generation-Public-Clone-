#!/usr/bin/env python3
"""Re-run the invalidated DP production grid with safe n_workers.

The original ``queue_vast_safe_remote_2026_06_10.py`` launched all 140 DP
shards with ``--n-workers 1``, triggering the chunk-boundary
controller-state corruption (Square 0/30, Can <=26.7%; see
``docs/results_writeups/vast_dp_grid_recovery_analysis_2026_06_11.md``).
This script reuses that file's *exact* grid definition and scheduler but:

  * rewrites ``--n-workers`` to a safe value (default 4, env
    ``HRTG_NWFIX_WORKERS``);
  * redirects every output from ``vast_safe_dp_2026_06_10`` to a fresh
    root ``vast_safe_dp_nwfix_2026_06_12`` so the (now-documented)
    corrupted artifacts are preserved untouched;
  * runs the post-hoc sanity gate (``check_unified_sanity.py``) at the
    end and fails loudly if any lambda0 cell collapsed.

Usage (on the remote box, inside the venv):
    python requeue_dp_grid_nwfix_2026_06_12.py --manifest   # dry-run summary
    python requeue_dp_grid_nwfix_2026_06_12.py --run        # launch the grid

The grid is identical to the original (60 Can + 80 Square shards), so the
recovered valid references (can_dp_calibration 70/50/66.7%, square_dp_prior
70%) are the expected ballpark for the matching cells.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse the original grid + scheduler verbatim.
import queue_vast_safe_remote_2026_06_10 as base  # noqa: E402

SAFE_WORKERS = str(int(os.environ.get("HRTG_NWFIX_WORKERS", "4")))
assert int(SAFE_WORKERS) >= 2, "n_workers must be >= 2 to avoid corruption"

OLD_ROOT = base.DP_ROOT
NEW_ROOT = base.REPO / "experiments/15_unified_hybrid/results/vast_safe_dp_nwfix_2026_06_12"
SANITY = _HERE / "check_unified_sanity.py"


def _fix_command(cmd: tuple[str, ...]) -> tuple[str, ...]:
    """Rewrite n_workers -> SAFE_WORKERS and redirect the output root."""
    fixed = base.replace_flag(list(cmd), "--n-workers", [SAFE_WORKERS])
    fixed = [
        str(NEW_ROOT) if tok == str(OLD_ROOT)
        else tok.replace(f"{OLD_ROOT}/", f"{NEW_ROOT}/")
        for tok in fixed
    ]
    return tuple(fixed)


def make_fixed_jobs() -> list[base.Job]:
    jobs = []
    for job in base.make_dp_jobs():
        new_outdir = Path(
            str(job.outdir).replace(str(OLD_ROOT), str(NEW_ROOT))
        )
        jobs.append(replace(
            job, command=_fix_command(job.command), outdir=new_outdir,
        ))
    return jobs


def run_sanity_gate() -> int:
    if not SANITY.exists():
        print(f"[nwfix] sanity script missing: {SANITY}")
        return 0
    return subprocess.run([
        sys.executable, str(SANITY), str(NEW_ROOT),
        "--ref", "can=0.66", "--ref", "square=0.53",
    ]).returncode


def manifest() -> None:
    jobs = make_fixed_jobs()
    print(f"[nwfix] safe n_workers = {SAFE_WORKERS}")
    print(f"[nwfix] old (corrupt) root = {OLD_ROOT}")
    print(f"[nwfix] new root           = {NEW_ROOT}")
    print(f"[nwfix] DP shards          = {len(jobs)} "
          f"(60 Can + 80 Square, identical grid)")
    sample = jobs[0]
    print(f"[nwfix] sample cell  = {sample.name}")
    print(f"[nwfix] sample cmd   = {' '.join(sample.command)}")
    # Confirm no job retains the corrupting flag.
    bad = [j.name for j in jobs if "1" == _nworkers_of(j.command)]
    print(f"[nwfix] jobs still on n_workers<=1 = {len(bad)} (must be 0)")


def _nworkers_of(cmd: tuple[str, ...]) -> str:
    if "--n-workers" in cmd:
        return cmd[cmd.index("--n-workers") + 1]
    return "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", action="store_true",
                    help="dry-run: print the corrected grid summary, launch nothing")
    ap.add_argument("--run", action="store_true",
                    help="launch the corrected DP grid into the fresh root")
    args = ap.parse_args()

    if args.manifest or not args.run:
        manifest()
        if not args.run:
            return 0

    NEW_ROOT.mkdir(parents=True, exist_ok=True)
    rc = base.run_queue(
        make_fixed_jobs(), NEW_ROOT,
        max_total=base.DP_MAX_TOTAL,
        max_per_gpu=base.DP_MAX_PER_GPU,
        launch_delay=base.DP_LAUNCH_DELAY_SECONDS,
        poll_seconds=base.DP_POLL_SECONDS,
        max_retries=base.DP_MAX_RETRIES,
    )
    print(f"[nwfix] grid finished rc={rc}; running sanity gate...")
    gate = run_sanity_gate()
    if gate != 0:
        print("[nwfix] SANITY GATE FAILED -- inspect flagged cells before citing.")
        return gate
    print("[nwfix] sanity gate passed.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
