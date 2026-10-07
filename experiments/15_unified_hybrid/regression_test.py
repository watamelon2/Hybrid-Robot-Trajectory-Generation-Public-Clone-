"""Phase-1 acceptance gate for the unified hybrid driver.

Runs the unified driver with two flag combinations and asserts each
reproduces a published headline number to within ``+/- TOLERANCE``
successes out of n=60. If either fails, the consolidation has a
semantic bug and Phase 1 is NOT complete.

References (with the new [256, 256] BC, post 2026-05-14 re-run):

    Cell                                                target
    ---------------------------------------------------------
    exp 07 lambda=5 single-shot (--mpc-replans=0)       20/60 = 33.3 %
    exp 10 K=10 R=2 warm-start-only (--lambda-prior=0)  11/60 = 18.3 %

The latter uses ``--lambda-prior=0`` to switch off the score-bridge,
making the unified driver behave like exp 05/10 (warm-start-only
MPC). The former uses ``--mpc-replans=0`` to switch off MPC,
making the unified driver behave like exp 07 (score-bridge single-
shot). Together they pin both legacy code paths.

Modes
-----
``--smoke``: tiny config (n=2 per cell), ~3 min total. Verifies
wiring only; does NOT enforce the +/- TOLERANCE bar. Use this
before committing or to spot-check on a new machine.

(default): n=60 per cell, ~3 h total. The actual Phase-1
acceptance gate. Run overnight or after other long-running tasks
finish (CPU contention will slow both).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
_RESULTS_DIR = _THIS_DIR / "results"

# Acceptance bar: +/- this many successes out of n=60.
TOLERANCE = 1


# Cells to regress against. Each entry: (label, flags, target_succ, n_total, target_pct).
CELLS = [
    {
        "label": "exp07_lambda5_singleshot",
        "flags": [
            "--prior=bc",
            "--lambda-prior=5",
            "--mpc-replans=0",
            "--K=10", "--M=16",
            "--sigma-max=0.25",
            "--lambda_=0.5",
            "--output-name-suffix=_regress",
        ],
        "target_succ": 20,
        "n_total": 60,
        "reference": "exp 07 lambda=5 single-shot, BC prior",
    },
    {
        "label": "exp10_K10_R2_warmstart",
        "flags": [
            "--prior=bc",
            "--lambda-prior=0",
            "--mpc-replans=2",
            "--K=10", "--M=16",
            "--sigma-max=0.25",
            "--lambda_=0.5",
            "--output-name-suffix=_regress",
        ],
        "target_succ": 11,
        "n_total": 60,
        "reference": "exp 10 K=10 R=2, BC prior, warm-start-only (lambda_prior=0)",
    },
]


def _flag_value(flags: list[str], key: str, default: str) -> str:
    """Return the value of --key=value in a flag list."""
    prefix = key + "="
    for f in flags:
        if f.startswith(prefix):
            return f[len(prefix):]
    return default


def _output_path(cell: dict, smoke: bool) -> Path:
    """Predict the output JSON path that run_can.py will write to."""
    K = _flag_value(cell["flags"], "--K", "10")
    R = _flag_value(cell["flags"], "--mpc-replans", "0")
    lam_str = _flag_value(cell["flags"], "--lambda-prior", "0")
    # run_can.py formats lam with str(float(lam)); reproduce that.
    lam_fmt = str(float(lam_str))
    suffix = _flag_value(cell["flags"], "--output-name-suffix", "")
    return _RESULTS_DIR / (
        f"unified_bc_lam-{lam_fmt}_K-{K}_R-{R}{suffix}.json"
    )


def run_cell(cell: dict, *, smoke: bool, n_demos: int, n_rollouts: int,
             n_workers: int = 1) -> dict:
    """Invoke run_can.py with the cell's flags, parse the output JSON."""
    cmd = [
        sys.executable, "-m", "experiments.15_unified_hybrid.run_can",
        f"--n-demos={n_demos}",
        f"--n-rollouts={n_rollouts}",
        "--seed=0",
        f"--n-workers={n_workers}",
        "--no-live-view" if False else "",  # placeholder; flag not in unified driver
        *cell["flags"],
    ]
    cmd = [c for c in cmd if c]
    print(f"\n[regression_test] running {cell['label']}: {' '.join(cmd)}")
    env = {"PYTHONPATH": str(_REPO_ROOT)}
    import os
    env.update(os.environ)
    p = subprocess.run(
        cmd, cwd=str(_REPO_ROOT), env=env,
        capture_output=False, check=False,
    )
    if p.returncode != 0:
        return {"error": f"exit code {p.returncode}", "cell": cell}

    out_path = _output_path(cell, smoke=smoke)
    if not out_path.exists():
        return {"error": f"output not found at {out_path}", "cell": cell}
    return json.loads(out_path.read_text())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--smoke", action="store_true",
        help="Run with n_demos=2, n_rollouts=1 (~3 min). Verifies "
             "wiring only, does NOT enforce the +/- TOLERANCE bar.",
    )
    p.add_argument(
        "--n-workers", type=int, default=1,
        help="Pass-through to run_can's --n-workers. Default 1 "
             "(single-threaded MBD evaluation). On a 16-core M-series "
             "machine, 4-6 is the sweet spot.",
    )
    args = p.parse_args()

    if args.smoke:
        n_demos, n_rollouts = 2, 1
        print("[regression_test] SMOKE MODE -- wiring check only")
    else:
        n_demos, n_rollouts = 20, 3  # n=60 per cell
        print(
            "[regression_test] FULL MODE -- enforces +/- "
            f"{TOLERANCE}/60 against published references"
        )

    all_pass = True
    results: list[dict] = []
    for cell in CELLS:
        out = run_cell(cell, smoke=args.smoke,
                       n_demos=n_demos, n_rollouts=n_rollouts,
                       n_workers=args.n_workers)
        results.append({"cell": cell, "result": out})

        if "error" in out:
            print(
                f"[regression_test] {cell['label']}: ERROR -- "
                f"{out['error']}"
            )
            all_pass = False
            continue

        succ = out["n_success"]
        n = out["n_total"]
        pct = 100 * succ / max(n, 1)

        if args.smoke:
            ok = isinstance(succ, int) and 0 <= succ <= n
            verdict = "OK (smoke)" if ok else "FAIL (smoke)"
            print(
                f"[regression_test] {cell['label']}: "
                f"{succ}/{n} = {pct:.1f} %  -- {verdict}"
            )
            if not ok:
                all_pass = False
        else:
            target = cell["target_succ"]
            diff = abs(succ - target)
            ok = diff <= TOLERANCE
            verdict = f"PASS (|{succ}-{target}|={diff}<={TOLERANCE})" if ok else \
                      f"FAIL (|{succ}-{target}|={diff}>{TOLERANCE})"
            print(
                f"[regression_test] {cell['label']}: "
                f"{succ}/{n} = {pct:.1f} %  "
                f"vs target {target}/{n} "
                f"({cell['reference']})  -- {verdict}"
            )
            if not ok:
                all_pass = False

    # Write a summary JSON.
    summary = {
        "mode": "smoke" if args.smoke else "full",
        "tolerance_successes": TOLERANCE,
        "all_pass": all_pass,
        "cells": [
            {
                "label": r["cell"]["label"],
                "target_succ": r["cell"].get("target_succ"),
                "n_total": r["cell"].get("n_total"),
                "reference": r["cell"].get("reference"),
                "observed": (
                    None if "error" in r["result"]
                    else r["result"]["n_success"]
                ),
                "error": r["result"].get("error"),
            }
            for r in results
        ],
    }
    summary_path = _RESULTS_DIR / (
        "regression_summary"
        + ("_smoke" if args.smoke else "")
        + ".json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(
        f"\n[regression_test] summary -> {summary_path}  "
        f"{'all_pass' if all_pass else 'FAILED'}"
    )

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
