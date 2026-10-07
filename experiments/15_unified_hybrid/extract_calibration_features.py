#!/usr/bin/env python3
"""Per-rollout feature table for the test-time-guidance calibration study.

Reads any number of ``unified_*.json`` result files (across one or more
results roots) and emits a tidy JSONL/CSV with one row per rollout, where
each row carries the *guidance-calibration signals* that are already
logged by ``unified_loop.py`` -- no new compute, no re-running.  This is
step 2 of ``docs/todos/next_direction_2026_06_11.md`` ("re-analyze
existing regimes"): regress the hybrid lift (guided vs lambda0) on these
signals across the valid prior ladder.

Per-rollout signals extracted from the chunk logs:
  * ``success`` / ``final_reward``       -- the outcome.
  * ``n_chunks`` / ``total_elapsed_s``   -- run shape (and the corruption
    slowdown tell).
  * ``used_score_fn``                    -- was the prior pull active.
  * ``mean_cost_drop``                   -- mean over chunks of
    (init_best_cost - final_best_cost): how much MBD improved the plan.
  * ``mean_ess`` / ``min_ess``           -- effective sample size of the
    MBD softmax (low ESS => guidance collapsing onto one rollout, the
    over-optimization tell).
  * ``mean_final_best_cost``             -- achieved cost level.
  * ``mean_gate_frac`` / ``mean_ldem_minus_lmod`` -- the §6.3 gate
    diagnostic (NaN unless --log-gate-diagnostic was on).

Signals that still need NEW logging (see the companion design note
``docs/results_writeups/calibration_diagnostics_spec_2026_06_11.md``):
kNN distance to the demo bank, kNN action disagreement, and the across-M
warm-start action variance.  Those require a small hook in
``unified_loop.py`` and are intentionally NOT faked here.

Usage:
    python extract_calibration_features.py ROOT [ROOT ...] \
        --out features.jsonl [--csv features.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

FIELDS = [
    "file", "cell", "task", "prior", "checkpoint", "lambda_prior",
    "combine_mode", "score_mode", "sigma_max", "sigma_demo",
    "mpc_replans", "K", "rollout_kind", "init_seed", "rollout_idx",
    "success", "final_reward", "n_chunks", "total_elapsed_s",
    "used_score_fn", "mean_cost_drop", "mean_final_best_cost",
    "mean_ess", "min_ess", "mean_gate_frac", "mean_ldem_minus_lmod",
]


def _ckpt_name(d: dict) -> str:
    return Path(str(d.get("prior_checkpoint", ""))).name


def _safe_mean(xs: list[float]) -> float:
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return statistics.fmean(xs) if xs else float("nan")


def _safe_min(xs: list[float]) -> float:
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return min(xs) if xs else float("nan")


def _rollout_row(d: dict, r: dict, file: Path) -> dict:
    params = d.get("params", {}) or {}
    chunks = r.get("chunks", []) or []
    cost_drops, finals, esss, gates, ldiffs = [], [], [], [], []
    used = False
    for c in chunks:
        ib, fb = c.get("init_best_cost"), c.get("final_best_cost")
        if isinstance(ib, (int, float)) and isinstance(fb, (int, float)):
            if not (math.isnan(ib) or math.isnan(fb)):
                cost_drops.append(ib - fb)
                finals.append(fb)
        used = used or bool(c.get("used_score_fn"))
        for p in c.get("iter_curve", []) or []:
            esss.append(p.get("ess"))
            gates.append(p.get("gate_frac"))
            ldiffs.append(p.get("ldem_minus_lmod"))
    return {
        "file": str(file),
        "cell": file.parent.name,
        "task": params.get("task") or d.get("task"),
        "prior": d.get("prior"),
        "checkpoint": _ckpt_name(d),
        "lambda_prior": r.get("lambda_prior", d.get("lambda_prior")),
        "combine_mode": params.get("combine_mode"),
        "score_mode": params.get("score_mode"),
        "sigma_max": params.get("sigma_max"),
        "sigma_demo": params.get("sigma_demo"),
        "mpc_replans": d.get("mpc_replans"),
        "K": params.get("K"),
        "rollout_kind": r.get("rollout_kind"),
        "init_seed": r.get("init_seed"),
        "rollout_idx": r.get("rollout_idx"),
        "success": int(bool(r.get("success"))),
        "final_reward": r.get("final_reward"),
        "n_chunks": len(chunks),
        "total_elapsed_s": r.get("total_elapsed_s") or r.get("plan_elapsed_s"),
        "used_score_fn": int(used),
        "mean_cost_drop": _safe_mean(cost_drops),
        "mean_final_best_cost": _safe_mean(finals),
        "mean_ess": _safe_mean(esss),
        "min_ess": _safe_min(esss),
        "mean_gate_frac": _safe_mean(gates),
        "mean_ldem_minus_lmod": _safe_mean(ldiffs),
    }


def collect(roots: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for root in roots:
        files = (
            [root] if root.is_file()
            else [f for f in root.rglob("unified_*.json")
                  if "summary" not in f.name]
        )
        for f in sorted(files):
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            for r in d.get("per_rollout", []) or []:
                rows.append(_rollout_row(d, r, f))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=Path("calibration_features.jsonl"))
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    rows = collect(args.roots)
    with args.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    if args.csv is not None:
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in FIELDS})
    n_succ = sum(r["success"] for r in rows)
    print(f"[features] wrote {len(rows)} rollout rows -> {args.out}"
          + (f" + {args.csv}" if args.csv else ""))
    print(f"[features] overall success = {n_succ}/{len(rows)}"
          f" = {100*n_succ/max(len(rows),1):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
