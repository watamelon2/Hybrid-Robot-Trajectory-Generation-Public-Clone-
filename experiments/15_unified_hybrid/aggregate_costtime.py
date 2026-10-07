#!/usr/bin/env python
"""Aggregate slate_costtime per-cell JSONs into a 3-curve summary.

Reads the 12 per-cell JSONs in
  experiments/15_unified_hybrid/results/slate_costtime_2026_05_25/
parses (method, lambda, dx) from each, computes Wilson 95% CIs,
and dumps:

  - AGGREGATE.json   structured per-cell results + table
  - aggregate.csv    flat CSV for the plot script / spreadsheet
  - cost_time.png    3-curve plot (hybrid / pure-MBD / BC vs dx)
  - cost_time.pdf    same, paper-ready

Method recovery rule:
  prior == "bc"        -> BC bridge
  prior == "demo_1nn"  -> hybrid (lambda_prior == 1.0) or
                          pure-MBD (lambda_prior == 0.0)

Companion to docs/adaptation/01_cost_time_spec.md sec 5
(headline plot + Wilson CIs + RESULT.md).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "slate_costtime_2026_05_25"
)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson 95% CI for a binomial rate k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (
        z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    )
    return (max(0.0, centre - half), min(1.0, centre + half))


def classify(record: dict) -> tuple[str, float]:
    """Return (method_label, dx_meters) from a per-cell JSON record."""
    prior = record["prior"]
    lam = float(record["lambda_prior"])
    dx = float(record["params"]["goal_perturbation_xyz"][0])
    if prior == "bc":
        return ("BC", dx)
    if prior == "demo_1nn":
        if lam == 1.0:
            return ("hybrid", dx)
        if lam == 0.0:
            return ("pure_MBD", dx)
    raise ValueError(
        f"Unrecognised method: prior={prior} lambda={lam}"
    )


def main() -> int:
    cells = []
    for path in sorted(RESULTS_DIR.glob("unified_*.json")):
        # Skip the summary files; they aggregate over lambdas and
        # do not carry per-lambda success counts.
        if "_summary_" in path.name:
            continue
        with path.open() as fh:
            record = json.load(fh)
        method, dx = classify(record)
        k = int(record["n_success"])
        n = int(record["n_total"])
        lo, hi = wilson_ci(k, n)
        cells.append(
            {
                "method": method,
                "dx_m": dx,
                "k": k,
                "n": n,
                "rate": k / n if n else 0.0,
                "ci_lo": lo,
                "ci_hi": hi,
                "wall_s": float(record.get("wall_s", 0.0)),
                "file": path.name,
            }
        )

    # Sort by (method, dx) for stable tabulation.
    method_order = {"hybrid": 0, "BC": 1, "pure_MBD": 2}
    cells.sort(key=lambda c: (method_order[c["method"]], c["dx_m"]))

    # Console summary.
    print("=" * 78)
    print("slate_costtime aggregate")
    print("=" * 78)
    print(
        f"{'method':<10} {'dx (m)':<8} {'succ':<8} "
        f"{'rate':<8} {'95% Wilson CI':<18} {'wall (min)':<10}"
    )
    print("-" * 78)
    for c in cells:
        ci = f"[{c['ci_lo']*100:5.1f}, {c['ci_hi']*100:5.1f}]"
        print(
            f"{c['method']:<10} {c['dx_m']:<8.2f} "
            f"{c['k']:>3}/{c['n']:<3} {c['rate']*100:6.1f}%  "
            f"{ci:<18} {c['wall_s']/60:6.1f}"
        )
    print("=" * 78)

    # Per-method, sorted-by-dx pivot for plotting.
    methods = ("hybrid", "pure_MBD", "BC")
    pivot = {m: [c for c in cells if c["method"] == m] for m in methods}
    for m in methods:
        pivot[m].sort(key=lambda c: c["dx_m"])

    # AGGREGATE.json.
    aggregate = {
        "slate": "slate_costtime_2026_05_25",
        "spec": "docs/adaptation/01_cost_time_spec.md",
        "task": "can",
        "operating_point": {
            "K": 10, "M": 16, "H": 120,
            "sigma_max": 0.25, "sigma_min": 0.05,
            "lambda_hybrid": 1.0,
            "lambda_bc_bridge": 5.0,
            "lambda_pure_mbd": 0.0,
            "n_demos": 20, "n_rollouts": 3, "seed": 0,
        },
        "perturbation_axis_m": [0.00, 0.05, 0.10, 0.20],
        "cells": cells,
        "by_method": pivot,
    }
    with (RESULTS_DIR / "AGGREGATE.json").open("w") as fh:
        json.dump(aggregate, fh, indent=2)
    print(f"wrote {RESULTS_DIR / 'AGGREGATE.json'}")

    # aggregate.csv.
    with (RESULTS_DIR / "aggregate.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "method", "dx_m", "k", "n", "rate",
                "ci_lo", "ci_hi", "wall_s", "file",
            ],
        )
        writer.writeheader()
        for c in cells:
            writer.writerow(c)
    print(f"wrote {RESULTS_DIR / 'aggregate.csv'}")

    # Plot: 3 curves, hybrid / pure-MBD / BC vs dx, with Wilson CIs.
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    style = {
        "hybrid":   ("#1f77b4", "o", "hybrid  (demo_1nn, $\\lambda=1$)"),
        "BC":       ("#ff7f0e", "s", "BC bridge  ($\\lambda=5$)"),
        "pure_MBD": ("#2ca02c", "^", "pure MBD  ($\\lambda=0$)"),
    }
    for m in methods:
        rows = pivot[m]
        dxs = [r["dx_m"] for r in rows]
        rates = [r["rate"] * 100 for r in rows]
        lo = [r["rate"] * 100 - r["ci_lo"] * 100 for r in rows]
        hi = [r["ci_hi"] * 100 - r["rate"] * 100 for r in rows]
        color, marker, label = style[m]
        ax.errorbar(
            dxs, rates, yerr=[lo, hi], fmt=f"-{marker}",
            color=color, capsize=3, lw=1.5, ms=7, label=label,
        )

    ax.set_xlabel("Goal-pose perturbation $\\Delta x$ (m, +x only)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title(
        "Slate CostTime: cost-perturbation adaptation on Can\n"
        "20 demos $\\times$ 3 rollouts = 60 evals/cell, "
        "single seed=0  (Wilson 95% CIs)"
    )
    ax.set_ylim(-5, 100)
    ax.set_xlim(-0.01, 0.22)
    ax.set_xticks([0.00, 0.05, 0.10, 0.20])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "cost_time.png", dpi=150)
    fig.savefig(RESULTS_DIR / "cost_time.pdf")
    print(f"wrote {RESULTS_DIR / 'cost_time.png'}")
    print(f"wrote {RESULTS_DIR / 'cost_time.pdf'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
