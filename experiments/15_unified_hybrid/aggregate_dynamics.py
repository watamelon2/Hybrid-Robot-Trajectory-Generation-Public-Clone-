#!/usr/bin/env python3
"""Aggregate slate_dynamics_2026_05_26 12 cells into AGGREGATE.json.

Schema per cell: (mass_scale, prior, lambda_prior) -> {n_total, n_success,
success_rate, wilson_lo, wilson_hi, wall_s}.

Wilson 95% interval per cell. Pooled stats per method across masses.
"""
import json
import math
from pathlib import Path

DYN_DIR = Path("/private/tmp/HRTG-dynamics-tmp/experiments/15_unified_hybrid/results/slate_dynamics_2026_05_26")
MASSES = [0.70, 1.00, 1.30, 1.50]
SLUG = {0.70: "m070", 1.00: "m100", 1.30: "m130", 1.50: "m150"}


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Two-sided Wilson 95% CI for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> None:
    cells: list[dict] = []
    for m in MASSES:
        slug = SLUG[m]
        for prior, tag in [("demo_1nn", "demo1nn"), ("bc", "bc")]:
            summary_path = DYN_DIR / f"unified_summary_{prior}_K-10_R-1_{tag}_{slug}.json"
            if not summary_path.exists():
                print(f"MISSING: {summary_path}")
                continue
            with open(summary_path) as f:
                summary = json.load(f)
            for entry in summary["per_lambda"]:
                k = entry["n_success"]
                n = entry["n_total"]
                lo, hi = wilson_ci(k, n)
                cell = {
                    "mass_scale": m,
                    "prior": prior,
                    "lambda_prior": entry["lambda_prior"],
                    "n_success": k,
                    "n_total": n,
                    "success_rate": entry["success_rate"],
                    "wilson_lo": lo,
                    "wilson_hi": hi,
                    "wall_s": entry["wall_s"],
                }
                cells.append(cell)

    # Tag each cell with a method label
    for c in cells:
        if c["prior"] == "demo_1nn" and c["lambda_prior"] == 0.0:
            c["method"] = "pure_mbd"
        elif c["prior"] == "demo_1nn" and c["lambda_prior"] == 1.0:
            c["method"] = "hybrid_demo1nn"
        elif c["prior"] == "bc" and c["lambda_prior"] == 5.0:
            c["method"] = "bc_bridge"
        else:
            c["method"] = "UNKNOWN"

    # Pooled stats per method (sum across masses)
    pooled: dict = {}
    for method in ("hybrid_demo1nn", "pure_mbd", "bc_bridge"):
        k = sum(c["n_success"] for c in cells if c["method"] == method)
        n = sum(c["n_total"] for c in cells if c["method"] == method)
        if n == 0:
            continue
        lo, hi = wilson_ci(k, n)
        pooled[method] = {
            "n_success": k,
            "n_total": n,
            "success_rate": k / n,
            "wilson_lo": lo,
            "wilson_hi": hi,
        }

    out = {
        "experiment": "15_unified_hybrid/slate_dynamics_2026_05_26",
        "purpose": "D1 mass-ladder adaptation experiment on Can",
        "spec": "docs/adaptation/02_dynamics_spec.md",
        "scope": "Single canonical lambda per method (12 cells = 4 masses x 3 methods)",
        "canonical_mass": 1.00,
        "n_per_cell": 60,
        "total_trials": sum(c["n_total"] for c in cells),
        "K": 10,
        "R": 1,
        "sigma_max": 0.25,
        "methods": {
            "hybrid_demo1nn": "demo_1nn prior with lambda=1 (Slate E pooled peak)",
            "pure_mbd": "demo_1nn prior with lambda=0 (effectively zero prior; BC ckpt unused)",
            "bc_bridge": "bc prior with lambda=5 (Slate E E2 BC peak)",
        },
        "cells": cells,
        "pooled_by_method": pooled,
        "ci_method": "Wilson 95% (z=1.96)",
    }

    with open(DYN_DIR / "AGGREGATE.json", "w") as f:
        json.dump(out, f, indent=2)

    # Print human-readable table
    print(f"=== {len(cells)} cells, {out['total_trials']} total trials ===\n")
    print(f"{'mass':>6} | {'method':<16} | {'lambda':>7} | {'k/n':>7} | {'succ':>6} | {'95% CI':<18} | wall (min)")
    print("-" * 90)
    for m in MASSES:
        for method in ("hybrid_demo1nn", "pure_mbd", "bc_bridge"):
            for c in cells:
                if c["mass_scale"] == m and c["method"] == method:
                    succ = f"{c['success_rate']*100:5.1f}%"
                    ci = f"[{c['wilson_lo']*100:4.1f}, {c['wilson_hi']*100:5.1f}]"
                    kn = f"{c['n_success']}/{c['n_total']}"
                    wall_min = c["wall_s"] / 60
                    print(f"{m:6.2f} | {method:<16} | {c['lambda_prior']:7.1f} | {kn:>7} | {succ:>6} | {ci:<18} | {wall_min:6.1f}")

    print()
    print("=== Pooled across masses ===")
    print(f"{'method':<16} | {'k/n':>9} | {'succ':>6} | {'95% CI':<18}")
    print("-" * 60)
    for method, p in pooled.items():
        kn = f"{p['n_success']}/{p['n_total']}"
        succ = f"{p['success_rate']*100:5.1f}%"
        ci = f"[{p['wilson_lo']*100:4.1f}, {p['wilson_hi']*100:5.1f}]"
        print(f"{method:<16} | {kn:>9} | {succ:>6} | {ci:<18}")

    print(f"\nWrote {DYN_DIR / 'AGGREGATE.json'}")


if __name__ == "__main__":
    main()
