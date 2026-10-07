"""Aggregate the 6-cell demo-scaling slate (slate_demoscale_2026_05_25)
into one summary JSON + CSV + two-panel scaling-law figure.

Run from any worktree (paths are absolute to the canonical results dir
in the main repository):

    python experiments/15_unified_hybrid/aggregate_demoscale.py

Outputs land in
``Hybrid-Robot-Trajectory-Generation/experiments/15_unified_hybrid/results/slate_demoscale_2026_05_25/``:

- ``AGGREGATE.json``  -- structured per-cell + mode-coverage breakdown
- ``aggregate.csv``   -- spreadsheet-friendly version
- ``scaling_law.png`` / ``scaling_law.pdf`` -- the two-panel figure

The figure mirrors the design in
``docs/adaptation/next_steps_2026_05_25.md`` section 1.2:

- Left panel:  log2(N_b) -> success rate, Wilson 95% CI bars,
               canonical 73.8% baseline annotated, knee at N_b=20 annotated.
- Right panel: per-N_b in-bank vs out-of-bank mode-coverage decomposition.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Paths -- always write to the canonical (main-worktree) results directory.
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(
    "/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/"
    "Hybrid-Robot-Trajectory-Generation/experiments/15_unified_hybrid/"
    "results/slate_demoscale_2026_05_25"
)

BANK_SIZES = (5, 10, 20, 50, 100, 200)
N_EVAL_DEMOS = 20            # --n-demos passed to the slate
N_ROLLOUTS_PER_DEMO = 3      # --n-rollouts passed to the slate

# Canonical 4-seed pool reference from Slate E (2026-05-19)
CANONICAL_POOL_RATE = 177 / 240    # = 0.7375
CANONICAL_POOL_CI = (0.678, 0.789)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def wilson_ci(n_succ: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a binomial proportion. Returns (lo, hi)."""
    if n_total == 0:
        return (0.0, 0.0)
    p = n_succ / n_total
    denom = 1 + z * z / n_total
    centre = (p + z * z / (2 * n_total)) / denom
    half = (z * math.sqrt(p * (1 - p) / n_total + z * z / (4 * n_total ** 2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def demo_idx(per_rollout_entry: dict) -> int:
    """Parse the integer index from a 'demo_N' string."""
    s = per_rollout_entry["demo"]
    return int(s.rsplit("_", 1)[1])


# ---------------------------------------------------------------------------
# Per-cell aggregation
# ---------------------------------------------------------------------------

def aggregate_cell(json_path: Path, n_b: int) -> dict:
    """Load one cell JSON and compute the structured summary."""
    with open(json_path) as f:
        d = json.load(f)

    per_rollout = d["per_rollout"]

    # Total breakdown
    n_total = len(per_rollout)
    n_success = sum(1 for r in per_rollout if r["success"])
    rate = n_success / n_total
    lo, hi = wilson_ci(n_success, n_total)

    # In-bank vs OOB breakdown.  Eval set is demos 0..19; "in-bank" =
    # demo_idx < n_b, "out-of-bank" = demo_idx >= n_b.
    ib_total = 0
    ib_succ = 0
    oob_total = 0
    oob_succ = 0
    for r in per_rollout:
        i = demo_idx(r)
        is_success = bool(r["success"])
        if i < n_b:
            ib_total += 1
            if is_success:
                ib_succ += 1
        else:
            oob_total += 1
            if is_success:
                oob_succ += 1

    ib_rate = ib_succ / ib_total if ib_total else None
    oob_rate = oob_succ / oob_total if oob_total else None
    ib_lo, ib_hi = wilson_ci(ib_succ, ib_total) if ib_total else (None, None)
    oob_lo, oob_hi = wilson_ci(oob_succ, oob_total) if oob_total else (None, None)

    # Reward stats
    rewards = np.array([r["final_reward"] for r in per_rollout])
    mean_reward = float(rewards.mean())
    median_reward = float(np.median(rewards))

    return {
        "n_b": n_b,
        "json_file": json_path.name,
        "n_total": n_total,
        "n_success": n_success,
        "success_rate": rate,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "in_bank": {
            "n_total": ib_total,
            "n_success": ib_succ,
            "rate": ib_rate,
            "ci95_lo": ib_lo,
            "ci95_hi": ib_hi,
        },
        "out_of_bank": {
            "n_total": oob_total,
            "n_success": oob_succ,
            "rate": oob_rate,
            "ci95_lo": oob_lo,
            "ci95_hi": oob_hi,
        },
        "mean_reward": mean_reward,
        "median_reward": median_reward,
        "wall_s": d.get("wall_s"),
    }


# ---------------------------------------------------------------------------
# Aggregate JSON + CSV writers
# ---------------------------------------------------------------------------

def write_aggregate_json(cells: list[dict], out_path: Path) -> None:
    body = {
        "slate": "slate_demoscale_2026_05_25",
        "task": "can",
        "prior": "demo_1nn",
        "operating_point": {
            "lambda_prior": 1.0,
            "sigma_max": 0.25,
            "sigma_min": 0.01,
            "K": 10,
            "M": 16,
            "H": 120,
            "n_demos": N_EVAL_DEMOS,
            "n_rollouts_per_demo": N_ROLLOUTS_PER_DEMO,
            "seed": 0,
            "mpc_replans": 1,
            "mpc_chunk_steps": 60,
        },
        "canonical_pool_reference": {
            "source": "slate_e_2026_05_19",
            "n_seeds": 4,
            "n_total": 240,
            "rate": CANONICAL_POOL_RATE,
            "ci95": list(CANONICAL_POOL_CI),
        },
        "cells": cells,
    }
    with open(out_path, "w") as f:
        json.dump(body, f, indent=2)


def write_csv(cells: list[dict], out_path: Path) -> None:
    fields = [
        "n_b", "n_total", "n_success", "rate", "ci95_lo", "ci95_hi",
        "ib_n", "ib_succ", "ib_rate",
        "oob_n", "oob_succ", "oob_rate",
        "mean_reward", "wall_s",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for c in cells:
            w.writerow([
                c["n_b"], c["n_total"], c["n_success"], c["success_rate"],
                c["ci95_lo"], c["ci95_hi"],
                c["in_bank"]["n_total"], c["in_bank"]["n_success"],
                c["in_bank"]["rate"],
                c["out_of_bank"]["n_total"], c["out_of_bank"]["n_success"],
                c["out_of_bank"]["rate"],
                c["mean_reward"], c["wall_s"],
            ])


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(cells: list[dict], out_png: Path, out_pdf: Path) -> None:
    n_bs = np.array([c["n_b"] for c in cells], dtype=float)
    rates = np.array([c["success_rate"] for c in cells])
    lo = np.array([c["ci95_lo"] for c in cells])
    hi = np.array([c["ci95_hi"] for c in cells])

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    # ---------- Left panel: scaling curve --------------------------------
    err_lo = rates - lo
    err_hi = hi - rates
    ax_l.errorbar(
        n_bs, rates, yerr=[err_lo, err_hi],
        fmt="o-", color="#2E5BFF", ecolor="#2E5BFF", capsize=3,
        markersize=7, linewidth=1.8, label="demo_1nn @ canonical $\\lambda{=}1, \\sigma_{\\max}{=}0.25$",
    )

    # Canonical 4-seed pool reference (horizontal band)
    ax_l.axhspan(
        CANONICAL_POOL_CI[0], CANONICAL_POOL_CI[1],
        color="#FFB000", alpha=0.18, label="Canonical 4-seed pool 73.8% CI [67.8, 78.9]",
    )
    ax_l.axhline(CANONICAL_POOL_RATE, color="#FFB000", linestyle="--", linewidth=1.0, alpha=0.7)

    # Knee annotation at N_b = 20 (predicted: bank coverage = eval set)
    knee_idx = list(BANK_SIZES).index(20)
    ax_l.annotate(
        f"saturation knee\n$N_b{{=}}20$ = eval set size",
        xy=(20, rates[knee_idx]),
        xytext=(35, rates[knee_idx] - 0.18),
        fontsize=9, ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
    )

    ax_l.set_xscale("log")
    ax_l.set_xticks(BANK_SIZES)
    ax_l.set_xticklabels([str(n) for n in BANK_SIZES])
    ax_l.set_xlim(4, 250)
    ax_l.set_ylim(0.0, 1.0)
    ax_l.set_xlabel("Prior bank size  $N_b$  (log scale)")
    ax_l.set_ylabel("Closed-loop success rate (Wilson 95% CI)")
    ax_l.set_title("(a) Bank-coverage scaling law on PickPlaceCan")
    ax_l.grid(True, which="both", alpha=0.3)
    ax_l.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # ---------- Right panel: mode-coverage decomposition -----------------
    x = np.arange(len(BANK_SIZES))
    bar_w = 0.36

    ib_rates = []
    ib_los = []
    ib_his = []
    oob_rates = []
    oob_los = []
    oob_his = []
    for c in cells:
        ib = c["in_bank"]
        oob = c["out_of_bank"]
        ib_rates.append(ib["rate"] if ib["rate"] is not None else np.nan)
        ib_los.append(ib["ci95_lo"] if ib["ci95_lo"] is not None else np.nan)
        ib_his.append(ib["ci95_hi"] if ib["ci95_hi"] is not None else np.nan)
        if oob["n_total"] > 0:
            oob_rates.append(oob["rate"])
            oob_los.append(oob["ci95_lo"])
            oob_his.append(oob["ci95_hi"])
        else:
            oob_rates.append(np.nan)
            oob_los.append(np.nan)
            oob_his.append(np.nan)

    ib_rates = np.array(ib_rates)
    oob_rates = np.array(oob_rates)

    # Bars + per-bar Wilson CI (clip NaN where OOB doesn't exist)
    def _yerr(rates_, los_, his_):
        rates_ = np.array(rates_, dtype=float)
        los_ = np.array(los_, dtype=float)
        his_ = np.array(his_, dtype=float)
        err_lo_ = rates_ - los_
        err_hi_ = his_ - rates_
        return np.where(np.isnan(rates_), 0.0, err_lo_), np.where(np.isnan(rates_), 0.0, err_hi_)

    ib_err = _yerr(ib_rates, ib_los, ib_his)
    oob_err = _yerr(oob_rates, oob_los, oob_his)

    bar_ib = ax_r.bar(
        x - bar_w / 2, np.nan_to_num(ib_rates, nan=0.0), bar_w,
        yerr=ib_err, capsize=3, color="#2E5BFF", alpha=0.85,
        label="In-bank rate (eval demo idx $<\\,N_b$)",
    )
    bar_oob = ax_r.bar(
        x + bar_w / 2, np.nan_to_num(oob_rates, nan=0.0), bar_w,
        yerr=oob_err, capsize=3, color="#FF6B35", alpha=0.85,
        label="Out-of-bank rate (eval demo idx $\\geq N_b$)",
    )

    # Mark cells where OOB doesn't exist (full coverage)
    for i, oob in enumerate(oob_rates):
        if math.isnan(oob):
            ax_r.text(
                x[i] + bar_w / 2, 0.02, "n/a\n(full\ncoverage)",
                ha="center", va="bottom", fontsize=8, color="#666666",
            )

    ax_r.set_xticks(x)
    ax_r.set_xticklabels([str(n) for n in BANK_SIZES])
    ax_r.set_ylim(0.0, 1.0)
    ax_r.set_xlabel("Prior bank size  $N_b$")
    ax_r.set_ylabel("Closed-loop success rate (Wilson 95% CI)")
    ax_r.set_title("(b) Mode-coverage decomposition")
    ax_r.grid(True, axis="y", alpha=0.3)
    ax_r.legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.suptitle(
        "Demo-bank scaling law for the zero-training retrieval prior (demo_1nn, Can)",
        fontsize=12, y=1.02,
    )

    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    cells: list[dict] = []
    for n_b in BANK_SIZES:
        path = RESULTS_DIR / f"unified_demo_1nn_lam-1.0_K-10_R-1_Nb{n_b}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing cell JSON: {path}")
        cells.append(aggregate_cell(path, n_b))

    write_aggregate_json(cells, RESULTS_DIR / "AGGREGATE.json")
    write_csv(cells, RESULTS_DIR / "aggregate.csv")
    make_figure(
        cells,
        RESULTS_DIR / "scaling_law.png",
        RESULTS_DIR / "scaling_law.pdf",
    )

    def _fmt_pct(x: float | None, w: int = 8) -> str:
        return " " * (w - 3) + "n/a" if x is None else f"{x:>{w}.1%}"

    print("Aggregated 6 cells:")
    print(f"  {'N_b':>5}  {'rate':>7}  {'CI95':>14}  {'IB rate':>8}  {'OOB rate':>8}  {'wall_s':>8}")
    for c in cells:
        ib_r = c["in_bank"]["rate"]
        oob_r = c["out_of_bank"]["rate"]
        wall = c.get("wall_s")
        wall_s = "    n/a" if wall is None else f"{wall:>6.0f}s"
        print(
            f"  {c['n_b']:>5}  "
            f"{c['success_rate']:>6.1%}  "
            f"[{c['ci95_lo']:>4.1%},{c['ci95_hi']:>5.1%}]  "
            f"{_fmt_pct(ib_r)}  "
            f"{_fmt_pct(oob_r)}  "
            f"{wall_s}"
        )

    print()
    print(f"Wrote:")
    print(f"  {RESULTS_DIR / 'AGGREGATE.json'}")
    print(f"  {RESULTS_DIR / 'aggregate.csv'}")
    print(f"  {RESULTS_DIR / 'scaling_law.png'}")
    print(f"  {RESULTS_DIR / 'scaling_law.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
