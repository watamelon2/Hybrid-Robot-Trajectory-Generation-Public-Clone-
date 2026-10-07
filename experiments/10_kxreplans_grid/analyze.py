"""Post-hoc analysis of the K x n_replans pareto grid.

Reads the per-cell JSONs written by ``experiments/10_kxreplans_grid/run.sh``
(which are themselves produced by ``experiments/05_hybrid_method/run_can.py``)
and emits three artefacts:

  results/summary.json   -- aggregate per-cell stats, machine-readable
  results/summary.md     -- markdown table suitable for paste into a doc
  results/summary.png    -- 2-panel figure:
                              (a) success-rate heatmap over K x R
                              (b) hybrid vs BC paired bars per cell

The summary is intentionally lightweight: no MuJoCo, no env, no policy.
It only consumes the fields exp 05 already writes per rollout
(``success``, ``final_exec_reward``, ``bc_success``, ``bc_exec_reward``).

Usage:

    python experiments/10_kxreplans_grid/analyze.py
    # or with explicit input dir:
    python experiments/10_kxreplans_grid/analyze.py \
        --results experiments/10_kxreplans_grid/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Defer matplotlib import until after argparse so --help works on hosts
# without a display backend.
_THIS_DIR = Path(__file__).resolve().parent
# Data archived 2026-05-15 into results/_archive/ since exp 10 was
# superseded by experiments/15_unified_hybrid/ (see docs/results.md).
_DEFAULT_RESULTS = _THIS_DIR / "results" / "_archive"


def _load_cells(results_dir: Path) -> list[dict[str, Any]]:
    """Parse every K{K}_R{R}.json in `results_dir` into a flat row."""
    rows = []
    for f in sorted(results_dir.glob("K*_R*.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            print(f"  [warn] skipping {f.name}: {e}", file=sys.stderr)
            continue
        # Parse K and R out of the filename for safety (some cells may
        # not have these as top-level fields in the JSON).
        name = f.stem  # e.g. "K10_R2"
        try:
            k_str, r_str = name.split("_")
            K = int(k_str.lstrip("K"))
            R = int(r_str.lstrip("R"))
        except (ValueError, IndexError):
            print(f"  [warn] could not parse K,R from {name}", file=sys.stderr)
            continue
        params = d.get("params", {}) or {}
        rows.append({
            "cell":            name,
            "K":               K,
            "R":               R,
            "n_total":         d.get("n_total", 0),
            "n_success":       d.get("n_success", 0),
            "success_rate":    d.get("success_rate", 0.0),
            "bc_success":      d.get("bc_success_total", 0),
            "bc_success_rate": d.get("bc_success_rate", 0.0),
            "delta_vs_bc":     d.get("success_rate", 0.0)
                               - d.get("bc_success_rate", 0.0),
            "total_wall_min":  d.get("total_wall_s", 0.0) / 60.0,
            "M":               params.get("M"),
            "sigma_max":       params.get("sigma_max"),
            "warm_start":      d.get("warm_start"),
            "seed":            d.get("seed"),
            "source":          str(f),
        })
        # Compute MBD>BC fraction over per_rollout list.
        rolls = d.get("per_rollout") or []
        if rolls:
            mbd_beats_bc = sum(
                1 for r in rolls
                if r.get("final_exec_reward", 0.0)
                   > r.get("bc_exec_reward", 0.0)
            )
            rows[-1]["mbd_beats_bc_frac"] = mbd_beats_bc / len(rolls)
        else:
            rows[-1]["mbd_beats_bc_frac"] = None
    return rows


def _write_summary_json(rows: list[dict[str, Any]], out: Path) -> None:
    out.write_text(json.dumps(rows, indent=2))
    print(f"  wrote {out}")


def _write_summary_md(rows: list[dict[str, Any]], out: Path) -> None:
    lines = []
    lines.append("# K x n_replans pareto grid -- summary\n")
    lines.append(
        f"{len(rows)} cells, all at fixed seed=0, n_demos=20, "
        f"n_rollouts=3 (so n_total=60 per cell), M=16, H=120, "
        f"sigma_max=0.25, warm_start=BC.\n"
    )
    # Sort rows for the markdown table: by K, then R.
    rows = sorted(rows, key=lambda r: (r["K"], r["R"]))
    hdr = ("| cell  | K  | R | hybrid succ | BC succ | "
           "delta  | MBD>BC | wall (min) |")
    sep = ("|-------|----|---|-------------|---------|"
           "--------|--------|------------|")
    lines.append(hdr)
    lines.append(sep)
    for r in rows:
        mbd_pct = (f"{r['mbd_beats_bc_frac']*100:.0f}%"
                   if r['mbd_beats_bc_frac'] is not None else "?")
        lines.append(
            f"| {r['cell']:<5} | {r['K']:<2} | {r['R']:<1} "
            f"| {r['success_rate']*100:>5.1f}% ({r['n_success']:>2}/{r['n_total']}) "
            f"| {r['bc_success_rate']*100:>5.1f}% ({r['bc_success']:>2}/{r['n_total']}) "
            f"| {r['delta_vs_bc']*100:>+5.1f}% "
            f"| {mbd_pct:>6} "
            f"| {r['total_wall_min']:>6.1f} |"
        )
    lines.append("")
    # Headline summary
    best = max(rows, key=lambda r: r["success_rate"])
    worst = min(rows, key=lambda r: r["success_rate"])
    avg_delta = sum(r["delta_vs_bc"] for r in rows) / len(rows)
    n_beat_bc = sum(1 for r in rows if r["delta_vs_bc"] > 0)
    lines.append("## Headline\n")
    lines.append(
        f"- **Best cell**: {best['cell']} -- "
        f"{best['success_rate']*100:.1f}% success "
        f"(BC: {best['bc_success_rate']*100:.1f}%, "
        f"delta: {best['delta_vs_bc']*100:+.1f}%)"
    )
    lines.append(
        f"- **Worst cell**: {worst['cell']} -- "
        f"{worst['success_rate']*100:.1f}% success "
        f"(BC: {worst['bc_success_rate']*100:.1f}%, "
        f"delta: {worst['delta_vs_bc']*100:+.1f}%)"
    )
    lines.append(
        f"- **Cells where hybrid beats BC**: {n_beat_bc}/{len(rows)}"
    )
    lines.append(
        f"- **Mean delta (hybrid - BC)**: {avg_delta*100:+.1f} pp"
    )
    lines.append("")
    lines.append(
        "## Reading the result\n\n"
        "BC's success rate varies cell-to-cell even with fixed seed=0\n"
        "because the BC baseline rollout is re-run inside each MBD\n"
        "rollout from a slightly different reset state (the planner's\n"
        "first plan introduces stochasticity even before execution).\n"
        "So 'BC succ' here is a per-cell sample of BC on 60 episodes,\n"
        "not the canonical BC baseline number from exp 03.\n\n"
        "The cells where hybrid clearly wins are R>=1 (multi-shot MPC)\n"
        "with moderate K -- specifically K5/R1, K10/R2, K20/R2.\n"
        "R=0 (single-shot) consistently underperforms BC, which\n"
        "matches the open-loop / compounding-error story from the\n"
        "exp 04 baseline.\n"
    )

    out.write_text("\n".join(lines))
    print(f"  wrote {out}")


def _write_summary_png(rows: list[dict[str, Any]], out: Path) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ks = sorted({r["K"] for r in rows})
    Rs = sorted({r["R"] for r in rows})
    by_kr = {(r["K"], r["R"]): r for r in rows}

    # ---- panel (a): success-rate heatmap K x R ----
    grid = np.full((len(Ks), len(Rs)), np.nan)
    for i, K in enumerate(Ks):
        for j, R in enumerate(Rs):
            if (K, R) in by_kr:
                grid[i, j] = by_kr[(K, R)]["success_rate"] * 100.0

    fig, (ax_h, ax_b) = plt.subplots(1, 2, figsize=(13, 5.5))
    im = ax_h.imshow(grid, cmap="viridis", vmin=0, vmax=max(10.0, np.nanmax(grid)))
    ax_h.set_xticks(range(len(Rs)), [f"R={R}" for R in Rs])
    ax_h.set_yticks(range(len(Ks)), [f"K={K}" for K in Ks])
    ax_h.set_xlabel("n_replans (R)")
    ax_h.set_ylabel("MBD iterations (K)")
    ax_h.set_title("Hybrid success rate (%)\nK x n_replans pareto grid")
    for i in range(len(Ks)):
        for j in range(len(Rs)):
            val = grid[i, j]
            if not np.isnan(val):
                ax_h.text(j, i, f"{val:.1f}%", ha="center", va="center",
                          color="white" if val < 5 else "black", fontsize=11)
    fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04, label="success rate (%)")

    # ---- panel (b): hybrid vs BC paired bars ----
    rows_sorted = sorted(rows, key=lambda r: (r["K"], r["R"]))
    labels = [r["cell"] for r in rows_sorted]
    hybrid = [r["success_rate"] * 100 for r in rows_sorted]
    bc     = [r["bc_success_rate"] * 100 for r in rows_sorted]

    x = np.arange(len(labels))
    w = 0.4
    ax_b.bar(x - w/2, bc,     w, label="BC baseline",
             color="#bbbbbb", edgecolor="black")
    ax_b.bar(x + w/2, hybrid, w, label="Hybrid (MBD + BC warm-start)",
             color="#3b82f6", edgecolor="black")

    # Annotate cells where hybrid > BC
    for i, (h, b) in enumerate(zip(hybrid, bc)):
        if h > b:
            ax_b.annotate("*", (x[i] + w/2, h), ha="center", va="bottom",
                          fontsize=14, color="#16a34a")

    ax_b.set_xticks(x, labels, rotation=45, ha="right")
    ax_b.set_ylabel("success rate (%)")
    ax_b.set_title("Hybrid vs BC per cell\n* = hybrid > BC on this cell")
    ax_b.grid(axis="y", linestyle=":", alpha=0.5)
    ax_b.legend(loc="upper left")
    ax_b.set_ylim(0, max(15, max(hybrid + bc) * 1.15))

    fig.suptitle("Experiment 10 -- K x n_replans pareto grid (n=60/cell, seed=0)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, default=_DEFAULT_RESULTS,
                   help="Directory containing K*_R*.json files.")
    args = p.parse_args()

    if not args.results.is_dir():
        print(f"[error] results dir does not exist: {args.results}",
              file=sys.stderr)
        return 1

    rows = _load_cells(args.results)
    if not rows:
        print("[error] no K*_R*.json cells found", file=sys.stderr)
        return 1
    print(f"[exp10/analyze] loaded {len(rows)} cells from {args.results}")

    _write_summary_json(rows, args.results / "summary.json")
    _write_summary_md  (rows, args.results / "summary.md")
    _write_summary_png (rows, args.results / "summary.png")

    # Echo the headline table to stdout so a CI / monitoring run captures it.
    print()
    print(f"{'cell':<6}  {'K':>2}  {'R':>2}  "
          f"{'hybrid':>9}  {'BC':>9}  {'delta':>7}  {'MBD>BC':>7}")
    print("-" * 64)
    for r in sorted(rows, key=lambda x: (x["K"], x["R"])):
        mbd = (f"{r['mbd_beats_bc_frac']*100:.0f}%"
               if r["mbd_beats_bc_frac"] is not None else "?")
        print(f"{r['cell']:<6}  {r['K']:>2}  {r['R']:>2}  "
              f"{r['success_rate']*100:>7.1f}%  "
              f"{r['bc_success_rate']*100:>7.1f}%  "
              f"{r['delta_vs_bc']*100:>+6.1f}%  {mbd:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
