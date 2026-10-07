"""Pareto plot for the T2 incremental-adaptation slate.

Reads ``results/slate_incremental_aggregate/T2_pareto_data.csv`` and
produces a wall-clock x success-rate scatter with Wilson 95% CIs.

The x-axis is *adaptation* wall-clock (the cost of moving from the
canonical operating point to a mass=1.3-adapted operating point):

    BC_unadapted       0 s   (no adaptation, use canonical policy.pt as-is)
    hybrid+5 demos     1 s   (re-index DemoBank with 5 mass=1.3 demos)
    hybrid+20 demos    1 s   (re-index DemoBank with 20 mass=1.3 demos)
    BC_retrained_205  ~80 s  (retrain BC on 200+5 mass=1.3 demos, 500 epochs)

The y-axis is mass=1.3 closed-loop success rate (n=30 trials per cell,
single seed=0, identical evaluation harness).  Error bars are Wilson
95% CIs on the binomial proportion.

Headline finding: the hybrid points sit upper-left of BC_retrained,
substantiating the Pareto-dominance claim of the incremental-prior
adapter.

Usage:

    python experiments/15_unified_hybrid/plot_incremental_pareto.py

Outputs into the same aggregate dir:

    T2_pareto.png  (150 dpi)
    T2_pareto.pdf
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


_THIS_DIR = Path(__file__).resolve().parent
_AGG_DIR = _THIS_DIR / "results" / "slate_incremental_aggregate"
_CSV = _AGG_DIR / "T2_pareto_data.csv"


# Per-method plotting style.  Colors picked to match the cost-time
# aggregate so cross-figure comparison is easier (hybrid = blue,
# BC = orange/red).
_STYLE = {
    "hybrid+5demos":     dict(color="#1f77b4", marker="o", label="hybrid + 5 mass=1.3 demos (T2.1)"),
    "hybrid+20demos":    dict(color="#1f77b4", marker="s", label="hybrid + 20 mass=1.3 demos (T2.4)"),
    "BC_unadapted":      dict(color="#d62728", marker="^", label="BC unadapted (T2.2)"),
    "BC_retrained_205":  dict(color="#d62728", marker="v", label="BC retrained on 205 demos (T2.3)"),
}

# Slight x-offset for the two hybrid points so the markers don't overlap.
_X_JITTER = {
    "hybrid+5demos":   -0.15,
    "hybrid+20demos":  +0.15,
    "BC_unadapted":     0.0,
    "BC_retrained_205": 0.0,
}


def _load() -> list[dict]:
    if not _CSV.exists():
        raise FileNotFoundError(
            f"{_CSV} not found.  Run aggregate_incremental.py --write first."
        )
    rows = []
    with open(_CSV) as f:
        for r in csv.DictReader(f):
            rows.append({
                "method":  r["method"],
                "wall_s":  float(r["adaptation_wall_s"]),
                "rate":    float(r["succ_rate"]),
                "lo":      float(r["wilson_lo"]),
                "hi":      float(r["wilson_hi"]),
            })
    return rows


def main() -> int:
    rows = _load()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    # Plot each point with Wilson error bar.  Track for Pareto-frontier
    # line connecting upper-left dominators.
    for r in rows:
        m = r["method"]
        style = _STYLE.get(m)
        if style is None:
            continue
        x = r["wall_s"] + _X_JITTER.get(m, 0.0)
        y = 100 * r["rate"]
        lo = 100 * (r["rate"] - r["lo"])
        hi = 100 * (r["hi"] - r["rate"])
        ax.errorbar(
            [x], [y], yerr=[[lo], [hi]],
            fmt=style["marker"], color=style["color"],
            ms=10, capsize=4, lw=1.8, label=style["label"],
        )

    # Pareto frontier overlay: connect the upper-left non-dominated
    # set so the visual hierarchy is unambiguous.  Sort by wall_s.
    pts = sorted(rows, key=lambda r: r["wall_s"])
    frontier = []
    best_rate = -1.0
    for p in pts:
        if p["rate"] > best_rate:
            frontier.append(p)
            best_rate = p["rate"]
    if len(frontier) >= 2:
        fx = [p["wall_s"] + _X_JITTER.get(p["method"], 0.0) for p in frontier]
        fy = [100 * p["rate"] for p in frontier]
        ax.plot(
            fx, fy, "--", color="#888888", lw=1.2, alpha=0.6,
            label="Pareto frontier", zorder=0,
        )

    # Annotate each point with the cell label.
    for r in rows:
        m = r["method"]
        if m not in _STYLE:
            continue
        cell = {
            "hybrid+5demos":   "T2.1",
            "hybrid+20demos":  "T2.4",
            "BC_unadapted":    "T2.2",
            "BC_retrained_205": "T2.3",
        }[m]
        x = r["wall_s"] + _X_JITTER.get(m, 0.0)
        y = 100 * r["rate"]
        ax.annotate(
            cell,
            xy=(x, y),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=9, color=_STYLE[m]["color"], weight="bold",
        )

    ax.set_xlabel(
        "Adaptation wall-clock (s)  [training + bank build]   "
        r"$\longleftarrow$ better",
        fontsize=11,
    )
    ax.set_ylabel(
        r"better $\longrightarrow$   Success rate at mass=1.3  (%)",
        fontsize=11,
    )
    ax.set_title(
        "T2 Pareto: hybrid retrieval prior vs BC retraining\n"
        "Can mass=1.3, 30 trials/cell, seed=0, Wilson 95% CIs",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 100)
    # Set x-lim to give the BC_retrained point breathing room.
    max_x = max(r["wall_s"] for r in rows)
    ax.set_xlim(-5, max(max_x * 1.15, 30))
    ax.legend(loc="center right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    out_png = _AGG_DIR / "T2_pareto.png"
    out_pdf = _AGG_DIR / "T2_pareto.pdf"
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
