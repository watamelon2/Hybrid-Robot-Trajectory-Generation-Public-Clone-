"""Generate the figures used in the Week-8 presentation.

Reads the result JSONs already produced by the experiment suite
and writes PDFs into ``presentation/figures/``.  Each plot
is a thin function so individual figures can be rebuilt as
numbers update.

    python -m presentation.scripts.make_headline_figures --all

Available plots
---------------
- crosstask        cross-task bar chart (Slide 5)
- green_slices     lambda-slice + K-slice at ep0149 (Slide 6)
- heatmap_ckpt_K   2-D heatmap, optional supplemental (Slide 6)
- heatmap_ckpt_lam 2-D heatmap, optional supplemental (Slide 6)
- scaling_law      bank-size scaling law (Slide 7)

Result-JSON paths are hard-coded relative to the repo root so
the script "just runs" without further argument plumbing.  If
you re-aggregate, the inputs below are the only thing to edit.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "presentation" / "figures"

# ---------------------------------------------------------------------------
# Data sources (edit these if aggregation paths change)
# ---------------------------------------------------------------------------
SRC_CROSSTASK = {
    # task -> {prior -> {"succ": int, "n": int}}.  Filled from the
    # cell JSONs under experiments/15_unified_hybrid/results/.
    # Numbers below mirror docs/results.md as of 2026-06-07.
    "lift":      {"bc": (0, 60), "mbd": (0, 60),
                  "retrieval": (60, 60), "hybrid_dp": (60, 60)},
    "can":       {"bc": (4, 60), "mbd": (0, 60),
                  "retrieval": (177, 240), "hybrid_dp": (26, 30)},
    "square":    {"bc": (0, 60), "mbd": (0, 60),
                  "retrieval": (15, 30), "hybrid_dp": (5, 60)},
    # tool_hang is intentionally excluded from the headline cross-task
    # chart: upstream robosuite ships sparse-only rewards
    # (`robosuite-1.5/.../tool_hang.py:217`); the project-side dense-
    # reward wrapper (`env_dense_rewards.py`, commit 18af830) is
    # smoke-verified but the scored re-runs (Slate J) are pending.
    # The numbers below are pre-fix (0/40 across all priors) and are
    # NOT honestly comparable to Lift/Can/Square.  Re-include after
    # Slate J lands.
    # "tool_hang": {"bc": (0, 60), "mbd": (0, 40),
    #               "retrieval": (0, 40), "hybrid_dp": (0, 40)},
}
SRC_CROSSTASK_SOTA = {
    "lift": 0.993,
    "can": 0.923,
    "square": 0.747,
}

# Phase E + F aggregated numbers at ep0149
SRC_LAMBDA_SLICE = {
    # lambda -> (succ, n)
    0.1: (18, 30),
    0.5: (44, 60),
    2.0: (26, 30),
}
SRC_K_SLICE = {
    # K -> (succ, n)
    10: (44, 60),
    20: (25, 30),
}

SRC_HEATMAP_CKPT_K = REPO_ROOT / (
    "experiments/17_dp_hybrid/results/phase_d_2026_05_30/q2q3_AGGREGATE.json"
)
SRC_BANK_SIZE_SCALING = [
    (5, 14, 60),
    (10, 21, 60),
    (20, 39, 60),
    (50, 41, 60),
    (100, 42, 60),
    (200, 41, 60),
]
SRC_T2_PARETO_CSV = REPO_ROOT / (
    "experiments/15_unified_hybrid/results/"
    "slate_incremental_aggregate/T2_pareto_data.csv"
)
SRC_T2_MAGNITUDE_JSON = REPO_ROOT / (
    "experiments/15_unified_hybrid/results/"
    "slate_incremental_T2_magnitude_aggregate/summary.json"
)

# Audience-facing label / colour / marker map for the dynamics-adaptation
# methods.  The internal slate-cell IDs (T2.0 / T2.1 / T2.2 / T2.3 / T2.4)
# do not appear on the deck; the human-readable name on the left is what
# the audience sees.
T2_METHOD_STYLE = {
    "hybrid+5demos":     {"label": "Hybrid + 5 perturbed demos",
                          "color": "#1a7f3c", "marker": "o", "msize": 95},
    "hybrid+20demos":    {"label": "Hybrid + 20 perturbed demos",
                          "color": "#1a7f3c", "marker": "s", "msize": 85},
    "BC_unadapted":      {"label": "BC (no adaptation)",
                          "color": "#c23333", "marker": "^", "msize": 90},
    "BC_retrained_205":  {"label": "BC retrained (205 demos)",
                          "color": "#c23333", "marker": "v", "msize": 90},
    # T2_0 (canonical hybrid, no adaptation) is shown on the magnitude
    # figure but not on the Pareto figure because it sits at the same
    # x=0 point as BC_unadapted and would clutter.
    "hybrid_no_adapt":   {"label": "Hybrid (no adaptation)",
                          "color": "#3399aa", "marker": "s", "msize": 85},
}


# ---------------------------------------------------------------------------
# Utility: Wilson 95 % CI half-width
# ---------------------------------------------------------------------------
def wilson_ci(succ: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = succ / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return centre - half, centre + half


# ---------------------------------------------------------------------------
# Plot 1: cross-task bar chart
# ---------------------------------------------------------------------------
def plot_crosstask(out: Path) -> None:
    # tool_hang intentionally excluded -- see SRC_CROSSTASK comment.
    tasks = ["lift", "can", "square"]
    priors = ["bc", "mbd", "retrieval", "hybrid_dp"]
    colors = {"bc": "#888", "mbd": "#c33",
              "retrieval": "#39a", "hybrid_dp": "#1a7f3c"}
    labels = {"bc": "BC", "mbd": "MBD alone",
              "retrieval": "Retrieval + MBD",
              "hybrid_dp": "DP + MBD (ref)"}

    x = np.arange(len(tasks))
    w = 0.18
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    for i, prior in enumerate(priors):
        rates, lo, hi = [], [], []
        for task in tasks:
            s, n = SRC_CROSSTASK[task][prior]
            r = s / n if n else 0.0
            lo_, hi_ = wilson_ci(s, n)
            rates.append(r)
            lo.append(max(0.0, r - lo_))
            hi.append(max(0.0, hi_ - r))
        ax.bar(x + (i - (len(priors) - 1) / 2) * w, rates, width=w,
               yerr=[lo, hi], capsize=2,
               color=colors[prior], label=labels[prior])

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "-").title() for t in tasks])
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, ls=":", lw=0.5, c="k", alpha=0.3)
    ax.legend(loc="upper right", frameon=False, fontsize=8.2)
    ax.set_title(
        "Cross-task headline: retrieval + MBD, with trained-DP reference",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out}")


# ---------------------------------------------------------------------------
# Plot 2: lambda-slice + K-slice at ep0149 (the two GREEN cells)
# ---------------------------------------------------------------------------
def plot_green_slices(out: Path) -> None:
    fig, (axL, axK) = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)

    # --- lambda-slice (left)
    lams = sorted(SRC_LAMBDA_SLICE)
    rates, los, his = [], [], []
    for lam in lams:
        s, n = SRC_LAMBDA_SLICE[lam]
        r = s / n
        lo, hi = wilson_ci(s, n)
        rates.append(r)
        los.append(max(0.0, r - lo))
        his.append(max(0.0, hi - r))

    axL.errorbar(lams, rates, yerr=[los, his], fmt="o-",
                 color="#1a7f3c", lw=2.4, capsize=4, ms=8.5)
    axL.set_xscale("log")
    axL.set_xticks(lams)
    axL.set_xticklabels([str(l) for l in lams])
    axL.set_xlabel(r"$\lambda$ (bridge weight)")
    axL.set_ylabel("Success rate")
    axL.set_ylim(0.35, 1.0)
    axL.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axL.grid(axis="y", ls=":", lw=0.6, alpha=0.45)
    axL.axhline(SRC_LAMBDA_SLICE[0.5][0] / SRC_LAMBDA_SLICE[0.5][1],
                ls=":", c="grey", alpha=0.55)
    axL.text(0.11, SRC_LAMBDA_SLICE[0.5][0] / SRC_LAMBDA_SLICE[0.5][1] + 0.015,
             "canonical 73.3%", fontsize=8, color="0.35")
    for lam, rate in zip(lams, rates):
        axL.text(lam, rate - 0.075, f"{rate * 100:.0f}%",
                 ha="center", va="top", fontsize=8.5, color="#1a7f3c")
    axL.annotate(
        "GREEN: 86.7%\n+13.4 pp",
        xy=(2.0, 26 / 30), xytext=(0.30, 0.905),
        ha="center", fontsize=9.0, color="#1a7f3c", weight="bold",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
        arrowprops=dict(arrowstyle="->", color="#1a7f3c", lw=1),
    )
    axL.set_title(r"Bridge weight at $K\!=\!10$", fontsize=10)

    # --- K-slice (right)
    Ks = sorted(SRC_K_SLICE)
    rates, los, his = [], [], []
    for K in Ks:
        s, n = SRC_K_SLICE[K]
        r = s / n
        lo, hi = wilson_ci(s, n)
        rates.append(r)
        los.append(max(0.0, r - lo))
        his.append(max(0.0, hi - r))

    axK.errorbar(Ks, rates, yerr=[los, his], fmt="s-",
                 color="#1a7f3c", lw=2.4, capsize=4, ms=8.5)
    axK.set_xticks(Ks)
    axK.set_xlabel(r"$K$ (MBD denoising iterations)")
    axK.set_ylim(0.35, 1.0)
    axK.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axK.grid(axis="y", ls=":", lw=0.6, alpha=0.45)
    for K, rate in zip(Ks, rates):
        axK.text(K, rate - 0.065, f"{rate * 100:.0f}%",
                 ha="center", va="top", fontsize=8.5, color="#1a7f3c")
    axK.annotate(
        "GREEN: 83.3%\n+10.0 pp",
        xy=(20, 25 / 30), xytext=(12.9, 0.905),
        ha="center", fontsize=9.0, color="#1a7f3c", weight="bold",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
        arrowprops=dict(arrowstyle="->", color="#1a7f3c", lw=1),
    )
    axK.set_title(r"Refinement budget at $\lambda\!=\!0.5$", fontsize=10)

    fig.suptitle(
        "MBD helps when the prior is coherent and has headroom (ep0149)",
        fontsize=11.5, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out}")


# ---------------------------------------------------------------------------
# Plot 3 (optional): 2-D heatmap ckpt x K
# ---------------------------------------------------------------------------
def plot_heatmap_ckpt_K(out: Path) -> None:
    if not SRC_HEATMAP_CKPT_K.exists():
        print(f"[figures] skip heatmap (no {SRC_HEATMAP_CKPT_K})")
        return
    raw = json.loads(SRC_HEATMAP_CKPT_K.read_text())
    # Actual q2q3_AGGREGATE.json schema (2026-06-01):
    # {"cells": [{"ckpt": int, "lambda_prior": float, "K": int,
    #             "n_test_success": int, "n_test_total": int,
    #             "p_hat": float, ...}, ...]}
    cells = [c for c in raw.get("cells", [])
             if abs(c.get("lambda_prior", 0.5) - 0.5) < 1e-9]
    if not cells:
        print(f"[figures] skip heatmap (no lam=0.5 cells)")
        return
    ckpts = sorted({c["ckpt"] for c in cells})
    Ks = sorted({c["K"] for c in cells})
    grid = np.full((len(ckpts), len(Ks)), np.nan)
    for c in cells:
        i = ckpts.index(c["ckpt"])
        j = Ks.index(c["K"])
        n = c["n_test_total"]
        if n:
            grid[i, j] = c["n_test_success"] / n

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    im = ax.imshow(grid, aspect="auto", vmin=0, vmax=1,
                   cmap="viridis", origin="lower")
    ax.set_xticks(range(len(Ks)))
    ax.set_xticklabels([str(K) for K in Ks])
    ax.set_yticks(range(len(ckpts)))
    ax.set_yticklabels([str(ck) for ck in ckpts])
    ax.set_xlabel(r"$K$")
    ax.set_ylabel("DP checkpoint")
    for i in range(len(ckpts)):
        for j in range(len(Ks)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i,j]*100:.0f}",
                        ha="center", va="center",
                        color="white" if grid[i, j] < 0.6 else "k",
                        fontsize=8)
    fig.colorbar(im, ax=ax, label="success rate")
    ax.set_title(r"Prior-quality $\times$ MBD-budget (lam=0.5)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out}")


# ---------------------------------------------------------------------------
# Plot 4: methods comparison -- training time vs final success rate
# ---------------------------------------------------------------------------
# Sources:
#   BC          : experiments/03_bc_baseline/results/training_log.json
#                 -> total_wall_s = 71.9 s, 79.6 k params
#   DP v3b @ep0149: experiments/08_flow_matching_prior/results/
#                   v_theta_dp_tobs2_h16_v3b_ep0149.pt
#                 -> wall_s = 1019 s, 63.6 M params
#   DP v3b @ep0299: same prefix, ep0299
#                 -> wall_s = 2039 s, 63.6 M params
#   Flow v_theta v3 : exp 08 train_v3.py docstring says ~30 min on a
#                     GPU at the default config; logs only carry
#                     per-epoch loss, no wall.  Listed as approximate.
#   Retrieval     : zero training; bank-build is ~5 s for 200 demos.
#   MBD-alone     : no learnt component; "0 s" placeholder.
#
# Success rates on Can/ph (n>=60 each, except where noted):
#   BC = 6.7 %, MBD = 0.0 %, Retrieval = 73.8 %.
#   Flow-matching: the EDM "sigma-specialist" v3 is non-functional
#     (0 % standalone, reproduced across v3_eval_first + v4 + the
#     2026-06-04 50-seed slate) and is EXCLUDED.  The usable flow prior
#     is CFM v6 (82.7 M params): 56 % standalone, 70 % +MBD bridge
#     (lambda=2.0).  Sources: results/headline_priors_2026_06_04/
#     standalone_v6_cfm_test50.json (28/50) and eval_dp_ema_e300/
#     ...v6pull_lam2p0.json (21/30).
#   DP-v3b: standalone 73.3 % at ep0149 (the bridge-lift ckpt) and
#     76 % on the fresh 50-seed bucket at ep0199 (38/50; within noise);
#     hybrid GREEN = 86.7 % at ep0149 K=10 lambda=2.0.  We plot the
#     ep0149 pair so the standalone->hybrid lift is a single-ckpt delta.
SRC_METHODS = [
    # (label, train_seconds, success_alone, success_hybrid, params, color, marker)
    # Labels are deliberately short / audience-readable -- no internal
    # checkpoint / experiment versioning leaks through to the slide.
    # The mapping from human-readable name to ckpt is:
    #   MBD-only         -- experiments/04_mbd_baseline
    #   Retrieval (1-NN, zero-train)
    #                    -- experiments/15_unified_hybrid demo_1nn prior
    #   BC (MLP, 80 k params)
    #                    -- experiments/03_bc_baseline canonical_200_cpu
    #   Flow-matching (CFM, 83 M params)
    #                    -- experiments/08_flow_matching_prior v_theta_v6_unet_xl
    #                    (the EDM sigma-specialist v3 is broken/0% -> excluded)
    #   Diffusion Policy (trained, 64 M params)
    #                    -- experiments/08_flow_matching_prior v_theta_dp_tobs2_h16_v3b ep0149
    ("MBD only (no prior)",
                      0.0,   0.000, None,     0,        "#c23333", "x"),
    ("Retrieval (1-NN, zero-train)",
                      5.0,   0.000, 0.738,    0,        "#3399aa", "o"),
    ("BC (MLP, 80 k params)",
                      71.9,  0.067, None,    79_623,    "#888888", "s"),
    ("Flow-matching (CFM, 83 M params)",
                      1800.0, 0.560, 0.700, 82_702_599, "#a66bb5", "^"),
    ("Diffusion Policy (trained, 64 M params)",
                      1019.0, 0.733, 0.867, 63_649_415, "#1a7f3c", "D"),
]


def plot_methods_comparison(out: Path) -> None:
    """Two-panel: (left) bar chart of success rates;
                  (right) Pareto scatter of train-time vs success."""
    fig, (ax_bar, ax_scatter) = plt.subplots(
        1, 2, figsize=(10.5, 4.0),
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )

    # ----- (a) success-rate bar chart, "alone" vs "hybrid" -----
    labels = [m[0] for m in SRC_METHODS]
    alone  = [m[2] for m in SRC_METHODS]
    hybrid = [m[3] if m[3] is not None else m[2] for m in SRC_METHODS]
    colors = [m[5] for m in SRC_METHODS]

    y = np.arange(len(labels))
    width = 0.40
    ax_bar.barh(y - width / 2, alone, height=width,
                color=colors, edgecolor="black", linewidth=0.4)
    # Overlay the hybrid bar in same colour but with a hatched
    # second band so the lift is visible.
    ax_bar.barh(y + width / 2,
                [(h if h is not None else 0) for h in hybrid],
                height=width, color=colors, alpha=0.55,
                edgecolor="black", linewidth=0.4,
                hatch="///")
    # Numerical annotation on each bar
    for yi, (a, h) in enumerate(zip(alone, hybrid)):
        if a is not None:
            ax_bar.text(min(a + 0.02, 0.95), yi - width / 2,
                        f"{a*100:.1f}%", va="center", ha="left",
                        fontsize=8)
        if h is not None and h != a:
            ax_bar.text(min(h + 0.02, 0.95), yi + width / 2,
                        f"{h*100:.1f}%", va="center", ha="left",
                        fontsize=8, fontweight="bold")

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1.05)  # +0.05 headroom so 86.7% label clears the frame
    ax_bar.set_xlabel("Success rate on Can/ph (n>=60)")
    ax_bar.axvline(0.5, ls=":", c="grey", alpha=0.4)
    # Use neutral-grey proxy patches for the legend so the swatches
    # don't pick up any single method's colour (the previous version
    # showed two red squares because matplotlib used the first bar's
    # colour for the whole BarContainer's legend entry).  Anchor the
    # legend OUTSIDE the data area, above the bars on the right, so
    # it never overlaps the longest bar.
    legend_handles = [
        Patch(facecolor="lightgrey", edgecolor="black", linewidth=0.4,
              label="standalone"),
        Patch(facecolor="lightgrey", edgecolor="black", linewidth=0.4,
              hatch="///", alpha=0.55, label="+ MBD bridge"),
    ]
    # Legend inside the plot at upper-right -- that corner is empty
    # because the top three bars (MBD-only, Retrieval standalone,
    # BC standalone) sit at 0-7 %, leaving room from x~0.55 to x=1.05
    # in the y~[-0.5, 1.5] band.  Cannot overlap the longest bars
    # (Diffusion Policy 73.3/86.7 %) because those are at the BOTTOM
    # of the inverted y-axis.
    ax_bar.legend(
        handles=legend_handles, fontsize=8, frameon=True,
        framealpha=0.95, edgecolor="lightgrey",
        loc="upper right",
        handlelength=1.6, borderaxespad=0.4,
    )
    ax_bar.set_title("(a) Final success: standalone vs + MBD",
                     fontsize=10)

    # ----- (b) Pareto scatter: train time vs (best) success rate -----
    # log-x with a small floor so the "zero-train" methods are
    # visible at the left edge instead of at x=-inf.
    floor = 1.0  # 1 second
    xs = [max(m[1], floor) for m in SRC_METHODS]
    ys = [(m[3] if m[3] is not None else m[2]) for m in SRC_METHODS]
    for xi, yi, m in zip(xs, ys, SRC_METHODS):
        ax_scatter.scatter([xi], [yi], s=110, c=m[5], marker=m[6],
                           edgecolors="black", linewidth=0.7,
                           zorder=4)
        # Right-aligned label for the high-success methods
        # (retrieval and DP) to avoid overlap with the scatter point.
        offs = (8, 6) if yi > 0.3 else (8, -8)
        ax_scatter.annotate(m[0], (xi, yi), textcoords="offset points",
                            xytext=offs, fontsize=8, ha="left")

    ax_scatter.set_xscale("log")
    ax_scatter.set_xlabel("Training wall-clock time (s, log scale)")
    ax_scatter.set_ylabel("Final success rate on Can/ph")
    ax_scatter.set_xlim(0.5, 5000)
    ax_scatter.set_ylim(-0.05, 1.0)
    ax_scatter.axhline(0.5, ls=":", c="grey", alpha=0.4)
    ax_scatter.grid(True, which="both", ls=":", alpha=0.3)
    ax_scatter.set_title("(b) Pareto: train cost vs final success",
                         fontsize=10)

    fig.suptitle(
        "Methods comparison: zero-training retrieval is on the "
        "Pareto frontier; trained DP only wins via the hybrid bridge",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out}")


# ---------------------------------------------------------------------------
# Plot 5: method architecture diagram (one MBD denoising iteration)
# ---------------------------------------------------------------------------
def plot_method_architecture(out: Path) -> None:
    """A schematic of one MBD iteration with the prior bridge.

    Three boxes + two arrows, rendered in pure matplotlib so the
    slide tool doesn't have to draw it.
    """
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="black", fontsize=10,
            fontweight="normal"):
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            facecolor=fc, edgecolor=ec, linewidth=1.0,
        ))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight)

    def arrow(x1, y1, x2, y2, text=None, color="black"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.4,
                                    color=color))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, text,
                    ha="center", fontsize=8, color=color,
                    style="italic")

    # Trajectory at iteration t
    box(0.2, 2.0, 1.8, 1.0, r"$\tau_t$" + "\n(current\nplan)",
        fc="#e8e8e8", fontsize=11)

    # Perturbation -> cost evaluation
    box(2.4, 3.3, 2.4, 1.0, r"M perturbations" + "\n" + r"$\tau_t^{(m)} = \tau_t + \sigma_t\,\epsilon^{(m)}$",
        fc="#fff4d0", fontsize=9)
    box(2.4, 0.7, 2.4, 1.0, r"Prior pull" + "\n" + r"$-\lambda(\sigma_t)\,(\tau_t^{(m)} - \tau^{\rm prior})/\sigma_t^2$",
        fc="#d0e8ff", fontsize=9)

    # Cost & SNIS combine
    box(5.4, 3.3, 2.4, 1.0, r"$J(\tau_t^{(m)})$" + "\nSNIS weights",
        fc="#fff4d0", fontsize=10)
    box(5.4, 0.7, 2.4, 1.0, r"$\tau^{\rm prior}$" + "\n(learned)",
        fc="#d0e8ff", fontsize=10)

    # Update step
    box(8.2, 2.0, 1.6, 1.0, r"$\tau_{t-1}$" + "\n(denoised)",
        fc="#d4edd1", ec="#1a7f3c", fontsize=11, fontweight="bold")

    # Arrows
    arrow(2.0, 2.7, 2.4, 3.5, "noise σ_t", color="#a87a00")
    arrow(2.0, 2.3, 2.4, 1.5, "prior bridge", color="#0f5e9e")
    arrow(4.8, 3.8, 5.4, 3.8)
    arrow(4.8, 1.2, 5.4, 1.2)
    arrow(7.8, 3.8, 8.5, 3.0)
    arrow(7.8, 1.2, 8.5, 2.0)

    # Labels
    ax.text(5.0, 4.7, "MBD analytical-cost branch",
            ha="center", fontsize=10, fontweight="bold",
            color="#a87a00")
    ax.text(5.0, 0.2, "Learned-prior bridge (this work)",
            ha="center", fontsize=10, fontweight="bold",
            color="#0f5e9e")
    ax.text(8.95, 0.95,
            r"$\tau_{t-1} = \mathrm{SNIS}[\tau_t^{(m)}; J] + \text{prior pull}$",
            ha="center", fontsize=8.5,
            color="#1a7f3c", style="italic")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out}")


# ---------------------------------------------------------------------------
# Plot 6: bank-size scaling law
# ---------------------------------------------------------------------------
def plot_scaling_law(out: Path) -> None:
    xs = np.array([row[0] for row in SRC_BANK_SIZE_SCALING], dtype=float)
    succ = np.array([row[1] for row in SRC_BANK_SIZE_SCALING], dtype=float)
    n = np.array([row[2] for row in SRC_BANK_SIZE_SCALING], dtype=float)
    rates = succ / n
    lows, highs = [], []
    for s, total, r in zip(succ, n, rates):
        lo, hi = wilson_ci(int(s), int(total))
        lows.append(max(0.0, r - lo))
        highs.append(max(0.0, hi - r))

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.errorbar(xs, rates * 100.0,
                yerr=[np.array(lows) * 100.0, np.array(highs) * 100.0],
                fmt="o-", color="#3399aa", lw=2.0, ms=7, capsize=4)
    ax.axvline(20, color="#1a7f3c", ls="--", lw=1.2, alpha=0.8)
    ax.axhspan(65.0, 70.0, color="#1a7f3c", alpha=0.08)
    ax.annotate(
        "knee: most gain\nby 20 demos",
        xy=(20, 65.0), xytext=(8, 78),
        fontsize=9, color="#1a7f3c",
        arrowprops=dict(arrowstyle="->", color="#1a7f3c", lw=1),
    )
    ax.text(75, 66.3, "50-200 demo plateau", fontsize=9,
            color="#1a7f3c", style="italic")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(x)) for x in xs])
    ax.set_xlabel("Retrieval bank size $N_b$ (log scale)")
    ax.set_ylabel("Success rate on Can/ph (%)")
    ax.set_ylim(0, 90)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_title(
        "Bank-size scaling: finite mode coverage, not full-bank memorization",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out}")


# ---------------------------------------------------------------------------
# Plot 7: Pareto -- adaptation wall-clock vs success at Can mass=1.3.
# ---------------------------------------------------------------------------
# Reads the canonical aggregator's CSV (which carries clean method keys
# like 'hybrid+5demos' and 'BC_retrained_205' rather than internal slate
# cell IDs).  The audience-facing labels come from T2_METHOD_STYLE so
# the slide never shows "T2.X" internal identifiers.
def plot_t2_pareto(out: Path) -> None:
    if not SRC_T2_PARETO_CSV.exists():
        print(f"[figures] skip t2-pareto (no {SRC_T2_PARETO_CSV})")
        return

    with SRC_T2_PARETO_CSV.open() as fh:
        rows = list(csv.DictReader(fh))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    # Plot every row using its style entry; rows whose 'method' key is
    # not in T2_METHOD_STYLE are skipped with a print so we never silently
    # mis-label a cell.
    for row in rows:
        key = row["method"]
        if key not in T2_METHOD_STYLE:
            print(f"[figures] t2_pareto: skipping unknown method '{key}'")
            continue
        style = T2_METHOD_STYLE[key]
        x = float(row["adaptation_wall_s"])
        y = float(row["succ_rate"])
        lo = max(0.0, y - float(row["wilson_lo"]))
        hi = max(0.0, float(row["wilson_hi"]) - y)
        ax.errorbar(
            [x], [y * 100.0],
            yerr=[[lo * 100.0], [hi * 100.0]],
            fmt=style["marker"], color=style["color"],
            markersize=10, capsize=4, linewidth=0,
            elinewidth=1.5,
            markeredgecolor="black", markeredgewidth=0.6,
            label=style["label"],
        )
        # Inline annotation right of the marker.
        ax.annotate(
            style["label"], (x, y * 100.0),
            xytext=(12, 4), textcoords="offset points",
            fontsize=8.5,
        )

    # Pareto frontier guide line: vertical at x=1 then horizontal at y=73.3
    ax.axvspan(0, 2, color="#1a7f3c", alpha=0.04, zorder=0)
    ax.text(1.0, 95.0, "Pareto-optimal region\n(adaptation in ~1 s)",
            ha="center", fontsize=8, color="#0f5e1f", style="italic")

    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlim(-0.5, 200)
    ax.set_xticks([0, 1, 10, 78])
    ax.set_xticklabels(["0", "1", "10", "78"])
    ax.set_xlabel("Adaptation wall-clock (s)  -- training + bank build")
    ax.set_ylabel("Success rate at Can mass=1.3 (%)")
    ax.set_ylim(-5, 105)
    ax.axhline(50, ls=":", c="grey", alpha=0.4)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_title(
        "Dynamics adaptation: hybrid retrieval vs BC retraining\n"
        "(Can mass=1.3, 30 trials/cell, Wilson 95 % CIs)",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out}")


# ---------------------------------------------------------------------------
# Plot 8: Dynamics-adaptation magnitude sweep (bite-zone curve).
# ---------------------------------------------------------------------------
# Three lines across Can mass = 1.1..1.5:
#   hybrid + canonical bank (no adaptation)       -- T2.0 internally
#   hybrid + canonical + 5 perturbed demos        -- T2.1 internally
#   BC retrained on canonical 200 + 5 perturbed   -- T2.3 internally
# Data source: canonical aggregator JSON written by
# experiments/15_unified_hybrid/aggregate_magnitude.py (matches
# docs/results_writeups/t2_magnitude.md exactly).  No T2.X labels in the
# rendered figure -- the audience sees only the human-readable names.
def plot_t2_magnitude(out: Path) -> None:
    if not SRC_T2_MAGNITUDE_JSON.exists():
        print(f"[figures] skip t2-magnitude (no {SRC_T2_MAGNITUDE_JSON})")
        return
    summary = json.loads(SRC_T2_MAGNITUDE_JSON.read_text())

    masses = sorted(float(k) for k in summary.keys())

    def _series(cell_key: str):
        rates, lo, hi = [], [], []
        for m in masses:
            cell = summary[str(m)][cell_key]
            s, n = cell["n_success"], cell["n_total"]
            r = s / n
            wl, wh = wilson_ci(s, n)
            rates.append(r * 100.0)
            lo.append(max(0.0, (r - wl)) * 100.0)
            hi.append(max(0.0, (wh - r)) * 100.0)
        return rates, lo, hi

    r_t20, lo_t20, hi_t20 = _series("T2_0")
    r_t21, lo_t21, hi_t21 = _series("T2_1")
    r_t23, lo_t23, hi_t23 = _series("T2_3")

    fig, ax = plt.subplots(figsize=(6.8, 4.0))

    ax.errorbar(
        masses, r_t21, yerr=[lo_t21, hi_t21],
        marker="o", color="#1a7f3c", linewidth=2.0, markersize=7,
        capsize=3, label="Hybrid + 5 perturbed demos (~1 s adaptation)",
    )
    ax.errorbar(
        masses, r_t20, yerr=[lo_t20, hi_t20],
        marker="s", color="#3399aa", linewidth=1.5, markersize=6,
        capsize=3, linestyle="--",
        label="Hybrid (no adaptation, 0 s)",
    )
    ax.errorbar(
        masses, r_t23, yerr=[lo_t23, hi_t23],
        marker="v", color="#c23333", linewidth=2.0, markersize=7,
        capsize=3, label="BC retrained (205 demos, 78 s)",
    )

    # Highlight the BC-collapse band (everything <= ~15 %).
    ax.axhspan(0, 15, color="#c23333", alpha=0.07, zorder=0)
    ax.text(1.49, 12.0, "BC collapse zone", ha="right",
            fontsize=8, color="#822222", style="italic")

    ax.set_xlabel(r"Can mass scale  (1.0 = canonical, 1.3 = anchor)")
    ax.set_ylabel("Success rate at perturbed mass (%)")
    ax.set_ylim(-5, 105)
    ax.set_xticks(masses)
    ax.set_xticklabels([f"{m:.1f}" for m in masses])
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_title(
        "Dynamics magnitude sweep: hybrid flat across mass=1.1-1.5; BC stays at floor",
        fontsize=10,
    )
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--plot", choices=[
        "crosstask", "green_slices", "heatmap_ckpt_K",
        "scaling_law", "methods_comparison",
        "method_architecture", "t2_pareto", "t2_magnitude", "all",
    ], default="all")
    p.add_argument(
        "--ext", choices=["pdf", "png"], default="pdf",
        help="Output extension; use 'png' to also render preview thumbnails.",
    )
    args = p.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = FIG_DIR if args.ext == "pdf" else FIG_DIR / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = args.ext

    if args.plot in ("crosstask", "all"):
        plot_crosstask(out_dir / f"headline_bar_crosstask.{ext}")
    if args.plot in ("green_slices", "all"):
        plot_green_slices(out_dir / f"green_slices.{ext}")
    if args.plot in ("heatmap_ckpt_K", "all"):
        plot_heatmap_ckpt_K(out_dir / f"heatmap_ckpt_K.{ext}")
    if args.plot in ("scaling_law", "all"):
        plot_scaling_law(out_dir / f"scaling_law.{ext}")
    if args.plot in ("methods_comparison", "all"):
        plot_methods_comparison(out_dir / f"methods_comparison.{ext}")
    if args.plot in ("method_architecture", "all"):
        plot_method_architecture(out_dir / f"method_architecture.{ext}")
    if args.plot in ("t2_pareto", "all"):
        # Re-drawn from T2_pareto_data.csv with audience-facing labels;
        # no T2.X internal cell IDs reach the slide.
        plot_t2_pareto(out_dir / f"t2_pareto.{ext}")
    if args.plot in ("t2_magnitude", "all"):
        plot_t2_magnitude(out_dir / f"t2_magnitude.{ext}")


if __name__ == "__main__":
    main()
