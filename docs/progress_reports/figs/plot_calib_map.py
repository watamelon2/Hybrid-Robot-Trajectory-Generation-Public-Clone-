#!/usr/bin/env python3
"""Generate the calibration-map figure for the HRTG final report.

Plots the three flagship lambda-curves (the inverted-U) from the deepened
DP grid (N=60-90/cell). Data transcribed from
HRTG-square-dp-followup/.../results/nwfix_calibration_analysis_2026_06_14.txt.
Output: figs/calib_map.png (PNG, since docs/**/*.pdf is gitignored).

Run: python docs/progress_reports/figs/plot_calib_map.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_map.png")

# (lambda, success) over the interesting [0,2] inverted-U region.
can010 = [(0, .511), (0.5, .600), (0.75, .667), (1, .622), (1.5, .656), (2, .656)]
can025 = [(0, .517), (0.5, .583), (0.75, .617), (1, .650), (1.5, .617), (2, .533)]
sqr199 = [(0, .544), (0.5, .544), (1, .689), (2, .622)]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.7,
    "mathtext.fontset": "dejavusans",
})

fig, ax = plt.subplots(figsize=(3.35, 2.2))
for data, lab, mk, c in [
    (can010, r"can ep0199, $\sigma_{\max}=0.10$", "o-", "#1f77b4"),
    (can025, r"can ep0199, $\sigma_{\max}=0.25$", "s--", "#ff7f0e"),
    (sqr199, r"square ep0199", "^-", "#2ca02c"),
]:
    xs, ys = zip(*data)
    ax.plot(xs, ys, mk, color=c, ms=4.5, lw=1.6, label=lab,
            markeredgecolor="white", markeredgewidth=0.5,
            clip_on=False, zorder=3)

# shaded interior-optimum band
ax.axvspan(0.5, 2.0, color="#9ecae1", alpha=0.20, lw=0, zorder=0)
ax.text(1.25, 0.405, "interior optimum", ha="center", va="center",
        fontsize=6.8, style="italic", color="0.4")
ax.annotate("over-anchoring", xy=(2.0, 0.565), xytext=(1.30, 0.470),
            fontsize=6.8, style="italic", color="0.4",
            arrowprops=dict(arrowstyle="->", color="0.55", lw=0.9))
ax.set_xlabel(r"prior-pull weight $\lambda$", fontsize=8.5)
ax.set_ylabel("success rate", fontsize=8.5)
ax.set_ylim(0.38, 0.74)
ax.set_xlim(-0.08, 2.1)
ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
ax.tick_params(labelsize=7, length=3, width=0.6)
ax.grid(True, alpha=0.35, lw=0.4, ls=":")
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=6.3, loc="upper left", framealpha=0.0,
          handlelength=1.8, borderpad=0.2, labelspacing=0.35)
fig.tight_layout(pad=0.3)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
