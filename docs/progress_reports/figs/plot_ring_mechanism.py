"""Ring-manifold mechanism toy (illustrative) for the final report.

Reuses the experiment-08 flow-prior + experiment-04 ``denoise_loop`` API.
A tiny flow prior is trained on a unit-ring data manifold (its samples are
the light backdrop, confirming the learned prior recovers the ring). The
hybrid solution is then driven through the *real* ``denoise_loop`` against
a cost whose optimum lies OFF the ring, using the manifold's analytic
noise-aware score for the prior pull (the tiny 2-D learned field is too
coarse for a legible sweep). As the prior-pull weight ``lambda`` grows, the
solution descends radially from the off-manifold cost optimum onto the
manifold at the cost-closest point -- the paper's mechanism in miniature.

Run (from the repo's python env with numpy/scipy/torch/matplotlib):
    python docs/progress_reports/figs/plot_ring_mechanism.py
Writes ``ring_mechanism.png`` next to this script.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve()
ROOT = _THIS.parents[3]                       # repo root
EXP08 = ROOT / "experiments" / "08_flow_matching_prior"
EXP04 = ROOT / "experiments" / "04_mbd_baseline"
for p in (EXP08, EXP04):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from mbd import MBDParams, cosine_sigma_schedule, denoise_loop  # noqa: E402
from model import (  # noqa: E402
    FlowConfig, VelocityField, conditional_fm_loss, sample_trajectory,
)
from adapter import FlowPrior  # noqa: E402

torch.manual_seed(0)
rng = np.random.default_rng(0)

# --- unit-ring data manifold p_data + tiny trained flow prior ---
n = 1024
theta = rng.uniform(0, 2 * np.pi, size=n).astype(np.float32)
X = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
S = np.zeros((n, 1), dtype=np.float32)
config = FlowConfig(obs_dim=1, action_dim=1, horizon=2,
                    hidden_sizes=(64, 64), time_embed_dim=8)
model = VelocityField(config)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
X_t, S_t = torch.from_numpy(X), torch.from_numpy(S)
for _ in range(400):
    loss = conditional_fm_loss(model, X_t, S_t)
    opt.zero_grad(); loss.backward(); opt.step()
prior_samples = sample_trajectory(
    model, S_t[:400], n_steps=64).detach().cpu().numpy().reshape(-1, 2)

# --- cost optimum OFF the ring; start the planner there ---
target = np.array([1.85, 0.78], dtype=np.float32)
nearest = target / np.linalg.norm(target)


def rollout(traj):
    return traj.reshape(traj.shape[0], 2)


def cost(states, _traj):
    return np.linalg.norm(states - target, axis=-1)


K = 30
params = MBDParams(H=2, M=64, K=K, action_dim=1, lambda_=0.5,
                   sigma_schedule=cosine_sigma_schedule(K, 0.5, 0.05))
RING_DELTA = 0.05
_SCHED = np.asarray(params.sigma_schedule, dtype=np.float32)


def score_fn(mu: np.ndarray, k: int) -> np.ndarray:
    """Noise-aware grad-log of the unit ring (bounded radial pull)."""
    x = np.asarray(mu, dtype=np.float32).reshape(2)
    r = float(np.linalg.norm(x)) + 1e-8
    sig = float(_SCHED[int(k)])
    s = -((r - 1.0) / (RING_DELTA ** 2 + sig ** 2)) * (x / r)
    return s.reshape(2, 1).astype(np.float32)


mu0 = target.reshape(2, 1).astype(np.float32)

lambdas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 10.0]
ends = np.array([
    denoise_loop(mu0, rollout, cost, params, np.random.default_rng(0),
                 prior_score_fn=score_fn, prior_lambda=lam)[0].reshape(-1)
    for lam in lambdas
])

lam_dense = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0])
cost_d, offman_d = [], []
for lam in lam_dense:
    e = denoise_loop(mu0, rollout, cost, params, np.random.default_rng(0),
                     prior_score_fn=score_fn, prior_lambda=lam)[0].reshape(-1)
    cost_d.append(float(np.linalg.norm(e - target)))
    offman_d.append(float(abs(np.linalg.norm(e) - 1.0)))
cost_d, offman_d = np.array(cost_d), np.array(offman_d)

# --- figure (stacked, single-column) ---
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Dedicated colorbar column so both panel axes share column 0 (same
# width) and line up; datalim-equal keeps the ring circular while filling
# that column.
fig = plt.figure(figsize=(3.35, 3.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.045],
                      height_ratios=[1.5, 1.0], hspace=0.55, wspace=0.04)
axg = fig.add_subplot(gs[0, 0])
cax = fig.add_subplot(gs[0, 1])
axt = fig.add_subplot(gs[1, 0])

# panel (a): geometry; solution vs prior pull lambda
ph = np.linspace(0, 2 * np.pi, 400)
axg.plot(np.cos(ph), np.sin(ph), color="0.55", lw=1.0,
         label="data manifold $p^{data}$")
axg.scatter(prior_samples[:, 0], prior_samples[:, 1], s=4, alpha=0.10,
            color="tab:blue", label="prior samples")
axg.plot(ends[:, 0], ends[:, 1], color="0.6", lw=0.8, zorder=3)
sc = axg.scatter(ends[:, 0], ends[:, 1], c=lambdas, cmap="viridis",
                 s=26, zorder=5, edgecolor="k", linewidth=0.3)
cb = fig.colorbar(sc, cax=cax)
cb.set_label(r"prior pull $\lambda$", fontsize=8)
cb.ax.tick_params(labelsize=7)
axg.annotate("increasing " r"$\lambda$", xy=ends[2], xytext=(0.05, 0.62),
             fontsize=6.5, color="0.3",
             arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.0))
axg.scatter(*target, marker="*", s=80, color="tab:red", zorder=6,
            label=r"cost optimum ($\lambda{=}0$)")
axg.scatter(*nearest, marker="o", s=52, facecolors="none",
            edgecolors="tab:green", linewidth=1.5, zorder=6,
            label="on-manifold solution")
axg.set_aspect("equal", adjustable="datalim")
axg.set_xlim(-1.3, 2.35); axg.set_ylim(-1.45, 1.5)
axg.tick_params(labelsize=7)
axg.legend(fontsize=5.8, loc="lower left", framealpha=0.9)
axg.set_title(r"(a) solution moves onto $p^{data}$ as $\lambda$ grows",
              fontsize=8)

# panel (b): cost vs off-manifold trade
axt.plot(lam_dense, cost_d, "o-", color="tab:red", ms=3.5, lw=1.2,
         label="cost: dist. to optimum")
axt.plot(lam_dense, offman_d, "s-", color="tab:blue", ms=3.5, lw=1.2,
         label="off-manifold dist. $||x|-1|$")
axt.axvspan(0.5, 2.0, color="0.85", alpha=0.6, zorder=0)
axt.text(1.20, 0.50 * max(cost_d.max(), offman_d.max()),
         "calibrated\nband", fontsize=6, ha="center", va="center",
         color="0.35")
axt.set_xlabel(r"prior pull $\lambda$", fontsize=8)
axt.set_ylabel("distance", fontsize=8)
axt.set_xlim(-0.3, 10.3)
axt.tick_params(labelsize=7)
axt.legend(fontsize=6, loc="center right")
axt.set_title("(b) cost vs. manifold-realism trade", fontsize=8)

fig.subplots_adjust(left=0.135, right=0.86, top=0.93, bottom=0.115,
                    hspace=0.55)
out = _THIS.parent / "ring_mechanism.png"
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
print(f"wrote {out}")
