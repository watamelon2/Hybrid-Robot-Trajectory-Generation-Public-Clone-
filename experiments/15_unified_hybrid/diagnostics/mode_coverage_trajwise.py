"""Step-resolved mode-coverage diagnostic.

For each sampled state, compute the action-std across top-5 NN at
multiple lookahead steps along the H=120 warm-start window.  If the
std grows with lookahead, the multi-modality story is at the
*trajectory* level, not the per-step level.
"""
import sys
from pathlib import Path

ROOT = Path("/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation")
for p in [
    ROOT / "experiments",
    ROOT / "experiments" / "03_bc_baseline",
    ROOT / "experiments" / "15_unified_hybrid",
]:
    sys.path.insert(0, str(p))

import numpy as np
from priors_zerotrain import load_demo_bank
from common import default_dataset_path

K = 5
H = 120
LOOKAHEADS = [0, 5, 10, 20, 40, 60, 100]

for task in ("can",):
    bank = load_demo_bank(default_dataset_path(task), task=task)
    rs = np.random.RandomState(0)
    n_sample = 1000
    idx_sample = rs.choice(bank.n_total, size=n_sample, replace=False)

    # For each sample, compute the (H, A) windows of top-5 NN, then the
    # per-step std across the 5 neighbours.
    std_at_t = {t: np.empty(n_sample) for t in LOOKAHEADS}
    for j, i in enumerate(idx_sample):
        s = bank.obs_norm[i]
        d2 = ((bank.obs_norm - s[None, :]) ** 2).sum(axis=-1)
        top = np.argpartition(d2, K)[:K]
        # Action window per neighbour.
        windows = np.stack(
            [bank.slice_actions(int(idx), H) for idx in top], axis=0
        )  # (K, H, A)
        for t in LOOKAHEADS:
            # Per-dim std across neighbours, mean over action dims.
            std_at_t[t][j] = float(windows[:, t, :].std(axis=0).mean())

    print(f"=== task = {task!r} ===  (n_sample={n_sample}, K={K})")
    print("  step    median     q90      q99      max     pct>0.3")
    for t in LOOKAHEADS:
        s = std_at_t[t]
        pct_high = float((s > 0.3).mean() * 100)
        print(
            f"  t+{t:<3d}  {np.median(s):.3f}   {np.quantile(s, 0.9):.3f}   "
            f"{np.quantile(s, 0.99):.3f}   {s.max():.3f}   {pct_high:.1f}%"
        )

    # Correlation between t=0 std and t=60 std: do the same states
    # that are ambiguous at t=0 stay ambiguous at t=60?
    corr_06 = float(np.corrcoef(std_at_t[0], std_at_t[60])[0, 1])
    print(f"\n  corr(std@t0, std@t60) = {corr_06:.3f}")
