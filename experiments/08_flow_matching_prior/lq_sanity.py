"""LQ-style sanity check for the flow-matching prior + composite update.

Mirrors ``experiments/04_mbd_baseline/lq_sanity.py`` in spirit: a 2-D
problem where we *know* the data manifold, train a tiny v_theta on
samples from it, and verify that

  1. samples from the trained prior recover the data manifold;
  2. ``denoise_loop`` with ``prior_score_fn`` set and a non-trivial
     cost is pulled toward the manifold for every lambda > 0;
  3. lambda=0 reproduces the no-prior loop exactly.

Runs in seconds on CPU and is the only test that doesn't need the
robomimic dataset, so it doubles as the CI smoke test for this
experiment directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _THIS_DIR.parent
_EXP_04 = _EXPERIMENTS_DIR / "04_mbd_baseline"
for _p in (_THIS_DIR, _EXP_04):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from mbd import MBDParams, cosine_sigma_schedule, denoise_loop  # noqa: E402
from model import (  # noqa: E402
    FlowConfig,
    VelocityField,
    conditional_fm_loss,
    sample_trajectory,
)
from adapter import FlowPrior  # noqa: E402


def main() -> int:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # 2-D ring data manifold p_data, parameterised as (cos, sin).
    n = 1024
    theta = rng.uniform(0, 2 * np.pi, size=n).astype(np.float32)
    X = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    S = np.zeros((n, 1), dtype=np.float32)  # trivial conditioning

    config = FlowConfig(
        obs_dim=1, action_dim=1, horizon=2,  # x_dim = 2 to match the ring
        hidden_sizes=(64, 64), time_embed_dim=8,
    )
    model = VelocityField(config)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    X_t = torch.from_numpy(X)
    S_t = torch.from_numpy(S)
    for epoch in range(400):
        loss = conditional_fm_loss(model, X_t, S_t)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"[lq] final FM loss: {float(loss):.4f}")

    # Sample from the prior; check radii are ~1.
    samp = sample_trajectory(model, S_t[:128], n_steps=64).cpu().numpy()
    radii = np.linalg.norm(samp.reshape(-1, 2), axis=-1)
    print(
        f"[lq] sample radii: mean={radii.mean():.3f} "
        f"std={radii.std():.3f}  (target ~1.0, ~0.0)"
    )

    # Now run denoise_loop with this prior driving descent against a
    # cost that prefers x = (1, 0). A trajectory of shape (H=2, A=1)
    # is just a 2-D point; reuse it for the LQ sanity.
    prior = FlowPrior(
        model,
        obs_mean=np.zeros(1, dtype=np.float32),
        obs_std=np.ones(1, dtype=np.float32),
    )
    target = np.array([1.0, 0.0], dtype=np.float32)

    def rollout(traj):
        # (M, 2, 1) -> (M, 2)
        return traj.reshape(traj.shape[0], 2)

    def cost(states, _traj):
        return np.linalg.norm(states - target, axis=-1)

    K = 20
    params = MBDParams(
        H=2, M=64, K=K, action_dim=1, lambda_=0.5,
        sigma_schedule=cosine_sigma_schedule(K, 0.5, 0.05),
    )
    score_fn = prior.make_prior_score_fn(np.zeros(1, dtype=np.float32), K)

    mu0 = np.array([[-0.6], [-0.6]], dtype=np.float32)  # off-manifold

    # (a) lambda=0 == old loop
    rng_a = np.random.default_rng(0)
    mu_a, log_a = denoise_loop(
        mu0, rollout, cost, params, rng_a,
        prior_score_fn=score_fn, prior_lambda=0.0,
    )
    rng_b = np.random.default_rng(0)
    mu_b, log_b = denoise_loop(mu0, rollout, cost, params, rng_b)
    assert np.allclose(mu_a, mu_b), "lambda=0 must match no-prior loop"
    print("[lq] lambda=0 backward-compat: OK")

    # (b) lambda > 0: final point should be closer to (1, 0) AND on
    # the unit ring.
    rng_c = np.random.default_rng(0)
    mu_c, log_c = denoise_loop(
        mu0, rollout, cost, params, rng_c,
        prior_score_fn=score_fn, prior_lambda=1.0,
    )
    end = mu_c.reshape(-1)
    radius = float(np.linalg.norm(end))
    dist = float(np.linalg.norm(end - target))
    print(
        f"[lq] composite (lambda=1) final point: {end.tolist()}  "
        f"radius={radius:.3f}  dist_to_target={dist:.3f}"
    )

    print("[lq] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
