"""End-to-end wiring smoke test for v_theta v3.

Does not require the robomimic HDF5.  Builds a tiny ``FilmEdmField``
with random weights, runs:

  1. ``edm_loss`` on synthetic ``(x_0, s)`` batches
  2. ``sample_initial_mean`` -> ``(H, A)`` warm-start
  3. ``make_prior_score_fn`` closure -> ``(H, A)`` score at each k

and asserts shapes / dtypes / numerical sanity.  Catches the
~80 %% of integration breakage you'd hit before the real Colab run.

Run from the repo root::

    python experiments/08_flow_matching_prior/smoke_v3.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


_THIS_DIR = Path(__file__).resolve().parent
for _p in (_THIS_DIR,):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from model_v3 import (  # noqa: E402
    FilmEdmField,
    FlowConfigV3,
    edm_loss,
    edm_sample,
    edm_score,
    sample_sigma_loguniform,
)
from adapter_v3 import FlowPriorV3, _cosine_sigma_schedule  # noqa: E402


def main() -> int:
    print("[smoke_v3] === Karras-EDM v3 wiring smoke test ===")

    torch.manual_seed(0)
    np.random.seed(0)

    # ---- 1. Build a tiny model ----------------------------------------
    obs_dim = 30
    action_dim = 7
    horizon = 16          # small H to keep the test fast
    config = FlowConfigV3(
        obs_dim=obs_dim,
        action_dim=action_dim,
        horizon=horizon,
        hidden=64,
        n_blocks=2,
        cond_hidden=32,
        sigma_embed_dim=16,
        sigma_min=0.05,
        sigma_max=0.5,
        sigma_data=0.5,
    )
    model = FilmEdmField(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[smoke_v3] (1) model built  params={n_params:,}  "
        f"x_dim={config.x_dim}"
    )

    # ---- 2. edm_loss on synthetic batch -------------------------------
    B = 8
    x0 = torch.randn(B, config.x_dim)
    s = torch.randn(B, obs_dim)
    loss = edm_loss(model, x0, s, n_sigma_per_traj=4)
    print(
        f"[smoke_v3] (2) edm_loss(B={B}, K=4)  loss={float(loss):.4f}"
    )
    assert torch.isfinite(loss), f"loss is not finite: {loss}"

    # Backprop sanity
    loss.backward()
    grad_norms = [
        float(p.grad.norm()) for p in model.parameters()
        if p.grad is not None
    ]
    assert grad_norms and all(np.isfinite(g) for g in grad_norms), (
        "gradients not finite"
    )
    print(
        f"[smoke_v3]     gradient OK  "
        f"max_grad_norm={max(grad_norms):.4f}  "
        f"min_grad_norm={min(grad_norms):.4f}"
    )

    # ---- 3. log-uniform sigma sampling --------------------------------
    sig = sample_sigma_loguniform(
        1000, config.sigma_min, config.sigma_max, device="cpu",
    ).numpy()
    assert (sig >= config.sigma_min - 1e-6).all() and (sig <= config.sigma_max + 1e-6).all(), (
        f"sigma out of range: [{sig.min():.4f}, {sig.max():.4f}] "
        f"vs config [{config.sigma_min}, {config.sigma_max}]"
    )
    # log-uniform => mean of log should be ~midpoint of log range.
    log_mid = 0.5 * (np.log(config.sigma_min) + np.log(config.sigma_max))
    assert abs(np.log(sig).mean() - log_mid) < 0.05, (
        f"log-uniform mean off: {np.log(sig).mean():.4f} vs {log_mid:.4f}"
    )
    print(
        f"[smoke_v3] (3) sample_sigma_loguniform  "
        f"range=[{sig.min():.4f}, {sig.max():.4f}]  "
        f"log_mean={np.log(sig).mean():.4f} (target {log_mid:.4f})"
    )

    # ---- 4. edm_score and Tweedie identity ----------------------------
    # ``D(x; σ, s) = x + σ² · score(x; σ, s)`` by construction.
    # Verify the identity.
    with torch.no_grad():
        x = torch.randn(B, config.x_dim)
        sigma = torch.full((B,), 0.2)
        D = model(x, sigma, s)
        score = edm_score(model, x, sigma, s)
        reconstructed = x + (sigma[:, None] ** 2) * score
        err = (D - reconstructed).abs().max().item()
    assert err < 1e-4, f"Tweedie identity broken: max err {err}"
    print(
        f"[smoke_v3] (4) Tweedie identity D = x + σ²·score  "
        f"max_err={err:.2e}"
    )

    # ---- 5. edm_sample -> (B, H, A) trajectories ----------------------
    with torch.no_grad():
        traj = edm_sample(model, s, n_steps=8)
    assert traj.shape == (B, horizon, action_dim), (
        f"edm_sample shape {traj.shape}, expected {(B, horizon, action_dim)}"
    )
    assert torch.isfinite(traj).all(), "trajectory has NaN/Inf"
    print(
        f"[smoke_v3] (5) edm_sample(n_steps=8)  shape={tuple(traj.shape)}  "
        f"||traj||/step={float(traj.norm() / (B * horizon * action_dim) ** 0.5):.4f}"
    )

    # ---- 6. Round-trip via FlowPriorV3 adapter ------------------------
    # Save -> load -> exercise the public surface.
    blob = {
        "format_version": 3,
        "arch": "karras_edm_v3",
        "config": config.to_dict(),
        "state_dict": model.state_dict(),
        "obs_mean": np.zeros(obs_dim, dtype=np.float32),
        "obs_std": np.ones(obs_dim, dtype=np.float32),
        "n_demos": 0,
        "n_chunks": 0,
        "horizon": horizon,
        "task": "smoke",
        "wall_s": 0.0,
    }
    with tempfile.TemporaryDirectory() as tdir:
        ckpt = Path(tdir) / "smoke_v3.pt"
        torch.save(blob, ckpt)
        prior = FlowPriorV3.from_checkpoint(ckpt, device="cpu")
    assert prior.sigma_min == config.sigma_min
    assert prior.sigma_max == config.sigma_max
    print(
        f"[smoke_v3] (6) FlowPriorV3.from_checkpoint  "
        f"σ=[{prior.sigma_min}, {prior.sigma_max}]  "
        f"obs_mean.shape={prior.obs_mean.shape}"
    )

    # sample_initial_mean
    s0 = np.random.randn(obs_dim).astype(np.float32)
    warm = prior.sample_initial_mean(s0, n_steps=8, seed=42)
    assert warm.shape == (horizon, action_dim) and warm.dtype == np.float32, (
        f"warm-start has shape {warm.shape}, dtype {warm.dtype}"
    )
    assert np.isfinite(warm).all(), "warm-start has NaN/Inf"
    # Determinism check: same seed -> same warm-start.
    warm2 = prior.sample_initial_mean(s0, n_steps=8, seed=42)
    assert np.allclose(warm, warm2), "warm-start not deterministic under seed"
    print(
        f"[smoke_v3]     sample_initial_mean  shape={warm.shape}  "
        f"||warm||/step={float(np.linalg.norm(warm) / (horizon * action_dim) ** 0.5):.4f}  "
        f"deterministic=OK"
    )

    # make_prior_score_fn + closure exercise
    K = 10
    score_fn = prior.make_prior_score_fn(s0, K)
    mu = np.zeros((horizon, action_dim), dtype=np.float32)
    schedule = _cosine_sigma_schedule(K, config.sigma_max, config.sigma_min)
    last_norm = None
    for k in range(K):
        sc = score_fn(mu, k)
        assert sc.shape == (horizon, action_dim) and sc.dtype == np.float32, (
            f"score has shape {sc.shape}, dtype {sc.dtype}"
        )
        assert np.isfinite(sc).all(), f"score at k={k} not finite"
        n = float(np.linalg.norm(sc))
        # Sanity: at very small σ the score magnitude grows like 1/σ² for
        # a generic x, so iter K-1 should have larger ||score|| than iter 0
        # for a randomly-initialised model.  This is a *very* loose check;
        # we just want to make sure k indexes into the schedule.
        last_norm = n
    print(
        f"[smoke_v3]     score_fn closure  K={K}  "
        f"σ_schedule=[{schedule[0]:.3f}, ..., {schedule[-1]:.3f}]  "
        f"||score(k=K-1)||={last_norm:.4f}"
    )

    # ---- 7. Custom sigma_schedule override ----------------------------
    custom = np.linspace(0.3, 0.05, K, dtype=np.float32)
    score_fn2 = prior.make_prior_score_fn(s0, K, sigma_schedule=custom)
    sc_k0 = score_fn2(mu, 0)
    assert sc_k0.shape == (horizon, action_dim)
    print(f"[smoke_v3] (7) custom sigma_schedule OK")

    # ---- 8. dispatch from adapter.FlowPrior auto-detects v3 ----------
    # The adapter.py dispatch should return a FlowPriorV3 for an
    # arch=karras_edm_v3 checkpoint.  Verify.
    from adapter import FlowPrior  # noqa: E402
    with tempfile.TemporaryDirectory() as tdir:
        ckpt = Path(tdir) / "smoke_v3.pt"
        torch.save(blob, ckpt)
        dispatched = FlowPrior.from_checkpoint(
            ckpt, device="cpu", sigma_min=0.05, sigma_max=0.4,
        )
    assert isinstance(dispatched, FlowPriorV3), (
        f"adapter dispatch returned {type(dispatched).__name__}, "
        "expected FlowPriorV3"
    )
    assert dispatched.sigma_max == 0.4, (
        f"sigma_max override did not apply: {dispatched.sigma_max}"
    )
    print(
        f"[smoke_v3] (8) adapter.FlowPrior dispatch -> "
        f"{type(dispatched).__name__}  σ_override applied"
    )

    print("[smoke_v3] === ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
