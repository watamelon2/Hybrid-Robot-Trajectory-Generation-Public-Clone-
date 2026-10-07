"""Unit tests for the tempered/gated combination family (mbd.py).

Validates ``tempered_combine`` endpoints (product / max-gate) and the
``combine_mode="tempered"`` path of ``denoise_loop`` -- all without
robosuite, using a synthetic identity rollout + quadratic cost so the
tests run in milliseconds. Covers the implementation backing
``docs/theory/09_mbd_data_guided_comparison.md`` §6.1/§6.2/§6.3 and
follow-up #4 (sigma-mode toggle).

Run: ``python -m pytest experiments/04_mbd_baseline/test_tempered_combine.py``
or ``python experiments/04_mbd_baseline/test_tempered_combine.py``.
"""

from __future__ import annotations

import numpy as np

from mbd import MBDParams, denoise_loop, tempered_combine


def test_product_limit_equals_loglinear_pool():
    """p -> 0 reproduces the weighted log-linear pool (= product/PoE)."""
    rng = np.random.default_rng(0)
    lm = rng.standard_normal(64)
    ld = rng.standard_normal(64)
    out, _ = tempered_combine(lm, ld, p=0.0, mix_model=1.0, mix_demo=3.0)
    expected = (1.0 / 4.0) * lm + (3.0 / 4.0) * ld
    assert np.allclose(out, expected, atol=1e-9)


def test_max_gate_limit():
    """p -> +inf reproduces MBD-Demo's elementwise max-gate (Eq. 13)."""
    rng = np.random.default_rng(1)
    lm = rng.standard_normal(64)
    ld = rng.standard_normal(64)
    out, _ = tempered_combine(lm, ld, p=np.inf, mix_model=1.0, mix_demo=1.0)
    assert np.allclose(out, np.maximum(lm, ld))


def test_finite_p_between_product_and_max():
    """Finite p sits between the product mean and the max, elementwise."""
    lm = np.array([0.0, 1.0, -2.0])
    ld = np.array([1.0, -1.0, 0.5])
    prod, _ = tempered_combine(lm, ld, p=1e-9)
    big, _ = tempered_combine(lm, ld, p=50.0)
    mx = np.maximum(lm, ld)
    # Generalised mean is monotone increasing in p; large p ~ max.
    assert np.all(prod <= mx + 1e-9)
    assert np.allclose(big, mx, atol=1e-3)


def test_gate_frac():
    """gate_frac = fraction of samples where the demo branch wins."""
    lm = np.array([0.0, 0.0, 0.0, 0.0])
    ld = np.array([1.0, -1.0, 2.0, -3.0])  # 2 of 4 exceed lm
    _, gf = tempered_combine(lm, ld, p=np.inf)
    assert gf == 0.5


def test_convexity_recovers_branches():
    """Zero mixture coeff on one branch returns the other branch."""
    lm = np.array([1.0, 2.0, 3.0])
    ld = np.array([-1.0, -2.0, -3.0])
    out_m, _ = tempered_combine(lm, ld, p=2.0, mix_model=1.0, mix_demo=0.0)
    out_d, _ = tempered_combine(lm, ld, p=2.0, mix_model=0.0, mix_demo=1.0)
    assert np.allclose(out_m, lm)
    assert np.allclose(out_d, ld)


# --- denoise_loop integration (synthetic, no robosuite) --------------------

def _make_quadratic_problem(H=4, action_dim=2, a_star=0.0):
    """Identity rollout + quadratic cost toward ``a_star`` (per-dim)."""
    def rollout_fn(tau):  # states == actions
        return np.asarray(tau, dtype=np.float32)

    def cost_fn(states, traj):
        t = np.asarray(traj, dtype=np.float64)
        return ((t - a_star) ** 2).sum(axis=(1, 2))

    return rollout_fn, cost_fn


def _params(H=4, action_dim=2, M=64, K=6):
    sched = np.linspace(0.5, 0.05, K).astype(np.float32)
    return MBDParams(
        H=H, M=M, K=K, action_dim=action_dim,
        lambda_=0.5, sigma_schedule=sched, clip_action=None,
    )


def test_tempered_loop_runs_and_logs_diagnostics():
    rng = np.random.default_rng(7)
    H, A = 4, 2
    rollout_fn, cost_fn = _make_quadratic_problem(H, A, a_star=0.0)
    params = _params(H, A)
    demo = np.full((H, A), 0.8, dtype=np.float32)  # demo far from cost-opt

    mu, log = denoise_loop(
        np.zeros((H, A), dtype=np.float32), rollout_fn, cost_fn, params, rng,
        combine_mode="tempered", demo_target=demo,
        sigma_demo=0.25, temper_p=0.0, prior_lambda=1.0,
    )
    assert mu.shape == (H, A)
    assert np.all(np.isfinite(mu))
    for e in log["per_iter"]:
        assert 0.0 <= e["gate_frac"] <= 1.0
        assert np.isfinite(e["ldem_minus_lmod"])


def test_product_pulls_toward_demo_relative_to_pure_cost():
    """temper_p=0 with demo pull lands between cost-opt (0) and demo (0.8)."""
    H, A = 4, 2
    rollout_fn, cost_fn = _make_quadratic_problem(H, A, a_star=0.0)
    params = _params(H, A)
    demo = np.full((H, A), 0.8, dtype=np.float32)

    mu_cost, _ = denoise_loop(
        np.zeros((H, A), dtype=np.float32), rollout_fn, cost_fn, params,
        np.random.default_rng(3),
    )  # pure MBD (no prior): should sit near 0
    mu_prod, _ = denoise_loop(
        np.zeros((H, A), dtype=np.float32), rollout_fn, cost_fn, params,
        np.random.default_rng(3),
        combine_mode="tempered", demo_target=demo,
        sigma_demo=0.5, temper_p=0.0, prior_lambda=5.0,
    )
    # Product target is pulled toward the demo (more positive) than pure cost.
    assert mu_prod.mean() > mu_cost.mean()
    assert mu_prod.mean() < demo.mean()


def test_score_mode_unchanged_when_no_prior():
    """combine_mode='score' (default) with no prior == legacy loop."""
    H, A = 4, 2
    rollout_fn, cost_fn = _make_quadratic_problem(H, A, a_star=0.3)
    params = _params(H, A)
    mu_a, _ = denoise_loop(
        np.zeros((H, A), dtype=np.float32), rollout_fn, cost_fn, params,
        np.random.default_rng(11),
    )
    mu_b, _ = denoise_loop(
        np.zeros((H, A), dtype=np.float32), rollout_fn, cost_fn, params,
        np.random.default_rng(11), combine_mode="score",
    )
    assert np.array_equal(mu_a, mu_b)


def test_sigma_mode_const_vs_decay_differ_with_prior():
    """#4: prior_sigma_mode toggles the pull width and changes the result."""
    H, A = 4, 2
    rollout_fn, cost_fn = _make_quadratic_problem(H, A, a_star=0.0)
    params = _params(H, A)
    demo = np.full((H, A), 0.8, dtype=np.float32)

    def score_factory(s0, K):
        state = {"a_bc": demo}

        def score_fn(mu, k):
            return (state["a_bc"] - mu).astype(np.float32)
        score_fn.state = state
        return score_fn

    sf = score_factory(None, params.K)
    mu_decay, _ = denoise_loop(
        demo.copy(), rollout_fn, cost_fn, params, np.random.default_rng(5),
        prior_score_fn=sf, prior_lambda=1.0, prior_sigma_mode="decay",
    )
    sf2 = score_factory(None, params.K)
    mu_const, _ = denoise_loop(
        demo.copy(), rollout_fn, cost_fn, params, np.random.default_rng(5),
        prior_score_fn=sf2, prior_lambda=1.0, prior_sigma_mode="const",
    )
    assert not np.array_equal(mu_decay, mu_const)


def test_adaptive_sigma_runs_and_logs():
    """#2: demo_sigma_mode='adaptive' runs, finite mu, valid diagnostics."""
    H, A = 4, 2
    rollout_fn, cost_fn = _make_quadratic_problem(H, A, a_star=0.0)
    params = _params(H, A)
    demo = np.full((H, A), 0.8, dtype=np.float32)
    mu, log = denoise_loop(
        np.zeros((H, A), dtype=np.float32), rollout_fn, cost_fn, params,
        np.random.default_rng(31),
        combine_mode="tempered", demo_target=demo,
        sigma_demo=0.5, temper_p=0.0, prior_lambda=1.0,
        demo_sigma_mode="adaptive", demo_sigma_adapt_beta=1.0,
    )
    assert mu.shape == (H, A) and np.all(np.isfinite(mu))
    for e in log["per_iter"]:
        assert 0.0 <= e["gate_frac"] <= 1.0
        assert np.isfinite(e["ldem_minus_lmod"])


def test_adaptive_sigma_differs_from_const():
    """#2: adaptive scaling changes the result vs fixed sigma (gap != 0)."""
    H, A = 4, 2
    rollout_fn, cost_fn = _make_quadratic_problem(H, A, a_star=0.0)
    params = _params(H, A)
    demo = np.full((H, A), 0.8, dtype=np.float32)  # off cost-optimum -> gap!=0
    common = dict(
        combine_mode="tempered", demo_target=demo,
        sigma_demo=0.5, temper_p=0.0, prior_lambda=3.0,
    )
    mu_const, _ = denoise_loop(
        np.zeros((H, A), dtype=np.float32), rollout_fn, cost_fn, params,
        np.random.default_rng(5), demo_sigma_mode="const", **common,
    )
    mu_adapt, _ = denoise_loop(
        np.zeros((H, A), dtype=np.float32), rollout_fn, cost_fn, params,
        np.random.default_rng(5), demo_sigma_mode="adaptive",
        demo_sigma_adapt_beta=1.0, **common,
    )
    assert not np.array_equal(mu_const, mu_adapt)


def test_gate_diagnostic_is_passive():
    """#3/§6.3: log_gate_diagnostic logs gate_frac/ldem but never changes mu."""
    H, A = 4, 2
    rollout_fn, cost_fn = _make_quadratic_problem(H, A, a_star=0.3)
    params = _params(H, A)
    demo = np.full((H, A), 0.8, dtype=np.float32)

    mu_plain, _ = denoise_loop(
        np.zeros((H, A), dtype=np.float32), rollout_fn, cost_fn, params,
        np.random.default_rng(21),
    )
    mu_diag, log = denoise_loop(
        np.zeros((H, A), dtype=np.float32), rollout_fn, cost_fn, params,
        np.random.default_rng(21),
        log_gate_diagnostic=True, demo_target=demo, sigma_demo=1.0,
    )
    # Passive: identical trajectory, but diagnostics are populated.
    assert np.array_equal(mu_plain, mu_diag)
    for e in log["per_iter"]:
        assert 0.0 <= e["gate_frac"] <= 1.0
        assert np.isfinite(e["ldem_minus_lmod"])


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
