"""Low-frequency action-basis parameterisation for MBD perturbations.

The Exp 05 README hypothesises that millimetre-scale grasp
alignment is the bottleneck and that the planner's ``M=16``
cosine-noise perturbations rarely close the gap. One reason: the
default perturbation lives in raw per-step action space, so the
``M`` samples explore a roughly ``H * action_dim``-dimensional
hyperball, the vast majority of which is high-frequency jitter
that the controller filters out anyway.

This module provides an alternative: perturb in a low-frequency
*basis* of size ``B << H``. Sampling ``M`` perturbations in a
B-dim space gives the same "perturbation budget" but every
sample now corresponds to a coherent shift across many
timesteps. We use a DCT-II basis (raised cosine), which is
orthonormal and has a closed-form expression that avoids any
extra dependencies.

Pairs with a small wrapper around ``denoise_loop`` that overrides
the per-iteration perturbation step. We do **not** modify the
upstream ``mbd.py``; instead this module subclasses behaviour by
replacing the ``rng.standard_normal`` call with a basis-projected
version. See ``denoise_loop_basis`` below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_EXP_04 = _THIS_DIR.parent / "04_mbd_baseline"
if str(_EXP_04) not in sys.path:
    sys.path.insert(0, str(_EXP_04))

from mbd import MBDParams  # noqa: E402,F401


def dct_basis(H: int, n_modes: int) -> np.ndarray:
    """Return a ``(n_modes, H)`` orthonormal DCT-II basis matrix.

    ``B[m, t] = sqrt(2/H) * cos(pi * (t + 0.5) * m / H)``  (m >= 1)
    plus the ``m=0`` constant row at ``sqrt(1/H)``.

    A perturbation ``eps_basis`` of shape ``(M, n_modes, action_dim)``
    is mapped to a per-step perturbation ``(M, H, action_dim)`` via
    ``eps = einsum('mna,nh->mha', eps_basis, B)``.
    """
    if n_modes < 1 or n_modes > H:
        raise ValueError(f"n_modes={n_modes} must satisfy 1 <= n_modes <= H={H}")
    t = np.arange(H, dtype=np.float64)
    m = np.arange(n_modes, dtype=np.float64)[:, None]
    B = np.sqrt(2.0 / H) * np.cos(np.pi * (t + 0.5) * m / H)
    B[0, :] = np.sqrt(1.0 / H)
    return B.astype(np.float32)


def denoise_loop_basis(
    initial_mean: np.ndarray,
    rollout_fn,
    cost_fn,
    params: MBDParams,
    rng: np.random.Generator,
    *,
    n_modes: int = 8,
    iter_callback=None,
    prior_score_fn=None,
    prior_lambda: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """MBD with perturbations sampled in a length-``n_modes`` DCT basis.

    Behaviour exactly matches ``mbd.denoise_loop`` (including the new
    ``prior_score_fn`` hook) except the per-iteration noise tensor
    ``eps`` is constructed as ``eps_basis @ B`` instead of being
    drawn directly. This preserves the cosine-schedule contract on
    the *full* per-step std at each iteration (we rescale so the
    induced per-step std matches ``sigma_k``).
    """
    mu = np.asarray(initial_mean, dtype=np.float32).copy()
    H, action_dim = mu.shape
    if H != int(params.H) or action_dim != int(params.action_dim):
        raise ValueError("initial_mean shape disagrees with params")
    schedule = np.asarray(params.sigma_schedule, dtype=np.float32)
    M, K = int(params.M), int(params.K)
    lambda_ = float(params.lambda_)
    sigma_max = float(schedule.max()) if schedule.size else 1.0

    B = dct_basis(H, n_modes)  # (n_modes, H)
    # Each per-step entry is sum_m eps_basis_m * B[m, t]; under
    # eps_basis ~ N(0, I) the per-step variance is sum_m B[m, t]^2.
    per_step_std = np.linalg.norm(B, axis=0)  # (H,)
    # Normalise so that the full-resolution per-step std matches
    # sigma_k after the rescale.
    norm = float(per_step_std.mean()) or 1.0

    use_prior = prior_score_fn is not None and prior_lambda != 0.0
    prior_lambda_f = float(prior_lambda)
    prior_lambda_inf = use_prior and not np.isfinite(prior_lambda_f)

    log: dict = {"per_iter": []}
    for k in range(K):
        sigma = float(schedule[k])
        eps_basis = rng.standard_normal((M, n_modes, action_dim)).astype(np.float32)
        # eps[M, H, A] = (sigma / norm) * sum_m eps_basis[M, m, A] * B[m, H]
        eps = (sigma / norm) * np.einsum("mna,nh->mha", eps_basis, B)
        tau = mu[None, :, :] + eps
        if params.clip_action is not None:
            low, high = params.clip_action
            tau = np.clip(tau, low, high).astype(np.float32)

        states = rollout_fn(tau)
        costs = np.asarray(cost_fn(states, tau), dtype=np.float64)
        shifted = (costs - float(costs.min())) / lambda_
        weights = np.exp(-shifted); weights /= weights.sum()
        weights_f32 = weights.astype(np.float32)
        mu_planner = (weights_f32[:, None, None] * tau).sum(axis=0)

        prior_step_norm = 0.0
        lr_k = 0.0
        if use_prior:
            score = np.asarray(
                prior_score_fn(mu, k), dtype=np.float32
            )
            mu_prior = mu + (sigma * sigma) * score
            if params.clip_action is not None:
                low, high = params.clip_action
                mu_prior = np.clip(mu_prior, low, high).astype(np.float32)
            prior_step_norm = float(np.linalg.norm(mu_prior - mu))
            if prior_lambda_inf:
                mu = mu_prior.astype(np.float32)
            else:
                lr_k = float(np.clip(
                    prior_lambda_f * (sigma / max(sigma_max, 1e-8)),
                    0.0, 1.0,
                ))
                mu = ((1.0 - lr_k) * mu_planner + lr_k * mu_prior).astype(
                    np.float32
                )
        else:
            mu = mu_planner

        ess = float(1.0 / (weights ** 2).sum())
        log_entry = {
            "iter": int(k),
            "sigma": sigma,
            "best_cost": float(costs.min()),
            "mean_cost": float(costs.mean()),
            "ess": ess,
            "prior_lr": lr_k,
            "prior_step_norm": prior_step_norm,
            "n_modes": int(n_modes),
        }
        log["per_iter"].append(log_entry)
        if iter_callback is not None:
            iter_callback(k, mu, sigma, costs, log_entry)

    return mu, log
