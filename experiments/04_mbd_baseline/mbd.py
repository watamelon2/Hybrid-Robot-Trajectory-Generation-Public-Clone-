"""Online Model-Based Diffusion (MBD) denoising loop -- algorithm only.

Following Pan et al. 2024 (https://arxiv.org/abs/2407.01573), we draw
samples ``tau ~ p*(tau) ∝ exp(-J(tau)/lambda)`` over a horizon-``H``
trajectory by iterating a softmax-weighted Monte-Carlo update of the
trajectory mean while annealing the perturbation noise.

This module is **algorithm only**. It takes a ``rollout_fn`` and a
``cost_fn`` as injected callables so the same ``denoise_loop`` works
for the 2-D LQ sanity check (``lq_sanity.py``) and the full MJX-based
can task (``run_can.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np


@dataclass(frozen=True)
class MBDParams:
    """Hyperparameters of the MBD denoising loop.

    Attributes:
        H: horizon (number of action steps in a trajectory).
        M: number of perturbed copies sampled per denoising step.
        K: number of denoising iterations.
        action_dim: dimensionality of one action.
        lambda_: temperature controlling the softmax sharpness.
        sigma_schedule: per-step perturbation std, length-K from large
            to small (e.g. cosine schedule from 1.0 down to 0.05).
        clip_action: optional ``(low, high)`` to clip every sampled
            action to (e.g. the env's action_spec).
    """

    H: int
    M: int
    K: int
    action_dim: int
    lambda_: float = 1.0
    sigma_schedule: np.ndarray = field(default_factory=lambda: np.array([]))
    clip_action: tuple[np.ndarray, np.ndarray] | None = None


class RolloutFn(Protocol):
    """Sim rollout signature: ``(traj_batch) -> states_batch``.

    Inputs:
        traj_batch: ``(M, H, action_dim)`` numpy/jax array of action
            sequences sharing the same initial state.
    Returns:
        states_batch: arbitrary -- whatever ``cost_fn`` will consume.
            Typical shapes: ``(M, H+1, state_dim)`` for state traces.
    """

    def __call__(self, traj_batch: np.ndarray) -> np.ndarray: ...


class CostFn(Protocol):
    """Per-trajectory scalar cost: ``(states_batch, traj_batch) -> (M,)``.

    Sign convention: lower is better (we minimise).
    """

    def __call__(
        self, states_batch: np.ndarray, traj_batch: np.ndarray
    ) -> np.ndarray: ...


def cosine_sigma_schedule(
    K: int, sigma_max: float = 1.0, sigma_min: float = 0.05
) -> np.ndarray:
    """Cosine-shaped descent from ``sigma_max`` to ``sigma_min`` over K steps.

    Roughly the schedule used in Pan et al.; the exact shape of the
    schedule does not seem to matter much as long as it is monotone
    and ends near zero.
    """
    ts = np.linspace(0.0, np.pi, K, dtype=np.float32)
    cos = (1.0 + np.cos(ts)) * 0.5  # 1 -> 0
    return sigma_min + (sigma_max - sigma_min) * cos


def tempered_combine(
    log_w_model: np.ndarray,
    log_w_demo: np.ndarray,
    p: float,
    *,
    mix_model: float = 1.0,
    mix_demo: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Generalised-mean fusion of two per-sample log-weights.

    Implements the tempered-mixture family from
    ``docs/theory/09_mbd_data_guided_comparison.md`` §5/§6.2 that
    interpolates between our product-of-experts and MBD-Demo's
    max-gate (Pan et al. 2024, Eqs. 12-13):

        log w_i = (1/p) * logsumexp_j( p * log_w_j_i + log mix_j )

    Endpoints:
      * ``p -> +inf`` : ``max_j(log_w_j_i)`` -- MBD-Demo's hard
        max-gate (Eq. 13), an OR/selection between model and demo.
      * ``p -> 0``    : ``sum_j mix_j_norm * log_w_j_i`` -- the
        log-linear pool, i.e. our product-of-experts (AND). With
        ``mix_model=1`` and ``mix_demo=lambda_prior`` this is exactly
        ``log p_model + lambda_prior * log p_demo``.

    Parameters
    ----------
    log_w_model, log_w_demo
        ``(M,)`` per-sample log-weights of the model (cost-Boltzmann)
        and demo (Gaussian pull) branches. Unnormalised; any additive
        constant cancels in the downstream softmax.
    p
        Tempering exponent. ``np.inf`` -> max-gate; values near ``0``
        -> product. Negative ``p`` gives the min-gate (not used in
        the report but supported for completeness).
    mix_model, mix_demo
        Non-negative mixture coefficients. Their ratio sets the
        demo's relative pull at the ``p -> 0`` product endpoint
        (``mix_demo`` plays the role of ``lambda_prior``).

    Returns
    -------
    (log_w_combined, gate_frac)
        ``log_w_combined`` is ``(M,)``; ``gate_frac`` is the fraction
        of samples where the demo branch strictly wins
        (``log_w_demo > log_w_model``) -- MBD-Demo's ``eta = 1``
        fraction, used by the §6.3 likelihood-ratio diagnostic.
    """
    lm = np.asarray(log_w_model, dtype=np.float64)
    ld = np.asarray(log_w_demo, dtype=np.float64)
    if lm.shape != ld.shape:
        raise ValueError(
            f"log_w_model shape {lm.shape} != log_w_demo shape {ld.shape}"
        )
    if mix_model < 0.0 or mix_demo < 0.0:
        raise ValueError("mix_model and mix_demo must be >= 0")
    if mix_model == 0.0 and mix_demo == 0.0:
        raise ValueError("at least one of mix_model/mix_demo must be > 0")

    gate_frac = float(np.mean(ld > lm)) if ld.size else 0.0

    # Degenerate single-branch cases (one mixture coeff is zero).
    if mix_demo == 0.0:
        return lm.copy(), gate_frac
    if mix_model == 0.0:
        return ld.copy(), gate_frac

    log_mix_m = float(np.log(mix_model))
    log_mix_d = float(np.log(mix_demo))

    if not np.isfinite(p):
        if p > 0:  # max-gate (MBD-Demo)
            combined = np.maximum(lm + log_mix_m, ld + log_mix_d)
        else:      # min-gate
            combined = np.minimum(lm + log_mix_m, ld + log_mix_d)
        return combined, gate_frac

    if abs(p) < 1e-8:  # product / log-linear pool (p -> 0 limit)
        s = mix_model + mix_demo
        combined = (mix_model / s) * lm + (mix_demo / s) * ld
        return combined, gate_frac

    # General finite p: (1/p) logsumexp over the two weighted branches.
    a = p * lm + log_mix_m
    b = p * ld + log_mix_d
    m = np.maximum(a, b)
    combined = (m + np.log(np.exp(a - m) + np.exp(b - m))) / p
    return combined, gate_frac


IterCallback = Callable[[int, np.ndarray, float, np.ndarray, dict], None]
"""Type alias for ``denoise_loop``'s optional per-iteration hook.

Signature: ``(k, mu_k, sigma_k, costs_k, log_entry) -> None``.

Receives the just-completed iteration index ``k``, the updated mean
trajectory ``mu_k`` of shape ``(H, action_dim)``, the perturbation std
used at this step, the per-perturbation cost vector of shape ``(M,)``,
and the log dict that was just appended. Used by ``run_can.py`` to
stream a live mujoco preview of the planning process; pure inspection,
no return value.
"""


PriorScoreFn = Callable[[np.ndarray, int], np.ndarray]
"""Optional per-step data-prior score hook.

Signature: ``(mu_k, k) -> score`` where ``score`` has the same shape
as ``mu_k`` (``(H, action_dim)``). Convention follows score-based /
flow-matching priors: ``score`` approximates ``grad log p_data(mu_k)``
(or, for a flow-matching velocity field, the velocity that pushes
``mu_k`` toward the data manifold at the current schedule).

When ``prior_score_fn`` is provided to ``denoise_loop``, the per-step
mean update is::

    mu_new = (1 - lr_k) * sum_m w_m * tau_m + lr_k * (mu + sigma_k**2 * score(mu, k))

with ``lr_k = prior_lambda * (sigma_k / sigma_max)`` (linear in the
noise level). Two endpoints recover the limits used in the report:

  - ``prior_score_fn=None`` or ``prior_lambda=0``: identical to the
    pre-existing zero-prior loop. Backwards-compatible.
  - ``prior_lambda=inf``: planner contribution is ignored and the
    update is pure prior gradient ascent on ``log p_data``.
"""


def denoise_loop(
    initial_mean: np.ndarray,
    rollout_fn: RolloutFn,
    cost_fn: CostFn,
    params: MBDParams,
    rng: np.random.Generator,
    iter_callback: IterCallback | None = None,
    *,
    prior_score_fn: PriorScoreFn | None = None,
    prior_lambda: float = 0.0,
    prior_lambda_schedule: np.ndarray | None = None,
    prior_sigma_mode: str = "decay",
    combine_mode: str = "score",
    demo_target: np.ndarray | None = None,
    demo_cost: float | None = None,
    sigma_demo: float = 0.25,
    demo_sigma_mode: str = "const",
    demo_sigma_adapt_beta: float = 1.0,
    temper_p: float = 0.0,
    log_gate_diagnostic: bool = False,
) -> tuple[np.ndarray, dict]:
    """Run K denoising iterations and return the final mean trajectory.

    Args:
        initial_mean: ``(H, action_dim)`` -- the starting trajectory
            mean ``mu_K``. Use zeros for "no prior", or a BC rollout
            for exp 05's hybrid setup.
        rollout_fn: see ``RolloutFn``.
        cost_fn: see ``CostFn``.
        params: ``MBDParams``.
        rng: numpy generator for the per-step Gaussian perturbations.
        iter_callback: optional hook invoked after each iteration. See
            ``IterCallback`` for the signature; default ``None``.
        prior_score_fn: optional ``(mu_k, k) -> score_array`` hook for
            the per-step composite update used by Tier 1b of the
            roadmap. Default ``None`` reproduces the no-prior loop
            verbatim. See ``PriorScoreFn`` docstring for the math.
        prior_lambda: weight of the prior contribution. ``0`` (default)
            disables the prior even if ``prior_score_fn`` is provided;
            ``np.inf`` ignores the planner contribution and is
            equivalent to gradient ascent on the prior alone.
        prior_lambda_schedule: optional per-iteration multiplier on
            the prior weight (A2, added 2026-05-20).  Shape ``(K,)``.
            When ``None`` (default) the effective rate is the
            historical ``prior_lambda * (sigma_k / sigma_max)``
            (i.e. constant=1 multiplier).  When provided, the
            effective rate becomes ``prior_lambda *
            prior_lambda_schedule[k] * (sigma_k / sigma_max)``.
            Use cases:
              * ``np.linspace(1, 0, K)`` (linear anneal-out): bridge
                strong early when noise is high, faded out at the
                low-sigma tail so MBD's local refinement dominates.
              * ``np.linspace(0, 1, K)`` (inverse-linear anneal-in):
                bridge weak early, strong late.
              * ``np.ones(K)``: same as ``None``.
        prior_sigma_mode: scaling of the score-pull step in the legacy
            ``combine_mode="score"`` path. ``"decay"`` (default) uses
            the historical ``mu + sigma_k**2 * score`` (the
            ``sigma_t``-annealed pull). ``"const"`` uses
            ``mu + sigma_max**2 * score`` (a fixed-width pull, MBD-style)
            -- the §8.3 / follow-up #4 ablation. No effect when the
            prior is off.
        combine_mode: ``"score"`` (default, legacy mean-space blend of
            planner + prior pull) or ``"tempered"`` (per-sample
            tempered-mixture reweighting of the model vs demo branches,
            §6.1/§6.2). ``"tempered"`` ignores ``prior_score_fn`` and
            instead uses ``demo_target`` directly.
        demo_target: ``(H, action_dim)`` demonstration / warm-start
            trajectory used as the Gaussian-pull centre in
            ``combine_mode="tempered"``. Required for tempered mode.
        demo_cost: scalar cost ``J(demo_target)`` providing the demo
            branch's fixed Boltzmann constant ``exp(-J_demo/lambda_)``
            (MBD Eq. 11's ``p_J(Y_demo)``). When ``None`` it is computed
            once by rolling out ``demo_target``.
        sigma_demo: observation-noise std of the demo Gaussian
            ``N(tau | demo_target, sigma_demo**2 I)``.
        demo_sigma_mode: ``"const"`` (fixed ``sigma_demo``, MBD-faithful)
            or ``"decay"`` (``sigma_demo * sigma_k / sigma_max``, our
            annealed variant) -- the #4 ablation for the tempered path.
        temper_p: tempering exponent for ``tempered_combine``.
            ``0`` -> product (PoE); ``np.inf`` -> MBD-Demo max-gate;
            finite ``>0`` -> the interpolation of §6.2/#7. ``mix_demo``
            is taken from ``prior_lambda`` so ``temper_p=0`` reproduces
            our product with weight ``lambda_prior``.
        log_gate_diagnostic: when ``True`` and ``demo_target`` is given,
            compute and log the §6.3 likelihood-ratio diagnostic
            (``gate_frac`` and ``ldem_minus_lmod``) every iteration
            **without changing the update** -- a passive probe usable on
            the legacy product/score cells (and pure-MBD) to test the §5
            prediction that the off-distribution collapse coincides with
            ``ldem << lmod``. Costs one extra demo roll-out per call.

    Returns:
        mu_0: ``(H, action_dim)`` final denoised trajectory.
        log: dict with per-iteration stats (best cost, mean cost,
            ess, etc.) for the README's diagnostic plots.

    Implementation note:
        At each iteration k = 0, ..., K-1 we:
          1. Draw ``M`` perturbed copies ``tau_m = mu + sigma_k * eps_m``
             with ``eps_m ~ N(0, I)``.
          2. (Optionally) clip them into ``params.clip_action`` so the
             roll-out function never sees out-of-spec values.
          3. Roll them out through ``rollout_fn`` and score with
             ``cost_fn``.
          4. Form softmax weights ``w_m = exp(-(c_m - c_min)/lambda)``,
             normalise, and update ``mu = sum_m w_m * tau_m``.
        ``c_m - c_min`` keeps the softmax numerically stable; the
        constant cancels in the normalisation. Equivalent to the
        Pan-et-al "score gradient" update because the weights sum to 1
        and ``mu_new = mu + sigma_k^2 * score_estimate``.
    """
    mu = np.asarray(initial_mean, dtype=np.float32).copy()
    H, action_dim = mu.shape
    if action_dim != int(params.action_dim):
        raise ValueError(
            f"initial_mean has action_dim={action_dim}, "
            f"params.action_dim={params.action_dim}"
        )
    if H != int(params.H):
        raise ValueError(
            f"initial_mean has H={H}, params.H={params.H}"
        )

    schedule = np.asarray(params.sigma_schedule, dtype=np.float32)
    if schedule.shape != (params.K,):
        raise ValueError(
            f"sigma_schedule has shape {schedule.shape}, "
            f"expected ({params.K},)"
        )

    M = int(params.M)
    K = int(params.K)
    lambda_ = float(params.lambda_)
    if lambda_ <= 0.0:
        raise ValueError(f"lambda_ must be > 0, got {lambda_}")

    log: dict = {"per_iter": []}
    sigma_max = float(schedule.max()) if schedule.size > 0 else 1.0
    use_prior = prior_score_fn is not None and prior_lambda != 0.0
    prior_lambda_f = float(prior_lambda)
    prior_lambda_is_inf = use_prior and not np.isfinite(prior_lambda_f)

    if prior_sigma_mode not in ("decay", "const"):
        raise ValueError(
            f"prior_sigma_mode must be 'decay' or 'const', got "
            f"{prior_sigma_mode!r}"
        )
    if combine_mode not in ("score", "tempered"):
        raise ValueError(
            f"combine_mode must be 'score' or 'tempered', got "
            f"{combine_mode!r}"
        )
    if demo_sigma_mode not in ("const", "decay", "adaptive"):
        raise ValueError(
            f"demo_sigma_mode must be 'const', 'decay' or 'adaptive', got "
            f"{demo_sigma_mode!r}"
        )

    # Tempered/gated combination (§6.1/§6.2) setup.  Reweights the
    # per-sample model (cost-Boltzmann) vs demo (Gaussian-pull) log-
    # weights via ``tempered_combine``; ``temper_p=0`` -> product,
    # ``temper_p=inf`` -> MBD-Demo max-gate.
    use_tempered = combine_mode == "tempered"
    # The demo log-weight is needed when we *fuse* with it (tempered) or
    # merely *measure* it (the §6.3 likelihood-ratio diagnostic, which is
    # passive logging on any run -- including the legacy product/score
    # cells -- when ``log_gate_diagnostic`` is on and a demo is given).
    want_demo_branch = use_tempered or (
        log_gate_diagnostic and demo_target is not None
    )
    mix_demo = 1.0
    if want_demo_branch:
        if demo_target is None:
            raise ValueError(
                "combine_mode='tempered' requires demo_target"
            )
        demo_tgt = np.asarray(demo_target, dtype=np.float32)
        if demo_tgt.shape != mu.shape:
            raise ValueError(
                f"demo_target shape {demo_tgt.shape} != mu shape {mu.shape}"
            )
        if float(sigma_demo) <= 0.0:
            raise ValueError(f"sigma_demo must be > 0, got {sigma_demo}")
        # Demo branch's fixed Boltzmann constant ``p_J(Y_demo)``
        # (MBD Eq. 11).  Computed once by rolling out the demo target
        # when not supplied.
        if demo_cost is None:
            demo_states = rollout_fn(demo_tgt[None].astype(np.float32))
            demo_cost_val = float(
                np.asarray(
                    cost_fn(demo_states, demo_tgt[None]), dtype=np.float64
                ).reshape(-1)[0]
            )
        else:
            demo_cost_val = float(demo_cost)
        if use_tempered:
            mix_demo = (
                float(prior_lambda_f) if np.isfinite(prior_lambda_f) else 1.0
            )
            if mix_demo < 0.0:
                raise ValueError(
                    "prior_lambda (mix_demo) must be >= 0 in tempered mode"
                )

    # Per-iteration multiplier on the prior weight (A2).  Default
    # ``None`` keeps the historical constant=1 multiplier verbatim.
    if prior_lambda_schedule is None:
        lambda_sched = None
    else:
        lambda_sched = np.asarray(prior_lambda_schedule, dtype=np.float32)
        if lambda_sched.shape != (K,):
            raise ValueError(
                f"prior_lambda_schedule has shape {lambda_sched.shape}, "
                f"expected ({K},)"
            )

    for k in range(K):
        sigma = float(schedule[k])
        eps = rng.standard_normal((M, H, action_dim)).astype(np.float32)
        tau = mu[None, :, :] + sigma * eps  # (M, H, action_dim)

        if params.clip_action is not None:
            low, high = params.clip_action
            tau = np.clip(tau, low, high).astype(np.float32)

        states = rollout_fn(tau)
        costs = np.asarray(cost_fn(states, tau), dtype=np.float64)
        if costs.shape != (M,):
            raise ValueError(
                f"cost_fn returned shape {costs.shape}, expected ({M},)"
            )

        # Numerically stable softmax: subtract min before exponentiating.
        shifted = (costs - float(costs.min())) / lambda_
        weights = np.exp(-shifted)
        weights /= weights.sum()
        weights_f32 = weights.astype(np.float32)

        # Planner contribution: weighted mean of perturbed trajectories.
        # Equivalent to "mu + sigma^2 * score" since sum(w) = 1.
        mu_planner = (weights_f32[:, None, None] * tau).sum(axis=0)

        prior_norm = 0.0
        prior_step_norm = 0.0
        lr_k = 0.0
        gate_frac = float("nan")
        ldem_minus_lmod = float("nan")
        # Per-sample model-vs-demo log-weights.  Computed whenever we
        # need the demo branch -- to fuse (tempered) OR just to measure
        # (the §6.3 likelihood-ratio diagnostic, passive on any run).
        if want_demo_branch:
            l_mod = -costs / lambda_  # (M,)
            if demo_sigma_mode == "decay":
                sd_mult = sigma / max(sigma_max, 1e-8)
            elif demo_sigma_mode == "adaptive":
                # Off-distribution rescue (follow-up #2, §6c): shrink the
                # demo Gaussian (-> gate fires) when the model's BEST
                # rollout is worse than the demo, widen it (-> trust the
                # model) when the model beats the demo.  The gap is
                # measured on the cost-Boltzmann scale ``lambda_`` so it
                # matches the log-weights' units; clipped to keep sd in a
                # sane band.  This is the only static-sigma fix for the
                # negative #1/#8 result (no single fixed sigma both
                # rescues off-dist and preserves on-dist).
                gap = (float(costs.min()) - demo_cost_val) / lambda_
                sd_mult = float(np.clip(
                    np.exp(-float(demo_sigma_adapt_beta) * gap),
                    0.05, 20.0,
                ))
            else:
                sd_mult = 1.0
            sd = sigma_demo * sd_mult
            sd = max(float(sd), 1e-6)
            sq = ((tau - demo_tgt[None]) ** 2).sum(axis=(1, 2))  # (M,)
            l_dem = -sq / (2.0 * sd * sd) - demo_cost_val / lambda_
            gate_frac = float(np.mean(l_dem > l_mod))
            ldem_minus_lmod = float(np.mean(l_dem - l_mod))

        if use_tempered:
            # Reweight per sample: temper_p=0 -> product, inf -> max-gate.
            log_w, _ = tempered_combine(
                l_mod, l_dem, temper_p,
                mix_model=1.0, mix_demo=mix_demo,
            )
            log_w = log_w - float(log_w.max())
            w = np.exp(log_w)
            w /= w.sum()
            mu = (w[:, None, None].astype(np.float32) * tau).sum(axis=0)
            ess = float(1.0 / (w ** 2).sum())
        elif use_prior:
            score = np.asarray(
                prior_score_fn(mu, k), dtype=np.float32
            )
            if score.shape != mu.shape:
                raise ValueError(
                    f"prior_score_fn returned shape {score.shape}, "
                    f"expected {mu.shape}"
                )
            prior_norm = float(np.linalg.norm(score))
            # ``prior_sigma_mode`` toggles the pull width (#4 ablation):
            # 'decay' = historical sigma_t-annealed pull (sigma_k**2);
            # 'const' = fixed-width MBD-style pull (sigma_max**2).
            pull_scale = (sigma * sigma) if prior_sigma_mode == "decay" else (
                sigma_max * sigma_max
            )
            mu_prior = mu + pull_scale * score
            if params.clip_action is not None:
                low, high = params.clip_action
                mu_prior = np.clip(mu_prior, low, high).astype(np.float32)
            prior_step_norm = float(np.linalg.norm(mu_prior - mu))

            if prior_lambda_is_inf:
                mu = mu_prior.astype(np.float32)
            else:
                sched_mult = (
                    1.0 if lambda_sched is None else float(lambda_sched[k])
                )
                lr_k = (
                    prior_lambda_f
                    * sched_mult
                    * (sigma / max(sigma_max, 1e-8))
                )
                lr_k = float(np.clip(lr_k, 0.0, 1.0))
                mu = ((1.0 - lr_k) * mu_planner + lr_k * mu_prior).astype(
                    np.float32
                )
            ess = float(1.0 / (weights ** 2).sum())
        else:
            mu = mu_planner
            ess = float(1.0 / (weights ** 2).sum())

        log_entry = {
            "iter": int(k),
            "sigma": sigma,
            "best_cost": float(costs.min()),
            "mean_cost": float(costs.mean()),
            "ess": ess,
            "prior_lr": float(lr_k),
            "prior_score_norm": float(prior_norm),
            "prior_step_norm": float(prior_step_norm),
            # Tempered/gated diagnostics (NaN in legacy/score mode).
            # ``gate_frac`` = MBD-Demo eta=1 fraction; ``ldem_minus_lmod``
            # is the §6.3 likelihood-ratio diagnostic.
            "gate_frac": float(gate_frac),
            "ldem_minus_lmod": float(ldem_minus_lmod),
        }
        log["per_iter"].append(log_entry)

        if iter_callback is not None:
            iter_callback(k, mu, sigma, costs, log_entry)

    return mu, log
