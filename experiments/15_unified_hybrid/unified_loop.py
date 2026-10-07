"""Unified hybrid plan-and-execute loop.

Fuses the *score-bridge* per-step prior correction (exp 07) with
the *MPC chunking* outer loop (exp 05). Replaces the divergent
``run_can.py`` implementations in exp 05 / 07 / 10.

The fusion is structurally simple:

1. **Outer loop** is exp 05's MPC: split the H-step execution
   into ``n_replans + 1`` chunks of ``chunk_steps`` env-steps
   each (last chunk takes the remainder). At each chunk boundary,
   snapshot the live env state and re-plan from there.

2. **Inner loop** is exp 04's ``denoise_loop`` with exp 07's
   ``prior_score_fn`` argument wired in. At every MBD denoising
   iteration ``k``, the loop adds ``lambda_prior * grad_log p_prior(mu_k)``
   to its update -- which is exactly the score-bridge formulation
   in eq. 1 of the Week-4 report. Setting ``lambda_prior=0``
   recovers exp 05's warm-start-only behaviour bit-for-bit;
   setting ``n_replans=0`` recovers exp 07's single-shot.

3. **Prior is pluggable** via the ``prior_funcs`` argument, which
   is a tuple ``(initial_mean_fn, score_fn_factory)`` produced
   by ``run_can._load_prior``. The two implemented priors mirror
   exp 07's --prior choices: ``bc`` (deterministic MLP from exp
   03, score is ``a_bc - mu``) and ``flow`` (flow-matching from
   exp 08, score is the FM-derived approximate score).

Returns the same dict shape as exp 05's ``mpc_plan_and_execute``
plus a ``per_chunk_score_fn_state`` debug field that records the
score-fn closure state per chunk (useful for diagnosing whether
the BC anchor was updated correctly each chunk).
"""

from __future__ import annotations

import collections
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

# This module is imported by run_can.py *after* it has done its
# sys.path.insert dance for the digit-prefixed exp dirs, so the
# following bare imports just work.
from common import check_success, set_sim_state  # noqa: E402
from mbd import MBDParams, denoise_loop  # noqa: E402


# Type alias for clarity.
InitialMeanFn = Callable[[Any, np.ndarray, np.ndarray], np.ndarray]
ScoreFnFactory = Callable[[np.ndarray, int], Callable]
# (chunk_idx) -> seed_uint32.  When supplied, the loop calls
# ``torch.manual_seed(...)`` before each chunk's ``initial_mean_fn``,
# eliminating the prior sampler's dependence on torch's global RNG
# state and making (warm-start, MBD perturbations, exec trajectory)
# fully deterministic per (base_seed, demo, rollout, chunk).
# Discovered by the 2026-05-16 Phase 8 diagnostic -- without this
# fix, the v_theta_v2 prior's success rate at a fixed cell drifts
# by ~8/60 between drivers because each driver leaves torch's
# global state in a different position by the time
# ``sample_initial_mean`` runs.
WarmstartSeedFn = Callable[[int], int]


def _snapshot_sim_state(env) -> np.ndarray:
    """Flat MuJoCo state vector for the live env.

    Copied verbatim from exp 05 so MPC chunks pick up where the
    previous chunk left off."""
    return np.asarray(env.sim.get_state().flatten())


def _make_obs_vec_for_prior(
    env, obs_keys: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Return the live env's flat obs in the prior's OBS_KEYS order.

    ``obs_keys`` controls which keys (and in which order) are
    concatenated; defaults to exp 03's Can ``OBS_KEYS`` for backward
    compat with pre-Slate-F callers.  Cross-task callers (Slate F:
    Square) pass ``obs_keys=get_obs_keys("square")`` instead.

    Copied from exp 07 ``run_can._make_obs_vec_for_prior`` -- the
    score-bridge priors all condition on this exact concatenation.

    Phase-C (2026-05-29) extension: ``obs_keys`` may include the
    DP paper's synthetic-relative-pose keys (``Can_to_eef_pos_synth``,
    ``Can_to_eef_quat_synth``) which are not exposed by robosuite's
    obs dict.  Those are computed at runtime from their base sources
    (``Can_pos`` + ``robot0_eef_pos``, etc.) via
    ``compute_synthetic_obs_keys``, mirroring the train-time
    construction in ``dataset_v3.load_chunks``.  Bit-identical train
    and runtime features by construction -- no robosuite-1.4/1.5
    reference-frame divergence.
    """
    if obs_keys is None:
        from dataset import OBS_KEYS  # exp 03's Can obs ordering

        obs_keys = OBS_KEYS

    obs_dict = env._get_observations(force_update=True)

    # Synthetic-key support: if any requested key is a synth derived
    # from base obs keys (e.g. Can_to_eef_pos_synth), compute it and
    # inject into the obs_dict before the per-key lookup below.
    # ``get_synthetic_keys(task)`` returns an empty dict for tasks
    # that don't declare synthetics, so this is a no-op for the
    # legacy 46-D ``can`` schema and for ``square`` / ``lift``.
    synth_needed = [k for k in obs_keys if k.endswith("_synth")]
    if synth_needed:
        from priors_zerotrain import (
            get_synthetic_keys, compute_synthetic_obs_keys,
        )
        # Aggregate all task-level synthetic recipes that might apply.
        # Iterate over tasks that could supply these synth keys; the
        # union covers ``can_dp`` today, ``lift_dp`` / ``square_dp`` in
        # the future without re-touching this code path.
        synth_recipes: dict[str, tuple[str, str]] = {}
        for _task in ("can_dp", "lift_dp", "square_dp"):
            for k, srcs in get_synthetic_keys(_task).items():
                if k in synth_needed and k not in synth_recipes:
                    synth_recipes[k] = srcs
        # Build the (1, D)-flat base-key arrays compute_synthetic
        # expects (it broadcasts over the leading axis).
        base_dict = {
            src: np.asarray(obs_dict[src], dtype=np.float32).reshape(1, -1)
            for srcs in synth_recipes.values()
            for src in srcs
            if src in obs_dict
        }
        synth_dict = compute_synthetic_obs_keys(base_dict, synth_recipes)
        for k, arr in synth_dict.items():
            # Strip the leading singleton axis we added above.
            obs_dict[k] = np.asarray(arr, dtype=np.float32).reshape(-1)

    parts = []
    for k in obs_keys:
        if k not in obs_dict:
            raise KeyError(
                f"runtime obs missing {k!r} (available: "
                f"{sorted(obs_dict.keys())})"
            )
        parts.append(np.asarray(obs_dict[k], dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=-1)


def _compact_step_obs(env) -> dict[str, list[float]]:
    obs_dict = env._get_observations(force_update=True)
    keys = (
        "robot0_eef_pos",
        "Can_pos",
        "cube_pos",
        "SquareNut_pos",
        "SquareNut_quat",
        "SquareNut_to_eef_pos",
        "SquareNut_to_robot0_eef_pos",
        "SquareNut_to_robot0_eef_quat",
        "Tool_pos",
        "tool_pos",
        "tool_quat",
        "tool_to_robot0_eef_pos",
        "tool_to_robot0_eef_quat",
        "frame_pos",
        "frame_quat",
        "frame_to_robot0_eef_pos",
        "frame_to_robot0_eef_quat",
        "frame_is_assembled",
        "base_pos",
        "base_quat",
        "base_to_robot0_eef_pos",
        "base_to_robot0_eef_quat",
        "tool_on_frame",
        "tool_hole_pos",
        "target_pos",
        "robot1_eef_pos",
        "payload_pos",
        "payload_quat",
        "payload_in_target_bin",
        "trash_pos",
        "trash_quat",
        "trash_in_trash_bin",
        "target_bin_pos",
        "trash_bin_pos",
        "lid_handle_pos",
        "lid_handle_quat",
        "gripper0_to_lid_handle",
        "gripper0_to_payload",
        "gripper1_to_payload",
        "gripper1_to_trash",
    )
    out: dict[str, list[float]] = {}
    for key in keys:
        if key in obs_dict:
            out[key] = np.asarray(obs_dict[key], dtype=np.float32).reshape(-1).tolist()
    return out


def unified_plan_and_execute(
    env,
    init_state: np.ndarray,
    *,
    prior_funcs: tuple[InitialMeanFn, ScoreFnFactory] | None,
    lambda_prior: float,
    params: MBDParams,
    rng: np.random.Generator,
    n_replans: int,
    chunk_steps: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    no_warm_start: bool = False,
    n_workers: int = 1,
    dataset_path: Path | None = None,
    perturbations: dict | None = None,
    make_rollout_fn=None,
    cost_fn=None,
    warmstart_seed_fn: WarmstartSeedFn | None = None,
    obs_keys: tuple[str, ...] | None = None,
    prior_lambda_schedule: np.ndarray | None = None,
    action_history_len: int = 0,
    max_env_steps: int | None = None,
    warmstart_noise_sigma: float = 0.0,
    save_executed_actions: bool = False,
    combine_mode: str = "score",
    temper_p: float = 0.0,
    sigma_demo: float = 0.25,
    demo_sigma_mode: str = "const",
    demo_sigma_adapt_beta: float = 1.0,
    prior_sigma_mode: str = "decay",
    log_gate_diagnostic: bool = False,
    save_step_trace: bool = False,
) -> dict:
    """Unified plan-and-execute combining MPC chunking + score-bridge prior.

    Parameters
    ----------
    env, init_state
        Live env handle and the chunk-zero state we're planning from.
    prior_funcs
        ``(initial_mean_fn, score_fn_factory)`` from
        ``run_can._load_prior``. Pass ``None`` for the pure-MBD
        baseline (no warm-start, no score correction).
    lambda_prior
        Weight on the per-step prior score in MBD's denoise loop.
        ``0.0`` -> warm-start-only (exp 05's behaviour).
        ``inf`` -> pure-prior plan (MBD's gradient zeroed out).
    params
        ``MBDParams`` with ``H``, ``M``, ``K``, ``sigma_schedule``,
        ``clip_action``, and ``lambda_`` (the softmax temperature,
        distinct from ``lambda_prior``).
    rng
        Numpy RNG, used once per chunk for the MBD perturbations.
    n_replans, chunk_steps
        MPC outer-loop knobs. ``n_replans=0`` -> single-shot exp 07
        behaviour. ``chunk_steps`` is the number of env-steps
        executed per chunk before re-planning.
    action_low, action_high
        Per-dim action bounds for clipping the warm-start.
    no_warm_start
        If True or ``prior_funcs is None``, use ``initial_mean=zeros``
        (the pure-MBD ablation).
    n_workers, dataset_path
        Forwarded to exp 04's worker pool.
    perturbations
        Optional dict of runtime env perturbations (cost-time goal
        translation, dynamics mass/friction scaling).  Forwarded to
        ``make_rollout_fn`` -> ``worker._worker_init`` so workers'
        envs see the same perturbations as the main env.  See
        ``worker._worker_init`` docstring for recognised keys.
        ``None`` (default) means no perturbations, i.e. the canonical
        slate-E / slate-G operating regime.
    make_rollout_fn, cost_fn
        Injected so unified_loop doesn't have to import exp 04
        at module top level (which conflicts with exp 03's basename).
    warmstart_seed_fn
        Optional ``(chunk_idx) -> seed`` callable.  When provided, the
        loop calls ``torch.manual_seed(warmstart_seed_fn(chunk_idx))``
        immediately before each chunk's ``initial_mean_fn`` call,
        making the flow prior's sampler deterministic per
        (base_seed, demo, rollout, chunk).  See module-level
        ``WarmstartSeedFn`` comment for context.

    Returns
    -------
    dict
        Same keys as exp 05's ``mpc_plan_and_execute`` plus
        ``"chunks"`` is unchanged.
    """
    if make_rollout_fn is None or cost_fn is None:
        raise ValueError(
            "make_rollout_fn and cost_fn must be supplied; "
            "see run_can.main for the canonical wiring."
        )

    n_chunks = int(n_replans) + 1
    H = params.H
    action_dim = params.action_dim

    # Receding-horizon-compatible execution budget.
    #
    # Historically this loop assumed ``H`` is *both* the MBD planner
    # horizon *and* the total episode length: each chunk executed
    # ``chunk_steps`` of the H-length plan, and the cumulative
    # ``steps_executed`` budget capped at ``H``.  That made sense for
    # the exp05/07/15 setups where ``H = 120`` was the whole rollout.
    #
    # The Diffusion-Policy-style closed-loop setup uses a much shorter
    # ``H`` (e.g. 16) as the *receding-horizon* planner depth, while
    # the actual episode runs for many more env steps (e.g. 240 = 30
    # chunks of 8).  In that regime we need to decouple the planner
    # horizon from the total budget, otherwise the loop bails after
    # the second chunk -- exactly the silent failure that produced
    # the "30 chunks but only 2 actually ran" bug observed in the
    # asymmetric-MBD evaluation of v_theta_dp_tobs2_h16.
    #
    # New rule: ``max_env_steps`` (when supplied) is the total budget;
    # if omitted we fall back to ``n_chunks * chunk_steps`` capped at
    # ``H`` (the pre-fix behaviour for backwards-compat with the
    # H=120 exp05/15 cells).  Callers that genuinely want the old
    # H-capped semantics can pass ``max_env_steps=H`` explicitly.
    if max_env_steps is not None:
        total_budget = int(max_env_steps)
    elif int(n_chunks) * int(chunk_steps) > H:
        total_budget = int(n_chunks) * int(chunk_steps)
    else:
        total_budget = H

    current_state = np.asarray(init_state)
    chunks_log: list[dict] = []
    cumulative_reward = 0.0
    # Track "success-ever" across every env.step in the rollout, not
    # only at the final state.  Without this we under-count successes
    # for any prior whose plan succeeds mid-rollout and then keeps
    # moving (e.g., DP's chunked policy can place the can correctly
    # then drift -- the env's success flag flips False even though
    # the task was completed).
    #
    # standalone_dp.py uses per-step check_success WITH early
    # termination at line 171-175; we keep stepping (because MBD
    # multi-chunk plans rely on env continuing to evolve), but record
    # success the moment it first appears.  Verified 2026-05-26:
    # accounts for the ~3 pp standalone (26.7 %) vs unified K=0
    # (23.3 %) gap on Can/ph.
    success_ever = False

    # Position env at init_state ONCE at the top of the rollout.
    # The previous design did env.reset()+set_sim_state at the top of
    # each chunk's Stage A *and* Stage D, but the reset+restore cycle
    # is not bit-identical to natural propagation -- robosuite's
    # controller / solver fixed-point state isn't fully captured in
    # ``sim.get_state().flatten()``, so the post-restore observation
    # drifts by ~9e-3 in 40+ dims (verified by
    # ``experiments/17_dp_hybrid/diagnose_chunk0.py``).  For obs-
    # conditioned DP T_obs=2 priors this drift compounds across
    # chunks and collapses success to 0/30.  For T_obs=1 priors
    # dominated by cube-position obs the drift is benign, which is
    # why the v6 CFM 75.9% headline never tripped this bug.
    #
    # The fix: position env at init_state once here, then between
    # chunks just let env.step()'s natural propagation maintain
    # state.  We only need to re-snapshot/restore when the
    # ``n_workers <= 1`` MBD path clobbers the main env (see Stage
    # D below).
    env.reset()
    set_sim_state(env, current_state)

    # When n_workers <= 1, exp 04's rollout_one does
    # ``env.reset() + set_sim_state(env, init_state)`` on the *main*
    # env for each of its M perturbed rollouts, leaving the main
    # env at an unknown post-rollout state.  We then need to
    # re-snapshot before executing the action chunk.  When
    # n_workers > 1 the workers each own a forked env and the main
    # env is untouched, so no re-snapshot is needed.
    _mbd_touches_main_env = (
        int(n_workers) <= 1 or dataset_path is None
    )

    # 2026-05-31: build the rollout worker pool ONCE per rollout and
    # reuse it across all MPC chunks.  Previously (commit 31b58a3 and
    # earlier) a fresh ``mp.Pool`` was created+torn down inside the
    # chunk loop, which on cloud VMs costs ~30s/worker for robosuite
    # env init -- ~25min of pure overhead per rollout at
    # --mpc-replans 49.  ``rollout_fn.update_init_state`` (added the
    # same day in exp 04's ``make_rollout_fn``) mutates the anchor
    # without rebuilding the pool, so per-chunk semantics are
    # preserved.  We seed the pool with ``current_state`` as a safe
    # default; the chunk loop re-anchors via ``update_init_state``
    # before each ``denoise_loop`` call.
    rollout_fn = make_rollout_fn(
        env, current_state,
        n_workers=n_workers,
        dataset_path=dataset_path,
        perturbations=perturbations,
    )

    plan_t0 = time.perf_counter()
    steps_executed = 0

    # A1 (2026-05-20): per-rollout buffer of recently executed actions.
    # Mirrors the bank-side ``_build_action_history`` convention:
    # oldest first, newest last; zero-padded when fewer than
    # ``action_history_len`` actions have been executed.  When
    # ``action_history_len = 0`` the buffer is empty and the
    # downstream ``hist_vec`` concatenation is a no-op, keeping
    # bit-for-bit compat with pre-A1 cells.
    if action_history_len > 0:
        recent_actions_buf = collections.deque(
            (np.zeros(action_dim, dtype=np.float32)
             for _ in range(int(action_history_len))),
            maxlen=int(action_history_len),
        )
    else:
        recent_actions_buf = None

    for chunk_idx in range(n_chunks):
        steps_left = total_budget - steps_executed
        if steps_left <= 0:
            break
        is_last = chunk_idx == n_chunks - 1
        # ``H`` still caps a single chunk's commitment (we can never
        # commit more actions than the planner produced).
        per_chunk_cap = min(chunk_steps, H)
        steps_this_chunk = (
            min(steps_left, H) if is_last
            else min(per_chunk_cap, steps_left)
        )

        # --- Stage A: build initial_mean for this chunk's planner.
        # The s0 vector is what the prior conditions on; the env is
        # already at ``current_state`` (positioned once at the top
        # of this function, then propagated naturally by Stage D's
        # env.step() loop in subsequent chunks), so we observe
        # directly -- no reset+restore.  See the rationale at the
        # top of this function for why this matters for T_obs > 1
        # priors.
        s0_vec = _make_obs_vec_for_prior(env, obs_keys=obs_keys)

        # A1: append action history to s0_vec when augmentation is on.
        # The bank's obs vectors carry the same trailing
        # ``action_history_len * action_dim`` columns (see
        # ``priors_zerotrain.load_demo_bank``), so retrieval
        # distances are computed in the augmented space directly.
        if recent_actions_buf is not None:
            hist_vec = np.concatenate(
                list(recent_actions_buf), dtype=np.float32
            )
            s0_vec = np.concatenate(
                [s0_vec.astype(np.float32), hist_vec], dtype=np.float32
            )

        if no_warm_start or prior_funcs is None:
            initial_mean = np.zeros((H, action_dim), dtype=np.float32)
            initial_mean_fn = None
        else:
            initial_mean_fn, _ = prior_funcs
            # Seed torch's global RNG so ``initial_mean_fn`` is
            # deterministic per (rollout, chunk).  See module
            # docstring for context.
            #
            # 2026-05-26 fix: closures may opt into *private-generator*
            # seeding via the ``_set_chunk_seed`` attribute.  This is
            # required for the DP T_obs=2 flow prior because
            # ``torch.manual_seed(X) + sample_initial_mean(s, n_steps)``
            # produces a uniformly-bad sample from the posterior
            # (verified empirically: 0/30 success on Can), while
            # ``sample_initial_mean(s, n_steps, seed=X)`` -- which
            # routes through a private ``torch.Generator`` --
            # recovers the trained rate.  The legacy
            # ``torch.manual_seed`` path is kept for retrieval and BC
            # priors that have always used it.
            if warmstart_seed_fn is not None:
                seed_value = int(warmstart_seed_fn(chunk_idx))
                if hasattr(initial_mean_fn, "_set_chunk_seed"):
                    initial_mean_fn._set_chunk_seed(seed_value)
                else:
                    torch.manual_seed(seed_value)
            initial_mean = initial_mean_fn(env, current_state, s0_vec)

        # Controlled warm-start corruption (added 2026-05-25).  When
        # warmstart_noise_sigma > 0, add iid Gaussian noise to the
        # warm-start *before* the clip-and-pass-to-MBD step.  The
        # noise rng is derived from warmstart_seed_fn so the
        # corruption is bit-reproducible per (base_seed, demo,
        # rollout, chunk); fall back to ``rng`` when
        # warmstart_seed_fn is None.  Use case: the controlled-
        # rescue probe (Slate L / K6) that deliberately pushes a
        # strong prior off-manifold to test MBD's pull-mode
        # recovery.
        if float(warmstart_noise_sigma) > 0.0 and not no_warm_start:
            if warmstart_seed_fn is not None:
                noise_rng = np.random.default_rng(
                    int(warmstart_seed_fn(chunk_idx)) ^ 0xC0FFEE
                )
            else:
                noise_rng = rng
            noise = noise_rng.standard_normal(
                size=initial_mean.shape
            ).astype(np.float32) * float(warmstart_noise_sigma)
            initial_mean = (initial_mean + noise).astype(np.float32)

        initial_mean = np.clip(initial_mean, action_low, action_high)
        initial_mean = initial_mean.astype(np.float32)

        # --- Stage B: build the per-step score_fn for this chunk.
        score_fn = None
        if prior_funcs is not None and not no_warm_start and lambda_prior != 0.0:
            _, score_fn_factory = prior_funcs
            score_fn = score_fn_factory(s0_vec, params.K)
            # BC fallback: the closure needs the chunk's BC trajectory
            # to be the per-step pull target. Flow prior has no such
            # state and the attribute simply isn't there.
            if hasattr(score_fn, "state"):
                score_fn.state["a_bc"] = initial_mean  # type: ignore[attr-defined]

        # --- Stage C: MBD denoising. Re-anchor the (reused) rollout
        # pool at current_state so MBD's perturbations are evaluated
        # as "what would happen if we executed mu starting right
        # here, right now".  The pool itself was built once before
        # the chunk loop (with `perturbations` threaded to each
        # worker via 016d603); we mutate its init_state cell here.
        rollout_fn.update_init_state(current_state)

        chunk_plan_t0 = time.perf_counter()
        # Build kwargs only if needed -- omitting prior_score_fn
        # keeps unified_loop bit-for-bit reproducible against exp 05
        # for the warm-start-only case (lambda_prior=0 with score
        # path stripped). With score_fn != None we pass both
        # arguments so denoise_loop's composite-update branch fires.
        denoise_kwargs = {}
        if score_fn is not None:
            denoise_kwargs["prior_score_fn"] = score_fn
            denoise_kwargs["prior_lambda"] = float(lambda_prior)
            denoise_kwargs["prior_sigma_mode"] = prior_sigma_mode
            if prior_lambda_schedule is not None:
                denoise_kwargs["prior_lambda_schedule"] = prior_lambda_schedule

        # Tempered/gated combination (§6.1/§6.2).  Uses the chunk's
        # warm-start (``initial_mean`` = the BC/demo trajectory) as the
        # Gaussian-pull demo target and reweights model-vs-demo
        # per-sample with ``temper_p`` (0 -> product, inf -> MBD-Demo
        # max-gate).  Requires a warm-start so ``initial_mean`` is the
        # demo, not zeros.
        if combine_mode == "tempered":
            if no_warm_start or prior_funcs is None:
                raise ValueError(
                    "combine_mode='tempered' requires a warm-start prior "
                    "(initial_mean is the demo target); pass --warm-start "
                    "with a --prior."
                )
            denoise_kwargs["combine_mode"] = "tempered"
            denoise_kwargs["demo_target"] = initial_mean
            denoise_kwargs["sigma_demo"] = float(sigma_demo)
            denoise_kwargs["demo_sigma_mode"] = demo_sigma_mode
            denoise_kwargs["demo_sigma_adapt_beta"] = float(demo_sigma_adapt_beta)
            denoise_kwargs["temper_p"] = float(temper_p)
            denoise_kwargs["prior_lambda"] = float(lambda_prior)
            # The per-step score_fn pull is unused in tempered mode.
            denoise_kwargs.pop("prior_score_fn", None)

        # §6.3 passive diagnostic: log gate_frac + ldem_minus_lmod on
        # ANY run (product/score/pure-MBD) without altering the update.
        # Needs a demo target -- use the chunk's warm-start when present.
        if log_gate_diagnostic and combine_mode != "tempered" \
                and not no_warm_start and prior_funcs is not None:
            denoise_kwargs["log_gate_diagnostic"] = True
            denoise_kwargs["demo_target"] = initial_mean
            denoise_kwargs["sigma_demo"] = float(sigma_demo)
            denoise_kwargs["demo_sigma_mode"] = demo_sigma_mode
            denoise_kwargs["demo_sigma_adapt_beta"] = float(demo_sigma_adapt_beta)

        mu_final, log = denoise_loop(
            initial_mean, rollout_fn, cost_fn, params, rng,
            **denoise_kwargs,
        )
        chunk_plan_elapsed = time.perf_counter() - chunk_plan_t0

        # --- Stage D: execute steps_this_chunk actions on the live env.
        # When MBD's worker pool clobbered the main env (n_workers <= 1
        # path in exp 04's ``rollout_one``), re-snapshot the main env
        # back to ``current_state`` before executing.  Otherwise the env
        # is still at ``current_state`` from the previous chunk's
        # env.step() loop (or from the top-of-function init for
        # chunk 0) and a reset+restore would re-introduce the
        # controller-state perturbation we just removed in Stage A.
        #
        # Phase-D fix (2026-05-30, exp17): also gate on params.K > 0.
        # At K = 0 the inner ``for k in range(K)`` in ``denoise_loop``
        # never iterates, so ``rollout_fn`` is never called, so the
        # main env is never clobbered -- yet the reset+restore would
        # still fire here, wiping robosuite's controller state
        # (OSC goal cache, interpolator, last-commanded pose) without
        # any clobbering to undo.  ``set_sim_state`` only restores
        # Mujoco's qpos/qvel; the controller's Python-side state is
        # lost.  Result on Phase-B's strong DP prior: K = 0 test
        # success collapses from ~80 % (standalone) to 0 %.  See
        # experiments/17_dp_hybrid/results/phase_d_2026_05_30/BRIEF.md.
        if _mbd_touches_main_env and int(params.K) > 0:
            env.reset()
            set_sim_state(env, current_state)
        chunk_reward = 0.0
        # C6 (2026-06-01): optional per-step action capture for the
        # (bias, variance) decomposition.  Cheap; only allocates when
        # the caller actually asks for it.
        executed_actions_this_chunk: list[np.ndarray] | None = (
            [] if save_executed_actions else None
        )
        step_trace_this_chunk: list[dict] | None = (
            [] if save_step_trace else None
        )
        for t in range(steps_this_chunk):
            action = mu_final[t]
            _obs, reward, _done, _info = env.step(action)
            chunk_reward += float(reward)
            # A1: push every actually-executed action onto the
            # circular history buffer so the next chunk's s0_vec
            # picks up the most recent ``action_history_len`` entries.
            if recent_actions_buf is not None:
                recent_actions_buf.append(
                    np.asarray(action, dtype=np.float32)
                )
            if executed_actions_this_chunk is not None:
                executed_actions_this_chunk.append(
                    np.asarray(action, dtype=np.float32)
                )
            # Track success-ever (see comment at top of this function).
            # Cheap: check_success is a single boolean attribute read
            # on the env; no overhead worth caring about.
            if not success_ever and check_success(env):
                success_ever = True
            if step_trace_this_chunk is not None:
                step_trace_this_chunk.append({
                    "step_idx": int(steps_executed + t),
                    "chunk_step_idx": int(t),
                    "reward": float(reward),
                    "success": bool(check_success(env)),
                    "obs": _compact_step_obs(env),
                })
        cumulative_reward += chunk_reward
        steps_executed += steps_this_chunk

        next_state = _snapshot_sim_state(env)

        chunk_log_entry = {
            "chunk_idx": int(chunk_idx),
            "n_chunks": int(n_chunks),
            "steps_executed": int(steps_this_chunk),
            "chunk_reward": float(chunk_reward),
            "plan_elapsed_s": float(chunk_plan_elapsed),
            # ``log["per_iter"]`` is empty when ``params.K == 0``
            # (the no-MBD control used by exp 17 diagnostics: pure
            # warm-start delivery through the unified driver).  Emit
            # NaN so downstream summarisers can detect the bypass.
            "init_best_cost": (
                float(log["per_iter"][0]["best_cost"])
                if log["per_iter"] else float("nan")
            ),
            "final_best_cost": (
                float(log["per_iter"][-1]["best_cost"])
                if log["per_iter"] else float("nan")
            ),
            "used_score_fn": bool(score_fn is not None),
            "iter_curve": [
                {
                    "iter": p["iter"],
                    "sigma": p["sigma"],
                    "best_cost": p["best_cost"],
                    "mean_cost": p["mean_cost"],
                    "ess": p["ess"],
                    # Tempered/gated diagnostics (NaN in legacy/score mode).
                    "gate_frac": p.get("gate_frac", float("nan")),
                    "ldem_minus_lmod": p.get("ldem_minus_lmod", float("nan")),
                }
                for p in log["per_iter"]
            ],
        }
        # C6 (2026-06-01): expose the per-step executed-actions chunk
        # to downstream consumers (measure_bias_variance.py).  Stored
        # as a nested list-of-list-of-float for clean JSON
        # serialisation; only present when the caller opted in.
        if executed_actions_this_chunk is not None:
            chunk_log_entry["executed_actions"] = [
                a.tolist() for a in executed_actions_this_chunk
            ]
        if step_trace_this_chunk is not None:
            chunk_log_entry["step_trace"] = step_trace_this_chunk
        chunks_log.append(chunk_log_entry)

        current_state = next_state

    # 2026-05-31: tear down the rollout pool now that all chunks are
    # done.  Mirrors the per-chunk shutdown that used to live inside
    # the chunk loop; without this, the n_workers>1 path leaks the
    # ctx.Pool and its worker procs each rollout.
    rollout_fn.shutdown()

    total_elapsed = time.perf_counter() - plan_t0
    # success-ever || success-now.  Catches both the standalone_dp.py
    # convention (success at any point in the rollout) and the
    # robosuite-native convention (success at the final state).  In
    # practice success_ever subsumes the final check because robosuite
    # set the env's internal flag the moment the can hit the bin.
    final_success = bool(success_ever or check_success(env))

    last = chunks_log[-1] if chunks_log else None
    return {
        "final_reward": float(cumulative_reward),
        "final_success": bool(final_success),
        "plan_elapsed_s": float(total_elapsed),
        "exec_elapsed_s": 0.0,
        "log": {"per_iter": last["iter_curve"] if last else []},
        "chunks": chunks_log,
    }
