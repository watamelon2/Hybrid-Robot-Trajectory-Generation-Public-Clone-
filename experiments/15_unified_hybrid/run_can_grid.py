"""Unified hybrid runner: score-bridge + MPC chunking + pluggable prior.

Replaces the divergent ``run_can.py`` in exp 05 (warm-start-only +
MPC) and exp 07 (score-bridge, single-shot). See
``unified_loop.py`` for the algorithm; this file is the CLI and IO
shell.

Flag surface is the union of exp 05 + exp 07 + a handful of new
no-op flags that branches A and B will wire (``--value-baseline-ckpt``,
``--shaped-cost``, ``--prior gaussian_bc``).

Regression-test invocations:

    # Reproduce exp 07 lambda=5 single-shot (target 20/60 = 33.3 %)
    python -m experiments.15_unified_hybrid.run_can \\
        --prior=bc --lambda-prior=5 --mpc-replans=0 \\
        --K=10 --M=16 --sigma-max=0.25 \\
        --n-demos=20 --n-rollouts=3 --seed=0

    # Reproduce exp 10 K=10 R=2 (target 11/60 = 18.3 %)
    python -m experiments.15_unified_hybrid.run_can \\
        --prior=bc --lambda-prior=0 --mpc-replans=2 \\
        --K=10 --M=16 --sigma-max=0.25 --lambda_=0.5 \\
        --n-demos=20 --n-rollouts=3 --seed=0
"""

from __future__ import annotations

import argparse
import importlib.util as _importlib_util
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Path / import setup (mirrors exp 07)
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _THIS_DIR.parent
_EXP_03 = _EXPERIMENTS_DIR / "03_bc_baseline"
_EXP_04 = _EXPERIMENTS_DIR / "04_mbd_baseline"
_EXP_05 = _EXPERIMENTS_DIR / "05_hybrid_method"
_EXP_08 = _EXPERIMENTS_DIR / "08_flow_matching_prior"

for _p in (
    _EXPERIMENTS_DIR, _EXP_03, _EXP_04, _EXP_05, _EXP_08, _THIS_DIR
):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from common import (  # noqa: E402
    default_dataset_path,
    get_action_bounds,
    make_env_for_dataset,
    set_sim_state,
)
from mbd import (  # noqa: E402
    MBDParams,
    cosine_sigma_schedule,
)

# exp 04's run_can holds the worker-pool factory + cost_fn.
_spec4 = _importlib_util.spec_from_file_location(
    "exp04_run_can", _EXP_04 / "run_can.py"
)
exp04_run = _importlib_util.module_from_spec(_spec4)
sys.modules["exp04_run_can"] = exp04_run
_spec4.loader.exec_module(exp04_run)

# exp 05's run_can holds bc_warmstart_actions used by the BC prior.
_spec5 = _importlib_util.spec_from_file_location(
    "exp05_run_can", _EXP_05 / "run_can.py"
)
exp05_run = _importlib_util.module_from_spec(_spec5)
sys.modules["exp05_run_can"] = exp05_run
_spec5.loader.exec_module(exp05_run)

# Local module.
from unified_loop import unified_plan_and_execute  # noqa: E402


DEFAULT_LAMBDAS = (0.0, 0.1, 0.5, 1.0, 5.0, float("inf"))


# ---------------------------------------------------------------------------
# Prior loader (extends exp 07's _load_prior with the no-op stubs
# for branches A and B)
# ---------------------------------------------------------------------------


def _load_prior(args, action_dim: int):
    """Return ``(initial_mean_fn, score_fn_factory, label)``.

    Mirrors exp 07's ``_load_prior`` exactly for ``--prior {bc, flow}``;
    raises NotImplementedError on the ``gaussian_bc`` stub so Branch
    B can wire it later without changing this file's surface.
    """
    if args.prior == "flow":
        from adapter import FlowPrior  # exp 08

        # Pass sigma_min/sigma_max so v3 (Karras-EDM) maps MBD's
        # cosine schedule to the right σ values.  v1/v2 ignore these
        # (their adapters don't take a σ schedule).
        prior = FlowPrior.from_checkpoint(
            args.prior_checkpoint,
            device=args.device,
            sigma_min=float(args.sigma_min),
            sigma_max=float(args.sigma_max),
        )

        prior_n_steps = int(getattr(args, "prior_n_steps", 32))

        # T_obs > 1 stateful mode (added 2026-05-26 to fix the
        # exp 17 DP-hybrid regression).  ``standalone_dp.py`` calls
        # ``prior.observe(s)`` before each ``sample_initial_mean``
        # so the adapter conditions on the trained ``[s_prev, s_curr]``
        # history.  Without this call the adapter falls back to
        # stateless mode (``[s_curr, s_curr]`` replication), which
        # matches DP's *training-time boundary* convention at chunk
        # 0 but is OOD for chunks 1..N-1.  For under-trained
        # checkpoints (e.g. v_theta_dp_tobs2_h16 @ 250 ep) the
        # mismatch collapses success rate from ~37% to ~0%.
        #
        # We expose ``prior`` as an attribute on the closure so the
        # rollout loop in ``main`` can call ``reset_obs_buffer()``
        # between rollouts -- the buffer is a deque of recent obs
        # and must be cleared at each env reset.
        #
        # Seed-passing channel (added 2026-05-26 part 2).  Empirically
        # ``torch.manual_seed(X) + sample_initial_mean(s, n_steps=N)``
        # produces DIFFERENT DP samples than
        # ``sample_initial_mean(s, n_steps=N, seed=X)`` -- the former
        # consumes torch's global RNG state via ``torch.randn`` while
        # the latter constructs a private ``torch.Generator(seed=X)``
        # and passes it through.  For this DP checkpoint the global-
        # state path lands in a uniformly-bad mode of the posterior
        # (0/30 on Can across all 30 trials), while the private-
        # generator path recovers the trained ~37% rate.  We stash
        # the per-chunk seed in a mutable dict that the unified-loop
        # driver writes to before each ``initial_mean_fn`` call;
        # see ``unified_loop.py`` Stage A for the writer side.
        _chunk_seed_slot = {"value": None}

        def initial_mean_fn(env, init_state, s0):
            del env, init_state
            if hasattr(prior, "observe"):
                prior.observe(s0)
            return prior.sample_initial_mean(
                s0, n_steps=prior_n_steps,
                seed=_chunk_seed_slot["value"],
            )

        def _set_chunk_seed(value):
            _chunk_seed_slot["value"] = (
                int(value) if value is not None else None
            )

        initial_mean_fn._prior_for_buffer_reset = prior  # type: ignore[attr-defined]
        initial_mean_fn._set_chunk_seed = _set_chunk_seed  # type: ignore[attr-defined]

        # Score-mode dispatch (added 2026-05-24).
        #
        # The original ``score_fn`` for trained priors evaluates the
        # learned vector field ``-ε(mu, t(σ_k))/σ_k`` at the *current*
        # MBD iterate ``mu``.  Empirically that's destructive for
        # strong DP priors (lam=1 on the 36.7% DP H=16 baseline
        # collapses to 0/30 because the non-stationary attractor
        # smears the carefully-constructed prior sample across the
        # K=10 inner MBD iterations).
        #
        # The retrieval-style priors (demo_1nn, BC) use a *stationary*
        # pull-toward-warm-start score: ``(a_target - mu)`` where
        # ``a_target`` is fixed for the whole chunk.  unified_loop.py
        # stashes the warm-start into ``score_fn.state["a_bc"]``
        # automatically if the score_fn exposes a ``.state`` dict.
        #
        # ``--score-mode pull`` builds a state-bearing pull-style
        # closure for the trained prior, ignoring its learned
        # vector field but keeping its observation-conditioned
        # warm-start.  This is the "trained-prior-as-retrieval"
        # operating mode that the demo_1nn pipeline already uses
        # to reach 73.8% headline.
        score_mode = str(getattr(args, "score_mode", "learned"))
        if score_mode == "learned":
            def score_fn_factory(s0, K):
                return prior.make_prior_score_fn(s0, K)
        elif score_mode == "pull":
            def score_fn_factory(s0, K):
                del s0, K
                state = {"a_bc": None}

                def score_fn(mu, k):
                    del k
                    a_t = state["a_bc"]
                    if a_t is None:
                        return np.zeros_like(mu, dtype=np.float32)
                    return (a_t - mu).astype(np.float32)

                score_fn.state = state  # type: ignore[attr-defined]
                return score_fn
        else:
            raise ValueError(
                f"--score-mode={score_mode!r} not recognised "
                f"(expected 'learned' or 'pull')"
            )

        # Tag v3 differently in output JSONs so the analysis layer
        # can distinguish flow-v3 cells from v1/v2 retroactively.
        label = "flow"
        if hasattr(prior, "config") and getattr(prior.config, "arch", "") == "karras_edm_v3":
            label = "flow_v3"
        if score_mode == "pull":
            label = f"{label}_pull"
        return initial_mean_fn, score_fn_factory, label

    if args.prior == "bc":
        from policy import Policy  # exp 03

        policy = Policy.from_checkpoint(
            args.bc_checkpoint, device=args.device
        )

        def initial_mean_fn(env, init_state, _s0):
            from common import get_action_bounds as _gab

            low, high = _gab(env)
            actions, _info = exp05_run.bc_warmstart_actions(
                env, init_state, policy,
                args.H,
                low.astype(np.float32),
                high.astype(np.float32),
            )
            return actions

        def score_fn_factory(s0, K):
            del s0
            state = {"a_bc": None}

            def score_fn(mu, k):
                a_bc = state["a_bc"]
                if a_bc is None:
                    return np.zeros_like(mu, dtype=np.float32)
                return (a_bc - mu).astype(np.float32)

            score_fn.state = state  # type: ignore[attr-defined]
            return score_fn

        return initial_mean_fn, score_fn_factory, "bc"

    if args.prior in (
        "demo_1nn", "demo_knn", "demo_random", "demo_mean",
        "demo_lwr", "demo_topk_random",
        "demo_1nn_proprio", "demo_1nn_object",
        "demo_softnn",
        "demo_1nn_contrastive",
    ):
        # Zero-training retrieval priors over the demo bank.  No
        # checkpoint; the bank is loaded once and reused across all
        # (demo, rollout, chunk) tuples for the current run.
        from priors_zerotrain import load_demo_bank, make_zero_train_prior

        # Bank source dispatcher
        # (feat/adapt-incremental-prior, 2026-05-25).  By default
        # the retrieval bank is read from the same dataset as the
        # evaluation init states (--dataset / --task), reproducing
        # the canonical headline cells bit-for-bit.  When
        # --prior-bank-dataset is set, the bank is read from THAT
        # dataset instead -- enabling cross-task transfer-prior
        # cells (T1) and incremental-adaptation cells (T2).  See
        # docs/adaptation/04_incremental_prior_spec.md.
        bank_dataset = (
            args.prior_bank_dataset
            if args.prior_bank_dataset is not None
            else (args.dataset or default_dataset_path(args.task))
        )
        bank_task = (
            args.prior_bank_task
            if args.prior_bank_task is not None
            else args.task
        )
        # Bank size truncation (feat/adapt-demo-scaling, 2026-05-25).
        # prior_bank_size scales the retrieval bank independently of
        # --n-demos (which controls evaluation rollout count).
        # Default 0 == use all demos in the dataset.  Orthogonal to
        # --prior-bank-dataset: truncation applies to whichever
        # dataset the dispatcher above selected.  See
        # docs/adaptation/03_demo_scaling_spec.md.
        prior_n_demos = (
            int(args.prior_bank_size)
            if int(args.prior_bank_size) > 0
            else None
        )
        bank = load_demo_bank(
            bank_dataset,
            n_demos=prior_n_demos,
            task=bank_task,
            action_history_len=int(args.action_history_len),
        )
        cross_task_note = (
            f"  cross-task: eval={args.task} bank={bank_task}"
            if bank_task != args.task else ""
        )
        print(
            f"[exp15] zero-train bank: {bank.n_demos} demos, "
            f"{bank.n_total} timesteps, obs_dim={bank.obs.shape[1]}, "
            f"action_dim={bank.action_dim}, task={bank_task}, "
            f"action_history_len={bank.action_history_len}"
            f"{cross_task_note}"
        )
        return make_zero_train_prior(
            kind=args.prior,
            bank=bank,
            H=int(args.H),
            knn_k=int(args.knn_k),
            rng_seed=int(args.seed),
            retrieval_beta=float(args.retrieval_beta),
            contrastive_encoder_path=args.contrastive_encoder_path,
        )

    if args.prior == "gaussian_bc":
        raise NotImplementedError(
            "--prior=gaussian_bc reserved for Branch B Task B4. "
            "Train a Gaussian-head BC first (see "
            "docs/plan_principled_score.md § B4), then implement "
            "this branch of _load_prior."
        )

    raise ValueError(f"Unknown --prior={args.prior!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    # Task / dataset
    p.add_argument("--task", type=str, default="can")
    p.add_argument("--dataset", type=Path, default=None)

    # Prior
    p.add_argument(
        "--prior",
        choices=(
            "bc", "flow", "gaussian_bc",
            "demo_1nn", "demo_knn", "demo_random", "demo_mean",
            "demo_lwr", "demo_topk_random",
            "demo_1nn_proprio", "demo_1nn_object",
            "demo_softnn",
            "demo_1nn_contrastive",
        ),
        default="bc",
        help="Which generative prior to use. 'bc' = exp 03 MLP-BC "
             "(default; matches exp 05 / 07 BC sweep). 'flow' = "
             "exp 08 flow-matching v_theta. 'gaussian_bc' = Branch B "
             "B4 stub (not yet implemented).  Zero-training retrieval "
             "priors over the demo bank: 'demo_1nn' (nearest neighbour "
             "on the obs vector), 'demo_knn' (k-NN, uniform mean of "
             "windows; see --knn-k), 'demo_random' (uniform-random "
             "demo), 'demo_mean' (dataset-mean action sequence). "
             "Slate F additions: 'demo_lwr' (k-NN with inverse-"
             "distance weights), 'demo_topk_random' (uniform sample "
             "from top-k NN), 'demo_1nn_proprio' / 'demo_1nn_object' "
             "(1-NN on robot-only / object-only subset of obs).  "
             "Slate G addition: 'demo_softnn' (A3 soft 1-NN with "
             "temperature beta).  Slate I prep: 'demo_1nn_contrastive' "
             "(C1, 1-NN in a learned contrastive obs-embedding space; "
             "requires --contrastive-encoder-path).  "
             "All reuse BC's pull-toward-warm-start score function so "
             "--lambda-prior > 0 gives a retrieved-trajectory score "
             "bridge.",
    )
    p.add_argument(
        "--knn-k", type=int, default=5,
        help="Number of nearest neighbours when --prior=demo_knn "
             "(or --prior=demo_lwr / demo_topk_random / demo_softnn). "
             "Ignored otherwise. Default 5.",
    )
    p.add_argument(
        "--retrieval-beta", type=float, default=1.0,
        help="Temperature for --prior=demo_softnn (A3, 2026-05-20). "
             "Weights are softmax(-beta * d2) over the top-k NN. "
             "beta=0 -> uniform mean over top-k (= demo_knn); "
             "beta->inf -> hard 1-NN (= demo_1nn). Sweeping beta "
             "draws the bias-variance tradeoff curve. Default 1.0.",
    )
    p.add_argument(
        "--contrastive-encoder-path", type=Path, default=None,
        help="C1 (2026-05-21): path to a contrastive obs-encoder "
             "checkpoint trained by "
             "experiments/15_unified_hybrid/train_contrastive_encoder.py. "
             "Required for --prior=demo_1nn_contrastive; ignored "
             "otherwise.  The encoder's input dim and "
             "action_history_len must match the runtime bank, or "
             "the prior factory raises.",
    )
    p.add_argument(
        "--action-history-len", type=int, default=0,
        help="A1 (2026-05-20): augment the retrieval obs vector with "
             "the last N executed actions (flattened, zero-padded "
             "before t=N).  Bank and runtime stay synchronised; the "
             "augmentation is bit-for-bit compat with --prior=bc/flow "
             "(which ignore the bank), but only the zero-train "
             "retrieval priors actually use the extra dims.  "
             "Default 0 (no augmentation).  Hypothesis: action "
             "history disambiguates contact phases on Square, where "
             "the same instantaneous state can correspond to "
             "approach / grasp / lift / insert depending on what the "
             "agent has been doing.",
    )
    p.add_argument(
        "--prior-checkpoint", type=Path, default=None,
        help="Path to the prior checkpoint. Defaults: exp 03 "
             "policy.pt for --prior=bc; exp 08 v_theta.pt for "
             "--prior=flow.",
    )
    p.add_argument(
        "--bc-checkpoint", type=Path,
        default=_EXP_03 / "results" / "policy.pt",
        help="Alias for --prior-checkpoint when --prior=bc.",
    )
    p.add_argument(
        "--lambda-prior", type=float, nargs="+", default=None,
        help=f"Weight(s) on the per-step prior score in MBD. "
             f"If multiple values, sweep is run. 0 = warm-start-only "
             f"(exp 05). inf = pure prior. "
             f"Default: {DEFAULT_LAMBDAS}.",
    )
    p.add_argument(
        "--warm-start", action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the prior to warm-start MBD's initial mean. "
             "Default True; --no-warm-start gives the pure-MBD ablation.",
    )

    # MBD core
    p.add_argument("--n-demos", type=int, default=20)
    p.add_argument("--n-rollouts", type=int, default=3)
    # Phase-C (2026-05-29): DP-paper-style test bucket.  Mirrors
    # the same flags on standalone_dp.py.  Each test rollout seeds
    # the env's placement_initializer with `test_start_seed + i`
    # and calls env.reset() to get a fresh init -- the apples-to-
    # apples bucket for comparing the unified pipeline at K=0/10/20
    # against Phase-B's standalone 80% test rate.  Defaults to 0 so
    # every legacy invocation runs train-init-only as before.
    p.add_argument(
        "--n-test-rollouts", type=int, default=0,
        help="Number of fresh-seed test rollouts (DP's n_test).  Each "
        "uses --test-start-seed + i to seed the placement_initializer "
        "and env.reset()s to get a fresh domain-randomised init.  "
        "Default 0 (legacy: only --n-demos x --n-rollouts train inits).  "
        "Set to 50 to match DP's Can/ph protocol.",
    )
    p.add_argument(
        "--test-start-seed", type=int, default=100_000,
        help="First seed used for the test-rollout sweep.  DP's "
        "Can/ph config uses 100_000.",
    )
    p.add_argument("--H", type=int, default=120)
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--K", type=int, default=10)
    p.add_argument(
        "--lambda_", type=float, default=0.5,
        help="MBD softmax temperature (NOT the prior weight; that's "
             "--lambda-prior).",
    )
    p.add_argument("--sigma-max", type=float, default=0.25)
    p.add_argument("--sigma-min", type=float, default=0.05)
    # Prior warm-start sample steps.  Previously hard-coded to 32 in
    # the ``initial_mean_fn`` closure for ``--prior flow``.  For DP
    # checkpoints (which use DDPM ancestral sampling) the standalone
    # eval shows full K=100 dominates K=18/K=32 -- so this flag must
    # match the standalone setting when comparing MBD to standalone.
    # The flag is wired only for ``--prior flow``; BC and zero-train
    # priors ignore it (their warm-start is deterministic / non-iterative).
    p.add_argument(
        "--prior-n-steps", type=int, default=32,
        help="Number of sampling steps the flow prior takes when "
             "generating the warm-start trajectory at the top of each "
             "chunk.  For v3/v6 (Karras-EDM) this is the Heun-sampler "
             "step count; for DP this is the DDPM ancestral / DDIM "
             "step count.  Default 32 (historical).  Set to 100 to "
             "match the DP paper-faithful inference setting; for v3/v6 "
             "18 is usually enough.",
    )
    # Score-mode for trained priors (added 2026-05-24, see _load_prior).
    # ``learned`` reproduces the original behaviour: query the trained
    # prior's score field ``-ε(mu, t(σ))/σ`` at the current MBD iterate.
    # ``pull`` replaces that score with the retrieval-style
    # ``(a_warmstart - mu)`` factory used by demo_1nn / BC -- i.e.
    # treat the trained prior as a one-shot warm-start generator and
    # use a stationary Gaussian pull-back for the per-iter score.
    # Empirically the ``learned`` path is destructive for strong DP
    # priors (-36.7 pp on the 36.7% DP H=16 baseline); ``pull`` is the
    # operating mode that produced the 73.8% demo_1nn headline.
    p.add_argument(
        "--score-mode",
        choices=("learned", "pull"),
        default="learned",
        help="How the trained prior's per-iter score is computed when "
             "--prior=flow.  'learned' (default) = query the model's "
             "ε-prediction at the current iterate -ε/σ; 'pull' = "
             "use the retrieval-style stationary pull (a_warmstart - mu) "
             "with a_warmstart the prior's chunk-start sample.  "
             "Ignored when --prior is not 'flow' (retrieval priors "
             "always use 'pull').",
    )
    # Warm-start corruption probe (added 2026-05-25).  When >0, adds
    # zero-mean Gaussian noise of the requested std (in normalised
    # action units) to the warm-start trajectory before clipping +
    # before MBD's inner loop sees it.  The noise is rng-deterministic
    # per (base_seed, demo, rollout, chunk) so re-runs are reproducible.
    # Use case: deliberately push a strong prior's warm-start off the
    # success manifold to test whether MBD's pull-mode rescue actually
    # *recovers* a corrupted warm-start (the "controlled rescue" probe
    # for the K6/Slate-L sweep).  Default 0.0 = no corruption (the
    # historical behaviour).
    p.add_argument(
        "--warmstart-noise-sigma", type=float, default=0.0,
        help="If > 0, add iid Gaussian noise of this std (in "
             "normalised action units) to the warm-start trajectory "
             "before passing it to MBD's inner loop.  Default 0.0 "
             "(no corruption).  Used by the controlled-rescue probe "
             "to test whether pull-mode MBD can recover an "
             "intentionally degraded warm-start.",
    )

    # MPC outer loop
    p.add_argument(
        "--mpc-replans", type=int, default=0,
        help="Number of mid-episode replans. 0 = single-shot (exp 07). "
             "1+ = MPC chunking (exp 05 / 10).",
    )
    p.add_argument(
        "--mpc-chunk-steps", type=int, default=0,
        help="Env steps per chunk before replanning. 0 = auto "
             "(H // (mpc_replans + 1)).",
    )
    # Receding-horizon execution budget.  Required when the planner
    # horizon ``H`` is *shorter* than the desired episode length
    # (Diffusion-Policy-style closed-loop control with H=16,
    # n_action_steps=8, episode>=240).  When unset, unified_loop
    # falls back to ``n_chunks * chunk_steps`` if that exceeds H,
    # else ``H`` (pre-fix backward-compat).  See unified_loop.py for
    # the full discussion.
    p.add_argument(
        "--max-env-steps", type=int, default=0,
        help="Hard cap on total env steps per rollout.  0 = auto: "
             "use ``n_chunks * chunk_steps`` when that exceeds H, "
             "else H.  Set explicitly to e.g. 240 to match the "
             "standalone DP eval's max_env_steps when running "
             "closed-loop MBD on top of a short-H model.",
    )

    # Misc
    # n_workers >= 2 routes MBD's M perturbations through subprocess
    # workers that each own a forked robosuite env, leaving the MAIN
    # env untouched.  Default was 1 for many months but produced a
    # subtle chunk-boundary controller-state corruption on the
    # n_workers=1 path (env.reset() between chunks wipes robosuite's
    # Python-side controller cache; set_sim_state only restores
    # Mujoco qpos/qvel).  This bit Phase-C catastrophically (K=0
    # test rate collapsed from 80% standalone to 0% unified).  We
    # default to 2 going forward; users who specifically need the
    # legacy in-process behavior (e.g. to reproduce pre-2026-05-30
    # numbers) can pass ``--n-workers 1``.  See
    # experiments/17_dp_hybrid/results/phase_d_2026_05_30/BRIEF.md.
    p.add_argument("--n-workers", type=int, default=2)
    p.add_argument(
        "--allow-unsafe-nworkers",
        action="store_true",
        help="Override the safety guard that refuses --n-workers<=1 when "
             "chunked MPC (--mpc-replans>=1) runs MBD (--K>0).  That "
             "combination triggers the chunk-boundary controller-state "
             "corruption documented in unified_loop.py:579-601 (env.reset() "
             "between chunks wipes robosuite's Python-side controller cache; "
             "set_sim_state only restores Mujoco qpos/qvel) and silently "
             "collapses unified success to ~0% (see "
             "docs/results_writeups/vast_dp_grid_recovery_analysis_2026_06_11.md). "
             "Only pass this to deliberately reproduce the pre-2026-05-30 "
             "broken numbers.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--deterministic-bc-reset",
        action=argparse.BooleanOptionalAction, default=True,
        help="Re-seed env before BC warm-start. Default ON in this "
             "unified driver (matches exp 07's de-facto behaviour); "
             "passing --no-deterministic-bc-reset reverts to exp 05's "
             "pre-fix behaviour.",
    )
    p.add_argument(
        "--deterministic-prior-warmstart",
        action=argparse.BooleanOptionalAction, default=True,
        help="Seed torch's global RNG before each rollout's prior "
             "warm-start sample so the flow-prior sampler is "
             "deterministic per (seed, demo, rollout, chunk). "
             "Default ON; --no-deterministic-prior-warmstart "
             "reverts to the pre-2026-05-16 behaviour where the "
             "warm-start drifts with whatever ambient torch state "
             "the driver leaves.  No-op for --prior=bc.",
    )

    # Branches A / B no-ops (Phase 1 just declares them; the
    # respective branches wire them up).
    p.add_argument(
        "--value-baseline-ckpt", type=Path, default=None,
        help="(Branch A Task A2) Path to a trained V_phi.pt. "
             "Currently a no-op in Phase 1.",
    )
    p.add_argument(
        "--shaped-cost", action="store_true",
        help="(Branch A Task A4) Use dense object-EEF / lift / grasp "
             "rewards. Currently a no-op in Phase 1.",
    )
    p.add_argument(
        "--raw-sparse-reward", action="store_true",
        help="(§7.1 sparse-reward probe) Skip the dense-reward shaping "
             "that wrap_env_with_dense_reward normally adds for "
             "sparse-only tasks (tool_hang/transport), leaving the raw "
             "upstream sparse terminal reward.  J is then flat over "
             "non-goal rollouts (ESS=M, no cost gradient) -- the "
             "degenerate regime where both product and max-gate fall "
             "back to BC.  Used to test the prediction that the product "
             "revives CONTINUOUSLY as the cost densifies while the "
             "max-gate is threshold-gated.  Default off (= dense "
             "shaping, the historical behaviour).",
    )
    p.add_argument(
        "--lambda-schedule",
        choices=(
            "constant", "linear", "inverse_linear", "cosine",
            "sigma_linear", "sigma_inv_linear", "sigma_step",
        ),
        default="constant",
        help="Per-denoising-iteration multiplier on the prior weight "
             "(A2, 2026-05-20; sigma-aware variants added 2026-06-01 "
             "follow-up C1).  'constant' (default) reproduces the "
             "historical behaviour bit-for-bit.  k-indexed shapes "
             "(falsified by Slate G on Can at lambda=1): "
             "'linear' 1.0 -> 0.0 in k, 'inverse_linear' 0.0 -> 1.0 "
             "in k, 'cosine' 0.5*(1+cos(pi*k/K)).  sigma-indexed shapes "
             "(new; couple lambda to the *actual* noise level "
             "sigma_k instead of iteration index k):  'sigma_linear' "
             "= (sigma_k - sigma_min) / (sigma_max - sigma_min) "
             "(strong at high noise, weak at low noise -- amplifies "
             "the implicit Pan sigma-decay).  'sigma_inv_linear' = "
             "1 - that (weak at high noise, strong at low noise -- "
             "cancels the implicit decay, peaks near sigma_min).  "
             "'sigma_step' = 1.0 in the first half of the sigma "
             "schedule, 0.0 in the second half (binary on/off at "
             "sigma midpoint).",
    )
    p.add_argument(
        "--K-truncate", type=int, default=0,
        help="(Branch C Phase 4) Truncate denoising at this many "
             "iterations (0 = use full K). Currently a no-op.",
    )
    p.add_argument(
        "--save-executed-actions", action="store_true",
        help="(C6, 2026-06-01) Capture every env.step()-executed "
             "action into the rollout JSON under "
             "chunks[i].executed_actions.  Enables per-step "
             "(bias, variance) decomposition via "
             "measure_bias_variance.py.  Default off (cheap to "
             "toggle, but adds ~H * action_dim floats per "
             "rollout to the JSON).",
    )
    p.add_argument(
        "--save-step-trace", action="store_true",
        help="Capture compact per-env-step observations and rewards under "
             "chunks[i].step_trace for failure-phase analysis. Default off.",
    )
    p.add_argument(
        "--replan-trigger",
        choices=("always", "uncertainty"), default="always",
        help="(Branch C Phase 5) When to invoke MBD vs falling back "
             "to direct BC execution. Currently only 'always' wired.",
    )

    # Tempered / gated combination rule (feat/mbd-gated-tempered-hybrid,
    # 2026-06-04).  Implements the product<->max-gate family from
    # docs/theory/09_mbd_data_guided_comparison.md §6.1/§6.2.  Default
    # 'score' is the legacy mean-space blend (all prior cells reproduce
    # bit-for-bit); 'tempered' switches to per-sample model-vs-demo
    # reweighting where --temper-p selects the rule (0 = product/PoE,
    # inf = MBD-Demo max-gate, finite>0 = the §6.2 interpolation).
    p.add_argument(
        "--combine-mode", choices=("score", "tempered"), default="score",
        help="Model/demo fusion rule. 'score' (default) = legacy "
             "mean-space score blend (product-of-experts via the "
             "per-step pull). 'tempered' = per-sample tempered-mixture "
             "reweighting (§6.1/§6.2); requires --warm-start + --prior. "
             "In tempered mode --lambda-prior is the demo mixture "
             "weight (mix_demo).",
    )
    p.add_argument(
        "--temper-p", type=float, default=0.0,
        help="Tempering exponent for --combine-mode=tempered. 0 = "
             "product (PoE, our rule); 'inf' = MBD-Demo max-gate "
             "(Eq. 13); finite >0 = the generalised-mean interpolation "
             "(§6.2/#7). Pass 'inf' as a float via e.g. --temper-p inf.",
    )
    p.add_argument(
        "--sigma-demo", type=float, default=0.25,
        help="Observation-noise std of the demo Gaussian "
             "N(tau|demo,sigma_demo^2 I) used by the tempered/gated "
             "combination. Default 0.25 (matches the canonical pull).",
    )
    p.add_argument(
        "--demo-sigma-mode", choices=("const", "decay", "adaptive"),
        default="const",
        help="Demo-pull width schedule in tempered mode. 'const' "
             "(default, MBD-faithful) holds sigma_demo fixed; 'decay' "
             "scales it by sigma_k/sigma_max (our annealed variant); "
             "'adaptive' (follow-up #2) scales it by the model-vs-demo "
             "best-cost gap so the gate engages off-distribution and "
             "relaxes on-distribution.",
    )
    p.add_argument(
        "--demo-sigma-adapt-beta", type=float, default=1.0,
        help="Sensitivity of --demo-sigma-mode=adaptive: larger beta "
             "shrinks/widens the demo Gaussian more aggressively per "
             "unit of (best_cost - demo_cost)/lambda gap. Ignored unless "
             "demo-sigma-mode=adaptive.",
    )
    p.add_argument(
        "--sigma-schedule", choices=("decay", "const"), default="decay",
        help="(#4 ablation) Pull-width schedule for the legacy "
             "--combine-mode=score path. 'decay' (default) = the "
             "historical sigma_t-annealed pull (sigma_k^2); 'const' = "
             "fixed-width MBD-style pull (sigma_max^2). No effect when "
             "the prior is off.",
    )
    p.add_argument(
        "--demo-phase-reward", action="store_true",
        help="(#3, §7.1 route 3) Replace the task's dense shaping with a "
             "zero-train demo-phase progress reward: reward = "
             "--demo-phase-scale * (fraction along the nearest demo). "
             "Densifies sparse-terminal tasks (tool_hang/transport) using "
             "the demo bank, no value training. Mutually exclusive with "
             "--raw-sparse-reward.",
    )
    p.add_argument(
        "--demo-phase-scale", type=float, default=10.0,
        help="Scale on the demo-phase progress reward (#3). Default 10 "
             "matches the dense-shaping subgoal magnitude.",
    )
    p.add_argument(
        "--log-gate-diagnostic", action="store_true",
        help="(#3, §6.3) Passively log the per-iter likelihood-ratio "
             "diagnostic (gate_frac + ldem_minus_lmod) on ANY run "
             "without changing the update -- tests whether the "
             "off-distribution collapse coincides with ldem << lmod. "
             "Requires --warm-start + --prior; costs one extra demo "
             "roll-out per chunk. Ignored in --combine-mode=tempered "
             "(already logged there).",
    )

    # Adaptation flags (feat/adapt-demo-scaling, 2026-05-25).
    # Decouples the retrieval bank size from --n-demos (which
    # controls EVALUATION rollout count).  Used to trace the
    # retrieval scaling law for the demo_*nn priors on Can.
    # Maps onto the p^data factor of the soft-control target
    # (docs/theory/01_soft_control.md sec 1).  See
    # docs/adaptation/03_demo_scaling_spec.md for the experiment.
    p.add_argument(
        "--prior-bank-size", type=int, default=0,
        help="Demo-bank scaling (feat/adapt-demo-scaling): limit "
             "the retrieval bank to the first N demos in the "
             "dataset.  0 (default) = use all demos = bit-for-bit "
             "reproduction of the canonical headline cells.  "
             "Positive N truncates the bank without affecting "
             "evaluation (--n-demos still controls the rollout "
             "count).  Used to trace the bank-size scaling law "
             "for the retrieval priors on Can; see "
             "docs/adaptation/03_demo_scaling_spec.md.",
    )

    # Adaptation flags (feat/adapt-incremental-prior, 2026-05-25).
    # Map onto the p^data factor of the soft-control target by
    # CHANGING THE BANK SOURCE at runtime -- either to a different
    # task's demos (cross-task transfer prior, T1) or to an
    # incrementally-augmented demo set under perturbed dynamics
    # (incremental adaptation, T2).  Orthogonal to --prior-bank-size
    # (which truncates whichever bank dataset is selected here).
    # See docs/adaptation/04_incremental_prior_spec.md.
    p.add_argument(
        "--prior-bank-dataset", type=Path, default=None,
        help="Incremental adaptation (feat/adapt-incremental-prior): "
             "load the retrieval bank from a DIFFERENT dataset HDF5 "
             "than the evaluation init-state dataset (--dataset / "
             "--task).  Default None = use --dataset for both = "
             "canonical headline cells reproduced bit-for-bit.  Set "
             "this to a cross-task dataset for the T1 transfer-prior "
             "cell (e.g. Lift demos for a Can eval), or to an "
             "incrementally-augmented bank for the T2 incremental-"
             "adaptation cell (canonical 200 Can demos + 5 fresh "
             "mass=1.3 demos).  See "
             "docs/adaptation/04_incremental_prior_spec.md.",
    )
    p.add_argument(
        "--prior-bank-task", type=str, default=None,
        help="Companion to --prior-bank-dataset: declares the "
             "task family of the bank dataset, so the correct "
             "task-aware obs schema is used (priors_zerotrain.get_obs_keys).  "
             "Default None = use --task for both eval and bank "
             "(matches canonical behaviour).  Must be set to a "
             "different task family iff --prior-bank-dataset is "
             "a cross-task dataset.",
    )

    # Adaptation flags (feat/adapt-cost-time + feat/adapt-dynamics,
    # 2026-05-25 + 2026-05-26).  Each flag perturbs ONE factor of the
    # soft-control target
    # q*(tau) = p^data(tau) * exp(-(lambda/beta) J(tau, x_goal, f)):
    #   --goal-perturbation-xyz       -> moves the goal body, affects
    #                                    J (the cost) via the deployed
    #                                    success check; see
    #                                    docs/adaptation/01_cost_time_spec.md
    #   --mass-scale / --friction-scale -> scales robot-link mass and
    #                                    object-table friction, affects
    #                                    f (the simulator dynamics); see
    #                                    docs/adaptation/02_dynamics_spec.md
    # Both invocations call into env_perturbations.py and propagate to
    # every multiprocessing worker's env (via the perturbations dict
    # threaded through unified_plan_and_execute -> make_rollout_fn ->
    # _worker_init), so MBD's planning sees the same cost+dynamics as
    # the executed rollout.
    p.add_argument(
        "--goal-perturbation-xyz",
        type=float, nargs=3, default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="Cost-time adaptation (feat/adapt-cost-time): translate "
             "the task goal body by (dx, dy, dz) metres in the world "
             "frame after set_sim_state.  Default (0, 0, 0) = "
             "no-op = canonical evaluation.  Non-zero values invoke "
             "env_perturbations.apply_goal_perturbation on the main "
             "env AND propagate the same perturbation to every "
             "multiprocessing worker's env (so MBD's planning sees "
             "the same cost as the executed rollout).  See "
             "docs/adaptation/01_cost_time_spec.md for the four-cell "
             "experiment plan (goal-swap and obstacle-injection).",
    )
    p.add_argument(
        "--mass-scale", type=float, default=1.0,
        help="Dynamics adaptation (feat/adapt-dynamics): scale every "
             "robot link mass by this factor before each rollout.  "
             "Default 1.0 = canonical evaluation.  Sweep "
             "{0.7, 1.0, 1.3, 1.5} for the headline dynamics-adaptation "
             "cell.  Non-1.0 values invoke "
             "env_perturbations.apply_dynamics_perturbation and "
             "propagate to MBD workers identically to goal_xyz.  See "
             "docs/adaptation/02_dynamics_spec.md.",
    )
    p.add_argument(
        "--friction-scale", type=float, default=1.0,
        help="Dynamics adaptation (feat/adapt-dynamics): scale the "
             "object-table tangential friction coefficient by this "
             "factor.  Default 1.0 = canonical.  Sweep "
             "{0.5, 1.0, 2.0}.  Same env_perturbations + worker-"
             "propagation path as --mass-scale.",
    )

    # IO
    p.add_argument(
        "--output-dir", type=Path,
        default=_THIS_DIR / "results",
        help="Directory to write per-lambda JSONs into.",
    )
    p.add_argument(
        "--output-name-suffix", type=str, default="",
        help="Optional suffix appended to output filename (after "
             "the auto-generated prior/lambda/K/R tag).",
    )

    p.add_argument(
        "--smoke", action="store_true",
        help="Tiny config (2 demos, 1 rollout, K=4, lambdas=[0, 1]) "
             "for end-to-end wiring tests.",
    )

    # Per-rollout frame capture (presentation/render_prior_grid.py).
    # Guarded: when --video-dir is unset the rollout path is untouched.
    p.add_argument(
        "--video-dir", type=Path, default=None,
        help="If set, capture each rollout's EXECUTED-step frames via "
             "mujoco.Renderer and save them as "
             "<video-dir>/<prior>_lam-<L>_<demo>_r<r>.npz (frames uint8 "
             "T,H,W,3 + success/reward metadata).  Use --n-workers>=2 so "
             "the main env is stepped only during execution.  "
             "Default None = no capture.",
    )
    p.add_argument("--video-camera", type=str, default="agentview",
                   help="MuJoCo camera name for --video-dir capture.")
    p.add_argument("--video-h", type=int, default=256,
                   help="Capture frame height for --video-dir.")
    p.add_argument("--video-w", type=int, default=256,
                   help="Capture frame width for --video-dir.")

    return p.parse_args()


def _smoke_overrides(args: argparse.Namespace) -> None:
    args.n_demos = 2
    args.n_rollouts = 1
    args.K = 4
    args.lambda_prior = [0.0, 1.0]


def _warn_unimplemented_flags(args: argparse.Namespace) -> None:
    """Print a one-line notice when a not-yet-wired flag is set."""
    notes: list[str] = []
    if args.shaped_cost:
        notes.append(
            "  --shaped-cost is a no-op in Phase 1 (Branch A A4)."
        )
    if args.K_truncate not in (0, args.K):
        notes.append(
            "  --K-truncate is a no-op in Phase 1 (Branch C Phase 4)."
        )
    if args.replan_trigger != "always":
        notes.append(
            f"  --replan-trigger={args.replan_trigger} is a no-op "
            "in Phase 1 (Branch C Phase 5)."
        )
    # --goal-perturbation-xyz, --mass-scale, --friction-scale are all
    # fully wired on this branch (registry + main env mutation +
    # worker propagation via the perturbations dict).  No warning
    # emitted when any of them is set.
    if notes:
        print("[exp15] Phase-1 unimplemented-flag notice:")
        for n in notes:
            print(n)


def _guard_nworkers(args: argparse.Namespace) -> None:
    """Refuse the n_workers<=1 + chunked-MPC + MBD combination.

    When ``n_workers<=1`` the MBD perturbation pool runs in-process and
    clobbers the main env, so unified_loop.py:599 forces an
    ``env.reset()`` + ``set_sim_state`` between every MPC chunk.  That
    reset wipes robosuite's Python-side controller cache (OSC goal,
    interpolator, last-commanded pose) which ``set_sim_state`` does NOT
    restore, silently collapsing unified success to ~0%.  This is the
    exact bug that invalidated the vast_safe_dp_2026_06_10 grid (Square
    0/30, Can <=26.7%, lambda0 53.3%->0%); see
    ``docs/results_writeups/vast_dp_grid_recovery_analysis_2026_06_11.md``.

    Single-shot runs (``mpc_replans==0``, no chunk boundary) and pure
    warm-start delivery (``K==0``, MBD never runs) are unaffected, so the
    guard only fires on the corrupting triple.  Pass
    ``--allow-unsafe-nworkers`` to deliberately reproduce pre-2026-05-30
    broken numbers.
    """
    corrupting = (
        int(args.n_workers) <= 1
        and int(args.mpc_replans) >= 1
        and int(args.K) > 0
    )
    if corrupting and not args.allow_unsafe_nworkers:
        raise SystemExit(
            "[exp15] REFUSING to run: --n-workers={nw} with chunked MPC "
            "(--mpc-replans={r}>=1) and MBD (--K={k}>0) triggers the "
            "chunk-boundary controller-state corruption documented in "
            "unified_loop.py:599 and "
            "docs/results_writeups/vast_dp_grid_recovery_analysis_2026_06_11.md "
            "(collapses unified success to ~0%). Re-run with "
            "--n-workers >= 2. If you genuinely need the broken legacy "
            "behaviour, pass --allow-unsafe-nworkers.".format(
                nw=args.n_workers, r=args.mpc_replans, k=args.K,
            )
        )
    if corrupting and args.allow_unsafe_nworkers:
        print(
            "[exp15] WARNING: --allow-unsafe-nworkers set; running the "
            "KNOWN-CORRUPTING n_workers<=1 + chunked-MPC + MBD path. "
            "Results will be invalid (~0% success). See "
            "vast_dp_grid_recovery_analysis_2026_06_11.md.",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    if args.smoke:
        _smoke_overrides(args)
    _warn_unimplemented_flags(args)
    _guard_nworkers(args)

    # Default prior_checkpoint per choice.
    if args.prior_checkpoint is None:
        if args.prior == "flow":
            args.prior_checkpoint = (
                _EXP_08 / "results" / "v_theta.pt"
            )
        elif args.prior == "bc":
            args.prior_checkpoint = args.bc_checkpoint
        else:
            # Zero-train priors / stubs: no checkpoint.  Use a
            # sentinel string so the output JSON has a stable field.
            args.prior_checkpoint = Path(f"<none:{args.prior}>")

    lambdas = (
        tuple(args.lambda_prior) if args.lambda_prior
        else DEFAULT_LAMBDAS
    )
    base_seed = int(args.seed)

    # Env + bounds + params.
    dataset = args.dataset or default_dataset_path(args.task)
    env, _env_meta = make_env_for_dataset(dataset)
    if hasattr(env, "reward_shaping"):
        env.reward_shaping = True
    # Phase-C (2026-05-29): disable hard reset so the test-rollout
    # bucket's placement_initializer.rng overrides survive env.reset().
    # Mirrors standalone_dp.py's setup; required for --n-test-rollouts > 0.
    # No-op for the legacy train-init bucket (set_sim_state restores the
    # full sim state directly, bypassing placement randomisation entirely).
    if hasattr(env, "hard_reset"):
        env.hard_reset = False
    # Patch dense reward for tasks that ship sparse-only upstream
    # (tool_hang, transport).  No-op for can/lift/square (they
    # already have dense shaping in robosuite) and for any future
    # task without a registered shaping function.  See
    # experiments/15_unified_hybrid/env_dense_rewards.py for the
    # per-task definitions and the root-cause analysis (Slate H
    # H1, 2026-05-21).
    if getattr(args, "raw_sparse_reward", False):
        # §7.1 sparse-reward probe: leave the upstream sparse terminal
        # reward UNshaped so J is flat over non-goal rollouts.  Tests
        # the prediction that as the cost densifies, the product revives
        # continuously while the max-gate is threshold-gated.
        if getattr(args, "demo_phase_reward", False):
            raise ValueError(
                "--raw-sparse-reward and --demo-phase-reward are mutually "
                "exclusive (one removes shaping, the other adds it)."
            )
        print("[run_can] --raw-sparse-reward: skipping dense-reward "
              "shaping (sparse J probe).")
    elif getattr(args, "demo_phase_reward", False):
        # §7.1 route 3 (#3): zero-train demo-phase progress reward in
        # place of the hand-crafted dense shaping.
        from demo_phase_reward import wrap_env_with_demo_phase_reward
        from priors_zerotrain import get_obs_keys as _gok
        wrap_env_with_demo_phase_reward(
            env, args.task, dataset, _gok(args.task),
            n_demos=(args.n_demos if args.n_demos > 0 else None),
            scale=float(args.demo_phase_scale),
        )
    else:
        from env_dense_rewards import wrap_env_with_dense_reward
        wrap_env_with_dense_reward(env, args.task)
    # Cost-time + dynamics adaptation hooks.  Each is a no-op when its
    # flag is at the canonical default; otherwise the corresponding
    # helper in env_perturbations.py is called once per env
    # instantiation.  See:
    #   docs/adaptation/01_cost_time_spec.md (goal-perturbation-xyz)
    #   docs/adaptation/02_dynamics_spec.md (mass-scale / friction-scale)
    goal_xyz = tuple(float(v) for v in args.goal_perturbation_xyz)
    mass_scale = float(args.mass_scale)
    friction_scale = float(args.friction_scale)

    if goal_xyz != (0.0, 0.0, 0.0):
        from env_perturbations import apply_goal_perturbation
        apply_goal_perturbation(env, args.task, goal_xyz)
    if mass_scale != 1.0 or friction_scale != 1.0:
        from env_perturbations import apply_dynamics_perturbation
        apply_dynamics_perturbation(
            env, args.task,
            mass_scale=mass_scale,
            friction_scale=friction_scale,
        )

    # Build the perturbations dict that's threaded through
    # unified_plan_and_execute -> make_rollout_fn -> _worker_init,
    # so multiprocessing workers' envs are perturbed identically to
    # this main env.  Without this propagation, workers would see
    # the canonical (unperturbed) goal/dynamics and MBD's planning
    # would silently optimise the wrong cost / state-transition.
    # See docs/adaptation/01_cost_time_spec.md and
    # docs/adaptation/02_dynamics_spec.md for the experiment plans.
    _any_perturbation = (
        goal_xyz != (0.0, 0.0, 0.0)
        or mass_scale != 1.0
        or friction_scale != 1.0
    )
    perturbations = (
        {
            "task": args.task,
            "goal_xyz": goal_xyz,
            "mass_scale": mass_scale,
            "friction_scale": friction_scale,
        }
        if _any_perturbation
        else None
    )

    low, high = get_action_bounds(env)
    action_dim = int(low.shape[0])
    low_f32 = low.astype(np.float32)
    high_f32 = high.astype(np.float32)

    # Task-aware obs keys for the score-bridge priors.  BC keeps its
    # legacy ``OBS_KEYS`` (Can-only) because the saved policy.pt was
    # trained against it; the zero-train priors honour ``--task``.
    # NOTE: import via ``priors_zerotrain`` (which loads exp 03's
    # dataset.py by absolute file path) to dodge the exp 08
    # ``dataset`` module collision -- bare ``from dataset import ...``
    # resolves to exp 08's flow-matching dataset here.
    from priors_zerotrain import get_obs_keys

    prior_obs_keys: tuple[str, ...] | None
    if args.prior in (
        "demo_1nn", "demo_knn", "demo_random", "demo_mean",
        "demo_lwr", "demo_topk_random",
        "demo_1nn_proprio", "demo_1nn_object",
        "demo_softnn",
        "demo_1nn_contrastive",
    ):
        prior_obs_keys = get_obs_keys(args.task)
    elif args.prior == "flow":
        # Phase-C (2026-05-29): flow priors carry a ``task`` field in
        # the ckpt blob (written by ``08_flow_matching_prior/train_dp.py``
        # since 2026-05-28).  Read it directly so the runtime obs
        # vector matches the schema the prior was trained on:
        #   - ``task == "can"``      -> 46-D legacy schema (every
        #                               pre-Phase-B flow ckpt).
        #   - ``task == "can_dp"``   -> 23-D DP paper schema
        #                               (Phase-B; ep0199 = 80 % test).
        # Backward-compat: legacy ckpts without ``task`` default to
        # ``"can"`` inside the helper, preserving the pre-Phase-C
        # behaviour bit-for-bit for any flow ckpt trained before
        # 2026-05-28.  We peek the ckpt blob for both ``task`` and
        # ``config["obs_dim"]`` so the assertion below catches any
        # train/runtime schema mismatch before the first rollout.
        if args.prior_checkpoint is not None and args.prior_checkpoint.is_file():
            _blob_peek = torch.load(
                args.prior_checkpoint, map_location="cpu", weights_only=False
            )
            _ckpt_task = str(_blob_peek.get("task", "can"))
            _ckpt_obs_dim = int(_blob_peek.get("config", {}).get("obs_dim", -1))
            del _blob_peek  # free the 500 MB state_dict we don't need here
        else:
            _ckpt_task = args.task  # fallback: trust the CLI
            _ckpt_obs_dim = -1
        prior_obs_keys = get_obs_keys(_ckpt_task)

        # Pre-flight schema check: build a probe obs vector via the
        # same code path the rollout loop uses and assert the trailing
        # dim matches the prior's expected obs_dim.  A mismatch is
        # almost always a schema bug; we fail loud and fast.
        if _ckpt_obs_dim > 0:
            from unified_loop import _make_obs_vec_for_prior as _probe
            _probe_vec = _probe(env, obs_keys=prior_obs_keys)
            if _probe_vec.shape[0] != _ckpt_obs_dim:
                raise RuntimeError(
                    f"obs schema mismatch: --prior=flow ckpt declares "
                    f"obs_dim={_ckpt_obs_dim} (task={_ckpt_task!r}, "
                    f"keys={prior_obs_keys}) but the runtime obs builder "
                    f"produces a {_probe_vec.shape[0]}-D vector.  This "
                    f"would silently corrupt the FiLM conditioning.  "
                    f"Re-train the prior on the correct schema, or "
                    f"override --task on the CLI to match the ckpt."
                )
        print(
            f"[exp15] --prior=flow ckpt task={_ckpt_task!r}  "
            f"obs_dim={_ckpt_obs_dim}  ({len(prior_obs_keys)} keys)"
        )
    else:
        # BC and other priors: use the task-aware obs schema.  Pre-2026-05-28
        # this branch hardcoded prior_obs_keys=None (which defaults to
        # exp 03's Can OBS_KEYS) because BC was Can-only.  After
        # feat/adapt-bc-multitask-obs merged on 2026-05-28, BC supports
        # --task square / lift / tool_hang and the runtime obs builder
        # MUST use the task-specific keys (e.g. SquareNut_pos for
        # square, vs Can_pos for can).  Without this fix, the Stage B
        # eval cells of the T2-Square slate crash at unified_loop:142
        # with KeyError: 'Can_pos' on a Square env -- documented in
        # docs/coordination/offload_plan_2026_06_01.md (2026-06-02
        # 02:22 PDT chain postmortem).
        prior_obs_keys = get_obs_keys(args.task)

    initial_mean_fn, score_fn_factory, prior_label = _load_prior(
        args, action_dim
    )

    # Branch A Task A2: load V_phi state-value baseline (control variate).
    # When set, the planner uses the obs-collecting rollout_fn and the
    # adapter's V_phi-subtracting cost_fn (exp 16/adapter.py).
    # NOTE: exp 08 also has a module named ``adapter`` (for the flow
    # prior), so we load exp 16's by absolute file location to avoid
    # the bare-import collision.
    if args.value_baseline_ckpt is not None:
        _EXP_16 = _EXPERIMENTS_DIR / "16_value_baseline"
        if str(_EXP_16) not in sys.path:
            sys.path.insert(0, str(_EXP_16))
        _spec16 = _importlib_util.spec_from_file_location(
            "exp16_adapter", _EXP_16 / "adapter.py"
        )
        exp16_adapter = _importlib_util.module_from_spec(_spec16)
        sys.modules["exp16_adapter"] = exp16_adapter
        _spec16.loader.exec_module(exp16_adapter)
        v_phi = exp16_adapter.ValuePhi.load(
            args.value_baseline_ckpt, device="cpu"
        )
        active_make_rollout_fn = exp04_run.make_rollout_fn_with_obs
        active_cost_fn = exp16_adapter.wrap_cost_fn_with_value_baseline(
            exp04_run.cost_fn, v_phi
        )
        value_baseline_label = f"V_phi:{args.value_baseline_ckpt}"
    else:
        active_make_rollout_fn = exp04_run.make_rollout_fn
        active_cost_fn = exp04_run.cost_fn
        value_baseline_label = "none"

    print(
        f"[exp15] prior={prior_label}  lambdas={lambdas}  "
        f"K={args.K}  R={args.mpc_replans}  "
        f"M={args.M}  sigma_max={args.sigma_max}  "
        f"warm_start={args.warm_start}  "
        f"value_baseline={value_baseline_label}"
    )

    schedule = cosine_sigma_schedule(
        args.K, sigma_max=args.sigma_max, sigma_min=args.sigma_min
    )
    params = MBDParams(
        H=args.H, M=args.M, K=args.K, action_dim=action_dim,
        lambda_=args.lambda_, sigma_schedule=schedule,
        clip_action=(low_f32, high_f32),
    )

    # A2: build the per-iteration prior-weight schedule.  ``constant``
    # leaves the historical behaviour bit-for-bit (pass ``None``).
    #
    # k-indexed shapes ('linear', 'inverse_linear', 'cosine') were
    # added 2026-05-20 and falsified on Can at lambda=1 in Slate G
    # (all underperformed 'constant' by 12-57 pp).  sigma-indexed
    # shapes were added 2026-06-01 (follow-up C1) to test whether
    # coupling lambda to the actual noise level sigma_k (instead of
    # the iteration index k) recovers any of the lift.
    lambda_schedule_arr: np.ndarray | None = None
    if args.lambda_schedule == "linear":
        # 1.0 at k=0  ->  0.0 at k=K-1
        lambda_schedule_arr = np.linspace(
            1.0, 0.0, num=int(args.K), dtype=np.float32
        )
    elif args.lambda_schedule == "inverse_linear":
        lambda_schedule_arr = np.linspace(
            0.0, 1.0, num=int(args.K), dtype=np.float32
        )
    elif args.lambda_schedule == "cosine":
        # 0.5 * (1 + cos(pi * k / max(K-1, 1)));  starts at 1, ends at 0
        ks = np.arange(int(args.K), dtype=np.float32)
        lambda_schedule_arr = (
            0.5 * (1.0 + np.cos(np.pi * ks / max(int(args.K) - 1, 1)))
        ).astype(np.float32)
    elif args.lambda_schedule in ("sigma_linear", "sigma_inv_linear", "sigma_step"):
        # sigma-indexed shapes: read sigma_k from the schedule and
        # build the multiplier as a function of sigma.  Note that
        # denoise_loop ALSO multiplies by (sigma_k / sigma_max)
        # unconditionally, so these are stacked on top of the
        # implicit Pan sigma-decay -- see the schedule definitions
        # in the --lambda-schedule help text.
        sigmas = np.asarray(schedule, dtype=np.float32)
        sigma_span = max(float(args.sigma_max) - float(args.sigma_min), 1e-8)
        if args.lambda_schedule == "sigma_linear":
            lambda_schedule_arr = (
                (sigmas - float(args.sigma_min)) / sigma_span
            ).astype(np.float32)
        elif args.lambda_schedule == "sigma_inv_linear":
            lambda_schedule_arr = (
                1.0 - (sigmas - float(args.sigma_min)) / sigma_span
            ).astype(np.float32)
        elif args.lambda_schedule == "sigma_step":
            mid_sigma = 0.5 * (float(args.sigma_max) + float(args.sigma_min))
            lambda_schedule_arr = (sigmas > mid_sigma).astype(np.float32)
        # Clamp to [0, 1] to guard against any tiny numerical
        # drift outside the unit interval.
        lambda_schedule_arr = np.clip(lambda_schedule_arr, 0.0, 1.0).astype(
            np.float32
        )

    # Auto chunk_steps if user passed 0.
    n_chunks = args.mpc_replans + 1
    chunk_steps = (
        args.mpc_chunk_steps
        if args.mpc_chunk_steps > 0
        else args.H // n_chunks
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load init states.
    with h5py.File(str(dataset), "r") as h5:
        demo_keys = sorted(
            h5["data"].keys(), key=lambda k: int(k.split("_")[1])
        )
        if args.n_demos > 0:
            demo_keys = demo_keys[: args.n_demos]
        init_states = {
            k: np.asarray(h5[f"data/{k}/states"][0]) for k in demo_keys
        }

    # Optional per-rollout frame capture (guarded; see --video-dir).
    # mujoco.Renderer draws directly from the env's MuJoCo model/data,
    # so no robosuite offscreen renderer is needed.  We wrap the MAIN
    # env's step() to grab one frame per EXECUTED action; with
    # --n-workers>=2 the main env is touched only during execution.
    _video_dir = args.video_dir
    _cur_frames: list = []
    _renderer = None
    if _video_dir is not None:
        import mujoco  # noqa: E402 (lazy; only when capturing)

        if int(args.n_workers) < 2:
            print("[exp15] WARNING --video-dir with --n-workers<2 captures "
                  "MBD planning frames too; pass --n-workers>=2.")
        _video_dir.mkdir(parents=True, exist_ok=True)
        _renderer = mujoco.Renderer(
            env.sim.model._model,
            height=int(args.video_h), width=int(args.video_w),
        )
        _orig_env_step = env.step

        def _capturing_step(action, *a, **kw):
            out = _orig_env_step(action, *a, **kw)
            try:
                _renderer.update_scene(
                    env.sim.data._data, camera=args.video_camera
                )
                _cur_frames.append(_renderer.render().copy())
            except Exception as _e:  # pragma: no cover - render best-effort
                print(f"[exp15] frame capture skipped: {_e!r}")
            return out

        env.step = _capturing_step
        print(f"[exp15] frame capture ON -> {_video_dir} "
              f"(cam={args.video_camera}, {args.video_w}x{args.video_h})")

    # Lambda sweep.
    summary_per_lambda: list[dict] = []
    for lam in lambdas:
        per_rollout: list[dict] = []
        wall0 = time.perf_counter()
        for i, key in enumerate(demo_keys):
            init_state = init_states[key]
            for r_idx in range(args.n_rollouts):
                rng_lam = np.random.default_rng(
                    np.uint32(hash(
                        ("lam", base_seed, key, r_idx)
                    ) & 0xFFFFFFFF)
                )

                # Per-rollout, per-chunk torch seed.  Only used by
                # the flow prior's ``sample_initial_mean``; BC's
                # ``initial_mean_fn`` is deterministic from s0 alone
                # so this is a no-op for --prior=bc.  See
                # unified_loop.WarmstartSeedFn for the motivation.
                warmstart_seed_fn = None
                if args.deterministic_prior_warmstart and prior_label in (
                    "flow", "flow_v3", "flow_pull", "flow_v3_pull",
                ):
                    def warmstart_seed_fn(
                        chunk_idx,
                        b=base_seed, _k=key, _r=int(r_idx),
                    ):
                        return int(np.uint32(hash((
                            "flow_warmstart", b, _k, _r, int(chunk_idx),
                        )) & 0xFFFFFFFF))

                # Reset T_obs > 1 prior's obs buffer at each rollout
                # boundary.  Mirrors ``standalone_dp.py``'s
                # ``prior.reset_obs_buffer()`` call before each
                # rollout; required so per-rollout obs histories
                # don't leak across rollouts.  No-op for stateless
                # priors (BC, demo_*, flow v1/v2/v3-T_obs=1).
                _p_for_reset = getattr(
                    initial_mean_fn, "_prior_for_buffer_reset", None
                )
                if _p_for_reset is not None and hasattr(
                    _p_for_reset, "reset_obs_buffer"
                ):
                    _p_for_reset.reset_obs_buffer()

                t0 = time.perf_counter()
                result = unified_plan_and_execute(
                    env, init_state,
                    prior_funcs=(initial_mean_fn, score_fn_factory),
                    lambda_prior=float(lam),
                    params=params,
                    rng=rng_lam,
                    n_replans=args.mpc_replans,
                    chunk_steps=chunk_steps,
                    action_low=low_f32,
                    action_high=high_f32,
                    no_warm_start=not args.warm_start,
                    n_workers=args.n_workers,
                    dataset_path=dataset,
                    perturbations=perturbations,
                    make_rollout_fn=active_make_rollout_fn,
                    cost_fn=active_cost_fn,
                    warmstart_seed_fn=warmstart_seed_fn,
                    obs_keys=prior_obs_keys,
                    prior_lambda_schedule=lambda_schedule_arr,
                    action_history_len=int(args.action_history_len),
                    save_executed_actions=bool(args.save_executed_actions),
                    max_env_steps=(
                        int(args.max_env_steps)
                        if args.max_env_steps > 0
                        else None
                    ),
                    warmstart_noise_sigma=float(args.warmstart_noise_sigma),
                    combine_mode=args.combine_mode,
                    temper_p=float(args.temper_p),
                    sigma_demo=float(args.sigma_demo),
                    demo_sigma_mode=args.demo_sigma_mode,
                    demo_sigma_adapt_beta=float(args.demo_sigma_adapt_beta),
                    prior_sigma_mode=args.sigma_schedule,
                    log_gate_diagnostic=bool(args.log_gate_diagnostic),
                    save_step_trace=bool(args.save_step_trace),
                )
                elapsed = time.perf_counter() - t0

                per_rollout.append({
                    "demo": key,
                    "rollout_idx": int(r_idx),
                    "rollout_kind": "train",
                    "lambda_prior": float(lam),
                    "final_reward": result["final_reward"],
                    "success": bool(result["final_success"]),
                    "plan_elapsed_s": float(result["plan_elapsed_s"]),
                    "total_elapsed_s": float(elapsed),
                    "chunks": result["chunks"],
                })
                print(
                    f"[exp15] prior={prior_label} lam={lam:>5}  "
                    f"R={args.mpc_replans}  K={args.K}  "
                    f"{len(per_rollout):>3}  train {key}#{r_idx}  "
                    f"reward={result['final_reward']:+.2f}  "
                    f"success={result['final_success']}"
                )

                # Flush this rollout's captured execution frames.
                if _video_dir is not None:
                    fr = np.asarray(_cur_frames, dtype=np.uint8)
                    out_npz = (
                        _video_dir
                        / f"{prior_label}_lam-{lam}_{key}_r{r_idx}.npz"
                    )
                    np.savez_compressed(
                        out_npz, frames=fr,
                        success=bool(result["final_success"]),
                        reward=float(result["final_reward"]),
                        demo=str(key), rollout_idx=int(r_idx),
                        lam=float(lam), prior=str(prior_label),
                    )
                    print(f"[exp15] captured {fr.shape[0]} frames "
                          f"-> {out_npz.name}")
                    _cur_frames.clear()

        # ---- Phase-C: DP-paper-style fresh-seed test bucket ---------------
        # For each test rollout, seed env.placement_initializer.rng with
        # ``test_start_seed + t_idx`` and call env.reset() to materialise
        # the seeded init.  Capture the resulting sim state and pass it to
        # unified_plan_and_execute as init_state -- the second env.reset()
        # inside that function re-randomises placement, but the immediate
        # set_sim_state restores the captured state, so the can ends up
        # exactly where this seed placed it.  Mirrors standalone_dp.py's
        # --n-test-rollouts protocol; required for the Phase-C K-sweep to
        # be apples-to-apples vs Phase-B's standalone 80 % test rate.
        for t_idx in range(int(args.n_test_rollouts)):
            test_seed = int(args.test_start_seed) + t_idx
            rng_pi = np.random.default_rng(test_seed)
            pi = getattr(env, "placement_initializer", None)
            if pi is not None:
                pi.rng = rng_pi
                child = getattr(pi, "samplers", None)
                if isinstance(child, dict):
                    for s in child.values():
                        s.rng = rng_pi
            np.random.seed(test_seed)
            env.reset()
            test_init_state = np.asarray(env.sim.get_state().flatten())

            rng_lam = np.random.default_rng(
                np.uint32(hash(
                    ("lam_test", base_seed, test_seed)
                ) & 0xFFFFFFFF)
            )

            warmstart_seed_fn = None
            if args.deterministic_prior_warmstart and prior_label in (
                "flow", "flow_v3", "flow_pull", "flow_v3_pull",
            ):
                def warmstart_seed_fn(
                    chunk_idx,
                    b=base_seed, _s=int(test_seed),
                ):
                    return int(np.uint32(hash((
                        "flow_warmstart_test", b, _s, int(chunk_idx),
                    )) & 0xFFFFFFFF))

            _p_for_reset = getattr(
                initial_mean_fn, "_prior_for_buffer_reset", None
            )
            if _p_for_reset is not None and hasattr(
                _p_for_reset, "reset_obs_buffer"
            ):
                _p_for_reset.reset_obs_buffer()

            t0 = time.perf_counter()
            result = unified_plan_and_execute(
                env, test_init_state,
                prior_funcs=(initial_mean_fn, score_fn_factory),
                lambda_prior=float(lam),
                params=params,
                rng=rng_lam,
                n_replans=args.mpc_replans,
                chunk_steps=chunk_steps,
                action_low=low_f32,
                action_high=high_f32,
                no_warm_start=not args.warm_start,
                n_workers=args.n_workers,
                dataset_path=dataset,
                make_rollout_fn=active_make_rollout_fn,
                cost_fn=active_cost_fn,
                warmstart_seed_fn=warmstart_seed_fn,
                obs_keys=prior_obs_keys,
                prior_lambda_schedule=lambda_schedule_arr,
                action_history_len=int(args.action_history_len),
                save_executed_actions=bool(args.save_executed_actions),
                max_env_steps=(
                    int(args.max_env_steps)
                    if args.max_env_steps > 0
                    else None
                ),
                warmstart_noise_sigma=float(args.warmstart_noise_sigma),
                combine_mode=args.combine_mode,
                temper_p=float(args.temper_p),
                sigma_demo=float(args.sigma_demo),
                demo_sigma_mode=args.demo_sigma_mode,
                demo_sigma_adapt_beta=float(args.demo_sigma_adapt_beta),
                prior_sigma_mode=args.sigma_schedule,
                log_gate_diagnostic=bool(args.log_gate_diagnostic),
                save_step_trace=bool(args.save_step_trace),
            )
            elapsed = time.perf_counter() - t0

            per_rollout.append({
                "demo": f"test_seed_{test_seed}",
                "rollout_idx": int(t_idx),
                "rollout_kind": "test",
                "init_seed": int(test_seed),
                "lambda_prior": float(lam),
                "final_reward": result["final_reward"],
                "success": bool(result["final_success"]),
                "plan_elapsed_s": float(result["plan_elapsed_s"]),
                "total_elapsed_s": float(elapsed),
                "chunks": result["chunks"],
            })
            print(
                f"[exp15] prior={prior_label} lam={lam:>5}  "
                f"R={args.mpc_replans}  K={args.K}  "
                f"{len(per_rollout):>3}  test seed={test_seed}  "
                f"reward={result['final_reward']:+.2f}  "
                f"success={result['final_success']}"
            )

        wall = time.perf_counter() - wall0
        # Train / test / overall stats.  When --n-test-rollouts=0 the
        # test bucket is empty and test_n=test_n_succ=0.
        train_rollouts = [r for r in per_rollout if r.get("rollout_kind") == "train"]
        test_rollouts  = [r for r in per_rollout if r.get("rollout_kind") == "test"]
        train_n = len(train_rollouts)
        train_n_succ = sum(int(r["success"]) for r in train_rollouts)
        test_n = len(test_rollouts)
        test_n_succ = sum(int(r["success"]) for r in test_rollouts)
        n = len(per_rollout)
        n_succ = train_n_succ + test_n_succ

        # Output filename: self-describing.
        out_name = (
            f"unified_{prior_label}_lam-{lam}"
            f"_K-{args.K}_R-{args.mpc_replans}"
            f"{args.output_name_suffix}.json"
        )
        out = args.output_dir / out_name
        out.write_text(json.dumps({
            "experiment": "15_unified_hybrid/run_can",
            "prior": prior_label,
            "prior_checkpoint": str(args.prior_checkpoint),
            "lambda_prior": float(lam),
            "warm_start": bool(args.warm_start),
            "mpc_replans": int(args.mpc_replans),
            "mpc_chunk_steps": int(chunk_steps),
            "deterministic_prior_warmstart": bool(args.deterministic_prior_warmstart),
            # Phase-C: train / test split.  Mirrors the standalone_dp.py
            # output schema so phase_c_<date>/ JSONs can be compared
            # apples-to-apples against the Phase-B per-rollout JSONs.
            # When --n-test-rollouts=0, the train_* fields equal n_total /
            # n_success / success_rate and the test_* fields are zero.
            "n_demos": int(args.n_demos),
            "n_rollouts": int(args.n_rollouts),
            "n_test_rollouts": int(args.n_test_rollouts),
            "test_start_seed": int(args.test_start_seed),
            "train_n": train_n,
            "train_n_succ": train_n_succ,
            "train_succ_rate": (train_n_succ / train_n) if train_n else 0.0,
            "test_n": test_n,
            "test_n_succ": test_n_succ,
            "test_succ_rate": (test_n_succ / test_n) if test_n else 0.0,
            "n_total": n,
            "n_success": n_succ,
            "success_rate": (n_succ / n) if n else 0.0,
            "params": {
                "H": args.H, "M": args.M, "K": args.K,
                "lambda_softmax": args.lambda_,
                "sigma_max": args.sigma_max,
                "sigma_min": args.sigma_min,
                "lambda_schedule": args.lambda_schedule,
                "score_mode": str(getattr(args, "score_mode", "learned")),
                "warmstart_noise_sigma": float(args.warmstart_noise_sigma),
                "retrieval_beta": float(args.retrieval_beta),
                "action_history_len": int(args.action_history_len),
                "contrastive_encoder_path": (
                    str(args.contrastive_encoder_path)
                    if args.contrastive_encoder_path is not None
                    else None
                ),
                "task": args.task,
                "prior_bank_size": int(args.prior_bank_size),
                "prior_bank_dataset": (
                    str(args.prior_bank_dataset)
                    if args.prior_bank_dataset is not None
                    else None
                ),
                "prior_bank_task": args.prior_bank_task,
                "goal_perturbation_xyz": list(goal_xyz),
                "mass_scale": float(args.mass_scale),
                "friction_scale": float(args.friction_scale),
                "combine_mode": str(args.combine_mode),
                "temper_p": float(args.temper_p),
                "sigma_demo": float(args.sigma_demo),
                "demo_sigma_mode": str(args.demo_sigma_mode),
                "demo_sigma_adapt_beta": float(args.demo_sigma_adapt_beta),
                "sigma_schedule": str(args.sigma_schedule),
                "log_gate_diagnostic": bool(args.log_gate_diagnostic),
                "save_step_trace": bool(args.save_step_trace),
                "raw_sparse_reward": bool(args.raw_sparse_reward),
                "demo_phase_reward": bool(args.demo_phase_reward),
                "demo_phase_scale": float(args.demo_phase_scale),
            },
            "seed": int(args.seed),
            "wall_s": wall,
            "per_rollout": per_rollout,
        }, indent=2))
        summary_per_lambda.append({
            "lambda_prior": float(lam),
            "n_total": n,
            "n_success": n_succ,
            "success_rate": (n_succ / n) if n else 0.0,
            "wall_s": wall,
            "output": str(out),
        })
        print(
            f"[exp15] wrote {out}  succ={n_succ}/{n} "
            f"= {100 * n_succ / max(n, 1):.1f}%"
        )

    # Cross-lambda summary.
    summary_path = args.output_dir / (
        f"unified_summary_{prior_label}"
        f"_K-{args.K}_R-{args.mpc_replans}"
        f"{args.output_name_suffix}.json"
    )
    summary_path.write_text(json.dumps({
        "experiment": "15_unified_hybrid/run_can",
        "prior": prior_label,
        "K": int(args.K),
        "R": int(args.mpc_replans),
        "per_lambda": summary_per_lambda,
    }, indent=2))
    print(f"[exp15] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
