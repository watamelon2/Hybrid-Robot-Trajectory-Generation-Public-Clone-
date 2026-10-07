"""Lambda sweep -- the per-step composite score (Tier 1b).

Wires the new ``prior_score_fn`` hook of ``mbd.denoise_loop`` to
either:

  - the flow-matching prior trained in exp 08 (``--prior flow``),
  - or the BC policy from exp 03 reinterpreted as a deterministic
    velocity field ``v_BC(x, t, s0) = a_BC(s0) - x_t/t`` after one
    closed-loop rollout (``--prior bc``); useful as a smoke wiring
    test before the flow-matching prior finishes training.

Endpoints intentionally recover earlier rows in the table:

  - ``--prior bc --warm-start --lambda-prior 0`` ~ exp 05's headline
    (BC warm-start, planner-only update).
  - ``--no-warm-start --lambda-prior 0`` ~ exp 04's no-prior
    baseline.
  - ``--lambda-prior inf`` is "ignore the planner, follow the prior";
    should match a deterministic rollout of the prior.

The actual lambda grid recommended in `docs/plans.md`:

    lambda_prior in {0, 0.1, 0.5, 1.0, 5.0, inf}

Each value writes its own JSON under
``results/lambda_<value>.json`` so the post-hoc plotter can iterate
files trivially.

Usage (smoke):

    python experiments/07_lambda_sweep/run_can.py --smoke

Full sweep on can/ph (20 x 3 per lambda; ~6 lambda values * ~80 min):

    python experiments/07_lambda_sweep/run_can.py \
        --prior flow \
        --prior-checkpoint experiments/08_flow_matching_prior/results/v_theta.pt \
        --n-demos 20 --n-rollouts 3 --seed 0 --no-live-view
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


_THIS_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _THIS_DIR.parent
_EXP_03 = _EXPERIMENTS_DIR / "03_bc_baseline"
_EXP_04 = _EXPERIMENTS_DIR / "04_mbd_baseline"
_EXP_05 = _EXPERIMENTS_DIR / "05_hybrid_method"
_EXP_08 = _EXPERIMENTS_DIR / "08_flow_matching_prior"

for _p in (_EXPERIMENTS_DIR, _EXP_03, _EXP_04, _EXP_05, _EXP_08, _THIS_DIR):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from common import (  # noqa: E402
    check_success,
    default_dataset_path,
    get_action_bounds,
    make_env_for_dataset,
    set_sim_state,
)
from mbd import (  # noqa: E402
    MBDParams,
    cosine_sigma_schedule,
    denoise_loop,
)


# Load exp 04's run_can by file path (basename collision prevention).
_spec = _importlib_util.spec_from_file_location(
    "exp04_run_can", _EXP_04 / "run_can.py"
)
exp04_run = _importlib_util.module_from_spec(_spec)
sys.modules["exp04_run_can"] = exp04_run
_spec.loader.exec_module(exp04_run)

_spec5 = _importlib_util.spec_from_file_location(
    "exp05_run_can", _EXP_05 / "run_can.py"
)
exp05_run = _importlib_util.module_from_spec(_spec5)
sys.modules["exp05_run_can"] = exp05_run
_spec5.loader.exec_module(exp05_run)

DEFAULT_LAMBDAS = (0.0, 0.1, 0.5, 1.0, 5.0, float("inf"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", type=str, default="can")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--prior", choices=("flow", "bc"), default="flow")
    p.add_argument(
        "--prior-checkpoint", type=Path,
        default=_EXP_08 / "results" / "v_theta.pt",
        help="For --prior flow: v_theta.pt from exp 08. "
             "For --prior bc: policy.pt from exp 03.",
    )
    p.add_argument(
        "--bc-checkpoint", type=Path,
        default=_EXP_03 / "results" / "policy.pt",
    )
    p.add_argument(
        "--lambda-prior", type=float, nargs="+", default=None,
        help=f"Lambda values to sweep. Default: {DEFAULT_LAMBDAS}.",
    )
    p.add_argument(
        "--bridge-formula", choices=("linear", "quadratic"),
        default="linear",
        help="Score-bridge denominator for the flow prior: "
             "'linear' (1-t) reproduces all 2026-05-16 results; "
             "'quadratic' (1-t)^2 is the theoretical bridge under "
             "linear-interp + N(0,I) noising (Phase 7 contingency). "
             "Ignored when --prior bc.",
    )
    p.add_argument("--warm-start", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--n-demos", type=int, default=5)
    p.add_argument("--n-rollouts", type=int, default=1)
    p.add_argument("--H", type=int, default=120)
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--lambda_", type=float, default=0.5,
                   help="MBD softmax temperature (separate from "
                        "--lambda-prior, which is the prior weight).")
    p.add_argument("--sigma-max", type=float, default=0.25)
    p.add_argument("--sigma-min", type=float, default=0.05)
    p.add_argument("--n-workers", type=int, default=1,
                   help="Number of parallel worker processes for MBD "
                        "rollouts.  Default 1 (sequential).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output-dir", type=Path,
        default=_THIS_DIR / "results",
    )
    p.add_argument("--live-view", action=argparse.BooleanOptionalAction,
                   default=False)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--smoke", action="store_true",
                   help="2 demos, 1 rollout, K=4, lambdas={0,1}.")
    return p.parse_args()


def _load_prior(args, action_dim: int):
    """Return ``(initial_mean_fn, score_fn_factory, label)``.

    initial_mean_fn(env, init_state, s0) -> np.ndarray (H, action_dim)
    score_fn_factory(s0, K) -> callable(mu, k) -> score
    """
    if args.prior == "flow":
        from adapter import FlowPrior  # noqa: F401  -- exp 08

        prior = FlowPrior.from_checkpoint(
            args.prior_checkpoint, device=args.device
        )

        def initial_mean_fn(env, init_state, s0):
            del env, init_state
            return prior.sample_initial_mean(s0, n_steps=32)

        bridge = getattr(args, "bridge_formula", "linear")

        def score_fn_factory(s0, K):
            return prior.make_prior_score_fn(s0, K, bridge=bridge)

        label = f"flow[{bridge}]"
        return initial_mean_fn, score_fn_factory, label

    # BC fallback: warm-start = closed-loop BC rollout (same as exp 05);
    # score = (a_BC_t - mu_t) / max(sigma, eps), i.e. "pull mu toward
    # the BC actions". Crude but exercises the wiring.
    from policy import Policy  # noqa: F401  -- exp 03

    policy = Policy.from_checkpoint(args.bc_checkpoint, device=args.device)

    def initial_mean_fn(env, init_state, _s0):
        from common import get_action_bounds as _gab

        low, high = _gab(env)
        actions, _info = exp05_run.bc_warmstart_actions(
            env, init_state, policy,
            args.H, low.astype(np.float32), high.astype(np.float32),
        )
        return actions

    def score_fn_factory(s0, K):
        del s0
        # We need the BC trajectory to define the score, but at score
        # time we don't have the env handy. Closure that the caller
        # populates via attribute assignment before the denoise loop:
        state = {"a_bc": None}

        def score_fn(mu, k):
            a_bc = state["a_bc"]
            if a_bc is None:
                return np.zeros_like(mu, dtype=np.float32)
            return (a_bc - mu).astype(np.float32)

        score_fn.state = state  # type: ignore[attr-defined]
        return score_fn

    label = "bc"
    return initial_mean_fn, score_fn_factory, label


def _make_obs_vec_for_prior(env) -> np.ndarray:
    """Return the live env's flat obs in exp 03/08's OBS_KEYS order."""
    from dataset import OBS_KEYS

    obs_dict = env._get_observations(force_update=True)
    parts = []
    for k in OBS_KEYS:
        if k not in obs_dict:
            raise KeyError(
                f"runtime obs missing {k!r} (available: "
                f"{sorted(obs_dict.keys())})"
            )
        parts.append(np.asarray(obs_dict[k], dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=-1)


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.n_demos = 2
        args.n_rollouts = 1
        args.K = 4
        args.lambda_prior = [0.0, 1.0]

    lambdas = tuple(args.lambda_prior) if args.lambda_prior else DEFAULT_LAMBDAS
    rng = np.random.default_rng(args.seed)

    dataset = args.dataset or default_dataset_path(args.task)
    env, env_meta = make_env_for_dataset(dataset)
    if hasattr(env, "reward_shaping"):
        env.reward_shaping = True
    low, high = get_action_bounds(env)
    action_dim = int(low.shape[0])
    low_f32 = low.astype(np.float32)
    high_f32 = high.astype(np.float32)

    initial_mean_fn, score_fn_factory, prior_label = _load_prior(
        args, action_dim
    )
    print(f"[exp07] prior={prior_label}  lambdas={lambdas}")

    schedule = cosine_sigma_schedule(
        args.K, sigma_max=args.sigma_max, sigma_min=args.sigma_min
    )
    params = MBDParams(
        H=args.H, M=args.M, K=args.K, action_dim=action_dim,
        lambda_=args.lambda_, sigma_schedule=schedule,
        clip_action=(low_f32, high_f32),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(dataset), "r") as h5:
        demo_keys = sorted(
            h5["data"].keys(), key=lambda k: int(k.split("_")[1])
        )
        if args.n_demos > 0:
            demo_keys = demo_keys[: args.n_demos]
        init_states = {
            k: np.asarray(h5[f"data/{k}/states"][0]) for k in demo_keys
        }

    # One "rollout RNG" per (demo, rollout_idx, lambda) so different
    # lambdas see the same perturbation noise -- a paired comparison.
    base_seed = int(args.seed)

    summary_per_lambda: list[dict] = []
    for lam in lambdas:
        per_rollout: list[dict] = []
        wall0 = time.perf_counter()
        for i, key in enumerate(demo_keys):
            init_state = init_states[key]
            for r_idx in range(args.n_rollouts):
                # Same seed across lambdas at fixed (demo, rollout):
                rng_lam = np.random.default_rng(
                    np.uint32(hash(("lam", base_seed, key, r_idx)) & 0xFFFFFFFF)
                )

                # Build initial mean and prior context.
                env.reset()
                set_sim_state(env, init_state)
                s0_vec = _make_obs_vec_for_prior(env)
                if args.warm_start:
                    initial_mean = initial_mean_fn(env, init_state, s0_vec)
                else:
                    initial_mean = np.zeros(
                        (args.H, action_dim), dtype=np.float32
                    )
                initial_mean = np.clip(
                    initial_mean, low_f32, high_f32
                ).astype(np.float32)

                score_fn = score_fn_factory(s0_vec, args.K)
                if hasattr(score_fn, "state"):
                    score_fn.state["a_bc"] = initial_mean  # type: ignore[attr-defined]

                rollout_fn = exp04_run.make_rollout_fn(
                    env, init_state, n_workers=args.n_workers,
                    dataset_path=dataset,
                )
                t0 = time.perf_counter()
                mu_final, log = denoise_loop(
                    initial_mean, rollout_fn, exp04_run.cost_fn,
                    params, rng_lam,
                    prior_score_fn=score_fn,
                    prior_lambda=float(lam),
                )
                rollout_fn.shutdown()
                plan_elapsed = time.perf_counter() - t0

                # Execute mu_final.
                env.reset(); set_sim_state(env, init_state)
                total_reward = 0.0
                for action in mu_final:
                    _obs, reward, _done, _info = env.step(action)
                    total_reward += float(reward)
                success = bool(check_success(env))

                per_rollout.append({
                    "demo": key,
                    "rollout_idx": int(r_idx),
                    "lambda_prior": float(lam),
                    "init_best_cost": float(log["per_iter"][0]["best_cost"]),
                    "final_best_cost": float(log["per_iter"][-1]["best_cost"]),
                    "final_exec_reward": float(total_reward),
                    "success": success,
                    "plan_elapsed_s": float(plan_elapsed),
                    "iter_curve": log["per_iter"],
                })
                print(
                    f"[exp07] lam={lam:>5}  {len(per_rollout):>3}  "
                    f"{key}#{r_idx}  exec={total_reward:+.2f}  "
                    f"success={success}"
                )
        wall = time.perf_counter() - wall0
        n = len(per_rollout)
        n_succ = sum(int(r["success"]) for r in per_rollout)
        # Filename disambiguates by prior so BC and flow runs don't
        # collide on disk. Flow keeps the historical `lambda_{lam}.json`
        # naming for backwards compatibility with existing fetch/analysis
        # scripts; BC uses `lambda_bc_{lam}.json` to match the names the
        # GCE pipeline already produced for the two finished cells.
        out_name = (
            f"lambda_{lam}.json"
            if prior_label == "flow"
            else f"lambda_{prior_label}_{lam}.json"
        )
        out = args.output_dir / out_name
        out.write_text(json.dumps({
            "experiment": "07_lambda_sweep/run_can",
            "prior": prior_label,
            "lambda_prior": float(lam),
            "warm_start": bool(args.warm_start),
            "n_total": n,
            "n_success": n_succ,
            "success_rate": (n_succ / n) if n else 0.0,
            "params": {
                "H": args.H, "M": args.M, "K": args.K,
                "lambda_softmax": args.lambda_,
                "sigma_max": args.sigma_max, "sigma_min": args.sigma_min,
            },
            "wall_s": float(wall),
            "per_rollout": per_rollout,
        }, indent=2))
        summary_per_lambda.append({
            "lambda_prior": float(lam),
            "n_success": int(n_succ),
            "n_total": int(n),
            "success_rate": (n_succ / n) if n else 0.0,
            "wall_s": float(wall),
            "output": str(out),
        })
        print(f"[exp07] wrote {out}  succ={n_succ}/{n}")

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_per_lambda, indent=2))
    print(f"[exp07] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
