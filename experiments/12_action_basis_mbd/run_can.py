"""Experiment 12 -- DCT-basis action perturbations for MBD on PickPlaceCan.

This is the ablation hypothesised in the Exp 05 README: the planner's
``M=16`` cosine-noise samples each live in an ``H * action_dim``-dim
hyperball, the vast majority of which is high-frequency jitter that
the controller filters out. By projecting the perturbations onto a
length-``n_modes`` DCT-II basis (``basis.dct_basis``) we keep the
same per-iteration budget but force every sample to be a coherent
multi-step shift -- the kind of motion that actually changes where
the can ends up.

This script is a thin wrapper around the Exp 05 single-shot
configuration (same env, same BC warm-start, same cost, same
schedule); the only swap is ``basis.denoise_loop_basis`` in place of
``mbd.denoise_loop``. Pair with exp 07's lambda sweep to ask whether
basis perturbations *and* a flow prior compose. Output schema
matches exp 05's ``can_ph.json`` so ``experiments/06_failure_taxonomy/
run.py`` consumes the result with no changes.

CLI mirrors exp 05's headline runner. New flags:

  --n-modes N         Number of DCT modes for the perturbation basis.
                      4 / 8 / 16 / 32 / H are the suggested sweep
                      values. ``H == n_modes`` recovers full-rank
                      perturbations (a slow check that
                      denoise_loop_basis is rank-preserving).
  --no-warm-start     Same as exp 05: ablate the BC warm-start.

Usage:

    # Wiring smoke (~2 minutes; 2 demos, K=4, basis with 4 modes):
    python experiments/12_action_basis_mbd/run_can.py --smoke --no-live-view

    # Single value of n_modes at exp 05's headline protocol:
    python experiments/12_action_basis_mbd/run_can.py \\
        --n-demos 20 --n-rollouts 3 --seed 0 --no-live-view \\
        --n-modes 8 \\
        --output experiments/12_action_basis_mbd/results/can_basis_m8.json

The suggested n_modes sweep ({4, 8, 16, 32, 120}) is left to a small
bash wrapper alongside this file; the README documents the protocol.
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

# Same sys.path layout as exp 05's run_can.py: the experiments/
# shared dir + each sibling experiment + this dir. Order matters so
# this script's own ``run_can`` doesn't shadow exp 04/05's.
for _p in (_EXPERIMENTS_DIR, _EXP_03, _EXP_04, _EXP_05, _THIS_DIR):
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
from mbd import MBDParams, cosine_sigma_schedule  # noqa: E402

# basis.py lives in this directory.
from basis import denoise_loop_basis  # noqa: E402

# Import exp 04 / exp 05 run_can.py explicitly by file path so the
# basename collision (every experiment names its driver run_can.py)
# does not confuse the import system.
_spec4 = _importlib_util.spec_from_file_location(
    "exp04_run_can", _EXP_04 / "run_can.py"
)
exp04_run = _importlib_util.module_from_spec(_spec4)
sys.modules["exp04_run_can"] = exp04_run
_spec4.loader.exec_module(exp04_run)

_spec5 = _importlib_util.spec_from_file_location(
    "exp05_run_can", _EXP_05 / "run_can.py"
)
exp05_run = _importlib_util.module_from_spec(_spec5)
sys.modules["exp05_run_can"] = exp05_run
_spec5.loader.exec_module(exp05_run)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--task", type=str, default="can")
    p.add_argument("--n-modes", type=int, default=8,
                   help="DCT basis size. 4, 8, 16, 32 are the typical "
                        "sweep points; n_modes == H is full-rank.")
    p.add_argument("--bc-checkpoint", type=Path,
                   default=_EXP_03 / "results" / "policy.pt")
    p.add_argument("--no-warm-start", action="store_true",
                   help="Ablation: ignore BC, start MBD from zeros.")
    p.add_argument("--n-demos", type=int, default=5)
    p.add_argument("--n-rollouts", type=int, default=1)
    p.add_argument("--H", type=int, default=120)
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--lambda_", type=float, default=0.5,
                   help="MBD softmax temperature.")
    p.add_argument("--sigma-max", type=float, default=0.25)
    p.add_argument("--sigma-min", type=float, default=0.05)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path,
                   default=_THIS_DIR / "results" / "can_basis.json")
    p.add_argument("--video-dir", type=str,
                   default="",
                   help="Where to save MP4s. Empty = disable (default).")
    p.add_argument(
        "--live-view", action=argparse.BooleanOptionalAction, default=False,
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--deterministic-bc-reset",
        action=argparse.BooleanOptionalAction, default=True,
        help="Re-seed the env before each BC warm-start. Default ON "
             "(mirrors exp 14's seed-sweep configuration).",
    )
    p.add_argument("--smoke", action="store_true",
                   help="Tiny config: 2 demos, 1 rollout, K=4, n_modes=4.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.n_demos = 2
        args.n_rollouts = 1
        args.K = 4
        args.n_modes = 4

    rng = np.random.default_rng(args.seed)

    dataset = args.dataset or default_dataset_path(args.task)
    if not dataset.exists():
        raise SystemExit(
            f"Dataset not found: {dataset}\n"
            "Run: python -m robomimic.scripts.download_datasets "
            f"--tasks {args.task} --dataset_types ph"
        )

    if not args.no_warm_start and not args.bc_checkpoint.exists():
        raise SystemExit(
            f"BC checkpoint not found: {args.bc_checkpoint}\n"
            "Train it first: python experiments/03_bc_baseline/train.py"
        )

    env, env_meta = make_env_for_dataset(dataset)
    if hasattr(env, "reward_shaping"):
        env.reward_shaping = True
    low, high = get_action_bounds(env)
    action_dim = int(low.shape[0])
    low_f32 = low.astype(np.float32)
    high_f32 = high.astype(np.float32)

    # Validate the basis size before we burn any compute.
    if not (1 <= args.n_modes <= args.H):
        raise SystemExit(
            f"--n-modes must satisfy 1 <= n_modes <= H={args.H}; "
            f"got {args.n_modes}"
        )

    policy = None
    if not args.no_warm_start:
        from policy import Policy  # noqa: F401  -- exp 03

        policy = Policy.from_checkpoint(args.bc_checkpoint, device=args.device)

    schedule = cosine_sigma_schedule(
        args.K, sigma_max=args.sigma_max, sigma_min=args.sigma_min
    )
    params = MBDParams(
        H=args.H, M=args.M, K=args.K, action_dim=action_dim,
        lambda_=args.lambda_, sigma_schedule=schedule,
        clip_action=(low_f32, high_f32),
    )

    with h5py.File(str(dataset), "r") as h5:
        demo_keys = sorted(
            h5["data"].keys(), key=lambda k: int(k.split("_")[1])
        )
        if args.n_demos > 0:
            demo_keys = demo_keys[: args.n_demos]
        init_states = {
            k: np.asarray(h5[f"data/{k}/states"][0]) for k in demo_keys
        }

    print(
        f"[exp12] n_modes={args.n_modes}  H={args.H}  M={args.M}  K={args.K}  "
        f"warm_start={not args.no_warm_start}  "
        f"sigma=({args.sigma_min}, {args.sigma_max})"
    )

    per_rollout: list[dict] = []
    wall0 = time.perf_counter()
    n_total = len(demo_keys) * args.n_rollouts
    for i, key in enumerate(demo_keys):
        init_state = init_states[key]
        for r_idx in range(args.n_rollouts):
            child_rng = np.random.default_rng(
                rng.integers(0, 2**31 - 1)
            )

            # 1. Warm-start: BC closed-loop rollout (exp 05) or zeros.
            bc_info: dict = {}
            if args.no_warm_start or policy is None:
                initial_mean = np.zeros(
                    (args.H, action_dim), dtype=np.float32
                )
            else:
                bc_reset_seed = (
                    int(np.uint32(
                        hash((int(args.seed), key, int(r_idx))) & 0xFFFFFFFF
                    )) if args.deterministic_bc_reset else None
                )
                initial_mean, bc_info = exp05_run.bc_warmstart_actions(
                    env, init_state, policy,
                    args.H, low_f32, high_f32,
                    reset_seed=bc_reset_seed,
                )

            # 2. Build rollout closure for this init_state.
            rollout_fn = exp04_run.make_rollout_fn(
                env, init_state,
                n_workers=args.n_workers,
                dataset_path=dataset,
            )

            # 3. MBD with DCT-basis perturbations.
            plan_t0 = time.perf_counter()
            mu_final, log = denoise_loop_basis(
                initial_mean, rollout_fn, exp04_run.cost_fn,
                params, child_rng, n_modes=args.n_modes,
            )
            rollout_fn.shutdown()
            plan_elapsed = time.perf_counter() - plan_t0

            # 4. Execute mu_final on the live env.
            env.reset()
            set_sim_state(env, init_state)
            total_reward = 0.0
            exec_t0 = time.perf_counter()
            for action in mu_final:
                _obs, reward, _done, _info = env.step(action)
                total_reward += float(reward)
            exec_elapsed = time.perf_counter() - exec_t0
            success = bool(check_success(env))

            tag = f"{key}#{r_idx}" if args.n_rollouts > 1 else key
            init_cost = log["per_iter"][0]["best_cost"]
            final_cost = log["per_iter"][-1]["best_cost"]
            print(
                f"[exp12] {len(per_rollout) + 1:>3}/{n_total:<3}  "
                f"{tag:<15}  init={init_cost:>+7.2f}  "
                f"final={final_cost:>+7.2f}  exec={total_reward:>+7.2f}  "
                f"success={success!s:<5}  ({plan_elapsed:.1f}s plan)"
            )

            per_rollout.append({
                "demo": key,
                "rollout_idx": int(r_idx),
                "init_best_cost": float(init_cost),
                "final_best_cost": float(final_cost),
                "final_exec_reward": float(total_reward),
                "success": success,
                "plan_elapsed_s": float(plan_elapsed),
                "exec_elapsed_s": float(exec_elapsed),
                "warm_start": bool(not args.no_warm_start),
                "n_modes": int(args.n_modes),
                "bc_exec_reward": bc_info.get("bc_exec_reward"),
                "bc_success": bc_info.get("bc_success"),
                "bc_elapsed_s": bc_info.get("bc_elapsed_s"),
                "iter_curve": [
                    {
                        "iter": p["iter"],
                        "sigma": p["sigma"],
                        "best_cost": p["best_cost"],
                        "mean_cost": p["mean_cost"],
                        "ess": p["ess"],
                    }
                    for p in log["per_iter"]
                ],
            })

    total_wall = time.perf_counter() - wall0
    n = len(per_rollout)
    n_success = sum(int(r["success"]) for r in per_rollout)
    bc_success_total = sum(int(bool(r.get("bc_success"))) for r in per_rollout)

    summary = {
        "experiment": "12_action_basis_mbd/run_can",
        "dataset": str(dataset),
        "env_name": env_meta.get("env_name"),
        "env_type": env_meta.get("type"),
        "seed": int(args.seed),
        "n_demos_attempted": len(demo_keys),
        "n_rollouts_per_demo": int(args.n_rollouts),
        "n_total": n,
        "n_success": n_success,
        "success_rate": (n_success / n) if n else 0.0,
        "bc_success_total": int(bc_success_total),
        "bc_success_rate": (
            float(bc_success_total) / n if n and not args.no_warm_start else None
        ),
        "warm_start": bool(not args.no_warm_start),
        "bc_checkpoint": (
            None if args.no_warm_start else str(args.bc_checkpoint)
        ),
        "params": {
            "H": int(args.H),
            "M": int(args.M),
            "K": int(args.K),
            "lambda": float(args.lambda_),
            "sigma_max": float(args.sigma_max),
            "sigma_min": float(args.sigma_min),
            "action_dim": action_dim,
            "n_modes": int(args.n_modes),
        },
        "total_wall_s": float(total_wall),
        "per_rollout": per_rollout,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(
        f"[exp12] success rate: {n_success}/{n} = "
        f"{summary['success_rate']:.1%}   wall={total_wall:.1f}s   "
        f"wrote {args.output}"
    )

    # Cost-curve PNG matching exp 04/05's plotting convention.
    try:
        import matplotlib.pyplot as plt

        if per_rollout:
            curves = np.array(
                [
                    [p["best_cost"] for p in r["iter_curve"]]
                    for r in per_rollout
                ]
            )  # (n_rollouts, K)
            mean_curve = curves.mean(axis=0)
            fig, ax = plt.subplots(figsize=(7, 4))
            for c in curves:
                ax.plot(c, alpha=0.25, color="tab:purple")
            ax.plot(mean_curve, color="tab:purple", lw=2,
                    label="mean across rollouts")
            ax.set_xlabel("MBD denoising iteration $k$")
            ax.set_ylabel("best cost in batch ($-\\sum r_t$)")
            warm_str = "" if not args.no_warm_start else " (zero-init)"
            ax.set_title(
                f"Exp 12: DCT-basis MBD{warm_str}  "
                f"(n_modes={args.n_modes}, M={args.M}, K={args.K}, "
                f"H={args.H})"
            )
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            plot_path = args.output.with_suffix(".png")
            fig.savefig(plot_path, dpi=120)
            plt.close(fig)
            print(f"[exp12] cost-curve plot: {plot_path}")
    except ImportError:
        print("[exp12] matplotlib not available, skipping cost-curve plot")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
