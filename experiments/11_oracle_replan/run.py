"""Oracle-replan ablation: only replan once, only at strong-reach states.

Question motivated by Exp 05b: is the MPC win "the second plan
gets a near-grasp env state to refine against" (mechanism claim)
or "two independent draws of the perturbation lottery"?

This script answers it cleanly: we replay every Exp-05 single-shot
rollout that landed in the `strong_reach` bucket from `t=0` up to
`t=H/2`, then *only there* run a fresh BC warm-start + MBD
denoising loop and execute the resulting `mu_final`. Rollouts not
in the strong-reach bucket are skipped (their first chunk would
have nothing to refine).

Compare:

  - Exp 05 single-shot success rate inside strong_reach (3.1 %).
  - Exp 05b MPC success rate inside strong_reach (14.8 %).
  - This script's success rate -- isolates the "second plan from a
    near-grasp state" effect.

If our number is close to 14.8 %, the mechanism claim holds; if
it's closer to 3.1 %, the MPC win was mostly variance reduction.

Implementation: we don't store env state in the existing JSON, so
we replay deterministically. With ``--deterministic-bc-reset`` the
chunk-0 BC trajectory and chunk-0 execution are bit-identical to
the original Exp-05 single-shot run, so reaching `t=H/2` lands at
the same env state every time.

Usage:

    python experiments/11_oracle_replan/run.py \
        --source experiments/05_hybrid_method/results/_archive/can_ph.json \
        --n-rollouts 3 --seed 0
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
_EXP_06 = _EXPERIMENTS_DIR / "06_failure_taxonomy"

for _p in (_EXPERIMENTS_DIR, _EXP_03, _EXP_04, _EXP_05, _EXP_06, _THIS_DIR):
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
from run import classify  # noqa: E402  -- exp 06

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

from policy import Policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--source", type=Path,
        default=_EXP_05 / "results" / "can_ph.json",
        help="Existing single-shot Exp 05 results JSON. Strong-reach "
             "rollouts are picked from here.",
    )
    p.add_argument("--task", type=str, default="can")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument(
        "--bc-checkpoint", type=Path,
        default=_EXP_03 / "results" / "policy.pt",
    )
    p.add_argument("--H", type=int, default=120)
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--lambda_", type=float, default=0.5)
    p.add_argument("--sigma-max", type=float, default=0.25)
    p.add_argument("--sigma-min", type=float, default=0.05)
    p.add_argument(
        "--chunk-steps", type=int, default=60,
        help="Where to perform the oracle replan (env step index).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-rollouts", type=int, default=3)
    p.add_argument(
        "--output", type=Path,
        default=_THIS_DIR / "results" / "oracle.json",
    )
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    source = json.loads(args.source.read_text())
    rollouts = source.get("per_rollout", [])
    strong = [
        r for r in rollouts
        if classify(
            float(r.get("final_exec_reward", 0.0)),
            bool(r.get("success", False)),
        ) == "strong_reach"
    ]
    print(f"[exp11] strong-reach rollouts in source: {len(strong)}/{len(rollouts)}")
    if not strong:
        raise SystemExit("no strong-reach rollouts in source; nothing to do")

    dataset = args.dataset or default_dataset_path(args.task)
    env, _meta = make_env_for_dataset(dataset)
    if hasattr(env, "reward_shaping"):
        env.reward_shaping = True
    low, high = get_action_bounds(env)
    action_dim = int(low.shape[0])
    low_f32 = low.astype(np.float32)
    high_f32 = high.astype(np.float32)

    policy = Policy.from_checkpoint(args.bc_checkpoint, device=args.device)

    schedule = cosine_sigma_schedule(
        args.K, sigma_max=args.sigma_max, sigma_min=args.sigma_min
    )
    params = MBDParams(
        H=args.H, M=args.M, K=args.K, action_dim=action_dim,
        lambda_=args.lambda_, sigma_schedule=schedule,
        clip_action=(low_f32, high_f32),
    )

    # Look up init_state per demo from the dataset.
    with h5py.File(str(dataset), "r") as h5:
        init_states = {
            k: np.asarray(h5[f"data/{k}/states"][0])
            for k in {r["demo"] for r in strong}
        }

    out_rollouts: list[dict] = []
    wall0 = time.perf_counter()
    for idx, src_r in enumerate(strong):
        key = src_r["demo"]
        r_idx = int(src_r.get("rollout_idx", 0))
        init_state = init_states[key]

        # Stage 1: replay first chunk_steps actions deterministically.
        # We re-do the BC warm-start with deterministic reset so chunk 0
        # is byte-identical to the source rollout.
        # Mask the hash to a uint32 BEFORE casting so this works on
        # numpy 2.x where np.uint32() of a negative int raises
        # OverflowError. Mirrors the fix applied to exp05/run_can.py.
        seed = int(np.uint32(hash((int(args.seed), key, int(r_idx))) & 0xFFFFFFFF))
        a_bc, _bc_info = exp05_run.bc_warmstart_actions(
            env, init_state, policy, args.H, low_f32, high_f32,
            reset_seed=seed,
        )
        # Run MBD chunk 0 to get the original mu_final, then execute
        # only the first chunk_steps of it. Match Exp 05b's MPC chunk-0
        # behaviour exactly.
        rollout_fn = exp04_run.make_rollout_fn(env, init_state)
        c0_rng = np.random.default_rng(
            np.uint32(hash(("chunk0", seed)) & 0xFFFFFFFF)
        )
        mu0, _ = denoise_loop(a_bc, rollout_fn, exp04_run.cost_fn, params, c0_rng)

        # Stage 2: snapshot env state at t=chunk_steps after executing mu0.
        env.reset(); set_sim_state(env, init_state)
        for t in range(args.chunk_steps):
            env.step(mu0[t])
        snapshot = np.asarray(env.sim.get_state().flatten())

        # Stage 3: oracle replan. BC + MBD from snapshot for full H steps.
        a_bc1, _ = exp05_run.bc_warmstart_actions(
            env, snapshot, policy, args.H, low_f32, high_f32,
            reset_seed=seed + 1,
        )
        rollout_fn1 = exp04_run.make_rollout_fn(env, snapshot)
        c1_rng = np.random.default_rng(
            np.uint32(hash(("chunk1", seed)) & 0xFFFFFFFF)
        )
        mu1, log1 = denoise_loop(a_bc1, rollout_fn1, exp04_run.cost_fn, params, c1_rng)

        # Stage 4: execute mu1 from snapshot; report.
        env.reset(); set_sim_state(env, snapshot)
        # Mirror Exp 05b: execute remaining (H - chunk_steps) of mu1,
        # so the total executed steps stay at H.
        steps_remaining = args.H - args.chunk_steps
        total_reward = 0.0
        for t in range(steps_remaining):
            _o, r, _d, _i = env.step(mu1[t])
            total_reward += float(r)
        success = bool(check_success(env))

        out_rollouts.append({
            "demo": key,
            "rollout_idx": r_idx,
            "source_exec_reward": float(src_r.get("final_exec_reward", 0.0)),
            "source_success": bool(src_r.get("success", False)),
            "oracle_chunk1_reward": float(total_reward),
            "oracle_success": success,
            "init_best_cost": float(log1["per_iter"][0]["best_cost"]),
            "final_best_cost": float(log1["per_iter"][-1]["best_cost"]),
        })
        print(
            f"[exp11] {idx + 1:>3}/{len(strong)}  {key}#{r_idx}  "
            f"src_rew={src_r.get('final_exec_reward', 0.0):+.2f}  "
            f"oracle_rew={total_reward:+.2f}  "
            f"oracle_success={success}"
        )

    n = len(out_rollouts)
    n_succ = sum(int(r["oracle_success"]) for r in out_rollouts)
    summary = {
        "experiment": "11_oracle_replan",
        "source": str(args.source),
        "n_strong_reach_in_source": len(strong),
        "n_total_in_source": len(rollouts),
        "n_oracle_attempted": n,
        "n_oracle_success": n_succ,
        "oracle_success_share": (n_succ / n) if n else 0.0,
        "wall_s": time.perf_counter() - wall0,
        "params": {
            "H": args.H, "M": args.M, "K": args.K,
            "lambda": args.lambda_,
            "sigma_max": args.sigma_max, "sigma_min": args.sigma_min,
            "chunk_steps": args.chunk_steps,
        },
        "per_rollout": out_rollouts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(
        f"[exp11] strong-reach -> success conversion (oracle replan): "
        f"{n_succ}/{n} = {summary['oracle_success_share']:.1%}"
    )
    print(f"[exp11] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
