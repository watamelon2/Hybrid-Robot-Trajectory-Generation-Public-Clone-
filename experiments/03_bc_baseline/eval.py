"""Closed-loop evaluation of a trained BC policy on the can/ph env.

Mirrors the protocol used by experiment 02 (random baseline) so the
three rows of the Week-4 results table are directly comparable:

  - 20 init-states drawn from the dataset's first 20 demos
  - 3 independent rollouts per init-state (gives n=60 for tighter
    confidence than n=20 on a stochastic policy; note BC itself is
    deterministic, but the env's contact dynamics + initialisation are
    not exactly idempotent when re-stepped, so we keep the multi-
    rollout structure for protocol compatibility)
  - rollout length = the demo's recorded horizon (so step counts match
    exps 01 / 02; see README "Implementation notes" for caveats)

At each step, the live robosuite obs dict is concatenated in the same
``OBS_KEYS`` order used during training, normalised with the saved
``obs_mean / obs_std``, fed to the MLP, and the tanh-squashed output is
clipped to ``env.action_spec`` before stepping.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

# Make sibling experiment scripts importable without dotted-package
# syntax (this directory's name starts with a digit).
_THIS_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _THIS_DIR.parent
for _p in (_THIS_DIR, _EXPERIMENTS_DIR):
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
from dataset import OBS_KEYS  # noqa: E402
from policy import Policy  # noqa: E402


# Live-env -> training-key remapping. With the post-bug-fix OBS_KEYS
# (Can_pos / Can_quat instead of the broken concatenated ``object``)
# every key in OBS_KEYS is present verbatim in robosuite 1.5's obs
# dict, so this remap is intentionally empty -- it stays here as the
# documented hook in case future obs-set tweaks need to translate
# names again.
_RUNTIME_KEY_REMAP: dict[str, str] = {}


def _get_runtime_obs(env_obs: dict) -> dict:
    """Translate the live env's obs dict into the dataset's key space."""
    out: dict = {}
    for k in OBS_KEYS:
        runtime_k = _RUNTIME_KEY_REMAP.get(k, k)
        if runtime_k not in env_obs:
            raise KeyError(
                f"runtime obs missing {runtime_k!r} (dataset key {k!r}); "
                f"available: {sorted(env_obs.keys())}"
            )
        out[k] = np.asarray(env_obs[runtime_k], dtype=np.float32)
    return out


def _concat_obs(obs: dict) -> np.ndarray:
    return np.concatenate([obs[k] for k in OBS_KEYS], axis=-1)


def _select_demos(
    dataset_path: Path, n_demos: int
) -> list[tuple[str, np.ndarray, int]]:
    """Return ``(demo_id, init_state, horizon)`` triples in numeric order."""
    out: list[tuple[str, np.ndarray, int]] = []
    with h5py.File(str(dataset_path), "r") as h5:
        demos = sorted(
            h5["data"].keys(), key=lambda k: int(k.split("_")[1])
        )
        if n_demos > 0:
            demos = demos[:n_demos]
        for demo_id in demos:
            d = h5["data"][demo_id]
            init = np.asarray(d["states"][0], dtype=np.float64)
            horizon = int(d["actions"].shape[0])
            out.append((demo_id, init, horizon))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=_THIS_DIR / "results" / "policy.pt",
        help="Path to a policy.pt produced by train.py.",
    )
    p.add_argument("--task", type=str, default="can")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument(
        "--n-demos",
        type=int,
        default=20,
        help="Number of init-states (demos) to evaluate (default: 20, "
        "matches exp 02). Use -1 for all 200.",
    )
    p.add_argument(
        "--n-rollouts",
        type=int,
        default=3,
        help="Independent rollouts per init-state (default: 3, "
        "matches exp 02).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_THIS_DIR / "results" / "can_ph.json",
        help="Where to write the per-rollout JSON.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset or default_dataset_path(args.task)

    print(f"[eval ] checkpoint = {args.checkpoint}")
    print(f"[eval ] dataset    = {dataset_path}")
    print(f"[eval ] device     = {args.device}")
    print(f"[eval ] n_demos    = {args.n_demos}")
    print(f"[eval ] n_rollouts = {args.n_rollouts}")
    print(f"[eval ] seed       = {args.seed}")

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Run train.py first."
        )
    policy = Policy.from_checkpoint(args.checkpoint, device=args.device)

    n_demos = args.n_demos if args.n_demos > 0 else 0
    demos = _select_demos(dataset_path, n_demos=n_demos or 200)
    print(f"[eval ] selected {len(demos)} demos")

    env, env_meta = make_env_for_dataset(dataset_path)
    action_low, action_high = get_action_bounds(env)
    print(
        f"[eval ] env_name = {env_meta['env_name']}, "
        f"action_dim = {action_low.shape[0]}"
    )

    per_rollout: list[dict] = []
    n_success = 0
    t_total = time.perf_counter()

    for demo_id, init_state, horizon in demos:
        for r in range(args.n_rollouts):
            env.reset()
            set_sim_state(env, init_state)
            obs_dict = env._get_observations(force_update=True)

            t_ep = time.perf_counter()
            last_done = False
            for _ in range(horizon):
                obs_named = _get_runtime_obs(obs_dict)
                obs_vec = _concat_obs(obs_named)
                action = policy.act(obs_vec)
                action = np.clip(action, action_low, action_high)
                obs_dict, _reward, last_done, _info = env.step(action)
            elapsed = time.perf_counter() - t_ep
            success = check_success(env)
            n_success += int(success)

            per_rollout.append(
                {
                    "demo": demo_id,
                    "rollout_idx": r,
                    "horizon": horizon,
                    "success": success,
                    "done": bool(last_done),
                    "elapsed_s": float(elapsed),
                    "steps_per_s": float(horizon / max(elapsed, 1e-9)),
                }
            )
            print(
                f"[eval ] {demo_id} r{r}  "
                f"H={horizon:3d}  success={success}  "
                f"({elapsed:.2f}s)"
            )

    total_wall = time.perf_counter() - t_total
    n_total = len(per_rollout)
    success_rate = n_success / max(n_total, 1)

    summary = {
        "experiment": "03_bc_baseline",
        "checkpoint": str(args.checkpoint),
        "dataset": str(dataset_path),
        "env_name": env_meta["env_name"],
        "env_type": env_meta.get("type", 1),
        "seed": int(args.seed),
        "n_demos_attempted": len(demos),
        "n_rollouts_per_demo": int(args.n_rollouts),
        "n_total": n_total,
        "n_success": int(n_success),
        "success_rate": float(success_rate),
        "action_low": action_low.tolist(),
        "action_high": action_high.tolist(),
        "obs_keys": list(OBS_KEYS),
        "total_wall_s": float(total_wall),
        "per_rollout": per_rollout,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    print()
    print(
        f"[eval ] {n_success}/{n_total} = "
        f"{success_rate:.3f} success on {env_meta['env_name']}"
    )
    print(f"[eval ] total wall {total_wall:.1f}s")
    print(f"[eval ] results -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
