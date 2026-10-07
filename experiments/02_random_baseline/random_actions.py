"""Experiment 02: Random-action baseline.

Counterpart to ``01_demo_replay``: instead of replaying the recorded
expert action sequence open-loop, we replay a **uniformly random**
action sequence of the same horizon, started from the same initial
state. This anchors the *lower bound* on success rate for every method
we compare in the Week-4 results table.

Concretely, for each demo we:

  1. Reconstruct the matching robosuite environment from the dataset's
     embedded ``env_args`` metadata.
  2. Reset the env to the demo's recorded initial state.
  3. Roll out an action sequence ``a_t ~ Uniform(low, high)`` for the
     same horizon as the demo, where ``(low, high) = env.action_spec``.
  4. Record whether the env reports task success at the final step.

We optionally repeat step 3 for ``--n-rollouts`` independent random
seeds per init-state, to get a lower-variance estimate. The default of
1 matches exp01's protocol.

Expected result: success rate ≈ 0 % on contact-rich manipulation
tasks like ``PickPlaceCan``. If it isn't, the task's success criterion
is too lax and we should pick a harder one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    default_dataset_path,
    get_action_bounds,
    make_env_for_dataset,
    run_episode,
)


def random_rollout(env, h5: h5py.File, key: str, rng: np.random.Generator) -> dict:
    """One uniformly-random rollout from a demo's initial state."""
    states: np.ndarray = h5[f"data/{key}/states"][...]
    horizon = int(h5[f"data/{key}/actions"].shape[0])

    low, high = get_action_bounds(env)
    actions = rng.uniform(low=low, high=high, size=(horizon, low.shape[0]))

    out = run_episode(env, init_state=states[0], actions=actions)
    out["demo"] = key
    return out


def run(
    dataset_path: Path,
    n_demos: int,
    n_rollouts: int,
    output_path: Path,
    seed: int = 0,
) -> dict:
    rng_master = np.random.default_rng(seed)

    env, env_meta = make_env_for_dataset(dataset_path)
    print(f"[exp02] env loaded: {env_meta.get('env_name', '?')}")
    print(f"[exp02] dataset:   {dataset_path}")
    low, high = get_action_bounds(env)
    print(
        f"[exp02] action_dim={low.shape[0]}, "
        f"low={np.round(low, 2).tolist()}, high={np.round(high, 2).tolist()}"
    )

    with h5py.File(str(dataset_path), "r") as h5:
        demo_keys = sorted(h5["data"].keys(), key=lambda k: int(k.split("_")[1]))
        if n_demos > 0:
            demo_keys = demo_keys[:n_demos]

        per_rollout: list[dict] = []
        wall0 = time.perf_counter()
        for i, key in enumerate(demo_keys):
            for r_idx in range(n_rollouts):
                # Independent stream per (demo, rollout) so the results
                # are reproducible even if we change the order.
                child = rng_master.spawn(1)[0]
                r = random_rollout(env, h5, key, rng=child)
                r["rollout_idx"] = r_idx
                per_rollout.append(r)
                tag = f"{key}#{r_idx}" if n_rollouts > 1 else key
                print(
                    f"[exp02] {len(per_rollout):>3}/"
                    f"{len(demo_keys) * n_rollouts:<3} {tag:<14} "
                    f"H={r['horizon']:<4} success={r['success']!s:<5} "
                    f"({r['elapsed_s']:.2f}s, {r['steps_per_s']:.0f} step/s)"
                )
        wall = time.perf_counter() - wall0

    n = len(per_rollout)
    n_success = sum(int(r["success"]) for r in per_rollout)
    summary = {
        "experiment": "02_random_baseline",
        "dataset": str(dataset_path),
        "env_name": env_meta.get("env_name"),
        "env_type": env_meta.get("type"),
        "seed": seed,
        "n_demos_attempted": len(demo_keys),
        "n_rollouts_per_demo": n_rollouts,
        "n_total": n,
        "n_success": n_success,
        "success_rate": n_success / n if n else 0.0,
        "action_low": low.tolist(),
        "action_high": high.tolist(),
        "total_wall_s": wall,
        "per_rollout": per_rollout,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[exp02] success rate: {n_success}/{n} = "
        f"{summary['success_rate']:.1%}   wrote {output_path}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--task", type=str, default="can")
    p.add_argument(
        "--n-demos",
        type=int,
        default=20,
        help=(
            "Number of distinct initial states (demos) to start from. "
            "Use -1 for all."
        ),
    )
    p.add_argument(
        "--n-rollouts",
        type=int,
        default=1,
        help=(
            "Independent random rollouts per init-state; total runs is "
            "n_demos * n_rollouts. Default 1 to mirror exp01's protocol."
        ),
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset or default_dataset_path(args.task)
    if not dataset.exists():
        raise SystemExit(
            f"Dataset not found: {dataset}\n"
            "Run: python -m robomimic.scripts.download_datasets "
            f"--tasks {args.task} --dataset_types ph"
        )
    output = (
        args.output
        or Path(__file__).resolve().parent / "results" / f"{args.task}_ph.json"
    )
    run(
        dataset_path=dataset,
        n_demos=args.n_demos,
        n_rollouts=args.n_rollouts,
        output_path=output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
