"""Experiment 01: Robomimic proficient-human demo replay sanity check.

For each demo in a Robomimic HDF5 file, this script:

  1. Reconstructs the matching robosuite environment from the dataset's
     embedded ``env_args`` metadata.
  2. Resets the env to the demo's recorded initial state.
  3. Replays the demo's recorded action sequence open-loop.
  4. Records whether the env reports task success at the final step.

The result is a per-demo success array and an aggregate success rate,
which is the project's *Method 0 / replay* baseline:

  - "How well does the recorded simulator agree with the action-state
    trajectory the dataset was collected on?"

If replay success is far below ~80 % on the same dataset, our env / sim
config is wrong; if it is near 100 %, we have a clean foundation to
compare learned policies against.

Usage
-----
    python experiments/01_demo_replay/replay.py \
        --dataset <path/to/low_dim_v141.hdf5> \
        --n-demos 5

Pass ``--n-demos -1`` to replay the entire dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

# Make ``experiments/common.py`` importable from this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402  (sys.path tweak above)
    default_dataset_path,
    make_env_for_dataset,
    run_episode,
)


def replay_demo(env, h5: h5py.File, key: str) -> dict:
    """Replay one demo and return a small result dict (with the demo key)."""
    actions: np.ndarray = h5[f"data/{key}/actions"][...]
    states: np.ndarray = h5[f"data/{key}/states"][...]
    out = run_episode(env, init_state=states[0], actions=actions)
    out["demo"] = key
    return out


def run(
    dataset_path: Path, n_demos: int, output_path: Path, seed: int = 0
) -> dict:
    """Run replay on the first ``n_demos`` demos and write a results JSON."""
    np.random.seed(seed)

    env, env_meta = make_env_for_dataset(dataset_path)
    print(f"[exp01] env loaded: {env_meta.get('env_name', '?')}")
    print(f"[exp01] dataset:   {dataset_path}")

    with h5py.File(str(dataset_path), "r") as h5:
        demo_keys = sorted(h5["data"].keys(), key=lambda k: int(k.split("_")[1]))
        if n_demos > 0:
            demo_keys = demo_keys[:n_demos]

        per_demo: list[dict] = []
        wall0 = time.perf_counter()
        for i, key in enumerate(demo_keys):
            r = replay_demo(env, h5, key)
            per_demo.append(r)
            print(
                f"[exp01] {i + 1:>3}/{len(demo_keys):<3} {key:<10} "
                f"H={r['horizon']:<4} success={r['success']!s:<5} "
                f"({r['elapsed_s']:.2f}s, {r['steps_per_s']:.0f} step/s)"
            )
        wall = time.perf_counter() - wall0

    n = len(per_demo)
    n_success = sum(int(r["success"]) for r in per_demo)
    summary = {
        "experiment": "01_demo_replay",
        "dataset": str(dataset_path),
        "env_name": env_meta.get("env_name"),
        "env_type": env_meta.get("type"),
        "n_demos_attempted": n,
        "n_success": n_success,
        "success_rate": n_success / n if n else 0.0,
        "total_wall_s": wall,
        "per_demo": per_demo,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[exp01] success rate: {n_success}/{n} = "
        f"{summary['success_rate']:.1%}   wrote {output_path}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Path to robomimic HDF5 dataset. Default: the can/ph "
            "low_dim_v141.hdf5 placed by robomimic.scripts.download_datasets."
        ),
    )
    p.add_argument(
        "--task",
        type=str,
        default="can",
        help="Task name used to find the default dataset (can / square / ...).",
    )
    p.add_argument(
        "--n-demos",
        type=int,
        default=5,
        help="Number of demos to replay. Use -1 for all.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Where to write the results JSON. Default: "
            "experiments/01_demo_replay/results/<task>_ph.json"
        ),
    )
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
        output_path=output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
