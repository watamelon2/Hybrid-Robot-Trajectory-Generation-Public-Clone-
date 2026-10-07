"""Replay flow-prior warm-starts directly in robosuite (no MBD).

Diagnostic for the v_theta v2 → 0/60 finding.  If this script reports
>0/60 success, the prior IS task-capable and MBD is interfering.  If
it reports 0/60, the prior alone cannot succeed and no λ_prior tuning
will recover -- the score-bridge would need to combine with a
*successful* warm-start source (e.g. BC).

CLI:
    python experiments/08_flow_matching_prior/replay_prior.py \
        --checkpoint experiments/08_flow_matching_prior/results/v_theta_v2.pt \
        --n-demos 20 --n-rollouts 3 --H 120 --seed 0 \
        --n-workers 10 \
        --output experiments/08_flow_matching_prior/results/replay_v2.json

Schema of the output JSON matches a strict subset of the
``04_mbd_baseline/run_can.py`` output so downstream analysis
scripts can ingest it identically.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

# sys.path: experiments/ and experiments/08_*/
_THIS_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _THIS_DIR.parent
for _p in (_THIS_DIR, _EXPERIMENTS_DIR):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
# Also experiments/04_mbd_baseline for worker module
sys.path.insert(0, str(_EXPERIMENTS_DIR / "04_mbd_baseline"))

from adapter import FlowPrior  # noqa: E402
from common import default_dataset_path  # noqa: E402
from worker import _worker_init, _worker_rollout  # noqa: E402

# Local dataset loader (08_flow_matching_prior/dataset.py is different
# from 03_bc_baseline/dataset.py; load by explicit path).
import importlib.util  # noqa: E402

_ds08_spec = importlib.util.spec_from_file_location(
    "ds08_dataset", _THIS_DIR / "dataset.py"
)
ds08 = importlib.util.module_from_spec(_ds08_spec)
_ds08_spec.loader.exec_module(ds08)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to v_theta_v?.pt")
    p.add_argument("--dataset", type=Path, default=None,
                   help="Path to low_dim_v141.hdf5; default=robomimic-resolved")
    p.add_argument("--task", type=str, default="can")
    p.add_argument("--n-demos", type=int, default=20)
    p.add_argument("--n-rollouts", type=int, default=3,
                   help="Independent prior samples per demo.")
    p.add_argument("--H", type=int, default=120)
    p.add_argument("--n-prior-steps", type=int, default=32,
                   help="ODE-integrator steps inside prior.sample_initial_mean.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu",
                   help="torch device for the prior forward passes.")
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset or default_dataset_path(args.task)
    if not dataset_path.exists():
        raise SystemExit(f"dataset not found: {dataset_path}")

    print(f"[replay_prior] checkpoint  : {args.checkpoint}")
    print(f"[replay_prior] dataset     : {dataset_path}")
    print(f"[replay_prior] n_demos     : {args.n_demos}")
    print(f"[replay_prior] n_rollouts  : {args.n_rollouts}")
    print(f"[replay_prior] H           : {args.H}")
    print(f"[replay_prior] n_workers   : {args.n_workers}")

    # 1. Load prior, sample warm-starts in main process.
    print("[replay_prior] loading prior...", flush=True)
    prior = FlowPrior.from_checkpoint(str(args.checkpoint), device=args.device)
    print(f"[replay_prior] prior loaded: {type(prior).__name__}", flush=True)

    print(f"[replay_prior] loading {args.n_demos} demos...", flush=True)
    actions_real, obs0, _, demo_ids = ds08.load_trajectories(
        dataset_path, horizon=args.H, n_demos=args.n_demos
    )
    print(f"[replay_prior]   obs0 {obs0.shape}  actions {actions_real.shape}", flush=True)

    # Load env init states from the dataset (mjData states[0] per demo).
    import h5py
    with h5py.File(str(dataset_path), "r") as f:
        all_demo_keys = sorted(
            f["data"].keys(), key=lambda k: int(k.split("_")[1])
        )[: args.n_demos]
        init_states = [
            np.asarray(f[f"data/{k}/states"][0]) for k in all_demo_keys
        ]
    print(f"[replay_prior]   init_states[0].shape = {init_states[0].shape}", flush=True)

    # 2. Sample warm-starts for all (demo, rollout) pairs.  Seed by
    # (seed + demo_idx * n_rollouts + r_idx) for reproducibility.
    print(f"[replay_prior] sampling {args.n_demos * args.n_rollouts} "
          f"warm-starts from prior ({args.n_prior_steps} ODE steps each)...",
          flush=True)
    t_sample = time.perf_counter()
    plans = []  # list of (demo_idx, r_idx, init_state, traj)
    for i in range(args.n_demos):
        for r in range(args.n_rollouts):
            seed_ir = args.seed + i * args.n_rollouts + r
            traj = prior.sample_initial_mean(
                obs0[i], n_steps=args.n_prior_steps, seed=seed_ir
            )
            # robosuite OSC_POSE action bounds = [-1, 1].
            traj = np.clip(traj, -1.0, 1.0).astype(np.float32)
            plans.append((i, r, init_states[i], traj))
    print(f"[replay_prior]   sampled in {time.perf_counter() - t_sample:.1f}s",
          flush=True)

    # 3. Spawn worker pool, replay each (init_state, traj).
    print(f"[replay_prior] starting pool of {args.n_workers} workers...",
          flush=True)
    t_pool = time.perf_counter()
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        args.n_workers,
        initializer=_worker_init,
        initargs=(str(dataset_path),),
    )
    print(f"[replay_prior]   pool ready in {time.perf_counter() - t_pool:.1f}s",
          flush=True)

    print(f"[replay_prior] replaying {len(plans)} plans...", flush=True)
    t_replay = time.perf_counter()
    args_list = [(init_state, traj) for _, _, init_state, traj in plans]
    chunksize = max(1, len(plans) // (args.n_workers * 2))
    results = pool.map(_worker_rollout, args_list, chunksize=chunksize)
    pool.close()
    pool.join()
    t_replay_done = time.perf_counter() - t_replay
    print(f"[replay_prior]   replayed in {t_replay_done:.1f}s "
          f"({t_replay_done / len(plans):.2f}s/plan avg)", flush=True)

    # 4. Aggregate + save.
    per_rollout = []
    n_success = 0
    for (i, r, init_state, traj), (total_reward, success) in zip(plans, results):
        per_rollout.append({
            "demo": f"demo_{i}",
            "rollout_idx": int(r),
            "exec_reward": float(total_reward),
            "success": bool(success),
        })
        if success:
            n_success += 1

    rate = n_success / len(plans) if plans else 0.0
    print(f"\n{'=' * 60}")
    print(f"[replay_prior] SUCCESS: {n_success}/{len(plans)} = {rate * 100:.1f}%")
    print(f"{'=' * 60}")

    out = {
        "experiment": "08_flow_matching_prior/replay_prior",
        "checkpoint": str(args.checkpoint),
        "dataset": str(dataset_path),
        "n_demos": int(args.n_demos),
        "n_rollouts": int(args.n_rollouts),
        "n_total": int(len(plans)),
        "n_success": int(n_success),
        "success_rate": float(rate),
        "H": int(args.H),
        "n_prior_steps": int(args.n_prior_steps),
        "seed": int(args.seed),
        "wall_total_s": float(time.perf_counter() - t_sample),
        "wall_replay_s": float(t_replay_done),
        "per_rollout": per_rollout,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[replay_prior] wrote {args.output}")


if __name__ == "__main__":
    main()
