"""Train a small MLP behaviour-cloning policy on the can/ph dataset.

Default hyperparameters are tuned for a CPU-only Mac: ~100 epochs of
batch-256 Adam over the 200-demo / 23k-transition training set finishes
in roughly two minutes on M-series silicon.

The script writes two artefacts under
``experiments/03_bc_baseline/results/``:

  - ``policy.pt`` -- a single ``torch.save`` blob (gitignored) bundling
    the trained weights, the obs-normalisation stats, and the model
    config. ``Policy.from_checkpoint`` re-hydrates everything.
  - ``training_log.json`` -- per-epoch loss curve, total wall time, and
    the hyperparameters used. Small enough to commit; gives the
    Week-4 report a citable curve without dragging a binary checkpoint
    into git.

Usage examples are at the top of ``README.md``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Make sibling experiment scripts importable without dotted-package
# syntax (this directory's name starts with a digit, see policy.py).
_THIS_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _THIS_DIR.parent
for _p in (_THIS_DIR, _EXPERIMENTS_DIR):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from common import default_dataset_path  # noqa: E402
from dataset import build_tensor_dataset, get_obs_keys, get_object_slices  # noqa: E402
from policy import MLPPolicy, Policy, PolicyConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to the robomimic low_dim HDF5. Defaults to "
        "default_dataset_path('can').",
    )
    p.add_argument(
        "--task",
        type=str,
        default="can",
        help="Task name passed to default_dataset_path (used only if "
        "--dataset is not given). Default: can.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_THIS_DIR / "results" / "policy.pt",
        help="Where to save the trained checkpoint (gitignored).",
    )
    p.add_argument(
        "--log",
        type=Path,
        default=_THIS_DIR / "results" / "training_log.json",
        help="Where to write the per-epoch training log (committed).",
    )
    p.add_argument(
        "--n-demos",
        type=int,
        default=-1,
        help="Number of demos to train on (default: all 200). Use a "
        "small positive value for smoke runs.",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    dataset_path = args.dataset or default_dataset_path(args.task)
    n_demos = None if args.n_demos in (-1, 0) else int(args.n_demos)

    print(f"[train] dataset      = {dataset_path}")
    print(f"[train] device       = {device}")
    print(f"[train] n_demos      = {n_demos if n_demos is not None else 'all'}")
    print(f"[train] epochs       = {args.epochs}")
    print(f"[train] batch_size   = {args.batch_size}")
    print(f"[train] learning_rate= {args.lr}")
    print(f"[train] seed         = {args.seed}")

    obs_keys = get_obs_keys(args.task)
    object_slices = get_object_slices(args.task)
    ds, stats, demo_ids = build_tensor_dataset(
        dataset_path,
        obs_keys=obs_keys,
        n_demos=n_demos,
        object_slices=object_slices,
    )
    print(f"[train] task         = {args.task}")
    print(f"[train] obs_keys     = {obs_keys}")
    print(f"[train] loaded {len(demo_ids)} demos, {len(ds)} transitions")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    obs_dim = ds[0][0].shape[0]
    config = PolicyConfig(
        obs_dim=int(obs_dim),
        hidden_sizes=(args.hidden, args.hidden),
        obs_keys=obs_keys,
    )
    net = MLPPolicy(config).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[train] policy: {config.hidden_sizes} MLP, {n_params} params")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    per_epoch: list[dict] = []
    t_total = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        net.train()
        epoch_loss = 0.0
        n_seen = 0
        t_epoch = time.perf_counter()
        for obs_b, act_b in loader:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)
            pred = net(obs_b)
            loss = loss_fn(pred, act_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n = obs_b.size(0)
            epoch_loss += float(loss.item()) * n
            n_seen += n
        avg = epoch_loss / max(n_seen, 1)
        elapsed = time.perf_counter() - t_epoch
        per_epoch.append(
            {"epoch": epoch, "loss": avg, "elapsed_s": elapsed}
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"[train] epoch {epoch:>3d}/{args.epochs}  "
                f"loss={avg:.6f}  ({elapsed:.2f}s)"
            )
    total_wall = time.perf_counter() - t_total

    # ------------------------------------------------------------------
    # Persist artefacts
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Policy.save(
        args.output,
        net,
        stats,
        extra={
            "dataset": str(dataset_path),
            "n_demos": len(demo_ids),
            "n_transitions": len(ds),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "seed": int(args.seed),
            "device": str(device),
        },
    )
    print(f"[train] checkpoint -> {args.output}")

    log = {
        "experiment": "03_bc_baseline",
        "dataset": str(dataset_path),
        "task": args.task,
        "obs_keys": list(obs_keys),
        "n_demos": len(demo_ids),
        "n_transitions": len(ds),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "hidden_sizes": list(config.hidden_sizes),
        "obs_dim": int(obs_dim),
        "action_dim": int(config.action_dim),
        "n_params": int(n_params),
        "seed": int(args.seed),
        "device": str(device),
        "per_epoch": per_epoch,
        "final_loss": per_epoch[-1]["loss"] if per_epoch else None,
        "total_wall_s": float(total_wall),
    }
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(log, indent=2))
    print(f"[train] training log -> {args.log}")
    print(f"[train] total wall {total_wall:.1f}s, final loss {log['final_loss']:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
