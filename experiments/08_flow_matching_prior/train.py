"""Train the flow-matching velocity field v_theta on can/ph low-dim demos.

Tier 1a in `docs/plans.md`. Companion of exp 05's BC prior; same
dataset, same obs space, same action space, but represents a
*distribution* over H-step trajectories conditioned on s0 instead
of a deterministic per-step controller.

Outputs (under ``results/``):

  - ``v_theta.pt``: torch checkpoint with config + state_dict +
    obs-normalisation stats. Loadable by ``adapter.py`` for use as a
    `prior_score_fn` inside MBD or as an ``initial_mean`` source for
    exp 05's ``run_can.py --bc-checkpoint``-style entry point.
  - ``train_log.json``: per-epoch loss curve.

Default hyperparameters target a few-minute laptop run on the
20-demo headline slice. A `--smoke` flag uses 2 demos and 5 epochs
for end-to-end wiring tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


_THIS_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _THIS_DIR.parent
for _p in (_EXPERIMENTS_DIR, _THIS_DIR):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from common import default_dataset_path  # noqa: E402

from dataset import compute_obs_stats, load_trajectories  # noqa: E402
from model import FlowConfig, VelocityField, conditional_fm_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", type=str, default="can")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--n-demos", type=int, default=20)
    p.add_argument("--horizon", type=int, default=120)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output", type=Path,
        default=_THIS_DIR / "results" / "v_theta.pt",
    )
    p.add_argument(
        "--log", type=Path,
        default=_THIS_DIR / "results" / "train_log.json",
    )
    p.add_argument("--smoke", action="store_true",
                   help="Tiny config for end-to-end wiring tests.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.n_demos = 2
        args.epochs = 5
        args.hidden = 64
        args.n_layers = 2

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_path = args.dataset or default_dataset_path(args.task)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    t0 = time.perf_counter()
    actions, obs0, demo_lens, demo_ids = load_trajectories(
        dataset_path, horizon=args.horizon, n_demos=args.n_demos
    )
    print(
        f"[exp08] loaded {len(demo_ids)} demos  "
        f"actions={actions.shape}  obs0={obs0.shape}  "
        f"mean_demo_len={demo_lens.mean():.1f}  ({time.perf_counter() - t0:.1f}s)"
    )

    stats = compute_obs_stats(obs0)
    obs0_norm = stats.normalize(obs0).astype(np.float32)
    X = actions.reshape(actions.shape[0], -1).astype(np.float32)
    S = obs0_norm

    n = X.shape[0]
    A = int(actions.shape[2])
    config = FlowConfig(
        obs_dim=int(S.shape[1]),
        action_dim=A,
        horizon=args.horizon,
        hidden_sizes=tuple([args.hidden] * args.n_layers),
    )
    model = VelocityField(config).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[exp08] model params: {n_params:,}  config={config.to_dict()}")

    ds = TensorDataset(
        torch.from_numpy(X), torch.from_numpy(S)
    )
    loader = DataLoader(
        ds, batch_size=min(args.batch_size, n),
        shuffle=True, drop_last=False,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    log: list[dict] = []
    wall0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        epoch_losses: list[float] = []
        for x_batch, s_batch in loader:
            x_batch = x_batch.to(args.device)
            s_batch = s_batch.to(args.device)
            loss = conditional_fm_loss(model, x_batch, s_batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        avg = float(np.mean(epoch_losses))
        log.append({"epoch": epoch, "loss": avg})
        if epoch % max(args.epochs // 20, 1) == 0 or epoch == args.epochs - 1:
            print(f"[exp08] epoch {epoch:>4}/{args.epochs}  loss={avg:.4f}")
    wall = time.perf_counter() - wall0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "config": config.to_dict(),
            "state_dict": model.state_dict(),
            "obs_mean": stats.mean.astype(np.float32),
            "obs_std": stats.std.astype(np.float32),
            "n_demos": int(len(demo_ids)),
            "horizon": int(args.horizon),
            "wall_s": float(wall),
        },
        args.output,
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(log, indent=2))
    print(
        f"[exp08] wrote {args.output}  "
        f"final_loss={log[-1]['loss']:.4f}  wall={wall:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
