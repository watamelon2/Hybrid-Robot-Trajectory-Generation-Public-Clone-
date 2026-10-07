"""Train v_theta v6 (Conditional Flow Matching, Lipman 2023) on robomimic.

Drop-in successor to ``train_v3.py`` for the *flow-matching* objective
(rather than the EDM denoising-score-matching objective).  Same data
loader, same backbones, same dataset, same checkpoint format —
literally everything except the loss is shared, so any difference in
downstream env-success between v3 and v6 is attributable to the
training objective alone.

Loss (per chunk, per (t, ε) draw):
    ε ~ N(0, I)
    t ~ U[0, 1 - t_eps]
    x_t   = (1 - t) ε + t x_data
    target = x_data - ε
    loss   = || v_θ(x_t, t, s) - target ||²

Sampler (inference time): Heun 2nd-order ODE on t ∈ [0, 1 - t_eps].

Usage
-----

GPU (Colab / cluster, ~30 min for 200 epochs)::

    python experiments/08_flow_matching_prior/train_v6.py \
        --task can --n-demos 200 --horizon 120 \
        --epochs 200 --batch-size 256 --lr 3e-4 \
        --hidden 768 --n-blocks 4 --backbone conv1d_unet \
        --device cuda --amp \
        --output experiments/08_flow_matching_prior/results/v_theta_v6_unet.pt

CPU smoke (~30 s)::

    python experiments/08_flow_matching_prior/train_v6.py --smoke
"""

from __future__ import annotations

import argparse
import json
import math
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

from dataset_v3 import load_chunks  # noqa: E402
from model_v6_cfm import (  # noqa: E402
    CFMField,
    FlowConfigV6,
    cfm_loss,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", type=str, default="can")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--n-demos", type=int, default=200)
    p.add_argument("--horizon", type=int, default=120)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--cond-hidden", type=int, default=256)
    p.add_argument("--t-embed-dim", type=int, default=64)
    p.add_argument("--t-eps", type=float, default=1e-3,
                   help="Min |1 - t| at training and inference.  Prevents "
                        "the score-bridge denominator (1 - t) from blowing "
                        "up at t -> 1.")
    p.add_argument(
        "--backbone",
        type=str,
        default="conv1d_unet",
        choices=["film_mlp", "conv1d_unet", "transformer"],
        help="Trunk architecture.  Same options as train_v3.py: "
             "'film_mlp' (flat MLP, no temporal inductive bias), "
             "'conv1d_unet' (Diffusion-Policy style 1D conv U-Net), "
             "'transformer' (BESO-style encoder).  Default conv1d_unet "
             "because that's where v4_unet showed the biggest DSM-loss "
             "improvement -- worth seeing if CFM gets the same benefit.",
    )
    p.add_argument("--n-t-per-traj", type=int, default=8,
                   help="Independent (t, ε) draws per chunk per step. "
                        "Higher reduces variance of the gradient estimator. "
                        "Default 8 matches train_v3.py's n_sigma_per_traj.")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Global grad-norm clip.  Required because the "
                        "score-bridge implicit in (1 - t) -> 0 produces "
                        "occasional gradient spikes near t = 1.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--amp", action="store_true",
                   help="Use mixed precision (cuda only).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output", type=Path,
        default=_THIS_DIR / "results" / "v_theta_v6_unet.pt",
    )
    p.add_argument(
        "--log", type=Path,
        default=_THIS_DIR / "results" / "train_log_v6_unet.json",
    )
    p.add_argument("--print-every", type=int, default=0,
                   help="Print every N epochs (0 = ~20 lines total)")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny config for end-to-end wiring tests.")
    return p.parse_args()


def cosine_lr(epoch: int, total_epochs: int, base_lr: float, warmup: int) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.n_demos = 3
        args.epochs = 4
        args.hidden = 64
        args.n_blocks = 2
        args.cond_hidden = 64
        args.n_t_per_traj = 4
        args.batch_size = 32

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_path = args.dataset or default_dataset_path(args.task)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    t0 = time.perf_counter()
    bundle = load_chunks(
        dataset_path,
        horizon=int(args.horizon),
        task=args.task,
        n_demos=args.n_demos,
    )
    print(
        f"[exp08-v6] loaded {bundle.n_demos} demos -> "
        f"{bundle.n_chunks} chunks  "
        f"X={bundle.X.shape}  S={bundle.S_norm.shape}  "
        f"mean_demo_len={bundle.demo_lens.mean():.1f}  "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    X = bundle.X.reshape(bundle.X.shape[0], -1).astype(np.float32)
    S = bundle.S_norm.astype(np.float32)

    config = FlowConfigV6(
        obs_dim=int(S.shape[1]),
        action_dim=int(bundle.action_dim),
        horizon=int(args.horizon),
        backbone=str(args.backbone),
        hidden=int(args.hidden),
        n_blocks=int(args.n_blocks),
        cond_hidden=int(args.cond_hidden),
        t_embed_dim=int(args.t_embed_dim),
        t_eps=float(args.t_eps),
    )
    device = torch.device(args.device)
    model = CFMField(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[exp08-v6] arch=cfm_v6  params={n_params:,}  "
        f"backbone={config.backbone}  config={config.to_dict()}"
    )

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(S))
    loader = DataLoader(
        ds,
        batch_size=min(int(args.batch_size), X.shape[0]),
        shuffle=True,
        drop_last=False,
    )
    print(
        f"[exp08-v6] dataloader: n={len(ds)} chunks  "
        f"batch_size={loader.batch_size}  "
        f"n_batches/epoch={len(loader)}"
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if use_amp
        else torch.amp.autocast("cpu", enabled=False)
    )
    print(f"[exp08-v6] device={device}  amp={use_amp}")

    print_every = (
        args.print_every if args.print_every > 0
        else max(args.epochs // 20, 1)
    )

    grad_clip = (
        float(args.grad_clip)
        if args.grad_clip and args.grad_clip > 0
        else None
    )
    print(f"[exp08-v6] grad_clip={grad_clip}")

    log: list[dict] = []
    wall0 = time.perf_counter()
    n_t = int(args.n_t_per_traj)
    for epoch in range(int(args.epochs)):
        lr = cosine_lr(epoch, int(args.epochs), float(args.lr), int(args.warmup_epochs))
        for g in opt.param_groups:
            g["lr"] = lr

        model.train()
        epoch_losses: list[float] = []
        max_grad_norm = 0.0
        n_clipped = 0
        for batch_idx, (x_batch, s_batch) in enumerate(loader):
            x_batch = x_batch.to(device)
            s_batch = s_batch.to(device)

            # n_t independent (t, ε) draws per chunk; average the loss.
            with amp_ctx:
                loss = sum(
                    cfm_loss(model, x_batch, s_batch) for _ in range(n_t)
                ) / float(n_t)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[exp08-v6] non-finite loss at epoch={epoch} "
                    f"batch={batch_idx} loss={float(loss):.3e}."
                )
            opt.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(opt)
                    g_norm = float(torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip,
                    ))
                else:
                    g_norm = 0.0
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    g_norm = float(torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip,
                    ))
                else:
                    g_norm = 0.0
                opt.step()
            if grad_clip is not None:
                max_grad_norm = max(max_grad_norm, g_norm)
                if g_norm > grad_clip:
                    n_clipped += 1
            epoch_losses.append(float(loss.detach().cpu()))
        avg = float(np.mean(epoch_losses))
        log.append({
            "epoch": epoch, "loss": avg, "lr": lr,
            "max_grad_norm": float(max_grad_norm),
            "n_clipped": int(n_clipped),
        })
        if epoch % print_every == 0 or epoch == int(args.epochs) - 1:
            print(
                f"[exp08-v6] epoch {epoch:>5}/{args.epochs}  "
                f"loss={avg:.4f}  lr={lr:.2e}  "
                f"max_gn={max_grad_norm:.2f}  clipped={n_clipped}/{len(loader)}"
            )
    wall = time.perf_counter() - wall0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 6,
            "arch": "cfm_v6",
            "config": config.to_dict(),
            "state_dict": model.state_dict(),
            "obs_mean": bundle.obs_mean.astype(np.float32),
            "obs_std": bundle.obs_std.astype(np.float32),
            "n_demos": int(bundle.n_demos),
            "n_chunks": int(bundle.n_chunks),
            "horizon": int(args.horizon),
            "task": str(args.task),
            "wall_s": float(wall),
        },
        args.output,
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(log, indent=2))
    print(
        f"[exp08-v6] wrote {args.output}  "
        f"final_loss={log[-1]['loss']:.4f}  wall={wall:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
