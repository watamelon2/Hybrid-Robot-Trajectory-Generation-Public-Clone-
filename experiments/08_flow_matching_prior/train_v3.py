"""Train v_theta v3 (Karras-EDM noise-conditional score) on robomimic demos.

Drop-in successor to ``train_v2.py``.  Three differences:

1.  **Per-chunk dataset.**  ``dataset_v3.load_chunks`` emits one
    example per demo timestep ``t`` -- conditioning ``s_t`` is the
    obs at chunk start, target is the H-step action window
    ``a_{t:t+H}`` (edge-padded at the tail).  This is what fixes
    v2's mid-trajectory OOD failure under MPC replanning.

2.  **Karras EDM loss.**  Model output is a denoiser ``D(x; σ, s)``;
    loss is ``λ(σ) · ||D - x_0||²`` with σ ~ log-Uniform[σ_min, σ_max]
    and λ(σ) the Karras 2022 weighting.

3.  **σ range tied to MBD's schedule.**  The trainer takes
    ``--sigma-min`` and ``--sigma-max`` flags and bakes them into the
    checkpoint, so inference-time σ_k always lies inside the
    training-time distribution of σ.  Defaults match
    ``cosine_sigma_schedule(K, sigma_max=0.5, sigma_min=0.05)``
    used by the unified driver at the canonical operating point.

Usage
-----

GPU (Colab / cluster, ~30 min)::

    python experiments/08_flow_matching_prior/train_v3.py \\
        --task can --n-demos 200 --horizon 120 \\
        --epochs 200 --batch-size 256 --lr 3e-4 \\
        --hidden 768 --n-blocks 4 \\
        --sigma-min 0.05 --sigma-max 0.5 \\
        --device cuda --amp \\
        --output experiments/08_flow_matching_prior/results/v_theta_v3.pt

CPU smoke (~30 s, sanity-check the wiring only)::

    python experiments/08_flow_matching_prior/train_v3.py --smoke
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
from model_v3 import (  # noqa: E402
    FilmEdmField,
    FlowConfigV3,
    edm_loss,
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
    p.add_argument("--sigma-embed-dim", type=int, default=64)
    p.add_argument("--sigma-min", type=float, default=0.05,
                   help="Min σ during training.  MBD's cosine_sigma_schedule "
                        "min must be >= this.")
    p.add_argument("--sigma-max", type=float, default=0.5,
                   help="Max σ during training.  MBD's cosine_sigma_schedule "
                        "max must be <= this.")
    p.add_argument("--sigma-data", type=float, default=0.5,
                   help="Karras EDM prior σ_data.  Action range is [-1, 1] "
                        "so 0.5 is a sane default; tune if z-score normalised.")
    p.add_argument("--n-sigma-per-traj", type=int, default=8,
                   help="Independent (σ, ε) draws per chunk per step.")
    p.add_argument(
        "--backbone",
        type=str,
        default="film_mlp",
        choices=["film_mlp", "conv1d_unet", "transformer"],
        help="Trunk architecture (v4 ablation):  'film_mlp' = v3 default, "
             "flat-MLP with FiLM (no temporal inductive bias).  "
             "'conv1d_unet' = Diffusion-Policy-style 1D conv U-Net.  "
             "'transformer' = BESO-style Transformer encoder.  "
             "All three keep the same Karras EDM training objective.",
    )
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Global grad-norm clip (Karras 2022 § 5.1 uses 1.0). "
                        "Required because λ(σ) ~ 1/σ² at the low-σ tail produces "
                        "occasional gradient spikes that NaN out training.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--amp", action="store_true",
                   help="Use mixed precision (cuda only).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output", type=Path,
        default=_THIS_DIR / "results" / "v_theta_v3.pt",
    )
    p.add_argument(
        "--log", type=Path,
        default=_THIS_DIR / "results" / "train_log_v3.json",
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
        args.n_sigma_per_traj = 4
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
        f"[exp08-v3] loaded {bundle.n_demos} demos -> "
        f"{bundle.n_chunks} chunks  "
        f"X={bundle.X.shape}  S={bundle.S_norm.shape}  "
        f"mean_demo_len={bundle.demo_lens.mean():.1f}  "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    X = bundle.X.reshape(bundle.X.shape[0], -1).astype(np.float32)
    S = bundle.S_norm.astype(np.float32)

    config = FlowConfigV3(
        obs_dim=int(S.shape[1]),
        action_dim=int(bundle.action_dim),
        horizon=int(args.horizon),
        backbone=str(args.backbone),
        hidden=int(args.hidden),
        n_blocks=int(args.n_blocks),
        cond_hidden=int(args.cond_hidden),
        sigma_embed_dim=int(args.sigma_embed_dim),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        sigma_data=float(args.sigma_data),
    )
    device = torch.device(args.device)
    model = FilmEdmField(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[exp08-v3] arch=karras_edm_v3  params={n_params:,}  "
        f"config={config.to_dict()}"
    )

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(S))
    loader = DataLoader(
        ds,
        batch_size=min(int(args.batch_size), X.shape[0]),
        shuffle=True,
        drop_last=False,
    )
    print(
        f"[exp08-v3] dataloader: n={len(ds)} chunks  "
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
    print(f"[exp08-v3] device={device}  amp={use_amp}")

    print_every = (
        args.print_every if args.print_every > 0
        else max(args.epochs // 20, 1)
    )

    grad_clip = float(args.grad_clip) if args.grad_clip and args.grad_clip > 0 else None
    print(f"[exp08-v3] grad_clip={grad_clip}")

    log: list[dict] = []
    wall0 = time.perf_counter()
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
            with amp_ctx:
                loss = edm_loss(
                    model,
                    x_batch,
                    s_batch,
                    n_sigma_per_traj=int(args.n_sigma_per_traj),
                )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[exp08-v3] non-finite loss at epoch={epoch} batch={batch_idx} "
                    f"loss={float(loss):.3e}. This indicates numerical instability "
                    "in the EDM loss; check --sigma-data / --sigma-min / --grad-clip."
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
                f"[exp08-v3] epoch {epoch:>5}/{args.epochs}  "
                f"loss={avg:.4f}  lr={lr:.2e}  "
                f"max_gn={max_grad_norm:.2f}  clipped={n_clipped}/{len(loader)}"
            )
    wall = time.perf_counter() - wall0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 3,
            "arch": "karras_edm_v3",
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
        f"[exp08-v3] wrote {args.output}  "
        f"final_loss={log[-1]['loss']:.4f}  wall={wall:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
