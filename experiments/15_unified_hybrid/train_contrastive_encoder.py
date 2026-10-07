"""Train a contrastive obs embedding for retrieval priors (C1).

The motivation comes from Slate G's verdict:  five orthogonal axes
(prior identity, bridge lambda, sigma schedule, MBD compute, raw
obs vocabulary including action-history) all leave Square below 9 %
succ at the F1 sweet spot.  The remaining hypothesis is that the
**raw obs vector doesn't separate Square's contact phases** -- a
fingertip-on-peg state and a fingertip-just-released state look
nearly identical in 46-d raw obs but require very different
actions.  A contrastive encoder, trained on temporal proximity
inside demos, embeds time-adjacent states close together and
time-far states far apart, recovering phase identity even when
the instantaneous proprio/object reading is ambiguous.

This script trains a small MLP encoder via InfoNCE:

  * anchor: obs at time t (after the same z-score normalisation
    that the demo bank uses for retrieval).
  * positive: obs at time t + k for a small look-ahead k (sampled
    uniformly in [1, k_max] per pair) in the **same** demo.
  * negatives: the other (B - 1) anchors in the batch.

The encoder is a 3-layer MLP with LayerNorm and ReLU, producing a
``embed_dim`` (default 32) output.  Embeddings are L2-normalised
to the unit sphere, so InfoNCE collapses to a temperature-scaled
cosine similarity classification.

The saved checkpoint contains everything ``priors_zerotrain``
needs to reconstruct the encoder at inference time:

  {
    "state_dict":    nn.Module state dict,
    "input_dim":     int (D),
    "embed_dim":     int (E),
    "hidden_dim":    int,
    "task":          str,
    "obs_mean":      (D,) float32,   <- bank.obs_mean
    "obs_std":       (D,) float32,   <- bank.obs_std
    "action_history_len": int,       <- bank.action_history_len
    "hparams":       dict (seed, lr, epochs, batch_size, k_max, tau, ...),
  }

Slate I I1 hypothesis: training this encoder on Square demos and
plugging it into a ``demo_1nn_contrastive`` prior at the F1 sweet
spot lifts succ above 15 % (weak gate) -- proving the bottleneck
was retrieval, not the bridge / planner.  If it stays at the
~8 % AUX1 peak, the structural change must be in the bridge or
cost function instead, and C2/C3 become the next axes.

Usage
-----
    python experiments/15_unified_hybrid/train_contrastive_encoder.py \
        --task square \
        --output-path artifacts/contrastive_encoder_square.pt

CLI args (full list in ``build_arg_parser``):

    --task                  robomimic task name (default 'square').
    --n-demos               cap on demo count (default None = all).
    --embed-dim             output embedding dim (default 32).
    --hidden-dim            MLP hidden width (default 128).
    --epochs                training epochs (default 60).
    --batch-size            pairs per batch (default 512).
    --k-max                 max positive look-ahead in steps
                            (default 4; positive sampled uniformly
                            in [1, k_max] per pair).
    --tau                   InfoNCE temperature (default 0.1).
    --lr                    Adam learning rate (default 1e-3).
    --weight-decay          Adam weight decay (default 1e-5).
    --seed                  RNG seed (default 0).
    --device                torch device (default 'mps' on macOS,
                            else 'cpu').
    --output-path           checkpoint path (required).
    --action-history-len    A1 augmentation len for the bank
                            (default 0).  Must match the
                            ``action_history_len`` used at
                            retrieval time.
    --verbose               log per-epoch loss (default True).

Sanity:  prints the mean and stdev of in-batch positive-pair
similarity (should rise to ~0.9) and negative-pair similarity
(should settle near 0).  If these don't separate, the training
diverged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure exp 03 dataset module + experiments/common are importable,
# mirroring run_can.py's path bootstrap.
_HERE = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _HERE.parent
_REPO_ROOT = _EXPERIMENTS_DIR.parent
for _sp in (
    str(_EXPERIMENTS_DIR),
    str(_EXPERIMENTS_DIR / "03_bc_baseline"),
    str(_HERE),
):
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

from common import default_dataset_path  # noqa: E402
from priors_zerotrain import load_demo_bank  # noqa: E402


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ContrastiveEncoder(nn.Module):
    """3-layer MLP with LayerNorm; L2-normalised output."""

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.embed_dim = int(embed_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


# ---------------------------------------------------------------------------
# Pair index sampling
# ---------------------------------------------------------------------------


def _build_pair_pool(
    demo_starts: np.ndarray,
    demo_lengths: np.ndarray,
    k_max: int,
) -> np.ndarray:
    """Return ``(n_pairs, 2)`` int64 array of (anchor, positive) row indices.

    For each demo of length L, we enumerate all (t, t+k) pairs with
    ``t + k < L`` and ``1 <= k <= k_max``.  Positives never cross
    demo boundaries.  At training time we sample uniformly from
    this pool for each batch.
    """
    anchors: list[np.ndarray] = []
    positives: list[np.ndarray] = []
    for start, length in zip(demo_starts, demo_lengths):
        start = int(start)
        length = int(length)
        for k in range(1, int(k_max) + 1):
            if k >= length:
                break
            t = np.arange(0, length - k, dtype=np.int64)
            anchors.append(start + t)
            positives.append(start + t + k)
    if not anchors:
        raise ValueError(
            "no (anchor, positive) pairs could be built; check k_max "
            "against demo lengths"
        )
    a = np.concatenate(anchors)
    p = np.concatenate(positives)
    return np.stack([a, p], axis=1).astype(np.int64)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _infonce_loss(
    anchor_z: torch.Tensor,
    positive_z: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, float, float]:
    """Symmetric InfoNCE with in-batch negatives.

    Both ``anchor_z`` and ``positive_z`` are ``(B, E)`` L2-
    normalised, so the dot product is the cosine similarity.
    """
    logits = anchor_z @ positive_z.t() / float(tau)   # (B, B)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_p = F.cross_entropy(logits.t(), labels)
    loss = 0.5 * (loss_a + loss_p)
    with torch.no_grad():
        pos_sim = (anchor_z * positive_z).sum(dim=-1).mean().item()
        mask = ~torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        neg_sim = ((anchor_z @ positive_z.t())[mask]).mean().item()
    return loss, pos_sim, neg_sim


def train(
    bank,
    encoder: ContrastiveEncoder,
    pairs: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    tau: float,
    lr: float,
    weight_decay: float,
    seed: int,
    device: str,
    verbose: bool,
) -> dict:
    """Train ``encoder`` on InfoNCE pairs; return a metrics dict."""
    obs_norm = torch.as_tensor(bank.obs_norm, dtype=torch.float32, device=device)
    n_pairs = pairs.shape[0]
    rng = np.random.RandomState(int(seed))

    opt = torch.optim.AdamW(
        encoder.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )

    steps_per_epoch = max(1, n_pairs // int(batch_size))
    losses: list[float] = []
    pos_sims: list[float] = []
    neg_sims: list[float] = []
    for ep in range(int(epochs)):
        encoder.train()
        # Shuffle pair indices once per epoch.
        order = rng.permutation(n_pairs)
        ep_loss = 0.0
        ep_pos = 0.0
        ep_neg = 0.0
        for s in range(steps_per_epoch):
            sel = order[s * batch_size : (s + 1) * batch_size]
            if sel.size < 2:
                continue
            anchor_idx = pairs[sel, 0]
            pos_idx = pairs[sel, 1]
            a = obs_norm[anchor_idx]
            p = obs_norm[pos_idx]
            za = encoder(a)
            zp = encoder(p)
            loss, pos_sim, neg_sim = _infonce_loss(za, zp, tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            ep_pos += pos_sim
            ep_neg += neg_sim
        ep_loss /= steps_per_epoch
        ep_pos /= steps_per_epoch
        ep_neg /= steps_per_epoch
        losses.append(ep_loss)
        pos_sims.append(ep_pos)
        neg_sims.append(ep_neg)
        if verbose:
            print(
                f"[c1] epoch {ep+1:3d}/{epochs}  loss={ep_loss:.4f}  "
                f"pos_sim={ep_pos:+.3f}  neg_sim={ep_neg:+.3f}  "
                f"gap={ep_pos - ep_neg:+.3f}"
            )

    return {
        "final_loss": losses[-1],
        "final_pos_sim": pos_sims[-1],
        "final_neg_sim": neg_sims[-1],
        "epoch_losses": losses,
        "epoch_pos_sims": pos_sims,
        "epoch_neg_sims": neg_sims,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a C1 contrastive obs embedding for retrieval."
    )
    p.add_argument("--task", type=str, default="square",
                   help="Robomimic task (default: square).")
    p.add_argument("--dataset", type=Path, default=None,
                   help="Override dataset HDF5 path. "
                        "Default: default_dataset_path(--task).")
    p.add_argument("--n-demos", type=int, default=None,
                   help="Cap on demo count (default: all).")
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--k-max", type=int, default=4,
                   help="Max positive look-ahead in steps.")
    p.add_argument("--tau", type=float, default=0.1,
                   help="InfoNCE temperature.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None,
                   help="torch device; default mps on macOS, else cpu.")
    p.add_argument("--action-history-len", type=int, default=0,
                   help="A1 obs-augmentation length (must match the "
                        "value used at retrieval time).")
    p.add_argument("--output-path", type=Path, required=True)
    p.add_argument("--verbose", action=argparse.BooleanOptionalAction,
                   default=True)
    return p


def _autodetect_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = _autodetect_device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    dataset_path = args.dataset or default_dataset_path(args.task)
    print(f"[c1] loading bank: task={args.task}  dataset={dataset_path}")
    t0 = time.perf_counter()
    bank = load_demo_bank(
        dataset_path,
        n_demos=args.n_demos,
        task=args.task,
        action_history_len=int(args.action_history_len),
    )
    print(
        f"[c1] bank ready in {time.perf_counter() - t0:.1f}s: "
        f"{bank.n_demos} demos  {bank.n_total} steps  "
        f"obs_dim={bank.obs.shape[1]}  action_history_len={bank.action_history_len}"
    )

    pairs = _build_pair_pool(bank.demo_starts, bank.demo_lengths, args.k_max)
    print(
        f"[c1] built {pairs.shape[0]} (anchor, positive) pairs "
        f"with k in [1, {args.k_max}]"
    )

    encoder = ContrastiveEncoder(
        input_dim=bank.obs.shape[1],
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
    ).to(device)
    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(
        f"[c1] encoder: input_dim={encoder.input_dim} "
        f"hidden_dim={encoder.hidden_dim}  embed_dim={encoder.embed_dim}  "
        f"n_params={n_params}  device={device}"
    )

    t0 = time.perf_counter()
    metrics = train(
        bank,
        encoder,
        pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        tau=args.tau,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
        verbose=args.verbose,
    )
    train_wall_s = time.perf_counter() - t0
    print(
        f"[c1] training finished in {train_wall_s:.1f}s.  "
        f"final loss={metrics['final_loss']:.4f}  "
        f"pos_sim={metrics['final_pos_sim']:+.3f}  "
        f"neg_sim={metrics['final_neg_sim']:+.3f}  "
        f"gap={metrics['final_pos_sim'] - metrics['final_neg_sim']:+.3f}"
    )

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": encoder.state_dict(),
        "input_dim": int(encoder.input_dim),
        "hidden_dim": int(encoder.hidden_dim),
        "embed_dim": int(encoder.embed_dim),
        "task": args.task,
        "obs_mean": torch.as_tensor(bank.obs_mean, dtype=torch.float32),
        "obs_std": torch.as_tensor(bank.obs_std, dtype=torch.float32),
        "action_history_len": int(bank.action_history_len),
        "hparams": {
            "seed": int(args.seed),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "k_max": int(args.k_max),
            "tau": float(args.tau),
            "n_demos": (None if args.n_demos is None else int(args.n_demos)),
            "device": device,
            "train_wall_s": float(train_wall_s),
            "final_loss": float(metrics["final_loss"]),
            "final_pos_sim": float(metrics["final_pos_sim"]),
            "final_neg_sim": float(metrics["final_neg_sim"]),
        },
    }
    torch.save(payload, str(out_path))
    print(f"[c1] wrote encoder checkpoint -> {out_path}")

    # Sidecar JSON for at-a-glance inspection without unpickling torch.
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta = {k: v for k, v in payload.items()
            if k not in {"state_dict", "obs_mean", "obs_std"}}
    meta["obs_mean_first8"] = bank.obs_mean[:8].tolist()
    meta["obs_std_first8"] = bank.obs_std[:8].tolist()
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[c1] wrote sidecar metadata -> {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
