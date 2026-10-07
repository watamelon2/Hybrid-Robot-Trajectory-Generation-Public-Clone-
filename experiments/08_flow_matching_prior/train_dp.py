"""Train v_theta-dp (Diffusion Policy U-Net + DDPM objective) on robomimic.

Drop-in successor to ``train_v6.py`` / ``train_v3.py``: same data
pipeline (``dataset_v3.load_chunks``), same flags where possible,
different model + loss.  The point is a clean three-way comparison::

    v3   = our trunks + Karras EDM                (v3)
    v6   = our trunks + Lipman CFM                (v6)
    v7   = DP UNet1D + Ho et al. DDPM ε-pred      (this script)

All three train on the same 200-demo can/ph dataset at the same
horizon, same obs-dim, same z-score normalisation.  Any env-success
delta between them is attributable to the (architecture, objective)
choice and not to data shimmer.

Usage
-----

GPU (Colab T4, ~30 min full)::

    python experiments/08_flow_matching_prior/train_dp.py \
        --task can --n-demos 200 --horizon 120 \
        --epochs 200 --batch-size 256 --lr 1e-4 \
        --down-dims 256 512 1024 \
        --device cuda --amp \
        --output experiments/08_flow_matching_prior/results/v_theta_dp.pt

CPU smoke (~30 s)::

    python experiments/08_flow_matching_prior/train_dp.py --smoke
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

from dataset_v3 import load_chunks, stack_obs_history  # noqa: E402
from model_dp import (  # noqa: E402
    ConditionalUnet1D,
    DDPMSchedule,
    FlowConfigDP,
    dp_loss,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", type=str, default="can")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--n-demos", type=int, default=200)
    p.add_argument("--horizon", type=int, default=120)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    # DP paper trains with lr=1e-4 (lower than our v3/v6 default 3e-4)
    # because of the larger UNet capacity and the per-step ε-pred loss
    # which has less gradient signal per chunk.
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument(
        "--min-lr", type=float, default=0.0,
        help="Lower bound for the cosine LR schedule (default 0.0, "
             "i.e. anneal all the way to zero).  Set to 1e-6 (DP-paper "
             "convention) for runs > 500 epochs so weights keep moving "
             "in the late phase and EMA actually does averaging.  See "
             "docs/experiments/17_dp_hybrid.md §3.1 for the diagnosis.",
    )
    p.add_argument("--weight-decay", type=float, default=1e-6,
                   help="Lower than v3/v6 default (1e-5) because DP's "
                        "released config uses 1e-6; we follow their recipe.")
    # AdamW betas.  PyTorch default is (0.9, 0.999); DP paper / released
    # config uses (0.95, 0.999) for the lowdim configs (see
    # ``diffusion_policy/config/train_diffusion_unet_lowdim_workspace.yaml``
    # ``optimizer.betas``).  The higher β1=0.95 dampens short-term gradient
    # noise, which matters with the small per-step ε-pred loss; the
    # original PyTorch β1=0.9 is more reactive but noisier.  Phase-1/2/2c/2d
    # used PyTorch defaults; Phase-3 onward sets these to the DP values.
    p.add_argument("--adam-beta1", type=float, default=0.9,
                   help="AdamW β1 (default 0.9, PyTorch default).  DP paper "
                        "recipe is 0.95.")
    p.add_argument("--adam-beta2", type=float, default=0.999,
                   help="AdamW β2 (default 0.999, matches both PyTorch and "
                        "DP paper).")
    p.add_argument("--diffusion-step-embed-dim", type=int, default=256)
    p.add_argument("--down-dims", type=int, nargs="+",
                   default=[256, 512, 1024],
                   help="Channel dims for the U-Net encoder.  DP's "
                        "default is (256, 512, 1024) -> ~66M params at "
                        "horizon=120, action_dim=7.")
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--n-groups", type=int, default=8)
    p.add_argument("--no-cond-predict-scale", action="store_true",
                   help="Disable FiLM scale+shift conditioning (DP's default "
                        "is True; set this flag to fall back to additive "
                        "bias only, which is closer to a vanilla UNet).")
    p.add_argument("--n-obs-steps", type=int, default=1,
                   help="T_obs.  DP paper uses 2; we default to 1 to keep "
                        "apples-to-apples with v3/v6.  Setting >1 triggers "
                        "dataset_v3.stack_obs_history which builds a "
                        "(N, T_obs * obs_dim) conditioning vector with the "
                        "most-recent obs as the trailing block and "
                        "per-demo boundary replication.")
    p.add_argument("--num-train-timesteps", type=int, default=100)
    p.add_argument("--beta-schedule", type=str, default="squaredcos_cap_v2",
                   choices=["squaredcos_cap_v2", "linear"])
    p.add_argument("--grad-clip", type=float, default=1.0)
    # EMA (Polyak averaging).  DP paper recipe: decay=0.9999, applied
    # every optimisation step.  The saved checkpoint's ``state_dict``
    # is the EMA-averaged copy (what inference should use); the raw
    # model weights are kept under ``state_dict_raw`` for completeness.
    # Reference: diffusion_policy/model/diffusion/ema_model.py and
    # config diffusion_policy/config/task/lowdim/can_lowdim.yaml.
    p.add_argument(
        "--use-ema", action=argparse.BooleanOptionalAction, default=True,
        help="Maintain an exponential moving average of model "
             "parameters during training (DP paper recipe).  When on, "
             "the saved ``state_dict`` is the EMA-averaged copy; the "
             "raw model weights are also stored under "
             "``state_dict_raw``.  Default ON.",
    )
    p.add_argument(
        "--ema-decay", type=float, default=0.9999,
        help="EMA decay coefficient (Polyak).  Default 0.9999, matching "
             "DP's published config.",
    )
    # ---- val split + early stopping + intermediate ckpts ----
    # Added 2026-05-27 so Phase-2b can empirically locate the
    # generalization peak instead of guessing.  Phase-2 showed that
    # train-loss convergence (1000 ep -> 0.0008) corresponds to a
    # *worse* standalone (13.3 %) than the loss=0.034 peak at
    # epoch 50 (26.7 %).  Val loss + per-checkpoint env eval are the
    # only honest signals for where to stop.
    p.add_argument(
        "--val-frac", type=float, default=0.0,
        help="Fraction of demos held out as a validation split (per-"
             "demo split, not per-chunk, so adjacent chunks don't "
             "leak across train/val).  0.0 disables (legacy mode, "
             "no val signal).  Try 0.15-0.20 for ~200 demos.",
    )
    p.add_argument(
        "--save-every", type=int, default=0,
        help="Save an intermediate ckpt every N epochs (in addition "
             "to the final one and the best-val one).  Useful for "
             "mapping the generalization curve via env-eval at each "
             "checkpoint.  0 disables.  Ckpts go next to --output "
             "with a _epXXX suffix.",
    )
    p.add_argument(
        "--early-stop-patience", type=int, default=0,
        help="Stop training if val loss does not improve for this "
             "many epochs.  0 disables (run for full --epochs).  "
             "Requires --val-frac > 0.",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output", type=Path,
        default=_THIS_DIR / "results" / "v_theta_dp.pt",
    )
    p.add_argument(
        "--log", type=Path,
        default=_THIS_DIR / "results" / "train_log_dp.json",
    )
    p.add_argument("--print-every", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    # ---- action normalisation (DP paper recipe) ----
    # Added 2026-05-27 after the source-code audit (see
    # ``docs/experiments/17_dp_hybrid.md`` §4).  DP's
    # ``LinearNormalizer.fit(mode='limits')`` rescales every action
    # dim to ``[-1, 1]`` via per-dim min-max, computed over the
    # training-set actions.  This keeps the SNR uniform across the
    # DDPM noise schedule: with raw actions concentrated near zero,
    # high-σ_t timesteps see x_t dominated by ε and the model
    # learns ε ≈ x_t (a degenerate solution).  Defaults to ON to
    # match the reference; ``--no-normalize-actions`` reverts to
    # the legacy raw-action path for Phase-1/2/2c reproducibility.
    p.add_argument(
        "--normalize-actions",
        action=argparse.BooleanOptionalAction, default=True,
        help="Apply DP's LinearNormalizer-style min-max [-1, 1] "
             "normalisation to actions during training.  The fitted "
             "``action_min`` / ``action_max`` are stored in the "
             "checkpoint blob and the DP adapter denormalises at "
             "sample time so callers get env-space actions back.  "
             "Default ON.",
    )
    # ---- observation normalisation (DP paper recipe, Finding C) ----
    # Added 2026-05-28 after Phase-3 hit a 25-27 % eval ceiling that
    # was insensitive to training budget, EMA, dse_dim, or AdamW betas.
    # Obs norm is the only audit finding left untested.  DP's
    # ``LinearNormalizer.fit(mode='limits')`` rescales every obs dim
    # to ``[-1, 1]`` via per-dim min-max over the training-set obs --
    # exactly the same shape as the action-norm patch, just on the
    # conditioning vector.  When ON, the adapter switches its
    # ``_normalise_s_np`` from z-score to min-max so closed-loop
    # rollouts feed the model the same distribution it saw at train
    # time.  Default OFF so existing ckpts (Phase-1/2/2c/2d/3) keep
    # the legacy z-score recipe; Phase-3.5 notebook flips it.
    p.add_argument(
        "--normalize-obs",
        action=argparse.BooleanOptionalAction, default=False,
        help="Apply DP's LinearNormalizer-style min-max [-1, 1] "
             "normalisation to observations during training.  The "
             "fitted ``obs_min`` / ``obs_max`` are stored in the "
             "checkpoint blob and the DP adapter applies the same "
             "affine map at inference so the model sees an "
             "in-distribution conditioning vector.  Default OFF "
             "(legacy z-score path; Phase-1/2/2c/2d/3 reproducibility).",
    )
    return p.parse_args()


def cosine_lr(
    epoch: int,
    total_epochs: int,
    base_lr: float,
    warmup: int,
    min_lr: float = 0.0,
) -> float:
    """Linear warmup then cosine cooldown with an optional ``min_lr`` floor.

    Without a floor the schedule anneals to ~0 in the final epochs,
    which we found makes the EMA shadow a no-op (weights stop moving
    before the EMA averaging window kicks in) and contributes to the
    over-fitting failure mode documented in
    ``docs/experiments/17_dp_hybrid.md`` §3.1.  Passing
    ``min_lr=1e-6`` (DP-paper convention) keeps weights productively
    moving in the cooldown phase so EMA actually does smoothing.
    """
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))
    return float(max(cosine, min_lr))


class EMAModel:
    """Polyak / exponential moving average of model parameters with
    Cheng Chi's power-decay warmup schedule.

    Per-step update:

        theta_ema <- decay_t * theta_ema + (1 - decay_t) * theta

    where ``decay_t`` follows the DP paper recipe from
    ``diffusion_policy/model/diffusion/ema_model.py``::

        decay_t = clip(
            1 - (1 + (step - update_after_step) / inv_gamma) ** -power,
            min_decay, max_decay,
        )

    With the published defaults ``inv_gamma=1.0, power=0.75,
    max_decay=0.9999``, the decay grows smoothly::

        step =      0  ->  0.000   (shadow == raw, EMA disabled)
        step =     10  ->  0.822
        step =    100  ->  0.968
        step =   1000  ->  0.994
        step =  10000  ->  0.9989
        step = 100000  ->  0.99982  (clipped to max_decay=0.9999)

    This is critical when training from scratch: a fixed ``decay=0.9999``
    from step 0 leaves the EMA dominated by the *random* initial weights
    for the first ~10k steps, producing a useless inference checkpoint
    for short runs.  The power schedule solves this by accepting most
    of the early updates verbatim, then progressively smoothing once
    the weights have left the chaotic warmup region.

    Reference: diffusion_policy/model/diffusion/ema_model.py (Cheng
    Chi 2023), itself derived from k-diffusion / EDM.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 0.75,
        min_decay: float = 0.0,
    ):
        self.max_decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.min_decay = float(min_decay)
        self.step = 0

        # Shadow copy of every learnable parameter.  Buffers (e.g. GroupNorm
        # running stats) are kept on the raw model -- our ConditionalUnet1D
        # has no running buffers anyway (GroupNorm in eval mode is stateless).
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    # Backwards-compat alias: callers reading ``ema.decay`` get the
    # asymptotic max decay (= what they passed in as ``decay`` kwarg).
    @property
    def decay(self) -> float:
        return self.max_decay

    def get_decay(self, step: int) -> float:
        """DP-paper power-decay schedule for the current step."""
        step = max(0, step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        value = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        return float(max(self.min_decay, min(self.max_decay, value)))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.step += 1
        decay = self.get_decay(self.step)
        if decay <= 0.0:
            # Early warmup: shadow follows raw weights exactly.
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].copy_(param.detach())
            return
        one_minus = 1.0 - decay
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(decay).add_(
                    param.detach(), alpha=one_minus,
                )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: t.detach().clone() for name, t in self.shadow.items()}

    def copy_to(self, model: torch.nn.Module) -> None:
        """Overwrite ``model``'s parameters with the EMA shadow weights."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    param.copy_(self.shadow[name])


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.n_demos = 3
        args.epochs = 4
        args.diffusion_step_embed_dim = 64
        args.down_dims = [64, 128, 256]
        args.batch_size = 32

    if int(args.n_obs_steps) < 1:
        raise SystemExit(
            f"--n-obs-steps must be >= 1, got {args.n_obs_steps}"
        )

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
        f"[exp08-dp] loaded {bundle.n_demos} demos -> "
        f"{bundle.n_chunks} chunks  "
        f"X={bundle.X.shape}  S={bundle.S_norm.shape}  "
        f"mean_demo_len={bundle.demo_lens.mean():.1f}  "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    X_raw = bundle.X.astype(np.float32)                   # (N, H, A) env-space
    if bool(args.normalize_actions):
        X = bundle.normalize_actions(X_raw)               # (N, H, A) in [-1, 1]
        print(
            f"[exp08-dp] action normalisation: ON  "
            f"action_min={np.round(bundle.action_min, 3).tolist()}  "
            f"action_max={np.round(bundle.action_max, 3).tolist()}  "
            f"X_norm range=[{X.min():+.3f}, {X.max():+.3f}]"
        )
    else:
        X = X_raw
        print(
            "[exp08-dp] action normalisation: OFF "
            "(--no-normalize-actions; legacy Phase-1/2/2c path)"
        )
    # Obs conditioning branch (DP paper recipe, Finding C; 2026-05-28).
    # ``--normalize-obs`` swaps the legacy z-score path for a min-max
    # [-1, 1] affine map, matching DP's LinearNormalizer.fit(mode='limits').
    # ``obs_min`` / ``obs_max`` are saved alongside ``obs_mean`` / ``obs_std``
    # in the ckpt blob; the adapter reads ``normalize_obs`` to pick the
    # matching inference-time normalisation.
    if bool(args.normalize_obs):
        S = bundle.S_norm_minmax.astype(np.float32)       # (N, D_obs) in [-1, 1]
        print(
            f"[exp08-dp] obs    normalisation: ON (min-max)  "
            f"obs_min=[{bundle.obs_min.min():.3f}..{bundle.obs_min.max():.3f}] "
            f"obs_max=[{bundle.obs_max.min():.3f}..{bundle.obs_max.max():.3f}]  "
            f"S range=[{S.min():+.3f}, {S.max():+.3f}]"
        )
    else:
        S = bundle.S_norm.astype(np.float32)              # (N, D_obs) z-scored
        print(
            "[exp08-dp] obs    normalisation: OFF (z-score) "
            "(--no-normalize-obs; legacy Phase-1/2/2c/2d/3 path)"
        )

    # Stack obs history when T_obs > 1.  ``stack_obs_history`` is the
    # identity for T_obs=1 so this is a no-op in the standard path.
    if int(args.n_obs_steps) > 1:
        S_cond = stack_obs_history(S, bundle.demo_lens, int(args.n_obs_steps))
        print(
            f"[exp08-dp] stacked obs history: n_obs_steps={args.n_obs_steps}  "
            f"S_cond shape {S_cond.shape} (was {S.shape})"
        )
    else:
        S_cond = S

    config = FlowConfigDP(
        obs_dim=int(S.shape[1]),
        action_dim=int(bundle.action_dim),
        horizon=int(args.horizon),
        n_obs_steps=int(args.n_obs_steps),
        diffusion_step_embed_dim=int(args.diffusion_step_embed_dim),
        down_dims=tuple(int(d) for d in args.down_dims),
        kernel_size=int(args.kernel_size),
        n_groups=int(args.n_groups),
        cond_predict_scale=not bool(args.no_cond_predict_scale),
        num_train_timesteps=int(args.num_train_timesteps),
        beta_schedule=str(args.beta_schedule),
    )
    device = torch.device(args.device)
    model = ConditionalUnet1D(config).to(device)
    schedule = DDPMSchedule(
        config.num_train_timesteps,
        schedule=config.beta_schedule,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[exp08-dp] arch=dp_unet1d  params={n_params:,}  "
        f"down_dims={config.down_dims}  T_DDPM={config.num_train_timesteps}  "
        f"beta={config.beta_schedule}  config={config.to_dict()}"
    )

    # The DP model wants (B, T, A); we keep the flat tensor on disk and
    # reshape inside dp_loss so the dataset is a drop-in with v3/v6.
    X_flat = X.reshape(X.shape[0], -1)

    # ---- per-demo train/val split ----
    # ``demo_lens`` is in demo order; chunks were concatenated in the
    # same order by ``load_chunks``.  So a chunk's demo_id is
    # repeat(arange(M), demo_lens).  Splitting BY demo (not by chunk)
    # avoids adjacent-chunk leakage across train/val.
    val_frac = float(args.val_frac)
    if val_frac > 0.0:
        demo_lens_arr = np.asarray(bundle.demo_lens, dtype=np.int64)
        chunk_demo_id = np.repeat(
            np.arange(len(demo_lens_arr), dtype=np.int64),
            demo_lens_arr,
        )
        if chunk_demo_id.shape[0] != X_flat.shape[0]:
            raise RuntimeError(
                f"[exp08-dp] demo-id mismatch: "
                f"{chunk_demo_id.shape[0]} vs {X_flat.shape[0]} chunks"
            )
        rng = np.random.default_rng(int(args.seed))
        demo_ids_shuffled = rng.permutation(len(demo_lens_arr))
        n_val_demos = max(1, int(round(val_frac * len(demo_lens_arr))))
        val_demo_set = set(demo_ids_shuffled[:n_val_demos].tolist())
        val_mask = np.isin(
            chunk_demo_id, np.fromiter(val_demo_set, dtype=np.int64),
        )
        train_mask = ~val_mask
        X_train = X_flat[train_mask]
        S_train = S_cond[train_mask]
        X_val = X_flat[val_mask]
        S_val = S_cond[val_mask]
        print(
            f"[exp08-dp] val split: {len(val_demo_set)}/{len(demo_lens_arr)} "
            f"demos held out -> n_train={int(train_mask.sum())}  "
            f"n_val={int(val_mask.sum())}"
        )
        ds_train = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(S_train),
        )
        ds_val = TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(S_val),
        )
        val_loader = DataLoader(
            ds_val,
            batch_size=min(int(args.batch_size), X_val.shape[0]),
            shuffle=False,
            drop_last=False,
        )
    else:
        ds_train = TensorDataset(
            torch.from_numpy(X_flat), torch.from_numpy(S_cond),
        )
        val_loader = None

    loader = DataLoader(
        ds_train,
        batch_size=min(int(args.batch_size), len(ds_train)),
        shuffle=True,
        drop_last=False,
    )
    print(
        f"[exp08-dp] dataloader: n={len(ds_train)} chunks  "
        f"batch_size={loader.batch_size}  "
        f"n_batches/epoch={len(loader)}"
    )

    if int(args.early_stop_patience) > 0 and val_loader is None:
        raise SystemExit(
            "--early-stop-patience requires --val-frac > 0 "
            "(no val signal otherwise)"
        )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(float(args.adam_beta1), float(args.adam_beta2)),
    )
    print(
        f"[exp08-dp] AdamW: lr={float(args.lr):.2e}  "
        f"betas=({float(args.adam_beta1)}, {float(args.adam_beta2)})  "
        f"weight_decay={float(args.weight_decay):.2e}"
    )

    # EMA shadow weights (DP paper recipe).  Initialised right after the
    # raw model so step 0 of training has shadow = current weights.
    ema: EMAModel | None = None
    if bool(args.use_ema):
        ema = EMAModel(model, decay=float(args.ema_decay))
        print(
            f"[exp08-dp] EMA enabled: decay={ema.decay}  "
            f"shadow_params={sum(t.numel() for t in ema.shadow.values()):,}"
        )
    else:
        print("[exp08-dp] EMA disabled (--no-use-ema)")

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if use_amp
        else torch.amp.autocast("cpu", enabled=False)
    )
    print(f"[exp08-dp] device={device}  amp={use_amp}")

    print_every = (
        args.print_every if args.print_every > 0
        else max(args.epochs // 20, 1)
    )

    grad_clip = (
        float(args.grad_clip)
        if args.grad_clip and args.grad_clip > 0
        else None
    )
    print(f"[exp08-dp] grad_clip={grad_clip}")

    log: list[dict] = []
    wall0 = time.perf_counter()
    best_val_loss = float("inf")
    best_val_epoch = -1
    best_val_state: dict[str, torch.Tensor] | None = None
    epochs_since_improvement = 0

    def _eval_val_loss() -> float:
        """Compute val loss using the EMA shadow weights (what we deploy).

        Strategy: stash raw weights, copy EMA shadow into the model,
        run val pass, restore raw weights.  Done in no-grad mode so
        memory is bounded by the largest batch.
        """
        if val_loader is None:
            return float("nan")
        # Backup raw weights to CPU staging dict so we don't bloat
        # GPU memory with a duplicate.
        if ema is not None:
            raw_backup = {
                n: p.detach().clone()
                for n, p in model.named_parameters()
            }
            ema.copy_to(model)
        try:
            model.eval()
            losses: list[float] = []
            with torch.no_grad():
                for x_b, s_b in val_loader:
                    x_b = x_b.to(device)
                    s_b = s_b.to(device)
                    with amp_ctx:
                        vloss = dp_loss(model, schedule, x_b, s_b)
                    losses.append(float(vloss.detach().cpu()))
            return float(np.mean(losses)) if losses else float("nan")
        finally:
            if ema is not None:
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if n in raw_backup:
                            p.copy_(raw_backup[n])
            model.train()

    def _save_intermediate_ckpt(epoch: int) -> Path:
        """Save a snapshot ckpt at the current epoch using EMA shadow."""
        if ema is not None:
            sd_to_save = {k: v.cpu() for k, v in ema.state_dict().items()}
            sd_raw = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            sd_to_save = {k: v.cpu() for k, v in model.state_dict().items()}
            sd_raw = None
        snap = {
            "format_version": 7,
            "arch": "dp_unet1d",
            "config": config.to_dict(),
            "state_dict": sd_to_save,
            "obs_mean": bundle.obs_mean.astype(np.float32),
            "obs_std": bundle.obs_std.astype(np.float32),
            "obs_min": bundle.obs_min.astype(np.float32),
            "obs_max": bundle.obs_max.astype(np.float32),
            "normalize_obs": bool(args.normalize_obs),
            "action_min": bundle.action_min.astype(np.float32),
            "action_max": bundle.action_max.astype(np.float32),
            "normalize_actions": bool(args.normalize_actions),
            "n_demos": int(bundle.n_demos),
            "n_chunks": int(bundle.n_chunks),
            "horizon": int(args.horizon),
            "task": str(args.task),
            "wall_s": float(time.perf_counter() - wall0),
            "epoch": int(epoch),
            "use_ema": bool(ema is not None),
        }
        if ema is not None:
            snap["ema_decay"] = float(ema.decay)
            snap["state_dict_raw"] = sd_raw
        out_path = args.output.parent / (
            f"{args.output.stem}_ep{epoch:04d}{args.output.suffix}"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(snap, out_path)
        return out_path

    for epoch in range(int(args.epochs)):
        lr = cosine_lr(
            epoch,
            int(args.epochs),
            float(args.lr),
            int(args.warmup_epochs),
            min_lr=float(args.min_lr),
        )
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
                loss = dp_loss(model, schedule, x_batch, s_batch)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[exp08-dp] non-finite loss at epoch={epoch} "
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
            # EMA shadow update happens once per optimisation step,
            # AFTER the optimiser has stepped the raw weights.  Both
            # AMP and non-AMP paths share this.
            if ema is not None:
                ema.update(model)
            if grad_clip is not None:
                max_grad_norm = max(max_grad_norm, g_norm)
                if g_norm > grad_clip:
                    n_clipped += 1
            epoch_losses.append(float(loss.detach().cpu()))
        avg = float(np.mean(epoch_losses))

        # ---- val pass (uses EMA shadow weights) ----
        val_loss = _eval_val_loss() if val_loader is not None else float("nan")
        improved = False
        if val_loader is not None and math.isfinite(val_loss):
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = int(epoch)
                # Cache the EMA shadow at this epoch in CPU memory so we
                # can write it out at the end.  ~250-500 MB depending on
                # model size; on Colab with 16 GB CPU RAM this is fine.
                if ema is not None:
                    best_val_state = {
                        k: v.detach().cpu().clone()
                        for k, v in ema.state_dict().items()
                    }
                else:
                    best_val_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                epochs_since_improvement = 0
                improved = True
            else:
                epochs_since_improvement += 1

        log.append({
            "epoch": epoch, "loss": avg, "lr": lr,
            "max_grad_norm": float(max_grad_norm),
            "n_clipped": int(n_clipped),
            "val_loss": val_loss,
            "improved": bool(improved),
        })
        if epoch % print_every == 0 or epoch == int(args.epochs) - 1:
            val_str = f"  val={val_loss:.4f}" if val_loader is not None else ""
            best_str = (
                f"  best_val={best_val_loss:.4f}@ep{best_val_epoch}"
                if val_loader is not None and math.isfinite(best_val_loss)
                else ""
            )
            print(
                f"[exp08-dp] epoch {epoch:>5}/{args.epochs}  "
                f"loss={avg:.4f}{val_str}{best_str}  lr={lr:.2e}  "
                f"max_gn={max_grad_norm:.2f}  clipped={n_clipped}/{len(loader)}",
                flush=True,
            )
            # Also flush the log file so monitoring cells can read it
            # incrementally without waiting for end-of-training.  Costs
            # ~1ms; entirely negligible vs the epoch cost.
            args.log.parent.mkdir(parents=True, exist_ok=True)
            args.log.write_text(json.dumps(log, indent=2))

        # ---- intermediate ckpt save ----
        save_every = int(args.save_every)
        if save_every > 0 and (epoch + 1) % save_every == 0 \
                and (epoch + 1) < int(args.epochs):
            snap_path = _save_intermediate_ckpt(epoch)
            print(
                f"[exp08-dp]   -> saved intermediate ckpt: {snap_path.name}",
                flush=True,
            )

        # ---- early stopping ----
        patience = int(args.early_stop_patience)
        if patience > 0 and val_loader is not None \
                and epochs_since_improvement >= patience:
            print(
                f"[exp08-dp] early-stop: val loss has not improved "
                f"for {patience} epochs (best={best_val_loss:.4f} "
                f"@ep{best_val_epoch}, current={val_loss:.4f}@ep{epoch})",
                flush=True,
            )
            break
    wall = time.perf_counter() - wall0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # When EMA is enabled, the *primary* ``state_dict`` we write to
    # disk is the EMA-averaged copy -- that's what ``adapter_dp.py``
    # loads at inference and what gives the published DP numbers.
    # The raw last-iterate weights are kept under ``state_dict_raw``
    # for reproducibility and to allow A/B comparisons if needed.
    if ema is not None:
        state_dict_ema = ema.state_dict()
        # Ensure all tensors are on CPU before serialising so the
        # checkpoint is device-agnostic.
        state_dict_to_save = {k: v.cpu() for k, v in state_dict_ema.items()}
        state_dict_raw = {k: v.cpu() for k, v in model.state_dict().items()}
        print(
            f"[exp08-dp] saving EMA weights as primary state_dict "
            f"(decay={ema.decay}); raw weights under 'state_dict_raw'"
        )
    else:
        state_dict_to_save = {k: v.cpu() for k, v in model.state_dict().items()}
        state_dict_raw = None

    blob = {
        "format_version": 7,
        "arch": "dp_unet1d",
        "config": config.to_dict(),
        "state_dict": state_dict_to_save,
        "obs_mean": bundle.obs_mean.astype(np.float32),
        "obs_std": bundle.obs_std.astype(np.float32),
        "obs_min": bundle.obs_min.astype(np.float32),
        "obs_max": bundle.obs_max.astype(np.float32),
        "normalize_obs": bool(args.normalize_obs),
        "action_min": bundle.action_min.astype(np.float32),
        "action_max": bundle.action_max.astype(np.float32),
        "normalize_actions": bool(args.normalize_actions),
        "n_demos": int(bundle.n_demos),
        "n_chunks": int(bundle.n_chunks),
        "horizon": int(args.horizon),
        "task": str(args.task),
        "wall_s": float(wall),
        "use_ema": bool(ema is not None),
    }
    if ema is not None:
        blob["ema_decay"] = float(ema.decay)
        blob["state_dict_raw"] = state_dict_raw
    if val_loader is not None and math.isfinite(best_val_loss):
        blob["best_val_loss"] = float(best_val_loss)
        blob["best_val_epoch"] = int(best_val_epoch)
    torch.save(blob, args.output)

    # Also save the best-val ckpt separately (with _best_val suffix) so
    # downstream eval scripts can pick it without rerunning.  If the
    # final epoch IS the best, this is just a duplicate but it makes the
    # naming pattern uniform.
    if best_val_state is not None:
        best_blob = {**blob, "state_dict": best_val_state,
                     "epoch": int(best_val_epoch)}
        if ema is not None and "state_dict_raw" in best_blob:
            # We don't have the raw weights from the best-val epoch;
            # only the EMA shadow.  Drop the raw entry to avoid
            # confusion.
            best_blob.pop("state_dict_raw", None)
        best_path = args.output.parent / (
            f"{args.output.stem}_best_val{args.output.suffix}"
        )
        torch.save(best_blob, best_path)
        print(
            f"[exp08-dp] best-val ckpt: {best_path.name}  "
            f"val_loss={best_val_loss:.4f}@ep{best_val_epoch}",
            flush=True,
        )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(log, indent=2))
    print(
        f"[exp08-dp] wrote {args.output}  "
        f"final_loss={log[-1]['loss']:.4f}  wall={wall:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
