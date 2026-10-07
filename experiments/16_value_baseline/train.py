"""Train V_phi: cumulative-cost-to-go regression on BC-policy rollouts.

Procedure (mirrors the spec in
``feat/stronger-mbd-signal/docs/branch_plan.md`` Task A2):

1. Load the canonical BC policy from
   ``experiments/03_bc_baseline/results/policy.pt``.
2. Roll out the BC policy N times (default 500) on the live
   robosuite env, recording per-step ``(s_t, c_t)`` where ``s_t`` is
   the OBS_KEYS-ordered observation vector and ``c_t = -reward_t``.
3. Compute the empirical cost-to-go ``R(s_t) = sum_{t' >= t} c_{t'}``
   for every transition.
4. Fit ``V_phi(s) ≈ R(s)`` via Adam MSE regression. Reuse the BC
   policy's ``(obs_mean, obs_std)`` for normalisation so the
   feature axes match the BC checkpoint exactly.
5. Save V_phi to ``--output`` (default
   ``experiments/16_value_baseline/results/V_phi.pt``).

Acceptance gate (see branch_plan.md): final TD residual / mean(R)
ratio should be < 0.5; otherwise the value head hasn't learned
anything useful. Logged at the end of training.

Open questions documented in the branch plan (mitigations applied
here):
  - 500 trajectories may be too few -> ``--n-trajs`` defaults to
    500 but can be raised to 2000 in ~1 h.
  - BC rollouts may not cover the state-action manifold MBD
    explores -> ``--action-noise`` adds tanh-Gaussian noise to BC
    actions during data collection. Default 0.0; set 0.1 for
    coverage if pure-MBD eval (downstream) plateaus.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# experiments/ on sys.path for the digit-prefixed exp modules
_THIS_DIR = Path(__file__).resolve().parent
_EXP_DIR = _THIS_DIR.parent
for _p in (_THIS_DIR, _EXP_DIR, _EXP_DIR / "03_bc_baseline", _EXP_DIR / "04_mbd_baseline"):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import h5py  # type: ignore  # noqa: E402

from common import (  # type: ignore  # noqa: E402
    check_success,
    default_dataset_path,
    make_env_for_dataset,
    set_sim_state,
)
from dataset import OBS_KEYS, ObsStats  # type: ignore  # noqa: E402
from policy import Policy  # type: ignore  # noqa: E402

from value_model import MLPValue, ValueConfig, ValuePhi  # type: ignore  # noqa: E402


def _obs_dict_to_vec(obs_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Concat the OBS_KEYS-ordered observation into a flat vector."""
    parts = []
    for k in OBS_KEYS:
        if k not in obs_dict:
            raise KeyError(f"obs missing {k!r}; have {sorted(obs_dict.keys())}")
        parts.append(np.asarray(obs_dict[k], dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=-1)


def _rollout_bc_one(
    env,
    init_state: np.ndarray,
    policy: Policy,
    horizon: int,
    rng: np.random.Generator,
    action_noise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """One open-loop BC rollout. Returns ``(obs_traj, cost_traj, success)``.

    ``obs_traj`` shape ``(horizon, obs_dim)`` -- the obs *before* each
    action (last frame's post-action obs is not collected; V_phi is
    queried at the H planning indices, not at H+1 terminal indices).
    ``cost_traj`` shape ``(horizon,)`` is ``-reward`` per step.
    """
    env.reset()
    set_sim_state(env, init_state)

    obs_buf: list[np.ndarray] = []
    cost_buf: list[float] = []

    for _ in range(horizon):
        obs_dict = env._get_observations(force_update=True)
        obs_vec = _obs_dict_to_vec(obs_dict)
        obs_buf.append(obs_vec)

        action = policy.act(obs_vec)
        if action_noise > 0.0:
            action = action + action_noise * rng.standard_normal(action.shape)
            action = np.clip(action, -1.0, 1.0)
        action = action.astype(np.float32)

        _next_obs, reward, _done, _info = env.step(action)
        cost_buf.append(-float(reward))

    return (
        np.asarray(obs_buf, dtype=np.float32),
        np.asarray(cost_buf, dtype=np.float32),
        bool(check_success(env)),
    )


def collect_dataset(
    *,
    bc_checkpoint: Path,
    dataset_path: Path,
    n_trajs: int,
    horizon: int,
    seed: int,
    action_noise: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run BC and accumulate (obs, cost-to-go) pairs.

    Returns:
        obs:      (N, obs_dim) float32. N = n_trajs * horizon.
        ctg:      (N,) float32. Cumulative cost-to-go.
        meta:     dict for logging.
    """
    print(f"[16/train] BC checkpoint: {bc_checkpoint}")
    policy = Policy.from_checkpoint(bc_checkpoint, device="cpu")

    print(f"[16/train] making env from {dataset_path}")
    env, _env_meta = make_env_for_dataset(dataset_path)

    # Load per-demo initial sim states from the HDF5 (matches the
    # pattern in exp 04's run_can.py).
    with h5py.File(str(dataset_path), "r") as h5:
        demo_keys = sorted(
            h5["data"].keys(), key=lambda k: int(k.split("_")[1])
        )
        init_states = [
            np.asarray(h5[f"data/{k}/states"][0], dtype=np.float64)
            for k in demo_keys
        ]
    print(
        f"[16/train] loaded {len(init_states)} demo init-states from {dataset_path}"
    )
    rng = np.random.default_rng(seed)

    all_obs: list[np.ndarray] = []
    all_ctg: list[np.ndarray] = []
    n_success = 0

    t_start = time.perf_counter()
    for i in range(n_trajs):
        init_state = init_states[i % len(init_states)]
        obs_traj, cost_traj, success = _rollout_bc_one(
            env, init_state, policy, horizon, rng, action_noise=action_noise,
        )
        n_success += int(success)
        # cost-to-go via reverse-cumsum: R[t] = sum_{t'>=t} c[t']
        ctg = np.flip(np.cumsum(np.flip(cost_traj))).astype(np.float32)
        all_obs.append(obs_traj)
        all_ctg.append(ctg)
        if (i + 1) % 50 == 0 or i == n_trajs - 1:
            elapsed = time.perf_counter() - t_start
            avg = elapsed / (i + 1)
            eta = avg * (n_trajs - i - 1)
            print(
                f"[16/train]   rollouts {i + 1}/{n_trajs}  "
                f"success={n_success}  "
                f"elapsed={elapsed:.1f}s  eta={eta:.0f}s",
                flush=True,
            )

    obs_all = np.concatenate(all_obs, axis=0).astype(np.float32)
    ctg_all = np.concatenate(all_ctg, axis=0).astype(np.float32)

    meta = {
        "n_trajs": int(n_trajs),
        "horizon": int(horizon),
        "n_transitions": int(obs_all.shape[0]),
        "n_success": int(n_success),
        "success_rate": float(n_success / max(n_trajs, 1)),
        "ctg_mean": float(ctg_all.mean()),
        "ctg_std": float(ctg_all.std()),
        "ctg_min": float(ctg_all.min()),
        "ctg_max": float(ctg_all.max()),
        "action_noise": float(action_noise),
        "dataset_path": str(dataset_path),
        "bc_checkpoint": str(bc_checkpoint),
        "seed": int(seed),
    }
    return obs_all, ctg_all, meta


def train(
    obs: np.ndarray,
    ctg: np.ndarray,
    stats: ObsStats,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str = "cpu",
) -> tuple[MLPValue, dict[str, Any]]:
    """Fit V_phi(s) ≈ R(s) via MSE regression."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    obs_norm = stats.normalize(obs).astype(np.float32)
    N, obs_dim = obs_norm.shape

    config = ValueConfig(obs_dim=int(obs_dim))
    net = MLPValue(config).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x = torch.from_numpy(obs_norm).to(device)
    y = torch.from_numpy(ctg.astype(np.float32)).to(device)

    losses: list[float] = []
    t_start = time.perf_counter()
    for epoch in range(epochs):
        perm = rng.permutation(N)
        epoch_loss_sum = 0.0
        epoch_n_samples = 0
        for s in range(0, N, batch_size):
            idx = perm[s:s + batch_size]
            xb = x[idx]
            yb = y[idx]
            pred = net(xb)
            loss = criterion(pred, yb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss_sum += float(loss.item()) * len(idx)
            epoch_n_samples += len(idx)
        epoch_loss = epoch_loss_sum / max(epoch_n_samples, 1)
        losses.append(epoch_loss)
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            elapsed = time.perf_counter() - t_start
            print(
                f"[16/train]   epoch {epoch + 1:4d}/{epochs}  "
                f"loss={epoch_loss:.4f}  elapsed={elapsed:.1f}s",
                flush=True,
            )

    # Final TD-residual diagnostic.
    with torch.no_grad():
        pred_all = net(x).cpu().numpy()
    resid = pred_all - ctg
    rmse = float(np.sqrt((resid ** 2).mean()))
    mae = float(np.abs(resid).mean())
    rmse_over_mean = rmse / max(abs(ctg.mean()), 1e-6)

    print(
        f"[16/train] final RMSE={rmse:.4f}  MAE={mae:.4f}  "
        f"RMSE/|mean(R)|={rmse_over_mean:.3f}",
        flush=True,
    )

    train_meta = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "final_loss": float(losses[-1]) if losses else float("nan"),
        "rmse": rmse,
        "mae": mae,
        "rmse_over_abs_mean_R": rmse_over_mean,
        "seed_torch": int(seed),
        "device": str(device),
    }
    return net, train_meta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--bc-checkpoint", type=Path,
        default=Path("experiments/03_bc_baseline/results/policy.pt"),
    )
    p.add_argument(
        "--dataset", type=Path,
        default=default_dataset_path("can"),
    )
    p.add_argument("--n-trajs", type=int, default=500)
    p.add_argument("--horizon", type=int, default=120)
    p.add_argument("--action-noise", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--output", type=Path,
        default=Path("experiments/16_value_baseline/results/V_phi.pt"),
    )
    args = p.parse_args()

    print(f"[16/train] seed={args.seed}  n_trajs={args.n_trajs}  "
          f"horizon={args.horizon}  action_noise={args.action_noise}")

    obs, ctg, data_meta = collect_dataset(
        bc_checkpoint=args.bc_checkpoint,
        dataset_path=args.dataset,
        n_trajs=args.n_trajs,
        horizon=args.horizon,
        seed=args.seed,
        action_noise=args.action_noise,
    )
    print(
        f"[16/train] collected {data_meta['n_transitions']} transitions  "
        f"R(s) ~ N({data_meta['ctg_mean']:.3f}, {data_meta['ctg_std']:.3f})  "
        f"BC success {data_meta['n_success']}/{data_meta['n_trajs']}"
    )

    # Reuse BC's ObsStats so V_phi and policy share normalisation.
    bc_policy = Policy.from_checkpoint(args.bc_checkpoint, device="cpu")
    stats = bc_policy.stats

    net, train_meta = train(
        obs, ctg, stats,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )

    extra = {**data_meta, **train_meta}
    ValuePhi.save(args.output, net, stats, extra=extra)
    summary_path = args.output.with_suffix(".meta.json")
    with summary_path.open("w") as f:
        json.dump(extra, f, indent=2, default=str)

    print(f"[16/train] V_phi saved to {args.output}")
    print(f"[16/train] meta saved to {summary_path}")

    # Diagnostic gate: if RMSE/|mean(R)| > 0.5 the regression hasn't
    # fit the cost-to-go reliably; flag for follow-up.
    if train_meta["rmse_over_abs_mean_R"] > 0.5:
        print(
            f"[16/train] WARN: RMSE/|mean(R)| = "
            f"{train_meta['rmse_over_abs_mean_R']:.3f} > 0.5; "
            f"consider --n-trajs 2000 or different normalisation."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
