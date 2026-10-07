"""Experiment 05: Hybrid BC prior + online MBD on PickPlaceCan.

This is the "our method" row of the Week-4 results table. Conceptually
trivial: take experiment 04's MBD denoising loop and feed it a BC
warm-start instead of a zero initial mean. The BC policy comes from
experiment 03 (``experiments/03_bc_baseline/results/policy.pt``);
everything else (cost, env, schedule, success metric) is reused
verbatim from ``experiments/04_mbd_baseline/run_can.py``.

Pipeline per (demo, rollout):

1. Reset the live env to the demo's recorded ``init_state`` and roll
   the BC policy closed-loop for ``H`` steps. The resulting action
   sequence ``a_bc[0..H-1]`` is the warm-start trajectory.
2. Call ``plan_and_execute(..., initial_mean=a_bc)`` from
   ``experiments/04_mbd_baseline/run_can.py``. The MBD denoising loop
   now perturbs around ``a_bc`` instead of around zero.
3. Execute the final denoised mean on the live env and record
   ``check_success`` for the results JSON, matching the protocol of
   experiments 02 / 03 / 04 (20 demos x 3 rollouts = 60).

Why this should work (and the story the report wants this row to tell):
the BC policy from exp 03 reaches the can on ~70% of starts but only
grasps + lifts on ~7%; conversely, MBD-no-prior (exp 04) reached the
can on only ~30% of starts and never grasped from cold init. By
warm-starting MBD with a BC rollout, the planner's perturbations now
explore *around* a trajectory that already reaches the can, so the
optimisation budget is spent on the grasp / lift / place segment
instead of on discovering how to reach.

CLI flags mirror exp 04 with two additions:

  - ``--bc-checkpoint``: path to the policy.pt produced by exp 03.
  - ``--no-warm-start``: disable BC warm-start (initial_mean=0). This
    is the ablation that should reproduce the exp 04 baseline; useful
    as a self-check that running the same MBD math through this script
    matches exp 04's 0/60 within seed noise.

Outputs:

  - ``results/can_ph.json``: the headline 20x3 evaluation.
  - ``results/can_ph.png``: aggregated cost-vs-iteration plot.
  - ``results/videos/``: per-(demo, rollout) MP4s of the executed
    ``mu_final`` trajectory; gitignored.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np


_THIS_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _THIS_DIR.parent
_EXP_03 = _EXPERIMENTS_DIR / "03_bc_baseline"
_EXP_04 = _EXPERIMENTS_DIR / "04_mbd_baseline"

# Sibling experiment directories start with a digit, so they can't be
# imported via dotted-package syntax. Mirror the sys.path pattern used
# by the exp 03 / 04 entry scripts so plain ``import policy`` etc work.
# Order matters: experiments/ shared dir + exp 03 + exp 04 come *before*
# this script's own directory, otherwise ``import run_can`` would
# resolve to this file (since exp 04 and exp 05 both define run_can).
for _p in (_EXPERIMENTS_DIR, _EXP_03, _EXP_04, _THIS_DIR):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from common import (  # noqa: E402
    check_success,
    default_dataset_path,
    get_action_bounds,
    make_env_for_dataset,
    set_sim_state,
)
from mbd import (  # noqa: E402  -- experiments/04_mbd_baseline/mbd.py
    MBDParams,
    cosine_sigma_schedule,
    denoise_loop,
)

# experiments/04_mbd_baseline/run_can.py is the source of truth for the
# rollout / cost / live-view machinery. Load it explicitly by file path
# so there's no risk of it being shadowed by exp 05's own run_can.py
# (both files share the basename).
import importlib.util as _importlib_util  # noqa: E402

_spec = _importlib_util.spec_from_file_location(
    "exp04_run_can", _EXP_04 / "run_can.py"
)
exp04_run = _importlib_util.module_from_spec(_spec)
sys.modules["exp04_run_can"] = exp04_run
_spec.loader.exec_module(exp04_run)

# experiments/03_bc_baseline/policy.py + dataset.py.
from policy import Policy  # noqa: E402  -- experiments/03_bc_baseline/policy.py
from dataset import OBS_KEYS  # noqa: E402  -- experiments/03_bc_baseline/dataset.py


# ---------------------------------------------------------------------------
# Live-env obs -> training-key remap (duplicated from exp 03's eval.py
# rather than imported, because exp 03/eval.py also defines a CLI parser
# at module top that we don't want to drag in).
# ---------------------------------------------------------------------------


_RUNTIME_KEY_REMAP: dict[str, str] = {}


def _get_runtime_obs(
    env_obs: dict,
    obs_keys: tuple[str, ...] | None = None,
) -> dict:
    """Translate the live env's obs dict into the dataset's key space.

    ``obs_keys`` defaults to ``None`` -> Can's legacy ``OBS_KEYS`` for
    backward compatibility with the historical BC checkpoints.  Pass an
    explicit tuple (from ``policy.config.obs_keys``) for cross-task BC.
    """
    if obs_keys is None:
        obs_keys = OBS_KEYS
    out: dict = {}
    for k in obs_keys:
        runtime_k = _RUNTIME_KEY_REMAP.get(k, k)
        if runtime_k not in env_obs:
            raise KeyError(
                f"runtime obs missing {runtime_k!r} (dataset key {k!r}); "
                f"available: {sorted(env_obs.keys())}"
            )
        out[k] = np.asarray(env_obs[runtime_k], dtype=np.float32)
    return out


def _concat_obs(
    obs: dict,
    obs_keys: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Concatenate the obs dict's values in ``obs_keys`` order.

    ``obs_keys`` defaults to Can's legacy ``OBS_KEYS`` for backward
    compatibility; pass an explicit tuple for cross-task BC.
    """
    if obs_keys is None:
        obs_keys = OBS_KEYS
    return np.concatenate([obs[k] for k in obs_keys], axis=-1)


# ---------------------------------------------------------------------------
# BC warm-start rollout
# ---------------------------------------------------------------------------


def _deterministic_env_reset(env, init_state: np.ndarray, seed: int) -> None:
    """Reset robosuite env in a way that's invariant to call history.

    Pre-fix, ``env.reset()`` advanced robosuite's internal RNG and the
    placement initializer, which changed the post-``set_sim_state``
    observable state in subtle ways depending on how many resets had
    happened earlier in the run. That's the documented cause of the
    BC-alone "0/60 vs 5/60" discrepancy between Exp 05 single-shot
    and Exp 05b MPC at identical inputs (see Exp 05 README, "Known
    anomaly"). Re-seeding numpy + robosuite right before each
    reset makes the reset deterministic given (init_state, seed),
    which is what we want for an apples-to-apples BC-alone arm.
    """
    np.random.seed(int(seed) & 0xFFFFFFFF)
    if hasattr(env, "seed"):
        try:
            env.seed(int(seed) & 0xFFFFFFFF)
        except Exception:
            pass
    env.reset()
    set_sim_state(env, init_state)


def bc_warmstart_actions(
    env,
    init_state: np.ndarray,
    policy: Policy,
    H: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    reset_seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Roll the BC policy closed-loop for ``H`` steps and return its actions.

    Stepping is closed-loop on the live env (so each obs reflects the
    BC policy's previous action), but we *return* the action sequence
    so it can be replayed open-loop later by MBD. This mirrors the
    "deterministic BC rollout" baseline in the README's method outline.

    ``reset_seed`` (default ``None``) routes the reset through
    ``_deterministic_env_reset`` so two BC warm-starts at the same
    ``init_state`` give byte-identical action sequences regardless of
    how many ``env.reset()`` calls happened in between. Pass the
    rollout's stable identifier (e.g. ``hash((demo_idx, rollout_idx))``)
    here. ``None`` preserves the legacy behaviour to avoid changing
    pre-existing JSON outputs.
    """
    if reset_seed is None:
        env.reset()
        set_sim_state(env, init_state)
    else:
        _deterministic_env_reset(env, init_state, reset_seed)
    obs_dict = env._get_observations(force_update=True)

    action_dim = int(action_low.shape[0])
    actions = np.zeros((H, action_dim), dtype=np.float32)

    # Use the obs_keys baked into the BC policy's PolicyConfig (set at
    # train time from --task) so cross-task BC checkpoints route the
    # right slice of the live env's obs dict.  Falls back to Can's
    # legacy OBS_KEYS for any legacy checkpoint that lacks the field.
    # ``Policy.config`` is not a direct attribute -- it's reached via
    # ``policy.net.config`` (MLPPolicy.config).
    _policy_cfg = getattr(getattr(policy, "net", None), "config", None)
    policy_obs_keys = getattr(_policy_cfg, "obs_keys", None) or OBS_KEYS

    t0 = time.perf_counter()
    total_reward = 0.0
    for t in range(H):
        obs_named = _get_runtime_obs(obs_dict, policy_obs_keys)
        obs_vec = _concat_obs(obs_named, policy_obs_keys)
        a = policy.act(obs_vec)
        a = np.clip(a, action_low, action_high).astype(np.float32)
        actions[t] = a
        obs_dict, reward, _done, _info = env.step(a)
        total_reward += float(reward)
    elapsed = time.perf_counter() - t0

    return actions, {
        "bc_exec_reward": float(total_reward),
        "bc_success": bool(check_success(env)),
        "bc_elapsed_s": float(elapsed),
    }


# ---------------------------------------------------------------------------
# MPC (replanning) variant of plan_and_execute
# ---------------------------------------------------------------------------


def _snapshot_sim_state(env) -> np.ndarray:
    """Return a flat MuJoCo state vector for the live env.

    Counterpart to ``common.set_sim_state``; we use this between MPC
    chunks to remember where the planner should pick up next time.
    """
    return np.asarray(env.sim.get_state().flatten())


def mpc_plan_and_execute(
    env,
    init_state: np.ndarray,
    policy: Policy | None,
    params: MBDParams,
    rng: np.random.Generator,
    *,
    n_replans: int,
    chunk_steps: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    no_warm_start: bool = False,
    bc_reset_seed: int | None = None,
    video_path: Path | None = None,
    renderer=None,
    camera: str = "agentview",
    live_view: "exp04_run.LiveView | None" = None,
    n_workers: int = 1,
    dataset_path: Path | None = None,
) -> dict:
    """MPC variant of exp 04's ``plan_and_execute``.

    With ``n_replans=N``, the H-step execution is broken into N+1
    chunks of ``chunk_steps`` env steps each (the last chunk takes
    whatever's left so the total executed steps stay at ``params.H``).
    At each chunk boundary we snapshot the live env state, redo the
    BC warm-start from that state, and run a fresh K-iteration MBD
    denoising loop with horizon ``params.H``; we then execute only
    the first ``chunk_steps`` actions of the resulting ``mu_final``
    on the live env. The planner therefore always has full
    ``params.H``-step foresight, but the *commitment* to those
    actions is reduced from ``params.H`` to ``chunk_steps``, which
    is the whole point of MPC.

    Setting ``n_replans=0`` is *not* equivalent to exp 04's
    ``plan_and_execute`` for backward compat reasons (we re-snapshot
    the env state, which is a small but observable difference): pass
    that argument and the wrapper exp 04 function from the main
    loop directly when you want the single-shot baseline.

    Returns the same shape as exp 04's ``plan_and_execute`` plus an
    additional ``"chunks"`` list with per-chunk diagnostics. The
    rollout-level ``iter_curve`` field is the *last* chunk's curve
    (so the aggregate cost-vs-iteration plot still works); each
    chunk's curve is preserved under ``chunks[i]["iter_curve"]``.
    """
    n_chunks = int(n_replans) + 1
    H = params.H
    action_dim = params.action_dim

    current_state = np.asarray(init_state)

    chunks_log: list[dict] = []
    cumulative_reward = 0.0
    capture = video_path is not None and renderer is not None
    stream_live = live_view is not None and renderer is not None
    frames: list = []

    plan_t0 = time.perf_counter()
    steps_executed = 0

    for chunk_idx in range(n_chunks):
        steps_left = H - steps_executed
        if steps_left <= 0:
            break
        # Last chunk takes the remainder so we always execute H total.
        is_last = chunk_idx == n_chunks - 1
        steps_this_chunk = steps_left if is_last else min(chunk_steps, steps_left)

        # Stage A: BC warm-start from current_state for the planner's
        # full horizon H (not just steps_this_chunk -- the planner
        # benefits from seeing the rest of the episode even if we
        # only commit the next chunk_steps actions).
        if no_warm_start or policy is None:
            initial_mean = np.zeros((H, action_dim), dtype=np.float32)
            bc_info: dict = {}
        else:
            chunk_reset_seed = (
                None if bc_reset_seed is None
                else int(bc_reset_seed) + chunk_idx
            )
            initial_mean, bc_info = bc_warmstart_actions(
                env, current_state, policy, H, action_low, action_high,
                reset_seed=chunk_reset_seed,
            )

        # Stage B: MBD denoising for K iterations from current_state.
        rollout_fn = exp04_run.make_rollout_fn(
            env, current_state, n_workers=n_workers, dataset_path=dataset_path
        )
        iter_cb = None
        if live_view is not None and renderer is not None:
            # Wrap exp 04's iter callback so the live-view title shows
            # which MPC chunk we are in. We inline a tiny version
            # rather than reusing make_iter_callback to get the
            # chunk-prefixed label.
            subsample = 4

            def iter_cb(
                k, mu, sigma, costs, log_entry,
                _ci=chunk_idx, _nc=n_chunks, _cs=current_state,
                _K=params.K,
            ):
                env.reset()
                set_sim_state(env, _cs)
                for t, action in enumerate(mu):
                    env.step(action)
                    if t % subsample == 0:
                        renderer.update_scene(
                            env.sim.data._data, camera=camera
                        )
                        frame = renderer.render()
                        live_view.update(
                            frame,
                            f"chunk {_ci + 1}/{_nc}  iter {k + 1}/{_K}  "
                            f"sigma={sigma:.2f}  "
                            f"best={log_entry['best_cost']:+.2f}  "
                            f"ess={log_entry['ess']:.1f}",
                        )

        chunk_plan_t0 = time.perf_counter()
        mu_final, log = denoise_loop(
            initial_mean, rollout_fn, exp04_run.cost_fn,
            params, rng, iter_callback=iter_cb,
        )
        rollout_fn.shutdown()
        chunk_plan_elapsed = time.perf_counter() - chunk_plan_t0

        # Stage C: execute only the first steps_this_chunk actions of
        # mu_final on the live env, starting from current_state.
        env.reset()
        set_sim_state(env, current_state)
        chunk_reward = 0.0
        for t in range(steps_this_chunk):
            action = mu_final[t]
            _obs, reward, _done, _info = env.step(action)
            chunk_reward += float(reward)
            if (capture or stream_live) and renderer is not None:
                renderer.update_scene(env.sim.data._data, camera=camera)
                frame = renderer.render()
                if capture:
                    frames.append(frame)
                if stream_live and (t % 4 == 0):
                    live_view.update(
                        frame,
                        f"chunk {chunk_idx + 1}/{n_chunks} exec  "
                        f"t={steps_executed + t + 1}/{H}  "
                        f"r_chunk={chunk_reward:+.2f}  "
                        f"r_total={cumulative_reward + chunk_reward:+.2f}",
                    )
        cumulative_reward += chunk_reward
        steps_executed += steps_this_chunk

        # Snapshot env state for next chunk's planner anchor.
        next_state = _snapshot_sim_state(env)

        chunks_log.append({
            "chunk_idx": int(chunk_idx),
            "n_chunks": int(n_chunks),
            "steps_executed": int(steps_this_chunk),
            "chunk_reward": float(chunk_reward),
            "plan_elapsed_s": float(chunk_plan_elapsed),
            "bc_exec_reward": bc_info.get("bc_exec_reward"),
            "bc_success": bc_info.get("bc_success"),
            "bc_elapsed_s": bc_info.get("bc_elapsed_s"),
            "init_best_cost": float(log["per_iter"][0]["best_cost"]),
            "final_best_cost": float(log["per_iter"][-1]["best_cost"]),
            "iter_curve": [
                {
                    "iter": p["iter"],
                    "sigma": p["sigma"],
                    "best_cost": p["best_cost"],
                    "mean_cost": p["mean_cost"],
                    "ess": p["ess"],
                }
                for p in log["per_iter"]
            ],
        })

        current_state = next_state

    total_elapsed = time.perf_counter() - plan_t0
    final_success = bool(check_success(env))

    if video_path is not None and frames:
        import mediapy as media

        video_path.parent.mkdir(parents=True, exist_ok=True)
        media.write_video(str(video_path), frames, fps=20)

    # Use the LAST chunk's iter_curve for the rollout-level
    # iter_curve / init_best / final_best fields so the
    # cost-curve aggregator in main() keeps working.
    last = chunks_log[-1] if chunks_log else None
    return {
        "final_reward": float(cumulative_reward),
        "final_success": bool(final_success),
        "plan_elapsed_s": float(total_elapsed),
        "exec_elapsed_s": 0.0,  # Subsumed into plan_elapsed_s for MPC.
        "log": {
            "per_iter": last["iter_curve"] if last else [],
        },
        "chunks": chunks_log,
    }


# ---------------------------------------------------------------------------
# Argparse + main loop
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--task", type=str, default="can")
    p.add_argument(
        "--bc-checkpoint",
        type=Path,
        default=_EXP_03 / "results" / "policy.pt",
        help="Path to the policy.pt produced by experiment 03.",
    )
    p.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Ablation: ignore the BC policy and start MBD from "
             "zero (initial_mean=0). Should reproduce experiment 04.",
    )
    p.add_argument("--n-demos", type=int, default=5,
                   help="Number of init-states. Default 5 for an "
                        "in-session run; full eval matching exps 02/03/04 "
                        "would use 20.")
    p.add_argument("--n-rollouts", type=int, default=1,
                   help="Independent MBD plans per init-state.")
    p.add_argument("--H", type=int, default=120,
                   help="Trajectory horizon (env steps). Match exp 04.")
    p.add_argument("--M", type=int, default=16,
                   help="MBD perturbed copies per denoising step.")
    p.add_argument("--K", type=int, default=10,
                   help="MBD denoising iterations.")
    p.add_argument("--lambda_", type=float, default=0.5,
                   help="MBD softmax temperature.")
    p.add_argument(
        "--sigma-max", type=float, default=0.25,
        help="Cosine schedule top. Smaller than exp 04's 0.5 default "
             "because we start *near* a BC rollout instead of from "
             "zero; aggressive perturbations would erase the BC "
             "structure in iteration 1. Pass 0.5 to disable this "
             "concession to ablation 1 in the README.",
    )
    p.add_argument("--sigma-min", type=float, default=0.05)
    p.add_argument("--n-workers", type=int, default=1,
                   help="Number of parallel worker processes for MBD "
                        "rollouts.  Default 1 (sequential).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--mpc-replans", type=int, default=0,
        help="MPC variant: number of mid-episode replans. With "
             "N>=1 the H-step execution is split into N+1 chunks; "
             "at each chunk boundary the env state is snapshotted "
             "and a fresh BC warm-start + MBD denoising loop is "
             "run from there (the planner still plans H steps "
             "ahead, but only commits chunk-steps actions). "
             "Default 0 = exp 05's single-shot headline.",
    )
    p.add_argument(
        "--mpc-chunk-steps", type=int, default=0,
        help="Env steps executed per MPC chunk before replanning. "
             "Default 0 = H // (mpc_replans + 1), which produces "
             "evenly-sized chunks. Ignored when --mpc-replans=0.",
    )
    # NOTE: --seed is already declared above (next to --n-workers); the
    # duplicate declaration here was a copy-paste leftover that argparse
    # rejects at import time with `ArgumentError: conflicting option
    # string: --seed`, making every fan-out VM crash before running.
    p.add_argument("--output", type=Path,
                   default=_THIS_DIR / "results" / "can_ph.json")
    p.add_argument("--video-dir", type=str,
                   default=str(_THIS_DIR / "results" / "videos"),
                   help="Where to save MP4s of the executed final "
                        "trajectory. Pass an empty string to disable.")
    p.add_argument("--video-h", type=int, default=480)
    p.add_argument("--video-w", type=int, default=640)
    p.add_argument("--camera", type=str, default="agentview")
    p.add_argument(
        "--live-view",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pop up a matplotlib window streaming the current mu_k "
             "after every MBD iteration plus the final executed "
             "trajectory. Pass --no-live-view for headless runs.",
    )
    p.add_argument("--device", type=str, default="cpu",
                   help="torch device for the BC policy.")
    p.add_argument(
        "--deterministic-bc-reset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Re-seed the env right before each BC warm-start so the "
             "BC-alone arm is invariant to how many env.reset() calls "
             "happened earlier in the run. Retires the documented "
             "0/60-vs-5/60 single-shot/MPC discrepancy. Default OFF "
             "to keep existing JSON outputs reproducible bit-for-bit.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    dataset = args.dataset or default_dataset_path(args.task)
    if not dataset.exists():
        raise SystemExit(
            f"Dataset not found: {dataset}\n"
            "Run: python -m robomimic.scripts.download_datasets "
            f"--tasks {args.task} --dataset_types ph"
        )

    if not args.no_warm_start and not args.bc_checkpoint.exists():
        raise SystemExit(
            f"BC checkpoint not found: {args.bc_checkpoint}\n"
            "Run experiment 03's train.py first, or pass "
            "--no-warm-start to run the exp-04 zero-init ablation."
        )

    env, env_meta = make_env_for_dataset(dataset)
    # Same shaping flip as exp 04 -- the dataset's env_args have
    # reward_shaping=False (sparse). The success metric is unchanged.
    if hasattr(env, "reward_shaping"):
        env.reward_shaping = True
    low, high = get_action_bounds(env)
    action_dim = int(low.shape[0])
    low_f32 = low.astype(np.float32)
    high_f32 = high.astype(np.float32)

    # Resolve MPC chunk size now that args.H is known.
    mpc_chunk_steps = args.mpc_chunk_steps or (
        args.H // (args.mpc_replans + 1) if args.mpc_replans > 0 else 0
    )

    print(f"[exp05] env loaded: {env_meta.get('env_name', '?')}  "
          f"reward_shaping={getattr(env, 'reward_shaping', 'n/a')}")
    print(f"[exp05] dataset:    {dataset}")
    print(f"[exp05] action_dim={action_dim}  "
          f"H={args.H}  M={args.M}  K={args.K}  lambda={args.lambda_}  "
          f"sigma_max={args.sigma_max}  warm_start={not args.no_warm_start}")
    if args.mpc_replans > 0:
        print(f"[exp05] MPC: replans={args.mpc_replans}  "
              f"chunk_steps={mpc_chunk_steps}  "
              f"(=> {args.mpc_replans + 1} plans/rollout)")

    # Load the BC policy once (skipped under --no-warm-start).
    policy: Policy | None = None
    if not args.no_warm_start:
        policy = Policy.from_checkpoint(args.bc_checkpoint, device=args.device)
        print(f"[exp05] BC checkpoint: {args.bc_checkpoint}")

    schedule = cosine_sigma_schedule(
        args.K, sigma_max=args.sigma_max, sigma_min=args.sigma_min
    )
    params = MBDParams(
        H=args.H,
        M=args.M,
        K=args.K,
        action_dim=action_dim,
        lambda_=args.lambda_,
        sigma_schedule=schedule,
        clip_action=(low_f32, high_f32),
    )

    # Renderer + popup (lazy: only if videos or live-view requested).
    renderer = None
    video_dir: Path | None = None
    video_dir_arg = (args.video_dir or "").strip()
    need_renderer = bool(video_dir_arg) or bool(args.live_view)
    if need_renderer:
        import mujoco

        renderer = mujoco.Renderer(
            env.sim.model._model, height=args.video_h, width=args.video_w
        )

    if video_dir_arg:
        video_dir = Path(video_dir_arg)
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"[exp05] writing per-demo MP4s to {video_dir}")

    live_view: exp04_run.LiveView | None = None
    if args.live_view:
        try:
            live_view = exp04_run.LiveView(
                height=args.video_h, width=args.video_w,
                title="Exp 05 -- hybrid BC + MBD on PickPlaceCan",
            )
            print("[exp05] live preview window opened (--no-live-view to skip)")
        except Exception as e:
            print(f"[exp05] live view disabled (matplotlib backend "
                  f"unavailable: {e!r})")
            live_view = None

    # Iterate over demos.
    with h5py.File(str(dataset), "r") as h5:
        demo_keys = sorted(
            h5["data"].keys(), key=lambda k: int(k.split("_")[1])
        )
        if args.n_demos > 0:
            demo_keys = demo_keys[: args.n_demos]

        per_rollout: list[dict] = []
        wall0 = time.perf_counter()
        n_total = len(demo_keys) * args.n_rollouts
        for i, key in enumerate(demo_keys):
            init_state = np.asarray(h5[f"data/{key}/states"][0])
            for r_idx in range(args.n_rollouts):
                child_rng = np.random.default_rng(
                    rng.integers(0, 2**31 - 1)
                )

                video_path: Path | None = None
                if video_dir is not None:
                    video_path = (
                        video_dir
                        / f"demo_{i:02d}_{key}_r{r_idx}.mp4"
                    )

                bc_reset_seed = (
                    # Mask to the low 32 bits BEFORE np.uint32() so the
                    # call works on both numpy 1.x (silent truncation,
                    # DeprecationWarning) and numpy 2.x (raises
                    # OverflowError on out-of-range input). The masked
                    # value is bit-identical to numpy 1.x's truncation,
                    # so seeds remain comparable across the two paths.
                    int(np.uint32(hash((int(args.seed), key, int(r_idx))) & 0xFFFFFFFF))
                    if args.deterministic_bc_reset else None
                )
                if args.mpc_replans > 0:
                    # MPC branch: chunked plan/execute. The per-chunk
                    # BC warm-start happens inside mpc_plan_and_execute
                    # so we don't pre-compute it here. We still record
                    # the FIRST chunk's BC diagnostics in `bc_info` so
                    # the per-rollout JSON shape stays compatible with
                    # the single-shot branch below.
                    result = mpc_plan_and_execute(
                        env,
                        init_state,
                        policy,
                        params,
                        child_rng,
                        n_replans=args.mpc_replans,
                        chunk_steps=mpc_chunk_steps,
                        action_low=low_f32,
                        action_high=high_f32,
                        no_warm_start=args.no_warm_start,
                        bc_reset_seed=bc_reset_seed,
                        video_path=video_path,
                        renderer=renderer,
                        camera=args.camera,
                        live_view=live_view,
                        n_workers=args.n_workers,
                        dataset_path=dataset,
                    )
                    if result["chunks"]:
                        c0 = result["chunks"][0]
                        bc_info = {
                            "bc_exec_reward": c0.get("bc_exec_reward"),
                            "bc_success": c0.get("bc_success"),
                            "bc_elapsed_s": c0.get("bc_elapsed_s"),
                        }
                    else:
                        bc_info = {}
                else:
                    # Single-shot branch (exp 05's headline).
                    bc_info = {}
                    if args.no_warm_start or policy is None:
                        initial_mean = np.zeros(
                            (args.H, action_dim), dtype=np.float32
                        )
                    else:
                        initial_mean, bc_info = bc_warmstart_actions(
                            env, init_state, policy,
                            args.H, low_f32, high_f32,
                            reset_seed=bc_reset_seed,
                        )
                    result = exp04_run.plan_and_execute(
                        env,
                        init_state,
                        params,
                        child_rng,
                        initial_mean=initial_mean,
                        video_path=video_path,
                        renderer=renderer,
                        camera=args.camera,
                        live_view=live_view,
                        n_workers=args.n_workers,
                        dataset_path=dataset,
                    )

                tag = f"{key}#{r_idx}" if args.n_rollouts > 1 else key
                init_cost = result["log"]["per_iter"][0]["best_cost"]
                final_cost = result["log"]["per_iter"][-1]["best_cost"]
                bc_str = (
                    f"bc_rew={bc_info.get('bc_exec_reward', float('nan')):+6.2f} "
                    f"bc_succ={int(bc_info.get('bc_success', False))} "
                    if bc_info else ""
                )
                print(
                    f"[exp05] {len(per_rollout) + 1:>3}/{n_total:<3} "
                    f"{tag:<15}  {bc_str}"
                    f"init_best={init_cost:>+8.2f}  "
                    f"final_best={final_cost:>+8.2f}  "
                    f"exec_reward={result['final_reward']:>+8.2f}  "
                    f"success={result['final_success']!s:<5}  "
                    f"({result['plan_elapsed_s']:.1f}s plan)"
                )

                per_rollout.append(
                    {
                        "demo": key,
                        "rollout_idx": int(r_idx),
                        "init_best_cost": float(init_cost),
                        "final_best_cost": float(final_cost),
                        "final_exec_reward": float(result["final_reward"]),
                        "success": bool(result["final_success"]),
                        "plan_elapsed_s": result["plan_elapsed_s"],
                        "exec_elapsed_s": result["exec_elapsed_s"],
                        "video": (
                            str(video_path) if video_path is not None else None
                        ),
                        "warm_start": bool(not args.no_warm_start),
                        "bc_exec_reward": bc_info.get("bc_exec_reward"),
                        "bc_success": bc_info.get("bc_success"),
                        "bc_elapsed_s": bc_info.get("bc_elapsed_s"),
                        "iter_curve": [
                            {
                                "iter": p["iter"],
                                "sigma": p["sigma"],
                                "best_cost": p["best_cost"],
                                "mean_cost": p["mean_cost"],
                                "ess": p["ess"],
                            }
                            for p in result["log"]["per_iter"]
                        ],
                        # MPC-only: full per-chunk diagnostics. Stays
                        # None for single-shot rollouts so the
                        # downstream analysis script can tell which
                        # branch produced each row.
                        "chunks": result.get("chunks"),
                    }
                )
        total_wall = time.perf_counter() - wall0

    n = len(per_rollout)
    n_success = sum(int(r["success"]) for r in per_rollout)
    bc_success_total = sum(int(bool(r.get("bc_success"))) for r in per_rollout)

    summary = {
        "experiment": "05_hybrid_method/run_can",
        "dataset": str(dataset),
        "env_name": env_meta.get("env_name"),
        "env_type": env_meta.get("type"),
        "seed": int(args.seed),
        "n_demos_attempted": len(demo_keys),
        "n_rollouts_per_demo": int(args.n_rollouts),
        "n_total": n,
        "n_success": n_success,
        "success_rate": n_success / n if n else 0.0,
        "bc_success_total": int(bc_success_total),
        "bc_success_rate": (
            float(bc_success_total) / n if n and not args.no_warm_start else None
        ),
        "warm_start": bool(not args.no_warm_start),
        "bc_checkpoint": (
            None if args.no_warm_start else str(args.bc_checkpoint)
        ),
        "params": {
            "H": int(args.H),
            "M": int(args.M),
            "K": int(args.K),
            "lambda": float(args.lambda_),
            "sigma_max": float(args.sigma_max),
            "sigma_min": float(args.sigma_min),
            "action_dim": action_dim,
            "mpc_replans": int(args.mpc_replans),
            "mpc_chunk_steps": int(mpc_chunk_steps),
        },
        "total_wall_s": float(total_wall),
        "per_rollout": per_rollout,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(
        f"[exp05] success rate: {n_success}/{n} = "
        f"{summary['success_rate']:.1%}   wall={total_wall:.1f}s   "
        f"wrote {args.output}"
    )
    if not args.no_warm_start:
        print(
            f"[exp05] BC-alone (closed-loop, H={args.H}) success "
            f"rate: {bc_success_total}/{n} = "
            f"{summary['bc_success_rate']:.1%}"
        )

    # Aggregate per-iter cost curve plot (averaged over rollouts).
    try:
        import matplotlib.pyplot as plt

        if per_rollout:
            curves = np.array(
                [
                    [p["best_cost"] for p in r["iter_curve"]]
                    for r in per_rollout
                ]
            )  # (n_rollouts, K)
            mean_curve = curves.mean(axis=0)
            fig, ax = plt.subplots(figsize=(7, 4))
            for c in curves:
                ax.plot(c, alpha=0.25, color="tab:green")
            ax.plot(mean_curve, color="tab:green", lw=2,
                    label="mean across rollouts")
            ax.set_xlabel("MBD denoising iteration $k$")
            ax.set_ylabel("best cost in batch  ($-\\sum r_t$)")
            title_extra = " (zero-init ablation)" if args.no_warm_start else ""
            ax.set_title(
                f"Exp 05: hybrid BC + MBD on PickPlaceCan{title_extra}  "
                f"(M={args.M}, K={args.K}, H={args.H})"
            )
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            plot_path = args.output.with_suffix(".png")
            fig.savefig(plot_path, dpi=120)
            plt.close(fig)
            print(f"[exp05] cost-curve plot: {plot_path}")
    except ImportError:
        print("[exp05] matplotlib not available, skipping cost-curve plot")

    if live_view is not None:
        live_view.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
