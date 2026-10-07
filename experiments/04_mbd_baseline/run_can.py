"""Stage 2 of experiment 04: online MBD on the can task.

Closes the loop between ``mbd.denoise_loop`` (the algorithm validated
by ``lq_sanity.py``) and the same robosuite ``PickPlaceCan`` env used
by experiments 01-03. Uses the dataset's recorded initial state and
the dataset's native OSC_POSE 7-D action space, so results are
directly comparable to the demo-replay (Exp 01), random (Exp 02), and
behaviour-cloning (Exp 03) baselines.

Two rollout backends are available:

  * Robosuite (default): uses the full PickPlaceCan MJCF -- mesh can,
    mesh fingers, the exact physics that BC + evaluation use. Costs
    are negative robosuite shaped reward. Rollouts are sequential
    (or pool-of-workers if --n-workers>1); each eval is ~5 s on Mac.

  * MJX (``--mjx``): uses ``scenes/simplified_can.xml`` (capsule can,
    box bin, motor actuators) plus the JAX port of robosuite's OSC
    controller (``jax_osc.py``, math-parity'd to 8e-12 Nm against
    the numpy reference -- see ``test_osc_parity.py``). Costs come
    from ``mjx_rollout._pickplace_step_cost`` (smooth shaped cost,
    *not* robosuite's reward). All M perturbations are vmap'd into a
    single batched ``mjx.step`` scan, so per-iter wall time becomes
    near-constant in M instead of linear. The simplification trade-off:
    different physics model => the final plan must be re-evaluated in
    the true robosuite env to claim success on PickPlaceCan.

The MBD math is already validated by ``lq_sanity.py`` and is identical
for both backends; only the (rollout_fn, cost_fn) pair changes.

Cost: ``-sum_t reward_t``, where ``reward`` is robosuite's dense
shaped reward for ``PickPlaceCan`` (reaching + grasping + lifting +
hovering, plus the success bonus). Lower cost <-> higher episode
reward. We re-roll each perturbed copy from the demo's recorded
initial state so the planner sees the same start every iteration.

Visualisation:

  - ``--video-dir results/videos/`` (default ON): an MP4 of the
    executed final trajectory is written per demo via
    ``mujoco.Renderer`` (offscreen) + ``mediapy``, mirroring the
    inline visualisation pattern in
    ``notebooks/01_inline_render.ipynb``.
  - ``--live-view`` (default ON): a live matplotlib popup window
    streams the current ``mu_k`` after every MBD iteration plus the
    final executed trajectory, so you can watch the planner descend.
    Pass ``--no-live-view`` for headless / batch runs.
  - The per-iteration cost curve is saved as a matplotlib PNG next to
    the JSON.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import h5py
import numpy as np


_THIS_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _THIS_DIR.parent
for _p in (_THIS_DIR, _EXPERIMENTS_DIR):
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
from mbd import (  # noqa: E402
    MBDParams,
    cosine_sigma_schedule,
    denoise_loop,
)
from worker import _worker_init, _worker_rollout  # noqa: E402


# ---------------------------------------------------------------------------
# Single-trajectory rollout in the live robosuite env
# ---------------------------------------------------------------------------


def rollout_one(
    env,
    init_state: np.ndarray,
    actions: np.ndarray,
    *,
    capture_frames: bool = False,
    renderer=None,
    camera: str = "agentview",
) -> tuple[float, bool, list]:
    """Run a single open-loop rollout from ``init_state``.

    Returns ``(total_reward, success, frames)``. ``frames`` is empty
    unless ``capture_frames`` is True and a ``mujoco.Renderer`` is
    provided.
    """
    env.reset()
    set_sim_state(env, init_state)

    total_reward = 0.0
    frames: list = []
    for action in actions:
        _obs, reward, _done, _info = env.step(action)
        total_reward += float(reward)
        if capture_frames and renderer is not None:
            renderer.update_scene(env.sim.data._data, camera=camera)
            frames.append(renderer.render())
    return total_reward, bool(check_success(env)), frames


# ---------------------------------------------------------------------------
# Bridge between MBD's (rollout_fn, cost_fn) and the live env
# ---------------------------------------------------------------------------


def make_rollout_fn(
    env,
    init_state: np.ndarray,
    *,
    n_workers: int = 1,
    dataset_path: Path | None = None,
    perturbations: dict | None = None,
):
    """Closure: ``(traj_batch) -> (M,) total reward per trajectory``.

    When ``n_workers > 1`` and ``dataset_path`` is provided, M perturbed
    rollouts are fanned out across a persistent pool of worker processes,
    each holding its own robosuite env.  The pool is created once and
    reused across all denoising iterations; the caller MUST call
    ``rollout_fn.shutdown()`` after the last invocation to terminate
    workers cleanly.

    ``perturbations`` (optional) is forwarded to each worker's
    ``_worker_init`` and applied to its env after creation; see
    ``worker._worker_init`` for the recognised keys.  In the
    single-process branch (``n_workers <= 1``) the caller is
    expected to have already applied perturbations to ``env``
    before passing it in -- the single-process branch never spawns
    additional envs and therefore needs no extra propagation here.
    """

    # 2026-05-31: mutable cell so callers (notably exp 15's
    # unified_loop) can reuse the SAME worker pool across MPC chunks
    # by mutating ``init_state`` between chunks via the
    # ``rollout_fn.update_init_state(new_state)`` method below.  This
    # avoids creating a fresh ctx.Pool per chunk (which on cloud VMs
    # costs ~30s/worker for robosuite env init -- ~25min wasted per
    # rollout at --mpc-replans 49).  Existing callers (05, 07, 11, 12)
    # that rebuild the closure per chunk still work without change.
    _init_state_cell = [init_state]

    if n_workers <= 1 or dataset_path is None:
        def rollout_fn(traj_batch: np.ndarray) -> np.ndarray:
            M = traj_batch.shape[0]
            rewards = np.zeros(M, dtype=np.float64)
            cur = _init_state_cell[0]
            for m in range(M):
                r, _success, _frames = rollout_one(env, cur, traj_batch[m])
                rewards[m] = r
            return rewards

        rollout_fn.shutdown = lambda: None  # noqa: E731
        rollout_fn.update_init_state = (  # type: ignore[attr-defined]
            lambda s: _init_state_cell.__setitem__(0, s)
        )
        return rollout_fn

    n_workers = min(n_workers, mp.cpu_count())
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        n_workers,
        initializer=_worker_init,
        initargs=(str(dataset_path), perturbations),
    )

    def rollout_fn(traj_batch: np.ndarray) -> np.ndarray:
        M = traj_batch.shape[0]
        cur = _init_state_cell[0]
        args = [(cur, traj_batch[m]) for m in range(M)]
        chunksize = max(1, M // n_workers)
        results = pool.map(_worker_rollout, args, chunksize=chunksize)
        return np.array([r[0] for r in results], dtype=np.float64)

    def _shutdown() -> None:
        pool.close()
        pool.join()

    rollout_fn.shutdown = _shutdown  # type: ignore[attr-defined]
    rollout_fn.update_init_state = (  # type: ignore[attr-defined]
        lambda s: _init_state_cell.__setitem__(0, s)
    )
    return rollout_fn


def cost_fn(rewards_batch: np.ndarray, _traj_batch: np.ndarray) -> np.ndarray:
    """MBD wants lower-is-better. We minimise negative episode reward."""
    return -np.asarray(rewards_batch, dtype=np.float64)


# ---------------------------------------------------------------------------
# Obs-collecting rollout variant (Branch A V_phi control variate, exp 16)
# ---------------------------------------------------------------------------
# These mirror ``rollout_one`` / ``make_rollout_fn`` but additionally
# return per-step OBS_KEYS-ordered observations. The unified driver
# dispatches to this when ``--value-baseline-ckpt`` is set so that
# the cost-fn can subtract ``sum_t V_phi(s_t)`` per trajectory.


def _obs_dict_to_vec_for_value(obs_dict: dict) -> np.ndarray:
    """OBS_KEYS-ordered flat vector. Mirrors worker.py's helper."""
    from dataset import OBS_KEYS  # exp 03's obs ordering, on sys.path

    parts = []
    for k in OBS_KEYS:
        if k not in obs_dict:
            raise KeyError(
                f"runtime obs missing {k!r} (available: "
                f"{sorted(obs_dict.keys())})"
            )
        parts.append(np.asarray(obs_dict[k], dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=-1)


def rollout_one_with_obs(
    env,
    init_state: np.ndarray,
    actions: np.ndarray,
) -> tuple[float, bool, np.ndarray]:
    """Open-loop rollout that ALSO returns per-step obs trajectory.

    Returns ``(total_reward, success, obs_traj)``. ``obs_traj`` has
    shape ``(H, obs_dim)``: the observation *before* each of the H
    actions is collected.
    """
    env.reset()
    set_sim_state(env, init_state)

    obs_buf: list[np.ndarray] = []
    total_reward = 0.0
    for action in actions:
        obs_dict = env._get_observations(force_update=True)
        obs_buf.append(_obs_dict_to_vec_for_value(obs_dict))
        _next_obs, reward, _done, _info = env.step(action)
        total_reward += float(reward)
    obs_traj = np.asarray(obs_buf, dtype=np.float32)
    return total_reward, bool(check_success(env)), obs_traj


def make_rollout_fn_with_obs(
    env,
    init_state: np.ndarray,
    *,
    n_workers: int = 1,
    dataset_path: Path | None = None,
    perturbations: dict | None = None,
    **_unused_kwargs,
):
    """Closure returning a rollout_fn that yields ``(rewards, obs_trajs)``.

    Same role as ``make_rollout_fn`` but the returned ``rollout_fn``
    produces a 2-tuple instead of a bare ``rewards`` array, so the
    downstream ``cost_fn`` can apply V_phi to ``obs_trajs``. The
    closure exposes ``.shutdown()`` for the parallel-pool case
    exactly like ``make_rollout_fn``.

    ``perturbations`` (optional, added 2026-06-01 for exp 16's V_phi
    headline eval -- exp 15's ``unified_plan_and_execute`` always
    forwards ``perturbations`` to the rollout factory; the OBS variant
    must accept it to be substitutable).  In the parallel branch it
    is forwarded to each worker's ``_worker_init`` exactly like the
    non-OBS ``make_rollout_fn`` does (see its docstring).  In the
    single-process branch the caller is expected to have already
    applied perturbations to ``env`` before passing it in.
    ``**_unused_kwargs`` swallows any future-added kwargs from the
    non-OBS factory so the two factories remain substitutable
    without re-touching this file.

    The returned ``rollout_fn`` also exposes ``.update_init_state(s)``
    (added 2026-06-01 for parity with the non-OBS factory used by
    exp 15's ``unified_plan_and_execute``, which mutates the anchor
    between MPC chunks instead of rebuilding the closure -- saves
    ~30 s/worker pool init at high replan counts).
    """
    from worker import _worker_init, _worker_rollout_with_obs

    # 2026-06-01: mutable cell so callers (notably exp 15's
    # unified_loop) can re-anchor across MPC chunks via
    # ``rollout_fn.update_init_state(new_state)`` -- mirrors the
    # non-OBS ``make_rollout_fn`` API so the two factories are
    # substitutable.
    _init_state_cell = [init_state]

    if n_workers <= 1 or dataset_path is None:
        def rollout_fn(traj_batch: np.ndarray):
            M = traj_batch.shape[0]
            rewards = np.zeros(M, dtype=np.float64)
            obs_trajs: list[np.ndarray] = []
            cur = _init_state_cell[0]
            for m in range(M):
                r, _success, obs_traj = rollout_one_with_obs(
                    env, cur, traj_batch[m],
                )
                rewards[m] = r
                obs_trajs.append(obs_traj)
            return rewards, np.stack(obs_trajs, axis=0)

        rollout_fn.shutdown = lambda: None  # noqa: E731
        rollout_fn.update_init_state = (  # type: ignore[attr-defined]
            lambda s: _init_state_cell.__setitem__(0, s)
        )
        return rollout_fn

    n_workers = min(n_workers, mp.cpu_count())
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        n_workers,
        initializer=_worker_init,
        initargs=(str(dataset_path), perturbations),
    )

    def rollout_fn(traj_batch: np.ndarray):
        M = traj_batch.shape[0]
        cur = _init_state_cell[0]
        args = [(cur, traj_batch[m]) for m in range(M)]
        chunksize = max(1, M // n_workers)
        results = pool.map(_worker_rollout_with_obs, args, chunksize=chunksize)
        rewards = np.array([r[0] for r in results], dtype=np.float64)
        obs_trajs = np.stack([r[2] for r in results], axis=0)
        return rewards, obs_trajs

    def _shutdown() -> None:
        pool.close()
        pool.join()

    rollout_fn.shutdown = _shutdown  # type: ignore[attr-defined]
    rollout_fn.update_init_state = (  # type: ignore[attr-defined]
        lambda s: _init_state_cell.__setitem__(0, s)
    )
    return rollout_fn


# ---------------------------------------------------------------------------
# Live mujoco preview window (matplotlib-based, no `mjpython` needed)
# ---------------------------------------------------------------------------


class LiveView:
    """Lightweight matplotlib window that displays mujoco frames as they
    come in, with a status string overlaid as the figure title.

    Why matplotlib and not ``mujoco.viewer``: ``viewer.launch_passive``
    requires running under ``mjpython`` on macOS, which would force
    every script in this repo onto a different interpreter. matplotlib
    with the default macOS backend pops a window from regular ``python``
    and is good enough for a preview.

    Why not opencv: avoids adding ``opencv-python`` to the env;
    matplotlib is already a runtime dep via this script's cost-curve
    plot.
    """

    def __init__(self, height: int = 480, width: int = 640,
                 title: str = "Exp 04 stage 2 -- MBD planning"):
        # Lazy import; keeps the script importable in headless contexts.
        import matplotlib

        # Default backend on macOS (``macosx``) and on most installs of
        # ``hybrid-traj`` is interactive; we don't override unless we
        # detect the headless ``Agg`` fallback.
        if matplotlib.get_backend().lower() == "agg":
            for candidate in ("MacOSX", "TkAgg", "Qt5Agg"):
                try:
                    matplotlib.use(candidate, force=True)
                    break
                except Exception:
                    continue

        import matplotlib.pyplot as plt

        self._plt = plt
        self.fig, self.ax = plt.subplots(figsize=(width / 100, height / 100))
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        self._img = self.ax.imshow(
            np.zeros((height, width, 3), dtype=np.uint8)
        )
        self.ax.axis("off")
        self._title = self.ax.set_title("(initialising)", fontsize=10)
        self.fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    def update(self, frame: np.ndarray, label: str = "") -> None:
        self._img.set_data(frame)
        self._title.set_text(label)
        self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._plt.close(self.fig)
        except Exception:
            pass


def make_iter_callback(
    env,
    init_state: np.ndarray,
    renderer,
    live_view: LiveView,
    K: int,
    *,
    camera: str = "agentview",
    subsample: int = 4,
):
    """Return a ``denoise_loop`` ``iter_callback`` that streams ``mu_k``.

    After each MBD iteration we re-roll the just-updated mean trajectory
    in the live env (one extra rollout of length H per iteration) and
    push every ``subsample``-th frame to the popup window. Costs about
    ``H / subsample`` render calls per iteration.
    """

    def cb(k, mu, sigma, costs, log_entry):
        env.reset()
        set_sim_state(env, init_state)
        for t, action in enumerate(mu):
            env.step(action)
            if t % subsample == 0:
                renderer.update_scene(env.sim.data._data, camera=camera)
                frame = renderer.render()
                label = (
                    f"iter {k + 1}/{K}   sigma={sigma:.2f}   "
                    f"best_cost={log_entry['best_cost']:+.2f}   "
                    f"mean_cost={log_entry['mean_cost']:+.2f}   "
                    f"ess={log_entry['ess']:.1f}"
                )
                live_view.update(frame, label)

    return cb


# ---------------------------------------------------------------------------
# Planning + execution for a single (demo, rollout) pair
# ---------------------------------------------------------------------------


def plan_and_execute(
    env,
    init_state: np.ndarray,
    params: MBDParams,
    rng: np.random.Generator,
    *,
    initial_mean: np.ndarray | None = None,
    video_path: Path | None = None,
    renderer=None,
    camera: str = "agentview",
    live_view: LiveView | None = None,
    n_workers: int = 1,
    dataset_path: Path | None = None,
) -> dict:
    """Run MBD from ``initial_mean`` (default zeros), then execute it.

    Passing a non-zero ``initial_mean`` of shape ``(H, action_dim)`` is
    how experiment 05 wires the BC warm-start in: the BC policy is
    rolled out open-loop for H steps from ``init_state`` and the
    resulting action sequence is handed in as ``initial_mean`` instead
    of zeros. Same denoising loop afterwards.
    """
    rollout_fn = make_rollout_fn(
        env, init_state, n_workers=n_workers, dataset_path=dataset_path
    )

    iter_cb = None
    if live_view is not None and renderer is not None:
        iter_cb = make_iter_callback(
            env, init_state, renderer, live_view, params.K, camera=camera
        )

    if initial_mean is None:
        mu0 = np.zeros((params.H, params.action_dim), dtype=np.float32)
    else:
        mu0 = np.asarray(initial_mean, dtype=np.float32)
        if mu0.shape != (params.H, params.action_dim):
            raise ValueError(
                f"initial_mean shape {mu0.shape} != "
                f"(H={params.H}, action_dim={params.action_dim})"
            )
    t0 = time.perf_counter()
    mu_final, log = denoise_loop(
        mu0, rollout_fn, cost_fn, params, rng, iter_callback=iter_cb
    )
    rollout_fn.shutdown()
    plan_elapsed = time.perf_counter() - t0

    # Final execution -- captures frames if a video path was requested,
    # and additionally streams them to the live view so the run finishes
    # with the planned trajectory visible.
    capture = video_path is not None and renderer is not None
    stream_live = live_view is not None and renderer is not None

    t0 = time.perf_counter()
    env.reset()
    set_sim_state(env, init_state)
    total_reward = 0.0
    frames: list = []
    for t, action in enumerate(mu_final):
        _obs, reward, _done, _info = env.step(action)
        total_reward += float(reward)
        if (capture or stream_live) and renderer is not None:
            renderer.update_scene(env.sim.data._data, camera=camera)
            frame = renderer.render()
            if capture:
                frames.append(frame)
            if stream_live and (t % 4 == 0):
                live_view.update(
                    frame,
                    f"executing mu_final   t={t + 1}/{params.H}   "
                    f"reward_so_far={total_reward:+.2f}",
                )
    final_reward = total_reward
    final_success = bool(check_success(env))
    exec_elapsed = time.perf_counter() - t0

    if video_path is not None and frames:
        import mediapy as media

        video_path.parent.mkdir(parents=True, exist_ok=True)
        media.write_video(str(video_path), frames, fps=20)

    return {
        "mu_final": mu_final,
        "final_reward": float(final_reward),
        "final_success": bool(final_success),
        "plan_elapsed_s": float(plan_elapsed),
        "exec_elapsed_s": float(exec_elapsed),
        "log": log,
    }


# ---------------------------------------------------------------------------
# MJX rollout backend (Gate 3 of docs/mjx_migration.md)
# ---------------------------------------------------------------------------


def plan_and_execute_mjx(
    ctx,
    qpos_init,
    qvel_init,
    params: MBDParams,
    rng: np.random.Generator,
    *,
    initial_mean: np.ndarray | None = None,
    prior_score_fn=None,
    prior_lambda: float = 0.0,
) -> dict:
    """MBD planning + final-plan eval using the MJX rollout backend.

    Mirror of ``plan_and_execute`` for the ``--mjx`` path:

      * ``ctx`` is a ``mjx_rollout.RolloutContext`` (already JIT-compiled).
      * ``rollout_fn(tau)`` calls ``ctx.batched_rollout``, returning the
        per-trajectory smooth costs from
        ``mjx_rollout._pickplace_step_cost``.
      * ``cost_fn`` is the identity -- MJX already gives us costs in the
        right sign convention.
      * The "final execution" pass is just one more ``ctx.single_rollout``
        on ``mu_final`` so we record final cost and final qpos.

    No live preview / no MP4 output -- those would need an MJX-side
    renderer (out of scope for Gate 3; can be added later by rolling
    ``mu_final`` into a CPU ``mujoco.MjData`` and feeding ``mj_step``).
    """
    import jax.numpy as jnp  # local import: keeps file importable on
                              # boxes without jax installed.

    # JIT warm-up: trigger compilation by running one dummy batch so the
    # first denoise iteration doesn't pay compile latency. We use the
    # actual M of the upcoming loop, since shape changes would re-trigger
    # compilation.
    print("[mjx] warming up rollout JIT (one dummy batch)...", flush=True)
    M = int(params.M)
    H = int(params.H)
    action_dim = int(params.action_dim)
    dummy = np.zeros((M, H, action_dim), dtype=np.float32)
    t0 = time.perf_counter()
    costs_warm, _qf, _ps = ctx.batched_rollout(
        qpos_init, qvel_init, jnp.asarray(dummy)
    )
    costs_warm.block_until_ready()
    compile_s = time.perf_counter() - t0
    print(
        f"[mjx] warm-up done in {compile_s:.1f}s "
        f"(zero-action cost = {float(np.asarray(costs_warm).mean()):.3f})",
        flush=True,
    )

    def mjx_rollout_fn(tau: np.ndarray) -> np.ndarray:
        costs, _qf, _ps = ctx.batched_rollout(
            qpos_init, qvel_init,
            jnp.asarray(tau, dtype=jnp.float32),
        )
        return np.asarray(costs, dtype=np.float64)

    mjx_rollout_fn.shutdown = lambda: None  # type: ignore[attr-defined]

    def mjx_cost_fn(costs: np.ndarray, _tau: np.ndarray) -> np.ndarray:
        # mjx_rollout already returns smooth, lower-is-better costs.
        return np.asarray(costs, dtype=np.float64)

    if initial_mean is None:
        mu0 = np.zeros((H, action_dim), dtype=np.float32)
    else:
        mu0 = np.asarray(initial_mean, dtype=np.float32)
        if mu0.shape != (H, action_dim):
            raise ValueError(
                f"initial_mean shape {mu0.shape} != (H={H}, "
                f"action_dim={action_dim})"
            )

    t0 = time.perf_counter()
    mu_final, log = denoise_loop(
        mu0, mjx_rollout_fn, mjx_cost_fn, params, rng,
        prior_score_fn=prior_score_fn,
        prior_lambda=prior_lambda,
    )
    plan_elapsed = time.perf_counter() - t0

    # "Execute" the final mean trajectory once more so we record the
    # final cost and the final qpos for analysis. No env.step calls --
    # this is still MJX-side.
    t0 = time.perf_counter()
    final_cost_j, qpos_final_j, per_step_cost_j = ctx.single_rollout(
        qpos_init, qvel_init,
        jnp.asarray(mu_final, dtype=jnp.float32),
    )
    final_cost_j.block_until_ready()
    exec_elapsed = time.perf_counter() - t0

    final_cost = float(final_cost_j)
    qpos_final = np.asarray(qpos_final_j)
    per_step_cost = np.asarray(per_step_cost_j)

    return {
        "mu_final": np.asarray(mu_final),
        # Sign-flip to match robosuite's "reward" field for comparability
        # across the two backends in the JSON / printed table.
        "final_reward": float(-final_cost),
        "final_cost": float(final_cost),
        "final_success": False,  # set by caller (needs robosuite eval)
        "plan_elapsed_s": float(plan_elapsed),
        "exec_elapsed_s": float(exec_elapsed),
        "qpos_final": qpos_final,
        "per_step_cost": per_step_cost,
        "log": log,
    }


def _load_prior_for_mjx(args):
    """Load the flow prior + per-demo s0 vectors for the MJX path.

    Returns ``(prior_or_None, s0_array_or_None, label)``.

    For ``--prior=flow``:
      * Loads the v_theta checkpoint via ``FlowPrior.from_checkpoint``
        (the adapter auto-dispatches v1 vs v2 on the ``arch`` field).
      * Reads ``obs0`` from the robomimic dataset's first ``n_demos``
        trajectories to use as s0 conditioning. The MJX physics state
        (qpos_init, qvel_init) is not used to compute s0 -- that would
        require running a robosuite step. Using the dataset's recorded
        obs[0] is an OOD approximation but matches what the unified
        driver does.
      * Returns the prior and s0 array; the caller indexes s0[i] per
        plan.

    For ``--prior=none``: returns ``(None, None, "none")``.
    """
    if args.prior == "none":
        return None, None, "none"
    if args.prior != "flow":
        raise ValueError(f"Unknown --prior={args.prior!r} on MJX path")

    # Import lazily so the --prior=none path stays torch-free.
    sys.path.insert(0, str(_EXPERIMENTS_DIR / "08_flow_matching_prior"))
    import torch  # noqa: F401
    from adapter import FlowPrior  # auto-dispatches v1 vs v2
    from dataset import load_trajectories

    dataset_path = args.dataset or default_dataset_path(args.task)
    if not dataset_path.exists():
        raise SystemExit(
            f"--prior=flow needs the dataset at {dataset_path} to read "
            f"per-demo s0. Run: python -m robomimic.scripts.download_datasets "
            f"--tasks {args.task} --dataset_types ph"
        )
    actions, obs0, demo_lens, demo_ids = load_trajectories(
        dataset_path, horizon=args.H, n_demos=args.n_demos
    )
    print(
        f"[exp04/mjx] loaded {len(demo_ids)} demo s0 vectors from "
        f"{dataset_path.name}  obs0={obs0.shape}"
    )

    device = args.device
    if device == "auto":
        import torch as _t
        device = "cuda" if _t.cuda.is_available() else "cpu"
    prior = FlowPrior.from_checkpoint(args.prior_checkpoint, device=device)
    arch = type(prior).__name__
    print(
        f"[exp04/mjx] prior={args.prior} ({arch})  ckpt={args.prior_checkpoint}  "
        f"device={device}  warm_start={args.warm_start}  "
        f"lambda_prior={args.lambda_prior}"
    )
    return prior, obs0.astype(np.float32), f"flow({arch})"


def run_mjx_main(args: argparse.Namespace) -> int:
    """Entry point for the ``--mjx`` rollout backend.

    Parallels the robosuite path in ``main`` but does not touch
    robosuite at all -- planning runs entirely in JAX/MJX on the
    simplified MJCF. The robomimic dataset is only used (optionally)
    as the index for the per-rollout JSON entries so the schema
    matches the robosuite path.

    With ``--prior=flow``, the dataset's recorded obs[0] for demo i
    is used to condition the flow prior (both for the warm-start
    initial mean and for the per-step score-bridge). The MJX physics
    starts from the simplified scene's ``home`` keyframe regardless,
    so prior conditioning and physics init are decoupled -- a known
    approximation that matches the unified driver.

    Initial state for every plan is the ``home`` keyframe in
    ``simplified_can.xml`` (set once at context build time). We do
    NOT load per-demo init states from the dataset: those refer to
    the full robosuite MJCF and the qpos vectors do not line up with
    the simplified MJCF's joint ordering. Each "demo" in the MJX path
    is therefore just a unique RNG seed for an independent plan.
    """
    # Local imports keep the robosuite path importable without jax.
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    from mjx_rollout import build_rollout_context

    if not args.mjx_scene.exists():
        raise SystemExit(f"--mjx-scene not found: {args.mjx_scene}")

    rng = np.random.default_rng(args.seed)

    # Action space for OSC_POSE: 7 dims, all in [-1, +1]. Same as
    # robosuite's `get_action_bounds` returns for OSC_POSE.
    action_dim = 7
    low = -np.ones(action_dim, dtype=np.float32)
    high = +np.ones(action_dim, dtype=np.float32)

    print(f"[exp04/mjx] scene:      {args.mjx_scene}")
    print(
        f"[exp04/mjx] action_dim={action_dim}  H={args.H}  M={args.M}  "
        f"K={args.K}  lambda={args.lambda_}  n_substeps={args.mjx_n_substeps}"
    )
    # Print active JAX device so the user knows whether they got GPU.
    try:
        import jax as _jax
        backend = _jax.default_backend()
        devices = _jax.devices()
        print(
            f"[exp04/mjx] jax backend={backend!r}  "
            f"devices={[str(d) for d in devices]}"
        )
    except Exception:
        pass

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
        clip_action=(low, high),
    )

    print("[exp04/mjx] building rollout context (compiles MJCF + JIT)...",
          flush=True)
    t0 = time.perf_counter()
    ctx = build_rollout_context(
        str(args.mjx_scene),
        n_substeps=args.mjx_n_substeps,
    )
    qpos_init, qvel_init = (
        jnp.asarray(ctx.mj_model.key_qpos[
            int(__import__("mujoco").mj_name2id(
                ctx.mj_model,
                __import__("mujoco").mjtObj.mjOBJ_KEY,
                "home",
            ))
        ], dtype=jnp.float32),
        jnp.zeros((ctx.mj_model.nv,), dtype=jnp.float32),
    )
    ctx_build_s = time.perf_counter() - t0
    print(f"[exp04/mjx] context built in {ctx_build_s:.1f}s")

    # Load the prior (and per-demo s0 vectors from the robomimic dataset).
    prior, s0_list, prior_label = _load_prior_for_mjx(args)

    # MJX path: one plan per (demo_idx, rollout_idx) pair, but the
    # "demo" is just a seed slot. Number of plans = n_demos * n_rollouts.
    n_demos = max(1, args.n_demos)
    n_rollouts = max(1, args.n_rollouts)
    per_rollout: list[dict] = []
    wall0 = time.perf_counter()
    for i in range(n_demos):
        # Per-demo prior closures: initial_mean and score_fn both
        # depend on s0_i, so we build them once per demo and reuse
        # across the n_rollouts seed slots.
        initial_mean_i = None
        score_fn_i = None
        if prior is not None and s0_list is not None and i < len(s0_list):
            s0_i = s0_list[i]
            if args.warm_start:
                initial_mean_i = prior.sample_initial_mean(s0_i, n_steps=32)
                # Clip to action bounds; the prior is free to produce
                # large samples and MBD's clip happens inside the loop.
                initial_mean_i = np.clip(
                    initial_mean_i, low, high
                ).astype(np.float32)
            if args.lambda_prior > 0.0:
                score_fn_i = prior.make_prior_score_fn(s0_i, K=args.K)

        for r_idx in range(n_rollouts):
            child_rng = np.random.default_rng(
                rng.integers(0, 2**31 - 1)
            )
            result = plan_and_execute_mjx(
                ctx, qpos_init, qvel_init, params, child_rng,
                initial_mean=initial_mean_i,
                prior_score_fn=score_fn_i,
                prior_lambda=float(args.lambda_prior),
            )
            tag = f"mjx_{i:02d}#{r_idx}" if n_rollouts > 1 else f"mjx_{i:02d}"
            init_cost = result["log"]["per_iter"][0]["best_cost"]
            final_cost = result["log"]["per_iter"][-1]["best_cost"]
            print(
                f"[exp04/mjx] {len(per_rollout) + 1:>3}/"
                f"{n_demos * n_rollouts:<3} {tag:<15}  "
                f"init_best={init_cost:>+8.2f}  "
                f"final_best={final_cost:>+8.2f}  "
                f"exec_cost={result['final_cost']:>+8.2f}  "
                f"({result['plan_elapsed_s']:.1f}s plan, "
                f"{result['exec_elapsed_s']:.2f}s exec)"
            )
            per_rollout.append(
                {
                    "demo": tag,
                    "rollout_idx": int(r_idx),
                    "init_best_cost": float(init_cost),
                    "final_best_cost": float(final_cost),
                    "final_exec_cost": float(result["final_cost"]),
                    "final_exec_reward": float(result["final_reward"]),
                    "success": False,  # MJX path can't certify robosuite success
                    "plan_elapsed_s": result["plan_elapsed_s"],
                    "exec_elapsed_s": result["exec_elapsed_s"],
                    "video": None,
                    # mu_final saved here so a follow-up notebook cell can
                    # replay it through robosuite to certify success. Shape
                    # is (H, action_dim) ~= (120, 7) floats = ~3.4 KB per plan.
                    "mu_final": result["mu_final"].astype(np.float32).tolist(),
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
                }
            )
    total_wall = time.perf_counter() - wall0

    # Default output path differs from the robosuite path so the two
    # don't overwrite each other if the user forgets to pass --output.
    out_path = args.output
    if out_path == _THIS_DIR / "results" / "can_ph.json":
        out_path = _THIS_DIR / "results" / "can_mjx.json"

    summary = {
        "experiment": "04_mbd_baseline/run_can_mjx",
        "mjx_scene": str(args.mjx_scene),
        "env_name": "MJX::simplified_can",
        "env_type": "mjx",
        "seed": int(args.seed),
        "n_demos_attempted": n_demos,
        "n_rollouts_per_demo": n_rollouts,
        "n_total": len(per_rollout),
        "n_success": 0,  # see comment above
        "success_rate": 0.0,
        "params": {
            "H": int(args.H),
            "M": int(args.M),
            "K": int(args.K),
            "lambda": float(args.lambda_),
            "sigma_max": float(args.sigma_max),
            "sigma_min": float(args.sigma_min),
            "action_dim": action_dim,
            "mjx_n_substeps": int(args.mjx_n_substeps),
        },
        "total_wall_s": float(total_wall),
        "ctx_build_s": float(ctx_build_s),
        "per_rollout": per_rollout,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(
        f"[exp04/mjx] wall={total_wall:.1f}s   wrote {out_path}"
    )

    # Cost-curve PNG (same as robosuite path).
    try:
        import matplotlib.pyplot as plt

        if per_rollout:
            curves = np.array(
                [
                    [p["best_cost"] for p in r["iter_curve"]]
                    for r in per_rollout
                ]
            )
            mean_curve = curves.mean(axis=0)
            fig, ax = plt.subplots(figsize=(7, 4))
            for c in curves:
                ax.plot(c, alpha=0.25, color="tab:orange")
            ax.plot(mean_curve, color="tab:orange", lw=2,
                    label="mean across plans")
            ax.set_xlabel("MBD denoising iteration $k$")
            ax.set_ylabel("best cost in batch  (mjx smooth shaped cost)")
            ax.set_title(
                f"Exp 04 stage 2 -- MJX backend  "
                f"(M={args.M}, K={args.K}, H={args.H}, "
                f"n_substeps={args.mjx_n_substeps})"
            )
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            plot_path = out_path.with_suffix(".png")
            fig.savefig(plot_path, dpi=120)
            plt.close(fig)
            print(f"[exp04/mjx] cost-curve plot: {plot_path}")
    except ImportError:
        pass

    return 0


# ---------------------------------------------------------------------------
# Argparse + main loop
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--task", type=str, default="can")
    p.add_argument("--n-demos", type=int, default=5,
                   help="Number of init-states. Default 5 for an "
                        "in-session run; full eval matching exps 02/03 "
                        "would use 20.")
    p.add_argument("--n-rollouts", type=int, default=1,
                   help="Independent MBD plans per init-state.")
    p.add_argument("--H", type=int, default=120,
                   help="Trajectory horizon (env steps).")
    p.add_argument("--M", type=int, default=16,
                   help="MBD perturbed copies per denoising step. "
                        "Each adds H sequential env steps. M=8 is "
                        "too noisy for PickPlaceCan; M=16 is the "
                        "smallest that gives reliable descent in "
                        "this script's tuning runs.")
    p.add_argument("--K", type=int, default=10,
                   help="MBD denoising iterations.")
    p.add_argument("--lambda_", type=float, default=0.5,
                   help="MBD softmax temperature. Empirically, episode "
                        "reward differences on PickPlaceCan with this "
                        "H/M are ~2-4; lambda=0.5 gives the best "
                        "perturbation real weight (lambda=5 averages "
                        "too softly and mu drifts back to zero-action).")
    p.add_argument("--sigma-max", type=float, default=0.5,
                   help="Cosine schedule top. Action space is [-1, 1] "
                        "in OSC_POSE; 0.5 explores meaningfully.")
    p.add_argument("--sigma-min", type=float, default=0.05)
    p.add_argument("--n-workers", type=int, default=1,
                   help="Number of parallel worker processes for MBD "
                        "rollouts.  Each worker holds its own robosuite "
                        "env.  Default 1 (sequential).  Set to 4 for "
                        "e2-standard-4 VMs.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path,
                   default=_THIS_DIR / "results" / "can_ph.json")
    p.add_argument("--video-dir", type=str,
                   default=str(_THIS_DIR / "results" / "videos"),
                   help="Where to save MP4s of the executed final "
                        "trajectory per demo. Pass an empty string to "
                        "disable.")
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

    # ---- MJX rollout backend (Gate 3 of docs/mjx_migration.md) ----
    p.add_argument(
        "--mjx",
        action="store_true",
        default=False,
        help="Use the JAX/MJX batched rollout instead of robosuite. "
             "All M perturbations are vmap'd into one mjx.step scan "
             "per policy step, so per-iter wall time becomes near-"
             "constant in M. Uses scenes/simplified_can.xml + the "
             "JAX OSC port (jax_osc.py, math-parity'd to 8e-12 Nm). "
             "Cost is mjx_rollout._pickplace_step_cost (smooth, shaped "
             "toward pick-place), NOT robosuite's reward, so cross-"
             "backend comparison must re-evaluate the final plan in "
             "the true robosuite env.",
    )
    p.add_argument(
        "--mjx-scene",
        type=Path,
        default=_THIS_DIR / "scenes" / "simplified_can.xml",
        help="Path to the simplified MJCF used by --mjx. Default: "
             "experiments/04_mbd_baseline/scenes/simplified_can.xml",
    )
    p.add_argument(
        "--mjx-n-substeps",
        type=int,
        default=5,
        help="mjx.step calls per policy step under --mjx. Robosuite's "
             "policy_freq=20 / sim_freq=500 implies 25; we default to 5 "
             "for faster Gate-3 iteration. Bump to 25 for the official "
             "parity number.",
    )

    # ---- Flow / BC prior wiring (works under --mjx; see _load_prior_for_mjx) ----
    p.add_argument(
        "--prior",
        type=str,
        default="none",
        choices=["none", "flow"],
        help="Initial-mean source and per-step prior_score_fn for the MBD "
             "denoise loop. 'none' (default) reproduces the pure-MBD "
             "baseline. 'flow' loads a v_theta checkpoint (v1 or v2 -- "
             "the adapter auto-dispatches on the checkpoint's `arch` "
             "field) and conditions it on the per-demo robomimic obs[0] "
             "to produce (a) a warm-start initial trajectory and (b) a "
             "score-bridge term in the denoiser. NOTE on --mjx: the s0 "
             "used to condition the prior is read from the robomimic "
             "dataset, NOT from the simplified MJX scene (which has no "
             "obs dict). This is a known OOD approximation that matches "
             "the unified driver's behaviour.",
    )
    p.add_argument(
        "--prior-checkpoint",
        type=Path,
        default=_EXPERIMENTS_DIR / "08_flow_matching_prior" / "results" / "v_theta.pt",
        help="Path to the v_theta .pt for --prior=flow.",
    )
    p.add_argument(
        "--lambda-prior",
        type=float,
        default=0.0,
        help="Weight of the prior score term in the MBD update. 0 "
             "disables the prior contribution to the per-step update "
             "(initial mean is still used if --warm-start is set).",
    )
    p.add_argument(
        "--warm-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the prior's sample_initial_mean(s0) as the MBD seed "
             "trajectory. Pass --no-warm-start to seed with zeros even "
             "when --prior=flow.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="torch device for the prior network. 'auto' uses cuda if "
             "available else cpu.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # MJX backend dispatched here so the robosuite env is never built
    # under --mjx (saves ~30s and lets the script run on machines
    # without robomimic / robosuite installed).
    if args.mjx:
        return run_mjx_main(args)

    rng = np.random.default_rng(args.seed)

    dataset = args.dataset or default_dataset_path(args.task)
    if not dataset.exists():
        raise SystemExit(
            f"Dataset not found: {dataset}\n"
            "Run: python -m robomimic.scripts.download_datasets "
            f"--tasks {args.task} --dataset_types ph"
        )

    env, env_meta = make_env_for_dataset(dataset)
    # PickPlaceCan's recorded env_args have reward_shaping=False
    # (sparse: 1 only on success), which gives MBD zero gradient
    # signal across the planning horizon. Flip on shaping for
    # planning -- the success metric (`check_success`) is unchanged.
    if hasattr(env, "reward_shaping"):
        env.reward_shaping = True
    low, high = get_action_bounds(env)
    action_dim = int(low.shape[0])

    print(f"[exp04] env loaded: {env_meta.get('env_name', '?')}  "
          f"reward_shaping={getattr(env, 'reward_shaping', 'n/a')}")
    print(f"[exp04] dataset:    {dataset}")
    print(f"[exp04] action_dim={action_dim}  "
          f"H={args.H}  M={args.M}  K={args.K}  lambda={args.lambda_}")

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
        clip_action=(low.astype(np.float32), high.astype(np.float32)),
    )

    # Renderer + popup window (lazy: only if videos or live-view requested).
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
        print(f"[exp04] writing per-demo MP4s to {video_dir}")

    live_view: LiveView | None = None
    if args.live_view:
        try:
            live_view = LiveView(
                height=args.video_h, width=args.video_w,
                title="Exp 04 stage 2 -- MBD planning on PickPlaceCan",
            )
            print("[exp04] live preview window opened (--no-live-view to skip)")
        except Exception as e:
            print(f"[exp04] live view disabled (matplotlib backend "
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
                result = plan_and_execute(
                    env,
                    init_state,
                    params,
                    child_rng,
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
                print(
                    f"[exp04] {len(per_rollout) + 1:>3}/"
                    f"{len(demo_keys) * args.n_rollouts:<3} {tag:<15}  "
                    f"init_best={init_cost:>+8.2f}  "
                    f"final_best={final_cost:>+8.2f}  "
                    f"exec_reward={result['final_reward']:>+8.2f}  "
                    f"success={result['final_success']!s:<5}  "
                    f"({result['plan_elapsed_s']:.1f}s plan, "
                    f"{result['exec_elapsed_s']:.1f}s exec)"
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
                    }
                )
        total_wall = time.perf_counter() - wall0

    n = len(per_rollout)
    n_success = sum(int(r["success"]) for r in per_rollout)
    summary = {
        "experiment": "04_mbd_baseline/run_can",
        "dataset": str(dataset),
        "env_name": env_meta.get("env_name"),
        "env_type": env_meta.get("type"),
        "seed": int(args.seed),
        "n_demos_attempted": len(demo_keys),
        "n_rollouts_per_demo": int(args.n_rollouts),
        "n_total": n,
        "n_success": n_success,
        "success_rate": n_success / n if n else 0.0,
        "params": {
            "H": int(args.H),
            "M": int(args.M),
            "K": int(args.K),
            "lambda": float(args.lambda_),
            "sigma_max": float(args.sigma_max),
            "sigma_min": float(args.sigma_min),
            "action_dim": action_dim,
        },
        "total_wall_s": float(total_wall),
        "per_rollout": per_rollout,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(
        f"[exp04] success rate: {n_success}/{n} = "
        f"{summary['success_rate']:.1%}   wall={total_wall:.1f}s   "
        f"wrote {args.output}"
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
                ax.plot(c, alpha=0.25, color="tab:blue")
            ax.plot(mean_curve, color="tab:blue", lw=2,
                    label="mean across rollouts")
            ax.set_xlabel("MBD denoising iteration $k$")
            ax.set_ylabel("best cost in batch  ($-\\sum r_t$)")
            ax.set_title(
                f"Exp 04 stage 2: MBD on PickPlaceCan  "
                f"(M={args.M}, K={args.K}, H={args.H})"
            )
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            plot_path = args.output.with_suffix(".png")
            fig.savefig(plot_path, dpi=120)
            plt.close(fig)
            print(f"[exp04] cost-curve plot: {plot_path}")
    except ImportError:
        print("[exp04] matplotlib not available, skipping cost-curve plot")

    if live_view is not None:
        live_view.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

