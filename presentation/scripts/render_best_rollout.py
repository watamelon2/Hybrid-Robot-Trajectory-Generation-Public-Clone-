"""Render the *best* expert demo's full rollout as an MP4 per task.

For each of the five robomimic ph tasks, picks the
shortest-horizon successful demo (the cleanest expert path),
restores the env's MuJoCo state to that demo's first frame,
replays the recorded action sequence step-by-step with the
offscreen renderer enabled, and writes the result as an MP4.

Unlike ``render_trajectory_on_env.py`` (which only paints
ensemble lines on a *single* background frame), this script
shows the robot actually executing the task -- the visual the
audience sees when we say "this is what success looks like".

    # All 5 tasks
    python -m presentation.scripts.render_best_rollout --task all

    # Single task
    python -m presentation.scripts.render_best_rollout --task can \\
        --fps 30 --slow-factor 1.0

Outputs:
  videos/best_rollout_<task>.mp4

Implementation notes
--------------------
Robosuite's ``env.sim.render`` returns OpenGL-up-Y frames, so
we flip vertically before encoding.  Demo HDF5 ``actions`` are
already in the env's native command space (delta OSC pose +
gripper) so they can be replayed open-loop without re-planning.

For the long-horizon tasks (tool_hang ~330 steps, transport
~370 steps) you can pass ``--slow-factor 0.5`` to halve the
playback rate so the audience can actually see what's happening,
or ``--max-steps`` to truncate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
VID_DIR = REPO_ROOT / "presentation" / "videos"

# Reuse the task registry + env-build helpers from the
# trajectory-overlay script so the two stay in lock-step.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "presentation" / "scripts"))
from render_trajectory_on_env import (  # noqa: E402
    TASK_HDF5,
    make_env_with_offscreen,
)


# ---------------------------------------------------------------------------
# Pick the optimal (shortest-horizon) demo
# ---------------------------------------------------------------------------
def find_optimal_demo(data_hdf5: Path) -> str:
    with h5py.File(str(data_hdf5), "r") as f:
        keys = sorted(f["data"].keys(),
                      key=lambda k: int(k.split("_")[1]))
        # Use the demo with the shortest action sequence as the
        # "cleanest expert" -- robomimic ph datasets only contain
        # successful demos, so all are valid, and the shortest is
        # the most direct.
        horizons = [f[f"data/{k}/actions"].shape[0] for k in keys]
    opt_i = int(np.argmin(horizons))
    return keys[opt_i]


# ---------------------------------------------------------------------------
# Replay one demo in the env, capturing every frame
# ---------------------------------------------------------------------------
def replay_demo(
    data_hdf5: Path, demo_key: str,
    camera: str, height: int, width: int,
    max_steps: int | None = None,
    mode: str = "action",
) -> tuple[list[np.ndarray], int, bool]:
    """Return (frames, T_executed, succeeded?).

    Parameters
    ----------
    mode
        ``"action"``  -- open-loop ``env.step(actions[t])`` from
                         ``states[0]``.  Honest physics, but
                         diverges over long horizons (tool_hang /
                         transport often end ``success=False``).
        ``"state"``   -- ``set_sim_state(env, states[t])`` at every
                         step.  Pixel-perfect replay of the recorded
                         trajectory; no contact-solver drift.
                         Use this when the action-replay didn't end
                         in a success state.
    """
    from experiments.common import set_sim_state, check_success

    env, _ = make_env_with_offscreen(data_hdf5, camera, height, width)

    with h5py.File(str(data_hdf5), "r") as f:
        states = np.asarray(f[f"data/{demo_key}/states"])
        actions = np.asarray(f[f"data/{demo_key}/actions"])

    env.reset()
    set_sim_state(env, states[0])

    T = actions.shape[0] if max_steps is None else min(actions.shape[0],
                                                      max_steps)

    frames: list[np.ndarray] = []

    # Initial frame (before any action) so the video opens on the
    # static scene at t=0 for ~0.17 s of visual context.
    img0 = env.sim.render(width=width, height=height, camera_name=camera)
    frames.extend([np.flipud(img0)] * 5)

    if mode == "action":
        for t in range(T):
            env.step(actions[t])
            img = env.sim.render(width=width, height=height,
                                 camera_name=camera)
            frames.append(np.flipud(img))
    elif mode == "state":
        # states has shape (T+1, state_dim) in robomimic ph datasets;
        # iterate from t=1 since we already rendered states[0].
        for t in range(1, min(T + 1, states.shape[0])):
            set_sim_state(env, states[t])
            img = env.sim.render(width=width, height=height,
                                 camera_name=camera)
            frames.append(np.flipud(img))
    else:
        raise ValueError(f"unknown replay mode {mode!r}")

    # Hold the final frame for ~0.5 s after the demo ends so the
    # success state is visible without freeze-framing manually.
    if frames:
        frames.extend([frames[-1]] * 15)

    success = bool(check_success(env))
    return frames, T, success


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------
def encode_mp4(frames: list[np.ndarray], out: Path,
               fps: int = 30, slow_factor: float = 1.0) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    effective_fps = max(1, int(round(fps * slow_factor)))
    imageio.mimwrite(str(out), frames, fps=effective_fps,
                     codec="libx264", quality=8, macro_block_size=1)
    print(f"[best_rollout] wrote {len(frames)} frames @ {effective_fps} fps "
          f"-> {out}")


# ---------------------------------------------------------------------------
# Per-task render
# ---------------------------------------------------------------------------
def render_one_task(
    task: str, data_hdf5: Path, args,
) -> None:
    print(f"\n[best_rollout] === task={task!r} data={data_hdf5} ===")
    if not data_hdf5.exists():
        print(f"[best_rollout] SKIP {task}: dataset not found")
        return

    demo_key = find_optimal_demo(data_hdf5)
    print(f"[best_rollout] optimal demo = {demo_key}")

    out_mp4 = (args.out_mp4 if args.out_mp4 is not None
               else VID_DIR / f"best_rollout_{task}.mp4")

    frames, T, ok = replay_demo(
        data_hdf5, demo_key,
        camera=args.camera, height=args.height, width=args.width,
        max_steps=args.max_steps, mode=args.mode,
    )
    print(f"[best_rollout] replayed T={T} steps (mode={args.mode!r}); "
          f"success={ok}")

    encode_mp4(frames, out_mp4, fps=args.fps,
               slow_factor=args.slow_factor)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task", type=str, default="can",
        choices=sorted(TASK_HDF5) + ["all"],
    )
    p.add_argument("--data", type=Path, default=None,
                   help="Override demo HDF5 path.")
    p.add_argument("--camera", type=str, default="agentview")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--fps", type=int, default=30,
                   help="Source render fps; the actual playback fps "
                        "is fps * slow_factor.")
    p.add_argument(
        "--slow-factor", type=float, default=1.0,
        help="<1.0 plays slower (good for long-horizon tasks); "
             ">1.0 plays faster (good for time-boxed slides). "
             "Default 1.0 = real-time at --fps.",
    )
    p.add_argument(
        "--max-steps", type=int, default=None,
        help="Truncate the replay after N env steps.  Useful for "
             "tool_hang/transport (T~330-370) if you want a clip "
             "that fits in a single talk beat.",
    )
    p.add_argument(
        "--mode", type=str, default="action",
        choices=["action", "state"],
        help="How to replay: 'action' = env.step(demo_actions[t]) "
             "(honest physics, can diverge on long horizons); "
             "'state' = set_sim_state(env, demo_states[t]) every "
             "step (pixel-perfect, guarantees success final state). "
             "Default 'action'.  Use 'state' for tool_hang / "
             "transport where action-replay drifts.",
    )
    p.add_argument("--out-mp4", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks = sorted(TASK_HDF5) if args.task == "all" else [args.task]
    if args.task == "all" and args.out_mp4 is not None:
        raise SystemExit(
            "--out-mp4 cannot be combined with --task all "
            "(filenames would collide)."
        )
    for task in tasks:
        data_hdf5 = (args.data if args.data is not None
                     else TASK_HDF5[task])
        render_one_task(task, data_hdf5, args)


if __name__ == "__main__":
    main()
