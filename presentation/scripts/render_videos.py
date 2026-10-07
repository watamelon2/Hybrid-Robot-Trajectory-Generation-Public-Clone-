"""Render the three motivation-slide MP4s.

Three modes that all use the robomimic env wrapper directly,
bypassing the MBD planning stack entirely (since the canonical
``unified_plan_and_execute`` doesn't expose a frame-capture hook
and we only need *visually-representative* clips for the talk).

    python -m presentation.scripts.render_videos \
        --mode demo_success --out presentation/videos/

Modes
-----
demo_success     Replay a successful human demo from
                 ``can_mass1p3_220demos.hdf5``.  Visually equivalent
                 to a successful hybrid rollout.

random_failure   Sample uniform-random actions for ~120 steps from
                 a fresh init.  Approximates pure-MBD-no-prior
                 behaviour: erratic, never finds the basin.

stuck_failure    Zero-action policy (robot hovers at init pose).
                 Approximates BC's no-reach failure mode.

Each call writes one MP4 (and optionally a last-frame PNG with
``--save-still``).  All clips share the same camera angle and
resolution so they slot into a 3-up grid.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_HDF5 = (
    Path("/Users/szh/Documents/SZH/Courseworks/ECE M237/Project")
    / "HRTG-shared-data" / "can_mass1p3_220demos.hdf5"
)
DEFAULT_OUT = REPO_ROOT / "presentation" / "videos"

# Pre-flight: robosuite + robomimic
os.environ.setdefault("MUJOCO_GL", "glfw")  # try glfw first; fall back if needed
import robomimic.utils.env_utils as EnvUtils  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_env(hdf5_path: Path, render_size: tuple[int, int] = (480, 640)):
    """Build the robomimic env that matches the demo HDF5."""
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path=str(hdf5_path)
    )
    H, W = render_size
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=False,
    )
    # robosuite's underlying env can be tuned post-construction:
    if hasattr(env, "env") and hasattr(env.env, "camera_widths"):
        try:
            env.env.camera_widths = [W]
            env.env.camera_heights = [H]
        except Exception:
            pass
    return env


def capture_frame(env, height: int, width: int, camera: str) -> np.ndarray:
    """Render one RGB frame from the env's offscreen camera."""
    img = env.render(mode="rgb_array",
                     height=height, width=width,
                     camera_name=camera)
    return img.astype(np.uint8)


def write_mp4(frames: list[np.ndarray], out_path: Path,
              fps: int = 30, save_still: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out_path), frames, fps=fps, codec="libx264",
                     quality=8, macro_block_size=1)
    print(f"[videos] wrote {len(frames)} frames -> {out_path}")
    if save_still:
        still = out_path.with_suffix(".png")
        imageio.imwrite(str(still), frames[-1])
        print(f"[videos] wrote final still     -> {still}")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def render_demo_success(out: Path, demo_idx: int = 0,
                        camera: str = "agentview",
                        H: int = 480, W: int = 640,
                        save_still: bool = False) -> None:
    env = make_env(DATA_HDF5, render_size=(H, W))

    with h5py.File(DATA_HDF5, "r") as f:
        keys = sorted(f["data"].keys(),
                      key=lambda k: int(k.split("_")[1]))
        demo_key = keys[demo_idx]
        states = f[f"data/{demo_key}/states"][:]
        actions = f[f"data/{demo_key}/actions"][:]
        # Some robomimic datasets store initial state in attrs
        attrs = dict(f[f"data/{demo_key}"].attrs)
        model_file = (f[f"data/{demo_key}"].attrs.get("model_file") or
                      f["data"].attrs.get("model_file"))

    # Standard robomimic reset_to: pass the first state vector.
    env.reset()
    reset_dict = {"states": states[0]}
    if model_file is not None:
        reset_dict["model"] = (model_file.decode()
                               if isinstance(model_file, bytes)
                               else model_file)
    env.reset_to(reset_dict)

    frames: list[np.ndarray] = [capture_frame(env, H, W, camera)]
    for a in actions:
        env.step(a)
        frames.append(capture_frame(env, H, W, camera))

    write_mp4(frames, out, save_still=save_still)


def render_random_failure(out: Path, n_steps: int = 120,
                          seed: int = 0,
                          camera: str = "agentview",
                          H: int = 480, W: int = 640,
                          save_still: bool = False) -> None:
    env = make_env(DATA_HDF5, render_size=(H, W))
    env.reset()
    rng = np.random.default_rng(seed)

    # Probe action dimension from the env spec.
    try:
        action_dim = env.action_dimension
    except AttributeError:
        action_dim = env.action_space.shape[0]

    frames: list[np.ndarray] = [capture_frame(env, H, W, camera)]
    for _ in range(n_steps):
        a = rng.uniform(-0.5, 0.5, size=action_dim).astype(np.float32)
        # Clamp the gripper dim to a wide-open value so the failure
        # mode is "drifts" rather than "grips empty air repeatedly".
        if action_dim >= 1:
            a[-1] = -1.0
        env.step(a)
        frames.append(capture_frame(env, H, W, camera))

    write_mp4(frames, out, save_still=save_still)


def render_stuck_failure(out: Path, n_steps: int = 120,
                         camera: str = "agentview",
                         H: int = 480, W: int = 640,
                         save_still: bool = False) -> None:
    env = make_env(DATA_HDF5, render_size=(H, W))
    env.reset()

    try:
        action_dim = env.action_dimension
    except AttributeError:
        action_dim = env.action_space.shape[0]

    frames: list[np.ndarray] = [capture_frame(env, H, W, camera)]
    zero = np.zeros(action_dim, dtype=np.float32)
    zero[-1] = -1.0  # gripper open
    for _ in range(n_steps):
        env.step(zero)
        frames.append(capture_frame(env, H, W, camera))

    write_mp4(frames, out, save_still=save_still)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=["demo_success", "random_failure",
                 "stuck_failure", "all"],
        default="all",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--demo-idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--camera", type=str, default="agentview")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--n-steps", type=int, default=120)
    p.add_argument("--save-still", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.mode in ("demo_success", "all"):
        render_demo_success(
            args.out / "hybrid_success.mp4",
            demo_idx=args.demo_idx,
            camera=args.camera, H=args.height, W=args.width,
            save_still=args.save_still,
        )
    if args.mode in ("random_failure", "all"):
        render_random_failure(
            args.out / "mbd_failure.mp4",
            n_steps=args.n_steps, seed=args.seed,
            camera=args.camera, H=args.height, W=args.width,
            save_still=args.save_still,
        )
    if args.mode in ("stuck_failure", "all"):
        render_stuck_failure(
            args.out / "bc_failure.mp4",
            n_steps=args.n_steps,
            camera=args.camera, H=args.height, W=args.width,
            save_still=args.save_still,
        )


if __name__ == "__main__":
    main()
