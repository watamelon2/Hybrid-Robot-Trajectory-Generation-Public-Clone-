"""Render MP4 trajectory videos for the Week-8 presentation.

This script wraps the existing ``experiments/15_unified_hybrid/run_can.py``
rollout entry point to also dump rendered RGB frames and stitch
them into MP4s.  Three preset comparison rollouts are supported
out of the box:

    python -m presentation.scripts.render_rollouts \
        --preset comparison-trio --out presentation/videos/

This produces:
  - bc_failure.mp4       (BC prior, lambda=0, no MBD bridge)
  - mbd_failure.mp4      (no prior, lambda=0, K=10 -- pure MBD)
  - hybrid_success.mp4   (demo_1nn prior, lambda=1, K=10)

Or render a single arbitrary configuration:

    python -m presentation.scripts.render_rollouts \
        --prior demo_1nn --lambda-prior 1.0 --K 10 --seed 0 \
        --out-mp4 presentation/videos/custom.mp4

Implementation notes
--------------------
robosuite renders RGB frames via ``env.sim.render(...)``, which
this script wraps in a callback passed to ``unified_plan_and_execute``.
The callback collects frames every executed env step (NOT every
planning step -- we want a watchable real-time rollout, not the
internal MBD denoising).  Frames are downsampled to 480p and
encoded at 30 fps with ``imageio-ffmpeg``.

If the ``--out-stills`` flag is passed, a final-frame PNG is also
saved (useful for the title-slide hero image).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Lazy-import: only required when actually rendering.  We do not
# want this module to be expensive to import.
def _lazy_imageio():
    import imageio.v2 as imageio  # type: ignore
    return imageio


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Re-use the canonical rollout entry point.  Keeping this thin
# means we follow whatever the Phase-D run.py considers "the
# canonical operating point" without drift.
from experiments.unified_hybrid_compat import (  # type: ignore  # noqa: E402
    build_env,
    build_prior,
    canonical_args,
)
from experiments._fifteen.unified_loop import (  # type: ignore  # noqa: E402
    unified_plan_and_execute,
)


@dataclass
class Preset:
    name: str
    prior: str
    lambda_prior: float
    K: int
    seed: int
    description: str


PRESETS: dict[str, Preset] = {
    "bc_failure": Preset(
        name="bc_failure",
        prior="bc",
        lambda_prior=0.0,
        K=0,
        seed=0,
        description="BC policy alone; never reaches the can.",
    ),
    "mbd_failure": Preset(
        name="mbd_failure",
        prior="none",
        lambda_prior=0.0,
        K=10,
        seed=0,
        description=(
            "Pure MBD with random init; chaotic, 0/60 baseline."
        ),
    ),
    "hybrid_success": Preset(
        name="hybrid_success",
        prior="demo_1nn",
        lambda_prior=1.0,
        K=10,
        seed=0,
        description=(
            "Retrieval prior + MBD bridge; clean grasp & place."
        ),
    ),
}


def render_one(
    preset: Preset,
    out_path: Path,
    *,
    task: str = "can",
    fps: int = 30,
    height: int = 480,
    width: int = 640,
    camera: str = "agentview",
    max_steps: int = 120,
    save_still: bool = False,
) -> None:
    """Run a single rollout end-to-end and write its MP4."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = build_env(task=task, has_renderer=False, has_offscreen_renderer=True,
                    use_camera_obs=False, camera_names=[camera],
                    camera_heights=[height], camera_widths=[width])
    prior = build_prior(preset.prior)
    args = canonical_args(prior=preset.prior, K=preset.K,
                          lambda_prior=preset.lambda_prior,
                          seed=preset.seed, task=task)

    frames: list[np.ndarray] = []

    def on_env_step(env_, *, step_idx: int, **_):
        # Robosuite's sim.render returns HxWx3 uint8 (upside-down,
        # so we flip vertically).
        img = env_.sim.render(width=width, height=height,
                              camera_name=camera)
        frames.append(np.flipud(img))

    init_state = env.reset()
    _ = unified_plan_and_execute(
        env,
        init_state,
        params=args,
        prior=prior,
        on_env_step=on_env_step,
        max_steps=max_steps,
    )

    if not frames:
        raise RuntimeError(
            f"render_one({preset.name}): no frames captured -- check that "
            f"unified_plan_and_execute accepts an `on_env_step` kwarg.  "
            f"If not, patch it to call the callback inside its env-step loop."
        )

    imageio = _lazy_imageio()
    imageio.mimwrite(str(out_path), frames, fps=fps, codec="libx264",
                     quality=8, macro_block_size=1)
    print(f"[render_rollouts] wrote {len(frames)} frames -> {out_path}")

    if save_still:
        still_path = out_path.with_suffix(".png")
        imageio.imwrite(str(still_path), frames[-1])
        print(f"[render_rollouts] wrote final still     -> {still_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=list(PRESETS) + ["comparison-trio"],
                   default=None,
                   help="Render a named preset or the full trio.")
    p.add_argument("--prior", type=str, default=None,
                   help="Override prior family for a custom rollout.")
    p.add_argument("--lambda-prior", type=float, default=None)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task", type=str, default="can")
    p.add_argument("--out-mp4", type=Path, default=None)
    p.add_argument("--out", type=Path,
                   default=Path("presentation/videos"),
                   help="Directory for preset MP4s (preset mode).")
    p.add_argument("--save-still", action="store_true",
                   help="Also dump the final frame as PNG.")
    p.add_argument("--fps", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.preset == "comparison-trio":
        for name in ("bc_failure", "mbd_failure", "hybrid_success"):
            render_one(PRESETS[name],
                       out_path=args.out / f"{name}.mp4",
                       task=args.task, fps=args.fps,
                       save_still=args.save_still)
        return

    if args.preset is not None:
        render_one(PRESETS[args.preset],
                   out_path=args.out / f"{args.preset}.mp4",
                   task=args.task, fps=args.fps,
                   save_still=args.save_still)
        return

    # Custom rollout
    if args.prior is None or args.lambda_prior is None or args.K is None:
        raise SystemExit(
            "Custom rollout requires --prior, --lambda-prior, and --K "
            "(or use --preset).")

    if args.out_mp4 is None:
        raise SystemExit("Custom rollout requires --out-mp4.")

    preset = Preset(name="custom", prior=args.prior,
                    lambda_prior=args.lambda_prior, K=args.K,
                    seed=args.seed, description="user override")
    render_one(preset, out_path=args.out_mp4, task=args.task,
               fps=args.fps, save_still=args.save_still)


if __name__ == "__main__":
    main()
