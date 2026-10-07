"""Render an N x N grid of MuJoCo rollouts as a single tiled MP4.

Runs the unified hybrid planner (``experiments/15_unified_hybrid/run_can.py``)
for ``grid*grid`` rollouts -- each a different demo init-state under the
SAME prior config (default: retrieval-hybrid ``demo_1nn`` at lambda=1) --
captures every executed-step frame via that CLI's ``--video-dir`` hook,
then composites the rollouts into one synchronized grid video.

    # Full 4x4 retrieval-hybrid montage on can (moderate config)
    python -m presentation.scripts.render_prior_grid \
        --grid 4 --task can --mpc-replans 4

    # Fast pipeline check (2x2, cheap planner config), reuse later with --skip-run
    python -m presentation.scripts.render_prior_grid \
        --grid 2 --K 4 --mpc-replans 0 --frames-dir /tmp/grid_frames

    # Re-tile already-captured frames without re-running the planner
    python -m presentation.scripts.render_prior_grid \
        --grid 4 --skip-run --frames-dir /tmp/grid_frames

Rollouts have different lengths; shorter ones hold their final frame so
the grid stays synchronized.  A green/red border marks success/failure.

Implementation
--------------
Frame capture lives in the planner CLI (guarded by ``--video-dir``): it
wraps the main env's ``step()`` with a ``mujoco.Renderer`` and dumps each
rollout's frames to ``<dir>/<prior>_lam-<L>_<demo>_r<r>.npz``.  Run with
``--n-workers>=2`` so MBD's perturbation rollouts happen in worker envs
and only executed actions are captured.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CAN = REPO_ROOT / "experiments" / "15_unified_hybrid" / "run_can_grid.py"
VID_DIR = REPO_ROOT / "presentation" / "videos"

_DEMO_RE = re.compile(r"_demo_(\d+)_r(\d+)\.npz$")


def _lazy_imageio():
    import imageio.v2 as imageio  # type: ignore
    return imageio


# ---------------------------------------------------------------------------
# Run the planner CLI with frame capture enabled
# ---------------------------------------------------------------------------
def run_planner(args, frames_dir: Path) -> None:
    n = args.grid * args.grid
    cmd = [
        sys.executable, str(RUN_CAN),
        "--prior", args.prior,
        "--lambda-prior", str(args.lambda_prior),
        "--K", str(args.K),
        "--M", str(args.M),
        "--H", str(args.H),
        "--sigma-max", str(args.sigma_max),
        "--mpc-replans", str(args.mpc_replans),
        "--task", args.task,
        "--n-demos", str(n),
        "--n-rollouts", "1",
        "--n-workers", str(args.n_workers),
        "--seed", str(args.seed),
        "--video-dir", str(frames_dir),
        "--video-camera", args.camera,
        "--video-h", str(args.tile),
        "--video-w", str(args.tile),
        "--output-dir", str(frames_dir / "json"),
    ]
    if args.extra:
        cmd += args.extra
    print("[grid] running planner:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Load + order the captured rollouts
# ---------------------------------------------------------------------------
def load_rollouts(frames_dir: Path, n: int) -> list[dict]:
    files = sorted(
        (p for p in frames_dir.glob("*.npz") if _DEMO_RE.search(p.name)),
        key=lambda p: tuple(int(g) for g in _DEMO_RE.search(p.name).groups()),
    )
    if not files:
        raise SystemExit(
            f"[grid] no *.npz frame files found in {frames_dir}. "
            f"Did the planner run with --video-dir?"
        )
    if len(files) < n:
        print(f"[grid] WARNING only {len(files)} rollouts found, need {n}; "
              f"grid will be padded with blanks.")
    rolls = []
    for p in files[:n]:
        d = np.load(p, allow_pickle=True)
        rolls.append({
            "frames": d["frames"],            # (T, H, W, 3) uint8
            "success": bool(d["success"]),
            "demo": str(d["demo"]) if "demo" in d else p.stem,
        })
    return rolls


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------
def _add_border(frame: np.ndarray, color: tuple[int, int, int],
                bw: int) -> np.ndarray:
    out = np.empty(
        (frame.shape[0] + 2 * bw, frame.shape[1] + 2 * bw, 3), dtype=np.uint8
    )
    out[:] = color
    out[bw:bw + frame.shape[0], bw:bw + frame.shape[1]] = frame
    return out


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    """Best-effort caption in the top-left corner (no-op without PIL)."""
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except Exception:
        return frame
    im = Image.fromarray(frame)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, max(46, 8 * len(text)), 14], fill=(0, 0, 0))
    dr.text((2, 2), text, fill=(255, 255, 255))
    return np.asarray(im)


def build_grid_video(rolls: list[dict], grid: int, bw: int,
                     hold_start: int, hold_end: int) -> np.ndarray:
    tile_h, tile_w = rolls[0]["frames"].shape[1:3]
    n = grid * grid
    blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

    # Pad to a common length: shorter rollouts hold their last frame.
    t_max = max(r["frames"].shape[0] for r in rolls)
    padded: list[np.ndarray] = []
    for idx in range(n):
        if idx < len(rolls):
            fr = rolls[idx]["frames"]
            last = fr[-1]
            pad = np.repeat(last[None], t_max - fr.shape[0], axis=0) \
                if fr.shape[0] < t_max else fr[:0]
            seq = np.concatenate([fr, pad], axis=0) if pad.shape[0] else fr
            color = (40, 200, 40) if rolls[idx]["success"] else (210, 50, 50)
            seq = np.stack([
                _label(_add_border(f, color, bw),
                       f"{idx:02d} {'OK' if rolls[idx]['success'] else 'X'}")
                for f in seq
            ])
        else:
            seq = np.repeat(_add_border(blank, (60, 60, 60), bw)[None],
                            t_max, axis=0)
        # Hold first/last frame for readability.
        seq = np.concatenate([
            np.repeat(seq[:1], hold_start, axis=0), seq,
            np.repeat(seq[-1:], hold_end, axis=0),
        ], axis=0)
        padded.append(seq)

    # Tile: rows of hstack, then vstack the rows.
    T = padded[0].shape[0]
    grid_frames = np.empty(
        (T, grid * (tile_h + 2 * bw), grid * (tile_w + 2 * bw), 3),
        dtype=np.uint8,
    )
    th, tw = tile_h + 2 * bw, tile_w + 2 * bw
    for r in range(grid):
        for c in range(grid):
            cell = padded[r * grid + c]
            grid_frames[:, r * th:(r + 1) * th, c * tw:(c + 1) * tw] = cell
    return grid_frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", type=int, default=4, help="Grid side (NxN).")
    p.add_argument("--task", type=str, default="can")
    p.add_argument("--prior", type=str, default="demo_1nn")
    p.add_argument("--lambda-prior", type=float, default=1.0, dest="lambda_prior")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--H", type=int, default=120)
    p.add_argument("--sigma-max", type=float, default=0.25, dest="sigma_max")
    p.add_argument("--mpc-replans", type=int, default=4, dest="mpc_replans")
    p.add_argument("--n-workers", type=int, default=2, dest="n_workers")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--camera", type=str, default="agentview")
    p.add_argument("--tile", type=int, default=256, help="Per-tile pixels.")
    p.add_argument("--border", type=int, default=4, help="Tile border px.")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--hold-start", type=int, default=8, dest="hold_start")
    p.add_argument("--hold-end", type=int, default=15, dest="hold_end")
    p.add_argument("--frames-dir", type=Path, default=Path("/tmp/grid_frames"),
                   dest="frames_dir")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--skip-run", action="store_true",
                   help="Reuse existing frames in --frames-dir.")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Extra args passed through to run_can.py.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    n = args.grid * args.grid
    if not args.skip_run:
        run_planner(args, args.frames_dir)

    rolls = load_rollouts(args.frames_dir, n)
    n_ok = sum(r["success"] for r in rolls)
    print(f"[grid] loaded {len(rolls)} rollouts ({n_ok} success); compositing...")

    frames = build_grid_video(rolls, args.grid, args.border,
                              args.hold_start, args.hold_end)

    out = args.out or (VID_DIR / f"prior_grid_{args.prior}_{args.task}_"
                       f"{args.grid}x{args.grid}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio = _lazy_imageio()
    imageio.mimwrite(str(out), list(frames), fps=args.fps, codec="libx264",
                     quality=8, macro_block_size=1)
    print(f"[grid] wrote {frames.shape[0]} frames "
          f"({frames.shape[2]}x{frames.shape[1]}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
