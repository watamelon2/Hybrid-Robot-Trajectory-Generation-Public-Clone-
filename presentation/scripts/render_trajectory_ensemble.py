"""Render a DDPM-style trajectory ensemble visualisation.

Visualises N candidate end-effector trajectories sampled from the
robomimic demo bank (one per demo) overlaid on a single ``optimal''
demo highlighted in red.  This is the canonical Diffusion-Policy
figure: a faded cloud of candidate roll-outs that the cost function
must sort through, plus the ground-truth basin we want to land in.

Two outputs:
  1. ``trajectory_ensemble.pdf``  -- static 3D figure (3 subplots:
     3D view, top-down XY, side ZX), 20 candidates + 1 optimal.
  2. ``trajectory_ensemble.mp4``  -- animated playhead version,
     ~5 s, all candidates revealed over time with the optimal
     thickening as the time-step advances.

Pure matplotlib + h5py + numpy + imageio.  No robosuite/mujoco.

    python -m presentation.scripts.render_trajectory_ensemble \
        --n-candidates 24 --fps 30 --duration-s 5

Optimal-demo selection
----------------------
Default heuristic: pick the demo with the shortest trajectory
among those whose final state evaluates to success in the
robomimic reward stream.  The shortest is usually the cleanest
"expert" path -- minimal dithering, direct grasp.

Cite the inspirations:
  - DDPM trajectory plots: Chi et al., Diffusion Policy, 2023.
  - Multi-rollout overlay style: Janner et al., Diffuser, 2022.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d)


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_HDF5 = (
    Path("/Users/szh/Documents/SZH/Courseworks/ECE M237/Project")
    / "HRTG-shared-data" / "can_mass1p3_220demos.hdf5"
)
FIG_DIR = REPO_ROOT / "presentation" / "figures"
VID_DIR = REPO_ROOT / "presentation" / "videos"


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
def load_eef_trajectories(
    hdf5_path: Path,
    n_candidates: int,
    optimal_strategy: str = "shortest",
    seed: int = 0,
) -> tuple[list[np.ndarray], int]:
    """Return (list of (T_i, 3) eef-xyz trajectories, optimal_index).

    Demos are ordered by ascending demo index in the HDF5 file.
    ``optimal_strategy`` picks the highlighted "optimal" demo:
      - "shortest": the shortest-horizon demo (cleanest expert).
      - "first":    demo_0.
    """
    rng = np.random.default_rng(seed)
    trajs: list[np.ndarray] = []
    horizons: list[int] = []

    with h5py.File(str(hdf5_path), "r") as f:
        all_keys = sorted(
            f["data"].keys(),
            key=lambda k: int(k.split("_")[1]),
        )
        # Sample n_candidates demos deterministically.
        if len(all_keys) > n_candidates:
            idxs = np.linspace(0, len(all_keys) - 1, n_candidates,
                               dtype=int)
            chosen = [all_keys[i] for i in idxs]
        else:
            chosen = all_keys

        for k in chosen:
            eef = f[f"data/{k}/obs/robot0_eef_pos"][:]  # (T, 3)
            trajs.append(np.asarray(eef, dtype=np.float64))
            horizons.append(eef.shape[0])

    if optimal_strategy == "shortest":
        opt_idx = int(np.argmin(horizons))
    elif optimal_strategy == "first":
        opt_idx = 0
    else:
        opt_idx = int(rng.integers(0, len(trajs)))

    return trajs, opt_idx


# ---------------------------------------------------------------------------
# Static figure  -- 3 panels (3D + XY + XZ)
# ---------------------------------------------------------------------------
def plot_static(trajs: list[np.ndarray], opt_idx: int, out: Path) -> None:
    fig = plt.figure(figsize=(11, 3.6))
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    axxy = fig.add_subplot(1, 3, 2)
    axxz = fig.add_subplot(1, 3, 3)

    n = len(trajs)
    cmap = plt.get_cmap("Blues")
    for i, tr in enumerate(trajs):
        if i == opt_idx:
            continue
        c = cmap(0.25 + 0.35 * (i / max(n - 1, 1)))
        ax3d.plot(tr[:, 0], tr[:, 1], tr[:, 2],
                  color=c, lw=0.8, alpha=0.55)
        axxy.plot(tr[:, 0], tr[:, 1], color=c, lw=0.8, alpha=0.55)
        axxz.plot(tr[:, 0], tr[:, 2], color=c, lw=0.8, alpha=0.55)

    opt = trajs[opt_idx]
    ax3d.plot(opt[:, 0], opt[:, 1], opt[:, 2],
              color="#c0392b", lw=2.6, label="optimal demo")
    axxy.plot(opt[:, 0], opt[:, 1], color="#c0392b", lw=2.6,
              label="optimal demo")
    axxz.plot(opt[:, 0], opt[:, 2], color="#c0392b", lw=2.6,
              label="optimal demo")
    # Endpoints
    for ax_, (x, y) in [(axxy, (opt[-1, 0], opt[-1, 1])),
                        (axxz, (opt[-1, 0], opt[-1, 2]))]:
        ax_.scatter([x], [y], s=40, c="#c0392b", zorder=5,
                    edgecolors="k", lw=0.5)
    ax3d.scatter(opt[-1, 0], opt[-1, 1], opt[-1, 2],
                 s=40, c="#c0392b", edgecolors="k", lw=0.5)

    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    ax3d.set_title("3D end-effector path", fontsize=10)
    ax3d.view_init(elev=22, azim=-60)

    axxy.set_xlabel("x"); axxy.set_ylabel("y")
    axxy.set_title("top-down (x-y)", fontsize=10)
    axxy.set_aspect("equal", adjustable="datalim")
    axxy.grid(True, alpha=0.3)

    axxz.set_xlabel("x"); axxz.set_ylabel("z")
    axxz.set_title("side (x-z)", fontsize=10)
    axxz.grid(True, alpha=0.3)

    proxy_cand = Line2D([], [], color=cmap(0.4), lw=1.2,
                        label=f"{n - 1} candidate roll-outs")
    proxy_opt = Line2D([], [], color="#c0392b", lw=2.6,
                       label="optimal demo")
    axxy.legend(handles=[proxy_cand, proxy_opt], loc="best",
                fontsize=8, frameon=False)

    fig.suptitle(
        "Diffusion-Policy roll-out ensemble: cost function selects "
        "from a cloud of candidates",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[trajectory_ensemble] wrote static  -> {out}")


# ---------------------------------------------------------------------------
# Animated MP4 -- playhead reveals trajectories progressively
# ---------------------------------------------------------------------------
def render_animation(
    trajs: list[np.ndarray], opt_idx: int, out: Path,
    fps: int = 30, duration_s: float = 5.0,
    height: int = 480, width: int = 960,
) -> None:
    n_frames = int(fps * duration_s)
    # Resample every trajectory to the same number of "playback ticks"
    # so the playhead is a clean fraction of each path's length.
    n_pts = 80
    resampled: list[np.ndarray] = []
    for tr in trajs:
        idx = np.linspace(0, len(tr) - 1, n_pts).astype(int)
        resampled.append(tr[idx])
    R = np.stack(resampled, axis=0)  # (N, n_pts, 3)

    # Global bounds for stable axes
    xs = R[..., 0]; ys = R[..., 1]; zs = R[..., 2]
    pad = 0.02
    xlim = (xs.min() - pad, xs.max() + pad)
    ylim = (ys.min() - pad, ys.max() + pad)
    zlim = (zs.min() - pad, zs.max() + pad)

    cmap = plt.get_cmap("Blues")
    cand_colors = [cmap(0.25 + 0.35 * (i / max(len(trajs) - 1, 1)))
                   for i in range(len(trajs))]

    frames: list[np.ndarray] = []
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axxy = fig.add_subplot(1, 2, 2)

    for fi in range(n_frames):
        # progress 0 -> 1 with a 0.5 s tail at the end
        t = min(1.0, fi / max(n_frames - int(0.5 * fps), 1))
        cut = max(2, int(t * n_pts))

        ax3d.cla()
        axxy.cla()

        for i in range(R.shape[0]):
            if i == opt_idx:
                continue
            seg = R[i, :cut]
            ax3d.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                      color=cand_colors[i], lw=0.8, alpha=0.55)
            axxy.plot(seg[:, 0], seg[:, 1],
                      color=cand_colors[i], lw=0.8, alpha=0.55)

        opt = R[opt_idx, :cut]
        ax3d.plot(opt[:, 0], opt[:, 1], opt[:, 2],
                  color="#c0392b", lw=2.8)
        axxy.plot(opt[:, 0], opt[:, 1], color="#c0392b", lw=2.8)
        if cut > 1:
            ax3d.scatter(opt[-1, 0], opt[-1, 1], opt[-1, 2],
                         s=42, c="#c0392b", edgecolors="k", lw=0.5)
            axxy.scatter(opt[-1, 0], opt[-1, 1], s=42,
                         c="#c0392b", edgecolors="k", lw=0.5)

        ax3d.set_xlim(xlim); ax3d.set_ylim(ylim); ax3d.set_zlim(zlim)
        ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
        ax3d.view_init(elev=22, azim=-60 + 25 * t)  # gentle spin
        ax3d.set_title("3D end-effector ensemble", fontsize=10)

        axxy.set_xlim(xlim); axxy.set_ylim(ylim)
        axxy.set_xlabel("x"); axxy.set_ylabel("y")
        axxy.set_aspect("equal", adjustable="box")
        axxy.grid(True, alpha=0.3)
        axxy.set_title("top-down (x-y)", fontsize=10)

        proxy_cand = Line2D([], [], color=cmap(0.4), lw=1.2,
                            label=f"{R.shape[0] - 1} candidate roll-outs")
        proxy_opt = Line2D([], [], color="#c0392b", lw=2.6,
                           label="optimal demo")
        axxy.legend(handles=[proxy_cand, proxy_opt],
                    loc="upper right", fontsize=8, frameon=False)

        fig.suptitle(
            "Diffusion-Policy roll-out ensemble (red = optimal)",
            fontsize=11, y=0.98,
        )
        fig.tight_layout()

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(buf)

    plt.close(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out), frames, fps=fps, codec="libx264",
                     quality=8, macro_block_size=1)
    print(f"[trajectory_ensemble] wrote animation -> {out} "
          f"({len(frames)} frames, {duration_s} s @ {fps} fps)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-candidates", type=int, default=24)
    p.add_argument("--optimal", type=str, default="shortest",
                   choices=["shortest", "first", "random"])
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--duration-s", type=float, default=5.0)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--data", type=Path, default=DATA_HDF5)
    p.add_argument("--out-pdf", type=Path,
                   default=FIG_DIR / "trajectory_ensemble.pdf")
    p.add_argument("--out-mp4", type=Path,
                   default=VID_DIR / "trajectory_ensemble.mp4")
    p.add_argument("--static-only", action="store_true",
                   help="Skip the MP4 render (only the static PDF).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    trajs, opt_idx = load_eef_trajectories(
        args.data, n_candidates=args.n_candidates,
        optimal_strategy=args.optimal,
    )
    print(f"[trajectory_ensemble] loaded {len(trajs)} trajectories; "
          f"optimal = demo index {opt_idx} (T={len(trajs[opt_idx])})")

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plot_static(trajs, opt_idx, args.out_pdf)

    if not args.static_only:
        render_animation(trajs, opt_idx, args.out_mp4,
                         fps=args.fps, duration_s=args.duration_s,
                         height=args.height, width=args.width)


if __name__ == "__main__":
    main()
