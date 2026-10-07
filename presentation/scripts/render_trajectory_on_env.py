"""Render the candidate-trajectory ensemble overlaid on the env.

Supports the five robomimic ``ph`` tasks bundled with the project's
conda env: ``lift``, ``can``, ``square``, ``tool_hang``, ``transport``.
For each task the script builds the matching robosuite env with the
offscreen renderer enabled, captures one background frame at the
optimal demo's initial state, projects every demo's 3-D end-effector
trajectory into the image plane via MuJoCo's pinhole camera matrix,
and overlays the trajectories as matplotlib lines.  The optimal demo
(default: shortest-horizon expert) is in red; the rest in faded blue.

By default the candidate pool is restricted to demos whose object
starting xy is within ``--cluster-radius-xy`` (default 4 cm) of the
optimal demo, so the visualisation shows N attempts at the *same*
task -- which is what MBD's denoise-ensemble figure actually shows.

Outputs (per task):
  - figures/trajectory_on_env_<task>.pdf   (static single frame)
  - videos/trajectory_on_env_<task>.mp4    (animated playhead reveal,
                                            ~5 s @ 30 fps)

Camera projection uses MuJoCo's built-in pinhole model derived
from ``model.cam_fovy`` plus the world->camera transform from
``data.cam_xpos`` and ``data.cam_xmat``.  Image y-axis is
flipped (OpenGL convention) to match the saved RGB frame.

    # Render one task
    python -m presentation.scripts.render_trajectory_on_env \\
        --task can --n-candidates 24 --fps 30 --duration-s 5

    # Render all five tasks in a single invocation
    python -m presentation.scripts.render_trajectory_on_env \\
        --task all --n-candidates 24 --fps 30 --duration-s 5
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "presentation" / "figures"
VID_DIR = REPO_ROOT / "presentation" / "videos"

# Per-task HDF5 dataset paths.  All five robomimic ph datasets are
# bundled by the project's conda env under ``site-packages/datasets``.
# Override via ``--data`` to point at a custom HDF5 (e.g. the
# mass-modified can bank used for the headline experiments).
_DATASETS_ROOT = Path(
    "/Users/szh/anaconda3/envs/hybrid-traj/lib/python3.11/site-packages/datasets"
)
TASK_HDF5: dict[str, Path] = {
    "lift":      _DATASETS_ROOT / "lift"      / "ph" / "low_dim_v141.hdf5",
    "can":       _DATASETS_ROOT / "can"       / "ph" / "low_dim_v141.hdf5",
    "square":    _DATASETS_ROOT / "square"    / "ph" / "low_dim_v141.hdf5",
    "tool_hang": _DATASETS_ROOT / "tool_hang" / "ph" / "low_dim_v141.hdf5",
    "transport": _DATASETS_ROOT / "transport" / "ph" / "low_dim_v141.hdf5",
}

# Project-local imports
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Env construction (offscreen rendering ON)
# ---------------------------------------------------------------------------
def make_env_with_offscreen(data_hdf5: Path, camera: str,
                            height: int, width: int):
    """Build robosuite env with offscreen renderer enabled.

    Bypasses ``experiments.common.make_env_for_dataset`` which
    forces ``has_offscreen_renderer=False``.
    """
    from experiments.common import (  # noqa: E402
        read_env_meta,
        upgrade_controller_kwargs,
    )
    import robosuite  # noqa: E402

    env_meta = read_env_meta(data_hdf5)
    kwargs = dict(env_meta.get("env_kwargs", {}))
    kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        ignore_done=True,
        camera_names=[camera],
        camera_heights=[height],
        camera_widths=[width],
    )
    kwargs = upgrade_controller_kwargs(kwargs)
    env = robosuite.make(env_meta["env_name"], **kwargs)
    return env, env_meta


def reset_env_to_demo(env, data_hdf5: Path, demo_key: str) -> None:
    """Reset the env's MuJoCo state to a demo's first frame."""
    from experiments.common import set_sim_state

    with h5py.File(str(data_hdf5), "r") as f:
        state0 = f[f"data/{demo_key}/states"][0]
    env.reset()
    set_sim_state(env, state0)


# ---------------------------------------------------------------------------
# Pinhole camera projection from MuJoCo camera frame
# ---------------------------------------------------------------------------
class PinholeCamera:
    """Project (N, 3) world points to (N, 2) image pixel coords.

    Uses MuJoCo's convention: the camera looks down -Z in its own
    frame, with +X right and +Y up.  ``cam_xmat`` is the rotation
    from camera -> world; we invert it to take world -> camera.
    """

    def __init__(self, sim, camera_name: str, height: int, width: int):
        import mujoco  # noqa: E402 (lazy)

        model = sim.model._model
        data = sim.data._data
        cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if cam_id < 0:
            raise ValueError(f"no camera named {camera_name!r} in model")

        self.height = height
        self.width = width
        self.cam_pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64).copy()
        # cam_xmat is camera-in-world rotation, stored row-major (9,)
        self.R_cw = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)

        fovy_deg = float(model.cam_fovy[cam_id])
        self.fy = 0.5 * height / np.tan(np.radians(fovy_deg) * 0.5)
        self.fx = self.fy  # assume square pixels
        self.cx = width / 2.0
        self.cy = height / 2.0

    def project(self, p_world: np.ndarray) -> np.ndarray:
        """Project (N, 3) world points -> (N, 2) pixel coords.

        Returns NaN for points behind the camera.
        """
        p_world = np.asarray(p_world, dtype=np.float64).reshape(-1, 3)
        # World -> camera: p_c = R_cw.T @ (p_w - cam_pos)
        rel = p_world - self.cam_pos[None, :]
        p_cam = rel @ self.R_cw  # since R_cw is (cam_axes in world) cols
        # In MuJoCo/OpenGL the camera looks down -Z, so points in
        # front have p_cam_z < 0.  Pixel coords:
        #   u =  fx * (x / -z) + cx
        #   v =  fy * (y / -z) + cy   then flip y at the end
        x, y, z = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]
        in_front = z < 0.0
        safe_z = np.where(in_front, z, np.nan)
        u = self.fx * (x / -safe_z) + self.cx
        v = self.fy * (y / -safe_z) + self.cy
        # OpenGL image y-axis points UP; matplotlib image y points
        # DOWN.  Flip v so pixel (0, 0) is top-left.
        v = self.height - v
        return np.stack([u, v], axis=-1)


# ---------------------------------------------------------------------------
# Data loader (same heuristic as the ensemble plot)
# ---------------------------------------------------------------------------
def load_eef_trajectories(
    data_hdf5: Path,
    n_candidates: int,
    optimal_strategy: str = "shortest",
    task_filter: bool = True,
    cluster_radius_xy: float = 0.04,
) -> tuple[list[np.ndarray], int, list[str]]:
    """Load N candidate EEF trajectories.

    When ``task_filter`` is True (default), the candidates are
    restricted to demos whose can starting xy position is within
    ``cluster_radius_xy`` metres of the optimal demo's can xy.
    This gives "multiple roll-outs on the *same* task" visual
    rather than "multiple roll-outs on different tasks", which
    is what MBD's denoise-ensemble figure actually shows.

    The optimal demo itself is always included, and is selected
    from the unfiltered pool by ``optimal_strategy`` so that we
    don't bias the centre of the cluster.
    """
    with h5py.File(str(data_hdf5), "r") as f:
        all_keys = sorted(f["data"].keys(),
                          key=lambda k: int(k.split("_")[1]))
        # Cache: (T, 3) eef + (3,) can_pos_t0 + T for every demo
        cache: list[tuple[np.ndarray, np.ndarray, int]] = []
        for k in all_keys:
            eef = np.asarray(f[f"data/{k}/obs/robot0_eef_pos"][:],
                             dtype=np.float64)
            can_t0 = np.asarray(f[f"data/{k}/obs/object"][0, :3],
                                dtype=np.float64)
            cache.append((eef, can_t0, eef.shape[0]))

    horizons_all = [c[2] for c in cache]
    if optimal_strategy == "shortest":
        opt_idx_global = int(np.argmin(horizons_all))
    elif optimal_strategy == "first":
        opt_idx_global = 0
    else:
        opt_idx_global = 0
    opt_key = all_keys[opt_idx_global]
    opt_can_xy = cache[opt_idx_global][1][:2]

    # Filter the pool to demos with similar can starting xy
    if task_filter:
        d_xy = np.array([np.linalg.norm(c[1][:2] - opt_can_xy)
                         for c in cache])
        pool = [i for i in range(len(cache))
                if d_xy[i] <= cluster_radius_xy]
        # Always include the optimal demo
        if opt_idx_global not in pool:
            pool.append(opt_idx_global)
        # Sort by distance for deterministic selection
        pool.sort(key=lambda i: d_xy[i])
        # If the cluster is too small, fall back to nearest-N
        if len(pool) < min(n_candidates, 6):
            order = np.argsort(d_xy)
            pool = order[:n_candidates].tolist()
            if opt_idx_global not in pool:
                pool[-1] = opt_idx_global
        # Truncate to n_candidates, but guarantee the optimal demo
        # survives the truncation.  When every demo has identical
        # object xy (e.g. tool_hang has a fixed tool start), the
        # stable-sort tie-break may push opt outside the top-N.
        pool = pool[:n_candidates]
        if opt_idx_global not in pool:
            pool[-1] = opt_idx_global
    else:
        if len(cache) > n_candidates:
            idxs = np.linspace(0, len(cache) - 1, n_candidates,
                               dtype=int)
            pool = idxs.tolist()
        else:
            pool = list(range(len(cache)))
        if opt_idx_global not in pool:
            pool[0] = opt_idx_global

    chosen = [all_keys[i] for i in pool]
    trajs = [cache[i][0] for i in pool]
    can_xys = [cache[i][1][:2] for i in pool]
    opt_local = chosen.index(opt_key)

    # Diagnostic: cluster tightness
    if task_filter:
        d_pool = np.array([np.linalg.norm(xy - opt_can_xy)
                           for xy in can_xys])
        print(f"[trajectory_on_env] task_filter ON: kept {len(pool)} "
              f"demos within {cluster_radius_xy*100:.1f} cm of "
              f"demo {opt_key}'s can xy; max d = "
              f"{d_pool.max()*100:.1f} cm, mean d = "
              f"{d_pool.mean()*100:.1f} cm")
    return trajs, opt_local, chosen


# ---------------------------------------------------------------------------
# Render: static + animated
# ---------------------------------------------------------------------------
def _draw_frame(
    bg_img: np.ndarray,
    projected: list[np.ndarray],
    opt_idx: int,
    cut: int | None,
    cand_colors: list,
    title: str,
):
    H, W, _ = bg_img.shape
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(bg_img)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_xticks([]); ax.set_yticks([])

    n = len(projected)
    for i, pix in enumerate(projected):
        if i == opt_idx:
            continue
        seg = pix if cut is None else pix[:cut]
        # Mask out off-screen points (NaN or out-of-bounds)
        valid = (~np.isnan(seg[:, 0])) & (seg[:, 0] > -5) & \
                (seg[:, 0] < W + 5) & (seg[:, 1] > -5) & (seg[:, 1] < H + 5)
        if valid.sum() < 2:
            continue
        ax.plot(seg[valid, 0], seg[valid, 1],
                color=cand_colors[i], lw=1.4, alpha=0.65)

    pix_opt = projected[opt_idx] if cut is None else projected[opt_idx][:cut]
    valid = (~np.isnan(pix_opt[:, 0]))
    if valid.sum() >= 2:
        ax.plot(pix_opt[valid, 0], pix_opt[valid, 1],
                color="#c0392b", lw=3.0)
        last = pix_opt[valid][-1]
        ax.scatter([last[0]], [last[1]], s=60, c="#c0392b",
                   edgecolors="white", lw=1.2, zorder=5)

    proxy_cand = Line2D([], [], color=cand_colors[0], lw=1.4,
                        label=f"{n - 1} candidate roll-outs")
    proxy_opt = Line2D([], [], color="#c0392b", lw=3.0,
                       label="optimal demo")
    ax.legend(handles=[proxy_cand, proxy_opt], loc="upper left",
              fontsize=10, frameon=True, framealpha=0.9)
    ax.set_title(title, fontsize=11)
    fig.tight_layout(pad=0.4)
    return fig


def render_static(bg_img, projected, opt_idx, cand_colors, out: Path,
                  *, title: str | None = None) -> None:
    if title is None:
        title = "End-effector trajectory ensemble overlaid on env"
    fig = _draw_frame(
        bg_img, projected, opt_idx, cut=None, cand_colors=cand_colors,
        title=title,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[trajectory_on_env] wrote static  -> {out}")


def render_animation(
    bg_img, projected, opt_idx, cand_colors, out: Path,
    fps: int = 30, duration_s: float = 5.0,
    *, title: str | None = None,
) -> None:
    if title is None:
        title = "End-effector trajectory ensemble overlaid on env"
    # Resample every projected path to a common length for the playhead
    n_pts = 80
    proj_resampled = []
    for p in projected:
        if len(p) == 0:
            proj_resampled.append(p)
            continue
        idx = np.linspace(0, len(p) - 1, n_pts).astype(int)
        proj_resampled.append(p[idx])

    n_frames = int(fps * duration_s)
    frames: list[np.ndarray] = []
    for fi in range(n_frames):
        t = min(1.0, fi / max(n_frames - int(0.5 * fps), 1))
        cut = max(2, int(t * n_pts))
        fig = _draw_frame(
            bg_img, proj_resampled, opt_idx, cut=cut,
            cand_colors=cand_colors,
            title=title,
        )
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(buf)
        plt.close(fig)

    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out), frames, fps=fps, codec="libx264",
                     quality=8, macro_block_size=1)
    print(f"[trajectory_on_env] wrote animation -> {out} "
          f"({len(frames)} frames, {duration_s} s @ {fps} fps)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task", type=str, default="can",
        choices=sorted(TASK_HDF5) + ["all"],
        help="Task whose demo bank + env to render.  ``all`` renders "
             "one PDF + MP4 per task in TASK_HDF5.",
    )
    p.add_argument(
        "--data", type=Path, default=None,
        help="Override the demo HDF5 path (else uses TASK_HDF5[task]).",
    )
    p.add_argument("--n-candidates", type=int, default=24)
    p.add_argument("--optimal", type=str, default="shortest",
                   choices=["shortest", "first"])
    p.add_argument(
        "--task-filter", action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter candidates to demos with similar object starting "
             "position (so the cloud shows the same task).  Default "
             "True; use --no-task-filter to disable.",
    )
    p.add_argument(
        "--cluster-radius-xy", type=float, default=0.04,
        help="Radius (m) around the optimal demo's object xy used for "
             "task-filter clustering.  Default 4 cm.",
    )
    p.add_argument("--camera", type=str, default="agentview")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--duration-s", type=float, default=5.0)
    p.add_argument("--out-pdf", type=Path, default=None,
                   help="Override output PDF path.  Default: "
                        "figures/trajectory_on_env_<task>.pdf.")
    p.add_argument("--out-mp4", type=Path, default=None,
                   help="Override output MP4 path.  Default: "
                        "videos/trajectory_on_env_<task>.mp4.")
    p.add_argument("--static-only", action="store_true")
    return p.parse_args()


def render_one_task(task: str, data_hdf5: Path, args) -> None:
    """Render the static PDF + animated MP4 for a single task."""
    print(f"\n[trajectory_on_env] === task={task!r} data={data_hdf5} ===")

    out_pdf = (args.out_pdf if args.out_pdf is not None
               else FIG_DIR / f"trajectory_on_env_{task}.pdf")
    out_mp4 = (args.out_mp4 if args.out_mp4 is not None
               else VID_DIR / f"trajectory_on_env_{task}.mp4")

    # Load data
    trajs, opt_idx, chosen = load_eef_trajectories(
        data_hdf5, args.n_candidates, optimal_strategy=args.optimal,
        task_filter=args.task_filter,
        cluster_radius_xy=args.cluster_radius_xy,
    )
    print(f"[trajectory_on_env] {len(trajs)} candidates; "
          f"optimal = {chosen[opt_idx]} (T={len(trajs[opt_idx])})")

    # Build env and grab one background frame at the optimal demo's
    # initial state -- this gives the most realistic "stage".
    env, _ = make_env_with_offscreen(data_hdf5, args.camera,
                                     args.height, args.width)
    reset_env_to_demo(env, data_hdf5, chosen[opt_idx])
    bg = env.sim.render(width=args.width, height=args.height,
                        camera_name=args.camera)
    bg = np.flipud(bg).astype(np.uint8)
    print(f"[trajectory_on_env] background frame {bg.shape} from camera "
          f"{args.camera!r}")

    # Build the projector AFTER resetting the env so cam_xpos/xmat
    # reflect the current state.
    cam = PinholeCamera(env.sim, args.camera, args.height, args.width)

    # Project every demo
    projected = [cam.project(tr) for tr in trajs]

    cmap = plt.get_cmap("Blues")
    cand_colors = [cmap(0.30 + 0.45 * (i / max(len(trajs) - 1, 1)))
                   for i in range(len(trajs))]

    # Pretty task name for the slide title (use the env name from
    # robosuite, which is more descriptive than the task slug).
    from experiments.common import read_env_meta
    env_name = read_env_meta(data_hdf5)["env_name"]
    title = (f"End-effector trajectory ensemble overlaid on "
             f"{env_name} (task = {task})")

    render_static(bg, projected, opt_idx, cand_colors, out_pdf,
                  title=title)
    if not args.static_only:
        render_animation(bg, projected, opt_idx, cand_colors, out_mp4,
                         fps=args.fps, duration_s=args.duration_s,
                         title=title)


def main() -> None:
    args = parse_args()

    tasks = sorted(TASK_HDF5) if args.task == "all" else [args.task]
    if args.task == "all" and (args.out_pdf is not None or
                               args.out_mp4 is not None):
        raise SystemExit(
            "--out-pdf / --out-mp4 cannot be combined with --task all "
            "(filenames would collide).  Drop the overrides and the "
            "defaults trajectory_on_env_<task>.{pdf,mp4} will be used."
        )

    for task in tasks:
        data_hdf5 = (args.data if args.data is not None
                     else TASK_HDF5[task])
        if not data_hdf5.exists():
            print(f"[trajectory_on_env] SKIP {task}: dataset not found "
                  f"at {data_hdf5}")
            continue
        render_one_task(task, data_hdf5, args)


if __name__ == "__main__":
    main()
