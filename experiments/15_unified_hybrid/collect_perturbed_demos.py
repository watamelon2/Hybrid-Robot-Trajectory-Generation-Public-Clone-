"""Collect successful demos under perturbed dynamics for T2.

The T2 (incremental-adaptation) headline cell needs a retrieval bank
that includes a handful of demos collected under perturbed dynamics
(default: mass_scale=1.3 on the Can task).  Per spec
``docs/adaptation/04_incremental_prior_spec.md`` sec 1 + sec 3:

  "Append 5 fresh demos under perturbed (mass-1.3) dynamics to the
   bank."

This script produces those demos by **replaying the canonical Can
demos under perturbed dynamics** and keeping the first N that still
succeed.  Methodological note: replaying canonical action sequences
under perturbed dynamics generates *new* obs trajectories (different
states, joint configurations, contact dynamics).  The action sequence
itself is reused, but the obs the retrieval prior would index against
at runtime is the perturbed-dynamics trajectory -- which is what the
T2 bank needs.

Why not run the canonical hybrid (demo_1nn + pull-MBD) as the data-
collection policy?  That works too, but the hybrid at mass=1.3 has
~15-25% expected success per the dynamics-slate prior, so it would
take 20-35 attempts to harvest 5 successes (~30 min CPU).  Replaying
canonical demos under perturbed dynamics is cheaper (~10 min for all
200 demos on 1 worker, faster with parallel) and produces the same
deliverable: ``(obs, action)`` trajectories where the **obs is from
the perturbed-dynamics env** and the action sequence succeeds in
that env.

Output schema matches robomimic's ``low_dim`` HDF5 layout, restricted
to the obs keys consumed by ``priors_zerotrain.load_demo_bank``:

    data/
      demo_0/
        obs/
          object               (T, 14)  [pos(3), quat(4), to_eef_pos(3), to_eef_quat(4)]
          robot0_eef_pos       (T, 3)
          robot0_eef_quat      (T, 4)
          robot0_gripper_qpos  (T, 2)
          robot0_gripper_qvel  (T, 2)
          robot0_joint_pos     (T, 7)
          robot0_joint_pos_cos (T, 7)
          robot0_joint_pos_sin (T, 7)
          robot0_joint_vel     (T, 7)
        actions                (T, 7)
        states                 (T, 71)
        rewards                (T,)
        @num_samples = T
      demo_1/ ...
    data/
      @env_args = JSON-encoded env_meta (matches the source dataset's
                  env_args so downstream tools can re-instantiate the
                  same task family)
      @mass_scale, @friction_scale = perturbations applied during this
                                     collection run
      @source_dataset = path of the canonical dataset that was replayed

When ``--combine-with`` is set, also writes a SECOND HDF5 that
concatenates the canonical source dataset's demos + the new perturbed
demos in numeric demo-id order (demo_0..demo_(K-1) from source, then
demo_K..demo_(K+N-1) from this collection).  The loader is agnostic
to demo-id ordering as long as ``int(name.split("_")[1])`` sorts; we
guarantee this by renumbering the new demos to follow the source's
top index.

Usage::

    # collect 5 demos at mass=1.3
    python collect_perturbed_demos.py \\
        --task can --mass-scale 1.3 --n-demos-target 5 \\
        --output experiments/15_unified_hybrid/data/can_mass1p3_5demos.hdf5

    # collect AND produce the combined bank (canonical 200 + new 5)
    python collect_perturbed_demos.py \\
        --task can --mass-scale 1.3 --n-demos-target 5 \\
        --output experiments/15_unified_hybrid/data/can_mass1p3_5demos.hdf5 \\
        --combine-output experiments/15_unified_hybrid/data/can_mass1p3_205demos.hdf5

Spec / theory companion:
- docs/adaptation/04_incremental_prior_spec.md sec 3 (the T2 plan)
- docs/theory/01_soft_control.md sec 1 (p^model factor adaptation)
- docs/theory/03_pareto_bias_variance.md (Corollary 1, rho collapse)
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

# Add the experiments root so we can import common + env_perturbations.
for _p in (_EXPERIMENTS_DIR, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import (  # noqa: E402
    default_dataset_path,
    make_env_for_dataset,
    set_sim_state,
    check_success,
)
from env_perturbations import apply_dynamics_perturbation  # noqa: E402


# Obs keys we record per step.  Subset matches what load_demo_bank
# requests for the Can task via priors_zerotrain.get_obs_keys("can"):
# Can_pos + Can_quat come from the synthesised ``object`` field;
# the eight robot0_* keys come from the live robosuite obs dict.
_OBJECT_FIELD_KEYS_BY_TASK: dict[str, tuple[str, ...]] = {
    # For each task, the robosuite obs-dict keys whose
    # concatenation matches the canonical dataset's ``object`` field
    # layout.  Order matters.
    #
    # Can / Square: [obj_pos(3), obj_quat(4), to_eef_pos(3), to_eef_quat(4)] = 14 dims.
    # Lift: [cube_pos(3), cube_quat(4), gripper_to_cube_pos(3)] = 10 dims.
    #   The Lift env does NOT expose ``gripper_to_cube_quat`` as a
    #   separate obs key (verified 2026-05-28 on robosuite >= 1.5
    #   on the canonical Lift PH dataset whose ``object`` field is
    #   shape (T, 10) -- see dataset.py OBS_KEYS_LIFT for the
    #   per-task layout map).
    "can": ("Can_pos", "Can_quat", "Can_to_robot0_eef_pos", "Can_to_robot0_eef_quat"),
    "lift": ("cube_pos", "cube_quat", "gripper_to_cube_pos"),
    "square": (
        "SquareNut_pos", "SquareNut_quat",
        "SquareNut_to_robot0_eef_pos", "SquareNut_to_robot0_eef_quat",
    ),
}


_ROBOT_OBS_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "robot0_joint_pos",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--task", type=str, default="can",
        choices=("can", "lift", "square"),
        help="Task to collect demos for.  Default 'can'.",
    )
    p.add_argument(
        "--mass-scale", type=float, default=1.3,
        help="Mass-scale perturbation applied to the env "
             "(via env_perturbations.apply_dynamics_perturbation).  "
             "Default 1.3 per spec sec 3.",
    )
    p.add_argument(
        "--friction-scale", type=float, default=1.0,
        help="Friction-scale perturbation.  Default 1.0 (no-op).",
    )
    p.add_argument(
        "--n-demos-target", type=int, default=5,
        help="Number of SUCCESSFUL perturbed-dynamics demos to "
             "collect.  Stops as soon as this many succeed.",
    )
    p.add_argument(
        "--max-attempts", type=int, default=0,
        help="Maximum number of canonical demos to replay (each "
             "attempt is one replay under perturbed dynamics).  "
             "Default 0 = replay ALL canonical demos in order.",
    )
    p.add_argument(
        "--source-dataset", type=Path, default=None,
        help="Source dataset to replay.  Default = canonical "
             "robomimic dataset for the task (default_dataset_path).",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Output HDF5 path for the collected demos.",
    )
    p.add_argument(
        "--combine-output", type=Path, default=None,
        help="If set, also write a SECOND HDF5 that combines "
             "the canonical source dataset's demos + the new "
             "perturbed demos, in numeric demo-id order.  T2 uses "
             "this combined HDF5 directly as --prior-bank-dataset.",
    )
    p.add_argument(
        "--n-canonical-cap", type=int, default=0,
        help="When --combine-output is set, only include the first "
             "N canonical demos.  Default 0 = include ALL canonical "
             "demos (200 for Can/ph).",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for env.reset (only affects randomness "
             "robosuite injects on reset).  Defaults to 0.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print per-attempt success/fail with reward.",
    )
    return p.parse_args()


def _replay_one_demo(
    env,
    init_state: np.ndarray,
    actions: np.ndarray,
    task: str,
) -> dict:
    """Replay an action sequence in a (possibly perturbed) env.

    Returns a dict with: success (bool), final_reward (float),
    per-step obs dict (each value is a list of arrays of length T),
    per-step states (list of (state_vec,) T times), and per-step
    rewards (list of float T times).
    """
    env.reset()
    set_sim_state(env, init_state)

    # Robosuite returns observations only AFTER an env.step(), so we
    # take a no-op first step to get the initial obs.  But the demos
    # we record are aligned with action steps -- so per-step
    # recording starts AFTER each action's env.step().
    object_keys = _OBJECT_FIELD_KEYS_BY_TASK[task]
    obs_log: dict[str, list[np.ndarray]] = {
        "object": [],
        **{k: [] for k in _ROBOT_OBS_KEYS},
    }
    states_log: list[np.ndarray] = []
    rewards_log: list[float] = []

    for a in actions:
        obs, reward, _done, _info = env.step(a)
        # Build object field by concatenating the four sub-keys.
        # If any sub-key is missing (e.g. robosuite version diff),
        # this raises a clear KeyError early.
        try:
            object_field = np.concatenate(
                [np.asarray(obs[k], dtype=np.float32).reshape(-1)
                 for k in object_keys]
            ).astype(np.float32)
        except KeyError as e:
            raise KeyError(
                f"Missing obs key {e} for task={task!r}.  Expected "
                f"sub-keys {object_keys}; env returned "
                f"{sorted(obs.keys())[:10]}...  Check "
                f"_OBJECT_FIELD_KEYS_BY_TASK in this script."
            ) from e
        obs_log["object"].append(object_field)
        for k in _ROBOT_OBS_KEYS:
            obs_log[k].append(np.asarray(obs[k], dtype=np.float32))

        # Per-step sim state (robosuite stores 71-vector for Can)
        states_log.append(np.asarray(env.sim.get_state().flatten(), dtype=np.float64))
        rewards_log.append(float(reward))

    success = check_success(env)
    final_reward = float(rewards_log[-1]) if rewards_log else 0.0
    return {
        "success": success,
        "final_reward": final_reward,
        "obs_log": {k: np.stack(v, axis=0) for k, v in obs_log.items()},
        "states": np.stack(states_log, axis=0),
        "rewards": np.asarray(rewards_log, dtype=np.float64),
    }


def _read_canonical_demos(
    source_path: Path,
    n_cap: int = 0,
) -> tuple[list[str], list[np.ndarray], list[np.ndarray], dict]:
    """Read (demo_ids, init_states, action_sequences, env_args) from
    a robomimic-style HDF5.  Init state per demo is ``states[0]``.
    """
    with h5py.File(str(source_path), "r") as h5:
        all_demos = sorted(
            h5["data"].keys(), key=lambda k: int(k.split("_")[1])
        )
        if n_cap and n_cap > 0:
            all_demos = all_demos[:n_cap]

        env_args_raw = h5["data"].attrs["env_args"]
        if isinstance(env_args_raw, bytes):
            env_args_raw = env_args_raw.decode("utf-8")
        env_meta = json.loads(env_args_raw)

        init_states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        for d in all_demos:
            grp = h5["data"][d]
            init_states.append(np.asarray(grp["states"][0], dtype=np.float64))
            actions.append(np.asarray(grp["actions"][()], dtype=np.float32))

    return all_demos, init_states, actions, env_meta


def _write_demos_hdf5(
    out_path: Path,
    demos: list[dict],
    env_meta: dict,
    *,
    source_dataset: Path,
    mass_scale: float,
    friction_scale: float,
    starting_demo_id: int = 0,
) -> None:
    """Write the collected demos to an HDF5 in robomimic schema."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    total_steps = 0
    with h5py.File(str(out_path), "w") as h5:
        data_grp = h5.create_group("data")
        for i, d in enumerate(demos):
            demo_id = f"demo_{starting_demo_id + i}"
            g = data_grp.create_group(demo_id)
            obs_g = g.create_group("obs")
            for k, v in d["obs_log"].items():
                obs_g.create_dataset(k, data=v.astype(np.float32))
            g.create_dataset("actions", data=d["actions"].astype(np.float32))
            g.create_dataset("states", data=d["states"].astype(np.float64))
            g.create_dataset("rewards", data=d["rewards"].astype(np.float64))
            T = int(d["actions"].shape[0])
            g.attrs["num_samples"] = T
            total_steps += T

        data_grp.attrs["env_args"] = json.dumps(env_meta)
        data_grp.attrs["total"] = total_steps
        data_grp.attrs["mass_scale"] = float(mass_scale)
        data_grp.attrs["friction_scale"] = float(friction_scale)
        data_grp.attrs["source_dataset"] = str(source_dataset)


def _write_combined_hdf5(
    out_path: Path,
    source_path: Path,
    new_demos: list[dict],
    *,
    n_canonical_cap: int = 0,
    mass_scale: float = 1.0,
    friction_scale: float = 1.0,
) -> None:
    """Concatenate (source_demos[:n_canonical_cap], new_demos) into
    a single HDF5 with sequential demo_ids.  Numerical IDs are
    renumbered so the loader (which sorts on int(name.split('_')[1]))
    sees source demos first, then new demos.

    Uses h5py's group copy for the source demos (fast, no Python-side
    re-encoding) and the per-demo writer above for the new ones.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    with h5py.File(str(source_path), "r") as src, h5py.File(str(out_path), "w") as dst:
        data_dst = dst.create_group("data")
        # Carry over data-group attrs (env_args + total).  We'll
        # overwrite "total" after we know the final step count.
        for k, v in src["data"].attrs.items():
            data_dst.attrs[k] = v

        # Source demos in numeric order, capped.
        src_demos = sorted(
            src["data"].keys(), key=lambda k: int(k.split("_")[1])
        )
        if n_canonical_cap and n_canonical_cap > 0:
            src_demos = src_demos[:n_canonical_cap]

        total = 0
        for i, name in enumerate(src_demos):
            dst_name = f"demo_{i}"
            src.copy(src["data"][name], data_dst, name=dst_name)
            if "num_samples" in data_dst[dst_name].attrs:
                total += int(data_dst[dst_name].attrs["num_samples"])
            elif "actions" in data_dst[dst_name]:
                total += int(data_dst[dst_name]["actions"].shape[0])

        # New demos with sequential IDs after the source.
        next_id = len(src_demos)
        for i, d in enumerate(new_demos):
            demo_id = f"demo_{next_id + i}"
            g = data_dst.create_group(demo_id)
            obs_g = g.create_group("obs")
            for k, v in d["obs_log"].items():
                obs_g.create_dataset(k, data=v.astype(np.float32))
            g.create_dataset("actions", data=d["actions"].astype(np.float32))
            g.create_dataset("states", data=d["states"].astype(np.float64))
            g.create_dataset("rewards", data=d["rewards"].astype(np.float64))
            T = int(d["actions"].shape[0])
            g.attrs["num_samples"] = T
            total += T

        data_dst.attrs["total"] = total
        # Mark this HDF5 as an incremental-augmented bank for
        # traceability; T2's analysis pipeline reads these.
        data_dst.attrs["incremental_mass_scale"] = float(mass_scale)
        data_dst.attrs["incremental_friction_scale"] = float(friction_scale)
        data_dst.attrs["n_canonical_demos"] = len(src_demos)
        data_dst.attrs["n_perturbed_demos"] = len(new_demos)


def main() -> int:
    args = parse_args()

    source_path = args.source_dataset or default_dataset_path(args.task)
    if not source_path.exists():
        print(f"[collect] source dataset not found: {source_path}", file=sys.stderr)
        return 1

    print(f"[collect] task={args.task}  mass_scale={args.mass_scale}  "
          f"friction_scale={args.friction_scale}  "
          f"target={args.n_demos_target}  source={source_path}")

    # Read canonical demos.
    t0 = time.perf_counter()
    src_demo_ids, init_states, actions, env_meta = _read_canonical_demos(
        source_path,
        n_cap=args.max_attempts,
    )
    print(f"[collect] read {len(src_demo_ids)} canonical demos in "
          f"{time.perf_counter() - t0:.2f}s")

    # Build env + perturbation.
    print(f"[collect] building env (this may take ~10s)...")
    t0 = time.perf_counter()
    env, env_meta_live = make_env_for_dataset(source_path)
    apply_dynamics_perturbation(
        env, args.task,
        mass_scale=float(args.mass_scale),
        friction_scale=float(args.friction_scale),
    )
    print(f"[collect] env built in {time.perf_counter() - t0:.2f}s "
          f"(perturbation applied)")

    # Replay loop.
    succ_demos: list[dict] = []
    t_loop = time.perf_counter()
    for idx, (demo_id, init_state, demo_actions) in enumerate(
        zip(src_demo_ids, init_states, actions)
    ):
        result = _replay_one_demo(env, init_state, demo_actions, args.task)
        if args.verbose or (idx % 10 == 0):
            tag = "OK" if result["success"] else "..."
            print(f"[collect] {idx + 1:3d}/{len(src_demo_ids)} "
                  f"{demo_id}: {tag} reward={result['final_reward']:+.3f}  "
                  f"successes={len(succ_demos)}/{args.n_demos_target}")
        if result["success"]:
            result["actions"] = demo_actions
            result["source_demo_id"] = demo_id
            succ_demos.append(result)
            if len(succ_demos) >= args.n_demos_target:
                break

    wall = time.perf_counter() - t_loop
    print(f"[collect] done: {len(succ_demos)} successes / "
          f"{idx + 1} attempts in {wall:.1f}s  "
          f"(succ-rate {100 * len(succ_demos) / (idx + 1):.1f}%)")

    if not succ_demos:
        print("[collect] no successful demos collected -- nothing to "
              "write.  Try a smaller mass_scale or run --verbose.",
              file=sys.stderr)
        return 2

    if len(succ_demos) < args.n_demos_target:
        print(f"[collect] WARNING: only {len(succ_demos)} successes "
              f"(target was {args.n_demos_target}).  Writing what we "
              f"have; downstream T2 cells will use fewer perturbed "
              f"demos than planned.")

    # Write perturbed-only HDF5.
    _write_demos_hdf5(
        args.output,
        succ_demos,
        env_meta,
        source_dataset=source_path,
        mass_scale=args.mass_scale,
        friction_scale=args.friction_scale,
    )
    print(f"[collect] wrote {len(succ_demos)} perturbed demos -> "
          f"{args.output}")

    # Optional combined HDF5.
    if args.combine_output is not None:
        _write_combined_hdf5(
            args.combine_output,
            source_path,
            succ_demos,
            n_canonical_cap=args.n_canonical_cap,
            mass_scale=args.mass_scale,
            friction_scale=args.friction_scale,
        )
        with h5py.File(str(args.combine_output), "r") as h5:
            n_total = len(h5["data"].keys())
        print(f"[collect] wrote combined HDF5 with {n_total} demos -> "
              f"{args.combine_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
