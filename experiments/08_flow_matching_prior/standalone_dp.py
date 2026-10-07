"""Standalone Diffusion-Policy-style rollout (no MBD wrapper).

This eval treats a trained ``v_theta`` checkpoint as a *standalone
policy*, mirroring Chi 2023 § 5.3:

    while not done:
        obs        = env.get_obs()
        chunk_120  = prior.sample_initial_mean(obs)   # Heun ODE
        for t in range(n_action_steps):
            env.step(chunk_120[t])
        # then re-observe and re-plan

No MBD, no cost-aware denoising, no per-step bridge.  The point of
this eval is to establish the **standalone SOTA ceiling** for our
trained backbones on this Can slice — answering "would Diffusion
Policy by itself reach the published ~95 % on our 200-demo split?".

Why this script exists rather than reusing ``run_can.py``:

* ``run_can.py`` is the MBD wrapper, and there is no clean way to
  ask MBD to do "exactly nothing" — even with K=1, λ=0, M=0, MBD
  still applies one cost-guided denoising step that may modify
  the warm-start.  Sidestepping MBD entirely here gives a clean
  *just-the-prior* number for the brief table.

Init-state protocol (2026-05-28 audit, Phase-A reproduction):

The original protocol replayed training-demo init states only
(``--n-demos`` × ``--n-rollouts`` train inits).  DP's published
85 % number on Can/ph is dominated by fresh-seed test rollouts
(``n_train=6 + n_test=50`` per ``can_lowdim.yaml``), where each
test rollout calls ``np.random.seed(test_start_seed+i)`` then
``env.reset()`` so robosuite's domain-randomisation produces a
fresh init.  Pass ``--n-test-rollouts 50 --test-start-seed 100000``
to add the DP-style fresh-seed bucket.  The output JSON now
reports ``train_succ_rate`` and ``test_succ_rate`` separately;
the latter is the DP apples-to-apples reference.

Output: a JSON file in ``--output-dir`` recording per-rollout
``(init_kind, demo, rollout_idx, init_seed, final_reward, success)``
plus a summary with ``train_*`` / ``test_*`` / overall ``n``,
``n_succ``, ``succ_rate``.

Usage (legacy, train inits only)::

    python experiments/08_flow_matching_prior/standalone_dp.py \\
        --task can \\
        --prior-checkpoint .../v_theta.pt \\
        --n-demos 10 --n-rollouts 3 \\
        --n-action-steps 8 --max-env-steps 400 \\
        --output-dir .../results/

Usage (DP-aligned reproduction protocol)::

    python experiments/08_flow_matching_prior/standalone_dp.py \\
        --task can \\
        --prior-checkpoint .../v_theta.pt \\
        --n-demos 10 --n-rollouts 1 \\
        --n-test-rollouts 50 --test-start-seed 100000 \\
        --n-action-steps 8 --max-env-steps 400 \\
        --output-dir .../results/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

# Bring the relevant sibling experiments onto the path the same way
# run_can.py does.  Order: 15 (driver utils) > 03 (dataset/obs_keys)
# > 08 (adapter), so 'common' resolves to 15's version.
_THIS_DIR = Path(__file__).resolve().parent
_EXP_DIR = _THIS_DIR.parent
for _p in (
    _EXP_DIR / "15_unified_hybrid",
    _EXP_DIR / "03_bc_baseline",
    _THIS_DIR,
    _EXP_DIR,
):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from adapter import FlowPrior  # noqa: E402
from common import (  # noqa: E402
    check_success,
    default_dataset_path,
    make_env_for_dataset,
    set_sim_state,
)
# Task-aware obs helpers (added Phase-B, 2026-05-29).  ``OBS_KEYS`` is
# kept as the legacy default for backward compatibility, but the eval
# pipeline now resolves the actual key list from the ckpt's ``task``
# field so DP-schema ckpts (``task='can_dp'``) get the 23-D obs vector
# including the recomputed ``Can_to_eef_*_synth`` features.
from dataset import (  # noqa: E402
    OBS_KEYS,
    get_obs_keys,
    get_synthetic_keys,
    compute_synthetic_obs_keys,
)


def make_obs_vec(
    env,
    obs_keys: tuple[str, ...] = OBS_KEYS,
    synth_keys: dict[str, tuple[str, ...]] | None = None,
) -> np.ndarray:
    """Flat obs concatenation in the prior's expected key order.

    Reads each *base* key directly from the env's obs dict, then
    computes any *synthetic* keys (e.g. ``Can_to_eef_pos_synth``)
    from the just-loaded base keys via
    ``compute_synthetic_obs_keys``.  Same function used at training
    time in ``dataset_v3.load_chunks``, so train and runtime feature
    vectors are bit-identical.

    ``synth_keys`` is the task's synthetic-key registry (see
    ``get_synthetic_keys(task)``).  ``None`` means "no synthetics"
    (legacy 46-D path).
    """
    obs_dict_env = env._get_observations(force_update=True)
    obs_dict: dict[str, np.ndarray] = {}
    synth_keys = synth_keys or {}
    # First pass: load every base (non-synthetic) key directly from
    # the env's obs dict.  Synthetic keys are deferred to a second
    # pass so all their dependencies are already loaded.
    for k in obs_keys:
        if k in synth_keys:
            continue
        if k not in obs_dict_env:
            raise KeyError(
                f"runtime obs missing {k!r} "
                f"(available: {sorted(obs_dict_env.keys())})"
            )
        obs_dict[k] = np.asarray(obs_dict_env[k], dtype=np.float32).reshape(-1)
    # Second pass: compute synthetics in-place.
    if synth_keys:
        compute_synthetic_obs_keys(obs_dict, synth_keys)
    # Concatenate in the task's declared order.
    parts = [obs_dict[k].astype(np.float32) for k in obs_keys]
    return np.concatenate(parts).astype(np.float32)


def standalone_rollout(
    env,
    prior: FlowPrior,
    init_state: np.ndarray | None,
    *,
    n_action_steps: int,
    max_env_steps: int,
    n_sample_steps: int,
    seed: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    use_obs_buffer: bool = True,
    init_seed: int | None = None,
    obs_keys: tuple[str, ...] = OBS_KEYS,
    synth_keys: dict[str, tuple[str, ...]] | None = None,
) -> dict:
    """Run one rollout and return ``{final_reward, success, n_steps}``.

    Strategy: action-chunking inference, optionally closed-loop in
    the obs-buffer sense.

      1. Reset env to ``init_state`` (and reset the prior's obs buffer
         if it has one).
      2. Observe ``s_t`` from the env; push it onto the prior's obs
         buffer if the prior supports it (``T_obs > 1`` mode).
      3. Sample an H-step chunk conditioned on the buffered history.
      4. Execute the first ``n_action_steps`` actions of that chunk.
      5. If not done and not at ``max_env_steps``, re-observe and
         re-plan.

    The chunked-replan loop is "closed-loop" already in the sense
    that we re-observe and re-plan every ``n_action_steps`` env
    steps.  ``use_obs_buffer=True`` adds *obs history* into the
    conditioning: for ``T_obs=2`` priors this means each replan sees
    ``[s_{previous_replan}, s_{current}]`` instead of just
    ``[s_{current}, s_{current}]`` (stateless replication).  This
    matches DP's training distribution when ``n_action_steps`` is
    small (their published config uses 8).

    Init-state semantics:

    * ``init_state is not None``: reset env, then restore the
      MuJoCo state to ``init_state`` (replay of a training-demo
      initial condition).  Legacy behaviour.

    * ``init_state is None`` and ``init_seed is not None``: seed
      numpy's global RNG with ``init_seed``, then call ``env.reset()``.
      Robosuite's domain-randomisation reads ``np.random`` internally,
      so this reproduces DP's ``test_start_seed + i`` fresh-init
      protocol (see ``robomimic_lowdim_wrapper.py:reset``).
    """
    if init_state is None:
        if init_seed is None:
            raise ValueError(
                "standalone_rollout: must pass exactly one of "
                "init_state (train-init replay) or init_seed "
                "(DP-style fresh-seed reset).",
            )
        # DP-style fresh-seed protocol.  Robosuite 1.5+ uses a private
        # ``np.random.default_rng()`` instance inside its placement
        # initialiser, NOT the numpy global RNG that DP's wrapper
        # ``robomimic_lowdim_wrapper.py:reset`` seeds.  So
        # ``np.random.seed`` alone has no effect on can placement.
        # We replace the sampler's ``rng`` directly.  For a composite
        # sampler we also reseed its child samplers (the can-only
        # sampler stores its own rng reference at construction time).
        # Requires ``env.hard_reset = False`` (set in main()) so reset
        # doesn't rebuild the model and clobber the rng we just set.
        rng = np.random.default_rng(int(init_seed))
        pi = getattr(env, "placement_initializer", None)
        if pi is not None:
            pi.rng = rng
            child = getattr(pi, "samplers", None)
            if isinstance(child, dict):
                for s in child.values():
                    s.rng = rng
        # Belt-and-braces: also seed the numpy global state to mimic
        # DP's wrapper exactly, in case some piece of robosuite still
        # reads it (e.g., dynamics randomisation if enabled).
        np.random.seed(int(init_seed))
        env.reset()
    else:
        env.reset()
        set_sim_state(env, init_state)

    # Reset the prior's obs buffer if it has one.  v3 / v6 priors
    # don't, hence the ``hasattr`` guard.
    if use_obs_buffer and hasattr(prior, "reset_obs_buffer"):
        prior.reset_obs_buffer()

    total_reward = 0.0
    success = False
    steps_executed = 0

    # Per-rollout, per-chunk seed so warm-starts are reproducible.
    chunk_idx = 0
    while steps_executed < max_env_steps:
        # Observe.
        s_vec = make_obs_vec(env, obs_keys=obs_keys, synth_keys=synth_keys)

        # Push onto the obs buffer if supported.  ``observe`` switches
        # the adapter into stateful mode; subsequent ``sample_initial_mean``
        # uses the buffer instead of replicating s_vec.
        if use_obs_buffer and hasattr(prior, "observe"):
            prior.observe(s_vec)

        # Sample H actions from the prior.
        chunk = prior.sample_initial_mean(
            s_vec,
            n_steps=int(n_sample_steps),
            seed=int(seed) + chunk_idx * 1009,
        )  # (H, A)
        chunk = np.clip(chunk, action_low[None, :], action_high[None, :])

        # Execute the next ``n_action_steps`` actions.
        steps_this_chunk = min(
            int(n_action_steps), max_env_steps - steps_executed,
        )
        for t in range(steps_this_chunk):
            action = chunk[t]
            _obs, reward, _done, _info = env.step(action)
            total_reward += float(reward)
            steps_executed += 1
            # Check success after every step; bail out as soon as it
            # succeeds so we don't waste env steps.
            if check_success(env):
                success = True
                break
        if success:
            break
        chunk_idx += 1

    return {
        "final_reward": total_reward,
        "success": bool(success),
        "n_steps": int(steps_executed),
        "n_chunks": int(chunk_idx + 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="can")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument(
        "--prior-checkpoint", type=Path, required=True,
        help="A v3 or v4 checkpoint.  Dispatcher in adapter.py "
        "auto-detects the format.",
    )
    p.add_argument(
        "--n-demos", type=int, default=20,
        help="Number of training-demo initial conditions to evaluate "
        "(equivalent to DP's `n_train`).  Each demo is replayed "
        "`--n-rollouts` times.",
    )
    p.add_argument(
        "--n-rollouts", type=int, default=3,
        help="Number of repeated rollouts per training-demo init "
        "(used to estimate sample-seed variance).  Set to 1 to match "
        "DP's protocol exactly.",
    )
    p.add_argument(
        "--n-test-rollouts", type=int, default=0,
        help="Number of fresh-seed test rollouts (equivalent to DP's "
        "`n_test`).  Each uses `--test-start-seed + i` as the numpy "
        "global seed before env.reset(), reproducing DP's domain-"
        "randomisation protocol from `robomimic_lowdim_wrapper.py`.  "
        "Default 0 (legacy mode: only training inits).  Set to 50 "
        "to match DP's Can/ph config.",
    )
    p.add_argument(
        "--test-start-seed", type=int, default=100_000,
        help="First seed used for the test-rollout sweep.  DP's "
        "Can/ph config uses 100_000.",
    )
    p.add_argument(
        "--n-action-steps", type=int, default=8,
        help="How many actions to execute from each sampled chunk "
        "before re-sampling.  Diffusion Policy's published Can config "
        "uses 8; we default to that.",
    )
    p.add_argument(
        "--max-env-steps", type=int, default=400,
        help="Hard cap on env steps per rollout.  DP's Can/ph config "
        "uses 400 (= robosuite's native horizon).",
    )
    p.add_argument(
        "--n-steps", type=int, default=18,
        help="Heun sampler steps per chunk.  18 matches "
        "FlowPriorV3's default.",
    )
    p.add_argument("--sigma-max", type=float, default=0.25)
    p.add_argument("--sigma-min", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--output-name", type=str, default="standalone_dp.json")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset or default_dataset_path(args.task)

    # Build env.
    env, _meta = make_env_for_dataset(dataset)
    if hasattr(env, "reward_shaping"):
        env.reward_shaping = True
    # Disable hard reset so the placement initialiser (and its rng,
    # which we re-seed per test-rollout) is not clobbered by model
    # reconstruction.  Robosuite 1.5+ rebuilds the entire model on
    # hard_reset=True and re-creates the sampler with a fresh
    # ``np.random.default_rng()`` -- which would defeat our seeding.
    if hasattr(env, "hard_reset"):
        env.hard_reset = False
    # Bring the sim + placement initialiser online once before any
    # rollout-specific reset so subsequent resets are soft.
    env.reset()

    # Action bounds from env.
    low = np.asarray(env.action_spec[0], dtype=np.float32)
    high = np.asarray(env.action_spec[1], dtype=np.float32)

    # Resolve the obs schema from the ckpt blob's ``task`` field BEFORE
    # loading the adapter, so we can sanity-check obs_dim against the
    # model and produce a runtime obs vector that bit-matches the one
    # the model was trained on.  ``--task`` on the CLI is an override
    # (useful when the ckpt was saved before ``task`` was recorded).
    # The peek is cheap -- we re-load the same file inside FlowPrior
    # but the blob is small and disk-resident.
    _ckpt_blob = torch.load(
        str(args.prior_checkpoint), map_location="cpu", weights_only=False,
    )
    _ckpt_task = str(_ckpt_blob.get("task", "")).strip()
    del _ckpt_blob
    if (
        _ckpt_task
        and args.task
        and _ckpt_task == f"{args.task}_dp"
    ):
        resolved_task = _ckpt_task
    elif args.task and args.task != "can":
        # Explicit --task override.  Useful when the ckpt predates the
        # ``task`` field or when force-evaluating a ``can`` ckpt under
        # a non-default obs schema.
        resolved_task = args.task
    elif _ckpt_task:
        resolved_task = _ckpt_task
    else:
        resolved_task = args.task or "can"
    obs_keys = get_obs_keys(resolved_task)
    synth_keys = get_synthetic_keys(resolved_task)

    # Load prior (auto-dispatches to FlowPriorV3 / FlowPrior).
    prior = FlowPrior.from_checkpoint(
        args.prior_checkpoint,
        device=args.device,
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
    )
    # Sanity-check: the obs vector we build at runtime must have the
    # exact dimensionality the loaded prior expects.  A mismatch here
    # is almost always a schema bug (ckpt was trained on a different
    # ``--task``) and we want to fail loudly BEFORE the first rollout,
    # not silently slice a tensor and produce nonsense.
    _probe_obs = make_obs_vec(env, obs_keys=obs_keys, synth_keys=synth_keys)
    _expected = int(prior.config.obs_dim)
    if int(_probe_obs.shape[0]) != _expected:
        raise SystemExit(
            f"[standalone_dp] obs_dim mismatch: built {_probe_obs.shape[0]}-D "
            f"vector for task={resolved_task!r} (keys={obs_keys}), but the "
            f"loaded prior expects obs_dim={_expected}.  Re-check the "
            f"ckpt's recorded task or pass --task explicitly."
        )
    print(
        f"[standalone_dp] task={resolved_task!r}  obs_dim={_probe_obs.shape[0]}  "
        f"prior_arch={prior.config.arch}  prior_obs_dim={_expected}  "
        f"prior_T_obs={getattr(prior.config, 'n_obs_steps', 1)}",
        flush=True,
    )

    # Demos.
    with h5py.File(str(dataset), "r") as h5:
        demo_keys = sorted(
            h5["data"].keys(), key=lambda k: int(k.split("_")[1]),
        )
        if args.n_demos > 0:
            demo_keys = demo_keys[: args.n_demos]
        init_states = {
            k: np.asarray(h5[f"data/{k}/states"][0]) for k in demo_keys
        }

    per_rollout: list[dict] = []
    total_planned = (
        args.n_demos * args.n_rollouts + int(args.n_test_rollouts)
    )
    t_outer = time.perf_counter()

    # ---- Train-init rollouts (legacy protocol) -----------------------
    for i, key in enumerate(demo_keys):
        init_state = init_states[key]
        for r_idx in range(args.n_rollouts):
            # Per-rollout torch seed for reproducible warm-starts.
            rollout_seed = int(np.uint32(hash(
                ("standalone_dp", int(args.seed), key, r_idx)
            ) & 0xFFFFFFFF))
            t0 = time.perf_counter()
            result = standalone_rollout(
                env, prior, init_state,
                n_action_steps=int(args.n_action_steps),
                max_env_steps=int(args.max_env_steps),
                n_sample_steps=int(args.n_steps),
                seed=rollout_seed,
                action_low=low,
                action_high=high,
                obs_keys=obs_keys,
                synth_keys=synth_keys,
            )
            elapsed = time.perf_counter() - t0
            per_rollout.append({
                "init_kind": "train",
                "demo": key,
                "rollout_idx": int(r_idx),
                "init_seed": None,
                "final_reward": result["final_reward"],
                "success": result["success"],
                "n_steps": result["n_steps"],
                "n_chunks": result["n_chunks"],
                "elapsed_s": float(elapsed),
            })
            print(
                f"[standalone_dp]  {len(per_rollout):>3}/"
                f"{total_planned}  "
                f"train  {key}#{r_idx}  "
                f"reward={result['final_reward']:+.2f}  "
                f"success={result['success']}  "
                f"steps={result['n_steps']}  "
                f"chunks={result['n_chunks']}",
                flush=True,
            )

    # ---- Test fresh-seed rollouts (DP protocol) ----------------------
    for t_idx in range(int(args.n_test_rollouts)):
        test_seed = int(args.test_start_seed) + t_idx
        # Per-rollout torch seed for the prior is separate from
        # the env init seed (which goes to numpy globally).
        rollout_seed = int(np.uint32(hash(
            ("standalone_dp_test", int(args.seed), test_seed)
        ) & 0xFFFFFFFF))
        t0 = time.perf_counter()
        result = standalone_rollout(
            env, prior, None,
            init_seed=test_seed,
            n_action_steps=int(args.n_action_steps),
            max_env_steps=int(args.max_env_steps),
            n_sample_steps=int(args.n_steps),
            seed=rollout_seed,
            action_low=low,
            action_high=high,
            obs_keys=obs_keys,
            synth_keys=synth_keys,
        )
        elapsed = time.perf_counter() - t0
        per_rollout.append({
            "init_kind": "test",
            "demo": None,
            "rollout_idx": int(t_idx),
            "init_seed": int(test_seed),
            "final_reward": result["final_reward"],
            "success": result["success"],
            "n_steps": result["n_steps"],
            "n_chunks": result["n_chunks"],
            "elapsed_s": float(elapsed),
        })
        print(
            f"[standalone_dp]  {len(per_rollout):>3}/"
            f"{total_planned}  "
            f"test   seed={test_seed}  "
            f"reward={result['final_reward']:+.2f}  "
            f"success={result['success']}  "
            f"steps={result['n_steps']}  "
            f"chunks={result['n_chunks']}",
            flush=True,
        )

    wall = time.perf_counter() - t_outer

    # ---- Per-bucket summary ------------------------------------------
    def _bucket_stats(rows: list[dict]) -> dict:
        n = len(rows)
        n_succ = sum(int(r["success"]) for r in rows)
        return {
            "n": int(n),
            "n_succ": int(n_succ),
            "succ_rate": float(n_succ / max(n, 1)),
        }

    train_rows = [r for r in per_rollout if r["init_kind"] == "train"]
    test_rows = [r for r in per_rollout if r["init_kind"] == "test"]
    train_stats = _bucket_stats(train_rows)
    test_stats = _bucket_stats(test_rows)
    overall_stats = _bucket_stats(per_rollout)

    summary = {
        "experiment": "standalone_dp",
        "task": resolved_task,
        "task_cli": args.task,
        "obs_keys": list(obs_keys),
        "obs_dim": int(_probe_obs.shape[0]),
        "prior_checkpoint": str(args.prior_checkpoint),
        "prior_arch": str(prior.config.arch),
        "prior_backbone": str(getattr(prior.config, "backbone", "n/a")),
        "n_demos": int(args.n_demos),
        "n_rollouts": int(args.n_rollouts),
        "n_test_rollouts": int(args.n_test_rollouts),
        "test_start_seed": int(args.test_start_seed),
        "n_action_steps": int(args.n_action_steps),
        "max_env_steps": int(args.max_env_steps),
        "n_sample_steps": int(args.n_steps),
        "sigma_min": float(args.sigma_min),
        "sigma_max": float(args.sigma_max),
        # Per-bucket and overall.  ``test_*`` keys are the DP apples-to-
        # apples reference (DP's `n_test=50` rollouts dominate their
        # reported `mean_score`); ``train_*`` is the legacy protocol
        # (training-demo init states only); ``n / n_succ / succ_rate``
        # are the combined totals.
        "train_n": train_stats["n"],
        "train_n_succ": train_stats["n_succ"],
        "train_succ_rate": train_stats["succ_rate"],
        "test_n": test_stats["n"],
        "test_n_succ": test_stats["n_succ"],
        "test_succ_rate": test_stats["succ_rate"],
        "n": overall_stats["n"],
        "n_succ": overall_stats["n_succ"],
        "succ_rate": overall_stats["succ_rate"],
        "wall_s": float(wall),
        "per_rollout": per_rollout,
    }
    out = args.output_dir / args.output_name
    out.write_text(json.dumps(summary, indent=2))
    # Apples-to-apples DP comparison goes to `test_*` when n_test>0.
    n_total = overall_stats["n"]
    succ_total = overall_stats["n_succ"]
    if test_stats["n"] > 0:
        print(
            f"[standalone_dp] wrote {out}\n"
            f"  train succ = {train_stats['n_succ']}/{train_stats['n']} "
            f"= {100*train_stats['succ_rate']:.1f}%  (legacy protocol)\n"
            f"  test  succ = {test_stats['n_succ']}/{test_stats['n']} "
            f"= {100*test_stats['succ_rate']:.1f}%  (DP fresh seeds, "
            f"apples-to-apples ref)\n"
            f"  total succ = {succ_total}/{n_total} = "
            f"{100*succ_total/max(n_total,1):.1f}%   "
            f"wall={wall/60:.1f} min",
            flush=True,
        )
    else:
        print(
            f"[standalone_dp] wrote {out}  succ={succ_total}/{n_total} = "
            f"{100*succ_total/max(n_total,1):.1f}%  wall={wall/60:.1f} min",
            flush=True,
        )


if __name__ == "__main__":
    main()
