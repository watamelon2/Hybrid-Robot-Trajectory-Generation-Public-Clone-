"""
Diagnose why unified_loop's K=0 mode produces different DP plans than
standalone_dp.py for the same demo / rollout / chunk 0.

Strategy: build the env+prior once, then for demo_0 init_state, run
TWO chunk-0 plan extractions side by side:

  A) Standalone-style: env.reset() + set_sim_state(env, init_state) +
     prior.observe(s0) + prior.sample_initial_mean(s0, seed=fixed).
  B) Unified-style: same as A, then snapshot current_state, then do
     a *second* env.reset() + set_sim_state(env, current_state) +
     prior.observe(s0_b) + prior.sample_initial_mean(s0_b, seed=fixed).

If plan A != plan B, the bug is in the unified driver's
reset-snapshot-restore cycle between chunks.

Run from repo root::
    PYTHONPATH=experiments/15_unified_hybrid:experiments/04_mbd_baseline:experiments/03_imitation_baseline:experiments/08_flow_matching_prior \\
        python experiments/17_dp_hybrid/diagnose_chunk0.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Match run_can.py's sys.path setup.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
_EXP_DIR = _REPO_ROOT / "experiments"
for _p in (
    _EXP_DIR,
    _EXP_DIR / "03_bc_baseline",
    _EXP_DIR / "04_mbd_baseline",
    _EXP_DIR / "08_flow_matching_prior",
    _EXP_DIR / "15_unified_hybrid",
):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from adapter import FlowPrior  # exp 08
from common import (  # experiments/common.py
    default_dataset_path,
    make_env_for_dataset,
    set_sim_state,
)
from dataset import OBS_KEYS  # exp 03
from unified_loop import _make_obs_vec_for_prior, _snapshot_sim_state


def main():
    ckpt = _REPO_ROOT / "experiments" / "08_flow_matching_prior" / "results" / "v_theta_dp_tobs2_h16.pt"
    dataset = default_dataset_path("can")

    prior = FlowPrior.from_checkpoint(
        ckpt, device="cpu", sigma_min=0.05, sigma_max=0.25,
    )
    print(f"loaded prior, T_obs={prior.config.n_obs_steps}, H={prior.config.horizon}")

    env, _meta = make_env_for_dataset(str(dataset))
    import h5py
    with h5py.File(str(dataset), "r") as h5:
        demo_keys = sorted(
            h5["data"].keys(), key=lambda k: int(k.split("_")[1])
        )[:1]
        init_state = np.asarray(h5[f"data/{demo_keys[0]}/states"][0])
    print(f"demo {demo_keys[0]} init_state.shape = {init_state.shape}")

    # ---- Path A: standalone-style chunk-0 plan ----
    env.reset()
    set_sim_state(env, init_state)
    s0_A = _make_obs_vec_for_prior(env, obs_keys=OBS_KEYS)
    prior.reset_obs_buffer()
    prior.observe(s0_A)
    plan_A = prior.sample_initial_mean(s0_A, n_steps=100, seed=0)
    print(f"\nPATH A (standalone-style):")
    print(f"  s0_A[:5] = {s0_A[:5]}")
    print(f"  plan_A[0,:] = {plan_A[0, :]}")
    print(f"  plan_A[8,:] = {plan_A[8, :]}")
    print(f"  plan_A.mean = {plan_A.mean():.4f}, std = {plan_A.std():.4f}")

    # ---- Path B: unified-style (extra reset+snapshot+restore) ----
    env.reset()
    set_sim_state(env, init_state)
    current_state = _snapshot_sim_state(env)
    print(f"\nsnapshot == init_state? {np.allclose(current_state, init_state)}")
    print(f"  state_diff_norm = {np.linalg.norm(current_state - init_state):.6e}")

    env.reset()
    set_sim_state(env, current_state)
    s0_B = _make_obs_vec_for_prior(env, obs_keys=OBS_KEYS)
    prior.reset_obs_buffer()
    prior.observe(s0_B)
    plan_B = prior.sample_initial_mean(s0_B, n_steps=100, seed=0)
    print(f"\nPATH B (unified-style):")
    print(f"  s0_B[:5] = {s0_B[:5]}")
    print(f"  plan_B[0,:] = {plan_B[0, :]}")
    print(f"  plan_B[8,:] = {plan_B[8, :]}")
    print(f"  plan_B.mean = {plan_B.mean():.4f}, std = {plan_B.std():.4f}")

    # ---- Compare ----
    print(f"\n=== COMPARISON ===")
    s0_diff = np.linalg.norm(s0_A - s0_B)
    plan_diff = np.linalg.norm(plan_A - plan_B)
    print(f"  s0 L2 diff = {s0_diff:.6e}")
    print(f"  plan L2 diff = {plan_diff:.6e}")
    print(f"  plan max abs diff = {np.max(np.abs(plan_A - plan_B)):.6e}")

    if s0_diff > 1e-6:
        print("\n  >>> s0 vectors DIFFER -- reset+set_sim_state is not idempotent")
        # Find which obs keys differ
        idx = 0
        for k in OBS_KEYS:
            from dataset import OBS_DIMS  # in case it exists
            # fallback: count by 1
            break  # printing too much

    if plan_diff > 1e-6:
        print("  >>> plans DIFFER (likely due to s0 or torch RNG state)")
    else:
        print("  >>> plans MATCH")

    # ============================================================
    # Chunk-1 comparison: after executing chunk-0 actions, does the
    # unified driver's snapshot-reset-restore cycle produce the same
    # observation as standalone's natural propagation?
    # ============================================================
    print("\n\n===== CHUNK 1 COMPARISON =====")

    # ---- Path A_chunk1: standalone-style, no reset between chunks ----
    env.reset()
    set_sim_state(env, init_state)
    # Execute chunk-0 actions (first 8 of 16 in plan_A).
    for t in range(8):
        env.step(plan_A[t])
    s1_A = _make_obs_vec_for_prior(env, obs_keys=OBS_KEYS)
    # Now sample chunk 1 with stateful prior buffer.
    prior.reset_obs_buffer()
    prior.observe(s0_A)
    prior.observe(s1_A)
    plan_A1 = prior.sample_initial_mean(s1_A, n_steps=100, seed=1)
    print(f"PATH A (standalone-style):")
    print(f"  s1_A[:5] = {s1_A[:5]}")
    print(f"  plan_A1[0,:] = {plan_A1[0, :]}")
    print(f"  plan_A1.mean = {plan_A1.mean():.4f}, std = {plan_A1.std():.4f}")

    # ---- Path B_chunk1: unified-style, snapshot+reset+restore each chunk ----
    env.reset()
    set_sim_state(env, init_state)
    # snapshot+reset+restore for chunk 0 (the unified pattern)
    current_state_chunk0 = _snapshot_sim_state(env)
    env.reset()
    set_sim_state(env, current_state_chunk0)
    s0_B = _make_obs_vec_for_prior(env, obs_keys=OBS_KEYS)
    # Sample chunk 0 (will be identical to plan_A since we proved that).
    # Execute the same chunk-0 actions, then snapshot+reset+restore.
    for t in range(8):
        env.step(plan_A[t])  # same actions
    current_state_chunk1 = _snapshot_sim_state(env)
    env.reset()
    set_sim_state(env, current_state_chunk1)
    s1_B = _make_obs_vec_for_prior(env, obs_keys=OBS_KEYS)
    prior.reset_obs_buffer()
    prior.observe(s0_B)
    prior.observe(s1_B)
    plan_B1 = prior.sample_initial_mean(s1_B, n_steps=100, seed=1)
    print(f"\nPATH B (unified-style):")
    print(f"  s1_B[:5] = {s1_B[:5]}")
    print(f"  plan_B1[0,:] = {plan_B1[0, :]}")
    print(f"  plan_B1.mean = {plan_B1.mean():.4f}, std = {plan_B1.std():.4f}")

    s1_diff = np.linalg.norm(s1_A - s1_B)
    plan1_diff = np.linalg.norm(plan_A1 - plan_B1)
    state_diff = np.linalg.norm(current_state_chunk1 - current_state_chunk1)
    print(f"\n  sim_state L2 diff (post-chunk-0) = {state_diff:.6e}")
    print(f"  s1 L2 diff = {s1_diff:.6e}")
    print(f"  plan_1 L2 diff = {plan1_diff:.6e}")
    print(f"  plan_1 max abs diff = {np.max(np.abs(plan_A1 - plan_B1)):.6e}")
    if s1_diff > 1e-6:
        print("  >>> s1 vectors DIFFER -- post-chunk-0 snapshot/restore breaks obs!")
        # Identify which obs components differ
        diff = np.abs(s1_A - s1_B)
        big_idx = np.where(diff > 1e-6)[0]
        print(f"  >>> {len(big_idx)} obs dims differ; first few diffs:")
        for i in big_idx[:8]:
            print(f"      idx={i}: A={s1_A[i]:.6f} B={s1_B[i]:.6f} diff={diff[i]:.6e}")
    if plan1_diff > 1e-6:
        print("  >>> plans DIFFER at chunk 1")

    # ============================================================
    # End-to-end emulation of post-fix unified-driver pattern.
    # Reproduces unified_plan_and_execute exactly for the first
    # rollout (demo_0, r_idx=0) under K=0, and compares chunk-by-
    # chunk to standalone_dp.py's path A.
    # ============================================================
    print("\n\n===== END-TO-END (unified post-fix vs standalone) =====")

    def warmstart_seed_fn(chunk_idx, b=0, key="demo_0", r=0):
        return int(np.uint32(hash((
            "flow_warmstart", b, key, r, int(chunk_idx),
        )) & 0xFFFFFFFF))

    # ---- Standalone (path A continued) ----
    env.reset()
    set_sim_state(env, init_state)
    prior.reset_obs_buffer()
    total_reward_std = 0.0
    success_std = False
    from common import check_success
    rollout_seed = int(np.uint32(hash(
        ("standalone_dp", 0, demo_keys[0], 0)
    ) & 0xFFFFFFFF))
    print(f"rollout_seed={rollout_seed}")
    for chunk_idx in range(30):  # 30 chunks * 8 steps = 240 env steps
        s = _make_obs_vec_for_prior(env, obs_keys=OBS_KEYS)
        prior.observe(s)
        chunk = prior.sample_initial_mean(
            s, n_steps=100, seed=rollout_seed + chunk_idx * 1009,
        )
        for t in range(8):
            _, reward, _, _ = env.step(chunk[t])
            total_reward_std += float(reward)
            if check_success(env):
                success_std = True
                break
        if success_std:
            break
    print(f"STANDALONE: reward={total_reward_std:+.2f}  success={success_std}")

    # ---- Unified post-fix replica ----
    env.reset()
    set_sim_state(env, init_state)  # top-of-function reset
    prior.reset_obs_buffer()
    total_reward_uni = 0.0
    success_uni = False
    for chunk_idx in range(30):
        s = _make_obs_vec_for_prior(env, obs_keys=OBS_KEYS)
        prior.observe(s)
        torch.manual_seed(warmstart_seed_fn(chunk_idx))
        chunk = prior.sample_initial_mean(s, n_steps=100)  # no seed arg
        for t in range(8):
            _, reward, _, _ = env.step(chunk[t])
            total_reward_uni += float(reward)
            if check_success(env):
                success_uni = True
                break
        if success_uni:
            break
    print(f"UNIFIED (post-fix, hash-seed via torch.manual_seed): reward={total_reward_uni:+.2f}  success={success_uni}")

    # ---- Unified post-fix, matched seeds via prior.sample_initial_mean(seed=) ----
    env.reset()
    set_sim_state(env, init_state)
    prior.reset_obs_buffer()
    total_reward_uni2 = 0.0
    success_uni2 = False
    for chunk_idx in range(30):
        s = _make_obs_vec_for_prior(env, obs_keys=OBS_KEYS)
        prior.observe(s)
        chunk = prior.sample_initial_mean(
            s, n_steps=100, seed=rollout_seed + chunk_idx * 1009,
        )
        for t in range(8):
            _, reward, _, _ = env.step(chunk[t])
            total_reward_uni2 += float(reward)
            if check_success(env):
                success_uni2 = True
                break
        if success_uni2:
            break
    print(f"UNIFIED-seedmatched (post-fix, Generator(seed)): reward={total_reward_uni2:+.2f}  success={success_uni2}")


if __name__ == "__main__":
    main()
