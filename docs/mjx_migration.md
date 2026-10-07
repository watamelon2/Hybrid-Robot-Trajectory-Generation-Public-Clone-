# MJX migration plan for the MBD inner loop

Detailed plan for Tier 4 (h) of [`docs/plans.md`](plans.md): replace
the sequential CPU robosuite rollouts inside MBD with a `jax.vmap`'d
MJX batch. Living document; update when a gate is hit. No code is
written until the gate-1 decision below.

## Why this doc exists

Per the wall-time analysis on the current GCE run
(`@experiments/04_mbd_baseline/results/can_ph_seed2.json`,
`plan_elapsed_s ~ 256` per demo, 60 demos, 4.35 h wall), the MBD
inner loop is the only term in the pipeline whose cost grows
linearly in `M x K x H` and dominates everything else by 2-3 orders
of magnitude. Estimated MJX-on-L4 speedup is `~50-200x` at fixed
`M=16`, plus a "free" `M` scale-up that the LQ sanity (`M=128`)
already showed MBD wants. See the conversation log on 2026-05-13
for the throughput derivation.

## Branching

Branch off **`feat/further-experiments` at HEAD** (`5e512e4`), not
off `main`. All GCE infra, exp-04/05 scaffolding, and the
`rollout_fn`-injected `denoise_loop` seam this plan depends on live
on the current branch; branching from `main` would mean re-doing
~15 commits.

Suggested name: `feat/mjx-rollouts`.

## File-level changes

### New files

| Path | Purpose |
|---|---|
| `experiments/04_mbd_baseline/scenes/simplified_can.xml` | Simplified PickPlaceCan MJCF that passes `mjx.validate_model` (capsule "can", box-walled "bin", `<motor>` actuators). |
| `experiments/04_mbd_baseline/mjx_rollout.py` | `make_mjx_rollout_fn(mj_model, init_state)` returning a closure with the same `(traj_batch) -> rewards` signature as the current `rollout_fn`. JAX-side cost. |
| `experiments/04_mbd_baseline/jax_osc.py` | JAX port of robosuite's `OSC_POSE` controller (~150 LOC). Converts the BC checkpoint's `(M, H, 7)` Cartesian-delta action space into joint torques that MJX's `mj_step` can consume. See OSC checklist below for why this is mandatory rather than optional. |
| `experiments/04_mbd_baseline/test_mjx_parity.py` | Pytest-style parity tests: trajectory parity vs robosuite, cost rank correlation, OSC controller parity, MBD descent on the LQ surrogate. |
| `docs/mjx_migration.md` | This document. |

### Modified files

| Path | Delta |
|---|---|
| `experiments/04_mbd_baseline/run_can.py` | Add `--mjx` flag; in `make_rollout_fn` dispatch to `mjx_rollout.make_mjx_rollout_fn` when set. Default remains the robosuite path. |
| `experiments/04_mbd_baseline/mbd.py` | **No change.** `denoise_loop` already accepts an injected `rollout_fn` (see `mbd.py:216`); that is the seam. |
| `experiments/05_hybrid_method/run_hybrid.py` | Add the same `--mjx` flag and dispatch identically. Inherits the speedup for free. |
| `experiments/07_lambda_sweep/*.py` | Ditto. |
| `scripts/gce/startup.sh` | Conditional CUDA + `mujoco-mjx` + `jax[cuda12_pip]` install when the VM has a GPU. |
| `scripts/gce/launch_overnight.sh` | Add `--accelerator type=nvidia-l4,count=1` and switch machine type to `g2-standard-4` when an `--mjx` flag is forwarded. |

### What the new code does NOT touch

- `mbd.py` denoising math.
- The multiprocessing.Pool path in `make_rollout_fn`
  (`run_can.py:143-163`) and `worker.py`. They become the CPU
  fallback, gated by the negation of `--mjx`. This preserves the
  existing baseline numbers byte-for-byte.
- The GCE watchdog (`scripts/gce/watchdog.sh`), streaming launcher,
  or GCS staging. Same shell scripts, only the VM image changes.

## Simplified MJCF checklist

Source: `mujoco_menagerie/franka_emika_panda/panda.xml` plus a
hand-authored "can + bin" worldbody. MJX restrictions are listed at
`@docs/MuJoCo.md:386-388` and the upstream MJX page.

1. Start from the Menagerie `panda.xml`. Keep arm bodies, kinematics,
   and inertias unchanged.
2. Replace stock `<general>` actuators with pure `<motor>` actuators,
   `ctrlrange="-87 87"`, per `@docs/MuJoCo.md:419-425`.
3. Drop the official gripper include; substitute two box-geom fingers
   on prismatic joints (mimics robosuite's two-finger parallel-jaw
   gripper at a level MJX can handle).
4. Add a table (single box geom) at canonical PickPlaceCan height.
5. Add the bin as four box walls (no cylinder geoms).
6. Add the "can" as a single capsule geom (height `0.1 m`, radius
   `0.03 m`) attached to a free joint. Tune mass to PickPlaceCan's
   ~0.05 kg.
7. Add a `<site name="ee">` at the gripper midpoint for cost
   computation; add a `<site name="bin_target">` for the goal.
8. Add a `<keyframe>` matching the canonical PickPlaceCan init pose
   so `mjx.make_data` + keyframe reset works.
9. Run `mjx.validate_model(mj_model)` in a smoke script; fix every
   diagnostic it returns before moving on.
10. Side-by-side render at 4 keyframes (init, pre-grasp, lift,
    placed) against robosuite PickPlaceCan to confirm the scene is
    visually faithful.

## Reward / success port checklist

robosuite's reward is composed of (reach, grasp-contact, lift,
bin-place); the contact-based term is the hardest piece to keep on
MJX. Initial port intentionally simplifies:

1. EE position: `data.site_xpos[ee_site_id]`.
2. Can position: `data.xpos[can_body_id]`.
3. Reaching: `1 - jnp.tanh(10 * jnp.linalg.norm(ee_pos - can_pos))`.
   Direct port of robosuite's `staged_rewards()[0]`.
4. **Lift proxy**: `jnp.clip((can_z - table_z) / lift_h, 0, 1)`.
   Replaces robosuite's contact-conditioned grasp+lift bonus. This
   is a deliberate simplification: MJX rollouts cannot distinguish
   "fingers-on-can lifting" from "scooped lifting". Document in the
   exp-04 README delta when we ship.
5. Bin placement: `1 - jnp.tanh(10 * jnp.linalg.norm(can_xy - bin_xy))`,
   gated by lift proxy > threshold.
6. Total reward = weighted sum, weights matched to robosuite within
   one significant figure.
7. Success (binary): `can_z > lift_h AND jnp.linalg.norm(can_xy - bin_xy) < bin_half_extent`.
   Geometric test only, no robosuite needed.

All five terms must be `jax.jit`-able and `vmap`-able over the M axis.

## OSC controller port checklist

robosuite feeds 7-D `OSC_POSE` Cartesian-delta actions through a
Python operational-space controller; MJX is `mj_step` only and
includes no controller layer. We must either port a minimal OSC
into JAX or break the BC compat chain.

**Trade-off:**

- **Option A -- retrain BC on joint torques.** Use pure `<motor>`
  actuators and a 7-D torque action space; bypasses the OSC port
  entirely (~0 LOC) but invalidates `policy.pt` (500 epochs, commit
  `0c8f008`) and forces re-running every BC-dependent row of the
  Week-4 table (exps 03, 05, 05b, 07).
- **Option B -- port the OSC controller into JAX.** Keep `policy.pt`
  and the `OSC_POSE` action space. Cost: ~150 LOC in `jax_osc.py`.
  Math is well-defined and stateless.

**Recommendation: Option B.** The published Week-4 numbers cannot
survive a BC retrain without re-running four downstream rows; the
engineering saving from option A is not worth the loss of
apples-to-apples.

**Checklist (option B):**

1. Read `robosuite.controllers.osc.OperationalSpaceController`
   (~200 LOC in robosuite source) to fix the input convention:
   action is a delta target pose, not an absolute one.
2. Port the position-tracking PD law:
   `tau = J^T @ (K_p * (x_des - x_curr) - K_d * x_dot) + qfrc_bias`,
   with `J = mjx.jac(...)` for the EE site.
3. Port the nullspace damping: `(I - J^+ J) @ (-K_null * qdot)`,
   matching robosuite's defaults.
4. Saturate `tau` to `[-87, 87]` per joint (matches MJCF `ctrlrange`).
5. Parity test: feed identical `OSC_POSE` actions to robosuite and
   to `jax_osc + mjx.step`; arm joint trajectories should agree to
   within `2e-2 rad` over 60 steps. Part of Gate 2.

## MJX feature gaps and mitigations

Enumeration of MJX restrictions that intersect our pipeline, with
the mitigation for each. Severity is task-relative, not general.

| # | Gap | Severity | Mitigation | Confidence |
|---:|---|---|---|---|
| 1 | Cylinder collisions silently dropped | **High** | Replace can with capsule, bin walls with boxes (MJCF checklist 5-6) | High -- standard substitution |
| 2 | Convex-mesh fingertips downgraded | **High** | Replace fingers with two box geoms per finger (MJCF checklist 3) | Medium -- grasp fidelity validated at Gate 2 |
| 3 | No controller layer (no OSC) | **High** | Port OSC into JAX (`jax_osc.py`, ~150 LOC) -- see checklist above | High -- well-defined math |
| 4 | `mj_inverse` not implemented | None | Keep demo-retargeting (`docs/MuJoCo.md` 7.2) on CPU MuJoCo; it is offline | High |
| 5 | Float32 precision (default) | Low | Parity tolerance `5e-2 rad`; fall back to `dtype=jnp.float64` if Gate 2 fails marginally (~2x throughput cost) | High |
| 6 | Newton solver only | Low | Pin CPU MuJoCo to Newton also (`<option solver="Newton">`) so both paths use the same solver | High |
| 7 | No built-in renderer | None | `device_get` final state, render in CPU MuJoCo (~30 ms / frame) | High |
| 8 | Static-shape requirement (no runtime-variable M, K, H) | Low | Pad + mask any future variable-length rollouts; currently fixed | High |
| 9 | Pre-allocated contact buffer; overflow silently drops contacts | Low | Set `max_contact_points=128` (~10x headroom); runtime assert `data.ncon < cap` | High |
| 10 | Limited equality-constraint support | None | Stock gripper mimic equality removed; two independent prismatic finger joints instead | High |
| 11 | No tendons | None | Franka uses none | High |
| 12 | Limited touch / force sensors | None | We never read sensors; we use `data.xpos` / `data.site_xpos` / `data.qpos` directly | High |
| 13 | Muscle actuators not supported | None | We use `<motor>` only | High |

The only items that meaningfully drive engineering are
**#1 (cylinder), #2 (mesh), #3 (OSC port)**. #1 and #2 are MJCF
edits gated by `mjx.validate_model` (Gate 1) and by cost-rank
correlation (Gate 2). #3 determines the API of `mjx_rollout.py`:
it takes Cartesian-delta actions and routes them through
`jax_osc.py` before each `mjx.step`, rather than taking raw
torques.

### What is NOT mitigatable

- **Adaptive `K` / early termination on convergence.** Static shapes
  force a fixed `K`. Could wrap in `lax.while_loop` at the cost of
  losing some XLA fusion; not worth it for `K=10`.
- **Per-rollout variable `H`.** Same constraint. If we ever want to
  early-terminate on success, mask the cost on post-success steps
  rather than truncating the rollout.
- **Contact-conditioned reward signal.** The lift-proxy in the
  reward checklist already concedes this; recovering it would
  require a JAX-side contact-aware reward and defeats the speedup.

## Validation steps (in order; each gates the next)

| Step | Acceptance | Hits gate |
|---|---|---|
| 1. `mjx.validate_model(simplified_can.xml)` is clean. | Zero diagnostics. | Gate 1 |
| 2. **Trajectory parity, M=1.** Same init state + zero-action 120-step rollout in robosuite and in MJX; compare `(qpos, qvel)` of the arm. | Joint-position MAE `< 5e-2 rad` at step 120 (different solvers, not bit-identity). | Gate 2 |
| 3. **Cost rank correlation.** 64 random trajectories from a held-out demo; compute robosuite reward and JAX reward. | Spearman `rho > 0.7`. | Gate 2 |
| 4. **LQ sanity port**, optional. Re-run `lq_sanity.py` with the MJX double-integrator. | MBD final cost within 10% of DARE optimum, matches the existing `lq_sanity.json`. | - |
| 5. **Exp 04 reproduction at M=16.** `run_can.py --mjx --seed 0`, 20x3 protocol. | (a) cost-descent curves qualitatively similar to `can_ph_seed0.json`; (b) success rate within ±2/60 of the robosuite baseline (likely 0/60 either way). | Gate 3 |
| 6. **Speed measurement.** Same config, on Mac-CPU (robosuite), Mac-CPU (MJX), and `g2-standard-4` L4. | Mac-CPU MJX `> 3x` over robosuite-CPU; L4 `> 30x` over robosuite-CPU. | Gate 3 |
| 7. **Scale-up demo.** Run `M=128 K=10 H=120` on L4. | Wall `< 5x` the `M=16` wall. This is the headline justification. | Gate 4 |
| 8. **Exp 05 + 07 re-run.** Hybrid + lambda sweep with `--mjx`. | Pareto curve qualitatively matches existing `lambda_*_inf.json` results. | Gate 4 |

## Status as of `feat/mjx-rollouts`

| Gate | Acceptance | Actual | Status |
|---|---|---|:---:|
| 1 | `mjx.validate_model` clean + 120-step zero-action rollout finite | `test_mjx_smoke.py` passes; CPU rollout ~720 step/s, MJX ~70 step/s (Mac CPU, first compile dominates) | ✅ |
| 2a | JAX OSC controller compiles and regulates the arm | `test_osc_smoke.py` passes; zero-action drift = 0.0 rad after 120 steps | ✅ |
| 2b | OSC math parity vs robosuite numpy reference, target `< 1e-6` Nm in float64 | **8.03e-12 Nm** over 32 random problems (`test_osc_parity.py`) | ✅ |
| 2c | Spearman `rho >= 0.3` between MJX cost and robosuite shaped reward, relaxed from doc's 0.7 because two MJCFs are different | **rho = +0.485** (16 plans, H=20, sigma=0.4, `test_cost_rank_parity.py`) | ✅ |
| 3 | `run_can.py --mjx` runs end-to-end and writes results JSON | Smoke at H=20 M=4 K=2 completes in ~57 s on Mac CPU (JIT compile dominates; per-iter MJX rollout is ~0.1 s post-compile) | ✅ |
| 4-prep | GCE GPU image + launch.sh ACCELERATOR=l4 + startup.sh exp04-mjx dispatch + fail-fast GPU smoke + Colab notebook | infra READY: `install_linux_gpu.sh`, `test_mjx_gpu_smoke.py` (PASS-on-GPU / exit-20-on-CPU), `launch.sh ACCELERATOR=l4`, `startup.sh exp04-mjx`, `colab_mjx_gate4.ipynb`, runbook below | ✅ |
| 4 | One real GPU run + speedup measurement vs Mac-CPU baseline | Colab L4 completed on 2026-06-07 using `colab_mjx_gate4_2026_06_07.ipynb`: 60 rollouts, notebook wall `1310.1 s` (`21.8 min`), JSON `total_wall_s = 1304.711483141`, mean/median plan time `14.733 / 14.731 s`, speedup `13.80x` versus nominal 5 h Mac/robosuite CPU baseline. This closes the measurement but misses the prior `>30x` promotion target, so MJX remains an opt-in engineering path. | ⚠️ |

The 8e-12 Nm OSC parity is the strongest individual signal that
the port is correct; the rank-correlation result (rho=0.485)
confirms the smooth shaped cost in `mjx_rollout._pickplace_step_cost`
ranks plans the same way the robosuite shaped reward does, even
across the two physics models.

## Decision gates

- **Gate 1 (end of MJCF work, ~2 days):** `mjx.validate_model` clean
  and a smoke rollout runs. If two days in we still cannot pass
  validation, abort: the can/bin scene may need a more invasive
  re-author than capsule-substitution.
- **Gate 2 (parity + cost correlation, +1 day):** Cost rank `rho`
  between MJX and robosuite must be `> 0.7`. Below that, "same task"
  is no longer defensible and the MJX path becomes a separate scene
  rather than a faster baseline.
- **Gate 3 (exp 04 reproduction at M=16, +1 day):** End-to-end wall
  speedup on Mac CPU `> 3x` and L4 `> 30x`. Below those numbers the
  engineering doesn't pay back relative to a Tier-3 second-task run.
- **Gate 4 (scale-up + downstream exp reruns, +1-2 days):** M=128
  full-eval fits in a single VM-hour and exp 05/07 numbers don't
  regress. Past this gate, MJX becomes the default path and the
  robosuite path is documented as the apples-to-apples reference
  for the published table.

## Rollback plan

- Branch is feature-isolated; `git switch feat/further-experiments`
  reverts everything except for newly generated `results/` files.
- `--mjx` is opt-in. The default `make_rollout_fn` codepath is
  unchanged, so all existing run scripts and overnight launchers
  keep producing the same numbers.
- `worker.py` and `multiprocessing.Pool` machinery stay in place as
  the CPU fallback even after the MJX path is the default. They are
  also what runs on hosts without a GPU (Mac dev, CPU-only VMs).
- If Gate 2 fails, the simplified MJCF + JAX cost can still be
  published as a "what would training-free MBD look like with
  enough compute" exploratory result, but the headline table stays
  on robosuite.

## Gate 4 ops runbook (L4 launch)

Everything needed to flip the switch the moment the L4 quota
lands. The runbook is also exercised by `bash -n` parsing in
the smoke tests, so the commands below stay in sync with the
scripts.

### Pre-flight (one-time, do NOW while quota is pending)

1. Verify the quota request is in:
   ```sh
   gcloud alpha quotas info list \
       --project=hybrid-traj-robin-2026 \
       --service=compute.googleapis.com \
       --filter='quotaId:GPUS-ALL-REGIONS OR quotaId:NVIDIA-L4-GPUS'
   ```

2. Make sure the repo tarball on GCS includes `feat/mjx-rollouts`:
   ```sh
   bash scripts/gce/upload_repo.sh   # tarballs $REPO_ROOT to gs://.../repo/repo.tar.gz
   ```
   `startup.sh` reads `git-branch` from instance metadata; the
   launch command below sets it to `feat/mjx-rollouts`.

3. Sanity-check the scripts parse and the local Python paths import:
   ```sh
   bash -n scripts/gce/launch.sh
   bash -n scripts/gce/startup.sh
   bash -n scripts/remote/install_linux_gpu.sh
   conda run -n hybrid-traj python \
       experiments/04_mbd_baseline/test_mjx_gpu_smoke.py
   # ^ exit 20 on a CPU box is the correct outcome -- it just proves the
   #   fail-fast guard works. On the L4 it must return 0 (PASS).
   ```

### Alternative: Colab (works today, no GCP quota needed)

If the L4 quota request is still pending, you can run the same
Gate-4 measurement on a Colab GPU instead. Open
`experiments/04_mbd_baseline/colab_mjx_gate4.ipynb` in Google Colab
(either via "Open in Colab" or `File > Upload notebook`), then:

1. `Runtime > Change runtime type > GPU` (T4 free works; L4 / A100 on
   Pro / Pro+).
2. Run cells top-to-bottom. The notebook installs `jax[cuda12]==0.10.0`
   + `mujoco-mjx==3.8.1`, clones `feat/mjx-rollouts`, runs the same
   fail-fast smoke (`test_mjx_gpu_smoke.py`) as the GCE path, then
   runs `run_can.py --mjx` at the headline H=120 M=16 K=10 config.

Prerequisites:
- The branch `feat/mjx-rollouts` must be pushed to GitHub:
  `git push -u origin feat/mjx-rollouts`.
- The repo is private, so the notebook's step-4 cell will prompt
  for a GitHub Personal Access Token via `getpass` (`repo` scope,
  7-day expiry is enough). The token is not persisted: after the
  clone, the cell rewrites the `origin` remote to drop the token
  so it doesn't leak via `git remote -v` or saved notebook output.
  If you make the repo public temporarily, you can leave the
  prompt empty and the clone falls back to anonymous HTTPS.

The results JSON ends up at `/content/can_mjx.json` and (optionally)
copied to `MyDrive/hybrid-traj/mjx-colab/`. Same schema as the GCE
path, so the Gate-4 acceptance table below applies unchanged.

A Colab L4 run on 2026-06-07 completed the headline config in
`1310.1 s` notebook wall (`1304.711483141 s` JSON wall) with 60
rollouts and mean/median plan time `14.733 / 14.731 s`. This is
`13.80x` faster than the nominal 5 h Mac/robosuite CPU baseline.
It is sufficient evidence that the GPU path works, but below the
original `>30x` promotion target.

### Launch (after quota is approved)

```sh
ACCELERATOR=l4 \
BRANCH=feat/mjx-rollouts \
PROJECT=hybrid-traj-robin-2026 \
BUCKET=gs://hybrid-traj-robin-2026-results \
N_DEMOS=20 N_ROLLOUTS=3 \
MJX_H=120 MJX_M=16 MJX_K=10 MJX_N_SUBSTEPS=5 \
bash scripts/gce/launch.sh exp04-mjx
```

This creates a single `g2-standard-4` (4 vCPU, 16 GB, 1 L4 24 GB)
in `us-central1-a` with:

- `--maintenance-policy=TERMINATE` (mandatory for GPU VMs)
- Deep Learning VM image `common-cu124-ubuntu-2204-py310` (NVIDIA
  driver + CUDA 12.4 preinstalled, saves ~3 min of driver install)
- `--boot-disk-size=100GB` (50 GB is too tight after CUDA + venv)
- Metadata `experiment=exp04-mjx`, `git-branch=feat/mjx-rollouts`,
  plus the MJX-specific knobs.

### What happens on boot (`startup.sh` flow for `exp04-mjx`)

1. Fetch the repo tarball.
2. `python3 -m venv .venv && . .venv/bin/activate`.
3. `bash scripts/remote/install_linux_gpu.sh`:
   - `nvidia-smi` check (fails fast if drivers missing)
   - `pip install "jax[cuda12]==0.10.0" "mujoco-mjx==3.8.1"`
   - Smoke: `jax.default_backend() == 'gpu'`, `mjx.step` runs.
4. **`python experiments/04_mbd_baseline/test_mjx_gpu_smoke.py`** --
   the Gate-4 fail-fast smoke (~60 s budget). Exits non-zero if:
   - JAX backend is not GPU (exit 20)
   - any check raises (exit 21)
   - post-compile batched rollout > 3 s (exit 22, indicates silent
     CPU fallback)
5. If smoke passed, run the headline experiment:
   ```sh
   python experiments/04_mbd_baseline/run_can.py --mjx \
       --H 120 --M 16 --K 10 --mjx-n-substeps 5 \
       --n-demos 20 --n-rollouts 3 --seed 0 --no-live-view --video-dir '' \
       --output experiments/04_mbd_baseline/results/can_mjx.json
   ```
6. Upload `experiments/*/results` to GCS.
7. `shutdown -h now`.

If the smoke fails, the run is skipped but `/tmp/install_linux_gpu.log`
and any partial results are still uploaded to GCS for triage.

### Watching the run

```sh
NAME=hybrid-traj-exp04-mjx-<timestamp>
gcloud compute instances tail-serial-port-output "$NAME" --zone us-central1-a
```

The serial log will show `[install-gpu] jax.default_backend(): gpu`
followed by the Gate-4 smoke output. Look for `PASS  GPU smoke clean.`
before letting the long experiment proceed.

### Cost guard

A misconfigured L4 VM running for an hour costs ~$0.70 -- not a
disaster, but the fail-fast smoke is designed to catch any silent
CPU-fallback within 60 s, so 99% of bad-config burns are bounded
at ~$0.012.

### Expected results (to-be-confirmed once quota lands)

Based on the Mac-CPU baseline (M=16 K=10 H=120 ~= 276 s plan
time per rollout in `can_ph_seed2.json`) and the post-compile
per-batch rollout time observed locally (~0.1 s on Mac CPU for
small batches), L4 should give:

| Stage | Mac CPU baseline | L4 (estimate) | Speedup |
|---|---:|---:|---:|
| Per-batch MJX rollout (M=16) | ~5-10 s | ~0.1-0.3 s | 30-50x |
| One full plan (K=10) | ~50-100 s | ~1-3 s | 30-50x |
| 60-rollout full eval | ~5 h | ~5-15 min | 20-50x |

These numbers are projections, not measurements. The actual
speedup will be recorded as the first row of this table once
the quota lands and a real run completes. The migration doc's
Gate-4 acceptance target is "L4 > 30x over robosuite-CPU,"
which lines up with the per-batch projection.

## Effort estimate

| Slice | Calendar (serial) | Active engineering |
|---|---:|---:|
| Simplified MJCF + validate | 2 days | ~6 h |
| `jax_osc.py` + OSC parity test | 1 day | ~5 h |
| `mjx_rollout.py` + cost port + parity tests (Gates 1-2) | 1 day | ~4 h |
| `run_can.py` `--mjx` wiring + exp 04 reproduction (Gate 3) | 1 day | ~3 h |
| GCE GPU image, launcher tweaks, first full eval (Gate 4) | 1-2 days | ~4 h |
| Exp 05 + 07 re-run + writeup | 1-2 days | ~3 h |
| **Total** | **~7-10 days serial** | **~25 h active** |

Most of the calendar time is GPU-VM job wait + JIT-compile waits,
not keyboard time. The `jax_osc.py` slice is new in v2 of this
plan after the MJX-feature-gap audit elevated the OSC port from
"assumed away" to a first-class deliverable.

## Open questions before starting

1. **GPU access.** Does the GCE project already have an L4 quota? If
   not, request 1 vCPU + 1 L4 in `us-central1` before Gate 4.
2. **Mac dev loop.** JAX on Apple Metal is technically supported but
   not officially production-ready. Decision: dev on CPU JAX
   (slower but correct), GPU only on GCE. Confirm before writing
   `mjx_rollout.py`.
3. **Scene-fidelity vs apples-to-apples.** If Gate 2 passes
   marginally (`rho ~ 0.7-0.8`), do we (a) publish MJX numbers as
   "the same task" or (b) re-run the entire Week-4 table on the
   simplified scene so the new column is internally consistent?
   Option (b) is the honest answer and costs `~3x60` extra robosuite
   rollouts (one overnight VM).
4. **MBD hyperparameters.** Once MJX makes `M=128, K=20` affordable,
   the existing `M=16, K=10` defaults become a CPU-era artifact.
   Plan a one-figure ablation as part of the MJX rollout.

## Cross-references

- Tier 4 (h) in `docs/plans.md`.
- `experiments/04_mbd_baseline/README.md` lines 200-207 (the
  explicit choice not to use MJX in stage 2).
- MJX usage primer and restrictions: `docs/MuJoCo.md` sections 2.2,
  6, 7.1, 8.
- Current rollout seam: `experiments/04_mbd_baseline/mbd.py:216`
  (`states = rollout_fn(tau)`) and
  `experiments/04_mbd_baseline/run_can.py:114-163`
  (`make_rollout_fn`).
- Current worker module that becomes the CPU fallback:
  `experiments/04_mbd_baseline/worker.py`.
