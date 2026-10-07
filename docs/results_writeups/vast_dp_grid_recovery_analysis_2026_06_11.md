# Vast remote results — recovery + analysis (2026-06-11)

**Status:** complete. Recovered all remote result roots to local; analyzed the
per-shard numbers that were previously aggregate-only.

**Headline:** the recovered **140-shard DP "production" grid
(`vast_safe_dp_2026_06_10/`) is INVALID and must not be cited** — it was
launched with `--n-workers 1`, which triggers the documented chunk-boundary
controller-state corruption that collapses unified success to ~0%. The BC /
static remote roots (run with `--n-workers >= 2`) are valid and are summarized
below.

## 1. Recovery provenance

Original instance `40441157` (2x A40) went `exited` with its start queued
indefinitely on a full host. Vast offers no host-to-host migration / offline
disk export for stopped instances, so we:

1. rented a fresh similar instance `40673435` (2x A40, same image, 70 GB);
2. web-UI-copied `/workspace/hrtg-cloud` from `40441157` -> `40673435`;
3. pulled to local via tar-over-ssh (`/usr/bin/rsync` was non-executable):
   `ssh ... "tar -C <dir> -czf - <root>" | tar -xzf -`.

All roots landed under
`experiments/15_unified_hybrid/results/`. Checkpoints (~2.5 G of `.pt`) and
`03_bc_baseline/results` were already local, so only the result JSONs + logs
were transferred.

| Recovered root | summaries | rollouts | verdict |
|---|---:|---:|---|
| `vast_safe_dp_2026_06_10` | 140 | ~840 | **INVALID (n_workers=1)** |
| `tail_boost_2026_06_10` | 9 | ~480 | valid (BC, nw=2) |
| `tail_boost2_static_2026_06_10` | 5 | 300 | valid (BC, nw=12) |
| `offload_chain_2026_06_10` | 9 | 540 | valid (static, nw=4) |
| `research_queue_remote_remainder_2026_06_10` | 19 | ~545 | valid (nw>=2) |
| `vast_safe_validation_2026_06_10` | 2 | 2 | invalid smoke (nw=1) |
| `vast_static_validation_2026_06_10` | 1 | 1 | invalid smoke |

## 2. Why the DP grid is invalid

`queue_vast_safe_remote_2026_06_10.py` hardcodes `--n-workers 1` in **both** DP
job builders (`base_can_dp_command` L100, `base_square_dp_command` L131). The
unified driver explicitly warns against this (`run_can.py:565-577`):

> n_workers=1 ... produced a subtle chunk-boundary controller-state corruption
> (env.reset() between chunks wipes robosuite's Python-side controller cache;
> set_sim_state only restores Mujoco qpos/qvel). This bit Phase-C
> catastrophically (K=0 test rate collapsed from 80% standalone to 0% unified).
> We default to 2 going forward.

Four independent lines of evidence confirm the corruption fired here:

1. **Same config, opposite result.** The grid's Square cells use the *identical*
   recipe to the documented Square-DP run (`docs/results_writeups/square_dp_prior.md`):
   `--task square --prior flow --score-mode pull --H 16 --mpc-chunk-steps 8
   --mpc-replans 49 --K 10 --M 16 --sigma-max 0.10 --test-start-seed 100000`,
   same `square_dp ep0199` checkpoint, same 30 test seeds. The documented run
   (nw=2) gets **lambda0 = 53.3%, lambda0.5/1 = 70%**; the recovered grid (nw=1)
   gets **0/30 at every lambda**.
2. **lambda0 collapse.** lambda0 is pure warm-start + MBD (no prior pull,
   `used_score_fn=false`). Documented 53.3% -> recovered 0% is the exact
   "53/70% -> 0%" corruption signature.
3. **6.7x slowdown.** 1804 s/rollout (recovered) vs 269 s/rollout (documented) —
   consistent with serial in-process perturbations on the broken path.
4. **Flat cost.** Per-chunk `init_best_cost -0.0489 -> final_best_cost -0.0509`:
   MBD never improves the plan; episodes run the full 400 steps and fail.
5. **Decisive same-cell cross-check.** The *identical* Can ep0199 sigma0.1 cell
   was also run (proper workers) in
   `research_queue_2026_06_10/can_dp_calibration/` — same checkpoint
   (`v_theta_dp_tobs2_h16_v3b_ep0199.pt`), same `K-10 R-29 smax0.1` recipe.
   There it gives **lambda0 21/30 = 70.0%, lambda0.5 15/30 = 50.0%,
   lambda1 20/30 = 66.7%** at 377 s/rollout. In `vast_safe_dp` (nw=1) the same
   cell gives **6.7% / 6.7% / 3.3%** at 1804 s/rollout. Same checkpoint, same
   config, opposite outcome — the only difference is `--n-workers`.

### Recovered per-shard DP numbers (INVALID — for the record only)

Pooled across 5 seeds x n6 (= 30 rollouts/condition), Wilson 95% CI:

| Task / ckpt | lambda=0 | 0.5 | 1 | 2 | 5 | 10 |
|---|---|---|---|---|---|---|
| Can ep0199 smax0.10 | 6.7% | 6.7% | 3.3% | 0% | 0% | 3.3% |
| Can ep0199 smax0.25 | 26.7% | 10% | 13.3% | 0% | 0% | 0% |
| Square ep0049 | 0% | 0% | 0% | 0% | - | - |
| Square ep0099 | 0% | 0% | 0% | 0% | - | - |
| Square ep0149 | 0% | 0% | 0% | 0% | - | - |
| Square ep0199 | 0% | 0% | 0% | 0% | - | - |

These contradict the valid documented DP numbers (Can 66-73%, Square-DP 70%) and
are an **infrastructure artifact**, not a method result.

## 3. Valid recovered results (BC / static, n_workers >= 2)

**BC lambda sweep, Can** (`tail_boost`, nw=2 / `tail_boost2_static`, nw=12 — two
independent worker counts, consistent shapes => BC path is not worker-sensitive):

| lambda | tail_boost (nw2) | static (nw12) |
|---:|---:|---:|
| 0 | 25/60 = 41.7% | 28/60 = 46.7% |
| 1 | 21/60 = 35.0% | 18/60 = 30.0% |
| 2 | 26/60 = 43.3% | 19/60 = 31.7% |
| 5 | 22/60 = 36.7% | - |
| 10 | 28/60 = 46.7% | 31/60 = 51.7% |
| 20 | 24/60 = 40.0% | 18/60 = 30.0% |

Shape: weakly non-monotonic, peaks at lambda0 and lambda10, dip at lambda1-2.
Consistent with the settled finding that the rule/strength knob is **secondary**
and the additive score-bridge is the robust operating mode.

**Static K x R grid** (`offload_chain`, bc_kxr_grid, lambda5, nw=4): 9 cells span
**25-43%** (15-26 / 60) as K and R vary.

**Adaptation under mass/friction perturbation**
(`research_queue_remote_remainder`, adaptation_mass_friction, lambda1, nw>=2):
18 cells span **17-40%** (5-12 / 30).

**Sparse task (tool_hang-class)** (`sparse_task_extra`, nw=1): 0/5 everywhere —
unsolved, consistent with all prior tool_hang evidence (and nw is moot at 0%).

## 4. Consequences

1. **Do not cite the `vast_safe_dp` grid.** The "140/140 complete" marker only
   verified completion, not validity. The numbers are corruption artifacts.
2. **The valid DP headline numbers remain** the local `square_dp_prior.md`
   (Square-DP standalone 66.7%, hybrid 70%) and the documented Can DP cells —
   all run with nw>=2.
3. **Process fix:** any future remote DP/flow-prior offload MUST pass
   `--n-workers >= 2`; add a guard in the queue scripts and a post-run sanity
   check that flags lambda0 unified success far below the standalone reference.
4. If per-shard DP production numbers are genuinely needed, **re-run the grid
   with `--n-workers >= 2`** (the one place re-compute is justified).

## 5. Reproduce this analysis

- `/tmp/analyze_vast_dp_2026_06_11.py` — pooled per-(task,ckpt,smax,lambda)
  tables + Wilson CIs over `vast_safe_dp`.
- `/tmp/dump_roots_2026_06_11.py` — per-cell dump for the other roots.
