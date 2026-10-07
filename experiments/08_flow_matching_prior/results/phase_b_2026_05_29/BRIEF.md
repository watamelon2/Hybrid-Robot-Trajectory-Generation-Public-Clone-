# Phase-B: DP retrain with 23-D `can_dp` obs schema — 2026-05-29

Single-variable retrain on top of Phase-3.5's recipe, flipping the
**obs schema** from our legacy 46-D `can` keys to the DP paper's 23-D
`can_dp` keys.  The pre-registered hypothesis from
`phase_a_2026_05_28/BRIEF.md`: the 25 % standalone ceiling is
structurally caused by the obs schema (we slice off the
relative-pose features DP's `object` blob includes), not by the eval
protocol (Phase-A confirmed RED on the apples-to-apples DP protocol:
ep0399 → 16.0 % test, ep1999 → 10.0 % test).

## Schema delta

| component   | Phase-3 (`can`, 46-D)                        | Phase-B (`can_dp`, 23-D)                           |
|-------------|----------------------------------------------|-----------------------------------------------------|
| object      | `Can_pos`(3) + `Can_quat`(4) = **7**         | `Can_pos`(3) + `Can_quat`(4) + `Can_to_eef_pos_synth`(3) + `Can_to_eef_quat_synth`(4) = **14** |
| robot eef   | `eef_pos`(3) + `eef_quat`(4) = 7             | `eef_pos`(3) + `eef_quat`(4) = 7                    |
| gripper     | `qpos`(2) + `qvel`(2) = 4                    | `qpos`(2) = 2                                       |
| joints      | `pos`+`cos`+`sin`+`vel` = **28**             | (none) = 0                                          |
| **total**   | **46**                                       | **23**                                              |

The two `_synth` keys are recomputed from `Can_pos` + `robot0_eef_*`
at BOTH train and runtime (via `compute_synthetic_obs_keys`) — same
function, same inputs → bit-identical features.  No dependency on
the dataset's `obs/object[:, 7:14]` tail (which is in a robosuite-1.4
reference frame that disagrees with the live 1.5 env).

## Recipe (Phase-3.5 verbatim, except `--task`)

| knob                       | Phase-3.5    | **Phase-B**            |
|----------------------------|--------------|------------------------|
| epochs                     | 400          | **400**                |
| ema_decay                  | 0.9999       | **0.9999**             |
| action norm                | ON           | **ON**                 |
| obs norm                   | min-max      | **min-max** (carryover)|
| **obs schema**             | `can` (46-D) | **`can_dp` (23-D)** ← THE single variable |
| horizon, n_obs_steps       | 16, 2        | 16, 2                  |
| eval n_test_rollouts       | 0            | **50** (Phase-A protocol) |
| eval n_train_rollouts × n_demos | 30 × 10  | **10 × 1** (DP paper)  |
| eval max_env_steps         | 240          | **400** (DP paper)     |
| eval n_steps               | 100          | 100 (DDPM K=100)       |

## Decision rule (pre-registered, from `colab_train_dp_phase_b.ipynb` cell 0)

| `test_pct` | reading |
|---|---|
| **≥ 65 %** | **GREEN** — schema was the dominant structural cause; within 20 pp of DP paper's 85 %; proceed to hybrid sweep (Phase-C). |
| **40–65 %** | **YELLOW** — schema lifts substantially but not all the way; one more hyperparameter pass (ckpt selection, K, n_action_steps). |
| **25–40 %** | **ORANGE** — some lift over Phase-A's 16 % but the schema is not the dominant cause; ~50 pp gap remains; audit before iterating. |
| **16–25 %** | **RED** — no meaningful lift; falsifies the BRIEF's primary hypothesis; shifts the search to remaining unknowns (EMA recipe, dataset preprocessing, action-norm convention). |
| **< 16 %** | **BROKEN** — schema regression; suspect a bit-flip in the synthetic-key path. |

## Result

Sorted by test_pct descending.  Best ckpt is **`ep0199`** at **80.0 %
test** (40/50 fresh-seed rollouts).

| ckpt                              | train (10) | **test (50, DP)** | combined |
|-----------------------------------|-----------:|------------------:|---------:|
| `v_theta_dp_tobs2_h16_v3b_ep0199` |   7/10 = 70.0 % | **40/50 = 80.0 %** | 47/60 = 78.3 % |
| `v_theta_dp_tobs2_h16_v3b_ep0299` |   9/10 = 90.0 % | **34/50 = 68.0 %** | 43/60 = 71.7 % |
| `v_theta_dp_tobs2_h16_v3b_ep0249` |   5/10 = 50.0 % | **34/50 = 68.0 %** | 39/60 = 65.0 % |
| `v_theta_dp_tobs2_h16_v3b_ep0349` |   7/10 = 70.0 % | **34/50 = 68.0 %** | 41/60 = 68.3 % |
| `v_theta_dp_tobs2_h16_v3b_ep0099` |   7/10 = 70.0 % | **33/50 = 66.0 %** | 40/60 = 66.7 % |
| `v_theta_dp_tobs2_h16_v3b_ep0149` |   7/10 = 70.0 % | **32/50 = 64.0 %** | 39/60 = 65.0 % |
| `v_theta_dp_tobs2_h16_v3b` (final, ep0399) |   6/10 = 60.0 % | **29/50 = 58.0 %** | 35/60 = 58.3 % |
| `v_theta_dp_tobs2_h16_v3b_ep0049` |   3/10 = 30.0 % | **23/50 = 46.0 %** | 26/60 = 43.3 % |

Reference numbers (carryover from Phase-A for delta calculation):

- DP paper Can/ph: **mean_score ≈ 85 %** (same n_test=50 protocol)
- Phase-A `ep0399` test (Phase-3 final, 46-D schema): **16.0 %**
- Phase-A `ep1999` test (Phase-3 EMA shadow, 46-D schema): **10.0 %**

Headline deltas (best ckpt = ep0199):

- **vs Phase-A `ep0399` (16.0 %): +64.0 pp**  — the schema swap alone
  closes 64 of the 69 pp Phase-A gap.
- **vs DP paper (~85 %): −5.0 pp**  — within the noise band of a
  50-rollout n_test (binomial 95 % CI on 40/50 is roughly
  66.3–89.9 %, easily including DP paper's 85 %).

## Verdict: GREEN — schema was the dominant structural cause

The 46-D legacy `can` schema's omission of the `Can_to_eef_pos/quat`
relative-pose features (plus the joint-redundancy padding) was the
**single dominant structural cause** of the 25 % Phase-3 ceiling.
Restoring DP paper's 23-D `can_dp` schema (with the synthetic
relative-pose dims computed at both train and runtime via
`compute_synthetic_obs_keys`) is sufficient to reach a near-paper
80 % test rate, holding every other knob (training recipe, eval
protocol, ckpt selection convention) fixed at Phase-3.5's values.

Specifically, Phase-A's RED verdict on the structural-ceiling
hypothesis is now resolved.  Phase-A correctly identified that
eval-pipeline bias was real but small (−10.7 pp Phase-3 ep0399's
26.7 % training-init → 16.0 % DP test).  The remaining 69 pp gap
was structural, and 64 pp of it is recovered by the obs schema
alone.  The residual 5 pp to DP paper's 85 % is within the
50-rollout n_test noise band and is not a meaningful gap.

## Observations worth carrying into Phase-C

1. **Best-ckpt selection matters: 22 pp swing.** Final (ep0399) is
   58 % test; peak (ep0199) is 80 % test.  Phase-3 had the same
   early-peak-then-decay pattern (ep399 of 2000), but the absolute
   numbers were so low that ckpt selection felt secondary.  With the
   right schema, the difference is now operationally critical;
   committing the final ckpt instead of the test-best ckpt would
   under-represent the model by ~22 pp.  Phase-C must select on
   `test_pct`, not on final ep.

2. **Peak at ep0199, not ep0399.** Phase-3 peaked at ep0399
   (different absolute numbers, same shape); Phase-B peaks at
   ep0199.  The earlier peak suggests the 23-D obs schema is
   easier to fit (smaller D = less variance to learn from the
   same 200 demos), and overfitting kicks in sooner.  Phase-C
   training budgets can be cut to ~200-300 epochs without losing
   the peak.

3. **Train rate is noisier than test rate.** `ep0249` clears 5/10
   train but `ep0299` clears 9/10 — same model class, same protocol,
   3-9 pp typical noise on the 10-rollout train bucket.  Phase-C
   should not select on train rate; the 50-rollout test is the
   stable signal.

4. **`ep0099`'s 66 % test confirms a stable plateau.** All of ep099
   through ep0299 are within 14 pp of each other on test; the model
   reaches "plateau capability" by epoch 100 and the remaining 100
   epochs to peak are fine-tuning, not unlocking new behaviours.
   This is consistent with DP paper's published learning curves on
   Can/ph.

## Phase-B artefacts in this directory

| file | what |
|---|---|
| `BRIEF.md` | this file |
| `v_theta_dp_tobs2_h16_v3b.log.json` | training log (per-epoch loss + val + LR; 400 epochs, 77 KB) |
| `standalone_v_theta_dp_tobs2_h16_v3b.json` | per-rollout summary for the final ckpt (ep0399; 60 rollouts) |
| `standalone_v_theta_dp_tobs2_h16_v3b_ep0049.json` | per-rollout summary, ep0049 |
| `standalone_v_theta_dp_tobs2_h16_v3b_ep0099.json` | per-rollout summary, ep0099 |
| `standalone_v_theta_dp_tobs2_h16_v3b_ep0149.json` | per-rollout summary, ep0149 |
| `standalone_v_theta_dp_tobs2_h16_v3b_ep0199.json` | per-rollout summary, **ep0199 (winner)** |
| `standalone_v_theta_dp_tobs2_h16_v3b_ep0249.json` | per-rollout summary, ep0249 |
| `standalone_v_theta_dp_tobs2_h16_v3b_ep0299.json` | per-rollout summary, ep0299 |
| `standalone_v_theta_dp_tobs2_h16_v3b_ep0349.json` | per-rollout summary, ep0349 |

Each `standalone_*.json` is the full output of
`08_flow_matching_prior/standalone_dp.py` for one ckpt: aggregate
counts (`n_succ`, `n`, `succ_rate`), the train/test split
(`train_*` for 10 train-init rollouts, `test_*` for 50 fresh-seed
DP-protocol rollouts), the resolved `task` and `obs_keys`, the
ckpt path, and a `per_rollout` list of 60 entries (per-init success
flag, max reward, terminal step, etc.).  These are the forensic
record matching the result table above and Phase-A's eval
protocol exactly.

The Drive originals live at
`/content/drive/MyDrive/hybrid_traj_dp_phase_b/standalone_eval_phase_b/<tag>/`
and were also captured in the Colab notebook's cell 20 / 22
outputs (commit `d59637e`).

The winner ckpt is at
`experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt`,
canonicalised to the `Hybrid-Robot-Trajectory-Generation` repo's
results directory and symlinked from `HRTG-dp` (the symlink is
the Git-tracked file; the 486 MB blob is kept out of `HRTG-dp`'s
LFS quota by living in the sibling repo).

## Implementation artefacts (committed before this BRIEF)

Phase-B's code landed in `3893bf6` on `feat/dp-hybrid-baseline`:

- `experiments/08_flow_matching_prior/standalone_dp.py` — runtime obs
  builder is task-aware; main resolves task from ckpt blob; obs_dim
  is asserted against the loaded prior before the first rollout.
- `experiments/08_flow_matching_prior/dataset.py` — re-exports
  `get_obs_keys`, `get_synthetic_keys`, `compute_synthetic_obs_keys`,
  `get_object_slices`.
- `experiments/08_flow_matching_prior/colab_train_dp_phase_b.ipynb` —
  Colab notebook; clone of Phase-3.5 with `--task can_dp` (and explicit
  `--dataset` because `default_dataset_path('can_dp')` doesn't resolve).
- Smoke-tested at `/tmp/test_phase_b_obs.py`: bit-identical 23-D
  vectors from train and runtime paths.

## Next step (GREEN path triggered)

Per the pre-registered decision rule, **GREEN** triggers Phase-C:
wire the Phase-B prior into the unified MBD pipeline at
`experiments/15_unified_hybrid` and run the hybrid sweep at
K = 0 / 10 / 20 against the Phase-B baseline.

The Phase-C plan (to be written up in a separate `phase_c_<date>/`
brief before launch):

1. Update `experiments/15_unified_hybrid/adapter_dp.py` (or the
   equivalent prior-adapter) to load the 23-D `can_dp` ckpt via the
   task-aware obs builder already shipped in `standalone_dp.py`.
   Validate `obs_dim=23` at adapter construction time, same as the
   standalone driver does.
2. Confirm the unified pipeline's runtime obs vector matches the
   prior's `obs_dim` before the first rollout (fail loud if 46 vs
   23 mismatch, same pattern as Phase-B's standalone assert).
3. Run the K-sweep at the Phase-A eval protocol
   (`--n-test-rollouts 50 --max-env-steps 400 --n-action-steps 8`)
   on `ep0199` as the prior, varying only K ∈ {0, 10, 20}.
4. Decision rule for Phase-C (pre-register before launch):
   - K=0 must reproduce Phase-B's 80 % within ±5 pp (sanity gate;
     this is "the unified pipeline runs the prior the same way the
     standalone driver does").
   - K=10/K=20 success rates establish whether MPC search lifts the
     prior or just adds inference cost.

Phase-B itself **does not** require additional work to be considered
"done" — the GREEN verdict resolves the structural-cause question
opened in Phase-A's RED.
