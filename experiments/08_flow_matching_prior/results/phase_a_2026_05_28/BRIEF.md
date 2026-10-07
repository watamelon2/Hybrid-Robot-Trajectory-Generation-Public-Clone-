# Phase-A: DP eval-pipeline reproduction — 2026-05-28

Re-evaluates the two best Phase-3 checkpoints (`ep0399` peak, `ep1999`
final / EMA shadow) under the **exact DP paper eval protocol**, with
the goal of separating two hypotheses for the persistent 25 % standalone
ceiling on Can/ph:

1. **Eval-pipeline bias** — Phase-3 was eval'd on training-init replays
   at `max_steps=240`; DP paper reports on 50 fresh-seed test rollouts at
   `max_steps=400`. The training-init distribution may be either easier
   (model overfits to seen poses) or harder (one very bad init in the
   10-demo set drags the rate down).
2. **Structural ceiling** — the model itself is undertrained or has an
   architectural / data-pipeline defect that caps it ~25 % regardless of
   eval protocol.

Pre-registered decision rule (cell 22 of `colab_eval_dp_aligned.ipynb`):

| `test_pct` | reading |
|---|---|
| ≥ 50 % | **GREEN** — eval bias was the whole story. Done. |
| 30–50 % | **YELLOW** — eval bias matters, Phase-3.5 (obs min-max norm) still worth running. |
| ~25 %, train ≈ test | **RED** — 25 % ceiling is structural, Phase-B retrain needed. |
| train ≫ test | **ORANGE** — policy overfit to training inits. |

## Result

| ckpt | train (10) | **test (50, DP)** | combined | wall |
|---|---:|---:|---:|---:|
| `v_theta_dp_tobs2_h16_v3_ep0399` (peak) | 2/10 = 20.0 % | **8/50 = 16.0 %** | 10/60 = 16.7 % | 75.2 min |
| `v_theta_dp_tobs2_h16_v3` (ep1999, final) | 2/10 = 20.0 % | **5/50 = 10.0 %** | 7/60 = 11.7 % | 74.9 min |

Reference: DP paper Can/ph reports **mean_score ≈ 85 %** on the same
test protocol (n_test=50, max_steps=400, n_obs_steps=2, n_action_steps=8,
DDPM K=100).

## Verdict: RED (confirmed)

- **Eval bias is real but small.** Phase-3 ep0399's 26.7 % on the legacy
  training-init protocol drops to 16.0 % on the DP fresh-seed protocol.
  That's a **−10.7 pp** eval-pipeline correction.
- **The 25 % ceiling is structural.** Even on the apples-to-apples DP
  protocol, our best ckpt clears 16 % vs DP's 85 %. The **69 pp residual
  gap** is the model, not the eval.
- **train ≈ test** (within 4 pp for ep0399, 10 pp for ep1999). Not ORANGE
  — no policy-overfit-to-train-inits effect. The model is uniformly
  limited.
- **`ep0399` > `ep1999`** on the proper protocol (16 % vs 10 %), mirroring
  the Phase-3 finding that the model peaks early and overfits.

The cell-22 logic labelled this "inconclusive" because the threshold for
RED required `test_pct ~25 %` *and* `train ≈ test`; `test_pct = 16 %` was
below the lower band. Threshold should be tightened to `test_pct < 30 %`
in a future Phase-A re-run.

## Most likely structural cause: obs schema mismatch

Phase-3 was trained on a **46-D** obs vector. DP paper's `can_lowdim.yaml`
uses a **19-D** vector. The gap, decomposed:

| component | ours (46-D) | DP paper (19-D) | gap |
|---|---|---|---|
| `object` | sliced `[:7]` = `Can_pos(3) + Can_quat(4)` | **full 10-D blob** including `Can_to_eef_pos(3) + Can_to_eef_quat(4)` | DP has **7 extra "relative pose" dims we don't** |
| robot eef | `eef_pos(3) + eef_quat(4)` = 7 | `eef_pos(3) + eef_quat(4)` = 7 | same |
| gripper | `gripper_qpos(2) + gripper_qvel(2)` = 4 | `gripper_qpos(2)` = 2 | we have 2 redundant vel dims |
| joints | `pos + cos + sin + vel` = **28** | (none) | we have **28 dims of joint redundancy** |
| **total** | **46** | **19** | **+27 redundant, −7 critical** |

The critical loss: DP's `object` blob (10 dims) includes the
**object-to-EEF relative pose** directly. That's a learning-free feature
giving the policy "where is the can relative to my hand". Our schema
slices it off and forces the model to learn it from 28 dims of joint
redundancy + 7 dims of absolute EEF pose. This is, by an order of
magnitude, the most likely cause of the 25 % ceiling.

## Hypotheses ruled out

For the record:

- **Dataset chunking convention** — `dataset_v3.load_chunks` emits one
  chunk per timestep (`T` chunks per demo, edge-padded at the tail),
  not "only `t=0`" chunks. Phase-3 trained on 23 207 chunks from 200
  demos = ~116 per demo. DP's `SequenceSampler` with
  `pad_before=1, pad_after=8` would produce ~110 chunks per demo. We
  emit ~5 % **more** chunks than DP, not fewer. Not the bug.
- **`T_obs=1` vs `T_obs=2`** — Phase-3 was trained with `n_obs_steps=2`;
  log confirms `S_cond shape (23207, 92) (was (23207, 46))`. Adapter's
  `_global_cond_from_s0` correctly builds the 2-step history at
  inference. Not the bug.
- **DDIM stride** — fixed in Phase-3 (`model_dp.py:551-664`). Not the bug.
- **EMA bias** — Phase-3's training used EMA(0.9999) on a copy of weights,
  and ep0399 is the non-EMA snapshot, not the EMA shadow. The EMA shadow
  (`ep1999`) is *worse* on the DP protocol (10 % vs 16 %). EMA decay is
  not masking a hidden gain.

## Phase-A artefacts in this directory

| file | what |
|---|---|
| `BRIEF.md` | this file |
| `standalone_v_theta_dp_tobs2_h16_v3_ep0399.json` | per-rollout summary for `ep0399` (60 rollouts, recovered from the disconnected-Colab printed log of the first run + verified against in-notebook recovery cell) |
| `standalone_v_theta_dp_tobs2_h16_v3.json` | per-rollout summary for `ep1999` (60 rollouts, fresh from the 2nd resumed Colab run) |

The originals on Drive live at `/content/drive/MyDrive/hybrid_traj_dp_phase_a/`.

## Pre-registered next step

The RED outcome triggers **Phase-B**: implement `OBS_KEYS_CAN_DP` (19-D,
full `object` blob, no joint redundancy) + runtime obs builder + a
`train_dp.py --obs-schema dp` flag + a Phase-B training notebook;
retrain (~3 hr L4); re-eval with the unchanged Phase-A protocol.

Total wall budget: ~2 hr of agent work + ~3 hr Colab train + ~75 min
Colab eval ≈ **6 hr** end-to-end.

Decision deferred to user before launching the Phase-B work.
