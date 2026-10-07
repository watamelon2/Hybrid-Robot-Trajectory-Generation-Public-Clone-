# Brief — Phase-2 DP retrain (1000 ep + min_lr + ema=0.99): pivot

*Date:* 2026-05-27 PT
*Branch:* `feat/dp-hybrid-baseline`
*Worktree:* `HRTG-dp`
*Hardware:* Colab L4 GPU (NVIDIA L4 @ 23 GB), 1.9 h wall
*Checkpoint:* `experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v2.pt` (508 MB, 66.5 M params, EMA primary + `state_dict_raw`)
*Log:* `experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v2.log.json` (1000-epoch loss/lr trace)
*Eval JSON:* `experiments/17_dp_hybrid/results/colab_standalone_dp_phase2/standalone_dp_phase2_gpu.json`
*Status:* **negative — pivot per the pre-registered <35 % threshold.**

## TL;DR

Phase-2 ran the `docs/experiments/17_dp_hybrid.md` §3.2 recipe end
to end (cosine LR with `min_lr=1e-6` floor, EMA decay 0.99, 1000
epochs, identical 66 M-param arch). Training converged cleanly —
final loss 7.8e-4 vs the 50-ep baseline's 0.034 (**43× lower**),
and the LR genuinely sat at the 1 e-6 floor through epochs 940-999
instead of bottoming out near zero like the original 50-ep cosine
or the over-annealed 300-ep cousin. **Standalone evaluation on
Can/`ph` regressed to 4/30 = 13.3 %**, below the 50-ep baseline's
26.7 % and the 300-ep EMA cousin's 16.7 %. Falls in the
pre-registered `< 35 %` "pivot away" band of §3.2.4.

## Headline numbers

| Run | Train loss (final) | LR floor held | Standalone Can/ph (n=30) |
|---|---|---|---|
| 50-ep ref (`v_theta_dp_tobs2_h16.pt`) | 0.034 | no — drops to ~1 e-7 | **26.7 %** (8/30) |
| 300-ep EMA cousin (`v_theta_dp_tobs2_h16_ema_e300.pt`) | 0.0071 | no — drops to ~3 e-9 | **16.7 %** (5/30, prior diagnosis) |
| **Phase-2 v2 (this run)** | **0.00078** | **yes — 1 e-6 floor for the last ~60 ep** | **13.3 %** (4/30) |

(50-ep baseline number from `experiments/17_dp_hybrid/results/baseline_standalone_dp/standalone_dp_repro.json`; pre-registered planning-doc reference value is 36.7 % from an earlier run — the 26.7 % is the latest in-tree repro on the same eval grid as Phase-2.)

## What the eval looks like at the per-demo level

Bimodal — the four successes pile onto two demos:

| Demo | rollout 0 | rollout 1 | rollout 2 |
|---|---|---|---|
| demo_0 | fail (r=2.29) | fail (r=2.06) | fail (r=2.07) |
| demo_1 | fail (r=1.95) | fail (r=2.92) | fail (r=2.26) |
| demo_2 | fail (r=3.94) | fail (r=3.98) | **succ (r=29.3)** |
| demo_3 | fail (r=2.27) | fail (r=2.15) | fail (r=1.94) |
| **demo_4** | **succ (r=27.8)** | **succ (r=27.0)** | **succ (r=26.8)** |
| demo_5 | fail (r=1.71) | fail (r=2.05) | fail (r=1.67) |
| demo_6 | fail (r=1.34) | fail (r=1.18) | fail (r=1.39) |
| demo_7 | fail (r=2.15) | fail (r=2.10) | fail (r=2.27) |
| demo_8 | fail (r=9.70) | fail (r=8.92) | fail (r=8.59) |
| demo_9 | fail (r=4.29) | fail (r=6.05) | fail (r=5.07) |

Two of the three failure modes:

- **demo_4** is the "memorised" trajectory — three rollouts all complete in 118-120 steps with reward ~27 (matches success threshold).
- **demos 6, 8, 9** plateau at reward 1-10 across all 240 max-env-steps — the policy is stuck doing *something* (high enough reward to be non-trivial, demo_8 actually averages r≈9.1) but never closes out.
- **demos 0, 1, 3, 5, 7** show flat reward ~2 across rollouts — generic whiff.

This is the **same bimodal signature** that the planning doc §1.5
risk register flagged: "DP either fully succeeds (reward ≥ 27) or
whiffs entirely (reward ~2)." More training collapsed the success
set further (from 7 demos / 30 rollouts seeing partial wins → 2
demos / 30 rollouts).

## Recipe sanity check (the fix DID land)

The Phase-2 recipe was specifically designed to address the "no-op
EMA + premature LR cooldown" diagnosis in §3.1. Both knobs landed:

- **LR floor** — last 60 epochs at `lr=1e-6` (verified in the log
  JSON). In the 50-ep cosine, the floor was effectively zero
  (cosine bottoming below 1 e-7); EMA can't track when the model
  isn't moving.
- **EMA decay = 0.99** — half-life ~70 steps at 91 batches/epoch
  ≈ 0.77 epochs. Tracks the late-training dynamics rather than
  freezing on initialisation (the DP-paper default 0.9999 has
  half-life ~700 epochs, longer than the entire 50-ep baseline).

So the recipe is **not** the variable that broke. The training
ran exactly as designed.

## What this confirms / falsifies

This is direct evidence for the §3.1 "**overfitting + an
under-parameterised dataset**" diagnosis, **not** for the prior
"EMA-application bug at eval time" theory:

- Successfully implementing the §3.2 recipe (loss converged 43×
  lower, EMA correctly tracking) makes the standalone result
  **worse**, not better.
- The monotonic ordering 50 ep (26.7 %) → 300 ep (16.7 %) → 1000
  ep (13.3 %) tracks training duration almost perfectly. More
  compute = worse generalisation.
- The 66 M-param UNet against 200 demos × ~116 chunks ≈ 23 k
  training chunks (~0.35 M training tokens against 66 M params,
  a ~190× over-parameterisation ratio) is the structural issue
  the planning doc already named.

## Recommended next step

Per the pre-registered §3.2.4 rule for the `< 35 %` band:

> Recipe didn't fix the under-training. Pivot to: (a) v6 CFM
> prior route (already at 75.9 %), or (b) data scaling (200 →
> 600 demos).

**Pivot (a) is the cheap one** — v6 CFM is already trained, on the
shelf at `experiments/08_flow_matching_prior/results/v_theta_v6_unet_xl.pt`,
and hits 75.9 % standalone. Phase 1 (DP + MBD hybrid) was already
falsified at 7/30 in `BRIEF_D8_probe_null_2026_05_26.md`; Phase 2
falsifying the "long training fixes it" alternative collapses the
DP route entirely. v6 CFM becomes the production prior for any
remaining hybrid work this week.

**Pivot (b) (data scaling)** is a ~1-2 d effort; defer unless v6 CFM
also caps out and we need a learned-prior story past the report
deadline.

## Files of record

- `experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v2.pt` (508 MB; **gitignored** by `experiments/**/results/*.pt`; kept on disk for any later spot checks)
- `experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v2.log.json` (committed)
- `experiments/17_dp_hybrid/results/colab_standalone_dp_phase2/standalone_dp_phase2_gpu.json` (committed)
- This brief (committed)

## Reproduction

The Colab notebook that produced the checkpoint, with the
idempotent re-run guards added today:

```
experiments/08_flow_matching_prior/colab_train_dp_phase2.ipynb
```

Run cells 1-15 (training is idempotent now — cell 14 will skip if
the checkpoint exists on Drive). Cell 19 (robosuite install) +
cell 20 (standalone eval) reproduce this exact 4/30 number;
`PYTHONHASHSEED=0` + `seed=0` was set both times.
