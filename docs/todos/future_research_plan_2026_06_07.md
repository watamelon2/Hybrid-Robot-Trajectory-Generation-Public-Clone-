# Future research plan — after presentation submission

## Immediate priority: presentation claim validation

### 1. Replicate and localize Square

**Status.** Superseded by the Square-specific DP prior follow-up on
2026-06-08.  The earlier zero-training `H=300, λ=2` result remains a
useful diagnostic, but it is no longer the primary Square path.

**Resolved answer.** Training a task-specific `square_dp` prior reaches
20/30 = 66.7% standalone and 21/30 = 70.0% with the pull-mode hybrid at
`λ=0.5` / `λ=1.0`.  This resolves the immediate Square validation
question as a representation/schema issue rather than a generic
score-bridge failure.  See
`docs/results_writeups/square_dp_prior.md`.

**Original zero-training localization plan.**

| Cell | Purpose |
|---|---|
| `H=300, λ=2.0` replicate | Confirm `15/30 = 50.0%` |
| `H=300, λ=1.5` | Left neighbor |
| `H=300, λ=2.5` | Right neighbor |
| `H=300, λ=3.0` | Strong-bridge limit |

**Instrumentation.** Add or enable failure-phase logging: approach, grasp, lift, align, insert.

**Scaffold.** `experiments/15_unified_hybrid/slate_square_dp_failure_trace_2026_06_09.sh`
prints the exact Square-DP peak trace command by default and only runs if
`RUN=1` is set.  The paired analyzer is
`experiments/15_unified_hybrid/analyze_square_failure_phases.py`.

**2026-06-09 execution.** The `λ=1.0` trace completed at 17/30 = 56.7%,
below the earlier 21/30 co-peak.  The compact taxonomy found 17 successes,
3 `align_or_insert` failures, and 10 `post_contact_unclear` failures.
Because `λ=0.5` was the other 70.0% co-peak, trace `λ=0.5` next before
making a final residual-failure claim.

**Decision update.**

- Promote Square as a solved/partially solved cross-task result under the
  Square-specific DP prior.
- Keep the `H=300, λ=2` replicate/local sweep as optional robustness
  work, not as a blocking validation job.
- The remaining Square work is failure-phase logging for the 9/30
  failures at the `square_dp` hybrid peak, followed by object-relative
  or phase-conditioned subgoals only if the failure taxonomy warrants it.

## Next method direction: long-horizon sparse tasks

### 2. Move tool_hang / transport to subgoal or value signals

Evidence so far says tool_hang is not solved by more static MBD tuning:

- raw sparse reward: 0% cells,
- dense shaping: not enough,
- horizon increase: not enough,
- sigma sweeps: not enough,
- product/gate/temper-p: weak,
- demo-phase pseudo-cost: high dense rewards but not success.

**Plan.**

1. Define task phases or geometric subgoals.
2. Train or hand-code a phase/value signal.
3. Run short-horizon per-phase MBD rather than one long sparse objective.
4. Compare against the existing additive score-bridge, not against product/gate variants.

**Scaffold.** `experiments/15_unified_hybrid/tool_hang_subgoal_plan_2026_06_09.md`
defines the candidate phase predicates.  The manual key-discovery launcher is
`experiments/15_unified_hybrid/slate_tool_hang_subgoal_scaffold_2026_06_09.sh`
and does not execute unless `RUN=1` is set.

**2026-06-09 execution.** The tiny key-discovery rollout completed 0/1 with
120 traced steps.  The compact trace exposed `robot0_eef_pos` and `tool_pos`,
but not `target_pos` or `tool_hole_pos`, so any alignment/hang classifier
needs a richer observation extractor or task-specific simulator access.

**Deprioritized.** More adaptive `sigma_demo`, finite-p tempering, max-gate tuning, and lambda annealing.

## Trained-prior research question

### 3. When does MBD help a trained prior?

**Status.** Partly answered by the Can DP checkpoint sweep and the
Square-DP follow-up.  On Can, the moderate checkpoint/bridge operating
point gives the clearest lift; on Square, a strong task-specific prior
has a smaller but positive lift from 66.7% standalone to 70.0% hybrid.

**Hypothesis.** MBD helps a trained prior only when the prior score is coherent but still has headroom.

**Plan.**

| Prior checkpoint regime | Expected behavior |
|---|---|
| Weak / undertrained | poor score field, little or unstable lift |
| Moderate | coherent + headroom, largest bridge lift |
| Saturated / overtrained | strong standalone, small bridge lift |

Use matched seed buckets to close any remaining ambiguity around DP
checkpoint selection.  The outstanding targeted correction is to rerun
the prior T8 D1/D2 cells with the canonical `ep0199` checkpoint rather
than the weaker `ep0149` checkpoint before citing them as final.

**Scaffold.** `experiments/17_dp_hybrid/slate_ep0199_checkpoint_correction_2026_06_09.sh`
prints the D1/D2 ep0199 correction commands by default and only executes if
`RUN=1` is set.

**2026-06-09 execution.** The checkpoint-corrected cells completed:
D1 `λ=1, σ_max=0.10` reached 19/30 = 63.3%, and D2 `λ=10, σ_max=0.25`
reached 21/30 = 70.0%.  Use these as the canonical D1/D2 `ep0199`
numbers.

## Theory / paper-writing direction

### 4. Bank-size saturation as finite-coverage theory

**Empirical anchor.** Can bank-size scaling rises from 23.3% at `N_b=5` to 65.0% at `N_b=20`, then plateaus at 68-70% through `N_b=200`.

**Plan.**

1. Turn the curve into a paper figure.
2. Relate the knee to finite mode coverage.
3. Separate in-bank vs out-of-bank initial-state performance.
4. Derive or cite a finite-nearest-neighbor coverage bound with practical constants.

**Scaffold.** `experiments/15_unified_hybrid/plot_bank_size_saturation.py`
reads the already-completed June-07 JSONs and writes a paper-style saturation
figure without launching any experiments.

**2026-06-09 execution.** The plotting scaffold generated
`docs/results_writeups/bank_size_saturation.pdf`.

## Engineering appendix

### 5. Keep MJX as an opt-in acceleration path

Gate-4 currently works as an opt-in backend and reached 13.8x on Colab L4, but this is below the promotion threshold.  Keep it as an engineering appendix unless future batching/scene simplification clears the 20-30x bar.

## Suggested order after the presentation

1. Commit and merge the Square-DP prior branch.
2. Rerun / correct any DP checkpoint-comparison cells that still use
   `ep0149` where the canonical result requires `ep0199`.
3. Add Square failure-phase logging for the `square_dp` peak cells.
4. Prototype tool_hang value/subgoal decomposition; do not spend more
   time on static product/gate or adaptive-sigma tuning.
5. Turn bank-size saturation into a paper figure and finite-coverage
   explanation.
6. Keep MJX as an opt-in engineering accelerator unless a later backend
   clears the 20-30x promotion threshold.
