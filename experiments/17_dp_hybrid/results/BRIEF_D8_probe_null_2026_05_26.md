# D8 de-risking probe — null result (2026-05-26)

**TL;DR**: Pull-mode MBD at `K=10, λ=1, σ_max=0.10` on the under-trained DP
checkpoint lands at **7/30 = 23.3 %**, statistically identical to the K=0
unified baseline (also 7/30 = 23.3 %). McNemar exact two-sided p = 1.000.
**Narrative A ("MBD lifts under-trained DP") is falsified at this
operating point.**

## Headline table

| Config | Success | Wilson 95% CI | Notes |
|---|---:|:---:|---|
| `standalone_dp` (paper repro) | 8/30 = 26.7 % | [14.2 %, 44.4 %] | direct run of `standalone_dp.py` |
| Unified K=0 (baseline, D7full) | 7/30 = 23.3 % | [11.8 %, 40.9 %] | post three-bug-fix; warm-start only |
| Unified K=10 pull-mode (D8 probe) | 7/30 = 23.3 % | [11.8 %, 40.9 %] | `σ_max=0.10, σ_min=0.05, M=16, λ=1` |

All three CIs overlap heavily.  The ~3 pp standalone-vs-K=0 gap (8/30 vs
7/30) is statistical noise but consistently in the same direction —
possible residual unified-driver perturbation; revisit if time.

## What MBD actually does — bimodal shuffling, not lift

The 30-trial paired comparison (K=10 vs K=0, same seeds, same demos)
showed:

- **3 recovered** (K=0 failed → K=10 succeeded): demo_1#2, demo_4#0, demo_4#1.
  Reward jumps from ~2 (total whiff) to ~140-150 (clean success).
- **3 broken** (K=0 succeeded → K=10 failed): demo_0#0, demo_1#0, demo_9#2.
  Reward drops from ~127-157 (success) to 2-37 (whiff or near-miss).
- **24 unchanged** (22 both-fail, 4 both-succeed; the "easy" cluster).

MBD does NOT do small refinements (no "+5 pp from polish").  It either
flips a trial from full whiff (reward ~2) to full success (~150), or
flips a success to a near-miss / whiff.  Pattern is consistent with
MBD perturbing the prior plan enough to land in a *different basin
of attraction* — sometimes a better basin, sometimes a worse basin,
with no net directional bias because the cost-fn signal in the
failing regime is too sparse to steer the selection step.

This matches the structural risks predicted in `17_dp_hybrid.md` §1.5:

1. Failure mode is bimodal, not jittery → MBD cannot do small fixes.
2. R=29 amplifies chunk-level perturbations → small changes compound.
3. DP posterior is tight (σ_min=0.05); MBD perturbations at σ_max=0.10
   are ~2× DP's training scale → MBD moves into low-density regions
   that DP's score model doesn't extrapolate well.
4. Cost-fn signal sparse near failures → selection step is noise.

## Files

- `abl_D8_probe_K10_lam1_smax10/run.log` — full per-trial log
- `abl_D8_probe_K10_lam1_smax10/unified_summary_flow_pull_K-10_R-29_D8_probe.json` — aggregated metrics
- `abl_D7full_K0_30trials/` — K=0 baseline reference
- `baseline_standalone_dp/` — standalone DP reference

## Implications

### Narrative A is falsified at this operating point

The §1.5 thesis was that score-bridge MBD with the pull anchor would lift
the under-trained DP from 36.7 % → ≥ 50 %.  At the most-likely-to-work
operating point (K=10, λ=1, σ_max calibrated to DP's posterior width),
we see exactly zero net lift.  Three more operating points remain
worth testing before declaring the broader hypothesis dead:

| Probe | Cost | Expected outcome |
|---|---:|---|
| `σ_max=0.05` (DP's own σ_min — minimal perturbation) | ~2 h | Probably ≈ K=0; tests whether ANY perturbation is destabilising. |
| `σ_max=0.10, λ=0.5` (looser anchor → more MBD pull) | ~2 h | Could lift OR could destabilise more.  Diagnostic. |
| `σ_max=0.10, M=64` (4× search density at K=10) | ~5 h | If cost-fn has signal, denser search should extract it.  Most decisive test. |

If any of these three lifts the rate to ≥ 33 % (the §2.3 expansion
threshold), the headline survives.  If all three remain at ~23 %, the
falsification is robust.

### Narrative B (train DP longer) is now the higher-EV path

With Narrative A null-result-likely, the project's strongest path
forward is the Phase-2 recipe gap closure outlined in `17_dp_hybrid.md`
§3.2:

- LR schedule with `min_lr=1e-6` floor (weights keep moving).
- EMA decay 0.99 instead of 0.9999 (averaging tracks late dynamics).
- Temporal-jitter augmentation.
- 1000-2000 epoch training.
- Target: 60-80 % standalone.

Cost: ~1 week wall clock for the training; then re-run the hybrid
cells on the improved prior.  The headline becomes the more modest
"MBD adds X pp to a converged DP" rather than the Pareto headline,
but the underlying numbers are more reproducible and the result is
publishable as-is.

### Three-bug fix value is independent

Even with the null hybrid result, the three-bug fix in
`docs/experiments/17_dp_hybrid.md` §4.6 is independently valuable:

- It corrects observable defects in the unified driver that would
  have silently broken any future obs-conditioned T_obs > 1 prior.
- It documents the standalone↔unified parity contract explicitly.
- The 23 pp regression that motivated the diagnosis was real;
  closing it back to 23.3 % (statistically equal to standalone's
  26.7 %) makes the codebase trustworthy for future experiments.

## Recommended next action

**Pause Phase 1 ablation slate; document this null and pivot to Phase 2.**

Specifically:
1. Commit this brief + the D8 probe artifacts.
2. Update `17_dp_hybrid.md` §1.5 and §4 with the falsification
   outcome.
3. Either pursue Phase 2 (train DP longer with revised recipe) OR
   close exp 17 as a published-grade negative result.
