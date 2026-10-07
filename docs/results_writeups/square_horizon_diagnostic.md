# Square horizon diagnostic and representation next step

**Slate:** `experiments/15_unified_hybrid/slate_square_horizon_diagnostic_2026_06_07.sh`
**Status:** completed 2026-06-07.
**Primary question:** is Square still a pure floor on the canonical score-bridge path once the horizon is long enough?

**Primary answer:** Square is not a pure floor, but the rescue is not caused by horizon alone. Warm-start-only at `H=300` scores 3.3%, and `λ=1` remains weak at both `H=200` and `H=300`. The successful cell is the stronger bridge setting, `H=300, λ=2`, which reaches 15/30 = 50.0%.

## Motivation

Earlier Square results at H=120 were near zero and led to a mechanism-limited verdict. The gated/tempered follow-up found that the score path can reach nonzero success when Square is given a longer horizon. This slate reproduces that lesson on `main` without the MBD-branch-only combination-rule flags, then decides whether the next work should target representation/subgoals or abandon Square as out of scope.

## Cells

| Cell | Prior | λ | H | Replans | Chunk | Trials | Purpose |
|---|---|---:|---:|---:|---:|---:|---|
| `square_warmstart_h300` | `demo_1nn` | 0 | 300 | 4 | 75 | 30 | warm-start-only control |
| `square_score_h200_lam1` | `demo_1nn` | 1 | 200 | 3 | 66 | 30 | reproduce H>=200 rescue |
| `square_score_h300_lam1` | `demo_1nn` | 1 | 300 | 4 | 75 | 30 | longer-horizon score path |
| `square_score_h300_lam2` | `demo_1nn` | 2 | 300 | 4 | 75 | 30 | stronger bridge probe |

## Decision gates

- **Representation/subgoal path justified:** any score cell reaches at least 20% and beats warm-start-only by at least 10 pp. Next step: phase-conditioned / object-relative retrieval and insertion subgoal diagnostics.
- **Partial rescue only:** score cells land between 10% and 20%. Next step: failure-phase logging before new training.
- **Still floor:** every score cell remains under 10%. Square is not unlocked by horizon under the current representation; only proceed if the paper needs a hard-task section.

**Outcome:** the representation/subgoal path is justified. `square_score_h300_lam2` reaches 50.0%, beating the `H=300, λ=0` warm-start control by +46.7 pp, and its Wilson interval `[33.2%, 66.8%]` does not overlap the warm-start interval `[0.6%, 16.7%]`. This supersedes the earlier "Square is near-zero" reading from shorter-horizon runs.

## Analysis

- **Long-horizon warm-start alone fails.** `H=300, λ=0` gets only 1/30 = 3.3%, so simply replaying a retrieved long action window does not solve Square.
- **Horizon alone is insufficient at `λ=1`.** `H=200, λ=1` gets 10.0%, while `H=300, λ=1` gets 6.7%. Within this small n=30 slate, longer horizon does not improve success unless bridge strength is also increased.
- **Bridge strength is the key observed lever.** Moving to `H=300, λ=2` gives 50.0%, a large and statistically separated improvement over warm-start-only.
- **Reward suggests partial progress before success.** Mean reward increases from 36.319 to 61.459 to 79.364 for the first three cells, but success remains low until `λ=2` raises mean reward to 170.959. This points to near-miss / phase-completion structure, not a total inability to make progress.
- **Interpretation for the roadmap.** Square should move from "maybe out of scope" to "hard but unlocked by stronger score-bridge." The next work should focus on robustifying this setting and diagnosing failure phases, not returning to product/gate or warm-start-only variants.

## Recommended next experiment if gate passes

First, replicate and localize the new operating point:

1. rerun `H=300, λ=2` at larger n or a second seed band,
2. sweep nearby bridge strengths such as `λ ∈ {1.5, 2.0, 3.0}`,
3. log failure phase and reward decomposition for the unsuccessful rollouts.

Then create a Square representation slate with:

1. object-relative features around nut/peg alignment,
2. nearest-demo phase conditioned on insertion progress,
3. saved failure taxonomy for approach / grasp / lift / insert phases,
4. score-bridge only at first; do not reintroduce product/gate.

## Results table

| Cell | Success | Rate | Wilson 95% CI | Mean reward | Notes |
|---|---:|---:|---:|---:|---|
| `square_warmstart_h300` | 1/30 | 3.3% | [0.6%, 16.7%] | 36.319 | long-horizon warm-start control fails |
| `square_score_h200_lam1` | 3/30 | 10.0% | [3.5%, 25.6%] | 61.459 | partial rescue only |
| `square_score_h300_lam1` | 2/30 | 6.7% | [1.8%, 21.3%] | 79.364 | longer horizon alone does not rescue |
| `square_score_h300_lam2` | 15/30 | 50.0% | [33.2%, 66.8%] | 170.959 | gate pass; Square unlocked by stronger bridge |
