# Final report experiment plan — 2026-06-10

Purpose: define the remaining experiments needed for a coherent final HRTG report. The target is a positive, defensible report story, not a broad catalogue of failed negative results.

## Report goal

The final report should keep canonical results on robosuite and use MJX as a validated acceleration appendix / future direction. The central scientific story should be:

1. Zero-training retrieval can be surprisingly strong on Can.
2. MBD score-bridging helps most when the prior has enough coherence and enough residual headroom.
3. Square was not fundamentally impossible; it needed a task-specific DP representation/schema.
4. Hard sparse tasks such as tool_hang and transport remain future work that motivates subgoal/value decomposition, not more static bridge tuning.

## Current positive results already available

- Can zero-training retrieval + score bridge reaches roughly **73.8%**.
- Bank-size scaling shows a clear knee around **20-50 demos** and a plateau around **68-70%**.
- Can DP moderate-prior setting reaches **86.7%** in the strongest hybrid cell.
- Square-DP reaches **66.7%** standalone and **70.0%** hybrid peak.
- Lift score-bridge results are positive and can support the cross-task table if the artifact is easy to cite.
- MJX Gate-4 is implemented and measured at **13.8x** speedup, but remains appendix/future work because it is not yet a canonical robosuite evaluator.

## Must-have missing experiments

### 1. Square-DP replicate / confirmation

Why: Square-DP prevents the final report from reading as "Can works, Square partly fails, tool_hang fails." It gives the report a second positive task-specific result.

Run:

- Square-DP `ep0199`, `lambda=0.5`, second seed band, `n=30`.
- Square-DP `ep0199`, `lambda=1.0`, second seed band, `n=30`.
- Save step traces or failure tags for failures.

Optional:

- Standalone Square-DP second seed band.

Desired outcome: replicate or approximately confirm the current peak. Even **60-70%** keeps the story positive.

Estimated wall time: **6-12 hours**.

Priority: **highest**.

### 2. Clean Can-DP checkpoint/headroom comparison

Why: the report needs a clear checkpoint story.

Key distinction:

- `ep0199` is the canonical stronger standalone DP reference.
- `ep0149` is the moderate-prior hybrid-lift operating point.

Minimum final table:

| Checkpoint regime | Standalone / lambda=0 | Best hybrid | Interpretation |
|---|---:|---:|---|
| weak / early | low | limited | prior not coherent enough |
| `ep0149` | moderate | high | best headroom regime |
| `ep0199` | strong | similar or modest lift | near saturation |

Run or confirm:

- Selected `ep0149` best hybrid setting.
- Selected `ep0199` reference/hybrid settings under the same seed protocol.

Desired outcome: show that MBD has a headroom regime, not universal dominance.

Estimated wall time: **3-8 hours** selected, **10-16 hours** for fuller grid.

Priority: **highest**.

### 3. Fresh-seed / out-of-bank retrieval coverage check

Why: strengthens the bank-size curve against the criticism that it only replays training starts.

Run minimum:

- Can `demo_1nn`, `lambda=1`, bank sizes `20`, `50`, `200`.
- Fresh test seeds, `n=30` each.

Better:

- Two seed bands, `n=60` per bank size.

Desired outcome: same qualitative trend as original curve: small banks under-cover, `20` gives a major jump, `50/200` plateau.

Estimated wall time: **4-8 hours** minimum, **8-16 hours** stronger version.

Priority: **high**.

### 4. Baseline ablations for each headline

Why: every positive claim needs a paired baseline.

Check that the final tables include:

| Claim | Needed baseline |
|---|---|
| Can zero-training retrieval | `lambda=0` warmstart-only or pure retrieval baseline |
| Can-DP headroom | standalone / `lambda=0` DP |
| Square-DP | standalone Square-DP and MBD-only / `lambda=0` |
| Bank-size scaling | small-bank `N_b=5/10` cells |

Estimated wall time: **0 hours** if artifacts already exist, **2-6 hours** if one or two cells are missing.

Priority: **high**.

## Should-have experiments

### 5. Square-DP failure-phase labels

Why: turns remaining Square failures into concrete future work instead of generic failure reporting.

Label failures as:

- approach failure,
- grasp failure,
- alignment failure,
- insertion / placement failure,
- drift after success.

Estimated wall time: **2-5 hours**.

Priority: **medium-high**.

### 6. Lift spot check if the artifact is not clean

Why: a cross-task table with Lift, Can, and Square looks much more positive.

Desired table shape:

| Task | Best method | Success |
|---|---|---:|
| Lift | score bridge | about 100% |
| Can | zero-train retrieval | about 73.8% |
| Square | Square-DP hybrid | about 70.0% |

Run only if the existing Lift artifact is not easy to cite.

Estimated wall time: **0 hours** if artifact is clean, **2-5 hours** if rerun.

Priority: **medium**.

## Nice-to-have, not required

### 7. Adaptation mass/friction subset

Run only a small representative subset if adaptation becomes part of the final story:

- canonical bank vs adapted bank,
- one mass perturbation,
- one friction perturbation.

Estimated wall time: **4-12 hours** selected subset.

Priority: **medium-low**.

### 8. MJX Level-1 prototype

For the final report, do not use new MJX experiments as science results. Use existing Gate-4 results:

- OSC parity,
- Colab L4 speedup,
- **13.8x** measured acceleration,
- not promoted to canonical backend.

Priority for final report: **low**.

## Experiments not to chase before the final report

Do not spend final-report time on:

- more product/gate/tempered sweeps,
- adaptive sigma tuning,
- lambda annealing,
- more tool_hang static bridge tuning,
- transport static bridge tuning,
- full MJX rewrite,
- exhaustive Square/task MJX scene work,
- full SOTA benchmark reimplementation.

These either already have a negative conclusion or will not become a positive coherent story before the deadline.

## Recommended final experiment package

| Priority | Experiment | Purpose |
|---:|---|---|
| 1 | Square-DP `lambda=0.5/1.0` replicate | Confirm Square positive result |
| 2 | Can-DP selected checkpoint/headroom cells | Clean DP story |
| 3 | Can fresh-seed retrieval bank `20/50/200` | Defend finite coverage |
| 4 | Missing `lambda=0` / standalone baselines | Make tables interpretable |
| 5 | Square-DP failure labels | Make future work concrete |

Estimated wall time:

- Minimum: **18-30 hours**.
- Comfortable: **2-3 days** including aggregation and table cleanup.

## Final report structure supported by this plan

1. **Zero-training retrieval prior works on Can.** High success without learned parameters; bank-size scaling supports finite coverage.
2. **MBD score bridge helps in the headroom regime.** Moderate DP checkpoint improves strongly; strong checkpoint has less room.
3. **Square improves with task-specific DP representation.** Earlier Square failures were not simply "MBD fails"; representation/schema matters.
4. **Hard sparse tasks remain future work.** tool_hang / transport motivate subgoal/value decomposition.
5. **MJX is a systems future direction.** It is implemented and measured, but not used as a canonical result source yet.

## Bottom line

The project is missing confirmation and coherence, not a brand-new discovery. The most important missing experiments are Square-DP replication, Can-DP headroom cleanup, fresh-seed retrieval coverage, and headline baselines.
