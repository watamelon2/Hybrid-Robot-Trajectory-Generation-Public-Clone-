# Research review handoff — 2026-06-10

Purpose: hand this document to another agent as a compact review of the current HRTG results, writing state, novelty position, consistency issues, and recommended next directions.

## Executive verdict

- **Novelty assessment:** conditionally novel / strong class-project or workshop-level if framed correctly.
- **Best framing:** not "we beat Diffusion Policy" and not "retrieval is new." The strongest story is that a lightweight prior-guided MBD score bridge can turn demonstration retrieval or moderately trained priors into robust planners, and that its benefit appears only in specific regimes: prior coherence, residual headroom, and a stable pull anchor.
- **Core thesis:** MBD is not a black-box add-on to any prior. It helps when the prior provides a coherent reference but still leaves room for cost-guided correction.

## Most defensible contributions

1. **Zero-training retrieval + score bridge.** `demo_1nn` retrieval plus additive score bridge reaches **73.8% Can** with no trained parameters, measured as 177/240 with Wilson CI [67.8, 78.9].
2. **Trained DP + bridge.** A moderately trained DP checkpoint (`ep0149`) plus pull-mode MBD reaches **86.7% Can** at `K=10, lambda=2.0` and is corroborated by **83.3%** at `K=20, lambda=0.5`.
3. **Bank-size scaling.** Retrieval success rises from **23.3%** at `N_b=5` to **65.0%** at `N_b=20`, then plateaus around **68-70%** through `N_b=200`, supporting finite mode coverage rather than full-bank memorization.
4. **Square-DP follow-up.** Square improves sharply with a task-specific DP observation schema: **66.7% standalone** and **70.0% hybrid** at `lambda=0.5/1.0`, superseding the earlier urgent Square `H=300, lambda=2` replication plan.
5. **Informative negative results.** Product/gate/tempered variants, adaptive sigma, lambda annealing, and demo-phase pseudo-cost mostly fail or underperform additive score bridging. These negatives are useful for narrowing the method story.

## Critical consistency issues to fix

### 1. Active branch may be stale relative to `origin/main`

At the time of review, the checked-out branch was:

- `revise/final-presentation-slides-2026-06-07`

But `origin/main` contains newer Square-DP documentation and results:

- `docs/results_writeups/square_dp_prior.md`
- updated `docs/results.md`
- updated `docs/todos/future_research_plan_2026_06_07.md`

Before editing final reports or slides, merge/rebase the presentation branch against the current main branch or explicitly mark the branch as pre-Square-DP.

### 2. Square narrative is superseded

Older deck/report wording says Square is partially rescued by zero-training retrieval at:

- `H=300, lambda=2.0`: **15/30 = 50.0%**

This remains a useful diagnostic, but it is no longer the primary Square result. The newer Square-DP result is stronger:

- Standalone Square-DP: **20/30 = 66.7%**
- Hybrid `lambda=0.5`: **21/30 = 70.0%**
- Hybrid `lambda=1.0`: **21/30 = 70.0%**

Recommended wording: Square was not a generic score-bridge failure; it needed a task-specific representation/schema.

### 3. DP checkpoint roles must stay distinct

Do not substitute `ep0149` and `ep0199` casually.

- **`ep0199`: canonical stronger standalone DP reference.** It scores about **76-80%** depending on seed bucket and should be used when asking how strong DP is by itself.
- **`ep0149`: hybrid-lift operating point.** It is where the bridge produces the strongest positive lift: **86.7%** at `K=10, lambda=2.0` and **83.3%** at `K=20, lambda=0.5`.

Correct interpretation: `ep0199` is the stronger standalone checkpoint; `ep0149` is the checkpoint where MBD has the most headroom to help.

### 4. SOTA reference numbers need caution

`docs/references/sota_reference_numbers.md` is useful, but some entries should be verified before being used in a polished report or slide:

- The Diffusion Policy **94.7%** number mixes image/low-dim crosswalk language and should be tied to an exact table/row before being cited precisely.
- The Diffuser **~80-85%** manipulation comparison appears weakly supported and should remain approximate or be removed unless verified.
- Keep approximate values marked with `~` and footnote observation modality, split, and seed protocol.

## Evidence-strength assessment

### Strong evidence

- **Can zero-train retrieval:** 177/240 = 73.8%, pooled across four seeds.
- **Bank-size scaling:** n=60 per cell, clean saturating curve with a knee around 20-50 demos.
- **DP hybrid regime story:** the grid supports the claim that bridge lift appears at a moderate checkpoint but not at weak or saturated prior extremes.
- **Negative result triage:** enough failed variants exist to justify deprioritizing product/gate/adaptive-sigma/static-tempering directions.

### Medium evidence

- **Can DP hybrid 86.7%:** promising but n=30; confidence comes from the independent K=20 corroborating cell.
- **Square-DP 70.0%:** strong directional evidence but n=30; should be replicated if promoted as a headline.
- **Square zero-train 50.0%:** useful diagnostic, now secondary to Square-DP.

### Weak or heavily caveated evidence

- **MJX:** implemented and GPU-measured, but speedup is **13.8x**, below the 20-30x promotion threshold. Keep as engineering appendix.
- **tool_hang / transport:** not solved. Current evidence says static MBD tuning is insufficient.
- **CFM v6:** useful but not the strongest story. It has 56% standalone and 70% hybrid in the current notes, but the protocol caveats matter.

## Writing review

### What is working

- Strong provenance: raw JSON paths, checkpoints, seed buckets, and Wilson intervals are usually recorded.
- The docs honestly preserve negative results and decision gates.
- `bank_size_scaling.md`, `square_horizon_diagnostic.md`, and `mjx_gate4_closeout.md` are clear and decision-oriented.
- The presentation has a good narrative arc: motivation, composite score, cross-task results, DP lift, adaptation, lessons.

### What to improve

1. **Add a short canonical current-claims doc.** `docs/results.md` is huge and historical. Create or maintain a compact page such as `docs/current_claims.md` with only active claims, source files, and caveats.
2. **Label diagnostic vs headline results.** Example: Square `H=300, lambda=2` is diagnostic; Square-DP 70% is the headline/follow-up result.
3. **Use consistent names.** Recommended terms:
   - `zero-train retrieval + score-bridge`
   - `DP ep0199 standalone reference`
   - `DP ep0149 hybrid-lift operating point`
   - `Square-DP task-specific prior`
4. **Narrow broad claims.** Avoid saying "MBD helps trained priors" without qualification. Prefer: "MBD helps when the prior is coherent and has headroom."
5. **Archive or mark superseded narratives.** Do not leave old Square or CFM readings adjacent to newer results without labels.

## Suggested future directions

### Priority 1 — Replicate Square-DP peak

- Run a second seed band for Square-DP `lambda=0.5` and/or `lambda=1.0`.
- Add failure-phase labels for the 9 failures at the 21/30 peak.
- If failures cluster around alignment/insertion, add object-relative or peg-relative features.

### Priority 2 — Correct remaining DP checkpoint comparisons

- Re-run any T8 / DP-prior cells that mistakenly use `ep0149` where the canonical standalone/reference claim requires `ep0199`.
- Keep `ep0149` only for the moderate-prior headroom / hybrid-lift claim.

### Priority 3 — Turn bank-size scaling into a paper figure and theory section

- Plot the `N_b=5,10,20,50,100,200` curve.
- Add in-bank vs out-of-bank initial-state split if possible.
- Relate the knee to finite nearest-neighbor / mode-coverage theory.

### Priority 4 — Tool-hang and transport need subgoals/value signals

Current evidence says more static tuning is not the path. Do not prioritize:

- adaptive `sigma_demo`,
- product/gate/temper-p,
- lambda annealing,
- more static max-gate tuning.

Instead:

1. define task phases or geometric subgoals,
2. train or hand-code a phase/value signal,
3. run short-horizon per-phase MBD,
4. compare against additive score-bridge only.

### Priority 5 — Keep MJX as appendix unless speed improves

- Current Colab L4 Gate-4 speedup is **13.8x**.
- Keep as implemented, GPU-measured, and useful for engineering iteration.
- Do not promote as canonical evaluation backend unless batching/scene simplification clears 20-30x.

## Recommended immediate edits for another agent

1. Merge or compare the active presentation branch with `origin/main` so Square-DP docs are visible.
2. Update presentation Slide 5 and Slide 9 to mention Square-DP 70% or explicitly label the deck as pre-Square-DP.
3. Create/update a compact `docs/current_claims.md`.
4. Revise `docs/references/sota_reference_numbers.md` to mark uncertain SOTA rows as approximate until verified.
5. Audit all mentions of `ep0149` and `ep0199` for the checkpoint-role distinction.
6. Keep MJX out of headline figures unless the speedup threshold is met.

## Bottom line

The project is strongest as an empirical map of **when prior-guided MBD works**, not as a pure benchmark chase. The defensible research story combines:

- zero-training retrieval,
- additive score bridging,
- finite-coverage scaling,
- trained-prior headroom,
- Square representation/schema correction,
- and honest negative results for alternative combination rules.

This is a coherent narrative if the docs are updated to separate current claims from superseded diagnostics.
