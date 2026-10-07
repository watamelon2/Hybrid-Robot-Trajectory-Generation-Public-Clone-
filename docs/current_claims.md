# Current claims — canonical, active (2026-06-11)

Compact source-of-truth for **active** HRTG claims (recommended by
`research_review_handoff_2026_06_10.md`). `docs/results.md` remains the full
historical record; this page lists only what we currently stand behind, with
sources and evidence strength. Diagnostic/superseded results are labeled.

## Headline claims

| # | Claim | Number | Evidence | Source |
|---:|---|---|---|---|
| 1 | Zero-training retrieval + additive score-bridge solves Can | **177/240 = 73.8%** (Wilson [67.8, 78.9]) | Strong (pooled 4 seeds) | `results_writeups/bank_size_scaling.md`, handoff |
| 2 | Bank-size = finite mode coverage | 23.3% (N_b=5) → 65% (N_b=20) → plateau 68-70% (N_b≤200) | Strong (n=60/cell) | `results_writeups/bank_size_scaling.md` |
| 3 | Trained DP + bridge hybrid-lift (moderate checkpoint) | **86.7%** Can @ K=10,λ=2.0 (`ep0149`); 83.3% @ K=20,λ=0.5 | Medium (n=30 + corroborating cell) | handoff, `results.md` |
| 4 | DP `ep0199` is the stronger standalone reference | **76-80%** depending on seed bucket | Medium | handoff |
| 5 | Square needs a task-specific DP representation | **66.7%** standalone, **70.0%** hybrid @ λ=0.5/1.0 | Medium (n=30) | `results_writeups/square_dp_prior.md` |
| 6 | Fast adaptation by bank editing > BC retraining (bite zone) | T2.1 **67.8%** (61/90) vs T2.3 BC-retrained **6.7%** (6/90); Δ≈+61 pp | Strong (3-seed pool) | `results_writeups/t2_multiseed.md` |

## Supporting / robustness claims

- **λ robustness (adaptation):** plateau 73.3-80% for λ∈[1,5]; λ=0.5 collapses to 30% (`t2_lambda.md`).
- **Demo-count saturation:** adaptation saturates by **n=1** perturbed demo within the n=5 CI (`t2_democount.md`).
- **Magnitude bite zone:** T2.1−T2.3 gap stays +53 to +73 pp across mass∈[1.1,1.5] (`t2_magnitude.md`).
- **Pareto dominance:** T2.1 strictly Pareto-dominates T2.3 in proxy (bias², variance) (`pareto_analysis.md`).

## Negative / null results (load-bearing for the method story)

- **Combination rules:** product / gate / tempered (finite-p), adaptive `σ_demo`, λ-annealing, and demo-phase pseudo-cost **underperform additive score-bridging**; product/gate over-optimize.
- **tool_hang / transport:** unsolved (~0) across raw sparse, dense shaping, larger H, σ sweeps, and demo-phase cost. Static guidance tuning is not the path.
- **T2 → Square transfer:** NULL/blocked (`t2_square.md`); superseded by Square-DP (claim 5).
- **Cross-task retrieval (heterogeneous bank):** failed without task-family locality.

## Engineering

- **MJX Gate-4:** implemented, GPU-measured **13.8×** (Colab L4) — below the 20-30× promotion bar. Appendix only (`results_writeups/mjx_gate4_closeout.md`).

## Remote offload (2026-06-11, `HRTG-square-dp-followup`)

Completed and cleaned up: DP production **140/140** (first grid was nw=1-corrupt, now
recomputed & valid — see below); static BC K/R grid 9/9 + split λ sweep (canonical race2
5/5); friction-adaptation grid 18/18; transport ablation composite-completed. Details:
`docs/results_writeups/vast_remote_offload_2026_06_11.md` (in the followup repo).

**✅ Corrected & DEEPENED DP λ-grid (`vast_safe_dp_nwfix_2026_06_12`, nw≥4, N→60-90/cell, 2026-06-13):**

The `n_workers=1` corruption was fixed (nw≥2) and the grid deepened to **530 summaries**
(flagship cells now 10-15 seeds = **60-90 rollouts/cell**, Wilson 95% CIs, de-duplicated; pulled local).
This **supersedes BOTH** the INVALID `vast_safe_dp_2026_06_10` (nw=1) **and the N=30 snapshot
previously cited here**. **Do NOT cite the old N=30 Square "+43 pp CI-separated" number — it
was a small-sample / seed-lucky artifact that regressed under more seeds (see below).**

- **Over-anchoring (inverted-U in λ) — REAL but MODEST (~+15 pp), consistent across 3 curves:**
  - Can DP `ep0199` σ0.10: λ0 **51.1%** [41,61] (46/90) → broad peak ~**66.7%** [56,76] (60/90, λ0.75) → plateau 62-67% (λ≥0.5). +15.6 pp.
  - Can σ0.25: λ0 **51.7%** [39,64] → λ1 65.0% → peak λ10 **68.3%** [56,79]. +16.6 pp.
  - Square `ep0199`: λ0 **54.4%** [44,64] (49/90) → peak λ1 **68.9%** [59,78] (62/90) → λ1.5 67.8% → λ2 62.2%. +14.4 pp; clean inverted-U.
  - **No single flagship cell is CI-separated at N=90** (independent Wilson), but that ignores the matched-seed design. The **matched-seed POOLED within-seed lift λ0→λ1 across the 3 curves IS significant: +12.9 pp, 95% paired-bootstrap CI [+4.6,+21.2], sign-test p=0.008 (n=40 seed-blocks)** — see `paired_significance_2026_06_14.py`. (The N=30 Square effect collapsed: λ0 43→54%, λ1 87→69% as block-2 seeds landed; do NOT cite the old per-cell separation.)
- **Cross-prior ladder (Square, λ∈{0,0.5,1,2}, N=60-90):** guidance gives a modest lift at *all* prior qualities (peak vs λ0): ep0049 38.3%→50.0% (+12 pp); ep0099 48.3%→56.7% (+8 pp); ep0149 58.3%→63.3% (+5 pp); ep0199 54.4%→68.9% (+15 pp). The earlier **"middling checkpoints are hurt" finding did NOT survive densification** (ep0099 now mildly *helped*). Lift is largest for the strongest checkpoint; no checkpoint is significantly hurt at its optimal λ.
- **Robustness sub-study (dynamics shift, ep0199, N=30/cell; phase4 deepening to N=60-90 in flight):** test-time guidance (λ1 vs λ0) **buys robustness to MODERATE shift**:
  - Can: nominal +11.1 pp; mass×1.5 **+20 pp** (63→83%); friction×2.0 **+20 pp** (57→77%); friction×0.5 +10 pp.
  - Square: nominal +14.4 pp; friction×0.5 **+23.3 pp** (57→80%); mass×1.5 / friction×2.0 +3.3 pp.
  - **Hard failure mode:** mass×2.5 collapses to ~0% regardless of λ — guidance cannot rescue far-OOD dynamics. (N=30 deltas not yet CI-separated; phase4 is boosting the +20 pp shifts toward separation + mapping the mass1.5→2.5 breakdown knee.)

Full grid: `HRTG-square-dp-followup/experiments/15_unified_hybrid/results/nwfix_calibration_analysis_2026_06_13.txt` + `analyze_nwfix_grid_2026_06_13.py`. Checkpoints (Can-DP + 4 Square) + 530 (de-duplicated) summaries now local.

## SOTA reference (robomimic low-dim, ph split; Mandlekar 2021 Table 4)

Lift 99.3% · Can 92.3% · Square 74.7% · Transport 65.3% · tool_hang 17.3% (BC-RNN-GMM).
Treat DP 94.7% / Diffuser ~80-85% as **approximate** until verified per
`references/sota_reference_numbers.md`.

## Caveats

- n=30 cells unless pooled; report Wilson CIs.
- Keep `ep0149` (hybrid-lift) and `ep0199` (standalone) roles distinct.
- Label diagnostic (e.g. Square H=300 λ=2 = 50%) vs headline (Square-DP 70%).
