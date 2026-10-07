# Archived engineering plans

These docs are **historical** — they were the engineering plans
that produced the experimental results currently on `main`. They
are kept for traceability (so the paper / supplement can cite
which plan drove which experiment) but are no longer maintained.

For the *current* forward-looking roadmap, see
[`../plans.md`](../plans.md).

For the *empirical results* that these plans produced, see
[`../results.md`](../results.md).

## Index

| File                            | Drafted     | Status as archived             | Produced (results currently on `main`)                                                                       |
|---------------------------------|-------------|--------------------------------|---------------------------------------------------------------------------------------------------------------|
| `post_consolidation_plan.md`    | 2026-05-14  | Fully executed                 | Top-level "branches A / B / C" plan. Spawned the three feature branches that landed May 15.                  |
| `plan_hybrid_coupling.md`       | 2026-05-14  | Phase 1-2 done; 3-5 deferred   | Branch C's Phase 1 (unified driver) + Phase 2 (K×R grid). Phases 3-5 (adaptive λ, truncated chain, replan trigger) carried forward into `plans.md`. |
| `plan_stronger_mbd_signal.md`   | 2026-05-14  | Task A1 done; A2-A4 deferred   | Branch A's σ-max sweep + the M-ablation (sweet-spot inverted-U). Tasks A2 (V_φ baseline), A3 (particle ablation extension), A4 (reward shaping) carried forward into `plans.md`. |
| `plan_overnight_queue.md`       | 2026-05-15  | Fully executed                 | The Stage-0-through-Stage-6 overnight queue that produced the Phase 2 K×R grid, Job A/B/C, the M-ablation, and the in-flight closeout log of the May 15 section in `results.md`. |
| `charles_tuning_analysis.md`    | 2026-05-12  | Superseded by BC-swap re-run   | The May 12 analysis of Charles's `Charles-test` runs that motivated the BC swap from `[512, 512]` to `[256, 256]`. The actual BC-swap re-run analysis (n=60) is in [`../bc_swap_rerun_analysis.md`](../bc_swap_rerun_analysis.md). |

## What's *not* archived (still active)

| File                        | Why kept                                                                                                |
|-----------------------------|---------------------------------------------------------------------------------------------------------|
| `../plans.md`               | Forward-looking master roadmap. Updated at each milestone.                                              |
| `../mjx_migration.md`       | Tier 4 plan (MJX migration) — still in-flight on `feat/flow-mjx`. Living document.                       |
| `../bc_swap_rerun_analysis.md` | The n=60 BC-swap re-run analysis still referenced from `results.md` § BC-swap re-run.                |
| `../results.md`             | Canonical empirical results.                                                                            |
| `../run_log.md`             | Chronological run log. Append-only.                                                                     |
| `../MuJoCo.md`, `../remote_run.md` | Operational reference docs.                                                                       |
