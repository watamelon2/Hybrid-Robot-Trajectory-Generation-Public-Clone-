# Phase 1 — Frontier scan (2026-06-11 refresh)

**Topic.** Which single next experiment best finishes the Hybrid Robot Trajectory
Generation (HRTG) final-report story, now that the remote offload is complete?

**Why a refresh.** A complete deep-research pass exists at
`research_directions_2026_06_10/`. This refresh re-runs the frontier with live
arXiv + Semantic Scholar searches focused on **2025-2026** work and on the exact
decision question, and folds in the now-finished remote experiments (DP 140/140,
static lambda sweeps, friction-adaptation grid, transport composite).

**Repo grounding (current, post-offload).**

1. Zero-training `demo_1nn` retrieval + additive score-bridge MBD is the Can headline at **177/240 = 73.8%**.
2. Bank-size scaling rises from 23.3% (N_b=5) to ~65% (N_b=20) and plateaus at 68-70% through N_b=200 (finite mode coverage).
3. Square needed a **task-specific DP representation**: 66.7% standalone, 70.0% hybrid at λ=0.5/1.0.
4. Trained-prior lift is regime-dependent: DP `ep0149` is the hybrid-lift operating point (86.7% at K=10,λ=2.0); `ep0199` is the stronger standalone reference (76-80%).
5. Long-horizon sparse tasks (`tool_hang`, `transport`) are **not** solved by static product/gate/temper/sigma/anneal tuning (transport sparse-extra cells are 0/5).
6. Adaptation: cheap demo-bank augmentation can beat BC retraining in the Can mass/friction-shift bite zone.

**Search corpus.** 14 frontier queries × arXiv (25 each) + Semantic Scholar
retries wrote raw JSONL under `phase1_frontier/search_results/`. After
dedup/relevance ranking (`/tmp/dr_rank_2026_06_11.py`), the merged raw database
holds **167 unique records**; `paper_db.jsonl` keeps the **70** highest-scoring
project-relevant records.

## The single dominant 2025-2026 trend

**Test-time guidance / steering / adaptation of *pretrained* generative policies
is now its own active subfield** — and HRTG's MBD score-bridge is a member of it.

| Frontier cluster | Representative 2025-2026 papers | Why it matters for HRTG |
|---|---|---|
| Steering off-the-shelf diffusion policies with an external signal during denoising | DynaGuide [@bagaria2025dynaguide], GRITS [@grits2025], Projected Coupled Diffusion [@pcd2025], Fisher-Preserving Guidance [@fisher2026] | This is *exactly* HRTG's `lambda_prior` cost-guided denoising. HRTG should be framed as test-time cost guidance of a prior, not "hybrid beats BC". |
| Test-time adaptation of a pretrained DP without retraining | ADPro [@adpro2025], Neural-Dynamics-Augmented DP [@ndadp2025], DA-SIP [@dasip2025] | Confirms the test-time-correction framing and the few-shot/coverage angle; ADPro adds manifold + init constraints (an alternative to additive bridging). |
| **When/whether test-time guidance is well-behaved** (calibration, over-optimization) | Calibrated Test-Time Guidance [@calibrated2026], Test-time Alignment w/o Reward Over-optimization (ICLR'25) [@das2025], Guidance↔test-time-scaling anthology [@anthology2025], PGG [@pgg2025] | **Direct theory hook for HRTG's negative results.** Product/gate/temper failures are reward-over-optimization / miscalibrated-guidance phenomena. No one has a practical *predictor* of when to guide a manipulation prior. |
| Retrieval / nonparametric priors for imitation | DARP (Difference-Aware Retrieval Policies) [@darp2026], Robot Data Curation w/ kNN-MI [@curation2025], latent-NN-guided sparse control [@nnsparse2023] | DARP is the closest neighbor to `demo_1nn`, but it *trains* a kNN-conditioned model; HRTG uses retrieval as a *zero-training score prior inside MBD*. Coverage/quality diagnostics (kNN-MI) support the bank-coverage direction. |
| Hierarchy / contact guidance for contact-rich, long-horizon tasks | Hierarchical Diffusion Policy via Contact Guidance (T-RO) [@hdp2025], Reactive Diffusion Policy [@xue2025reactive], TacDiffusion [@tacdiffusion2024], DP+Residual-RL [@dprrl2025] | Supports the phase/contact direction for `tool_hang` and Square residuals: high-level contact/subgoal + low-level guided action; richer contact observability matters. |
| Rigorous, sample-efficient policy evaluation | Beyond Binary Success (SAVI anytime-valid) [@savi2026] | Directly relevant to HRTG's n=30 cells and Wilson CIs; a cheap way to add statistical rigor to the final report. |

## Frontier implication for the decision

The literature has moved decisively toward **test-time guidance of pretrained
policies**, and a 2026 theory thread now says these guidance schemes are often
**miscalibrated and prone to reward over-optimization**. HRTG already holds the
matched-seed evidence for this in manipulation: additive score-bridging helps in
specific regimes while product/gate/temper *over-optimize and fail*.

> **Sharpened thesis.** Test-time cost guidance helps a robot prior only when the
> prior supplies coherent, phase-correct geometry with residual headroom; the
> same guidance *over-optimizes* when the combination rule or representation is
> wrong. HRTG can contribute a **practical calibration/diagnostic** for *when and
> how strongly* to guide — the missing applied counterpart to the 2026 calibration
> theory.

This reframes HRTG's large bank of negative results from "things that didn't work"
into "evidence for a calibration boundary", which is the cheapest, lowest-risk,
highest-coherence way to finish the report.
