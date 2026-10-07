# Experiment 08 — Flow-matching prior `v_θ`

**Lifecycle:** **WIP at the abstract "beat retrieval with a single
learned prior" goal; verified at a narrower one** -- the v3-family
DP UNet1D checkpoints `v_theta_dp_tobs2_h16_v3b_ep*.pt` (trained
for exp 17's Phase-B / Phase-C / Phase-D research arc on
`feat/dp-phase-c`, merged into main 2026-06-01 as `8e97ca3`)
hit **70-80 % standalone** on Can / `ph` and **86.7 % under
MBD hybrid** at the (ep0149, K = 10, λ = 2.0) cell -- comparable
to, and in the hybrid cell exceeding, the retrieval 73.8 %
headline.  See [`17_dp_hybrid.md`](17_dp_hybrid.md) for the
full Phase-D 2-D grid.  Earlier learned-prior attempts (v1, v2,
the parallel v3 Karras-EDM work on `feat/flow-v3-karras`) remain
recorded as negative or partial results below.

**In-tree directory:**
[`experiments/08_flow_matching_prior/`](../../experiments/08_flow_matching_prior/).
*(No README on main; this doc is the entry point.)*

## Purpose

Train a learnable score / flow-matching prior `v_θ(x, t)` over
demonstration trajectories, then plug it into the
[`07_lambda_sweep`](07_lambda_sweep.md) /
[`15_unified_hybrid`](15_unified_hybrid.md) score-bridge as
`--prior=flow`. The end goal is to **beat the retrieval-prior 73.8 %
headline with a trained prior** so the project has a learned story
in addition to a retrieval-based one.

That goal is **not** achieved yet. The retrieval prior (exp 15 with
`--prior=demo_1nn`) is far ahead of any learned variant tried so far.

## Phase history

### Phase 1 — v1: bare CFM regression (May 13–15, abandoned)

- 3-layer MLP, conditional flow matching on `(x, t)` → velocity.
- Sample magnitudes ~1000× the action bound; required two retraining
  passes (Fix A, Fix B).
- **Result**: 0 / 60 success.
- Stored: `experiments/08_flow_matching_prior/results/v1_*` (archive).

### Phase 2 — v2: FiLM + x1-parameterisation (May 15–17)

- FiLM time-conditioning, predict `x_1` instead of velocity,
  Colab notebook for training.
- Phase-0 prior-only (no MBD) = 8.3 %.
- Phase-5 σ-preservation sweep best cell: σ=0.02 λ=0 K=10 M=16
  single-seed **10 / 60 = 16.7 %**.
- Phase 9 replanning ablation: R=0 = 4 / 60 (6.7 %), R=1 = 0 / 60
  ("MPC replanning hurts" for this prior; see
  [`docs/results.md`](../results.md)).
- **Verdict**: better than v1 but still far below retrieval.
- Stored: `experiments/08_flow_matching_prior/results/sweep_v2_2026_05_16/`,
  `results/v2_followup_2026_05_16/`.

### Phase 3a — v3 Karras-EDM (parallel branch, May 22-25, NOT on main)

*Earlier v3 attempts.  v3 was the Karras-EDM noise-conditional
rewrite; it lived on `feat/flow-v3-karras` which never merged.
Key learnings (capacity- vs schedule-bottleneck diagnostics) fed
into the v3b training recipe that subsequently landed on
`feat/dp-phase-c`.*

Lives on branch `feat/flow-v3-karras` (NOT on main):

- Karras-EDM noise-conditional score parameterisation.
- DP UNet1D + DDPM prior with `T_obs` stacking + closed-loop adapter.
- EMA training (DP paper recipe).
- DDIM sampler stride fix; asymmetric MBD evaluation.
- Slate K (May 25) `--score-mode {learned, pull}` diagnostic:
  **parity is the absolute ceiling** for the current learned
  priors under MBD's softmax weighting. Strong priors get destroyed
  by MBD unless `score-mode=pull` is used.

The Phase 3 result on main is the Slate-K diagnostic finding, not
a new headline number. The v3 work IS NOT merged; see
`feat/flow-v3-karras`.

### Phase 4 — v3b: DP UNet1D family on `feat/dp-phase-c` (May 26 - June 1, MERGED 2026-06-01)

The v3b training recipe (DP UNet1D with the Karras-EDM schedule
lessons baked in) shipped on `feat/dp-phase-c` and was evaluated
end-to-end via exp 17's Phase-B (standalone) + Phase-C / Phase-D
(hybrid + K/λ sweeps).  Checkpoint files:
`experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep{0049,0099,0149,0199,0249,0299,0349}.pt`.

**Standalone (Phase B)** on Can / `ph`:

  - ep0199 (50 fresh seeds, GCP T4): **40 / 50 = 80.0 %**.
  - ep0199 re-eval on this Mac (10 seeds): 7 / 10 = 70.0 %.

**Hybrid (Phase D)** on Can / `ph` (n=30/cell unless noted):

  - Q1 epoch curve (K = 10 λ = 0.5, n = 60 / ckpt):
    ep0049 / ep0099 / ep0149 / ep0199 / ep0249 / ep0299 / ep0349
    = 55 / 70 / **73** / 72 / 70 / 67 / 70 %.
  - **Two GREEN cells** at the Q1 peak ep0149:
    - (ep0149, K = 10, λ = **2.0**) = **26 / 30 = 86.7 %** [70.3, 92.7], **+13.4 pp** GREEN.
    - (ep0149, K = 20, λ = 0.5)   = 25 / 30 = 83.3 % [66.4, 92.7], +10.0 pp GREEN.

The v3b family is the **first learned prior** in this project
that is competitive with (or exceeds, in the hybrid) the
retrieval 73.8 % pooled headline.  See
[`experiments/17_dp_hybrid/results/phase_d_2026_05_30/CONCLUSION.md`](../../experiments/17_dp_hybrid/results/phase_d_2026_05_30/CONCLUSION.md)
for the mechanism reading: MBD's value emerges at the moderately-
trained prior, along either the bridge-weight knob (λ=2.0) or
the refinement-budget knob (K=20), with the cheaper λ-knob path
delivering the larger lift.

K=40 at ep0049 is deferred (paused via SIGSTOP at 2 / 25 rollouts
on 2026-06-01 01:31 PDT; will resume on a follow-up branch).

## Current best (on main)

| Variant | Success | Notes |
|---------|---------|-------|
| v1 + MBD | **0 / 60** | sample magnitude bug; archived |
| v2 + MBD, σ=0.02 K=10 M=16 (seed 0) | **10 / 60 = 16.7 %** | Phase-5 sweep peak |
| v2 + MBD, R=0 seeded | 4 / 60 = 6.7 % | Phase-9 ablation |
| v2 + MBD, R=1 seeded | **0 / 60** | replanning hurts |
| **v3b ep0199 standalone** (GCP T4, 50 seeds, DP apples-to-apples) | **40 / 50 = 80.0 %** | Phase B; v3b is the canonical learned prior on main as of 2026-06-01 |
| **v3b ep0149 + MBD, K=10 λ=2.0** | **26 / 30 = 86.7 %** | Phase D Phase-F GREEN cell; cheapest hybrid path |
| **v3b ep0149 + MBD, K=20 λ=0.5** | 25 / 30 = 83.3 % | Phase D Phase-E GREEN cell |
| v3b ep0149 + MBD, K=10 λ=0.5 | 44 / 60 = 73.3 % | Phase D Q1 baseline at the epoch-curve peak |
| **Square-DP standalone** (`square_dp`, ep0199) | **20 / 30 = 66.7 %** | Task-specific Square DP prior, 23-D compact schema |
| **Square-DP + MBD, K=10 λ=0.5/1.0** | **21 / 30 = 70.0 %** | Square hybrid peak; moderate pull bridge |

For the comparison numbers vs other priors, see
[`docs/results.md`](../results.md) row "v_theta v2 + MBD".

The Square-DP result resolves the earlier Square learned-prior failure
mode as a representation/schema problem rather than a generic DP-prior
or score-bridge limitation.  Details:
[`docs/results_writeups/square_dp_prior.md`](../results_writeups/square_dp_prior.md).

## Reproduce

```bash
conda activate hybrid-traj

# Train v2 (canonical) on CPU; ~15 min.
python experiments/08_flow_matching_prior/train_v2.py \
    --task can --n-demos 20 --horizon 120 \
    --epochs 50 --batch 256 \
    --output experiments/08_flow_matching_prior/results/v_theta_v2.pt

# Or use the Colab notebook (L4 GPU; ~3 min).
# See experiments/08_flow_matching_prior/colab_train_v2.ipynb.

# Plug into exp 15.
python experiments/15_unified_hybrid/run_can.py \
    --prior flow \
    --prior-checkpoint experiments/08_flow_matching_prior/results/v_theta_v2.pt \
    --lambda-prior 0 --K 10 --M 16 --sigma-max 0.02 \
    --n-demos 20 --n-rollouts 3 --seed 0

# Standalone prior-only ascent (phase-0 sanity check).
python experiments/08_flow_matching_prior/replay_prior.py \
    --checkpoint experiments/08_flow_matching_prior/results/v_theta_v2.pt \
    --n-demos 20 --n-rollouts 3 --seed 0
```

## Files

| Path | Role |
|------|------|
| [`experiments/08_flow_matching_prior/train.py`](../../experiments/08_flow_matching_prior/train.py) | v1 trainer (archived). |
| [`experiments/08_flow_matching_prior/train_v2.py`](../../experiments/08_flow_matching_prior/train_v2.py) | v2 trainer (FiLM + x1-param). |
| [`experiments/08_flow_matching_prior/model.py`](../../experiments/08_flow_matching_prior/model.py) | v1 MLP. |
| [`experiments/08_flow_matching_prior/model_v2.py`](../../experiments/08_flow_matching_prior/model_v2.py) | v2 model (FiLM-conditioned). |
| [`experiments/08_flow_matching_prior/adapter.py`](../../experiments/08_flow_matching_prior/adapter.py) | Wraps a checkpoint as a score callable for the unified driver. |
| [`experiments/08_flow_matching_prior/adapter_v2.py`](../../experiments/08_flow_matching_prior/adapter_v2.py) | v2 adapter (x1 → velocity). |
| [`experiments/08_flow_matching_prior/replay_prior.py`](../../experiments/08_flow_matching_prior/replay_prior.py) | Prior-only ascent (no MBD weighting). |
| [`experiments/08_flow_matching_prior/lq_sanity.py`](../../experiments/08_flow_matching_prior/lq_sanity.py) | 2-D LQ regression sanity for `v_θ`. |
| [`experiments/08_flow_matching_prior/sweep_v2.sh`](../../experiments/08_flow_matching_prior/sweep_v2.sh) | Phase 0-9 launcher (the bash orchestrator referenced in run_log). |
| [`experiments/08_flow_matching_prior/colab_train_v2.ipynb`](../../experiments/08_flow_matching_prior/colab_train_v2.ipynb) | Train v2 on Colab L4. |
| `experiments/08_flow_matching_prior/results/v_theta_v2.pt` | v2 checkpoint (gitignored / cached). |
| `experiments/08_flow_matching_prior/results/sweep_v2_2026_05_16/` | Phase-0/3/5/7/9 result JSONs. |

## Cross-references

- **Consumer**: [`15_unified_hybrid.md`](15_unified_hybrid.md)
  with `--prior=flow --prior-checkpoint .../v_theta_v2.pt`.
- **Replaced by (for the headline)**:
  [`15_unified_hybrid.md`](15_unified_hybrid.md) with
  `--prior=demo_1nn` (retrieval prior — no training required).
- **Plan**:
  [`docs/plan_principled_score.md`](../plan_principled_score.md)
  is the engineering plan for B1–B8 phases.
- **MJX migration**:
  [`docs/mjx_migration.md`](../mjx_migration.md) — once an L4 GPU is
  available, v_θ training and MBD evaluation can move there.

## Why this didn't beat retrieval

Documented in [`docs/results.md`](../results.md) § presentation
brief paragraph "Why retrieval beats trained":

1. **Mode-coverage trade-off**: MSE-style training on diverse demos
   averages across action modes; a 1-NN retrieval prior preserves
   each demo's mode exactly.
2. **MBD's softmax weighting**: the `--score-mode=learned` MBD
   update destroys strong learned priors at parity — see Slate K
   on `feat/flow-v3-karras`.
3. **Data scale**: 200 demos × 120 steps = 24 k transitions is
   marginal for the v_θ MLP/UNet capacity used; retrieval needs no
   density estimate.

A learned `v_θ` that beats retrieval is an open research question,
not just an engineering tuning question.

## WIP work (on `feat/flow-v3-karras`, NOT on main)

- v3 Karras-EDM noise-conditional score
- DP UNet1D + DDPM prior (May-22 onward)
- Slate K parity-ceiling diagnostic
- EMA + asymmetric-MBD experiments

If you want to revisit, check out `feat/flow-v3-karras` for the
latest state. Merge to main is gated on beating retrieval (or
producing a publishable negative result with a tighter mechanism
story).
