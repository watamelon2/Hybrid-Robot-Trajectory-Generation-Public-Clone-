# v_theta v3 — Karras-EDM noise-conditional score prior

**Status**: code complete, smoke test green; not yet trained on the
full ph slice.  This document is the design + run-book for the
training and evaluation, written so anyone can pick it up cold.

## Why v3 (motivation in one paragraph)

v2 (the FiLM x_1-parameterised flow) was diagnosed in
`results/sweep_v2_2026_05_16/MORNING_BRIEF.md` as having three
**structural** failure modes that no λ / σ / bridge-formula tuning
could fix:

1. **σ-preservation mismatch.**  v2 trained on the OT-CFM
   linear-interpolant `x_t = (1-t) x_0 + t x_1` with `x_0 ~ N(0,I)`.
   MBD iterates at `μ + σε` on the variance-exploding forward
   `p^data * N(0, σ²I)` — a *different* distribution.  v2 only
   worked at σ ≈ 0.02 (16.7 % single-seed) and collapsed at the
   canonical σ = 0.25 (0/60).
2. **t=0-only training distribution.**  v2's dataset was one
   example per demo (`s_0`, full demo).  Mid-trajectory states are
   OOD → MPC replanning regressed flow from 6.7 % R=0 to 0.0 % R=1.
3. **Wrong score object.**  Phase 7 verified the "quadratic" bridge
   `(t·x1_hat - x)/(1-t)²` does *not* recover the prior gradient —
   the trained model's velocity field is on the wrong marginal at
   intermediate σ, full stop.

v3 fixes all three by structure:

1. **Karras EDM training**: noise-conditional score on
   `x_0 + σε` with σ ∼ log-Uniform[σ_min, σ_max].  Inference σ
   range is **baked into the checkpoint** so the score is
   well-calibrated at every MBD iterate.
2. **Per-chunk dataset**: one example per timestep
   `(s_t, a_{t:t+H})`, edge-padded at the demo tail.  Matches BC's
   per-state training distribution that already survives MPC
   replanning in the unified driver.
3. **Direct score**: model output *is* the denoiser
   `D(x; σ, s)`; the score is `(D - x) / σ²` by Tweedie.  No
   bridge formula.

## File layout

| File                | Role                                                          |
|---------------------|---------------------------------------------------------------|
| `model_v3.py`       | `FilmEdmField` + Karras preconditioning + `edm_loss/score/sample` |
| `dataset_v3.py`     | Per-chunk `(s_t, a_{t:t+H})` extractor (`load_chunks`)        |
| `train_v3.py`       | CLI trainer (CPU/GPU/AMP) → `v_theta_v3.pt`                   |
| `adapter_v3.py`     | `FlowPriorV3` wrapper (direct score, σ-schedule-aware)        |
| `adapter.py`        | Auto-dispatches v1/v2/v3 by checkpoint `arch` field           |
| `smoke_v3.py`       | End-to-end wiring test (no HDF5 dependency, ~5 s on CPU)      |
| `README_v3.md`      | This file                                                     |

## Math (one screen, no proofs)

### Forward
$$
x_\sigma \;=\; x_0 + \sigma\,\varepsilon,\qquad
\varepsilon \sim \mathcal{N}(0, I),\quad
\sigma \sim \text{log-Uniform}[\sigma_{\min}, \sigma_{\max}].
$$

### Karras preconditioning (`edm_scalings` in `model_v3.py`)

$$
c_{\rm in}(\sigma) = \frac{1}{\sqrt{\sigma^2 + \sigma_{\rm data}^2}}, \quad
c_{\rm out}(\sigma) = \frac{\sigma\,\sigma_{\rm data}}{\sqrt{\sigma^2 + \sigma_{\rm data}^2}}, \quad
c_{\rm skip}(\sigma) = \frac{\sigma_{\rm data}^2}{\sigma^2 + \sigma_{\rm data}^2}, \quad
c_{\rm noise}(\sigma) = \tfrac14 \log \sigma.
$$

### Denoiser
$$
D(x;\sigma, s) \;=\; c_{\rm skip}(\sigma)\, x \;+\; c_{\rm out}(\sigma)\,F_\theta\!\left(c_{\rm in}(\sigma)\,x,\,c_{\rm noise}(\sigma),\,s\right).
$$

### Loss
$$
\mathcal{L}(\theta) \;=\; \mathbb{E}_{x_0, s, \sigma, \varepsilon}\!\left[\,\lambda(\sigma)\,\|D(x_0+\sigma\varepsilon;\sigma,s) - x_0\|^2\,\right], \qquad
\lambda(\sigma) = \frac{\sigma^2 + \sigma_{\rm data}^2}{(\sigma\,\sigma_{\rm data})^2}.
$$

### Score (used by MBD)
$$
s_\theta(x;\sigma, s) \;=\; \frac{D(x;\sigma, s) - x}{\sigma^2}
\;\overset{\text{Tweedie}}{=}\; -\frac{\varepsilon_\theta(x;\sigma, s)}{\sigma}.
$$

No bridge formula — the score is a one-line Tweedie inversion of
the denoiser.  Plugged into `mbd.denoise_loop`'s composite update
`μ ← μ + σ² · s_θ(μ; σ, s_0)`, this is `μ ← D(μ; σ, s_0)` exactly.

## How to train

### GPU (Colab L4 or equivalent, ~30 min on the full ph slice)

```bash
python experiments/08_flow_matching_prior/train_v3.py \
    --task can --n-demos 200 --horizon 120 \
    --epochs 200 --batch-size 256 --lr 3e-4 \
    --hidden 768 --n-blocks 4 \
    --sigma-min 0.05 --sigma-max 0.5 \
    --sigma-data 0.5 \
    --device cuda --amp \
    --output experiments/08_flow_matching_prior/results/v_theta_v3.pt
```

Why these defaults:

- `--sigma-min 0.05 --sigma-max 0.5` matches MBD's
  `cosine_sigma_schedule(K, sigma_max=0.5, sigma_min=0.05)` used at
  the canonical operating point in `experiments/15_unified_hybrid`.
  **Do not narrow this range below MBD's actual schedule** — any
  inference σ outside `[sigma_min, sigma_max]` will produce
  miscalibrated scores.  If you want to test σ ≥ 0.5, retrain.
- `--sigma-data 0.5` matches the action range [-1, 1] (max
  reasonable per-step std is ~0.5).  Karras's recipe: pick
  `sigma_data` close to the empirical std of the data.
- Per-chunk dataset balloon: 200 demos × ~115 steps/demo
  ≈ 23 000 chunks (vs v2's 200 chunks).  Hence the larger batch
  size (256) and fewer epochs (200) than v2's (128 / 4000) —
  total gradient updates are comparable.

### Quick wiring test (CPU, ~5 s)

```bash
python experiments/08_flow_matching_prior/smoke_v3.py
```

Builds a tiny `FilmEdmField`, runs `edm_loss` + `edm_sample` +
`make_prior_score_fn`, verifies Tweedie identity, round-trips
through `FlowPriorV3.from_checkpoint`, and confirms `adapter.py`
auto-dispatches `arch=karras_edm_v3` to `FlowPriorV3`.  No HDF5
needed.

### Train smoke (CPU, ~1 s, real data)

```bash
python experiments/08_flow_matching_prior/train_v3.py --smoke
```

3 demos × 4 epochs × tiny model.  Verifies dataset loading + loss
descent + checkpoint write.

## How to evaluate

The trained `v_theta_v3.pt` is consumed by
`experiments/15_unified_hybrid/run_can.py` via the existing
`--prior=flow` flag — `adapter.FlowPrior.from_checkpoint`
auto-detects the `arch=karras_edm_v3` field and returns a
`FlowPriorV3`.  No new CLI flag is needed.

### Recommended first eval (mirrors `slate_d` baseline cell on Can)

```bash
python experiments/15_unified_hybrid/run_can.py \
    --task can \
    --prior flow \
    --prior-checkpoint experiments/08_flow_matching_prior/results/v_theta_v3.pt \
    --warm-start \
    --lambda-prior 0 1 5 \
    --mpc-replans 0 1 \
    --K 10 --M 16 --H 120 \
    --sigma-max 0.25 --sigma-min 0.05 \
    --lambda_ 0.5 \
    --n-demos 20 --n-rollouts 3 \
    --seed 0 \
    --n-workers 10 \
    --output-dir experiments/08_flow_matching_prior/results/v3_eval_first \
    --no-live-view
```

**Important σ caveat**: the eval command's `--sigma-max 0.25` is
**inside** v3's training range `[0.05, 0.5]`, so the score is
well-calibrated.  If you change `--sigma-max` above 0.5, retrain
v3 with the matching range.

### Three pre-registered hypotheses

1. **σ-mismatch is fixed.**  At σ_max = 0.25 (where v2 was 0/60),
   v3 should produce non-zero hybrid success.  If v3 also gives
   0/60 at σ_max=0.25, the σ-preservation diagnosis was incomplete
   and we need to look at the warm-start quality directly.
2. **Replanning works.**  At λ=0 R=1 K=10 σ=0.25, v3 should
   match or exceed v2's σ=0.02 R=0 cell (16.7 %), and **not regress
   under replanning**.  If R=1 < R=0 for v3, the per-chunk
   dataset alone wasn't enough; the per-state action distribution
   in `dataset_v3._edge_pad_actions` might need refinement.
3. **Score-bridge contributes.**  At λ ∈ {1, 5} R=0, v3 should
   beat v3 at λ=0 R=0 (the warm-start-only ablation).  If λ>0
   regresses, the Karras score might be over-aggressive at the
   high-σ tail; consider `--sigma-data` retuning or σ-tail
   clipping.

## Comparison surface

| Prior                 | What it learns               | σ range | Per-state? | Score-bridge math |
|-----------------------|------------------------------|---------|------------|-------------------|
| `demo_1nn` (headline) | nothing — direct retrieval   | n/a     | yes        | quadratic pull    |
| BC (exp 03)           | `a = π(s)` MLP               | n/a     | yes        | quadratic pull    |
| v1 / v2 flow          | OT-CFM `v_θ(x_t, t, s)`      | mismatch | per-demo  | wrong (Phase 7)   |
| **v3 (this)**         | Karras score `s_θ(x, σ, s)`  | matched | **per-state** | direct Tweedie |

The right way to read v3's pitch: it's the *parametric* counterpart
of `demo_1nn` that doesn't suffer from the structural σ-mismatch
of v1/v2, and unlike BC it has a principled score-bridge instead
of an ad-hoc quadratic pull.

## Open risks (be honest)

1. **Per-chunk dataset is BC-like.**  At every timestep we tell the
   model what the next H actions are.  This is a strictly *more*
   informative training signal than v2's t=0-only setup, but it's
   also similar to BC and might not unlock anything `demo_1nn`
   already gets for free.  Empirically: if v3 beats `demo_1nn`'s
   73.8 % pooled on Can, we have a parametric advantage; if not,
   the result is "matched performance with smaller checkpoint and
   parametric extrapolation off the bank."  Either is publishable.
2. **σ_data tuning is empirical.**  Karras's `σ_data` is picked to
   match the data's empirical std.  Action range [-1, 1] gives
   ~0.5 as a defensible default, but if z-score normalisation is
   later added to the action dim (it isn't currently), `σ_data` has
   to be retuned to 1.0.
3. **Heun sampler at sample_initial_mean.**  Karras's Heun integrator
   is the right inference-time sampler for a noise-conditional
   score model, but it's a *new* code path vs v2's Euler.  The
   smoke test verifies finiteness + determinism but not
   *trajectory quality*.  Anomalies in v3's warm-start (e.g.,
   norms way off the demo distribution) should be diagnosed first
   via `replay_prior.py` before reading too much into the
   end-to-end success rates.

## Provenance

- Diagnosis source: `results/sweep_v2_2026_05_16/MORNING_BRIEF.md`
  (the 2026-05-16 sweep + 2026-05-16 PM follow-ups), specifically
  the "Three findings the data forced" section and the warm-start
  ablation table at the bottom.
- Architecture inspiration: Karras et al. 2022 NeurIPS
  ("Elucidating the design space of diffusion-based generative
  models"), Table 1 EDM column.
- Smoke artefacts: `smoke_v3.py` output verifies 8 invariants
  including the Tweedie identity to 0 floating-point error.
