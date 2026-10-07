# Theory: what is the score-bridge actually doing? *(post-Slate-E)*

After Slate E (May 19) we have three pooled empirical facts that
any theory has to explain:

1. **The lambda-sweep is unimodal**, with a peak strictly above
   either endpoint (lambda=0 = warm-start only vs lambda=infty =
   bridge-dominated).  `demo_1nn` peaks at lambda=1 = 73.8 %
   pooled CI [67.8, 78.9]; warm-start alone gives 11.7 %; bridge
   plateau gives 64.2 % pool.  See `docs/results.md` § Slate E.

2. **The peak location is prior-specific.**  BC peaks at lambda=5
   = 41.7 % (single seed), `demo_1nn` peaks at lambda=1 = 75.0 %.
   Both priors agree exactly at lambda=0 = 11.7 % (warm-start
   provenance alone).

3. **The sigma-sensitivity profile is lambda-specific.**  At
   lambda=5 the cell is flat in sigma across [0.05, 1.00]
   (63 - 68 %).  At lambda=1 it is peaked at sigma=0.25 = 75 %
   and collapses at sigma=1.00 to 23 %.

The framing below ties (1)-(3) to one mechanism.  No new results;
this is a paper-narrative document that fits the empirical record
into a Bayesian / bias-variance picture, with one offline
diagnostic that supports the trajectory-level multi-modality
story.

---

## 1.  The score-bridge is Gaussian classifier guidance with a
##     trajectory-shaped likelihood

MBD's denoising step at iteration k computes a softmax-weighted
mean of M trajectory samples around the current mean trajectory
`mu`.  The score-bridge adds a per-step pull:

```
mu_{k+1}  =  softmax-of-rollout-costs (cost-driven update)
            +  lambda_prior * (a_target - mu_k) / sigma_k^2  (prior pull)
```

The second term is *exactly* Gaussian classifier guidance with the
target distribution `N(a_target, I)` at noise level `sigma_k`.
This is the same algebraic shape as classifier-free guidance in
text-to-image diffusion (Ho & Salimans 2022), with two differences:

- The "classifier" is a function of the warm-start trajectory
  (`a_target = initial_mean[t]` at step t), not a learned net.
- The guidance scale `lambda_prior` multiplies a *fixed*
  per-step deviation rather than the difference between a
  conditional and unconditional score.

This identification predicts (and our data confirm):

- **Unimodal-in-lambda peak (fact 1).**  Classifier guidance is
  known to be non-monotone: weak guidance does not bias enough,
  strong guidance over-commits to the target's mode and
  suppresses the prior's exploration mass.  The lambda-sweep
  shape is a classical guidance-scale curve.

- **Sigma-dependence.**  The bridge's effective strength is
  `lambda_prior / sigma_k^2`.  At fixed lambda, large sigma
  weakens the bridge; small sigma amplifies it (until the
  per-step noise itself becomes negligible and the warm-start
  is followed essentially open-loop).  This sets up the
  lambda-sigma interaction in fact 3.

The novelty in our story is not guidance itself but **how the
guidance target is constructed** -- specifically the contrast
between BC (a smoothed parametric prior) and `demo_1nn` (a raw
non-parametric retrieval).

---

## 2.  Bias-variance framing of the prior target

Treat the warm-start trajectory `a_target` as an estimator of
"the right action sequence from this state".  Three priors with
different bias-variance profiles:

| Prior         | Bias source                                 | Variance source                    | Effective `a_target` |
|---------------|---------------------------------------------|------------------------------------|---------------------|
| **`demo_1nn`** | None (raw demo action -- in-distribution by definition) | Single-demo: depends on which demo is closest | One demo's actions |
| **`demo_knn`**, **`demo_lwr`** | Cross-demo averaging at mode-ambiguous states | Reduced (averages out demo-specific noise) | Mean of k nearest demos |
| **BC**          | MLP fits all demos with one network; smooths at multi-modal points | Low (MLP forward is deterministic) | Smoothed best-fit action |

The score-bridge does **not** know which of these the
`a_target` came from.  It applies the same Gaussian pull.  What
differs is **what the prior is biased toward**.

**The key claim (Story B's strongest form)**: when MBD denoises
near a multi-modal demo region, BC's smoothed target lives in
an action region that no actual demo would pick -- so pulling
MBD toward BC drags the trajectory *off* the demo manifold.
1-NN's target is on the manifold by construction, so the same
pull moves MBD *along* the manifold.  This is why 1-NN's
optimal lambda is small (lambda=1 -- a weak guide is enough)
while BC's is large (lambda=5 -- need a strong pull to escape
MBD's own cost-shaped distribution that is otherwise misaligned
with the smoothed BC target).

---

## 3.  Why "multi-modal demo region" is the relevant operating regime

Hypothesis: the BC-vs-1NN gap is concentrated at states where
the k-NN action sequence (i.e., what the demo bank would
predict next) is **highly variable across the top-k neighbours**.
At such states, averaging breaks; mode-selection wins.

We can test the *prevalence* of such states offline by
measuring the action-dispersion across each state's top-5 NN.

**Diagnostic (2026-05-20, n=1000 sampled timesteps from the
Can bank, K=5 NN)**:

| lookahead step | median NN action std | q99 NN action std | % states with std > 0.3 |
|---------------:|---------------------:|------------------:|-----------------------:|
| t + 0          | 0.040               | 0.194             | 0.0 %                   |
| t + 10         | 0.045               | 0.187             | 0.0 %                   |
| t + 40         | 0.038               | 0.288             | 0.6 %                   |
| t + 60         | 0.031               | 0.314             | 2.0 %                   |
| t + 100        | 0.000               | 0.364             | 4.1 %                   |

(Action range is bounded in [-1, 1] for the OSC controller, so
"std > 0.3" means neighbours disagree by > 60 % of the per-dim
range.)

Two reads:

- At t = 0 the top-5 neighbours essentially agree (median std
  = 0.04).  BC and 1-NN should be *indistinguishable* at the
  start of a rollout.  This is consistent with our flat
  lambda=0 numbers (11.7 % for both priors).

- By t = 60-100, a non-trivial 2-4 % of sampled states have
  high neighbour-disagreement.  Over a H=120 warm-start window
  that means **most warm-start trajectories pass through at
  least one such state**.  At these states, BC's averaged action
  pulls MBD off the demo manifold; 1-NN's commits to one mode
  and stays on-manifold.

- The correlation between `std@t0` and `std@t60` is only **0.14**.
  Neighbour-ambiguity at the chunk-start step does *not*
  predict ambiguity 60 steps in.  This rules out a simple
  "ambiguous states have ambiguous initial obs" story --
  ambiguity emerges from *trajectory divergence*, not local
  state geometry.

This is exactly the "trajectory-level multi-modality" picture
that the score-bridge + retrieval combination exploits and that
BC's smoothing destroys.

---

## 4.  What the framing predicts for Slate F and beyond

The bias-variance + classifier-guidance frame makes specific,
falsifiable predictions for the four ablations in Slate F's
Track B:

- **`demo_lwr_k5` (inverse-distance-weighted average)**:
  *prediction* -- between 1-NN and demo_knn (which collapsed
  to ~50 % at k=5 with uniform mean).  Distance weighting
  preserves some mode-selection.  Expected: ~60-70 %.

- **`demo_topk_random_k5`**:  *prediction* -- close to 1-NN
  (within ~5 pp), because at most states the top-5 demos agree
  (median std = 0.04).  The variance penalty kicks in only at
  the ~2 % of states with high neighbour-disagreement, so the
  cell-level average rate should be only mildly hit.

- **`demo_1nn_proprio`**:  *prediction* -- close to full-obs
  1-NN.  The retrieval signal is dominated by robot-config
  matching (~40/46 dims), so dropping the 6 object dims should
  barely change retrieval.

- **`demo_1nn_object`**:  *prediction* -- significantly below
  1-NN.  6 obs dims (3 pos + 3-of-4 quat -- the quat has
  effective dim 3) carry far less information than 40 proprio
  dims; many demos at the same object pose have very different
  robot configurations.  Expected: ~30-50 %.

And for Track A (Square):

- **F1 lambda-sweep on Square**:  *prediction* -- a clear
  inverted-U with a peak above both endpoints.  Peak location
  could be at lambda=1 (matching Can) or higher if Square's
  contact geometry rewards stronger commitment to the demo.
  Peak rate likely lower than Can in absolute terms (Square is
  harder: nut-on-peg insertion is more contact-rich than
  pick-place can).

- **F2 sigma-tail on Square at lambda=1**:  *prediction* -- a
  sigma-peaked curve (matching the Can lambda=1 regime), not a
  flat curve (which would indicate Square's optimum is at
  higher lambda).  If F2 is flat, that's signal that Square
  sits in the bridge-dominated regime at lambda=1 -- in which
  case the *real* Square peak is somewhere lambda > 1.

The single most informative result will be the *shape* of F1,
not its peak height.  If the shape is the same as Can's, the
mechanism transfers.  If the shape is different (e.g.,
monotone increasing, or flat), the mechanism is task-specific.

---

## 5.  Open theoretical question (deferred to paper-writing)

The score-bridge as written assumes the prior's target is a
**Gaussian** with the trajectory action as the mean.  But
multi-modal demo regions are exactly where this Gaussian
assumption fails -- the true conditional `p(action | state)`
is a mixture of modes, not a Gaussian.

If we replaced the bridge's Gaussian likelihood with a
mode-aware likelihood (e.g., conditional GMM, or a max-
likelihood-over-top-k rather than mean), would the optimal
lambda location shift?  In the current setup, 1-NN already
implements a poor-man's mode selection at the *target* level,
which is why it tolerates a weak bridge (lambda=1).  A
mode-aware bridge might let BC reach 1-NN's headline by
restoring mode-selection at the *bridge* level.

This is the natural Branch C "Phase 3 adaptive lambda(sigma)"
follow-up reframed: instead of adapting lambda along the
denoising chain, adapt the *target's likelihood shape* to
match the local prior's mode count.  Outside the scope of this
project's paper, but a clean follow-up.

---

## 6.  Status

- §§ 1, 2, 4 are the paper-ready narrative.  §§ 3 is the
  diagnostic that supports § 2's bias-variance picture with
  one concrete number.  §§ 5 is "future work".
- Re-read after Slate F lands; the Square cross-task results
  either substantiate § 4 or force a revision.

**Branch / source**: `feat/zero-train-priors`.  Diagnostic
script lives at
`experiments/15_unified_hybrid/diagnostics/mode_coverage_trajwise.py`
(run 2026-05-20).  Re-run with
`python experiments/15_unified_hybrid/diagnostics/mode_coverage_trajwise.py`
from the repo root to reproduce the table in § 3.
