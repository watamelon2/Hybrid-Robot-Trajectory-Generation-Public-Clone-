# Asymmetric Robustness of Hybrid vs BC under Incremental Adaptation

A formal model for the asymmetric-robustness finding from the
T2.3 mechanism diagnostic (commit `402998d`): on canonical Can
init-states, BC retrained on 200 canonical + 5 mass=1.3 demos
shifts its gripper output by 0.730 absolute (5.4x the
position-axis shifts; 5.4x the CPU/CUDA float-noise floor),
while the hybrid + same bank shows no analogous shift.

## Setup

Let
$$\mathcal{B} = \{(s_i, a_i)\}_{i=1}^N$$
be the canonical training bank and
$$\mathcal{B}^{+} = \mathcal{B} \cup \{(s_j^{p}, a_j^{p})\}_{j=1}^M$$
the augmented bank with $M$ perturbed demos.  In T2,
$N = 200,\ M = 5$.  Let $\pi_*(s)$ be the "canonical reference"
policy on a state $s$ (the canonical hybrid + canonical bank, on
the canonical dynamics).  We measure each method's action bias
against $\pi_*(s)$ and its variance across noise realisations.

## BC: parametric absorption

The BC policy minimises an MSE loss over the bank:
$$\theta_{\text{BC}}^* = \arg\min_\theta \sum_{(s, a) \in \mathcal{B}^{+}} \|\pi_\theta(s) - a\|^2.$$
For an MLP-class hypothesis $\pi_\theta$ with sufficient capacity,
the fixed point on each $s$ acts as a soft kernel regression over
demos:
$$\pi_{\text{BC}}^*(s) \;\approx\; \sum_{i \in \mathcal{B}^{+}} w_i(s)\, a_i, \quad \sum_i w_i(s) = 1.$$
The weights $w_i$ depend on $s$'s position in the latent space the
MLP carves out.  At canonical $s$ states with sparse canonical
coverage, the **nearest perturbed demo can dominate** the weighted
sum even when $M/N$ is small (here 2.4 %).  Consequently:
$$\text{bias}_{\text{BC}}(s) \;=\; \pi_{\text{BC}}^*(s) - \pi_*(s)
\;\propto\; \sum_{j} w_j^{p}(s)\, [a_j^{p} - \pi_*(s)]$$
where $w_j^{p}$ is the perturbed-demo weight.  The gripper
output shift of $0.730$ on the affected axis is exactly this
mechanism: at borderline grip-close decisions the nearest perturbed
demo (in latent space) pulls $a$ across the action-space
threshold.  **BC's parametric form absorbs the perturbed bias
directly into $\theta^*$.**

## Hybrid: cost-projected correction

The hybrid policy (MBD with prior score from the bank) solves a
soft-control problem
$$q^*(\tau) \propto p^{\text{bank}}(\tau) \cdot \exp\left(-\tfrac{1}{\beta} J(\tau, s, x_{\text{goal}})\right),$$
where $\tau$ is an action trajectory of length $H$ and $J$ is the
deployed cost.  Sampling proceeds by an MPC iteration that
descends both the prior score $\nabla_\tau \log p^{\text{bank}}(\tau)$
and the cost gradient $\nabla_\tau J(\tau)$.  Let $\mathcal{S}$
denote the success region of $J$ at the canonical $s$.  The
hybrid's expected output, in the small-$\beta$ regime where $J$
dominates, is
$$\pi_{\text{hyb}}^*(s) \;\approx\; \Pi_{\mathcal{S}}\left[
  \arg\max_\tau p^{\text{bank}}(\tau)\right]$$
where $\Pi_{\mathcal{S}}$ is a cost-induced projection onto
$\mathcal{S}$.

The same nearest-perturbed-demo pull happens in the prior score
term, but $\mathcal{S}$ at canonical $s$ is governed by the
**canonical dynamics** (the cost $J$ measures the canonical
success: cube/can in canonical-bin position).  Any perturbed-demo
bias that would push $\tau$ outside $\mathcal{S}$ is projected
back by the cost gradient.  **The hybrid's cost-projection
correction discards the bias before execution.**

## Predicted asymmetry

Combining the two:
- $\|\text{bias}_{\text{BC}}\|$ scales with the perturbed-demo
  weights $w^{p}$ at $s$, which can be $O(1)$ on borderline
  states even at $M/N \ll 1$.
- $\|\text{bias}_{\text{hyb}}\|$ is upper-bounded by the
  cost-projection slack $\|\tau_{\text{bank}} - \Pi_\mathcal{S}\tau_{\text{bank}}\|$,
  which is $O(\beta)$ in the small-temperature regime.

For $\beta \to 0$, the hybrid's bias vanishes regardless of
bank composition.  For finite $\beta$ (the canonical
$\lambda = 1, \sigma_{\max} = 0.25$ regime), the hybrid still
inherits some bank bias, but only along directions tangent to
the success region's boundary.

This explains the observed asymmetry (3-seed multiseed pool, N=90 rollouts/cell at mass=1.3):
| Metric                              | BC retrain (T2.3) | Hybrid + bank (T2.1) |
|---|---:|---:|
| Gripper $\|\Delta a\|$ on canonical $s$ | 0.730           | (small)              |
| Succ rate (3-seed pool)              | 6.7 % (6/90) [3.1, 13.8] | 67.8 % (61/90) [57.6, 76.5] |
| bias² proxy (pp²) vs T2.0 anchor    | **4 153.1**     | 11.1 (**374× lower**) |
| variance proxy (pp²)                | 29.6            | 17.3 (1.7× lower)    |
| MSE proxy (pp²)                     | **4 182.7**     | 28.4 (**147× lower**) |

The bias², variance, and MSE numbers are computed by
`experiments/15_unified_hybrid/proxy_bias_variance.py` from the
already-merged multiseed slate; see `docs/results_writeups/
pareto_analysis.md` for the full Pareto-dominance verification.

## Connection to the bias-variance Pareto theorem

`docs/theory/03_pareto_bias_variance.md` Corollary 1 states the
hybrid sits on a strictly-dominating front in (bias², variance)
space.  This branch's empirical T2 result, combined with the
asymmetric-bias model above, gives a **constructive
demonstration** of the theorem in adaptation space:

1. The 5-demo bias is delivered to both methods identically (same
   bank).
2. BC absorbs it parametrically → high bias on borderline states.
3. Hybrid projects it through cost → small bias on the same states.
4. Both methods see the same MBD-side stochasticity, so variance
   is comparable.

The asymmetry is quantified in two layers:

- **Cell-level (bias², variance) proxy** —
  `experiments/15_unified_hybrid/proxy_bias_variance.py` reads the
  merged multiseed slate and computes per-method
  $(\mu, v, b^2, b^2 + v)$ in pp² from the 3-seed succ-rate replicates.
  Lands on `main` with this branch (see table above).  Sufficient for
  the qualitative Pareto-dominance check.

- **Per-step action-level (bias², variance)** —
  `experiments/15_unified_hybrid/measure_bias_variance.py` would
  capture per-step executed actions and compute the standard
  $\mathrm{MSE} = \|\text{bias}\|^2 + \text{trace}(\text{Cov})$ in
  action-space directly, attributing bias to specific action dims
  (e.g. dim 6 = gripper, predicted by the model above).  Scaffolded;
  requires a `--save-executed-actions` hook in `run_can.py` and a
  full `replay_hybrid_t2_1`.  Deferred to a journal version.

## What this model does **not** explain

- **The Lift cross-task null**.  The Lift hybrid saturates under
  every available perturbation magnitude.  The model above
  predicts $\mathrm{bias}_{\text{hyb}} \to 0$ as $\beta \to 0$,
  but does not say what magnitude of perturbation is required to
  push $\mathcal{S}$ far enough that even the cost-projected
  hybrid degrades.  The Lift saturation is a property of the
  success criterion's geometric tolerance, not of the
  adaptation mechanism.

- **The catastrophic 0/30 for T2.3** (vs the 5.4x noise-floor
  metric).  The 0.730 shift is "large" relative to the
  $[-1, +1]$ action range, but the model so far only predicts
  bias direction, not magnitude.  A quantitative prediction
  would need a model of the BC kernel's effective bandwidth in
  the latent space.

These gaps are out of scope for the present writeup; they would
make natural follow-ups in a journal version.

## References

- `docs/theory/03_pareto_bias_variance.md` -- Pareto theorem.
- `docs/results.md`, "## Slates Incremental T1 + T2" -- empirical
  T2 headline and the T2.3 mechanism diagnostic.
- `docs/results_writeups/pareto_analysis.md` -- this branch's
  writeup with the proxy (bias², variance) table.
- Commit `402998d` -- the gripper-shift measurement that motivated
  this model.
- `experiments/15_unified_hybrid/proxy_bias_variance.py` --
  cell-level proxy measurement (lands with this branch).
- `experiments/15_unified_hybrid/measure_bias_variance.py` --
  scaffolded per-step measurement, deferred to journal version.
- `experiments/15_unified_hybrid/results/proxy_bias_variance/
  bias_variance_proxy.{json,md}` -- proxy outputs.
