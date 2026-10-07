# Bias-variance Pareto: summary-level proxy from T2 multiseed

**Mass perturbation:** 1.3.  **Anchor (zero-bias reference):** 
``hybrid_canonical`` (mean = 71.11 pp across 3 seeds).

Per-method 3-seed proxy decomposition:

| Cell | per-seed succ (k/n) | mean | var (pp²) | bias (pp) | bias² (pp²) | MSE (pp²) | Pareto-optimal? |
|---|---|---:|---:|---:|---:|---:|---|
| T2.0  hybrid + canonical bank | 21/30 · 21/30 · 22/30 | 71.1 | 2.47 | +0.00 | 0.0 | 2.5 | yes |
| T2.1  hybrid + bank + 5 demos | 22/30 · 19/30 · 20/30 | 67.8 | 17.28 | +3.33 | 11.1 | 28.4 | no, dom by hybrid_canonical, hybrid_20 |
| T2.2  BC unadapted (canonical) | 9/30 · 9/30 · 9/30 | 30.0 | 0.00 | +41.11 | 1690.1 | 1690.1 | yes |
| T2.3  BC retrained on 205 demos | 0/30 · 4/30 · 2/30 | 6.7 | 29.63 | +64.44 | 4153.1 | 4182.7 | no, dom by hybrid_canonical, hybrid_5, bc_unadapted, hybrid_20 |
| T2.4  hybrid + bank + 20 demos | 21/30 · 22/30 · 21/30 | 71.1 | 2.47 | +0.00 | 0.0 | 2.5 | yes |

## Pareto-dominance headline

T2.1 (hybrid+5):  bias² = **11.1**, variance = **17.28** (pp²).
T2.3 (BC retrain on 205): bias² = **4153.1**, variance = **29.63** (pp²).

**T2.1 strictly dominates T2.3 in (bias², variance) space:** YES.

Bias² ratio:  374x lower for T2.1.
Variance ratio:  1.71x lower for T2.1.
MSE ratio:  147x lower for T2.1.

This is the constructive demonstration of the bias-variance Pareto
theorem (``docs/theory/03_pareto_bias_variance.md`` Corollary 1) and
the asymmetric-robustness model
(``docs/theory/08_asymmetric_robustness_model.md``) in adaptation
space:  the same 5 perturbed demos that BC absorbs parametrically
(↑ bias) only enter the hybrid through a cost-projected score (↓ bias).

## Caveats

1. The proxy uses **per-cell success rates** rather than per-step
   actions.  The per-step decomposition that ``measure_bias_variance.py``
   scaffolds is still future work for a journal version.
2. n = 3 seeds is small for variance estimation; the qualitative
   ordering (which method has the higher bias / variance) is robust,
   but exact ratios should be read with caution.
3. The anchor (``hybrid_canonical`` at the same perturbation) is
   itself a method, not a ground-truth oracle, so the bias proxy is
   relative.  This is the right relative-bias quantity for the
   Corollary-1 dominance question (which is itself relative).
