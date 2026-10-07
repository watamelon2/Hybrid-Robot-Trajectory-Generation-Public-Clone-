# Theory bundle — index

The project's theoretical contribution lives in an
**eight-document bundle** under `docs/theory/`. Each document
states one main theorem, derives it from explicit assumptions,
matches it against the empirical record, and lists falsification
protocols. The bundle is **self-contained**: each doc cites only
its predecessors, so any individual theorem can be quoted with
at most three layers of cross-reference.

The headline rate of the bundle is

$$
\alpha(\lambda) \;=\; \frac{\lambda^2\, \sigma_{\min}^2}{2\, C_{\mathrm{LSI}}}
\;+\; \mathrm{(small corrections)},
$$

derived in doc 06 (Theorem 4) and applied throughout. The
**single number** the final report's theory section should
cite.

---

## The seven documents

| # | Document | Main theorem | One-line statement |
|---:|---|---|---|
| 01 | [`01_soft_control.md`](01_soft_control.md) | Claim 1 + 2 | Soft-control optimum is product-of-experts; LQ limit = one step of policy iteration. |
| 02 | [`02_snis_bound.md`](02_snis_bound.md) | Theorem 1 | $\mathbb{E}\|\hat s - s\|^2 \le D^2 V_w / (\sigma_k^2 M) + O(M^{-2})$. |
| 03 | [`03_pareto_bias_variance.md`](03_pareto_bias_variance.md) | Theorem 1' | $\mathrm{MSE}(\mathrm{hybrid}) \le \min(\mathrm{MSE}(\mathrm{data}), \mathrm{MSE}(\mathrm{model})) - \delta$. |
| 04 | [`04_sample_complexity.md`](04_sample_complexity.md) | Theorem 2 + Corollary 2.4 | $N_{\mathrm{hybrid}} \le N_{\mathrm{pure}} / e^{2 \alpha K / \alpha_{\mathrm{SM}}}$. |
| 05 | [`05_adaptive_robustness.md`](05_adaptive_robustness.md) | Theorem 3 + Corollary 3.5 | DRO radius $\delta(K) = \Omega(e^{2\alpha K})$. |
| 06 | [`06_predictor_corrector_sde.md`](06_predictor_corrector_sde.md) | Theorem 4 | $W_2^2(\mathcal{L}(\bar\tau_0^{\mathrm{hybrid}}), q^\star) \le e^{-2 \alpha K} W_2^2(p^\phi, q^\star) + C V_w/M$. |
| 07 | [`07_minimax_regret_synthesis.md`](07_minimax_regret_synthesis.md) | Theorem 5 + Theorem 6 | Linear composition is minimax optimal; MPC regret is $\tilde O(\sqrt{T})$. |
| 08 | [`08_asymmetric_robustness_model.md`](08_asymmetric_robustness_model.md) | Asymmetric-robustness model | Hybrid absorbs distribution-shift bias parametrically while BC absorbs it non-parametrically; explains the T2.1 vs T2.3 Pareto gap. |

---

## How the bundle composes

```
Doc 01 (soft-control optimum: bias side)
        |
        v
Doc 02 (SNIS bound: variance side)
        |
        v
Doc 03 (Pareto bias-variance synthesis)
        |
        +--------------+
        v              v
Doc 04            Doc 05
(shorter          (adaptive
 training)         robustness)
        \            /
         \          /
          v        v
       Doc 06 (W2 contraction;
                derives alpha(lambda))
                  |
                  v
       Doc 07 (minimax + regret;
                synthesis;
                open problems)
```

The dependency is **strictly forward**: doc $N$ cites only
docs $1, \ldots, N-1$. The full proof of any theorem requires at
most three layers of dependency.

---

## What the bundle proves

The hybrid (zero-training retrieval prior + score-bridge MBD)
is:

1. **Pareto-optimal in (bias, variance)** at finite compute
   (doc 03).
2. **Provably contractive** in $W_2$ at rate $e^{-\alpha K}$
   (doc 06).
3. **Sample-complexity reducing** at fixed deployment cost
   (doc 04 — answers the user's first sub-question, *"shorter
   training"*).
4. **DRO-certifying** at fixed training data (doc 05 —
   answers the user's second sub-question, *"added
   adaptability"*).
5. **Minimax-optimal** within the linear composition family
   (doc 07 Thm 5).
6. **Sub-linear regret** $\tilde O(\sqrt{T})$ in the MPC setting
   (doc 07 Thm 6).

The two answers in (3) and (4) collapse onto the **same
contraction rate** $\alpha(\lambda) = \lambda^2 \sigma_{\min}^2 /
(2 C_{\mathrm{LSI}})$ of doc 06. The single empirical anchor is
the **1.059× DARE excess** on the LQ companion (Week-4), which
matches the predicted rate constants to within rounding.

---

## What the bundle does not prove

Per doc 07 §7, seven open problems remain:

1. LSI verification on Robomimic (~3-5 days).
2. Mode-separation tightening on Square (~1 week).
3. Finite-$N$ theory with explicit constants (~2 weeks).
4. Beyond log-concavity via stochastic localisation (~3 weeks).
5. Connection to RLHF / preference learning (~3-4 weeks).
6. Continuous-time limit at infinite $K$ (~3-5 weeks).
7. The "very good prior" second-order theory (~2 weeks).

Items 1-3 are the immediate follow-up; items 4-7 are paper- or
research-program-grade.

---

## How to use this bundle in the final report

The proposed report-theory section (~5 pages) maps to the
bundle as:

1. **Setup + soft-control derivation** — doc 01 §§1-3
   (1.5 pages).
2. **SNIS variance bound** — doc 02 §§1-2 (1 page).
3. **Pareto bias-variance theorem** — doc 03 §5 (0.5 page).
4. **Sample-complexity reduction** — doc 04 §2 (1 page).
5. **DRO adaptation certificate** — doc 05 §2 (1 page).
6. **Synthesis paragraph** — doc 07 §6 (0.25 page).

Theorems 4 (doc 06), 5 (doc 07), and 6 (doc 07) go into an
appendix (3 pages) with full proofs. Open problems go into the
"Future Work" section. The 1.059× DARE empirical anchor is the
single numerical match-to-data cited in the main text.

---

## Status

- **Docs 01-03**: paper-ready (merged into `main` on 2026-05-25
  via `feat/theory`).
- **Docs 04-07**: paper-ready (merged into `main` on 2026-05-26
  via `feat/theory-extensions`, commit `6ec2ca6`).  Open problems
  of doc 07 §7 are the natural-next-theory line.
- **Doc 08** ([`08_asymmetric_robustness_model.md`](08_asymmetric_robustness_model.md)):
  paper-ready (merged into `main` on 2026-06-01 via
  `feat/adapt-pareto-analysis`, commit `aaefba2`).  Formal model
  for the empirical Hybrid-vs-BC asymmetry observed in the T2
  4-knob robustness slate; connects to doc 03's Pareto bound via
  the proxy `(bias², variance)` decomposition shipped in
  `experiments/15_unified_hybrid/proxy_bias_variance.py` (T2.1
  strictly Pareto-dominates T2.3 on the cell-level proxy).
- **Empirical companions**: most falsification protocols deferred
  per `docs/plans.md`'s no-sweep-before-consolidation rule.  The
  cell-level proxy of doc 03's Pareto bound IS now empirically
  verified via doc 08's measurement (see
  [`experiments/15_unified_hybrid/results/proxy_bias_variance/bias_variance_proxy.md`](../../experiments/15_unified_hybrid/results/proxy_bias_variance/bias_variance_proxy.md));
  the per-step action-level measurement is scaffolded in
  `experiments/15_unified_hybrid/measure_bias_variance.py` but
  deferred to a journal version.

The bundle is the **finished** version of the project's
theoretical contribution as of 2026-06-01 (eight docs, all on
`main`).
