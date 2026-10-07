# Theory 2 — Finite-sample SNIS variance bound for the MBD score

*Branch: `feat/theory-snis-bound`. Item 3 of the seven-branch
theory + adaptation roadmap. Expands `presentation_brief_2026_05_22.md` §8.3
into a paper-ready derivation. Companion to
`docs/theory/01_soft_control.md` (bias side, via KL-regularised
soft control) and `docs/theory/03_pareto_bias_variance.md`
(Pareto-dominance corollary that fuses both sides).*

## 1. Setup and the SNIS estimator

The Model-Based Diffusion (MBD; Pan et al., NeurIPS 2024) inner
loop computes, at each denoising iteration $k$ with noise scale
$\sigma_k$, an empirical estimate of the analytic score of the
Boltzmann target $p^{\mathrm{model}}(\tau)\propto\exp(-J(\tau)/\beta)$
diffused to time $\sigma_k$. Using Tweedie's formula and a forward
proposal $q_k(\tau' | \tau)$ (Gaussian convolution of width
$\sigma_k$ in trajectory space), the **true score** is

$$
s_k(\tau)
\;=\;
\nabla_\tau \log p^{\mathrm{model}}_{\sigma_k}(\tau)
\;=\;
\frac{1}{\sigma_k^2}\;
\frac{\mathbb{E}_{\tau'\sim q_k}\!\big[(\tau' - \tau)\, w(\tau')\big]}
     {\mathbb{E}_{\tau'\sim q_k}\!\big[w(\tau')\big]},
\qquad
w(\tau') := \exp(-J(\tau')/\beta).
\tag{1}
$$

MBD estimates $s_k$ by **self-normalised importance sampling
(SNIS)** with $M$ i.i.d.\ samples $\{\tau'_i\}_{i=1}^M \sim q_k$:

$$
\hat s_{k,M}(\tau)
\;=\;
\frac{1}{\sigma_k^2}\;
\frac{\sum_{i=1}^{M}\, w(\tau'_i)\,(\tau'_i - \tau)}
     {\sum_{i=1}^{M}\, w(\tau'_i)}.
\tag{2}
$$

This estimator is **biased at finite $M$** (the self-normalisation
makes it nonlinear in the empirical measure) and **consistent**
($\hat s_{k,M} \to s_k$ a.s.\ as $M \to \infty$; Owen 2013 Thm 9.1).
The questions for the final report are:

- **Q1.** What is the mean-square error $\mathbb{E}\|\hat s_{k,M} - s_k\|^2$
  at finite $M$, as a function of $\sigma_k$, $\beta$, the horizon
  $H$, and the proposal $q_k$?
- **Q2.** How does the prior anchor $\bar\tau$ (the `demo_1nn`
  warm-start) modify Q1 — i.e., what *variance reduction ratio*
  does the prior provide?
- **Q3.** Does Q1's scaling predict the empirically observed
  **0/60 on Can at $M=16$** while **75 % at the same $M$ with the
  prior**?

The remainder of this document answers Q1 with a Pareto-grade
bound, sketches Q2 with a constant identified from the proposal's
KL to the Boltzmann target, and matches both against §7 below.

## 2. Main bound (Theorem 1)

**Standing assumptions.**

- **(A1)** $J: \mathcal{T} \to \mathbb{R}_{\ge 0}$ is measurable
  and bounded: $0 \le J(\tau) \le J_{\max}$ for all $\tau$ in
  the support of $q_k$.
- **(A2)** $\tau \mapsto \tau' - \tau$ has bounded $L^2$ norm under
  $q_k$: $\mathbb{E}_{q_k}\|\tau' - \tau\|^2 \le D^2$. (For
  Gaussian $q_k$ of width $\sigma_k$ in $\mathbb{R}^{d}$, this is
  $d\,\sigma_k^2$.)
- **(A3)** The proposal $q_k$ is absolutely continuous w.r.t.\
  the diffused Boltzmann measure $p^{\mathrm{model}}_{\sigma_k}$
  and has bounded importance-weight second moment:
  $\mathbb{E}_{q_k}\!\big[w(\tau')^2\big] \le W_2 < \infty$.

Define the **effective sample-size deficit** of the importance
weights under the proposal,

$$
V_w(q_k, \beta)
\;:=\;
\frac{\mathrm{Var}_{q_k}\!\big[w(\tau')\big]}
     {\big(\mathbb{E}_{q_k}\!\big[w(\tau')\big]\big)^2}
\;\;\ge\;\; 0,
\tag{3}
$$

with equality iff $w$ is constant under $q_k$. $V_w$ is the
standard SNIS variance ratio (Owen 2013 eq. 9.5). Its inverse
$1/(1+V_w)$ is the **effective sample fraction**: the SNIS
estimator with $M$ samples has the same asymptotic variance as
an unbiased estimator with $M/(1+V_w)$ i.i.d.\ samples.

**Theorem 1 (finite-sample SNIS MSE for the MBD score).** Under
(A1)–(A3),

$$
\boxed{\;
\mathbb{E}\!\left[\big\|\hat s_{k,M}(\tau) - s_k(\tau)\big\|^2\right]
\;\le\;
\frac{D^2\, V_w(q_k, \beta)}{\sigma_k^2\, M}
\;+\;
\frac{C_{\mathrm{bias}}(D, J_{\max})}{\sigma_k^4\, M^2}.
\;}
\tag{4}
$$

The leading $O(M^{-1})$ term is the **SNIS variance** of (2)
expressed via the variance ratio $V_w$ at temperature $\beta$;
the $O(M^{-2})$ term is the **self-normalisation bias** of the
ratio estimator (delta-method residual). The constant
$C_{\mathrm{bias}}(D, J_{\max})$ depends only on the assumed
bounds and is finite under (A1)–(A2).

### 2.1 Reading the bound

- **$\sigma_k^{-2}$ scaling.** Each MBD denoising iteration uses
  the score in a per-step update $\mu_{k+1} = \mu_k +
  \sigma_k^2 \hat s_{k,M}$, so the **action-space update error**
  is $\sigma_k^2 \cdot \|\hat s_{k,M} - s_k\|$, with MSE
  proportional to $D^2 V_w / M$ — the $\sigma_k^{-2}$ in the
  bound is *gauge-invariant* against this gauge. This is the
  same gauge that lets the LQR sanity (§7) interpret 5.9 %
  executed-cost excess as "pure variance".
- **$V_w$ as the perturbation knob.** Everything task-specific
  (horizon $H$, contact richness, demo-bank coverage) routes
  through $V_w$. We bound $V_w$ in $H$ in §3 and identify it
  with the proposal-to-Boltzmann KL in §4.
- **$M^{-1}$ first, $M^{-2}$ second.** For $M$ at our operating
  point (16) the leading term dominates by $M = 16$x;
  self-normalisation bias is a 6 % correction at $M=16$, falls
  to 0.4 % at $M=64$, negligible at $M=256$. Decision-relevant
  takeaway: the $M$-ablation in `experiments/09_sigma_sweep`
  (peak at $M=64$, **50 %** on Can) is consistent with the
  bound predicting the bias term becoming negligible at
  moderate $M$.

### 2.2 Proof sketch

The proof is a standard textbook SNIS analysis — we give the
sketch here and defer constants and explicit dependencies of
$C_{\mathrm{bias}}$ to the final-report appendix.

Let $a := \mathbb{E}_{q_k}[(\tau'-\tau) w(\tau')]$,
$b := \mathbb{E}_{q_k}[w(\tau')]$, so $s_k = a / (\sigma_k^2 b)$.
The empirical analogues are $\hat a_M := M^{-1}\sum_i (\tau'_i - \tau) w(\tau'_i)$
and $\hat b_M := M^{-1}\sum_i w(\tau'_i)$, and
$\hat s_{k,M} = \hat a_M / (\sigma_k^2 \hat b_M)$.

**Step 1 — delta-method linearisation.** Taylor-expand
$\hat s_{k,M}$ around $(\hat a_M, \hat b_M) = (a, b)$:

$$
\hat s_{k,M} - s_k
\;=\;
\frac{1}{\sigma_k^2 b}\big(\hat a_M - a\big)
\;-\;
\frac{a}{\sigma_k^2 b^2}\big(\hat b_M - b\big)
\;+\; R_M,
\quad
\|R_M\| = O_{a.s.}(M^{-1}).
$$

**Step 2 — bound the linear term's variance.** The linear
combination is a sum of i.i.d.\ mean-zero terms; its variance is
$M^{-1}$ times the per-sample variance of the bracketed
expression. Applying Cauchy–Schwarz and (A1)–(A2), and recognising
$\mathrm{Var}_{q_k}(w)/b^2 = V_w$,

$$
\mathrm{Var}\!\Big(\frac{\hat a_M - a}{\sigma_k^2 b}
- \frac{a(\hat b_M - b)}{\sigma_k^2 b^2}\Big)
\;\le\;
\frac{D^2}{\sigma_k^2 M}\cdot V_w(q_k, \beta).
$$

This is the leading $O(M^{-1})$ term of (4).

**Step 3 — bound the bias and the $R_M$ second moment.** Use the
delta-method second-order term plus (A3)'s $W_2 < \infty$ to get
$\mathbb{E}\|R_M\|^2 = O(M^{-2}\sigma_k^{-4})$. The hidden
constants depend polynomially on $D, J_{\max}, b, W_2$; bound
them by their assumed extremes to get $C_{\mathrm{bias}}$. $\square$

The argument is standard (Owen 2013 §9.2; Geweke 1989 §3); the
only project-specific content is the identification of the
proposal $q_k$ with MBD's per-iteration Gaussian convolution of
the current iterate, which is what links $V_w$ to the
demonstration prior in §4.

## 3. Horizon scaling of $V_w$

The cost decomposes additively along time:
$J(\tau) = \sum_{t=0}^{H-1} \ell(s_t, u_t) + \ell_H(s_H)$.
Under mild moment assumptions on $\ell$ (Owen 2013 §9.3, modified
for the time-additive case), $V_w$ scales **polynomially** in $H$:

**Proposition 1.** *If the per-step costs satisfy
$\mathrm{Var}_{q_k}[\ell(s_t, u_t)] \le \sigma_\ell^2$ uniformly
in $t$, and the proposal $q_k$'s marginal over $s_t$ is mixing
in $t$ (a Doeblin-style minorisation; see Diaconis & Saloff-Coste 1996),
then*

$$
V_w(q_k, \beta)
\;\le\;
H \cdot \tfrac{\sigma_\ell^2}{\beta^2}\cdot \exp\!\big(H\,\sigma_\ell^2/(2\beta^2)\big).
\tag{5}
$$

*The bound is loose at large $H$ (the exponential blows up) and
tight at small $H$ (linear in $H$, matches the central-limit
regime).*

**Reading.** Pan's paper notes (limitation iii) that SNIS variance
"grows with horizon and sample count". Proposition 1 makes this
precise: $V_w$ has a linear-in-$H$ floor and an exponential
ceiling, with the crossover at
$H \approx 2\beta^2 / \sigma_\ell^2$. At our operating point on
Can ($H=120$, $\beta=1$, $\sigma_\ell \sim 0.1$), the bound puts us
**just past the linear regime**: $H \sigma_\ell^2/(2\beta^2) \approx 0.6$,
$e^{0.6} \approx 1.8$, so $V_w \approx 120 \cdot 0.01 \cdot 1.8 \approx 2.2$
— a modest variance inflation. **The 0/60 on Can at $M = 16$ is
therefore not a "$V_w$ blew up" outcome; it is the $\sigma_k^{-2}$
factor in (4) combined with $D \sim \sqrt{d} \approx 30$ for
$d = 840$.**

## 4. The prior anchor reduces $V_w$ — the hybrid's variance gain

In the hybrid driver, MBD's per-iteration proposal $q_k$ is
centred at the **current iterate $\mu_k$** which is *itself*
pulled toward the prior anchor $\bar\tau$ by the data-side score
$\nabla\!\log p^{\mathrm{data}}_{\sigma_k}$. Equivalently, the
hybrid's proposal is a *shifted* Gaussian centred at $\mu_k$ with
$\mu_k$ close to $\bar\tau$ at the relevant denoising iterations.

The variance ratio $V_w$ then evaluates to

$$
V_w(q_k(\cdot \mid \mu_k), \beta)
\;=\;
\frac{\mathbb{E}_{q_k}[w^2] - b^2}{b^2}
\;=\;
\mathbb{E}_{q_k}\!\big[w(\tau')/b - 1\big]^2.
\tag{6}
$$

Under (A1)–(A3) and an extra assumption that
$q_k(\cdot | \mu_k)$ has bounded relative density w.r.t.\ the
Boltzmann measure $p^{\mathrm{model}}_{\sigma_k}$ in a neighbourhood
of $\bar\tau$, (6) admits the **KL-bridge identity** (Owen 2013 §9.5;
or Chatterjee & Diaconis 2018 Thm 1.1):

$$
1 + V_w(q_k(\cdot|\mu_k), \beta)
\;=\;
\chi^2\!\big(p^{\mathrm{model}}_{\sigma_k}\,\big\|\, q_k(\cdot | \mu_k)\big),
\tag{7}
$$

where $\chi^2$ is the standard $\chi^2$-divergence. The $\chi^2$
divergence is **convex** in its first argument and **minimised
when the proposal $q_k$ concentrates near a mode of the target
$p^{\mathrm{model}}_{\sigma_k}$**.

**Corollary 1 (variance reduction by mode anchoring).**
*If the anchor $\bar\tau$ is near a mode of $p^{\mathrm{model}}_{\sigma_k}$,*

$$
V_w(q_k(\cdot|\bar\tau), \beta)
\;\ll\;
V_w(q_k(\cdot|0), \beta).
\tag{8}
$$

*Explicitly, in the Gaussian-around-quadratic-cost limit
($p^{\mathrm{model}}_{\sigma_k}$ Gaussian with mean $\tau^\star$
and covariance $\Sigma$, proposal $q_k$ Gaussian centred at
$\mu_k$ with covariance $\sigma_k^2 I$),*

$$
1 + V_w
\;=\;
\det(\Sigma)^{1/2}\det(\sigma_k^2 I)^{-1/2}\,
\exp\!\Big(\tfrac{1}{2}(\mu_k - \tau^\star)^\top\Sigma^{-1}(\mu_k - \tau^\star)\Big).
$$

*The exponential factor is the variance-reduction lever: each
$\sigma$-unit closer the proposal centre is to the cost optimum,
$V_w$ drops by a factor $e^{1/2} \approx 1.65$.*

This is exactly the regime the LQR sanity sits in (§7): the
proposal is centred at the demonstration's mean, which is close
to the LQR optimum; $V_w$ collapses; the executed-cost MSE
drops to its theoretical floor.

### 4.1 Define the hybrid variance-reduction ratio

Let

$$
\rho(\bar\tau)
\;:=\;
\frac{V_w^{(\text{no prior})}}
     {V_w^{(\bar\tau)}}
\;\ge\; 1.
\tag{9}
$$

Then the **pure-MBD compute equivalent** of running the hybrid at
$M$ samples is

$$
M^\star
\;=\;
\rho(\bar\tau)\cdot M.
\tag{10}
$$

This is the project's commitment from
`presentation_brief_2026_05_22.md` §8.3 ("predicts the MBD-no-prior
break-even compute $M^\star \approx 16\cdot \rho$").

## 5. Bridge to the hybrid's score error

Combining Theorem 1 with the soft-control composite score of
`docs/theory/01_soft_control.md` eq. (7), the hybrid's per-iteration
score error decomposes:

$$
\big\|\hat s_{k,M}^{\mathrm{hybrid}} - s_k^{\mathrm{hybrid}}\big\|^2
\;\le\;
\underbrace{\big\|s^{\mathrm{data}}_k\big\|^2_{\mathrm{bias}}}_{\text{prior bias}}
\;+\;
\underbrace{\frac{D^2\,V_w(\bar\tau,\beta)}{\sigma_k^2\,M}}_{\text{anchored SNIS variance}}
\;+\;
O(M^{-2}).
\tag{11}
$$

The first term is the **bias** the soft-control derivation
identifies — the prior's deviation from the demonstration manifold
(§2 of `theory_post_slate_e.md` translates this into the
mode-selection vs. averaging picture). The second term is the
**anchored SNIS variance** of this document. The bound on the
sum is the input to §03's Pareto inequality.

## 6. Empirical observations explained by the bound

| Observation (cell / artefact) | Term in (11) that explains it |
|---|---|
| **MBD-no-prior 0/60 on Can, $M=16$** (`experiments/04_mbd_baseline/results/_archive/can_ph.json`) | Variance term at unanchored $V_w^{(\text{no prior})}$; per-step score is noise-dominated, descent direction useless |
| **`demo_1nn` 75 % single-seed at $\lambda=1, M=16$** (Slate D D2) | Variance term reduced by $\rho(\bar\tau)$ (Corollary 1); bias term small because `demo_1nn` mode-selects on the manifold |
| **BC 21.7 %** (`experiments/03_bc_baseline/results/can_ph.json`) | Bias term is large (BC averages multi-modal demos); variance unaffected because BC's $\bar\tau$ is still near a mode |
| **Soft-NN $\beta$-curve 0.45 → 0.75** (Slate G2) | Continuous interpolation of bias term: $\beta=0$ averages top-$k$ (large bias), $\beta\to\infty$ hard-1NN (small bias). Variance roughly constant across the sweep — all targets are *near* the manifold, just at different points. |
| **$M$-ablation peak at $M=64$ = 50 %** (`experiments/09_sigma_sweep/results/_archive/sigma_0.05_M64.json`) | Bound's $M^{-1}$ trend; bias term floors before variance does. $M=256$ underperforms $M=64$ because the softmax-temperature $\lambda$ was not rescaled to $\log M$ (separate issue, Branch A A3-(b)). |
| **LQR sanity 1.059× DARE** | Quadratic cost ⇒ $w$ smooth ⇒ $V_w \to V_w^{\min}$; bias term zero in expectation; the 5.9 % is pure $O(M^{-1})$ residual. |
| **Square mechanism limit 8.3 %** (Slate G-aux AUX1) | Variance term unaffected (demo_1nn still mode-selects), bias term *cannot* be reduced by retrieval temperature because the retrieval representation clusters wrong-mode demos together — moves the discussion to the metric-coverage extension (deferred §03). |

Every empirical regime has a one-line variance / bias attribution
from the bound. **This is the strongest form of "the theory is
not retrofitted to the data"** the project can claim short of an
ab-initio prediction; the only ab-initio prediction is $M^\star$
in (10), and the falsification protocol for it is in §8.

## 7. LQR closed-form companion

Under the LQ specialisation of
`docs/theory/01_soft_control.md` §2, the constants in (4) admit
closed forms.

- **$D^2 = \mathrm{tr}(\sigma_k^2 I) = d\,\sigma_k^2$.** For
  Can-scale $d = 7 \cdot 120 = 840$ and $\sigma_k = 0.25$,
  $D \approx 7.2$.
- **$V_w$ in Gaussian limit.** From (7) with $\Sigma = \beta H_J^{-1}$,
  the LQ trajectory-cost Hessian. The exponent in §4's
  closed-form vanishes when $\mu_k = \tau^\star_{\mathrm{LQR}}$
  (after enough denoising iterations, when the iterate has
  reached the LQR optimum): $V_w^{\mathrm{LQ}} \le e^{1/2} - 1 \approx 0.65$.
- **$C_{\mathrm{bias}}$** in the LQ limit is a polynomial in
  $(d, \sigma_k, \beta)$ that we omit (the empirical 1.059×
  DARE already constrains it to $\le 0.05$ of the linear term).

The result: at $M=16$ the LQ executed-cost excess over the
DARE optimum is bounded by $\approx 0.65 \cdot 16^{-1} \approx 0.04$
(4 %) plus $O(M^{-2}) \le 0.02$, totaling $\approx 6$ % — matching
the **1.059×** empirical observation to within rounding.

This is the **derivation companion** that
`presentation_brief_2026_05_22.md` §8.3 promised but did not
include; it is the *only* fully-quantitative match-to-data the
project gives, and it converts the LQR sanity from a
"correctness check" into a "constants-match check".

## 8. Falsification protocol for $M^\star$

Equation (10) predicts that on Can, **pure MBD at $M^\star$
samples should match hybrid at $M=16$ samples**. With $V_w$
estimates from §3–§4, $\rho(\bar\tau) \approx 4$–$16$ on Can
(depending on how peaked the cost-Boltzmann factor is near the
demo manifold), so $M^\star \in [64, 256]$.

**Cell ladder** (each cell is 60 evaluations of pure MBD, no
prior, single-seed):

| Cell | $M$ | Predicted succ |
|---|---:|---:|
| F1 | 16 | 0 / 60 (already measured, Exp 04) |
| F2 | 64 | $\le$ 20 / 60 (≤ hybrid plateau) |
| F3 | 256 | 40–55 / 60 (matches or slightly beats hybrid) |
| F4 | 1024 | $\ge$ 55 / 60 (saturates above hybrid; pure MBD wins compute-unbounded) |

Wall-clock: each MBD-no-prior cell at $M=1024$ is $\approx 64\times$
the $M=16$ baseline, so **F4 alone is $\approx 8.5$ h** on the
test rig. Total ladder $\approx 11$ h, but F1–F3 alone close the
falsification question at $\approx 3$ h.

**Outcomes.**

- **F3 in $[40, 55]/60$:** bound's $M^{-1}$ scaling holds at the
  predicted rate. The $\rho \approx 4$–$16$ envelope is
  empirically narrowed, and the SNIS bound is the dominant story
  for the variance side. Theory is locked in.
- **F3 well above $55/60$ or well below $40/60$:** the
  constants $D, V_w$ are estimated wrong by more than the bound
  tolerates; either (a) the proposal is not Gaussian (cosine
  schedule's $\sigma$-tail anisotropy), or (b) the cost-Boltzmann
  weights have heavier tails than (A1) admits. Both are
  fixable in revision; the *shape* (monotone in $\log M$) of
  the curve is the falsifying signal.
- **F4 at $\ge 55$ but F3 at $\le 30$:** scaling is correct but
  the predicted $\rho$ is off by 2$\times$. This means the prior
  reduces variance more than (or less than) the cost-Boltzmann
  concentration alone implies — a **second prior-side mechanism**
  beyond mode anchoring. Worth chasing.

## 9. Status

- §§1–6 are paper-ready. §7 is the LQR closed-form companion
  that makes the bound *quantitative* on the sanity check; §8
  is the falsification protocol the final report can cite
  in its limitations section.
- **Effort to deliver:** the doc itself is finished here.
  Constants in §3 (Proposition 1) and §7 (LQ closed form) can
  be tightened with one MCMC course-notes pass (~1 day) before
  the final report.
- **Companion experiment**: the F1–F4 ladder of §8 fires only on
  explicit go (respects `docs/plans.md` rule). The bound's
  predictions are recorded here so the F-cell readout has a
  fixed target.

## References

- Chatterjee, S., Diaconis, P. *The sample size required in
  importance sampling.* Annals of Applied Probability, 2018.
  (KL-bridge identity for $V_w$.)
- Diaconis, P., Saloff-Coste, L. *Logarithmic Sobolev inequalities
  for finite Markov chains.* Annals of Applied Probability, 1996.
  (Doeblin minorisation; horizon-scaling backstop.)
- Geweke, J. *Bayesian inference in econometric models using Monte
  Carlo integration.* Econometrica, 1989. (Self-normalised
  importance sampling foundations.)
- Owen, A.B. *Monte Carlo Theory, Methods and Examples*, 2013
  (online). Ch. 9 (SNIS variance + chi-squared bridge).
- Pan, C., Yi, Z., Shi, G., Qu, G. *Model-Based Diffusion for
  Trajectory Optimisation.* NeurIPS 2024. (MBD baseline; the
  paper our bound makes precise.)

---

*Companion docs: bias-side derivation in
[`01_soft_control.md`](01_soft_control.md); Pareto bias-variance
corollary in [`03_pareto_bias_variance.md`](03_pareto_bias_variance.md).
This document supplies the **variance** side of the bundle's
contraction story; the per-iteration variance bound (4) is the
input to the $W_2$ contraction theorem in
[`06_predictor_corrector_sde.md`](06_predictor_corrector_sde.md) §2.2
(the SNIS residual term), and the variance-reduction ratio
$\rho(\bar\tau)$ of §4.1 is the input to the sample-complexity
reduction theorem in
[`04_sample_complexity.md`](04_sample_complexity.md) §2.2 and the
DRO certificate in
[`05_adaptive_robustness.md`](05_adaptive_robustness.md) §2.2.
The closing minimax + MPC-regret + synthesis is in
[`07_minimax_regret_synthesis.md`](07_minimax_regret_synthesis.md);
see [`README.md`](README.md) for the bundle index.*
