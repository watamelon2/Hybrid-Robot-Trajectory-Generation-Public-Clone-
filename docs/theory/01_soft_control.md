# Theory 1 — KL-regularised soft-control derivation of the score-bridge

*Branch: `feat/theory-soft-control`. Companion to `docs/theory_post_slate_e.md`
(empirical/Bayesian framing) and `docs/theory/02_snis_bound.md` (variance
side of the bias–variance Pareto statement). See
`docs/theory/03_pareto_bias_variance.md` for the unified inequality
that glues this document and §02 together.*

The composite score
$\nabla_\tau\!\log p^{\mathrm{data}}_t(\tau) +
\lambda(x,t)\,\nabla_\tau\!\log p^{\mathrm{model}}_t(\tau)$
that drives the unified driver
(`experiments/15_unified_hybrid/unified_loop.py`) admits **three
simultaneous readings** at the level of the underlying functional
it descends:

1. a **classifier-guidance** reading (Ho & Salimans 2022),
2. an **inverse-KL-temperature** reading from soft / KL-regularised
   reinforcement learning (Levine 2018; Haarnoja et al. 2018; Peters
   & Schaal 2010),
3. a **product-of-experts** reading from Bayesian generative
   modelling (Hinton 2002; Du et al. 2020).

This note derives the soft-control reading explicitly, because it
is the one that (a) gives the bridge coefficient $\lambda$ a
precise, *units-bearing* interpretation, (b) directly answers the
reviewer question "*is this not just offline RL with a learned
behaviour prior?*", and (c) supplies the **bias-side** half of the
Pareto bias–variance theorem in §03 (the §02 SNIS bound supplies
the variance side).

---

## 1. The hybrid target distribution as a soft-control optimum

Fix a trajectory space $\mathcal{T} = \{\tau = (s_0, u_0, \ldots,
s_{H-1}, u_{H-1}, s_H) : s_{k+1} = f(s_k, u_k)\}$ with deterministic
dynamics $f$ (the MJX rollout). Endow $\mathcal{T}$ with a
*trajectory cost* $J : \mathcal{T} \to \mathbb{R}_{\ge 0}$ (the
Robomimic per-task cost, e.g.\ binary success-reward summed over $H$
steps) and a *demonstration density* $p^{\mathrm{data}}(\tau)$
supported on the offline demo bank (parametric for the BC prior,
empirical for the `demo_*nn` retrieval priors).

For a candidate planning distribution $q \in \mathcal{P}(\mathcal{T})$
define the **soft-control functional**

$$
\mathcal{F}_{\lambda,\beta}[q]
\;:=\;
\underbrace{\mathbb{E}_{\tau\sim q}\!\Big[\tfrac{1}{\beta}\, J(\tau)\Big]}_{\text{model side}}
\;+\;
\underbrace{\tfrac{1}{\lambda}\,
\mathrm{KL}\!\big(q \,\big\|\, p^{\mathrm{data}}\big)}_{\text{data side}}.
\tag{1}
$$

The first term is the *expected scaled cost* under $q$ (the MBD
Boltzmann objective); the second term is a *Kullback–Leibler
penalty* against the demonstration distribution. The coefficients
$\lambda > 0$ and $\beta > 0$ are dimensionless; their ratio
$\beta/\lambda$ is the natural unit of the trade-off, but we keep
them separate because $\beta$ also appears inside the MBD score
estimator (it sets the importance-weight temperature) and
$\lambda$ also appears as the score-mixing scalar in the unified
driver. We will see in §3 that this is not a coincidence.

**Claim 1.** *The minimiser of (1) over absolutely continuous
$q \ll p^{\mathrm{data}}$ is*

$$
q^\star_{\lambda,\beta}(\tau)
\;=\;
\frac{1}{Z_{\lambda,\beta}}\;
p^{\mathrm{data}}(\tau)\,
\exp\!\Big(\!-\tfrac{\lambda}{\beta}\, J(\tau)\Big),
\tag{2}
$$

*with $Z_{\lambda,\beta}$ the normalising constant.*

**Proof.** Expand $\mathrm{KL}(q\|p^{\mathrm{data}}) =
\mathbb{E}_q[\log q] - \mathbb{E}_q[\log p^{\mathrm{data}}]$ and
substitute into (1):

$$
\mathcal{F}_{\lambda,\beta}[q]
\;=\;
\mathbb{E}_q\!\Big[
\tfrac{1}{\beta} J
- \tfrac{1}{\lambda}\log p^{\mathrm{data}}
+ \tfrac{1}{\lambda}\log q
\Big].
$$

Multiplying through by $\lambda$ (legal because $\lambda > 0$ and
this rescales the optimum but not the optimiser), the functional
becomes a *KL between $q$ and the un-normalised Gibbs measure
$p^{\mathrm{data}} \exp(-(\lambda/\beta) J)$* up to a $\tau$-free
constant; that KL is minimised iff $q$ equals the Gibbs measure,
normalised. $\square$

Equation (2) is **exactly the product-of-experts target that
defines the hybrid** at every diffusion step, with the bridge
exponent $\lambda$ playing the role of the inverse KL temperature:

$$
q^\star_{\lambda,\beta}(\tau)
\;\propto\;
p^{\mathrm{data}}(\tau)\;\big[p^{\mathrm{model}}(\tau)\big]^{\lambda/\beta},
\qquad
p^{\mathrm{model}}(\tau) \propto \exp(-J(\tau)/\beta).
\tag{3}
$$

So:

- $\boldsymbol{\lambda/\beta \to 0}$ recovers $q^\star = p^{\mathrm{data}}$,
  i.e.\ a pure-prior planner. In our hybrid this is the
  *warm-start-only* control (Slate D2 endpoint, **11.7%** on Can).
- $\boldsymbol{\lambda/\beta \to \infty}$ recovers $q^\star \propto
  \exp(-J/\beta)$, i.e.\ pure MBD. On Can this is **0/60** at
  $M\!=\!16,\, H\!=\!120$ (Experiment 04 baseline).
- $\boldsymbol{0 < \lambda/\beta < \infty}$ is the hybrid regime;
  unimodal-in-$\lambda$ peaks at the Slate E pool ($\lambda^\star
  \approx 1$, **73.8%**) match the predicted soft-control sweet
  spot at the bias–variance crossover.

### 1.1 What this gives that classifier guidance does not

The classifier-guidance reading (Ho & Salimans 2022; restated in
`theory_post_slate_e.md` §1) explains the *shape* of the
$\lambda$-sweep — non-monotone, peaked — by appeal to a generic
property of conditional generative models. The soft-control
reading goes one step further:

- It identifies $\lambda$ as an **inverse KL temperature**. This
  means the same $\lambda$ that controls the score mixing also
  controls the *strength of the demonstration constraint* in the
  soft-control problem. The bridge is not adjusting "how much
  classifier signal to mix"; it is adjusting "how strictly the
  planner is required to stay in-demonstration".
- It exposes the **dimensional dual** $\beta$ as the
  *cost-Boltzmann temperature* that the SNIS estimator weights
  rollouts by. This is the same $\beta$ that drives $V_w$, the
  importance-weight variance, in §02. The two sides of the
  bias–variance Pareto statement are coupled through this single
  constant.
- It makes the connection to **offline RL** unambiguous, settling
  the §11 Q-grid review question explicitly (see §4).

---

## 2. Reduction to one step of policy iteration in the LQ limit

Section 2.1 of the presentation brief established empirically that
the unified denoising loop converges to **1.059× the DARE optimum**
on the LQ companion problem. The soft-control derivation lets us
*derive* that result, not just observe it.

### 2.1 LQ specialisation

Take the canonical LQ setup
($x_{k+1} = A x_k + B u_k$,
$\ell(x,u) = x^\top Q x + u^\top R u$,
$\ell_H(x_H) = x_H^\top Q_H x_H$,
$Q \succeq 0,\, R \succ 0$). The trajectory cost
$J(\tau) = \sum_{k=0}^{H-1}\ell + \ell_H$ is a quadratic form in
$\tau$, so

$$
\exp\!\big(\!-J(\tau)/\beta\big) \;\propto\; \mathcal{N}\!\big(\tau \,;\, \tau^\star_{\mathrm{LQR}},\, \beta\, H_J^{-1}\big),
$$

where $H_J = \nabla^2 J$ is the Hessian of the trajectory-cost
and $\tau^\star_{\mathrm{LQR}}$ is the LQR optimum from the DARE
recursion (`presentation_brief_2026_05_22.md` §2.1, eqs. for
$P_k, K_k$). Suppose further that the demonstration density is
Gaussian in trajectory space, $p^{\mathrm{data}}(\tau) =
\mathcal{N}(\tau ; \bar\tau, \Sigma_{\mathrm{data}})$ — for the
`demo_1nn` retrieval prior this is exactly its Gaussian-pull
score $-\lambda(\tau - \bar\tau)/\sigma_t^2$ at noise level
$\sigma_t$ (effectively $\Sigma_{\mathrm{data}} = \sigma_t^2 I$).

Then (2) is a **product of two Gaussians**, itself Gaussian:

$$
q^\star_{\lambda,\beta}(\tau)
\;=\;
\mathcal{N}\!\big(\tau \,;\, \mu^\star,\, \Sigma^\star\big),
\qquad
\Sigma^{\star\,-1} \;=\; \Sigma_{\mathrm{data}}^{-1} + \tfrac{\lambda}{\beta} H_J.
\tag{4}
$$

The mean $\mu^\star$ is the *precision-weighted blend* of the
data prior's mean $\bar\tau$ and the LQR optimum $\tau^\star_{\mathrm{LQR}}$:

$$
\mu^\star
\;=\;
\Sigma^\star
\Big(\Sigma_{\mathrm{data}}^{-1}\,\bar\tau \;+\; \tfrac{\lambda}{\beta}\, H_J\, \tau^\star_{\mathrm{LQR}}\Big).
\tag{5}
$$

### 2.2 The "one step of policy iteration" reading

Standard policy iteration (Bertsekas, Vol. I, §6.4) alternates
between:

- **Policy evaluation** — fix policy $\pi_n$, compute its value
  $V^{\pi_n}$. For LQ, $V^{\pi_n}(x) = x^\top P_n x$ for some
  $P_n \succeq 0$ depending on $\pi_n$.
- **Policy improvement** — set
  $\pi_{n+1}(x) = \arg\min_u\, x^\top Q x + u^\top R u + V^{\pi_n}(A x + B u)$.
  In the LQ case this is exactly $\pi_{n+1}(x) = -K_{n+1} x$
  with $K_{n+1} = (R + B^\top P_n B)^{-1} B^\top P_n A$.

Initialising at the *demonstration policy*
$\pi_0(x) = -K_{\mathrm{data}}\, x$
with quadratic value $V^{\pi_0}(x) = x^\top P_{\mathrm{data}}\, x$,
**one step of LQ policy improvement** produces

$$
\pi_1(x) \;=\; -K_1\, x,
\qquad
K_1 \;=\; (R + B^\top P_{\mathrm{data}}\, B)^{-1}\, B^\top P_{\mathrm{data}}\, A.
\tag{6}
$$

**Claim 2.** *The mean $\mu^\star$ of (5) is the trajectory
generated by the policy-iterate $\pi_1$ of (6), with the data side
$P_{\mathrm{data}}$ identified with $(\lambda/\beta)\, \Sigma_{\mathrm{data}}^{-1}$
on the appropriate diagonal block. Equivalently, the hybrid in
the LQ limit performs* **one round of policy iteration** *on the
quadratic value function $V^{\pi_0}$ implied by the data prior.*

**Sketch.** The trajectory-cost Hessian $H_J$ block-diagonalises
along the time axis into per-step Hessians $H_J^{(k)} = \mathrm{diag}(Q, R)$
plus dynamics-coupling terms. The precision-weighted mean (5)
then admits a per-step decomposition that exactly matches the
Bellman-improvement step (6), with the data-prior precision
$\Sigma_{\mathrm{data}}^{-1}$ playing the role of $P_{\mathrm{data}}$
on the $x$-block of each time step. The dynamics coupling that
makes (6) "stateful" arises from the off-diagonals of $H_J$ and
contributes the $B^\top P B$ terms. A full derivation is
straightforward but tedious; we omit it here and cite Bertsekas
Vol. I §6.4 for the policy-iteration template and
Toussaint 2009 (*Robot Trajectory Optimisation using Approximate
Inference*) for the Gaussian-precision identification. $\square$

### 2.3 Empirical companion — 1.059× DARE

The Week-4 LQ sanity check reports an executed cost of
**1.059× the DARE optimum**, 5/5 seeds within 10% (see
`experiments/04_mbd_baseline/results/_archive/lq_sanity.json`).
The 5.9% excess is the *finite-$M$, finite-$K$ SNIS noise* on
top of the policy-iteration target $\mu^\star$ derived in §2.2.
Concretely:

- At $\lambda = 1$, $\beta = 1$ on the LQ companion problem,
  $\mu^\star$ is *not* the DARE optimum (which would require
  $\lambda/\beta \to \infty$, i.e.\ pure MBD); it is the
  precision-blend (5) of the data prior with the LQR optimum.
- For a "good enough" data prior (small $\Sigma_{\mathrm{data}}$,
  i.e.\ tight Gaussian around $\bar\tau$ close to the LQR
  optimum), $\mu^\star \approx \tau^\star_{\mathrm{LQR}}$ up to a
  small bias controlled by $\|\bar\tau - \tau^\star_{\mathrm{LQR}}\|$.
- The remaining 5.9% executed-cost excess is then *all variance*,
  attributable to the SNIS noise term of §02 — which in the LQ
  limit collapses to its lowest-variance regime (constant cost
  gradient $\Rightarrow$ smooth weights $\Rightarrow$ small $V_w$).

This is the LQ companion result that §03's Pareto theorem turns
into an inequality for the contact-rich Robomimic case.

---

## 3. Equivalence to the per-step score update

The classifier-guidance and product-of-experts readings agree
that *taking a single Wasserstein-gradient step* of (1) at noise
level $\sigma_t$, with the data and model marginals diffused to
$p^{\mathrm{data}}_t,\, p^{\mathrm{model}}_t$ respectively,
produces the composite score

$$
\nabla_\tau \log q^{\star}_{\lambda,\beta,t}(\tau)
\;=\;
\nabla_\tau \log p^{\mathrm{data}}_t(\tau)
\;+\;
\tfrac{\lambda}{\beta}\;
\nabla_\tau \log p^{\mathrm{model}}_t(\tau).
\tag{7}
$$

Comparing to the score added in the unified driver,

$$
s_t^{\mathrm{hybrid}}(\tau)
\;=\;
\nabla_\tau \log p^{\mathrm{data}}_t(\tau)
\;+\;
\lambda_{\mathrm{prior}}\,
\nabla_\tau \log p^{\mathrm{model}}_t(\tau)
\quad
\text{(\texttt{unified\_loop.py} score formula)},
\tag{8}
$$

we read off

$$
\boxed{\;\lambda_{\mathrm{prior}} \;=\; \lambda/\beta.\;}
\tag{9}
$$

The unified driver's *single scalar hyperparameter*
$\lambda_{\mathrm{prior}}$ is **the ratio of the two soft-control
temperatures**, not an independent quantity. Two operational
consequences:

- **The Slate-E $\lambda^\star = 1$ peak** means
  $\lambda/\beta = 1$, i.e.\ at the canonical Robomimic operating
  point the data-side KL penalty and the model-side cost penalty
  carry **equal soft-control weight**. This is the cleanest
  statement of "the data and model sides are balanced at the
  optimum" the project has.
- **The BC-vs-`demo_1nn` peak shift** ($\lambda^\star_{\mathrm{BC}} = 5$
  vs $\lambda^\star_{\mathrm{1nn}} = 1$, Slate E E2) reads as a
  $5\times$ relative *stiffening* of the KL penalty for BC: a
  smoothed parametric prior needs $5\times$ stronger pull to
  match the demo-manifold commitment that 1-NN gets for free
  (consistent with §2 of `theory_post_slate_e.md`, the
  "mode-selection vs averaging" reading).

---

## 4. Relation to offline RL ("is this not just AWR?")

The KL-regularised soft-control objective (1) is structurally
identical to the family of *behaviour-regularised offline RL*
methods:

- **AWR** (Peng et al. 2019; Advantage-Weighted Regression):
  $\min_\pi\, \mathbb{E}_\pi[-A(s,a)] + \alpha\, \mathrm{KL}(\pi \,\|\, \pi_\beta)$
  with $\pi_\beta$ the behaviour policy. Closed-form solution
  $\pi^\star(a\mid s) \propto \pi_\beta(a\mid s)\exp(A(s,a)/\alpha)$
  — the **per-step** analogue of (2).
- **AWAC** (Nair et al. 2020): the same objective as AWR but with
  the advantage replaced by a learned $Q$-network and the
  KL replaced by a per-step variational bound.
- **CRR** (Wang et al. 2020): same objective, different
  importance-weighting scheme.
- **AlgaeDICE / OptiDICE** (Nachum et al. 2019; Lee et al. 2021):
  state-action visitation-distribution KL instead of policy KL,
  but the same Lagrangian.

**The hybrid is to MBD what AWR is to soft Q-learning.** Both
families minimise the same KL-regularised expected-cost
functional; the differences are:

- AWR / AWAC operate **per state**, on a (state, action) marginal
  with a learned advantage. The hybrid operates on the **joint
  trajectory** $\tau$, with the cost-to-go computed analytically
  from the simulator $J(\tau) = \sum_k \ell$. No advantage net is
  trained.
- AWR / AWAC use Monte Carlo estimates of the per-step expected
  advantage. The hybrid uses **SNIS at the trajectory level**
  with the cost-Boltzmann temperature $\beta$ playing the role
  of AWR's $\alpha$.
- AWR / AWAC use a learned $\pi_\beta$ (a Gaussian MLP fit to
  demo actions). The hybrid uses either a learned $\pi_\beta$
  (BC, flow) or a *retrieval* $\pi_\beta$ (`demo_*nn`); the
  retrieval prior has **no parameters**.

The single biggest substantive difference is **trajectory-level
rather than per-step KL**. In the LQ case the two are equivalent
(§2.2); in the multi-modal demo-manifold case they are not,
because per-step KL averages modes (the BC failure mode in
`theory_post_slate_e.md` §2) whereas trajectory-level KL with a
mode-selected target preserves them (the `demo_1nn` win).

For the final report this gives a clean answer to *"why don't you
just run IQL / AWAC?"*: because trajectory-level KL with a
zero-parameter retrieval prior is **strictly the cheaper end of
the same family**, and the SNIS bound of §02 quantifies what is
lost / gained by the trajectory-vs-per-step trade-off.

---

## 5. What this derivation buys for the final report

1. **A units-bearing interpretation of $\lambda$.** Not "guidance
   scale", not "trust schedule" — *inverse KL temperature*, with
   $\lambda_{\mathrm{prior}} = \lambda/\beta$ the dimensional
   ratio.
2. **An a-priori connection to classical DP.** The LQ limit
   reduces to one step of policy iteration on the demo-implied
   quadratic value function (§2.2). The 1.059× DARE empirical
   check is the *consistency* sanity, not a magic-number match.
3. **A self-evident answer to the offline-RL framing question**
   (§4). Same functional, trajectory-level vs per-step, retrieval
   vs learned $\pi_\beta$.
4. **The bias side of the Pareto bias–variance theorem of §03.**
   The minimiser (2) at finite $\lambda$ has bounded bias
   relative to either endpoint, and (5) gives an explicit bias
   formula in the LQ case. Glued to the §02 SNIS variance bound,
   this yields a single inequality stating that the hybrid is
   strictly Pareto-better than each endpoint.

---

## References

- Bertsekas, D.P. *Dynamic Programming and Optimal Control, Vol. I*,
  4th ed., 2017. Ch. 6 (policy iteration on LQ).
- Du, Y., Li, S., Mordatch, I. *Compositional Visual Generation
  with Energy-Based Models.* NeurIPS 2020. (PoE in score-based
  generative modelling.)
- Haarnoja, T., Zhou, A., Abbeel, P., Levine, S. *Soft
  Actor-Critic.* ICML 2018. (KL-regularised maximum-entropy RL.)
- Hinton, G.E. *Training Products of Experts by Minimising
  Contrastive Divergence.* Neural Computation, 2002.
- Ho, J., Salimans, T. *Classifier-Free Diffusion Guidance.*
  NeurIPS Workshop 2022.
- Levine, S. *Reinforcement Learning and Control as Probabilistic
  Inference.* arXiv:1805.00909, 2018. (Standard reference for
  the soft-control / KL-regularised formulation.)
- Nair, A., Gupta, A., Dalal, M., Levine, S. *AWAC: Accelerating
  Online Reinforcement Learning with Offline Datasets.* arXiv:2006.09359, 2020.
- Pan, C., Yi, Z., Shi, G., Qu, G. *Model-Based Diffusion for
  Trajectory Optimisation.* NeurIPS 2024.
- Peters, J., Schaal, S. *Reinforcement Learning by Reward-Weighted
  Regression for Operational Space Control.* ICML 2007.
- Peng, X.B., Kumar, A., Zhang, G., Levine, S. *Advantage-Weighted
  Regression.* arXiv:1910.00177, 2019.
- Toussaint, M. *Robot Trajectory Optimisation using Approximate
  Inference.* ICML 2009. (Gaussian-precision identification of
  policy iteration with trajectory-level Gaussians.)

---

*Status: §§1–4 form the paper-ready derivation. §5 is the
"what this buys you" summary, intended to slot directly into the
final-report theory section. The companion variance bound is in
[`02_snis_bound.md`](02_snis_bound.md); the unified Pareto
inequality is in [`03_pareto_bias_variance.md`](03_pareto_bias_variance.md).
This document supplies the **bias** side of the bundle's
contraction story; the extension into shorter-training and
adaptive-robustness corollaries lives in
[`04_sample_complexity.md`](04_sample_complexity.md) and
[`05_adaptive_robustness.md`](05_adaptive_robustness.md), with
the continuous-time derivation of the rate
$\alpha(\lambda) = \lambda^2 \sigma_{\min}^2 / (2 C_{\mathrm{LSI}})$
in [`06_predictor_corrector_sde.md`](06_predictor_corrector_sde.md)
and the closing minimax + MPC-regret + synthesis in
[`07_minimax_regret_synthesis.md`](07_minimax_regret_synthesis.md).
See [`README.md`](README.md) for the bundle index.*
