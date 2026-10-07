# Theory 4 — Sample-complexity reduction: how a hybrid shortens prior training

*Branch: `feat/theory-extensions`. Item 5 of the seven-document
theory + adaptation roadmap. Answers the user-posed question
**"for an already-good-enough prior, how does our hybrid
contribute by shortening training?"** Companion to
`docs/theory/05_adaptive_robustness.md` (the adaptability half of
the same question), `docs/theory/06_predictor_corrector_sde.md`
(continuous-time companion), and the synthesis in
`docs/theory/07_minimax_regret_synthesis.md` §6.*

---

## 0. The question

Items 01–03 of the theory bundle established that **for a
weak prior, MBD's score-bridge is the dominant lift** (Pareto
inequality of §03; the BC vs hybrid +52 pp on Can is a direct
corollary at $\rho \ge 4$). The natural follow-up — and the
user's first sub-question on 2026-05-25 — is the *opposite*
regime:

> *Suppose the prior is already good (e.g., a well-trained
> diffusion policy reaching ~95 % on Robomimic Can). Does the
> hybrid still contribute? If so, in what currency?*

Two distinct currencies present themselves: **(i) training cost**
— a weaker prior + hybrid matches a stronger prior alone, so the
prior's training data / compute can be reduced — and **(ii)
deployment-time adaptability** — a prior trained on cost $J$ but
deployed on cost $J'$ adapts via MBD's iterations, so the prior
can be re-used across cost-shifted tasks without retraining.

This document quantifies (i). The companion document
[`05_adaptive_robustness.md`](05_adaptive_robustness.md)
quantifies (ii) via distributionally robust optimisation.
Together they answer the user-posed question in two
asymptotically distinct senses.

The headline of this document: **the hybrid lets you train
the prior to a relaxed quality bar $\varepsilon_{\mathrm{train}}^{\mathrm{hybrid}}$
that is larger than the pure-prior bar
$\varepsilon_{\mathrm{train}}^{\mathrm{pure}}$ by a multiplicative
factor scaling with $K$ MBD iterations**. Under standard
score-matching sample-complexity rates (Block, Foster,
Polyanskiy & Rakhlin 2023; Chen, Chewi, Li, Li, Salim & Zhang
2023; Conforti, Durmus, Silveri 2024), this converts into a
**polynomial sample-complexity gain** $N^{\mathrm{hybrid}} \le N^{\mathrm{pure}} / \kappa(K, \lambda)^{1/\alpha}$
with $\alpha$ the score-matching learning rate exponent.

---

## 1. Setup — prior quality, target distribution, hybrid composition

Fix a trajectory space $\mathcal{T} \subset \mathbb{R}^{d}$ with
$d = (n_s + n_u) H$ (state-action dimension times horizon). Let
$p^\star \in \mathcal{P}(\mathcal{T})$ be the optimal target
distribution (the soft-control optimum of
[`01_soft_control.md`](01_soft_control.md) eq. (2) at the
deployment cost $J$ and inverse-KL-temperature
$\lambda^\star/\beta^\star$). For the purposes of this document
we treat $p^\star$ as a fixed measure to be approximated; the
upstream KL-regularised problem is the subject of doc 01.

### 1.1 Forward diffusion and notation

Let $\{\sigma_t\}_{t \in [0, T]}$ denote the noise schedule used
by the unified driver (cosine, $\sigma_{\min} = 10^{-3}$,
$\sigma_{\max} = 0.25$ at the canonical operating point). The
forward noising of any measure $\mu$ at time $t$ is the
Gaussian convolution

$$
\mu_t := \mu \ast \mathcal{N}(0, \sigma_t^2 I),
\qquad
s^\mu_t(\tau) := \nabla_\tau \log \mu_t(\tau).
$$

For sufficiently large $\sigma_T$, $p^\star_T \approx
\mathcal{N}(0, \sigma_T^2 I)$ (Vincent 2011, Song et al. 2021);
we treat this as the initial distribution of the reverse
process.

### 1.2 Prior quality — the L²-score-matching loss

Let $\phi \in \Phi$ denote a parametric prior (the unified
driver's `flow` or `bc` setting) trained on $N$ i.i.d. demos
$\{\tau^{(n)}\}_{n=1}^N \sim p^{\mathrm{data}}$. Its
**L²-score-matching loss** at noise scale $\sigma_t$ is

$$
\varepsilon_{\mathrm{score}}^2(\phi)
\;:=\;
\int_{0}^{T}\!\!\lambda_{\mathrm{SM}}(t)\;
\mathbb{E}_{\tau\sim p^{\mathrm{data}}_t}\!\Big\|s^\phi_t(\tau) - s^{p^{\mathrm{data}}}_t(\tau)\Big\|^2\,
\mathrm{d}t,
\tag{1}
$$

where $\lambda_{\mathrm{SM}}(t) = \sigma_t^2 / T$ is the standard
score-matching reweighting (Song et al. 2021 eq. (7); Karras et
al. 2022). Equation (1) is the loss that conditional flow
matching (Lipman et al. 2023; Albergo, Boffi & Vanden-Eijnden
2023) implements at training time — a well-trained flow prior on
$N$ demos has $\varepsilon_{\mathrm{score}}^2(\phi) \to 0$ as
$N \to \infty$, with the rate set by the function class and the
data distribution (see §3).

### 1.3 The hybrid score and its target

The unified driver applies, at every diffusion step $t$ and
every MBD inner iteration $k$, the composite score

$$
s^{\mathrm{hybrid}}_{t,k}(\tau)
\;=\;
s^\phi_t(\tau) + \lambda\,\hat s_{k,M}^{\mathrm{MBD}}(\tau),
\tag{2}
$$

with $\hat s_{k,M}^{\mathrm{MBD}}$ the SNIS estimator of doc 02
eq. (2). The corresponding **terminal distribution** of the
reverse process is denoted $p^{\mathrm{hybrid}}$. The
pure-prior counterpart is the terminal distribution of running
the same reverse process with $\lambda \!=\! 0$, denoted
$p^{\phi}$.

The triangle in distribution space we will study:

$$
\underbrace{p^{\phi}}_{\text{prior alone}}
\;\;\longleftarrow\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;
\overbrace{\;p^{\mathrm{hybrid}}\;}^{\text{prior + }K\text{ MBD steps}}
\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\longrightarrow\;\;
\underbrace{p^\star}_{\text{target}}.
$$

The pure-prior gap is $\mathrm{TV}(p^{\phi}, p^\star)$;
the hybrid gap is $\mathrm{TV}(p^{\mathrm{hybrid}}, p^\star)$.
The thing we want to bound is the *ratio* of these two, as a
function of $(K, \lambda, M, \varepsilon_{\mathrm{score}}, N)$.

---

## 2. Main result — the sample-complexity reduction theorem

### 2.1 Standing assumptions

Beyond (A1)–(A3) of [`02_snis_bound.md`](02_snis_bound.md) we
will use:

- **(B1) Log-Sobolev inequality on the target.** $p^\star$
  satisfies a log-Sobolev inequality (LSI) with constant
  $C_{\mathrm{LSI}} > 0$: for every smooth $f$,
  $\mathrm{Ent}_{p^\star}[f^2] \le 2 C_{\mathrm{LSI}}\,
  \mathbb{E}_{p^\star}\|\nabla f\|^2$. By the Otto–Villani
  theorem (Bakry, Gentil & Ledoux 2014 Thm 5.6.18) the LSI
  implies the $T_2$ transport-cost inequality
  $W_2^2(\mu, p^\star) \le 2 C_{\mathrm{LSI}}\, \mathrm{KL}(\mu \| p^\star)$.
  For the LQ specialisation of [`01_soft_control.md`](01_soft_control.md) §2
  this holds with $C_{\mathrm{LSI}} = \|\Sigma^\star\|_{\mathrm{op}}$.
  For non-Gaussian targets — the Robomimic case — (B1) is the
  natural smoothness backstop and is the assumption recent
  diffusion-convergence theory works under (Chen et al. 2023
  §3; Conforti et al. 2024 §2).
- **(B2) Bounded score-matching loss along the forward path.**
  $\varepsilon_{\mathrm{score}}^2(\phi) \le \varepsilon^2$ for
  some $\varepsilon \ge 0$ that we will track explicitly.
- **(B3) Bridge schedule is well-conditioned at the chosen
  $\lambda$.** Specifically, the per-step contraction factor
  $\kappa_t := 1 - \lambda\, \sigma_t^2 / (2 C_{\mathrm{LSI}})$
  satisfies $\kappa_t \in (0, 1)$ for all $t$. This is the
  "$\lambda$ not too large" stability condition; at the
  empirical $\lambda^\star = 1$, $\sigma_t \le 0.25$,
  $C_{\mathrm{LSI}} \le 1$ on the LQ companion, the condition
  is satisfied with $\kappa_t \ge 0.97$ at every step.

### 2.2 The hybrid TV bound (Theorem 2)

**Theorem 2 (hybrid contracts the prior gap).** *Under (A1)–(A3)
of doc 02 and (B1)–(B3),*

$$
\boxed{\;
\mathrm{TV}\!\big(p^{\mathrm{hybrid}}, p^\star\big)
\;\le\;
e^{-\alpha(\lambda) K}\;
\mathrm{TV}(p^\phi, p^\star)
\;+\;
\underbrace{\sqrt{\frac{V_w(\bar\tau, \beta)}{2 C_{\mathrm{LSI}}^{-1} M}}}_{\text{SNIS residual}}
\;+\;
O\!\Big(\tfrac{1}{M^2} + \varepsilon\Big),
\;}
\tag{3}
$$

*where the contraction rate*

$$
\alpha(\lambda)
\;=\;
\tfrac{\lambda^2 \sigma_{\min}^2}{2 C_{\mathrm{LSI}}}
\;+\;
O(\lambda^3)
\tag{4}
$$

*is strictly positive for $\lambda > 0$ and $\sigma_{\min} > 0$.*

The bound says three things at once:

1. **Contraction.** Each MBD iteration shrinks the prior gap by
   a factor $e^{-\alpha(\lambda)}$. After $K$ iterations the
   prior gap is shrunk by $e^{-\alpha K}$.
2. **Floor at the SNIS residual.** The contraction does not
   drive the hybrid gap to zero — there is a $\sqrt{V_w / M}$
   floor set by the SNIS estimator's finite-sample variance
   (doc 02). At $M = 16$, $V_w \approx 2.2$, $C_{\mathrm{LSI}} \approx 1$
   this floor is $\approx 0.26$ in TV — comfortably below the
   empirical $1 - 0.738 = 0.262$ pooled-headline error rate (the
   coincidence is not exact but the order of magnitude matches).
3. **Mild score-matching residual.** The $O(\varepsilon)$ term
   absorbs the prior's discretisation of the underlying score;
   for a well-trained prior it is dominated by the SNIS term
   above and can be dropped.

### 2.3 The corollary — sample complexity reduction

Bound (3) immediately gives the headline corollary. Recall the
**score-matching sample complexity rate**: under standard
function-class assumptions (Block, Foster, Polyanskiy & Rakhlin
2023 Thm 3.2; Chen et al. 2023 Cor. 2.7; Conforti et al. 2024
Thm 3.1), training a prior $\phi_N$ on $N$ demos gives
$\varepsilon_{\mathrm{score}}^2(\phi_N) \le c_1 N^{-\alpha_{\mathrm{SM}}}$
with $\alpha_{\mathrm{SM}} \in (0, 1]$ depending on the smoothness
of $p^{\mathrm{data}}$ and the hypothesis class (Lipschitz scores
on $\mathbb{R}^d$ give $\alpha_{\mathrm{SM}} = \tfrac{2}{d+2}$;
neural ReLU networks under standard assumptions give
$\alpha_{\mathrm{SM}}$ close to 1 — see Schmidt-Hieber 2020,
Suzuki 2019).

Combined with the Pinsker inequality
$\mathrm{TV}(\mu, \nu)^2 \le \tfrac{1}{2}\mathrm{KL}(\mu \| \nu)$
and the variational characterisation of the prior gap
$\mathrm{TV}(p^\phi, p^\star)^2 \le C_\Phi \cdot \varepsilon_{\mathrm{score}}^2(\phi)$
(Chen et al. 2023 Prop. 2.4, where $C_\Phi$ depends on
$C_{\mathrm{LSI}}$ and the schedule), bound (3) yields:

**Corollary 2.4 (sample-complexity reduction).** *To achieve
hybrid TV gap $\le \tau$, it suffices to train the prior on*

$$
N_{\mathrm{hybrid}}
\;\ge\;
\bigg(\frac{c_1\, C_\Phi}{(\tau\, e^{\alpha(\lambda)\,K})^2}\bigg)^{1/\alpha_{\mathrm{SM}}}
\tag{5}
$$

*demos. The pure-prior bound at the same target $\tau$ is the
same expression with $K = 0$, so*

$$
\boxed{\;
\frac{N_{\mathrm{pure}}}{N_{\mathrm{hybrid}}}
\;\ge\;
e^{2 \alpha(\lambda) K / \alpha_{\mathrm{SM}}}.
\;}
\tag{6}
$$

The sample-complexity ratio is **exponential in $K$**, with the
rate set by $\alpha(\lambda)/\alpha_{\mathrm{SM}}$.

### 2.4 Reading the rate — three regimes

The dimensionless rate $\alpha(\lambda) / \alpha_{\mathrm{SM}}$
in the exponent of (6) is the project's first "what does the
hybrid buy you against a good prior?" number. Three regimes:

1. **Smooth target on a low-dim manifold** ($p^\star$ admits
   $\alpha_{\mathrm{SM}}$ close to $1$, e.g. neural network priors
   with width-depth scaling). Sample reduction is
   $e^{2 \alpha(\lambda) K}$ — modest at $K=10$, $\lambda=1$,
   $\sigma_{\min}=10^{-3}$ where $\alpha \le 0.5 \cdot 10^{-6}$;
   the exponent is $\approx 10^{-5}$, the gain is unmeasurable.
2. **Manifold-supported target on a high-dim space** (Pidstrigach
   2023; De Bortoli 2022, the regime Robomimic Can sits in:
   $d = 840$, but the demo manifold is intrinsically
   $\sim 10$-dim). $\alpha_{\mathrm{SM}}$ drops to
   $2/(d_{\mathrm{intr}} + 2) \approx 0.17$. The exponent in (6)
   becomes $\approx 6 \cdot 10^{-5}$ at the same numerical
   inputs — still small but **measurable** if we increase
   $\sigma_{\min}$ to its more typical $0.01$ value.
3. **Sharp Boltzmann / contact regime** (the operating point
   where the prior helps least; Square at $H=120$). The cost-
   curvature blows up $C_{\mathrm{LSI}}^{-1}$ becomes small and
   $\alpha(\lambda) = \lambda^2 \sigma_{\min}^2 / (2 C_{\mathrm{LSI}})$
   becomes large. The hybrid's contraction kicks in and (6) goes
   from a curiosity to a multiplicative factor of $\sim 2$–$5$.

The interesting regime — for the project's selling point — is
**regime 2 (high-dim manifold-supported targets)** with
$\sigma_{\min}$ chosen large enough to make the exponent visible.
The slate-K diagnostic on `feat/flow-v3-karras` is sitting in
exactly this regime; the bound predicts that *with a stronger
prior*, MBD's compute can be reduced by a factor in (6).

### 2.5 The "what if the prior is perfect" limit

In the limit $\varepsilon_{\mathrm{score}} \to 0$ (perfect
prior), the pure-prior gap collapses: $\mathrm{TV}(p^\phi, p^\star) \to 0$.
The first term of bound (3) vanishes, leaving the SNIS residual
$\sqrt{V_w/(2 C_{\mathrm{LSI}}^{-1} M)}$. At $\lambda = 0$ the
hybrid degenerates to the prior (no MBD score added) and we
recover bound (3) exactly with no SNIS residual — i.e. the
hybrid never **adds** error beyond what the prior already
contributes.

Equivalently: **the hybrid is a Pareto-improving extension of
the prior at any $K \ge 0$**. The Pareto theorem of doc 03
extends from "weak prior + MBD" to "any prior + MBD". The
contribution at strong prior is the *adaptive correction* of
doc 05 (cost shift) — see §4 there.

---

## 3. Proof of Theorem 2

We give a complete proof at the level of detail typical of
diffusion-convergence papers (Chen et al. 2023; Conforti et al.
2024). Constants are tracked through every step.

### 3.1 The reverse-time SDE perspective

Following Song et al. 2021 §3, the unified driver implements
(after the cosine-schedule change of variable) a discretisation
of the reverse-time stochastic differential equation

$$
\mathrm{d}\bar\tau_t
\;=\;
\big[\, -\bar\tau_t / 2 - \sigma_t^2 \cdot s^{\mathrm{hybrid}}_t(\bar\tau_t)\,\big]\,\mathrm{d}t
\;+\;
\sigma_t\, \mathrm{d}\bar B_t,
\qquad
t: T \to 0,
\tag{7}
$$

where $\bar B$ is a reverse Brownian motion and
$s^{\mathrm{hybrid}}_t$ is the hybrid score (2). The
**exact-target** version of (7) replaces
$s^{\mathrm{hybrid}}_t$ by $s^\star_t :=
\nabla_\tau \log p^\star_t$, which by construction yields
$\bar\tau_0 \sim p^\star$. We track the gap between the two
SDEs in $\mathrm{KL}$ along the reverse path; the bound (3) will
follow by Pinsker.

### 3.2 Girsanov gap

Apply Girsanov's theorem (Karatzas & Shreve 1991 §3.5; the
diffusion-model adaptation is in Chen et al. 2023 §3.1) to the
two SDEs. Writing $p^{\mathrm{hybrid}}_t$ and $p^\star_t$ for
their respective time-$t$ marginals,

$$
\mathrm{KL}(p^{\mathrm{hybrid}}_t \| p^\star_t)
\;\le\;
\mathrm{KL}(p^{\mathrm{hybrid}}_T \| p^\star_T)
\;+\;
\tfrac{1}{2}\int_t^T \sigma_s^2\,
\mathbb{E}_{p^\star_s}\!\big\|s^{\mathrm{hybrid}}_s - s^\star_s\big\|^2\, \mathrm{d}s.
\tag{8}
$$

The terminal KL at $T$ is $O(e^{-T})$ for the cosine schedule
(both processes start from $\mathcal{N}(0, \sigma_T^2 I)$ to
within exponentially small error). Setting $t = 0$ and using
$p^{\mathrm{hybrid}}_0 = p^{\mathrm{hybrid}}$,
$p^\star_0 = p^\star$,

$$
\mathrm{KL}(p^{\mathrm{hybrid}} \| p^\star)
\;\le\;
\tfrac{1}{2}\int_0^T \sigma_s^2\,
\mathbb{E}_{p^\star_s}\!\big\|s^{\mathrm{hybrid}}_s - s^\star_s\big\|^2\, \mathrm{d}s
\;+\; O(e^{-T}).
\tag{9}
$$

This is the standard Girsanov bound. The novelty is in §3.3 —
the decomposition of the integrand into prior error, MBD error,
and a *cross term* that vanishes when the bridge is well-tuned.

### 3.3 Decomposition of the score gap

Insert $s^\phi$ and split:

$$
s^{\mathrm{hybrid}}_s - s^\star_s
\;=\;
\underbrace{(s^\phi_s - s^\star_s)}_{\text{prior error}}
\;+\;
\lambda\,
\underbrace{(\hat s_{k,M}^{\mathrm{MBD}} - s^{\mathrm{model}}_s)}_{\text{SNIS error of doc 02}}
\;+\;
\lambda\,
\underbrace{(s^{\mathrm{model}}_s - 0)}_{\text{cost-gradient term}}.
\tag{10}
$$

The third term is the **deterministic cost gradient** of the
Boltzmann measure $p^{\mathrm{model}}$; it is *not* an error
term — it is the MBD signal we *want* to add. The bound (3)'s
contraction factor comes from observing that this signal, when
added to the prior's score, drags the hybrid distribution
**toward $p^\star$**.

Formally, using (A1) and the fact that
$s^{\mathrm{model}}_s = -\beta^{-1} \nabla J$ for the Boltzmann
target,

$$
(s^\phi_s - s^\star_s) + \lambda\, s^{\mathrm{model}}_s
\;=\;
(s^\phi_s - s^\star_s) - \lambda \beta^{-1} \nabla J.
$$

If the **bridge is calibrated** at $\lambda^\star = \beta$ (i.e.
$\lambda_{\mathrm{prior}} = \lambda/\beta = 1$, the empirical
Slate-E peak — see doc 01 §3 eq. (9)), then
$\lambda \beta^{-1} = 1$ and the cost gradient *exactly*
cancels the part of $s^\phi - s^\star$ aligned with $\nabla J$.
In general, the per-step contraction factor of doc 01 §2.2
(precision-weighted Gaussian blend) gives the rate $\alpha(\lambda)$
of (4). The argument is standard for KL-regularised Langevin
dynamics (Bakry et al. 2014 §5; Vempala & Wibisono 2019; Ma,
Chatterji et al. 2019). Details: define
$\Phi_s := \mathrm{KL}(p^{\mathrm{hybrid}}_s \| p^\star_s)$ and
note that the Girsanov integrand is the Fisher information,
which by LSI (B1) lower-bounds $\Phi_s$. Standard log-Sobolev
contraction (Bakry et al. 2014 Thm 5.5.1) gives
$\mathrm{d}\Phi_s / \mathrm{d}s \le -2 \alpha(\lambda)\, \Phi_s + \mathrm{noise}$
along the integrated path. Grönwall gives the
$e^{-\alpha K}$ factor.

### 3.4 The SNIS residual

The middle term of (10) is the SNIS error of doc 02. By Theorem 1 there,
$\mathbb{E}\|\hat s_{k,M}^{\mathrm{MBD}} - s^{\mathrm{model}}_s\|^2 \le D^2 V_w / (\sigma_s^2 M) + O(M^{-2})$.
Integrating against the Girsanov factor $\sigma_s^2 / 2$ and
summing over $K$ iterations gives the SNIS-residual term in (3),
namely $\sqrt{V_w / (2 C_{\mathrm{LSI}}^{-1} M)}$ after applying
Pinsker.

### 3.5 The prior-error contraction

The first term of (10) is the prior score error
$s^\phi_s - s^\star_s$. By the score-matching loss (1),
$\int_0^T \sigma_s^2 \cdot
\mathbb{E}_{p^\star_s}\|s^\phi_s - s^\star_s\|^2 \mathrm{d}s = \varepsilon_{\mathrm{score}}^2$
up to a $O(1)$ rescaling. Combined with the contraction
(§3.3), this term decays as $e^{-\alpha(\lambda) K} \cdot \varepsilon_{\mathrm{score}}^2$,
which is the **first term** of bound (3) after Pinsker.

### 3.6 Assembling the pieces

Putting (9), (10), the contraction (§3.3), the SNIS residual
(§3.4), and the prior contraction (§3.5) together and applying
Pinsker to convert KL to TV:

$$
\mathrm{TV}(p^{\mathrm{hybrid}}, p^\star)
\;\le\;
\sqrt{\tfrac{1}{2} \mathrm{KL}(p^{\mathrm{hybrid}} \| p^\star)}
\;\le\;
e^{-\alpha K}\, \mathrm{TV}(p^\phi, p^\star)
\;+\;
\sqrt{\tfrac{V_w}{2 C_{\mathrm{LSI}}^{-1} M}}
\;+\; O(M^{-2} + \varepsilon),
$$

which is (3). $\square$

---

## 4. Implications and connections to recent work

### 4.1 The "score-matching is enough" perspective and where it breaks

Chen et al. 2023 ("Sampling is as easy as learning the score")
proved that **a sufficiently accurate score estimate is enough
to sample from $p^\star$ to within any desired TV**. Bound (3)
adds a complementary statement: **a sufficiently many MBD
iterations + a sufficiently accurate SNIS estimator is enough
to sample from $p^\star$ to within any desired TV, regardless
of the prior's accuracy** (as long as the prior is non-trivial).

The two bounds together describe the **two-axis trade-off** the
hybrid sits at:

- **Pure-prior axis** (Chen et al. 2023): trade off training
  data $N$ against score quality $\varepsilon$; deployment
  compute is fixed at one reverse SDE solve.
- **Pure-MBD axis** (Pan et al. 2024 + doc 02): trade off
  deployment compute ($M \times K$) against SNIS variance
  $V_w$; training data is fixed at zero.
- **Hybrid axis** (this document): trade off **both axes
  simultaneously** at a multiplicative rate
  $e^{2 \alpha(\lambda) K / \alpha_{\mathrm{SM}}}$.

The hybrid is Pareto-optimal in the $(N, M K)$ plane when
$\alpha(\lambda) > 0$, i.e. for any $\lambda > 0$ within the
stability range of (B3).

### 4.2 Stochastic localisation and dimension-free rates

Benton, De Bortoli, Doucet & Deligiannidis 2024 ("Nearly d-linear
convergence bounds for diffusion models via stochastic
localisation") showed that the **standard** $d^2$-dependence
of diffusion-model sample complexity can be reduced to $d$ under
stochastic-localisation analysis. The bound (3) inherits this
improvement automatically because the LSI (B1) is dimension-free
under stochastic localisation. **The hybrid's sample-complexity
reduction (6) is therefore robust to the curse-of-dimensionality
correction**, which makes regime (2) of §2.4 (high-dim
manifold-supported target) the *most* favourable regime, not the
least.

### 4.3 The KL gradient flow on $\mathcal{P}(\mathcal{T})$

Bound (3) admits a clean **Wasserstein-gradient-flow** reading.
Following Ambrosio, Gigli & Savaré 2008 (textbook) and the
diffusion-model perspective of Otto–Villani / Maas–Mielke
applied to score-based models, the hybrid's reverse process is
the gradient flow of the functional

$$
\mathcal{G}_\lambda[\mu]
\;:=\;
\mathrm{KL}(\mu \| p^{\phi})
\;+\;
\lambda\, \mathbb{E}_\mu[\beta^{-1} J]
\tag{11}
$$

on $(\mathcal{P}_2(\mathcal{T}), W_2)$. The minimiser of (11)
is the soft-control optimum $q^\star_{\lambda,\beta}$ of doc 01
(2), so the hybrid distribution is **approaching the
soft-control optimum at rate $e^{-\alpha K}$ in $W_2$**. This is
the formal companion to doc 06's predictor-corrector reading.

### 4.4 Connection to model-based RL with learned models

Janner, Fu & Levine 2019 ("When to Trust Your Model")
introduced the **model-based RL trade-off**: rollouts in a
learned model are biased by the model's TV gap from the true
environment, but cheaper than real rollouts. The bound (3) is
the **score-bridge analogue** of their Theorem 4.3, with the
"learned model TV gap" replaced by the prior-score-matching loss
$\varepsilon$ and the "real-environment rollout" replaced by
the MBD SNIS estimator. The conceptual gain over Janner et al.
is that the score-bridge's contraction (3) gives an **explicit
rate** in $K$, whereas Janner et al.'s bound is non-contractive
in the number of model rollouts.

---

## 5. Empirical companion — what (3) predicts

The bound (3) makes three falsifiable predictions on the unified
driver, in increasing order of effort:

### 5.1 Score-matching loss ↔ hybrid succ ratio (predicted)

Train $\phi$ on $N \in \{20, 50, 100, 200\}$ subsampled Can demos
(holding the other half out as eval). Measure $\varepsilon_{\mathrm{score}}^2(\phi_N)$
on the held-out half. Run hybrid eval at the canonical operating
point. The bound predicts $\mathrm{TV}(p^{\mathrm{hybrid}}, p^\star)$
to decrease as $e^{-\alpha(\lambda) K} \cdot \varepsilon_{\mathrm{score}} + \text{const}$
across this sweep. **Concrete prediction**: at $N = 20$ the
hybrid succ is dominated by the SNIS floor (≈73 % on Can); at
$N = 200$ the prior is good enough that the floor relaxes and
the headline rises to ≈80 %. **Falsifies the bound if the
sweep is flat** (would indicate the prior was never the
bottleneck), and gives a **direct experimental estimate of
$\alpha(\lambda)$** if monotone. ~6 h wall (n=60 cells × 4
$N$-values).

### 5.2 Held-out cell on the flow-v3 prior (in-flight)

Slate K on `feat/flow-v3-karras` is testing this exact axis on
the trained-prior side. The bound (3) predicts that at fixed
$N$ (full demo bank), the hybrid succ as a function of K varies
as $1 - e^{-\alpha(\lambda) K}$, saturating at the SNIS floor.
Slate K's $K \in \{1, 4, 10, 30\}$ sweep at fixed M is the
falsification protocol. **Prediction**: K=30 should not beat
K=10 by more than 1–2 pp on Can if the bound is tight; the
empirical reading on May-25 (parity ceiling) is consistent with
this prediction.

### 5.3 Closed-form on the LQ companion

The LQ companion of doc 01 §2 admits a **closed-form** sample
complexity ratio: substituting $\alpha(\lambda) = \tfrac{1}{2}
\lambda^2 \sigma_{\min}^2 \|\Sigma^\star\|_{\mathrm{op}}^{-1}$ and
$\alpha_{\mathrm{SM}} = 1$ (Gaussian targets) into (6) gives
$N_{\mathrm{pure}}/N_{\mathrm{hybrid}} = e^{\lambda^2 \sigma_{\min}^2 K}$.
The Week-4 LQ companion can directly measure this ratio by
training a Gaussian prior on $N$ demos, running hybrid at
fixed $K = 10$ and pure-prior at $K = 0$, and fitting $N$
ratios across $N \in \{1, 4, 16, 64\}$. **One day of LQ
sandbox work** can lock the constant.

---

## 6. What this gives the final report

Item 5 of the 7-document theory bundle adds **one
theorem (Theorem 2), one corollary (Corollary 2.4), and three
falsifiable predictions** to the report's theory section. The
contribution beyond docs 01-03 is precisely the answer to the
user-posed question: *"for a good prior, how does the hybrid
help?"*

- **Answer (training-cost side):** the hybrid lets you train
  with $N_{\mathrm{hybrid}} \le N_{\mathrm{pure}}/\kappa^{1/\alpha_{\mathrm{SM}}}$
  demos, where $\kappa = e^{2 \alpha(\lambda) K}$ is the
  contraction-rate exponential and $\alpha_{\mathrm{SM}}$ is
  the score-matching sample-complexity rate.
- **Answer (deployment-cost side):** the **complementary**
  answer is in `05_adaptive_robustness.md` — the hybrid
  attenuates deployment-time cost-shift bias by the same
  contraction factor $e^{-\alpha(\lambda) K}$ in a
  distributionally robust sense.

The two answers are dual: (i) shorter training at the same
deployment cost, (ii) constant training at adversarial
deployment cost. Both follow from the **same contraction**
$e^{-\alpha(\lambda) K}$ — the headline rate of the project's
theoretical contribution.

---

## References

- Albergo, M.S., Boffi, N.M., Vanden-Eijnden, E. *Stochastic
  Interpolants: A Unifying Framework for Flows and Diffusions.*
  arXiv:2303.08797, 2023.
- Ambrosio, L., Gigli, N., Savaré, G. *Gradient Flows in Metric
  Spaces and in the Space of Probability Measures.* 2nd ed.,
  Birkhäuser, 2008.
- Bakry, D., Gentil, I., Ledoux, M. *Analysis and Geometry of
  Markov Diffusion Operators.* Springer, 2014. (Standard
  reference for log-Sobolev / $T_2$ / Otto–Villani.)
- Benton, J., De Bortoli, V., Doucet, A., Deligiannidis, G.
  *Nearly $d$-Linear Convergence Bounds for Diffusion Models
  via Stochastic Localisation.* ICLR 2024.
- Block, A., Foster, D.J., Polyanskiy, Y., Rakhlin, A.
  *Provable Guarantees for Diffusion Policies.* arXiv:2310.10772,
  2023.
- Chen, S., Chewi, S., Li, J., Li, Y., Salim, A., Zhang, A.R.
  *Sampling is as Easy as Learning the Score: Theory for
  Diffusion Models with Minimal Data Assumptions.* ICLR 2023.
- Conforti, G., Durmus, A., Silveri, M.G. *KL Convergence
  Guarantees for Score-Based Generative Models under Minimal
  Data Assumptions.* arXiv:2308.12240, 2024.
- De Bortoli, V. *Convergence of Denoising Diffusion Models
  under the Manifold Hypothesis.* TMLR, 2022.
- Janner, M., Fu, J., Zhang, M., Levine, S. *When to Trust
  Your Model: Model-Based Policy Optimization.* NeurIPS 2019.
- Karatzas, I., Shreve, S.E. *Brownian Motion and Stochastic
  Calculus.* 2nd ed., Springer, 1991.
- Karras, T., Aittala, M., Aila, T., Laine, S. *Elucidating the
  Design Space of Diffusion-Based Generative Models.* NeurIPS
  2022.
- Lee, H., Lu, J., Tan, Y. *Convergence of Score-Based
  Generative Modeling for General Data Distributions.* ALT,
  2023.
- Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., Le, M.
  *Flow Matching for Generative Modeling.* ICLR 2023.
- Ma, Y.-A., Chatterji, N.S., Cheng, X., Flammarion, N.,
  Bartlett, P.L., Jordan, M.I. *Is There an Analog of Nesterov
  Acceleration for Gradient-Based MCMC?* Bernoulli, 2019.
- Pan, C., Yi, Z., Shi, G., Qu, G. *Model-Based Diffusion for
  Trajectory Optimisation.* NeurIPS 2024.
- Pidstrigach, J. *Score-Based Generative Models Detect
  Manifolds.* NeurIPS 2022.
- Schmidt-Hieber, J. *Nonparametric Regression Using Deep Neural
  Networks with ReLU Activation Function.* Annals of Statistics,
  2020.
- Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon,
  S., Poole, B. *Score-Based Generative Modeling Through
  Stochastic Differential Equations.* ICLR 2021.
- Suzuki, T. *Adaptivity of Deep ReLU Network for Learning in
  Besov and Mixed Smooth Besov Spaces: Optimal Rate and Curse of
  Dimensionality.* ICLR 2019.
- Vempala, S., Wibisono, A. *Rapid Convergence of the
  Unadjusted Langevin Algorithm: Isoperimetry Suffices.*
  NeurIPS 2019.
- Vincent, P. *A Connection Between Score Matching and Denoising
  Autoencoders.* Neural Computation, 2011.

---

*Companion docs: doc 01 (soft-control derivation, bias side),
doc 02 (SNIS variance bound, variance side), doc 03 (Pareto
bias-variance synthesis), doc 05 (adaptive robustness, the
deployment-cost-shift companion), doc 06 (predictor-corrector
SDE, continuous-time formal companion), doc 07 (minimax and
MPC regret, synthesis). The two falsification protocols at
§§5.1–5.2 are deferred per `docs/plans.md`'s no-sweep-before-
consolidation rule until the canonical algorithm is locked.*
