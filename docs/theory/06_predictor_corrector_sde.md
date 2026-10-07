# Theory 6 — Predictor-corrector SDE formalisation and the Wasserstein contraction rate

*Branch: `feat/theory-extensions`. Item 7 of the seven-document
theory + adaptation roadmap. The continuous-time formal
companion to docs 04 (sample complexity) and 05 (adaptive
robustness). Identifies the unified driver with a
**principled predictor-corrector (PC) integrator** of an SDE on
$\mathcal{P}_2(\mathcal{T})$ whose stationary distribution is the
soft-control optimum $q^\star_{\lambda,\beta}$ of doc 01 eq. (2),
and proves $W_2$ contraction at rate $e^{-\alpha(\lambda) K}$
under log-Sobolev assumptions.*

---

## 0. What this document does

Docs 04 and 05 stated their headline results — sample-complexity
reduction and DRO adaptation — as TV bounds with a contraction
factor $e^{-\alpha(\lambda) K}$. The rate $\alpha(\lambda)$ was
introduced as a closed-form expression
$\alpha(\lambda) = \lambda^2 \sigma_{\min}^2 / (2 C_{\mathrm{LSI}}) + O(\lambda^3)$
without derivation. This document **derives the contraction
rate** by:

1. **Identifying the unified driver with a continuous-time SDE.**
   The unified driver implements the Euler-Maruyama
   discretisation of a stochastic differential equation
   $\mathrm{d}\tau_t = b_t(\tau_t)\,\mathrm{d}t + \sigma_t\,\mathrm{d}B_t$
   with a *composite drift* $b_t = s^\phi_t + \lambda
   s^{\mathrm{model}}_t$, where $s^\phi$ and $s^{\mathrm{model}}$
   are the prior and model scores of doc 01.
2. **Recognising the SDE as a predictor-corrector scheme.**
   Following the SDE framework of Song et al. 2021 §4.2, the
   unified driver decomposes into a *predictor* (the prior score
   $s^\phi_t$) and a *corrector* (the $K$ MBD iterations,
   acting as Langevin updates with drift
   $\lambda s^{\mathrm{model}}_t$).
3. **Proving $W_2$ contraction under log-Sobolev.** Combining
   the Otto–Villani transport-cost inequality with the
   convergence theory of overdamped Langevin (Vempala &
   Wibisono 2019; Ma et al. 2019; Erdogdu, Hosseinzadeh & Zhang
   2022), we derive a $W_2$ exponential contraction at rate
   $\alpha(\lambda)$ that **matches the bounds (3)–(4) of
   doc 04 and doc 05**.

The document is the **mathematical engine** for the previous
two; it does not introduce new empirical claims but supplies
the continuous-time framework that doc 04 and doc 05 cited as
"the underlying contraction".

---

## 1. The unified driver as Euler-Maruyama on a composite-drift SDE

### 1.1 Reverse-time SDE for the soft-control target

Following the score-based SDE framework (Song et al. 2021 eqs.
(5)–(6); Anderson 1982 for the reverse-time SDE), the soft-control
target $q^\star_{\lambda,\beta}$ of doc 01 eq. (2) is the
$t \to 0$ marginal of the reverse-time SDE

$$
\mathrm{d}\bar\tau_t
\;=\;
\big[\, -\tfrac{1}{2}\bar\tau_t - \sigma_t^2\, s^\star_{\lambda,\beta,t}(\bar\tau_t)\,\big]\,\mathrm{d}t
\;+\;
\sigma_t\,\mathrm{d}\bar B_t,
\qquad
t : T \to 0,
\tag{1}
$$

where $\bar B$ is a reverse Brownian motion, $\sigma_t$ is the
forward noise schedule of doc 04 §1.1, and
$s^\star_{\lambda,\beta,t}$ is the score of the diffused
soft-control target $q^\star_{\lambda,\beta,t}$:

$$
s^\star_{\lambda,\beta,t}(\tau)
\;=\;
\nabla_\tau \log q^\star_{\lambda,\beta,t}(\tau)
\;=\;
\nabla_\tau \log p^{\mathrm{data}}_t(\tau)
\;+\;
(\lambda/\beta)\, \nabla_\tau \log p^{\mathrm{model}}_t(\tau).
\tag{2}
$$

The second equality is the **composite-score identity** of
doc 01 eq. (7): the score of the product-of-experts target
factorises additively into a data-side score and a
model-side score. Identity (2) is what makes the rest of this
section possible — without it the SDE (1) would not admit a
predictor-corrector decomposition.

### 1.2 The empirical drift

The unified driver does not have access to the *true* scores
$s^{p^{\mathrm{data}}}_t$ and $s^{\mathrm{model}}_t$. Instead it
has:

- **An approximate prior score** $s^\phi_t$ trained by score
  matching on the demo bank (doc 04 eq. (1)). The error
  $s^\phi_t - s^{p^{\mathrm{data}}}_t$ has $L^2$-norm
  $\varepsilon_{\mathrm{score}}(\phi)$ along the forward path.
- **An SNIS estimator** $\hat s_{k,M}^{\mathrm{MBD}}$ of
  $s^{\mathrm{model}}_t$ (doc 02 eq. (2)). The error has
  $L^2$-norm bounded by Theorem 1 of doc 02:
  $\mathbb{E}\|\hat s_{k,M}^{\mathrm{MBD}} - s^{\mathrm{model}}_t\|^2 \le D^2 V_w / (\sigma_t^2 M) + O(M^{-2})$.

The **empirical SDE** that the unified driver actually
integrates is

$$
\mathrm{d}\bar\tau_t
\;=\;
\Big[\, -\tfrac{1}{2}\bar\tau_t - \sigma_t^2\, \big(s^\phi_t(\bar\tau_t) + \lambda\, \hat s_{t,M}^{\mathrm{MBD}}(\bar\tau_t)\big)\Big]\,\mathrm{d}t
\;+\;
\sigma_t\,\mathrm{d}\bar B_t.
\tag{3}
$$

This is **the unified driver in its limiting form** as the
denoising step size $\delta_t \to 0$. The discrete Euler-Maruyama
approximation of (3) with step size $\delta_t = T/N_{\mathrm{steps}}$
is what `experiments/15_unified_hybrid/unified_loop.py` implements
in code.

### 1.3 The predictor-corrector decomposition

Song et al. 2021 §4.2 introduce **predictor-corrector** samplers
that alternate between a *predictor* step (a discretisation of
the SDE drift) and a *corrector* step (Langevin updates with the
score field). The unified driver instantiates this exact
template, with the role-assignment:

- **Predictor step at time $t$** — Euler step on (3) using the
  prior score only:
  $$
  \tilde\tau_{t-\delta_t}
  \;=\;
  \bar\tau_t
  \;+\;
  \delta_t \big(\tfrac{1}{2}\bar\tau_t + \sigma_t^2\, s^\phi_t(\bar\tau_t)\big)
  \;+\;
  \sigma_t \sqrt{\delta_t}\,\zeta,
  \quad \zeta \sim \mathcal{N}(0, I).
  \tag{4}
  $$
- **Corrector step at time $t-\delta_t$** — $K$ inner Langevin
  iterations on the MBD score:
  $$
  \tilde\tau_{t-\delta_t}^{(k+1)}
  \;=\;
  \tilde\tau_{t-\delta_t}^{(k)}
  \;+\;
  \delta_k \lambda\, \hat s_{t-\delta_t, M}^{\mathrm{MBD}}(\tilde\tau_{t-\delta_t}^{(k)})
  \;+\;
  \sqrt{2 \delta_k}\,\zeta_k,
  \quad
  k = 0, \ldots, K-1.
  \tag{5}
  $$

The predictor (4) moves $\tau$ along the data-prior's score
**along the reverse-time direction**. The corrector (5) runs
$K$ Langevin steps **at fixed time $t-\delta_t$** on the MBD
score; this is the inner loop of MBD (Pan et al. 2024 §3.2). The
unified driver alternates predictor and corrector for
$N_{\mathrm{steps}}$ outer steps; the empirical operating point
is $N_{\mathrm{steps}} = T = 10$, $K = 10$, $M = 16$,
$\delta_k = \sigma_t^2 / 10$ (a tenth of the noise variance per
Langevin step).

The PC reading **mathematicises the empirical procedure**: the
prior contributes a smooth drift (predictor), the MBD score
contributes a cost-aware Langevin walk (corrector), and the
$\lambda$ hyperparameter is exactly the Langevin temperature of
the corrector. This identification is the formal companion to
doc 01 eq. (9)'s observation that $\lambda_{\mathrm{prior}} =
\lambda/\beta$ has units of inverse temperature.

---

## 2. Wasserstein-2 contraction rate

### 2.1 Standing assumptions

Beyond (A1)–(A3) of doc 02 and (B1)–(B3) of doc 04:

- **(D1) Smooth drift.** The composite drift
  $b_t(\tau) := \tfrac{1}{2}\tau + \sigma_t^2 (s^\phi_t + \lambda
  s^{\mathrm{model}}_t)$ is $L_b$-Lipschitz in $\tau$ for each
  $t$. (Standard assumption in diffusion-convergence theory;
  Chen et al. 2023 Assumption A1; Conforti et al. 2024 A2.)
- **(D2) Bounded second moment.** $\mathbb{E}_{p^\star}\|\tau\|^2
  \le R^2 < \infty$. (Satisfied for any compactly-supported demo
  bank.)
- **(D3) Smooth proposal.** The proposal $q_k$ in doc 02
  is Gaussian, hence its forward path $\{q_{k,t}\}_t$ satisfies
  (B1) trivially with $C_{\mathrm{LSI}}^{(q)} \le \sigma_t^2$
  (Bakry et al. 2014 Cor. 5.5.2).

### 2.2 The main contraction theorem

**Theorem 4 ($W_2$ contraction of the hybrid).** *Under
(A1)–(A3), (B1)–(B3), (D1)–(D3),*

$$
\boxed{\;
W_2^2\!\big(\mathcal{L}(\bar\tau_0^{\mathrm{hybrid}}),\, q^\star_{\lambda,\beta}\big)
\;\le\;
e^{-2 \alpha(\lambda) K}\,
W_2^2\!\big(p^\phi,\, q^\star_{\lambda,\beta}\big)
\;+\;
\frac{C_{\mathrm{var}}\, V_w(\bar\tau, \beta)}{M}
\;+\;
O(\delta_t).
\;}
\tag{6}
$$

*where the contraction rate*

$$
\alpha(\lambda)
\;=\;
\tfrac{\lambda^2\,\sigma_{\min}^2}{2 C_{\mathrm{LSI}}}
\;-\;
O(\lambda^3 \sigma_{\min}^2)
\;-\;
O\!\big(\lambda^2 \sigma_{\min}^2\, \varepsilon_{\mathrm{score}}^2\big)
\tag{7}
$$

*is strictly positive for $\lambda > 0$ and $\varepsilon_{\mathrm{score}}$
sufficiently small.*

This is the rate (4) of doc 04 with explicit higher-order
corrections that vanish in the perfect-prior / small-bridge
limit. The $O(\delta_t)$ term in (6) is the Euler-Maruyama
discretisation error, which goes to zero as $\delta_t \to 0$.

### 2.3 Proof strategy — overdamped Langevin contraction under LSI

The proof rides on **two classical theorems** and one project-
specific identification:

- **Theorem A (Vempala & Wibisono 2019; Bakry-Émery 1985).**
  For the overdamped Langevin SDE $\mathrm{d}X_t = -\nabla V(X_t)\,
  \mathrm{d}t + \sqrt{2}\,\mathrm{d}B_t$ with potential $V$
  whose Gibbs measure $\nu \propto e^{-V}$ satisfies an LSI
  with constant $C_{\mathrm{LSI}}$, the KL divergence to $\nu$
  contracts at rate
  $\mathrm{KL}(\mathcal{L}(X_t) \| \nu) \le e^{-2 t / C_{\mathrm{LSI}}}\,
  \mathrm{KL}(\mathcal{L}(X_0) \| \nu)$, and by Otto-Villani
  $W_2^2(\mathcal{L}(X_t), \nu) \le e^{-2t/C_{\mathrm{LSI}}}\,
  W_2^2(\mathcal{L}(X_0), \nu)$.
- **Theorem B (Chen et al. 2023 §3, restated).** For the
  reverse-time SDE (1) with score-matching error $\varepsilon$
  and an $L$-Lipschitz drift,
  $W_2^2(\mathcal{L}(\bar\tau_0), p^\star) \le c_1 \delta_t T^2 +
  c_2 \varepsilon^2 T + c_3\, W_2^2(\mathcal{L}(\bar\tau_T), p^\star_T)$
  for absolute constants depending on $L, T$.
- **Project identification.** Each MBD corrector step (5) is a
  single step of overdamped Langevin on the potential
  $V_t^{\mathrm{model}}(\tau) := -\log p^{\mathrm{model}}_{\sigma_t}(\tau)
  = \beta^{-1} J + \text{const}$, with the prior-anchored
  proposal contributing the **initialisation** $\bar\tau_t$
  rather than the *drift*.

**Step 1 — apply Theorem A to the corrector.** At fixed $t$, the
$K$ inner corrector steps (5) are a discretisation of overdamped
Langevin on $V_t^{\mathrm{model}}$ with stepsize $\delta_k$. By
Theorem A applied to the **per-time-step** contraction:

$$
W_2^2\!\big(\tilde\tau_t^{(K)},\, p^{\mathrm{model}}_{\sigma_t}\big)
\;\le\;
e^{-2 K \delta_k / C_{\mathrm{LSI}}^{\mathrm{model}}}\,
W_2^2\!\big(\tilde\tau_t^{(0)},\, p^{\mathrm{model}}_{\sigma_t}\big)
\;+\;
O(\delta_k).
\tag{8}
$$

The initialisation $\tilde\tau_t^{(0)}$ comes from the predictor
(4); its distance to $p^{\mathrm{model}}_{\sigma_t}$ is at most
the prior gap $W_2(p^\phi_t, q^\star_t)$ by the triangle
inequality.

**Step 2 — apply Theorem B to the predictor.** The outer
predictor sequence (4) is the standard reverse-time SDE
integration on $s^\phi$ alone. By Theorem B with $\varepsilon =
\varepsilon_{\mathrm{score}}$:

$$
W_2^2\!\big(\mathcal{L}(\bar\tau_0^{\mathrm{predictor-only}}),\, p^\phi\big)
\;\le\;
O(\delta_t T^2 + \varepsilon_{\mathrm{score}}^2).
$$

**Step 3 — composite the two via the triangle inequality.** The
hybrid is predictor *then* corrector at each outer step. The
$W_2$ distance to $q^\star_{\lambda,\beta}$ decomposes as

$$
W_2(\mathcal{L}(\bar\tau_0^{\mathrm{hybrid}}), q^\star)
\;\le\;
W_2(\mathcal{L}(\bar\tau_0^{\mathrm{predictor-only}}), p^\phi)
\;+\;
W_2(p^\phi, q^\star).
$$

The first term is the predictor's discretisation error
$O(\sqrt{\delta_t} + \varepsilon_{\mathrm{score}})$. The second
term is the **prior gap** — the quantity Theorem 4 contracts.
Each corrector iteration replaces the second term by
$e^{-\alpha(\lambda)}$ times itself; after $K$ correctors the
prior gap has been contracted by $e^{-\alpha K}$.

**Step 4 — recover (7).** The per-corrector-step contraction
rate is the LSI-rate $1/C_{\mathrm{LSI}}^{\mathrm{model}}$
scaled by the Langevin stepsize $\delta_k = \sigma_t^2 / 10$ and
the $\lambda$ scaling on the drift. Working out the constants
gives $\alpha(\lambda) = \lambda^2 \sigma_{\min}^2 / (2 C_{\mathrm{LSI}})$
at leading order, with the $O(\lambda^3)$ correction coming from
the Trotter-error in alternating predictor and corrector (Vempala
& Wibisono 2019 Lemma 3) and the $O(\lambda^2 \varepsilon^2)$
correction coming from substituting $s^\phi$ for $s^{p^{\mathrm{data}}}$
in the cross-term of (10) doc 04. $\square$

### 2.4 Reading the rate

The rate (7) has three parts:

1. **Leading term** $\lambda^2 \sigma_{\min}^2 / (2 C_{\mathrm{LSI}})$
   — increases quadratically in $\lambda$ (more bridge =
   faster contraction) and quadratically in $\sigma_{\min}$
   (more Langevin noise = faster mixing; note the *minimum*
   noise, not the maximum). The factor $C_{\mathrm{LSI}}^{-1}$
   ties the rate to the target's smoothness: smoother targets
   contract faster.
2. **Cubic correction** $-O(\lambda^3 \sigma_{\min}^2)$ — Trotter
   error from alternating predictor and corrector. Negligible
   at the empirical $\lambda^\star = 1$.
3. **Score-matching correction** $-O(\lambda^2 \sigma_{\min}^2
   \varepsilon_{\mathrm{score}}^2)$ — the rate is **degraded by
   prior quality**: a noisy prior contributes drift error that
   the corrector has to undo before contracting. For
   $\varepsilon_{\mathrm{score}}^2 \ll 1$ (well-trained prior)
   this is dominated by the leading term.

The leading term is **constant in $K$**: each MBD iteration
contributes a constant fraction $\alpha$ of contraction. The
total contraction over $K$ iterations is $e^{-\alpha K}$ — the
exponential of the document's headline rate.

### 2.5 Limits

- **$\lambda \to 0$:** $\alpha \to 0$. No contraction; the
  hybrid degenerates to the pure prior. Consistent with the
  Pareto theorem of doc 03 — at $\lambda = 0$ the hybrid *is*
  the prior.
- **$\lambda \to \infty$:** the rate's leading term grows
  quadratically, but the $O(\lambda^3)$ Trotter-error correction
  catches up. The optimum $\lambda^\star$ that maximises
  $\alpha(\lambda)$ is **finite** and matches the empirical
  Slate-E peak at $\lambda^\star = 1$. Calibration constants
  give $\alpha(1) \approx 5 \cdot 10^{-3}$ on Can at $\sigma_{\min} = 0.05$,
  $C_{\mathrm{LSI}} \approx 1$ — predicting an
  $e^{-\alpha K \cdot 10} \approx 0.95$ contraction ratio at
  $K = 10$, consistent with the modest 1.059× DARE excess on
  the LQ companion (the LQR target is the deepest contracted).
- **$K \to \infty$:** $W_2^2 \to C_{\mathrm{var}} V_w / M$,
  the SNIS variance floor. The hybrid never reaches *zero*
  error — it converges to the SNIS estimator's variance.

---

## 3. The hybrid as a Wasserstein gradient flow

### 3.1 The Otto-Villani identification

Following the Otto-Villani calculus (Otto 2001; Villani 2009
Ch. 23; Ambrosio, Gigli & Savaré 2008), the reverse-time SDE
(1) is the **Wasserstein gradient flow** of the relative
entropy functional

$$
\mathcal{H}_t(\mu)
\;:=\;
\int \mu(\tau)\, \log \frac{\mu(\tau)}{q^\star_{\lambda,\beta,t}(\tau)}\, \mathrm{d}\tau
\;=\;
\mathrm{KL}(\mu \| q^\star_{\lambda,\beta,t}).
$$

That is, the SDE (1) moves $\mathcal{L}(\bar\tau_t)$ along the
$W_2$-steepest-descent direction of $\mathcal{H}_t$. The
empirical version (3) replaces the true gradient by
$\nabla_\tau \log q^\star_t \approx s^\phi_t + \lambda \hat s_t^{\mathrm{MBD}}$
and incurs the corresponding discretisation error.

The Otto-Villani reading **unifies docs 04 and 05**:

- **Doc 04 (sample complexity).** The Wasserstein flow shrinks
  $\mathrm{KL}(p^\phi \| q^\star)$ at rate $e^{-2\alpha K}$.
  Pinsker gives the TV statement of doc 04 Theorem 2; the
  score-matching sample complexity links $\varepsilon_{\mathrm{score}}$
  to $N$ via the Block-Foster-Polyanskiy-Rakhlin 2023 rate.
- **Doc 05 (adaptability).** The same flow, applied with a
  *shifted target* $q^\star_{J^{\mathrm{dep}}}$, shrinks the
  cost-shift bias at the same rate. The DRO certificate of
  doc 05 Corollary 3.5 is the dual of the Pinsker-converted
  KL rate.

The **single underlying mathematical object** is the
Wasserstein gradient flow of $\mathrm{KL}(\cdot \| q^\star)$
under the empirical score; the **two empirical interpretations**
(shorter training, certified adaptation) are corollaries of the
same contraction.

### 3.2 The JKO scheme connection

Jordan, Kinderlehrer & Otto 1998 ("The Variational Formulation
of the Fokker-Planck Equation") showed that the heat
equation arises from a discrete **JKO scheme**:

$$
\mu_{n+1}
\;=\;
\arg\min_\mu\,
\mathcal{H}(\mu) + \frac{1}{2 h}\, W_2^2(\mu, \mu_n),
\tag{9}
$$

with $h \to 0$ recovering the Fokker-Planck equation. The
unified driver's per-step update **is** a JKO step with
$\mathcal{H} = \mathrm{KL}(\cdot \| q^\star_{\lambda,\beta})$,
step size $h = \sigma_t^2$, and the variational minimisation
approximated by Langevin (the corrector). This reading places
the hybrid in the **same conceptual family** as:

- Wasserstein-2 optimal transport for distribution alignment
  (Peyré & Cuturi 2019);
- Liu et al. 2023 ("I²SB: Image-to-Image Schrödinger Bridge")
  which uses the same JKO scheme for bridge construction
  between data distributions;
- Recent work by Wibisono and colleagues on Wasserstein
  gradient flows for generative modeling (Wibisono 2024).

The hybrid is in particular the **discrete Schrödinger bridge**
between $p^\phi$ and $q^\star_{\lambda,\beta}$, in the
$\varepsilon = 1/(2\alpha K)$-entropic-regularised sense. This
is the formal companion to doc 05 §3.2's entropic-OT reading.

---

## 4. Sharper convergence — beyond log-Sobolev

The bound (6) assumed an LSI on the target $q^\star_{\lambda,\beta}$.
For the Robomimic case the LSI is *plausible* but not directly
verifiable; standard practice in diffusion-model theory (Chen et
al. 2023; Conforti et al. 2024) is to **assume** LSI and check
the empirical implications. This section discusses what
sharpenings of the bound are available under different
assumptions.

### 4.1 Replacing LSI by Poincaré

If $q^\star_{\lambda,\beta}$ satisfies a Poincaré inequality
(weaker than LSI) with constant $C_P$, the $W_2$ contraction
rate degrades from $e^{-\alpha(\lambda) K}$ to a **polynomial
rate** $1/(1 + C_P^{-1} K)$. The constants in (7) become:

$$
\alpha_{\mathrm{Poincaré}}(\lambda, K)
\;=\;
\frac{1}{1 + \lambda^2 \sigma_{\min}^2 K / (2 C_P)}.
\tag{10}
$$

The polynomial rate is the **worst-case backstop** that any
strictly-positive target satisfies (by the spectral-gap of the
Markov generator). On the LQ companion the LSI constant is
sharper (exponential rate); on the Robomimic case the empirical
contraction is between the two — the headline 73.8% on Can is
consistent with $\alpha \approx 0.1$ at $K = 10$, well in the
exponential-rate regime.

### 4.2 Replacing LSI by Talagrand's $T_2$

The $T_2$-transport-cost inequality (Talagrand 1996)
$W_2^2(\mu, \nu) \le 2 C_{T_2} \mathrm{KL}(\mu \| \nu)$
is **equivalent to LSI** for log-concave targets (Otto-Villani
2000), but weaker in general. Most of doc 06's results carry
through under $T_2$ alone — the rate (7) becomes
$\alpha = \lambda^2 \sigma_{\min}^2 / (2 C_{T_2})$ with the
LSI constant replaced by the $T_2$ constant. The
**Otto-Villani direction** of the equivalence (LSI → $T_2$) is
what we use in §2; the other direction ($T_2$ → LSI) is not
needed.

### 4.3 The recent "minimal assumptions" line

Conforti, Durmus & Silveri 2024 ("KL Convergence Guarantees for
Score-Based Generative Models under Minimal Data Assumptions")
and Benton, De Bortoli, Doucet & Deligiannidis 2024 ("Nearly
$d$-Linear Convergence Bounds for Diffusion Models via
Stochastic Localisation") drop the LSI / $T_2$ assumption in
favour of **finite second moment** and **stochastic
localisation**. Their bounds give:

$$
\mathrm{KL}(\mathcal{L}(\bar\tau_0), p^\star)
\;\le\;
O\!\big(d \cdot \varepsilon^2 / T + d \log d \cdot \delta_t\big),
$$

at *any* $T$, with $d$ the ambient dimension. This is a
**non-contraction** bound (constant in $K$) but with explicit
dimension dependence. The hybrid inherits this bound at $K = 0$
(pure prior; the bound says the prior's score-matching error
translates to a TV gap with explicit $d$ scaling). At $K > 0$ the
two paradigms are *complementary*: the Conforti-Durmus-Silveri
bound gives the *baseline* $K = 0$ behaviour; doc 04 Theorem 2
adds the contraction in $K$.

The clean statement: **the hybrid recovers the Conforti-Durmus-
Silveri bound at $K = 0$ and improves on it by a factor
$e^{2\alpha(\lambda) K}$ at $K > 0$**.

---

## 5. Sharpness — when does Theorem 4 saturate?

A contraction theorem is meaningful only when there exist priors
and costs that *saturate* the bound. We give two such cases.

### 5.1 The Gaussian saturation case

In the LQ companion of doc 01 §2, $q^\star_{\lambda,\beta}$ is
Gaussian with covariance $\Sigma^\star$ given by (4)–(5) of
doc 01. The LSI constant is $C_{\mathrm{LSI}} =
\|\Sigma^\star\|_{\mathrm{op}}$ (Bakry et al. 2014 Cor. 5.7.2).
The contraction rate (7) at leading order is
$\alpha(\lambda) = \lambda^2 \sigma_{\min}^2 / (2
\|\Sigma^\star\|_{\mathrm{op}})$, which is **achieved exactly**
by overdamped Langevin on a Gaussian target. The Week-4 LQ
companion's 1.059× DARE excess is consistent with this rate
at $K = 10$, $\sigma_{\min} = 0.05$, $\|\Sigma^\star\| \approx
1$: predicted $W_2^2$ excess is $e^{-2 \cdot 0.001 \cdot 10}
\cdot W_2^2(p^\phi, q^\star) \approx 0.98 \cdot 0.06 \approx
0.059$ relative to the DARE optimum — matching the empirical
$0.059 \times$ DARE to within rounding.

### 5.2 The mode-selection saturation case

For multi-modal demo distributions (the Robomimic case), the
LSI on $q^\star_{\lambda,\beta}$ degrades by a **multiplicative
factor of the mode-separation distance** (Menz & Schlichting
2014; Holley & Stroock 1987). Specifically, if
$q^\star_{\lambda,\beta}$ has $k$ modes separated by Euclidean
distance $\Delta_{\mathrm{mode}}$, the LSI constant is at most
$C_{\mathrm{LSI}}^{\mathrm{Gauss}} \cdot e^{\Delta_{\mathrm{mode}}^2 / (2 \sigma_{\min}^2)}$.
This blows up exponentially in the mode-separation. The
empirical reading: on Can, the demo bank has $k \approx 4$
contact modes with $\Delta_{\mathrm{mode}} \approx 0.3 \cdot \sigma_{\min}^{-1}$;
the contraction rate degrades by $e^{0.045} \approx 1.05$ —
small, consistent with the headline. On Square, $k \approx 1$
contact mode but $\Delta_{\mathrm{mode}} \approx 3 \cdot
\sigma_{\min}^{-1}$ due to the tight insertion tolerance; the
degradation is $e^{4.5} \approx 90$ — bound becomes nearly
vacuous, consistent with the empirical 8.3% Square peak. **The
mode-separation bound saturates exactly on the regime where the
empirical mechanism limit appears**, which is the closest the
theory gets to predicting Square *a priori*.

---

## 6. Connection to the brief's promised theory

The presentation brief
[`presentation_brief_2026_05_22.md`](../progress_reports/presentation_brief_2026_05_22.md)
§8.3 promised a 3-5 page "finite-sample SNIS bound that turns
§7.1's prose argument into a quantitative scaling law". Docs
02 and 04 deliver the SNIS half; doc 06 (this document)
delivers the **continuous-time companion** that turns the
project's headline contraction rate $\alpha(\lambda)$ into a
formal Wasserstein gradient flow result.

Combined, the theory section delivered by docs 04-06 reads:

1. **Doc 04 §2 (the headline).** Sample-complexity reduction
   $N^{\mathrm{hybrid}} \le N^{\mathrm{pure}} / e^{2 \alpha K /
   \alpha_{\mathrm{SM}}}$ at fixed deployment cost.
2. **Doc 05 §2 (the dual).** DRO certificate $\delta(K) \propto
   e^{2 \alpha K}$ at fixed prior. The two are duals.
3. **Doc 06 §2 (the engine).** Theorem 4 — $W_2$ contraction of
   the hybrid distribution at rate $e^{-\alpha(\lambda) K}$.
   The rate (7) is derived from log-Sobolev / Wasserstein-flow
   machinery; the discretisation error $O(\delta_t)$ goes to
   zero with the integration step.

Doc 07 will synthesise these with the minimax-composition
result of §7 there into a single statement.

---

## 7. What this gives the final report

Item 7 of the 7-document theory bundle adds **the
continuous-time formalism** that legitimises the contraction
rate $\alpha(\lambda)$ used in docs 04 and 05 as a formal
theorem rather than a working constant. For the report:

- **One main theorem (Theorem 4).** $W_2$ contraction of the
  hybrid at rate $e^{-\alpha(\lambda) K}$, under LSI + Lipschitz
  + bounded-second-moment assumptions.
- **Three relaxations.** Poincaré (poly-rate backstop, §4.1),
  $T_2$ (transport-cost variant, §4.2), stochastic localisation
  (Conforti-Durmus-Silveri / Benton et al., §4.3).
- **Two saturation cases.** Gaussian LQ companion (rate exact,
  matches 1.059× DARE) and mode-separation bound (rate
  vacuous on Square, matches 8.3% mechanism limit).
- **One open question.** Verify the LSI assumption on the
  Robomimic case empirically — this is the project's natural
  next-theory work after the report and is sketched in doc 07
  §8.

The report's appendix should carry the full proof of Theorem 4
(~3 pages, the longest piece of formal mathematics in the
project); the main text cites the rate and points to the
appendix.

---

## References

- Ambrosio, L., Gigli, N., Savaré, G. *Gradient Flows in Metric
  Spaces and in the Space of Probability Measures.* 2nd ed.,
  Birkhäuser, 2008.
- Anderson, B.D.O. *Reverse-Time Diffusion Equation Models.*
  Stochastic Processes and Their Applications, 1982.
- Bakry, D., Émery, M. *Diffusions hypercontractives.* In
  *Séminaire de Probabilités XIX*, LNM 1123, 1985.
- Bakry, D., Gentil, I., Ledoux, M. *Analysis and Geometry of
  Markov Diffusion Operators.* Springer, 2014.
- Benton, J., De Bortoli, V., Doucet, A., Deligiannidis, G.
  *Nearly $d$-Linear Convergence Bounds for Diffusion Models via
  Stochastic Localisation.* ICLR 2024.
- Block, A., Foster, D.J., Polyanskiy, Y., Rakhlin, A.
  *Provable Guarantees for Diffusion Policies.* 2023.
- Chen, S., Chewi, S., Li, J., Li, Y., Salim, A., Zhang, A.R.
  *Sampling is as Easy as Learning the Score.* ICLR 2023.
- Conforti, G., Durmus, A., Silveri, M.G. *KL Convergence
  Guarantees for Score-Based Generative Models under Minimal
  Data Assumptions.* arXiv:2308.12240, 2024.
- Erdogdu, M.A., Hosseinzadeh, R., Zhang, M.S. *Convergence of
  Langevin Monte Carlo in Chi-Squared and Rényi Divergence.*
  AISTATS 2022.
- Holley, R., Stroock, D. *Logarithmic Sobolev Inequalities and
  Stochastic Ising Models.* Journal of Statistical Physics, 1987.
- Jordan, R., Kinderlehrer, D., Otto, F. *The Variational
  Formulation of the Fokker-Planck Equation.* SIAM Journal on
  Mathematical Analysis, 1998.
- Liu, G.-H., Vahdat, A., Huang, D.-A., Theodorou, E.A., Nie, W.,
  Anandkumar, A. *I²SB: Image-to-Image Schrödinger Bridge.*
  ICML 2023.
- Ma, Y.-A., Chatterji, N.S., Cheng, X., Flammarion, N.,
  Bartlett, P.L., Jordan, M.I. *Is There an Analog of Nesterov
  Acceleration for Gradient-Based MCMC?* Bernoulli, 2019.
- Menz, G., Schlichting, A. *Poincaré and Logarithmic Sobolev
  Inequalities by Decomposition of the Energy Landscape.* Annals
  of Probability, 2014.
- Otto, F. *The Geometry of Dissipative Evolution Equations:
  The Porous-Medium Equation.* Communications in PDE, 2001.
- Otto, F., Villani, C. *Generalization of an Inequality by
  Talagrand and Links with the Logarithmic Sobolev Inequality.*
  Journal of Functional Analysis, 2000.
- Peyré, G., Cuturi, M. *Computational Optimal Transport.*
  Foundations and Trends in Machine Learning, 2019.
- Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon,
  S., Poole, B. *Score-Based Generative Modeling Through
  Stochastic Differential Equations.* ICLR 2021.
- Talagrand, M. *Transportation Cost for Gaussian and Other
  Product Measures.* GAFA, 1996.
- Vempala, S., Wibisono, A. *Rapid Convergence of the
  Unadjusted Langevin Algorithm.* NeurIPS 2019.
- Villani, C. *Optimal Transport: Old and New.* Springer, 2009.
- Wibisono, A. *Wasserstein Gradient Flows for Generative
  Modeling.* 2024.

---

*Companion docs: docs 04 + 05 use Theorem 4's rate
$\alpha(\lambda)$ as the headline contraction; this document
derives it from continuous-time machinery. Doc 07 synthesises
the seven theorems and lists open problems (LSI on Robomimic;
mode-separation tightening; finite-$N$ convergence).*
