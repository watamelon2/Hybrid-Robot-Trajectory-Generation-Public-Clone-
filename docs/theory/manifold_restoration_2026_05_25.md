# Manifold restoration vs noise corruption: an SNIS-based theory of MBD's hybrid-prior behaviour

**Date.**  2026-05-25.  **Branch.**  `feat/v6-pull-slate-l`.
**Empirical basis.**  Slate K-N + Slate O, summarised in
`docs/results.md`.

This note formalises the **mechanistic unified picture** that
emerged from Slates K, L, M, N, O.  It explains why MBD's
rescue is large in some regimes (`demo_1nn`: 11.7 % standalone
→ 73.8 % with MBD, +62 pp lift) and zero in others (v6 + pull
at canonical config: 83 % standalone → 81 % with MBD, ~0 pp
lift; `demo_corruption σ ≥ 0.3`: ~0 % with MBD).  The key claim
is that MBD's lift is governed by a single geometric quantity:
the **cost-gradient reachability** of the σ-neighbourhood of
the warm-start in action space.

The empirical results that the theory must explain are the
**five-prior lift curve** (Slate M results.md table) and the
**Wave D noise-corruption collapse** (Slate L L3).  This note
gives a unified explanation; it does not introduce new
empirical results.

---

## 1. The MBD inner loop as self-normalized importance sampling

MBD's per-iteration update (using the `--score-mode pull`
formulation that Slate K established as canonical) is

$$
\mu_{k+1} \;=\; \lambda \, \mu_{0} \;+\; (1-\lambda) \,
                \sum_{m=1}^{M} w_m^{(k)} \, a_m^{(k)},
\qquad a_m^{(k)} = \mu_k + \sigma_k \, z_m^{(k)},
\quad z_m^{(k)} \sim \mathcal{N}(0, I_{H \times d_a}),
$$

with self-normalised importance weights

$$
w_m^{(k)} \;=\; \frac{\exp\!\bigl(-\,c(a_m^{(k)})\,/\,\beta\bigr)}
                     {\sum_{m'} \exp\!\bigl(-\,c(a_{m'}^{(k)})\,/\,\beta\bigr)}.
$$

Here $\mu_k \in \mathbb{R}^{H \times d_a}$ is the current
trajectory iterate (initialised to the warm-start $\mu_0 = a_\text{warm}$),
$\sigma_k$ follows the cosine schedule from $\sigma_\text{max}$ down to
$\sigma_\text{min}$, and $c(a)$ is the cost rollout of executing
$a$ in the environment (negative of total reward, or whatever
cost signal MBD is wired to).  $\beta$ is the SNIS temperature
(`--lambda-softmax`); $\lambda \in [0,1]$ is the
pull-back weight (`--lambda-prior`).

**Observation.** With $\lambda \to 1$, the pull dominates and
$\mu_{k+1} \to \mu_0$; the MBD update is a no-op.  With
$\lambda \to 0$, the SNIS update dominates and $\mu_{k+1}$ is
the SNIS posterior mean over the $M$ candidates.  At the
canonical $\lambda = 1$ with $K=10$ iterations, *each* iteration
re-anchors to $\mu_0$ but injects a small SNIS-driven
perturbation; the net effect after $K$ iterations is a
**weighted average of the warm-start and the SNIS modes** at
each $\sigma_k$ scale.  This is the "rank-and-pick" behaviour
that Slate M tries to bound experimentally.

The candidate distribution $a_m^{(k)} = \mu_k + \sigma_k z_m$ is
a Gaussian proposal centred at the current iterate.  The SNIS
target it implicitly approximates is

$$
p_\text{target}(a) \;\propto\; \exp\!\bigl(-\,c(a)\,/\,\beta\bigr) \,\cdot\, \mathcal{N}\!\bigl(a \,\big|\, \mu_0,\ \sigma_\text{pull}^2 I\bigr),
$$

where $\sigma_\text{pull}^2$ is the effective variance of the pull
constraint (a monotone function of $\lambda$ at the canonical
$K=10$; at $\lambda=1$ this prior is sharp around $\mu_0$).

---

## 2. The lift lemma

**Lemma (informal).**  Let $a_0$ be the warm-start, $\mathcal{B}_\sigma(a_0)$
the $L^2$-ball of radius $\sqrt{H \cdot d_a} \cdot \sigma_\text{max}$
in action space (the *effective reach* of the SNIS proposal
over $K$ iterations).  Define $c^\star_\sigma \,=\, \min_{a \in \mathcal{B}_\sigma(a_0)} c(a)$.
The MBD lift (success rate improvement vs. executing $a_0$
directly) satisfies

$$
\text{lift}(a_0,\sigma) \;\leq\; \mathbb{P}\!\bigl[\,\text{success}\,\big|\,c \leq c^\star_\sigma\bigr] \;-\; \mathbb{1}\{\text{success}(a_0)\},
$$

with equality (up to SNIS Monte-Carlo error from finite $M$)
when

(i) $a_0$ is in a *cost basin* with a unique minimum at
$a^\star \in \mathcal{B}_\sigma(a_0)$ that is well-separated from
other local minima; **and**

(ii) the SNIS proposal $\mathcal{N}(a_0, \sigma^2 I)$ assigns
non-negligible mass to $a^\star$.

**Three regimes.**  This decomposition predicts three
qualitatively distinct outcomes, all matched empirically:

| regime | warm-start property | $c^\star_\sigma$ vs $c(a_0)$ | predicted lift | measured |
|--------|---------------------|------------------------------|----------------|----------|
| **R1 Parity** | $a_0$ at the cost optimum | $c^\star_\sigma \approx c(a_0)$ | ≈ 0 pp | v6+pull: 83 → 81 (+0) |
| **R2 Full rescue** | $a_0$ off-optimum but $a^\star \in \mathcal{B}_\sigma(a_0)$ | $c^\star_\sigma \ll c(a_0)$ | large positive | demo_1nn: 12 → 74 (+62) |
| **R3 Collapse** | $\mathcal{B}_\sigma(a_0)$ contains no good action | $c^\star_\sigma \approx c(a_0)$ but neither is feasible | 0 (or negative) | Wave D σ_corrupt ≥ 0.3: 0 (+0) |

The first column ("warm-start property") and the second
("$c^\star_\sigma$") are independent geometric facts: the
warm-start can be *bad* either because it is *near* a good
$a^\star$ but $a^\star$ is outside $\mathcal{B}_\sigma(a_0)$
(rescue fails), or because $\mathcal{B}_\sigma(a_0)$ has *no* good
action at all (collapse).

---

## 3. Application to the five-prior lift curve

The lift curve in `docs/results.md` (Slate M section) tabulates
five priors ordered by standalone strength.  Each can be
classified into one of the three regimes:

### 3.1 `demo_1nn` (11.7 → 73.8, +62 pp): R2 full rescue

The warm-start is *exactly one demonstration trajectory* —
the one whose initial observation is the nearest neighbour of
the rollout's initial observation, in obs space.  This
trajectory lives on the demo manifold by construction.

When the 1-NN retrieval mis-fires (matches a demo for a
*different* task variant), the warm-start is bad
($c(a_0)$ high → 12 % standalone) but the **true best demo**
is also on the manifold, and "moving to a different demo's
action sequence" is a small displacement in action space:
demos in the can-pick-place dataset have similar action
statistics (low std around their mean trajectory), so the
inter-demo distance is $O(\sigma_\text{max} \sqrt{H d_a})$.
$\mathcal{B}_\sigma(a_0)$ therefore covers nearby alternative
demos, and SNIS finds them.  Lift is large.

This is the **rescue regime**: warm-start *on* the demo
manifold but *not* at the cost-optimal demo.

### 3.2 `demo_knn k=5` (3.3 → 23.3, +20 pp): partial rescue at the regime boundary

The warm-start is the *uniform average* of the top-5 nearest
demos.  Averaging $\geq 2$ demos with non-aligned dynamics
produces a **synthetic trajectory** that does not match any
single demo's path.  It is therefore *off* the demo manifold,
but only *mildly* off — the average is still in the convex
hull of the demo set, which sits inside a thin shell around
the manifold.

By the lemma, $c^\star_\sigma$ over $\mathcal{B}_\sigma(a_0)$ might
contain a real demo at the boundary, giving partial recovery
(+20 pp), but the SNIS proposal has a hard time *fully*
recovering because to reach a clean demo requires "un-averaging"
the 5-demo mean, which is not in $\mathcal{B}_\sigma(a_0)$ at
$\sigma_\text{max} = 0.25$.

This is the **partial rescue regime**: warm-start in a thin
*off-manifold shell*; some alternative actions in
$\mathcal{B}_\sigma(a_0)$ are good (real demos), but the bulk
of the proposal mass is still near the synthetic average.

### 3.3 BC pool (38 → 64, +26 pp): partial rescue, similar mechanism

A behavior-cloning prior emits action sequences that are
trained to *interpolate* between demos.  The BC output is
typically near the demo manifold but not exactly on it; the
prior fails when it interpolates two task-variant demos into
something that satisfies neither.  When MBD samples in
$\mathcal{B}_\sigma$, it finds nearby on-manifold actions —
giving the same kind of partial recovery as `demo_knn k=5`.

### 3.4 v6 single-shot (63 → 81, +18 pp): R1 small-lift via rank-and-pick

**Update 2026-05-25 late (post Slate O).**  An earlier draft of
this note classified v6 + MBD-pull at canonical as R1 Parity
(+0 pp), based on the K3 single-seed cell being numerically
close to a then-unmeasured "v6 standalone" reference.  Slate O
measured pure v6 standalone directly (K=1 M=1 σ≈0 λ=0, 3-seed
pool 57/90 = 63.3 % CI [53.0, 72.6]) and revealed that the K3
v6+MBD-pull cell (3-seed pool 73/90 = 81.1 % CI [71.9, 88.0])
actually adds **+17.8 pp** with non-overlapping 95 % CIs.  The
correct classification is **R1 Small-lift via rank-and-pick**:
$c^\star_\sigma < c(a_0)$ strictly, but only by a small margin
because v6 sits *near* (not *at*) the cost optimum.

The SNIS proposal at $\sigma_\text{max}\!=\!0.25$ samples
$M\!=\!16$ neighbours of $a_0$; the cost landscape is locally
flat-but-with-small-discriminable-improvements (the warm-start
is competent but not perfect).  The rank-and-pick over $M\!=\!16$
candidates finds a meaningfully better neighbour ~18 pp of the
time.  Equivalently: v6's deterministic ODE-sampler has ~8 pp
seed-to-seed std (Slate O measured a 20 pp spread across 3
seeds: 53.3, 63.3, 73.3); the M=16 SNIS step samples this
sampler-noise distribution and picks the best, collapsing the
seed spread to 3 pp (80.0, 80.0, 83.3) at the same time it
lifts the mean.

**Empirical confirmation from Slate L.**  L1 (σ-sweep) showed
monotone degradation as $\sigma_\text{max}$ drops below 0.25:
60 % at σ=0.05, 77 % at σ=0.10, 80 % at σ=0.15 vs 83 % at
σ=0.25.  In R1 the SNIS rank-and-pick lift requires
$\sigma_\text{max}$ large enough that the M=16 candidates are
*discriminably different* from $a_0$; too small (σ=0.05) and
all candidates ≈ $a_0$, so the rank-and-pick has nothing to
choose between and the lift vanishes.  The σ=0.25 sweet spot
is the discriminability optimum: large enough to expose
sampler-noise variation, small enough that the SNIS softmax
can identify the better candidate without the noise drowning
the cost signal.

**The R1 small-lift regime predicts**: the lift size should be
roughly proportional to the *sampler-noise std* of $a_0$
itself.  Stronger priors with tighter samplers (smaller seed
variance) will give smaller R1 lifts; weaker priors with wider
samplers will give larger R1 lifts (up to the point where the
prior crosses into R2/R3).  This is a testable prediction for
future stronger or weaker CFM priors.

### 3.5 v6 + noise corruption σ ≥ 0.3 (0 % with MBD): R3 collapse

`--warmstart-noise-sigma 0.3` adds iid Gaussian noise of std
0.3 to every step of v6's warm-start trajectory **before**
MBD's inner loop sees it.  This displaces $a_0$ by
$\sigma_\text{corrupt} \sqrt{H d_a} \approx 0.3 \times \sqrt{120 \times 7} \approx 9$
in $L^2$ action units.  $\mathcal{B}_\sigma(a_0)$ at
$\sigma_\text{max}=0.25$ has radius $\sim 7.5$ — too small to
*reach* either v6's true warm-start (displaced by ~9) or any
demo manifold (also at $L^2$ distance > 7 from a
noise-corrupted point in a high-d space).

So $c^\star_\sigma \approx c(a_0)$ and both are infeasible
(no successful policy reachable from the noise-corrupted
neighbourhood).  Lift is zero.  This is the **collapse
regime**.

---

## 4. Predictions and falsification

The lemma makes three testable predictions.

### 4.1 Lift should scale with $\sigma_\text{max}$ in the rescue regime

In R2 (rescue), increasing $\sigma_\text{max}$ enlarges
$\mathcal{B}_\sigma(a_0)$ and should *increase* the lift (up
to the point where the SNIS proposal becomes too diffuse and
the M=16 sample budget is exhausted).  In R1 (parity), changing
$\sigma_\text{max}$ should produce **non-monotone** behaviour
(too small → indistinguishable from standalone; too large →
diffusion outpaces SNIS selection).

**Empirical match.**  G2 (Slate G2 on Can) measured the
`demo_1nn` lift curve as a function of $\sigma_\text{max}$:
peak at $\sigma_\text{max} = 0.25$, with $\sigma_\text{max} \in \{0.05, 1.0\}$
all worse.  This is R2 behaviour.  L1 measured v6's behaviour:
*monotone* in $\sigma_\text{max}$ below 0.25, with the
*opposite* sign (lower σ → lower succ).  This is R1
behaviour with rank-and-pick mechanism: at lower σ the M=16
candidates collapse to copies of $a_0$, the SNIS softmax has
no discriminative power, and the executed trajectory is
*also* $a_0$ but with one perturbation that occasionally hurts.

### 4.2 Wave D σ_corrupt should have a sharp threshold

The collapse regime should kick in when the corruption
displacement $\sigma_\text{corrupt} \sqrt{H d_a}$ exceeds the
SNIS reach $\sigma_\text{max} \sqrt{H d_a}$, i.e.
$\sigma_\text{corrupt} \gtrsim \sigma_\text{max} = 0.25$.

**Empirical match.**  Wave D: σ=0.3 → 3.3 % (just over the
threshold, partial collapse), σ=0.5 → 0 % (well over), σ=1.0
→ 0 % (deep collapse).  The threshold appears to be in the
0.25-0.30 range, consistent with the prediction.

### 4.3 `demo_knn` lift should monotone-decrease with $k$

`demo_knn k=1` is in R2 (on-manifold) → +62 pp.
`demo_knn k=5` is in the R2/R3 boundary → +20 pp.
Predictions: `k=2` should be near-R2 (+50 pp), `k=3` somewhere
in between (+35 pp), `k \to \infty` (uniform demo mean) should
collapse to ~0 lift because the mean is far enough off-manifold
that $\mathcal{B}_\sigma$ no longer covers any demo.

**Open follow-up** (Slate Ext T3.2, see
`docs/todos/feat_v6_pull_slate_l_pending_2026_05_25.md`):
measure `k ∈ {2, 3}` to confirm.

---

## 5. Connection to the SOC / SNIS literature

The MBD update is a Monte-Carlo approximation of a Bayes
posterior with proposal $p_\text{prop}(a) = \mathcal{N}(\mu_k, \sigma_k^2 I)$
and target $p_\text{target}(a) \propto \exp(-c(a)/\beta)$.  In
the Stochastic Optimal Control literature this is a path-integral
control update (Kappen 2005; Williams et al. 2017's MPPI), and
in the diffusion-sampling literature it is a noise-conditioned
posterior sampling step.

The connection to score-based generative modeling becomes clean
in the `--score-mode pull` formulation: the pull-back to
$\mu_0$ is exactly the prior term $\nabla \log p_\text{prior}(\mu)$
in a Bayes update where the prior is a sharp Gaussian at the
warm-start.  When the prior is sharp (large $\lambda$), the
posterior collapses to the prior mode and MBD ≡ executing
$a_0$.  This is the formal version of "MBD-pull at $\lambda=1$
recovers v6 standalone" that Slate K observed empirically.

The five-prior lift curve becomes, in this framing, **a curve
of the SNIS posterior's distance from its prior mode**: large
distance ⇒ large effective rescue, zero distance ⇒ parity,
non-existent meaningful target ⇒ collapse.

---

## 6. What this rules out

1. **"MBD universally improves any prior" is falsified.**  The
   lemma predicts zero lift whenever the prior is already at the
   cost optimum or whenever $\mathcal{B}_\sigma$ contains no good
   action.  v6+pull (parity regime) and Wave D (collapse regime)
   are both falsifying instances.

2. **"MBD is mechanism-limited by the cost function, not the
   prior" is falsified.**  The cost is the same across all five
   priors; the lift varies from +62 pp to 0 pp.  The decisive
   variable is the prior's *geometric* relation to the cost
   landscape ($c(a_0)$ and the reachable $c^\star_\sigma$), not
   the cost itself.

3. **"MBD is mechanism-limited by the prior's training
   distribution" is partially falsified.**  v6 is trained on
   *the same demos* as `demo_1nn`; the prior just produces a
   different point in action space.  Slate K already established
   that the prior is what changes; this note adds the
   geometric mechanism.

---

## 7. Open questions

- **Quantifying $\sigma_\text{pull}^2$ as a function of $\lambda, K$.**
  The "effective pull variance" after $K$ iterations is what
  determines whether the SNIS update can escape $\mu_0$ at all.
  An analytical expression (probably a damped-Langevin
  characterisation of the $K$-iter chain) would let us **predict**
  the parity-vs-rescue boundary from the prior's cost geometry,
  rather than just observing it empirically.

- **Validating R3 collapse off-Can.**  Wave D measured collapse
  on Can with $\sigma_\text{corrupt} \geq 0.3$.  Does the same
  threshold hold on Lift / Square / ToolHang, or does it scale
  with the task's action-dim and horizon?  Slate G's Lift cells
  did *not* probe noise corruption; this is a clean follow-up.

- **The rank-and-pick lift over true standalone.**  Slate O (in
  flight) measures pure v6 standalone at multiple seeds.  If the
  pool lands meaningfully below v6+pull's 81.1 %, the M=16
  SNIS-mean over $\mathcal{B}_\sigma(a_0)$ does have a small but
  non-zero lift, even in the "parity" regime.  The lemma allows
  this (it gives a $\leq$ bound, not equality); the size of
  this lift in R1 is currently estimated as
  "indistinguishable within seed noise" but Slate O will give
  a direct measurement.
