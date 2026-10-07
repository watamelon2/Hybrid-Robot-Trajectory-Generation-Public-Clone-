# Theory 9 — Our data-guided hybrid vs MBD §4.3 "Model-Based Diffusion with Demonstration" (Pan et al., NeurIPS 2024)

*Branch: `feat/mbd-data-guided-comparison`. Companion to
[`01_soft_control.md`](01_soft_control.md) (the product-of-experts /
soft-control derivation this note builds on) and the MBD briefing in
[`../papers/paper_briefings.md`](../papers/paper_briefings.md) §4.1.*

> **Source.** This note is now grounded in the actual text of MBD
> §4.3 ("Model-Based Diffusion with Demonstration", Eqs. 11–13),
> supplied directly. **It corrects an earlier draft of this file**,
> which mistakenly claimed MBD injects demonstrations as a *proposal*
> that washes out asymptotically. That was wrong: §4.3 injects data
> into the **target distribution**. The real difference is *how* the
> demo and model factors are combined — **product vs adaptive
> mixture** — which is a sharper and more useful distinction.

---

## 0. One-sentence answer

Both methods place the demonstration in the **target** and both use
the *same* Gaussian "pull-to-demo" atom; the difference is the
**combination rule**:

- **MBD §4.3** combines model and demo as an **additive mixture**
  (Eq. 11) selected by a **hard, per-sample max-gate** `η ∈ {0,1}`
  (Eqs. 12–13) — an **OR / selection** ("trust whichever of model or
  demo is locally more likely").
- **Our hybrid** combines them as a **multiplicative product**
  (product-of-experts) with a **soft, global** weight
  `λ_prior = λ/β` — an **AND / intersection** ("be simultaneously
  in-data *and* low-cost").

Everything else (our learned score vs their Gaussian demo, the
non-monotone λ-peak, the bias–variance Pareto statement, and — newly
— a *robustness* gap we can now name and fix) follows from
product-vs-mixture and soft-vs-gated.

---

## 1. MBD §4.3 as written (Eqs. 11–13)

MBD factorises its model target into three factors,
`p_0(Y⁰) ∝ p_d(Y⁰)·p_J(Y⁰)·p_g(Y⁰)`, where (MBD notation, **now
confirmed verbatim** against §4.2 / Eq. 10 of the PDF):

- `p_J(Y⁰) ∝ exp(−J(Y⁰)/λ)` — the **objective** (cost-Boltzmann).
- `p_d(Y⁰)` — **dynamic feasibility**, a **Dirac delta** that is
  non-zero only on dynamically feasible trajectories (PDF §4.2,
  line "pd is a Dirac delta function"). In the algorithm it is
  enforced *operationally* by re-rolling the control part `U`
  through the dynamics (the "shooting" step, Alg. 2 line 4), so the
  importance weight in the model-only case reduces to
  `w = p_J·p_g` (Eq. 10d).
- `p_g(Y⁰) ∝ ∏ₜ 1(gₜ(xₜ,uₜ) ≤ 0)` — a product of **constraint
  indicators** (PDF §4.2).

> ⚠️ **Notation clash to flag in the report.** MBD's `λ` is the
> **cost temperature** of the Boltzmann objective `exp(−J/λ)` — it
> plays the role of *our* `β`, **not** of our prior weight
> `λ_prior`. When we write `λ_prior = λ/β` it is *our* `λ` over
> *our* `β`; do not conflate it with MBD's `λ`. Throughout this note,
> "λ" inside an `exp(−J/·)` means a temperature (= our β).

Demonstrations enter as **noisy observations of the desired
trajectory**:

$$
p(Y_{\mathrm{demo}}\mid Y^{0})=\mathcal N(Y^{0},\sigma^2 I),
\qquad\Longrightarrow\qquad
p_{\mathrm{demo}}(Y^{0})\;\propto\;p(Y_{\mathrm{demo}}\mid Y^{0})\;\propto\;\mathcal N(Y^{0}\mid Y_{\mathrm{demo}},\sigma^2 I).
$$

MBD explicitly **rejects the naive posterior**
`p(Y⁰|Y_demo) ∝ p_0(Y⁰)·p(Y_demo|Y⁰)` because, with *suboptimal*
demos, "the demonstration likelihood could dominate `p_0` and mislead
the sampling process." That naive posterior is, notably, **exactly a
product-of-experts** — i.e. structurally *our* combination rule. MBD
deliberately moves away from it. Instead they form a **separated
mixture** target (Eq. 11):

$$
p_0'(Y^{0})\;\propto\;(1-\eta)\,\underbrace{p_d(Y^{0})p_J(Y^{0})p_g(Y^{0})}_{\text{model branch}}
\;+\;\eta\,\underbrace{p_{\mathrm{demo}}(Y^{0})\,p_J(Y_{\mathrm{demo}})\,p_g(Y_{\mathrm{demo}})}_{\text{demo branch}}.
\tag{11}
$$

The mixing weight `η` is chosen as a **hard, per-sample switch**
(Eq. 12):

$$
\eta=\begin{cases}1 & p_d p_J p_g(Y^{0}) < p_{\mathrm{demo}}(Y^{0})\,p_J p_g(Y_{\mathrm{demo}})\\[2pt]0 & \text{otherwise,}\end{cases}
\tag{12}
$$

i.e. *"when a sample has high model-likelihood, ignore the
demonstration and sample from the model; otherwise trust the
demonstration."* This is realised in the SN-IS score estimate by
taking the **max** of the two likelihoods as the importance weight
(Eq. 13):

$$
\bar Y^{0}=\frac{\sum_{Y^{0}\in\mathcal Y_d^{(i)}}Y^{0}\,w(Y^{0})}{\sum_{Y^{0}\in\mathcal Y_d^{(i)}}w(Y^{0})},
\qquad
w(Y^{0})=\max\Big\{\,p_d p_J p_g(Y^{0}),\;\; p_{\mathrm{demo}}(Y^{0})\,p_J p_g(Y_{\mathrm{demo}})\,\Big\}.
\tag{13}
$$

**Design intent (stated):** robustness to *imperfect / partial-state*
demonstrations — the max-gate guarantees a bad demo can never drag a
good model sample down, and a good demo can rescue a region where the
model is weak.

---

## 2. Our hybrid as written

Our unified driver descends the composite score
(`experiments/15_unified_hybrid/unified_loop.py`; derived in
`01_soft_control.md` eqs. 7–9)

$$
s_t^{\mathrm{hybrid}}(\tau)=\nabla_\tau\log p_t^{\mathrm{data}}(\tau)+\lambda_{\mathrm{prior}}\,\nabla_\tau\log p_t^{\mathrm{model}}(\tau),
\qquad\lambda_{\mathrm{prior}}=\lambda/\beta,
$$

whose stationary target is the **product-of-experts** measure

$$
q^\star(\tau)\;\propto\;p^{\mathrm{data}}(\tau)\,\big[p^{\mathrm{model}}(\tau)\big]^{\lambda/\beta},
\qquad p^{\mathrm{model}}(\tau)\propto\exp(-J(\tau)/\beta).
$$

This is **precisely the naive product posterior MBD §4.3 rejects** —
with one addition MBD does not have: a **tunable global exponent**
`λ_prior` that lets us anneal between pure-data (`λ_prior→0`) and
pure-cost (`λ_prior→∞`), with the empirical optimum at an interior
`λ_prior≈1` (`01_soft_control.md` §1, Slate-E 73.8%).

---

## 3. Core contrast: product (AND) vs mixture-max (OR)

| Axis | MBD §4.3 (MBD-Demo) | Our hybrid (PoE) |
|---|---|---|
| Combination of model & demo | **additive mixture** `(1−η)·model + η·demo` (Eq. 11) | **multiplicative product** `data·model^{λ/β}` |
| Set logic | **OR / union** of supports | **AND / intersection** of supports |
| Blend weight | **hard, per-sample** `η∈{0,1}` (Eq. 12); `w=max(·,·)` (Eq. 13) | **soft, global** scalar `λ_prior=λ/β` |
| Demo representation | Gaussian obs `N(Y⁰\|Y_demo,σ²I)` (nonparametric) | learned CFM/DP score **or** nonparametric `demo_*nn` |
| Effect of a *bad* demo | **gated off** (η=0) — cannot mislead | **multiplied in** — persistently biases the target |
| Effect of a *bad* model region | demo **rescues** it (η=1) | data factor only re-weights; cannot override 0-cost-signal |
| Free knob semantics | likelihood-ratio threshold (implicit) | inverse-KL temperature (units-bearing) |

The single deepest statement: **MBD-Demo is a robust *selector*; our
hybrid is a committed *intersector*.** MBD takes the better of two
explanations per sample; we require both to hold at once and tune how
hard the cost factor pushes.

---

## 4. The shared atom: both use a Gaussian "pull" to the demo

MBD's demo factor `p_demo(Y⁰) ∝ N(Y⁰|Y_demo,σ²I)` has score

$$
\nabla_{Y^{0}}\log p_{\mathrm{demo}}(Y^{0})=\frac{Y_{\mathrm{demo}}-Y^{0}}{\sigma^2}.
$$

This is **identical, up to the `1/σ²` scale, to our `--score-mode
pull` term** `(a_target − μ)` (`run_can.py:200–205`) and to the
`demo_1nn` Gaussian-pull score `−(τ−τ̄)/σ_t²`
(`01_soft_control.md` §2.1). So:

> **Our nonparametric / `pull`-mode prior *is* MBD-§4.3's
> demonstration factor.** The two methods share the demonstration
> atom exactly; they differ **only in the rule that fuses it with the
> cost term** — product+soft-λ (ours) vs mixture+max-η (theirs).

This also clarifies why our `--score-mode learned` (the trained
score-bridge `(t·v−x)/(1−t)`) is a *third* thing neither method
emphasises: a non-stationary, per-iterate data score that empirically
destabilises strong DP priors (−36.7 pp), which is why our headline
runs use `pull`, landing us back on MBD's shared demo atom.

---

## 4.5 What Pan's *own* appendix (A.3) says about the product — and why it favours us

Reading Appendix A.3 verbatim turns up **two passages Pan wrote that
are, in effect, arguments for our method.** Both should go in the
report.

**(a) Pan explicitly analyses the product (= our rule) and calls it a
"compromise".** A.3 considers exactly the naive posterior
`p(Y⁰|Y_demo) ∝ p_0(Y⁰)·p_demo(Y⁰)` (the product) and writes:

> *"it will yield update weights `w = w_demo·w_model`, which will
> draw samples to both model and demonstration data. If the
> demonstration data is not optimal, the final solution will be a
> compromise between the model and demonstration data."*

That "compromise" is **precisely the bias term** in our bias–variance
Pareto statement (`01_soft_control.md`). The crucial point: **Pan
treats the compromise as a defect to be eliminated; we treat it as a
*continuously tunable* quantity** indexed by `λ_prior`, with an
*interior optimum* (Slate-E `73.8%` at `λ_prior≈1`, vs `37.5%`
BC-only and the bridge-dominated plateau). MBD-Demo's hard max-gate
**throws away the entire interior of that trade-off** — it only ever
sits at one of the two corners. **Our headline result lives in the
interior MBD-Demo cannot represent.** This is the single cleanest
"ours is strictly more expressive" statement available.

**(b) The max-gate is just a one-time, constant-thresholded
warm-start.** A.3 spells out the gate's temporal behaviour:

> *"In the early stage while `p_J(Y⁰)` is low due to poor sample
> quality, `w_demo` will dominate thanks to the high `p_J(Y_demo)`.
> … As the sample quality improves and `p_J(Y⁰) > p_J(Y_demo)`,
> `w_model` will dominate and the sample will converge to the optimal
> point."*

So the gate flips **once**, when the running sample's cost beats the
**demo's fixed cost** `p_J(Y_demo)` (a constant). Three consequences,
all in our favour:

1. **MBD-Demo is structurally a warm-start, not a fusion.** It is
   "follow the demo until you're as good as the demo, then drop it" —
   the same family as our *rejected* `--score-mode pull` *warm-start*
   ablation, **not** our headline per-step product. We already
   measured that pure warm-starting is weaker than holding the prior
   on every denoising step; MBD-Demo is locked into the weaker mode.
2. **Its ceiling is the demo's own cost.** Because the demo is
   discarded the instant `p_J(Y⁰) > p_J(Y_demo)`, MBD-Demo can never
   use the demonstration's *shape* to push past the demo's cost level
   — the demo only ever supplies a basin to fall into. Our product
   keeps the data factor active at all noise levels, so a good prior
   keeps shaping the solution **after** the cost alone would have
   plateaued (this is exactly the `λ=0→11.7%` vs `λ=1→75%` Can gap:
   the prior is contributing *throughout*, not just as an init).
3. **The "robustness" of (a) in §5 is real but qualitative only.**
   See §8 — Pan demonstrates it on a 1-D toy (Fig. 6) and two
   *figure-only* tasks (Fig. 4), with **no success-rate table**. We
   quantify our own regime with full mass/friction sweeps. So even
   the one axis where MBD-Demo wins (off-distribution demo) is, in
   Pan's paper, *illustrated* rather than *measured* — whereas every
   number in our comparison is a 60-rollout cell.

**Net:** MBD-Demo collapses the data↔cost trade-off to a binary
warm-start switch whose ceiling is the demonstration's own cost. Our
product keeps both experts live on every step, exposes the whole
trade-off through `λ_prior`, and **empirically peaks in the interior
that MBD-Demo cannot reach.** That is the core "our model works better"
claim, and it now rests on Pan's own equations and appendix.

---

## 5. New theory — the max-gate is the robustness mechanism our product lacks

Write the per-sample log-weights `ℓ_mod = log p_d p_J p_g(Y⁰)` and
`ℓ_dem = log p_demo p_J p_g(Y_demo)`. Then:

- **MBD-Demo (Eq. 13):** `log w = max(ℓ_mod, ℓ_dem)`. A demo that is
  *off-distribution* (small `ℓ_dem`) is simply **dropped** whenever
  the model explains the sample better — bounded influence.
- **Ours (PoE):** `log w = ℓ_mod + (β/λ)·ℓ_dem` (the product). A demo
  with small `ℓ_dem` enters as a **large negative additive penalty**
  on *every* sample, dragging the whole target — **unbounded, and
  worst exactly where the demo is least applicable.**

This is not abstract: it **predicts our own asymmetric-robustness
failure**. Under large dynamics shift (this project's
`slate_T2_mass_extremes`: hybrid `0%` at mass 2.5 vs `73%` at mass
0.5), the canonical-operating-point prior `p^data` becomes
off-distribution. In a product, that bad prior keeps multiplying in
and dominates; MBD-Demo's max-gate (Eq. 12) would set `η=0` there and
fall back to pure model. **So MBD §4.3 has a built-in safeguard
against exactly the regime where our hybrid collapses.** Our §08
asymmetric-robustness model (`08_asymmetric_robustness_model.md`) can
be re-read as: *the product has no `η`-gate.*

Equivalently, the two combination rules sit at the endpoints of a
**generalised-mean / tempered-mixture family**:

$$
\log w_p \;=\; \tfrac1p\log\!\big(\tfrac12 e^{p\,\ell_{\mathrm{mod}}}+\tfrac12 e^{p\,\ell_{\mathrm{dem}}}\big)
\;\xrightarrow[p\to+\infty]{}\;\max(\ell_{\mathrm{mod}},\ell_{\mathrm{dem}})\ \text{(MBD-Demo)},\qquad
\text{product} \;\leftrightarrow\; \text{(weighted sum, the }p\to0\text{ / log-linear pool)}.
$$

MBD picks `p→∞` (hard max); we pick the log-linear pool (product).
The whole space in between *was* unexplored at proposal time; §6c records
the branch test that falsified the hoped-for finite-`p` advantage and
showed the additive score-bridge is the load-bearing path.

---

## 6. Tested idea — a gated / tempered-mixture hybrid

This section is preserved as the proposal that motivated branch
`feat/mbd-gated-tempered-hybrid`. The empirical verdict is now negative
for the hoped-for rescue/finite-`p` story: the family is informative as an
ablation, but not the next algorithmic direction.

1. **Hard-gated hybrid (direct MBD-Demo port).** In `unified_loop.py`
   we already compute both the cost-weight `exp(−J_i/β)` and (in
   `pull` mode) the demo target `a_target`. Add a per-sample gate
   `η_i = 1[ exp(−J_i/β) < κ·N(τ_i|a_target,σ²) ]` and set the SN-IS
   weight `w_i = max( exp(−J_i/β), κ·N(τ_i|a_target,σ²) )` (Eq. 13).
   Original prediction: **recovers success at mass = 2.5 / friction =
   3.0** where the fixed product collapses, at the cost of giving up the
   smooth λ-sweep there. Empirical update: not supported at calibrated
   `sigma_demo`; static max-gate and adaptive-sigma gate both remain weak
   or collapse.
2. **Soft-tempered hybrid (the interpolation MBD doesn't take).** Use
   the generalised-mean weight `w_p` of §5 with **finite, tunable
   `p`**. `p→0` recovers our product (best at the canonical operating
   point); large `p` recovers MBD's robustness. A **schedule
   `p(perturbation)`** — log-linear when in-distribution, max-like
   when the prior is off-distribution — would, if it works, **Pareto-
   dominate both** our fixed product and MBD's fixed max. This is a
   clean, novel, falsifiable contribution.
   > **Verdict (§6c): FALSIFIED.** Run at n=90, no finite `p`
   > beats the endpoints; `p=1` is not better than the endpoints, and even
   > at large `M` the tempered family stays far below the additive
   > score-bridge. Kept here as the original hypothesis.
3. **Likelihood-ratio diagnostic.** Even without changing the loop,
   logging `ℓ_dem − ℓ_mod` per rollout would *test* the §5 prediction:
   the mass-2.5 collapse should coincide with `ℓ_dem ≪ ℓ_mod` (prior
   off-distribution) — i.e. exactly where MBD would gate the demo off.

All three reused existing machinery and were tested in the follow-up
slates. The result is a clean negative: rule-level product/max/finite-`p`
fusion is a red herring compared with `combine_mode="score"` versus
`combine_mode="tempered"`.

### 6b. Additional follow-ups grounded in this branch's findings

Each is tied to a *specific* verified result from §1/§4.5/§8 and is a
**mechanism test**, not a hyperparameter sweep (so it is allowed under
the project's "no tuning before consolidation" rule — these resolve
algorithmic questions). Ordered cheapest-first.

4. **σ-annealing ablation (tests §8.3).** Run `demo_1nn` `pull` with
   **fixed σ** (MBD-style) vs our **σ_t-annealed** σ at the canonical
   Can operating point. If σ_t-annealing is what buys our edge, fixed-σ
   should drop toward the BC plateau. Isolates *one* of the two
   mechanisms (annealing) from the combination rule. ~3 cells, reuses
   `demo_1nn`; no new code beyond a `--sigma-schedule {const,decay}`
   flag.
5. **Warm-start vs persistent-product ablation (tests §4.5b — the
   claim "MBD-Demo ≈ warm-start").** Run our own pipeline in
   *warm-start-only* mode (prior active only for high-noise steps,
   gated off below a noise threshold) and compare to the *persistent*
   product (prior on every step). Prediction: warm-start-only lands
   near the MBD-Demo equivalence point and **below** the persistent
   product. If it does, that is direct empirical support for the
   §4.5b equivalence between the max-gate and a warm-start. Reuses the
   existing per-step λ schedule machinery (`lambda-schedule-wireup`).
6. **Demo-quality / gate-ceiling sweep (tests §4.5b-2).** Build demo
   pools at graded quality (optimal, +10%, +30% cost) and measure
   success vs demo cost for (i) the hard-gate port (§6.1) and (ii) the
   best-λ product. Prediction: the **gate plateaus at the demo's own
   cost** while the **product exceeds it** — the cleanest single plot
   for "persistent fusion beats warm-start." Needs the §6.1 gate plus
   a demo-subsampling-by-cost utility.
7. **Tempered-p rule selection (promotes §6.2 from idea → experiment).**
   Sweep the generalised-mean exponent `p ∈ {0, 1, 2, 5, ∞}` (0 =
   product, ∞ = MBD max-gate) at the canonical point **and** at mass
   2.5 / friction 3.0. Goal: show a **finite p** Pareto-dominates both
   fixed rules across the perturbation axis. This is the headline novel
   contribution; `p` is a *rule* choice, not a tuning knob, so it is in
   scope. Needs the `w_p` weight wired into `denoise_loop`.
   > **Done & FALSIFIED — see §6c.** No interior `p` survives at n=90.
8. **Off-distribution generality of the collapse (tests §5).** Confirm
   the product's collapse reproduces on *both* mass 2.5 and friction
   3.0 (orthogonal dynamics axes), and that the gate (§6.1) rescues
   *both*. Guards against the mass-2.5 result being a one-axis artifact.
   Pure re-run of existing slates with the gate flag.

### 6c. Empirical verdict (2026-06-05/06 slates) — what actually held

The combination-rule family from §5/§6.2 was implemented in
`feat/mbd-gated-tempered-hybrid` and then run through the staged slate
family. Implementation pieces:

- **`mbd.tempered_combine(log_w_model, log_w_demo, p, mix_model,
  mix_demo)`** — the generalised-mean fusion of §5. `p→0` = product
  (log-linear pool, our rule); `p→∞` = MBD-Demo max-gate (Eq. 13).
  Returns `gate_frac` (MBD's `η=1` fraction).
- **`denoise_loop(..., combine_mode="tempered", demo_target, sigma_demo,
  demo_sigma_mode, temper_p, prior_lambda)`** — per-sample model-vs-demo
  reweighting; computes the demo branch's fixed `p_J(Y_demo)` constant
  by rolling out the demo once. `combine_mode="score"` (default) is the
  legacy path. Also adds `prior_sigma_mode={"decay","const"}` for the
  #4 σ-ablation on the legacy path. Logs `gate_frac` +
  `ldem_minus_lmod` (§6.3 diagnostic).
- **`log_gate_diagnostic`** (#3, §6.3): passive per-iter logging of
  `gate_frac` + `ldem_minus_lmod` on any run without changing the
  update.
- **CLI** (`run_can.py`): `--combine-mode`, `--temper-p` (accepts
  `inf`), `--sigma-demo`, `--demo-sigma-mode`, `--sigma-schedule`,
  `--log-gate-diagnostic`, `--demo-phase-reward`.
- **Tests/analyzers/slates:** `test_tempered_combine.py`,
  `test_demo_phase_reward.py`, `analyze_tempered_slate.py`, plus the
  `slate_mbd_*_2026_06_04/05/06.sh` family.

Items 1–8 were run on Can (n=30, calibrated `σ_demo`; selected cells at
n=90 / M-scaled) plus Lift/Square/tool_hang. Full readout:
`experiments/15_unified_hybrid/results/SLATE_FINDINGS_2026_06_05.md`.
**The headline §6.2 claim did NOT survive; the §4.5b/§7 claims did.**

- **#7 tempered-p rule — FALSIFIED (the headline "interior-p Pareto-
  dominates both" is not supported).** At n=30 a finite `p=1` looked
  best (20% vs product 6.7% vs max-gate 3.3%), but at **n=90** the
  ranking collapses into noise: `p=0` 12.2%, `p=1` 6.7%, `p=2` 8.9%,
  `p=∞` 10.0% (CI ≈ ±7 pp), with `p=1` in fact the *lowest*. There is
  **no interior temper-p optimum.** The combination-*rule* knob (product
  vs gate vs finite-p) is a **red herring**; what matters is
  `combine_mode = score` vs `tempered` (next bullet).
- **score-bridge ≫ tempered/gated reweighting, and it is M-cheap (new,
  decisive).** The *additive* score-bridge (our actual operating mode —
  add `λ·∇log p_demo` to the cost gradient) gives Can **66.7%**, Lift
  **100%**, Square **23.3%**. The *multiplicative* tempered reweighting
  of the same demo factor gives only 3–20% at M=16. **M-scaling** shows
  why and how far: tempered `p=1` climbs 6.7→10→**30%** as M=16→64→256,
  while the score-bridge is **M-insensitive** (66.7% at M=16 = 63.3% at
  M=256). So tempered is **variance-limited *and* structurally weaker** —
  even at 16× the sample budget it plateaus at ~half the bridge. The
  practical recommendation is the additive score-bridge, not the
  generalised-mean fusion of §5/§6.2.
- **#4 σ-schedule — confirmed mild:** constant σ 73.3% > decayed σ
  66.7% (Can, score path).
- **#5 warm-start vs persistent — confirmed strongly:** pure warm-start
  (`λ=0`) 13.3% vs **persistent bridge 73.3%**; every annealed schedule
  (linear/cosine/inverse) is worse (20–40%). Direct support for "the
  fusion must persist every step," i.e. MBD-Demo's warm-start framing is
  the weaker regime (§4.5b).
- **#6 demo-quality gate-ceiling — confirmed:** soft product ≫ hard gate
  at every demo band (optimal +43.3, mid +36.7), both fail on a bad demo
  (+0.0). Cleanest "soft beats hard" plot.
- **#1/#8 hard-gate off-distribution rescue — NEGATIVE.** At the
  calibrated `σ_demo` the gate does **not** rescue mass-2.5/friction-3.0
  (both product and max-gate ≈0–3%, `gate_frac` stays ~0.3, never →1).
  The gate only dominates (`gate_frac→1`) at large `σ_demo`, but large
  `σ_demo` kills on-distribution performance — **no single static
  `σ_demo` both rescues off-dist and preserves on-dist.** (This re-opens
  the adaptive-`σ_demo` idea below.)
- **σ_demo calibration is task-specific:** balanced (`ℓ_dem−ℓ_mod≈0`) at
  σ≈1 for Can/Lift, σ>16 for Square; on tool_hang the gate never engages
  at any σ tested.

**Residual open items — now IMPLEMENTED & under test (2026-06-06):**
- **#1 dense-revival on a solvable task** (`slate_mbd_dense_revival`):
  the §7.1 raw-vs-dense × product/gate/score contrast on Can+Lift, where
  the gate actually engages (the sparse slate's tool_hang could not test
  it). No new code.
- **#2 adaptive `σ_demo`** (`--demo-sigma-mode adaptive`): scales the
  demo Gaussian by the model-vs-demo best-cost gap
  (`sd = σ_demo·exp(-β·(best_cost − demo_cost)/λ)`) so the gate fires
  off-dist and relaxes on-dist — the only static-σ fix for the negative
  #1/#8 result. Slate `slate_mbd_adaptive_sigma` (static vs adaptive on
  canonical + mass2.5 + fric3.0). Lower priority since the tempered/gated
  family loses to the additive bridge regardless, but it closes the
  off-dist-rescue question.
- **#3 demo-phase pseudo-cost** (`--demo-phase-reward`, route 3):
  zero-train dense reward = fraction along the nearest demo. Slate
  `slate_mbd_demo_phase` tests whether it gets tool_hang off 0% where
  hand-shaping (H=300) did not, and re-tests the §7.1 revival on Can.
- **Still open:** tool_hang may also need routes 2/4 (learned value /
  subgoal waypoints) if the demo-phase progress signal alone is
  insufficient — to be decided from the #3 results.

---

## 7. Where the distinction stops mattering — the flat-cost regime

When the in-horizon cost `J` is flat (sparse reward never reached;
this project's `tool_hang`/`transport` hard-task baselines: all 240
rollouts show `best_cost ≡ 0`, `ESS ≡ M`), `ℓ_mod` is the *same
constant* for every sample. Then:

- **MBD-Demo:** the max-gate (Eq. 12) trivially selects the demo
  branch everywhere (`η=1`) — it degenerates to **replaying the
  demo** (its Gaussian pull), with no cost-driven improvement.
- **Ours:** the product degenerates to `p^data` alone — **also pure
  behaviour-cloning** (`λ_prior→0` endpoint).

So under flat cost both methods collapse to "follow the demo," and
the product-vs-mixture distinction evaporates. The fix is a **dense /
shaped cost** (`feat/dense-reward-slate-K`), which is upstream of both
combination rules. Honest boundary: our advantage *and* MBD's
robustness gate both require in-horizon cost gradient.

### 7.1 How to address the sparse-reward regime (where our model underperforms)

**Root cause, stated precisely.** A sparse terminal reward gives a
*flat* `J` over the horizon for every rollout that never reaches the
goal, so the cost-Boltzmann `p_J ∝ exp(−J/β)` is **uniform across
samples** → SN-IS weights are uniform (`ESS ≡ M`) → there is **no
cost gradient to rank candidates**. The hybrid then has only one live
signal — the data/demo factor — so it can do no better than BC
(`λ_prior→0` endpoint). **This is not a combination-rule problem**
(product vs gate is irrelevant when `ℓ_mod` is constant); it is a
*signal-density* problem that must be fixed **upstream** of the fusion
rule. Four routes, in increasing training cost:

1. **Potential-based shaping (preferred — only route with an
   optimality guarantee).** Add a per-step shaping term
   `F_t = γΦ(s_{t+1}) − Φ(s_t)` to the cost, with `Φ` a progress
   potential (learned value, or distance-to-goal). Ng et al. (1999)
   guarantees the *optimal policy is unchanged*, but `J` is now dense,
   so `p_J` ranks samples again, `ESS < M`, and **the product's `λ`
   knob is live again**. This is the principled fix: unlike an ad-hoc
   dense reward, it cannot move the optimum.
2. **Learned cost-to-go / value as the cost itself (connects to the
   open `value-baseline-headline-eval` arm).** Replace the sparse
   terminal `J` with `V̂(x_T)` (or per-step `Σ γ^t r̂`). Dense by
   construction; the value-baseline arm already staged in this repo is
   exactly this fix and should be evaluated on `tool_hang`/`transport`.
   Empirical update: the Can pure-MBD `V_φ` control-variate cell did not
   rescue the baseline, but tool_hang still has no learned subgoal/value
   test; this remains the main viable route.
3. **Demo-phase pseudo-cost (zero-train, fits our nonparametric
   stack).** Use *arclength / phase along the nearest demo* as a
   monotone pseudo-cost (reward = fraction of the demo completed).
   Dense, needs no value training, and reuses the `demo_1nn`
   machinery — it converts the *static* demo target into a dense
   *progress* signal. Caveat: inherits the demo's suboptimality (a bad
   demo gives a misleading potential — same bias as a bad prior).
   Empirical update: implemented in `demo_phase_reward.py`; it produces
   large dense rewards but 0% tool_hang success across score/product/gate
   and sigma/scale variants, so demo-phase alone is closed as a rescue.
4. **Reachability fix (when the goal is simply unreachable in `H`).**
   In `tool_hang`/`transport`, `best_cost ≡ 0` partly because the goal
   is never reached *within the horizon*. Longer `H` or **demo-derived
   subgoal waypoints** (each segment gets a reachable terminal cost)
   restore a non-flat `J` without any reward engineering. Empirical
   update: longer `H=300` plus sigma calibration still leaves tool_hang
   at 0%, so the remaining reachability route is explicit subgoals rather
   than simply extending the horizon.

**Branch-specific punchline — our product revives faster than MBD's
gate as signal returns.** Once *any* weak dense signal is added, the
two rules respond differently:

- **Our product** shifts its target **continuously**, proportional to
  `λ_prior × ∇(shaping)`. Even a noisy or partial value/potential moves
  the solution off the BC endpoint immediately.
- **MBD's max-gate** only responds once the model branch's likelihood
  **crosses the demo's fixed threshold** `p_J(Y_demo)`. A weak shaping
  term that does not cross that threshold leaves `η = 1` — still pure
  demo-replay, no improvement.

So as you densify the cost, **our advantage returns continuously while
MBD-Demo's is gated behind a threshold** — a concrete, testable
prediction that ties directly to follow-up #6/#7 and the value-baseline
arm. The honest caveat: routes 2–3 introduce their own
mis-specification bias and (route 2) training cost; route 1 is the only
one that is provably optimum-preserving.

> **Empirical status (§6c sparse slate, tool_hang).** The *signal-
> density* half is **confirmed**: raw sparse `J` → mean reward `0`
> (flat, no gradient); adding dense shaping → mean reward 70–112. The
> *product-revives-faster-than-the-gate* half is **not yet shown**: at
> the σ_demo tested the gate never engaged (`gate_frac=0`) and every arm
> stayed at 0% success (the goal is unreachable in `H` — route 4), so the
> continuous-vs-threshold contrast needs a solvable dense task (Lift/Can
> with a shaped cost) before it can be claimed.

---

## 8. Verified against the PDF (was: open items)

All three items below were **checked word-for-word** against the
NeurIPS-2024 PDF (`docs/papers/MBD_Model_Based_Diffusion_NeurIPS2024.pdf`,
§4.2–4.3 + Appendix A.3). Resolutions:

1. **Factor definitions & whether the demo branch carries cost —
   RESOLVED (§1).** `p_J ∝ exp(−J/λ)`, `p_d` = Dirac feasibility (rollout-
   enforced), `p_g = ∏ₜ 1(gₜ≤0)`. The Eq. 11/13 terms `p_J(Y_demo)`,
   `p_g(Y_demo)` are evaluated **at the demo trajectory** and are stated
   to be **"two extra constant terms … to ensure the demonstration
   likelihood is properly scaled to match the model likelihood"** (PDF
   §4.3). So the demo branch *does* know the demo's own cost — **but
   only as a fixed scalar threshold**, never as a per-sample cost
   gradient. This is what makes the gate a *constant-thresholded*
   switch (§4.5b): the demo branch can localise the basin but cannot
   itself be cost-improved.

2. **Fig. 6 / A.3 robustness numbers — RESOLVED, and it favours us.**
   There is **no quantitative robustness table.** Fig. 6 is a **1-D
   nonconvex toy** (constraint `||Y|−0.4|>0.3`) showing convergence to
   the optimum from a *suboptimal* demo; the real-robot data-augmented
   cases (Car UMaze Navigation, Humanoid Jogging, Fig. 4) are
   **figure-only / qualitative** — no success rates, no seed bars. The
   headline MBD success table (Tables 2–3) is **vanilla MBD only (no
   demo)**, vs CMA-ES/CEM/MPPI/RL on Brax locomotion + PushT. So the
   *specific* MBD-Demo robustness claim we credit in §5 is **never
   numerically quantified in the paper** — our mass/friction sweeps are
   strictly more rigorous evidence on the same axis (§4.5b-3).

3. **`σ²` fixed vs annealed — RESOLVED: FIXED.** The PDF defines
   `p(Y_demo|Y⁰) ∼ N(Y⁰, σ²I)` with a **single fixed observation
   noise**; there is **no σ-schedule** tied to the diffusion level.
   Therefore our `σ_t`-scaled `demo_1nn` pull (which *tightens* as the
   noise level decays — `01_soft_control.md` §2.1) is a **genuine
   mechanism MBD-Demo lacks**: ours sharpens the data constraint over
   the chain, theirs holds a constant-width Gaussian. Add this to the
   "ours, distinctively" list.

---

## 9. Bottom line for the report

- **Corrected framing.** MBD §4.3 and our hybrid both put data in the
  target and share the *same* Gaussian demo-pull atom (= our
  `pull`/`demo_1nn` score). They differ in the **fusion rule**:
  MBD = adaptive **mixture** with a hard per-sample **max-gate**
  (Eqs. 11–13, robust-OR); ours = **product-of-experts** with a soft
  global **λ** (committed-AND).
- **Ours, distinctively (why ours works better):**
  1. **Expressivity.** Pan's own A.3 (§4.5a) calls the product a
     "compromise" between model and demo and *eliminates* it with a
     binary max-gate. We keep that compromise as a **continuously
     tunable** `λ_prior` with an **interior optimum** (Slate-E
     `73.8%`) that the max-gate's two corners cannot represent.
  2. **Persistent fusion, not warm-start.** A.3 (§4.5b) shows the
     gate flips once at the demo's fixed cost `p_J(Y_demo)`, so
     MBD-Demo is a demo→model **warm-start** capped at the demo's
     cost. Our product keeps both experts live every step (`λ=0→11.7%`
     vs `λ=1→75%` Can), so a good prior shapes the solution past where
     cost alone plateaus.
  3. **Noise-annealed data constraint.** `σ²` is **fixed** in MBD
     (§8.3); our `σ_t`-scaled `demo_1nn` pull tightens over the chain.
  4. **Rigour.** Our advantage axis is measured (60-rollout cells,
     mass/friction sweeps); MBD-Demo's robustness is **figure-only**
     (Fig. 4/6, no success table — §8.2).
  Plus a units-bearing `λ_prior=λ/β`, a learned data score option
  (CFM/DP), and the bias–variance Pareto guarantee.
- **Theirs, distinctively (honest):** a built-in robustness gate
  against suboptimal / off-distribution demos — which **our product
  lacks**, and which **predicts our mass/friction-extreme collapse**
  (§5). But this gate is *qualitative* in the paper (§8.2) and comes
  at the cost of expressivity (§4.5).
- **New contribution on the table (§6) — TESTED, §6c.** The proposed
  tempered/gated hybrid that interpolates product↔max was run on Can
  (n=90), Lift, Square and tool_hang. **It does *not* Pareto-dominate:**
  no interior temper-`p` beats the endpoints, and the whole
  *multiplicative* tempered/gated family loses to our *additive*
  score-bridge (Can 66.7% vs ≤30% even at 16× the MBD sample budget;
  the bridge is also M-insensitive). The defensible contributions that
  *did* hold are (a) **persistent fusion ≫ warm-start** (73.3% vs
  13.3%), (b) **soft product ≫ hard gate across demo quality**, and
  (c) the **flat-cost signal-density boundary** (§7/§7.1, confirmed:
  raw-sparse → 0 reward, dense shaping → reward returns). The max-gate
  earns no rescue at a calibrated `σ_demo` (§6c #1/#8, negative).
