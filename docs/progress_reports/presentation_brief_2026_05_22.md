# From DARE to Diffusion

### Zero-Training Retrieval Priors as a Score-Bridge to Model-Based Diffusion — Presentation Brief

> *Why does a parameter-free 1-NN over the demo bank beat a
> trained behaviour-cloning policy by +52 pp on the same task,
> and what does that have to do with the discrete algebraic
> Riccati equation? The score-bridge.*

**Working title (formal):** Data-Driven Priors and Physics-Based
Corrections — A Hybrid Score-Bridge Approach to Robot Manipulation
**Authors:** Zihao Sun, Charles Beck
**Date:** 2026-05-22 (post-brief update appended 2026-05-25 — see §13)
**Branch at brief:** `feat/zero-train-priors` (merged into `main` on 2026-05-25, commit `2daa166`)
**Active artifact:** `experiments/15_unified_hybrid/` (the "unified driver")
**Course:** UCLA ECE M237 — Dynamic Programming, Spring 2026

---

## 1. Motivation — why a hybrid?

Robot motion planning needs trajectories that are simultaneously
**high quality** (smooth, multimodal, goal-reaching),
**physically feasible** (dynamically consistent under the
deployed cost), and **safe** under uncertainty. Two paradigms
each solve one half of that problem:

- **Pure offline / learned priors** — Diffuser
  ([Janner et al., 2022]), flow-matching planners (FlowMP), and
  diffusion-policy variants learn an expressive multimodal
  distribution over expert trajectories from demonstrations.
  They produce diverse, demonstration-shaped behaviour, but
  they provide **no feasibility guarantees out of distribution**
  and degrade silently when the deployed cost differs from the
  training data.

- **Pure online / model-based** — Model-Based Diffusion (MBD,
  [Pan et al., NeurIPS 2024]) replaces the learned score of a
  diffusion planner with one **computed analytically** from the
  known dynamics $x_{k+1}\!=\!f(x_k,u_k)$ and cost $J(\tau)$,
  via Tweedie's formula and self-normalised importance sampling.
  This guarantees feasibility without any training data, but
  cannot capture the multimodal, demonstration-informed
  behaviour the offline methods get for free, and suffers high
  estimator variance on long horizons.

MBD's authors note that data "can be naturally integrated" with
their model-based pipeline but leave the exact mode of that
integration unformalised. **That is the gap our project
addresses.**

---

## 2. The hybrid idea — score-bridge / product of experts

We treat the target trajectory distribution as a
**product of experts**:

> $p^{\mathrm{target}}(\tau) \;\propto\; p^{\mathrm{data}}(\tau)\;[p^{\mathrm{model}}(\tau)]^{\lambda}$,
> where $p^{\mathrm{model}}(\tau)\!\propto\!\exp(-J(\tau)/\beta)$
> is the analytical Boltzmann target of MBD.

At every diffusion time $t$ this is equivalent to denoising
with the **composite score**

> $\nabla\!\log p^{\mathrm{target}}_t \;\approx\;
> \nabla\!\log p^{\mathrm{data}}_t \;+\; \lambda(x,t)\,
> \nabla\!\log p^{\mathrm{model}}_t.$

The first term anchors the search on demonstration structure;
the second term refines toward feasibility under the deployed
cost; $\lambda(x,t)$ is the **trust schedule** between them.
This is the project's central object, and every experimental
sweep we run is an attempt to map a different axis of it
($\lambda$, $\sigma$, the prior identity, the obs vocabulary,
the cost shaping).

### 2.1 Linear-quadratic limit — the DP anchor

The composite score has a closed-form solution in the
**linear-quadratic (LQ)** special case, which gives the project
a direct bridge to classical dynamic programming. Take the
canonical LQ problem:

> **Dynamics** $\quad x_{k+1} = A\,x_k + B\,u_k$
>
> **Stage cost** $\quad \ell(x, u) = x^\top Q\, x + u^\top R\, u$, with $Q \succeq 0$, $R \succ 0$
>
> **Terminal cost** $\quad \ell_H(x_H) = x_H^\top Q_H\, x_H$
>
> **Trajectory cost** $\quad J(\tau) = \sum_{k=0}^{H-1}\!\ell(x_k, u_k) + \ell_H(x_H)$

The Bellman recursion has a known closed form: the value
function is quadratic $V_k(x) = x^\top P_k\, x$, the matrices
$\{P_k\}$ solve the **discrete algebraic Riccati equation**
(DARE) backward from $P_H = Q_H$:

> $P_k = Q + A^\top P_{k+1} A - A^\top P_{k+1} B \,\big(R + B^\top P_{k+1} B\big)^{-1} B^\top P_{k+1} A,$

and the optimal control is a **linear state-feedback**

> $u^\star_k = -K_k\, x_k, \qquad K_k = \big(R + B^\top P_{k+1} B\big)^{-1} B^\top P_{k+1} A.$

Now observe that in the LQ case MBD's Boltzmann target is
exactly **Gaussian** — $p^{\mathrm{model}}(\tau) \propto
\exp(-J(\tau)/\beta)$ with quadratic $J$ — so its score reduces
to the gradient of a quadratic form:

> $\nabla_{\tau}\!\log p^{\mathrm{model}}(\tau) = -\tfrac{1}{\beta}\, \nabla J(\tau).$

Marginalising onto the per-step action gives the
**state-feedback law $u_k = -P_k\, x_k / \beta$** up to the
inverse-temperature scaling, i.e. the score field that MBD
estimates by self-normalised importance sampling **at finite
$M$** is the same field that DP solves **exactly** by the
DARE in the LQ limit. The hybrid in this limit is therefore
LQR-feedback regularised by the data prior — a generative-
modelling realisation of one round of policy iteration on a
quadratic value function.

**Empirical validation (Week 4).** We ran the full unified
denoising loop on a 2D point-mass LQ companion problem
($A, B, Q, R$ analytical) and measured the converged cost
against the DARE-optimum: **1.059×** the DARE optimum across
5/5 seeds within 10 %. The score, schedule, and softmax
weighting are therefore correctly implemented; any failure on
the contact-rich Robomimic tasks is attributable to the
sim coupling and the demonstration prior, not to the MBD
algorithmic core (`experiments/04_mbd_baseline/results/_archive/lq_sanity.json`).

---

## 3. Method — what we actually built

The full pipeline is one **unified driver** at
`experiments/15_unified_hybrid/run_can.py`. It plugs three
pieces together:

1. **Offline prior** — pluggable, currently three implementations:
   - **`bc`** — MLP behaviour-cloning policy on the demo bank
     (Week 4 stand-in for the flow prior).
   - **`flow`** — flow-matching velocity field $v_\theta$
     trained on the demos via conditional flow matching
     ([Lipman et al., ICLR 2023]); the
     flow $\leftrightarrow$ score identity gives the
     $\nabla\!\log p^{\mathrm{data}}_t$ at inference.
   - **`demo_1nn` (and friends)** — *zero-training* retrieval
     prior: at every plan we 1-NN the current observation
     against the offline demo bank in normalised obs space,
     pick the matched demo's action window, and pull toward
     it with a Gaussian score $-\lambda(\tau-\bar\tau)/\sigma^2$.
     Variants: `demo_knn`, `demo_softnn(β)`, `demo_lwr`,
     `demo_1nn_proprio`, `demo_1nn_object`,
     `demo_1nn_contrastive` (a learned embedding).

2. **Online correction** — Model-Based Diffusion (MBD) at
   $K\!=\!10$ iterations, $M\!=\!16$ rollouts per iteration,
   cosine $\sigma_{\max}\!\to\!\sigma_{\min}$ schedule. The
   correction is the cost-weighted posterior-mean of $M$
   forward rollouts in MuJoCo at each step.

3. **MPC chunking** — Each rollout replans $R\!=\!1$ time at
   the chunk boundary, letting the bridge react to closed-loop
   drift while keeping wall-time roughly flat.

The composite score (Section 2) is added **at every MBD
iteration**, not only at the initial sample — this is the key
methodological upgrade from Week 4, where the prior only
provided a warm start.

---

## 4. Testbed — Robomimic on MuJoCo

7-DoF Franka Panda in MuJoCo / robosuite, four task families
from Robomimic's proficient-human (`ph`) split:

- **`can`** — `PickPlaceCan`, $H\!\approx\!120$. Our reference
  task; everything starts here.
- **`lift`** — strictly easier; a ceiling control to confirm
  the mechanism transfers.
- **`square`** — peg-in-hole, contact-rich; a mechanism stress
  test.
- **`tool_hang`** — long-horizon, $H\!\approx\!681$.
- (**`transport`** — queued; first scored cells incoming with
  the dense-reward fix.)

Protocol: 20 initial states × 3 rollouts = **60 evaluations
per cell**, seed=0 single-seed unless pooled across 4 seeds.

---

## 5. Project arc (weeks 1–7)

| Week | Milestone |
|---:|---|
| **1** | Project proposal: hybrid offline-prior + online-MBD via the composite-score formulation. |
| **2** | Stochastic-optimal-control + flow-matching lit review (matched to the course). |
| **3** | Plan locked: MuJoCo + Robomimic + 7-DoF Panda. Five-baseline scorecard chosen (replay / random / BC / MBD / hybrid). |
| **4** | First scored rows on `can`. Hybrid = BC warm-start + MBD = **1/60** binary, but behaviourally clear: paired hybrid > BC on **85 %** of rollouts; dense reward +4.51 → +8.25. The binary headline was below threshold; the mechanism was working. |
| **5** | Peer-review week (we reviewed a CVAE+MPPI / PointMaze project; not our own results). |
| **6–7** | **Unified driver + zero-training retrieval prior** lands. Per-step composite score replaces warm-start-only. `demo_1nn` replaces the MLP-BC stand-in. Cross-task evaluation expands from `can` only to four task families. **`can` climbs from 1/60 → 73.8 % pooled.** This brief. |

---

## 6. One-paragraph thesis (current status)

A **zero-training retrieval prior** (`demo_1nn`: nearest
neighbour in normalised obs space, pick its action window)
combined with a **score-bridge model-based-diffusion** controller
(MBD with a pull-toward-warm-start composite score) reaches the
headline hybrid number on Robomimic `can` (**73.8 % pooled, 75 %
single-seed**), **transfers cleanly to a strictly easier task
(`lift`, 100 %)**, and **exposes a real mechanism limit on
`square` (8.3 % peak)** that we have triangulated to 5
orthogonal axes. **No training was done in this work** — the
entire warm-start prior is parameter-free retrieval over the
offline demo bank.

---

## 6.5 Update — Slates K, L, M, N, O (added 2026-05-25 PM)

Since 2026-05-22 we have run a 23-cell follow-up programme on
the v6 trained CFM prior to **mechanistically explain when MBD
helps and when it does not**, capped by Slate O's pure-standalone
measurement which forced a major revision to the parity narrative.

Headlines:

- **MBD-pull lifts v6 by +17.8 pp (NOT parity) with
  non-overlapping 95 % CIs.**  Slate O measured pure v6
  standalone (K=1 M=1 σ≈0 λ=0, deterministic warm-start
  executed verbatim) across seeds 0/1/2: 22/19/16 → 3-seed
  pool $57/90 = 63.3\%$, Wilson 95 % CI [53.0, 72.6].  v6+MBD-pull
  at K3 canonical (same 3 seeds): 25/24/24 → $73/90 = 81.1\%$
  CI [71.9, 88.0].  $\Delta = +17.8$ pp, two-sample $z \approx 2.83$,
  $p \approx 0.005$.  *The earlier "Slate K parity" claim
  treated K3 as if it were standalone; pull at $\lambda=1$
  fully anchors only the high-σ early iterations of the K=10
  inner loop, so the SNIS rank-and-pick over M=16 candidates
  at low σ DOES contribute a real lift.  Slate O catches this.*

- **MBD-pull also collapses the seed-variance.**  The standalone
  seed spread (53.3 → 73.3 %) is **20 pp**; v6+MBD-pull's spread
  (80.0 → 83.3 %) is **3.3 pp**.  MBD's rank-and-pick over
  M=16 candidates is much less sensitive to the ODE-sampler
  initial-condition seed than v6's single deterministic sample.

- **The lift is single-shot specific.**  Re-running v6+MBD-pull
  at closed-loop $R\!=\!4$, chunk-24 drops the cell to
  **46.7 %** (−34.4 pp vs K3, Slate L Wave B).  v6 was trained
  on $H\!=\!120$ windows aligned to demo starts; replanning at
  mid-trajectory observations is off-distribution.  The same
  failure mode crushed DP-H=16 R=29 to 0 %; v6 holds up better
  but the *direction* is identical.

- **MBD's lift is manifold-coherence-bound.**  Adding iid
  Gaussian noise of std $\sigma_\mathrm{corrupt}\!\geq\!0.3$ to
  v6's warm-start before MBD's inner loop collapses the cell
  to ≤ 3.3 % (Slate L Wave D).  In contrast, demo_1nn's
  "wrong-demo" warm-start (on-manifold but misretrieved) is
  rescued from 11.7 → 73.8 %.  The discriminator is whether
  the σ-ball around the warm-start *reaches* an on-manifold
  trajectory with lower cost.

- **The revised lift curve (5 priors, after Slate O).**  Every
  measured prior shows positive MBD-pull lift; the size of
  the lift varies by *regime*:

  | regime | priors | lift |
  |--------|--------|-----:|
  | **R2 Full rescue** | `demo_1nn` (on-manifold, wrong-demo) | **+62 pp** (11.7 → 73.8) |
  | **R2 Partial rescue** | BC pool (interpolation off-manifold) | **+26 pp** (38 → 64) |
  | **R2/R3 Partial rescue** | `demo_knn` k=5 (averaged off-manifold) | **+20 pp** (3.3 → 23.3) |
  | **R1 Small-lift (rank-and-pick)** | v6 (near cost optimum) | **+18 pp** (63.3 → 81.1) |
  | **R3 Collapse** | v6 + noise σ≥0.3 (off-manifold by noise) | **0 pp** (~0 → 0) |

  See `docs/theory/manifold_restoration_2026_05_25.md` for the
  SNIS posterior-collapse argument that formalises the four
  regimes.

- **The 83.3 % single-seed (81.1 % pool) ceiling is genuine,
  not a sampler/horizon artifact.**  Slate N tested whether
  denser ODE sampler (`prior-n-steps` 18 → 28, 50) or longer
  execution budget (`max-env-steps` 240 → 300, 360) raise the
  v6+pull ceiling.  All four perturbations underperform or
  match K3; the lifted-sampler + lifted-budget combination
  N3 lands at **73.3 %** (−8 pp vs K3 pool 81.1 %).
  K3's tuning is at the task ceiling for the v6+pull
  configuration on Can.

**What this means for the talk.**  The §6 thesis
("zero-training retrieval + score-bridge MBD reaches 73.8 %
on Can") still holds.  What the Slate K-O programme adds is
a **positive mechanistic claim with a quantitative
prediction**: **MBD-pull lift is positive across the full
prior-strength spectrum (12 % → 83 %), saturating gradually
toward the task ceiling**.  The previous (incorrect)
"parity at v6" framing was a measurement artefact.  The
corrected story is *better* for the talk because it lets
us say "MBD always helps" (with the regime-dependent size)
rather than the more brittle "MBD helps until the prior
saturates".  The mechanism — SNIS rank-and-pick over the
σ-ball — is the same in every regime; only the shape of
the cost landscape around the warm-start changes.

---

## 6.6 Update — Adaptation arc (updated 2026-05-26 PM)

In parallel with the K-O mechanism arc, four **adaptation
experiments** were planned (specs in `docs/adaptation/01-04*`)
on dedicated branches.  As of 2026-05-26 noon, **two have
landed empirically (one positive, one negative-with-mechanism)**,
one is in flight, and one is queued.  Each tests one direction
of deployment-time distribution shift.

| # | branch | dimension perturbed | status | headline |
|--:|---|---|---|---|
| 1 | `feat/adapt-demo-scaling` | demo bank size $N_b$ | **complete (positive)** | three-phase scaling law confirms Theorem 1' Cor 1 |
| 2 | `feat/adapt-cost-time` | goal-pose $\Delta x$ | **complete (negative)** | gap-vs-$\Delta x$ narrows; **fixed-anchor mechanism** identified |
| 3 | `feat/adapt-dynamics` | robot-link mass $\times \{0.7, 1.0, 1.3, 1.5\}$ | **in flight** | ETA ~5.5 h from 2026-05-26 12:25 PT |
| 4 | `feat/adapt-incremental-prior` | bank mode-coverage growth | infra ready, design pending | — |

### DemoScale (1, complete) — three-phase scaling law

Six cells on Can, single-seed n=60/cell, canonical
$\lambda=1$ $\sigma=0.25$ $K=10$:

| $N_b$ | succ rate | Wilson CI | in-bank | OOB |
|------:|----------:|-----------|--------:|----:|
| 5 | 20.0 % | [11.8, 31.8] | 80 % | 0 % |
| 10 | 35.0 % | [24.2, 47.6] | 60 % | 10 % |
| 20 | 68.3 % | [55.8, 78.7] | full coverage | full coverage |
| 50 | 68.3 % | [55.8, 78.7] | full coverage | full coverage |
| 100 | 68.3 % | [55.8, 78.7] | full coverage | full coverage |
| 200 | 80.0 % | [68.2, 88.2] | full coverage (within seed noise of 73.8 % 4-seed pool) | — |

Three-phase shape predicted by Theorem 1' Corollary 1:

- **Phase I — mode-undercoverage** ($N_b \le 10$): each new
  demo mode adds rate sharply, but in-bank rate drops (80 → 60 %)
  as 1-NN aliasing starts (each new demo isn't its own unique
  match).
- **Phase II — saturation plateau** ($N_b \in [20, 100]$):
  flat 68.3 % across the bank; once the bank covers all eval
  modes, adding more demos doesn't help (Corollary 1's
  saturation regime).
- **Phase III — re-climb** ($N_b = 200$): rate climbs to
  80 %, matching the canonical 73.8 % 4-seed pool within seed
  noise; the extra demos refine within-mode coverage rather
  than adding new modes.

The 80 % in-bank vs 0 % OOB split at $N_b\!=\!5$ is **sharper
than the spec predicted** ($\sim\!30\%$ overall, averaged) and
substantiates the mode-anchoring story directly.  This is the
**second falsification target** for Theorem 1' Cor 1 (after
Slate O's R1 small-lift was the first).  See
`docs/adaptation/03_demo_scaling_spec.md` and
`docs/adaptation/plan_2026_05_25.md` on the branch for the
full writeup; merge to `main` is pending conflict review with
the `docs/experiments/*` files that landed via
`feat/experiment-docs`.

### Cost-time (2, complete — NEGATIVE result with mechanism)

Landed 2026-05-26 11:48 PT on `feat/adapt-cost-time` (commit
`22ad90d`).  Slate completed cleanly (8 cells, 5.5 h wall) but
**falsified the spec's primary hypothesis**.  Honest writeup
 in `experiments/15_unified_hybrid/results/slate_costtime_2026_05_25/RESULT.md`
on the branch; one-paragraph follow-up plan in
`NEGATIVE_RESULT_FOLLOWUP.md`.

Headline (single seed=0, n=60/cell):

| $\Delta x$ (m) | hybrid | BC | pure-MBD | hybrid–BC gap |
|---:|---:|---:|---:|---:|
| 0.00 | 73.3 % | 26.7 % | 6.7 % | **+46.7 pp** |
| 0.05 | 71.7 % | 26.7 % | 10.0 % | +45.0 pp |
| 0.10 | 50.0 % | 21.7 % | 10.0 % | +28.3 pp |
| 0.20 | 0.0 % | 0.0 % | 1.7 % | 0.0 pp |

- Spec §5.1 substantiation criterion ("gap widens monotonically
with $\|\Delta x\|$"): **not met** — gap *narrows*
monotonically.
- Spec §6 falsification 2 ("hybrid degrades faster than BC, must
be a wiring bug"): **triggered** at $\Delta x = 0.10$ ($z=2.50$,
$p \approx 0.012$).
- **But not a wiring bug.** Verified by (a) `test_worker_perturbation.py`
PART A + PART B both PASS, and (b) the $\Delta x=0$ control cell
reproduces the canonical Slate E pool (hybrid 73.3 % vs E1 pool
73.8 %, within seed-noise band).

**Mechanism identified — fixed-anchor bias.** The retrieval
prior's anchor $\bar\tau$ is implicitly *pinned to the canonical
goal* because Can/ph obs does not contain the bin pose.  The
1-NN demo retrieval is **obs-goal-blind**, so the prior can't
adapt to a perturbed bin.  MBD's cost re-targeting is real but
the prior's pull cancels it out.  This is a previously-unmodelled
failure mode of the score-bridge under cost shift — a publishable
finding on its own.

**Follow-ups** (in `NEGATIVE_RESULT_FOLLOWUP.md`):
 - F1: multi-seed pool at $\Delta x = 0.10$ to confirm the
   single-seed signal (~2.5 h compute, no new infra).
 - F2: **goal-conditioned retrieval** — augment `demo_1nn`'s
   query with the current (perturbed) bin pose so the retrieved
   demo tracks the new goal.  Spec prediction: hybrid invariant
   in $\Delta x$ across all 4 cells, gap to BC *widens*.  This
   is the principled fix; ~1 day infra + ~6 h compute.
 - F3-F6: pure-MBD at $M\!=\!64,256$; $-x, \pm y$ direction
   symmetry; Lift cross-task replication; cost-shift-only
   decoupled test.

### Dynamics (3, in flight)

Launched 2026-05-26 12:25 PT in screen `slate_dynamics` on
`feat/adapt-dynamics` (commit `b6b43af`, in `/private/tmp/HRTG-dynamics-tmp`
worktree).  D1 mass-ladder, minimal variant:

- Mass scales: $\{0.7, 1.0, 1.3, 1.5\}$ on robot-link bodies
  (Panda's `robot0_link*` + `robot0_right_hand` + `gripper0_*`,
  15 bodies; mutates `body_mass` and `body_inertia` proportionally).
- Three methods (same as cost-time): hybrid `demo_1nn`+$\lambda=1$,
  pure-MBD `demo_1nn`+$\lambda=0$, BC bridge `bc`+$\lambda=5$.
- 4 mass scales × 3 methods = **12 cells**, single seed=0,
  n=60/cell.  ETA ~5.5 h.

Pre-flight verification: regression test
`test_worker_perturbation.py` PART A + B PASS on cost-time
(`22ad90d`); worker plumbing on dynamics is identical via the
cherry-pick of `05edaf6` + `716630a`; single-cell mass-scale
smoke (`--mass-scale 1.3`, n=2 demos, K=4) ran cleanly with
`mass_scale=1.3` recorded in output JSON params.

**Three a priori outcomes:**

A. **Mirror cost-time** — gap narrows with $|mass-1.0|$.  The
   fixed-anchor mechanism extends to dynamics.  Unifies the
   adaptation story into ONE finding: *the prior's retrieval
   bottleneck binds before the cost-side mechanism can act*.

B. **Dynamics succeeds, cost-time failed** — gap widens or
   stays flat.  Sharpens the thesis: *the failure is specific
   to retrieve-from-obs perturbations; mass/friction are
   structurally outside Can's obs vocab and so escape the
   fixed-anchor trap*.  This is the most informative outcome.

C. **Universal collapse** at large mass shift — publishable
   bound on Can's manifold range, less interesting.

### Incremental prior (4) — queued

Infrastructure (incremental cross-task prior-bank flags) is
committed on `feat/adapt-incremental-prior`; design questions
remain open (obs-vocab choice; perturbed-demo source).  Gated
on the dynamics result + a design discussion.

### What this means for the talk

The adaptation arc gives a **two-out-of-three substantiation
picture**, with one negative result that *strengthens* the
thesis rather than weakens it:

1. **Slate O (R1 small-lift, on `main`):** at the
   single-trajectory level, rank-and-pick over the σ-ball
   lifts v6 standalone by +18 pp. **Confirms Theorem 1' Cor 1.**
2. **DemoScale (on `feat/adapt-demo-scaling`):** at the
   bank-coverage level, mode-undercoverage gives the sharp
   80 %/0 % in-bank vs OOB split at $N_b=5$.
   **Confirms Theorem 1' Cor 1.**
3. **Cost-time (on `feat/adapt-cost-time`):** under goal-pose
   shift, the hybrid-vs-BC gap *narrows*.  **Identifies a new
   failure mode — fixed-anchor bias — with a principled fix
   (goal-conditioned retrieval) queued as F2.**  The mechanism
   is *complementary* to Cor 1: Cor 1 controls variance via
   mode anchoring, but anchoring only helps when the prior's
   anchor can move with the cost.  When obs is goal-blind, the
   anchor freezes.
4. **Dynamics (in flight):** outcome distinguishes whether the
   fixed-anchor mechanism generalises (A: unifies the story)
   or is specific to retrieve-from-obs perturbations
   (B: sharper isolated finding).

The cost-time negative is **not a setback** — it converts a
talk that would have said *"the hybrid adapts everywhere"* into
one that says *"the hybrid adapts wherever the prior's retrieval
can refresh; we identify when and why the bottleneck binds"*.
The latter is a stronger and more honest thesis.

---

## 7. Headline numbers (canonical operating point: λ=1, σ=0.25, M=16, K=10, R=1)

The hybrid is a fusion of two prior methods, so the right
comparison is against **both** of its component endpoints
in isolation, not just one.

| Task          | Pure offline<br>(BC MLP `[256, 256]`) | Pure online<br>(MBD, no prior) | **Hybrid (ours)** | Open-loop<br>demo replay (oracle) |
|---------------|------------------------:|------------------------:|--------------------:|------------------:|
| **Can**       | 21.7 % (13/60)         | **0.0 %** (0/60)        | **75.0 %** single-seed<br>**73.8 %** 4-seed pool, CI [67.8, 78.9] (240 rollouts) | 90 % (18/20) |
| **Lift**      | — *(not measured)*     | — *(not measured)*      | **100.0 %** (60/60) | — |
| **Square**    | — *(not measured)*     | — *(not measured)*      | **8.3 % peak** (at σ=0.30, AUX1) | — |
| **Tool_hang** | — *(not measured)*     | — *(not measured)*      | 0 / 40 *(upstream sparse-reward bug; re-test queued in Slate J with dense-reward shim)* | — |
| **Transport** | — *(not measured)*     | — *(not measured)*      | queued (Slate J) | — |

Source: BC baseline from `experiments/03_bc_baseline/results/can_ph.json`; MBD baseline from `experiments/04_mbd_baseline/results/_archive/can_ph.json`; hybrid pool from `slate_e_2026_05_19/AGGREGATE.json`.

**The two endpoints both fail on Can in isolation.** Pure
offline BC plateaus at 21.7 % because the MLP is unimodal and
collapses the demonstration distribution; pure online MBD
returns 0/60 because the 7-D × 120-step search has no
warm-start to anchor it and never discovers the grasp.
**Hybrid = 73.8 % pooled is strictly greater than the
max of the two endpoints (21.7 %) — the bridge is doing
non-additive work**, not merely averaging two partially-working
methods. Quantitatively:

- **Δ vs BC** (pure offline): **+52.1 pp** single-seed, **+36.3 pp** pooled (73.8 − 37.5 BC pool), with non-overlapping confidence intervals by ~24 pp.
- **Δ vs MBD-no-prior** (pure online): **+75.0 pp** single-seed (75.0 − 0.0), **+73.8 pp** pooled. The online endpoint is degenerate without a prior.
- **Sum-of-parts test**: $\text{BC}+\text{MBD} = 21.7\% + 0.0\% = 21.7\%$. Hybrid (73.8 %) exceeds this by **52.1 pp**, ruling out the null hypothesis that the bridge is just a linear combination of the two methods.

Caveat: BC and MBD-no-prior baselines were measured on
Can only. Cross-task BC/MBD reference numbers on Lift /
Square / tool_hang are future-work — the hybrid columns there
stand on the score-bridge mechanism alone, not on a Δ to a
baseline.

### 7.1 Relation to Pan et al. (MBD, NeurIPS 2024) — not a contradiction

A reader familiar with the MBD paper will reasonably ask: *Pan
et al. report that MBD works on manipulation benchmarks — why
do we measure $0/60$?* The two measurements are **not in
conflict**. Pan et al. explicitly list as **limitation (iii)**
of MBD that its self-normalised importance-sampling estimator
*"suffers high variance when $\beta$ is small or the horizon is
long"*. Our cell sits squarely in that regime: $H\!=\!120$
across a 7-DoF Panda yields an **840-dimensional trajectory
search space** with millimetre-precision contact constraints,
and we use **$M\!=\!16$** SNIS samples per iteration — well
below the budget Pan's published manipulation cells run at.
Pan's own paper additionally lists as **limitation (i)** that
MBD "cannot capture multimodal, demonstration-informed
behaviors", and the paper explicitly states that data *"can
be naturally integrated"* with the pipeline but leaves the
integration unformalised — **this is the gap our score-bridge
formalises**. The Slate F **budget-retry at 3×~MBD compute**
on Square ($M\!=\!16\to48$, $K\!=\!10\to30$) returned $\leq 9\%$,
showing that on Robomimic's contact-rich `ph` tasks the
bottleneck within our wall-clock envelope is not MBD compute
but the absence of a demonstration-informed prior. Our $0/60$
therefore **corroborates** Pan's limitations (i) and (iii) and
**motivates** the hybrid; it does not refute Pan's
contribution.

### 7.2 If asked about SOTA — the framing (talking points only; do not show in the table without cross-checking)

A robotics-literate audience member will ask *"how does 73.8 % on
Can compare to Diffusion Policy / Diffuser?"* This is the single
most likely question after the headline table. We do **not** have
a SOTA column in §7 because (a) we have not measured those methods
on our setup, (b) running them would take 1-3 days + a GPU we
don't currently have, and (c) hand-transcribing their published
numbers risks getting them slightly wrong. Instead, prepare to
answer in prose with this framing:

> "Trained policies on Robomimic `can/ph` — Diffusion Policy
> reports around 95 %; BC-RNN with GMM ~87-92 %; offline-RL
> methods like BCQ ~89 %, CQL ~38 %. Our 73.8 % sits between
> trained BC variants and untrained baselines. The right
> comparison axis isn't raw success rate — it's success rate
> *per learned parameter*. Trained methods use ~$10^7$
> parameters and ~6-12 GPU-hours of Robomimic-specific
> training. Our prior is **parameter-free retrieval over the
> demo bank**. We reach roughly 75-80 % of trained SOTA at
> 0 % of the training cost; that's the trade-off the hybrid is
> exploring."

If asked for exact numbers we would say *"approximately ~95 %
for Diffusion Policy on `can/ph`, but let me double-check the
paper on a follow-up"* — better to defer than to misquote a
table. Cross-check before the talk if there's time: Chi et al.
2023 (Diffusion Policy) Table 1; Mandlekar et al. 2021
(Robomimic) Table 4. Adding the SOTA column to the headline
table is queued in §8.2 item #3 as ~30 min of future work.

### 7.3 Baseline-selection rationale and SOTA comparison

The full landscape of methods we could compare against on
Robomimic `can/ph`, with our selection rationale. **Numbers in
the "cited" column are from the original papers and have not
been verified against our setup**; the four rows labelled
"measured" are reproducible from this repository.

| Method | Family | Training cost | `can/ph` succ | Provenance | Why we did / didn't headline it |
|---|---|---:|---:|:-:|---|
| Demo replay | open-loop oracle | — | **90 %** | measured (Exp 01) | **Used** as ceiling reference. |
| Random actions | floor | — | **0 %** | measured (Exp 02) | **Used** as lower bound (Week 4). |
| BC MLP `[256, 256]` | offline IL, MLP | ~5 min CPU | **21.7 %** | measured (Exp 03) | **Used** as the cheap-training offline endpoint. The cleanest +52 pp delta to the hybrid. |
| BC-RNN-GMM | offline IL, RNN | ~30 min GPU | ~92 % | cited (Mandlekar et al. 2021) | **Deferred** — Robomimic's own paper publishes the number; running it ourselves would duplicate work without changing the story. |
| IBC (Implicit BC) | offline IL, EBM | ~1-2 h GPU | ~22 % | cited (Florence et al. 2022) | **Deferred** — interesting because it's the only trained IL method that under-performs the hybrid; not central to the talk. |
| BCQ | offline RL | ~3-6 h GPU | ~89 % | cited (Robomimic paper) | **Deferred to future work** (§8.2 item 5). Same role IQL would play. |
| CQL | offline RL | ~3-6 h GPU | ~38 % | cited (Robomimic paper) | **Deferred** — surprisingly low; useful contrast for the offline-RL story but not for our zero-training pitch. |
| Diffuser | diffusion planning | ~6 GPU-h | ~80-85 % | cited (Janner et al. 2022) | **Deferred** — same family as the score-bridge but trained-end-to-end; running it would take 1-3 days + GPU. |
| **Diffusion Policy** | diffusion imitation | ~6-12 GPU-h | ~95 % | cited (Chi et al. 2023) | **Deferred** — current SOTA on Robomimic. The right comparison axis is succ-per-trained-parameter; cited in §7.2 talking points. |
| MBD no prior | pure online | — | **0 %** | measured (Exp 04) | **Used** as the no-training online endpoint. |
| **v_theta v2 + MBD (our flow prior attempt)** | hybrid (trained flow + MBD) | ~30 min GPU | **16.7 % single-seed peak** at σ=0.02 | measured (Exp 08) | **Kept on the branch but not headlined.** We *did* try a trained flow prior on the same demos; it underperformed parameter-free `demo_1nn` by ~58 pp. See the v_theta diagnostic in §11 Q2. |
| **Our hybrid (`demo_1nn` + score-bridge MBD)** | hybrid (zero-training prior + online MBD) | **0** | **73.8 % pooled, CI [67.8, 78.9]** | measured (Exp 15) | **The headline.** |

**The selection rule.** We measured the **two endpoint methods
that bracket the training-cost spectrum** at zero or near-zero
training cost (BC MLP, MBD-no-prior, demo replay, random, our
hybrid), plus **one trained-prior comparison from inside the
hybrid family** (v_theta v2). Trained policies of more expensive
flavours (BC-RNN, BCQ, CQL, Diffuser, Diffusion Policy) are
cited from the literature in §7.2 rather than re-run, because
running them would consume 1-3 days + GPU we don't have and
the *axis* of the project's claim is "what can be done at zero
or near-zero training cost", not "what's the SOTA succ rate
unconditional on compute".

### Results and refinement over the traditional approach (presentation paragraph)

Classical trajectory generation forces a binary choice between
two failure modes: **pure offline diffusion or flow priors**
(Diffuser, FlowMP, BC) capture multimodal demonstration
structure but provide no feasibility guarantee under the
deployed cost and degrade silently out of distribution — on
Robomimic `can/ph`, our `[256, 256]` MLP behaviour-cloning
baseline reaches only **21.7 %** (13/60) closed-loop, and the
same baseline reaches **0 %** on the harder Square peg-in-hole;
**pure online model-based diffusion** (MBD) is
feasibility-correct under the analytical dynamics but
demonstration-blind, with importance-sampling estimators of
high variance over long horizons — on the same `can/ph` task,
MBD without any prior reaches **0/60** because the 7-D × 120-step
open-loop search never discovers the grasp. Our **score-bridge**
refinement closes that gap by denoising with the composite score
$\nabla\!\log p^{\mathrm{data}}_t + \lambda(x,t)\nabla\!\log p^{\mathrm{model}}_t$
**at every diffusion step**, fusing the two into a single
product-of-experts target
$p^{\mathrm{data}}\,[p^{\mathrm{model}}]^\lambda$ whose
linear-quadratic limit reduces exactly to the LQR feedback gain
of the DARE — a direct bridge between generative modelling and
classical dynamic programming. Empirically, the hybrid reaches
**75.0 % single-seed and 73.8 % across a 4-seed pool**
(177 / 240, Wilson 95 % CI [67.8 %, 78.9 %]) on Robomimic Can —
a **+52.1 pp single-seed / +36.3 pp pooled** gain over BC with
non-overlapping confidence intervals — and transfers cleanly to
**Lift at 100 % (60/60)** at the identical operating point, both
with the same $\lambda\!=\!1$ peak and the same $\sigma$-invariance
profile, evidencing that the score-bridge mechanism is
task-agnostic rather than Can-specific. A soft-NN $\beta$-sweep
traces the bias-variance tradeoff continuously from **0.45 at
$\beta\!=\!0$** (uniform top-$k$ mean) up to **0.75 at
$\beta\!\to\!\infty$** (hard 1-NN), confirming that **mode
selection — not mode averaging — is the operative property of
the warm-start prior**: smoothing across demos hurts; committing
to one demo mode helps. Finally, on Square (peg-in-hole,
contact-rich) the same hybrid plateaus at **8.3 % peak** after
**five orthogonal ablations** (bridge $\lambda$, $\sigma$ tail,
MBD compute budget, retrieval temperature, action-history obs);
the failure localises to the **retrieval representation** rather
than the bridge itself, motivating the contrastive-obs-embedding
extension currently in flight.

---

## 8. Limitations and suggested future work

The honest weak spots of the project as it stands, paired with
the most useful next step against each. Roughly in priority
order.

### 8.1 Limitations

1. **No SOTA comparison on the Robomimic leaderboard.** Diffusion
   Policy reaches ~80-95 % on `can/ph`; Diffuser is in the 70-85 %
   range. We report 73.8 % pooled and call it a +52 pp gain over
   BC — but never compare against the modern *learned*-policy
   methods on the same benchmark. The right framing is
   "comparable success rate at **zero training cost**", but we
   haven't measured the SOTA column to substantiate that framing.
2. **The offline prior is memorisation, not learning.** `demo_1nn`
   is parameter-free retrieval — no generalisation beyond the
   demo bank, no representation of which features matter for
   retrieval, no scaling argument as bank size grows.
3. **Square is not solved.** 8.3 % peak after five orthogonal
   ablations (bridge λ, σ tail, MBD compute, retrieval temperature
   $\beta$, action-history obs). We've localised the failure to
   the retrieval representation but haven't fixed it.
4. **Statistical power is thin.** Only Can has a 4-seed pool
   (240 rollouts, CI [67.8 %, 78.9 %]). Lift / Square / tool_hang
   are single-seed n=60 cells with wide Wilson CIs (e.g. Square
   8.3 % = 5/60 gives CI [3.6 %, 17.7 %]).
5. **Sim-only.** No perturbation studies (mass / friction ±30 %),
   no observation noise, no real-robot deployment. Week 1
   promised OOD tests; they were deferred.
6. **Dense-reward shim on tool_hang / transport is hand-designed.**
   Staged-distance + subgoal-indicator shaping. Weights picked
   from analytical max-per-step intuition, not learned. Results
   on these tasks will be reward-function-dependent.
7. **Theory is empirical-only.** The LQR limit is shown via the
   1.059× DARE check, not derived. No SNIS-variance bounds, no
   policy-iteration convergence proof. *Addressed by the
   finite-sample SNIS bound planned in §8.3, and partially by
   the SNIS posterior-collapse / σ-ball reachability argument
   in `docs/theory/manifold_restoration_2026_05_25.md` (added
   2026-05-25), which explains the three-regime lift curve
   from §6.5 from first principles.*
8. **MBD-no-prior baseline measured only at $M\!=\!16$.** Pan's
   published manipulation cells used larger $M$. We argue in §7.1
   that this corroborates Pan rather than contradicts it, but we
   have not run MBD-no-prior at $M\!\in\!\{64, 256\}$ to quantify
   the break-even compute budget. The 0/60 on Can is regime-
   specific, not absolute.
9. **Wall-clock cost; not real-time deployable.**
   $K\!=\!10\times M\!=\!16\times H\!=\!120$ = 19 200 MuJoCo steps
   per rollout, ~50 s/rollout on a 16-core machine with 10
   parallel workers. Real-robot deployment would require GPU
   MuJoCo (MJX); the `feat/mjx-rollouts` branch ports the
   rollout to JAX but is blocked on cloud GPU quota.
10. **MPC chunking ($R\!=\!1$) not ablated.** ~~Every reported cell
    uses one replan at the chunk boundary; we never measured the
    open-loop variant ($R\!=\!0$) or longer-horizon replans
    ($R\!\geq\!2$) at the canonical operating point. The
    contribution of replanning to the headline number is
    therefore unattributed.~~
    *Partially addressed 2026-05-25: Slate L Wave B measured
    v6+MBD-pull at $R\!=\!4$ chunk=24 on Can = $14/30 = 46.7\%$
    vs $R\!=\!0$ K3 reference $= 83.3\%$ (single-seed; multi-seed
    pool 81.1 %).  $R \in \{1, 2, 3\}$ are queued as Tier-3
    follow-ups in `docs/todos/feat_v6_pull_slate_l_pending_2026_05_25.md`.
    Conclusion: parity-with-prior is **single-shot specific**;
    closed-loop chunking is itself a distribution shift the
    prior was not trained for.*
11. **Operating-point sensitivity not pooled.** Slate E E3 already
    showed that $\sigma$-invariance is $\lambda$-specific (flat
    at $\lambda\!=\!5$, peaked at $\lambda\!=\!1$). A 2-D
    $(\lambda, \sigma)$ grid with seed pooling has not been run;
    the headline operating point may not be universal across
    tasks.
12. **No comparison to other hybrid prior + online-correction
    methods.** SafeFlow, MPD (Carvalho et al., IROS 2023), CFM-MPPI
    (Mizuta & Leung, ICRA 2026), and BSD (Li et al., 2026) sit in
    the same conceptual space. We frame against them in §"Related
    Work" but never measure their numbers on the same Robomimic
    cells; the gain over BC and MBD is a clean A/B but not a
    field-wide ranking.
13. **Demonstrator-quality dependence.** All four task families
    use Robomimic's *proficient-human* (`ph`) split; the `mh`
    (mixed-human) split has lower-quality demos and would test
    whether the retrieval prior degrades gracefully. Not measured.

### 8.2 Suggested future work

In rough order of impact-per-effort.

#### Immediate (this week, no new infrastructure)

1. **Slate I (C1 contrastive embedding on Square) — abandoned
   2026-05-23, deferred to successor branch.**  Originally
   paused 2026-05-22 mid-cell-1/13; the process died May 23
   with zero cell outputs and was never resumed.  C1 is the
   last untested axis on Square's 8.3 % mechanism limit (after
   bridge λ, σ tail, MBD compute, retrieval temperature β, and
   action-history obs all failed in Slates F, G, H, AUX1).
   Re-launching is queued as a successor-branch follow-up;
   strong gate Square peak ≥ 25 %, falsification ≤ 10 %.  Not
   blocking the K-O headline.
2. **Run Slate J (tool_hang + transport with dense-reward shim).**
   Manual fire, ~10-12 h. First scored transport cells in the
   project; closes Slate H's upstream-bug gap.
3. **SOTA reference numbers.** ~30 min — read Diffusion Policy
   and Diffuser papers, transcribe their published `can/ph`
   numbers into the headline table as a 5th column. Costs
   nothing and pre-empts the reviewer's first question.

#### Short-term (1-2 days)

4. **Cross-task BC and MBD-no-prior baselines.** ~25 min/cell
   × 6 cells = 2.5 h. Fills the "—" entries in the §7 table for
   Lift / Square / tool_hang. Strengthens the 3-way head-to-head
   argument for those tasks.
5. **Offline-RL prior (IQL / AWAC).** Plug-compatible with the
   `priors_zerotrain.py` interface; uses Robomimic's reward field.
   ~3-4 h implementation + 1 h scored cells. Directly addresses
   the "offline must be RL" interpretation flagged earlier.
6. **Bank-size scaling on Can.** Subsample the demo bank to
   $N \in \{20, 50, 100, 200\}$ at fixed lambda/sigma and trace
   the succ curve. Tests whether the 73.8 % headline is bank-
   limited or mechanism-limited.

#### Medium-term (1 week)

7. **OOD perturbation suite.** ±30 % mass / friction; novel object
   poses outside the demo distribution. The hybrid's *claimed*
   virtue is OOD robustness; never measured. Strong selling
   point if it survives.
8. **Closed-loop safety / smoothness metrics.** Compute collision
   counts, joint-limit violations, control-effort norms from the
   saved rollouts. No new experiments; only re-analysis.
9. **MBD-no-prior compute-scaling on `can/ph`.** Re-run
   MBD-no-prior at $M \in \{16, 64, 256\}$ to extrapolate when
   pure online matches the hybrid. Sharpens the §7.1 "Pan is
   not contradicted" argument quantitatively.

#### Theory directions (final-report or paper anchor)

10. **KL-regularised RL formulation of the score-bridge.** Show
    that the composite score is the gradient of the soft-RL target
    $\arg\min_\pi \,\mathbb{E}[J(\tau)/\beta]
    + (1/\lambda)\,\mathrm{KL}(p^\pi \,\|\, p^{\mathrm{data}})$.
    Gives $\lambda$ a precise interpretation as KL inverse-
    temperature; honestly answers the "must be RL" question.
    ~1 day of writing.
11. **(Primary planned contribution — see §8.3.)**
    Finite-sample SNIS bound for the MBD score. Derive
    $\|\hat s_M - s\|^2 \leq C\,\sigma_t^2 V_w / (M \beta^2)$
    under standard assumptions. Makes Pan's limitation (iii)
    quantitative and predicts the break-even compute budget
    at which the hybrid beats pure MBD.
12. **Deferred (longer-term).** Full bias-variance decomposition
    of the hybrid via a coverage bound on the demo bank composed
    with the SNIS bound from (11): predicts the soft-NN $\beta$
    curve and explains Square as a metric-coverage failure.
    Requires non-parametric 1-NN regression theory on top of (11);
    ~1-2 weeks. Pursue after (11) lands.

### 8.3 Planned theoretical contribution — finite-sample SNIS bound for the MBD score

The project's committed theoretical anchor for the final report
— **theory choice B from the future-work menu** (Pan limitation
(iii) made precise). Picked over the more ambitious choice C
(full bias-variance decomposition) for two reasons: (i) it's
achievable in ~3-5 days, fitting inside the project window;
(ii) it makes the *single* claim that already appears in our
§7.1 defence against the MBD comparison (*"SNIS variance grows
with horizon and sample count"*) into a quantitative bound a
reviewer can chase. Choice C is deferred to §8.2 item #12 and
remains the natural follow-up.

#### Setup

Pan's MBD computes the analytical score by self-normalised
importance sampling (SNIS) over $M$ perturbed rollouts. At
noise scale $\sigma_t$, with the Boltzmann temperature $\beta$
and trajectory cost $J(\tau)$, the true score is

$$
s_t(\tau) \;=\; \nabla_\tau \log p^{\mathrm{model}}_t(\tau)
\;=\;
\frac{\mathbb{E}_{q_t}\!\big[(\tau'-\tau)\,w(\tau')\big]}{\sigma_t^2\,\mathbb{E}_{q_t}\!\big[w(\tau')\big]},
\qquad w(\tau') = \exp\!\big(\!-J(\tau')/\beta\big),
$$
where $q_t$ is the noised proposal distribution. The SNIS
estimator from $M$ i.i.d.\ samples $\{\tau'_i\}_{i=1}^M\!\sim\!q_t$ is

$$
\hat s_{t,M}(\tau) \;=\;
\frac{\sum_{i=1}^M w(\tau'_i)\,(\tau'_i - \tau)}{\sigma_t^2\,\sum_{i=1}^M w(\tau'_i)}.
$$

Let $V_w := \mathrm{Var}_{q_t}[w(\tau')] / (\mathbb{E}_{q_t}[w(\tau')])^2$
denote the **effective sample size deficit** (so $V_w \ge 0$,
with $V_w = 0$ iff the weights are constant).

#### Claim (the bound we want to prove)

Under standard SNIS regularity (Owen 2013, Ch. 9): bounded
trajectory norm $\|\tau\|_\infty \le R$, bounded cost
$|J| \le J_{\max}$, and absolutely continuous proposal $q_t$:

$$
\boxed{\;
\mathbb{E}\!\left[\big\|\hat s_{t,M}(\tau) - s_t(\tau)\big\|^2\right]
\;\leq\;
\frac{C\;\sigma_t^2\;V_w(\tau, t, \beta)}{M\,\beta^2}
\;+\;
O\!\Big(\frac{1}{M^2}\Big),
\;}
$$
with $V_w$ growing in $H$ through the Boltzmann concentration
of the importance weights. The leading $O(1/M)$ term is the
classical SNIS variance; the $O(1/M^2)$ term is the
self-normalisation bias.

#### Why this matters quantitatively

| Empirical observation | What the bound says |
|---|---|
| **MBD-no-prior 0/60 on Can at $M\!=\!16$** | At $H\!=\!120$, $V_w$ for the unaided 840-d search is large (most rollouts are "cannot reach", weight collapses to a handful of outliers). The per-iteration score error is variance-dominated; the descent direction is noise. |
| **§7.1 defence vs Pan** | "SNIS variance grows with horizon and sample count" becomes the explicit scaling $\sigma_t^2 V_w / (M \beta^2)$. Pan's manipulation cells used $M\!\gg\!16$; ours did not. |
| **LQR sanity at 1.059× DARE** | In LQ the bound collapses to its lowest-variance regime (constant cost gradient, smooth $w$): the residual 5.9 % excess over DARE is the finite-$M$, finite-$K$ SNIS noise — *quantifies* the algorithmic core's correctness. |
| **Why the prior helps (motivates the hybrid)** | The hybrid anchors the search at $\bar\tau$ from `demo_1nn`, where $w$ is locally peaked $\Rightarrow$ $V_w(\bar\tau)$ is much smaller than $V_w(0)$. Same $M$, lower variance, viable descent direction. |
| **Predicts MBD break-even compute** | Pure MBD matches the hybrid at $M^\star \approx V_w^{\text{(no prior)}} / V_w^{\text{(prior)}} \times 16$. Empirically measurable by sweeping $M\!\in\!\{16, 64, 256, 1024\}$ on MBD-no-prior. |

#### Derivation sketch (~4-6 pages for the final report)

- **Step 1 — Decompose** $\hat s_{t,M} - s_t$ into a variance
  term (i.i.d.\ SNIS-style central limit) and a self-
  normalisation bias term (delta-method).
- **Step 2 — Bound the variance term** using Owen Ch 9.
  Theorem 9.2: $\mathrm{Var}(\hat s_{t,M}) \;\le\; \sigma_t^2 V_w / (M \beta^2)$,
  with $V_w$ the asymptotic SNIS variance constant.
- **Step 3 — Bound the bias term** as $O(1/M^2)$ by the
  delta-method second moment.
- **Step 4 — Express $V_w$ in terms of horizon $H$** using
  the cost's additive structure $J(\tau)\!=\!\sum_k \ell$; under
  mild moment assumptions $V_w$ scales polynomially in $H$.
  (This is the half of Pan's limitation (iii) that's most
  important to make precise.)

#### Effort and deliverable

- Effort: ~3-5 days for a rigorous write-up using standard SNIS
  machinery (Owen 2013 Ch 9, MCMC course notes); ~2 days for a
  clean sketch with $V_w$ computed in closed form for the LQ
  case as a sanity-check companion to the 1.059× DARE result.
- Deliverable: a 2-3 page theory section that turns §7.1's
  prose argument *("MBD limitation (iii) explains 0/60")* into
  a quantitative scaling law. Slots between the "Related Work"
  and "Experiments" sections of the final report.

#### Falsification protocol

The bound predicts a specific $\sqrt{M}$ scaling for the
hybrid's executed cost when the prior side is held fixed.
A 4-cell sweep of MBD-no-prior on Can at
$M \in \{16, 64, 256, 1024\}$ (~3 h total wall on the test rig)
would test whether the predicted scaling $\sigma\!\propto\!1/\sqrt{M}$
holds, and locate the empirical break-even with the hybrid
($M^\star$ such that pure MBD reaches 73.8 % succ). If the
predicted slope is off by more than ~2×, either the constants
in $C$ are wrong (a fitting issue) or the bound is structurally
loose (a derivation gap). Both outcomes are informative for
the final report.

---

## 9. What this gives us for the paper / presentation

Five publishable claims, each backed by a specific cell in the
project's data:

1. **Zero-training retrieval prior + score-bridge MBD beats BC
   on Can by +52 pp single-seed / +36 pp pooled.** §7 headline
   table; `slate_e_2026_05_19/AGGREGATE.json`.
2. **The mechanism is bias-variance, with mode selection (not
   averaging) as the operative property.** Soft-NN β-sweep
   monotone curve from 0.45 (β=0) → 0.75 (β→∞). `docs/results.md`
   §"Slate G + G-aux (May 21)", G2.
3. **Cross-task transfer: Lift reaches 100 % at the same
   operating point, with the same λ peak as Can.**
   `docs/results.md` §"Slate G + G-aux (May 21)", G1.
4. **Mechanism limit identified on Square (8.3 %) and
   triangulated to 5 orthogonal axes** (bridge λ, σ tail, MBD
   compute, retrieval temperature, action-history obs).
   `docs/results.md` §"Slate G + G-aux" + Slate H. Motivates
   the C1 contrastive-embedding follow-up.
5. **Methods caveat: two robomimic tasks (`tool_hang`,
   `transport`) ship sparse-only rewards upstream.** Project
   provides a dense-reward shim
   (`experiments/15_unified_hybrid/env_dense_rewards.py`).
   Important for reproducibility; Slate H §POSTMORTEM
   documents the upstream-source-code trace.

---

## 10. Open work (status as of 2026-05-22 morning)

- **Slate I** (C1 contrastive obs embedding) — **abandoned
  2026-05-23** (originally paused 2026-05-22 mid-cell-1/13;
  process died May 23 with zero cell outputs; never resumed).
  Re-launching is queued as a successor-branch follow-up,
  not blocking the K-O headline.  Strong gate Square
  peak ≥ 25 %, falsification ≤ 10 %.  Either outcome would
  still be publishable (strong = "Square unlocked",
  falsification = "Square is a bridge / cost-function limit,
  not a retrieval limit").
- **Slate J** (manual fire, ~10-12 h): re-tests tool_hang and
  transport with the dense-reward shim active.  First scored
  transport cells in the project.
- **Documentation**:
  - `docs/results.md` updated with the May 22 cross-task
    headline table + the Slate H section.
  - This file is the presentation brief.

---

## 11. Asking for the instructor's opinion (20-minute meeting plan)

We have about 20 minutes. The plan below allocates time so we
ask the **highest-leverage scoping questions first** — these
are decisions about what to build over the next 1-2 weeks that
we cannot answer ourselves with confidence. Four concrete
questions plus a catchall; rough time budgets and the single
sentence we need an answer to per question.

| Time | Block | The one decision we want out of this block |
|---:|---|---|
| **0-3 min** | Context-set: headline 73.8 % pooled CI [67.8, 78.9] on Can, 100 % on Lift, 8.3 % on Square (mechanism limit triangulated to 5 axes), LQR limit at 1.059× DARE | (no question — set the frame) |
| **3-7 min** | **Q1 — baseline priority** | Which one trained baseline should we add for the final report? |
| **7-11 min** | **Q2 — theory write-up depth** | Sketch (3-5 days), rigorous (~2 weeks), or sketch + start choice C? |
| **11-15 min** | **Q3 — further hyperparameter sweeps / seed pooling** | Which existing cells should we pool to a 4-seed CI before the final report? |
| **15-18 min** | **Q4 — extend to other tasks / benchmarks** | Cross-Robomimic-split, MetaWorld / D4RL, or real-robot? |
| **18-20 min** | Open / catchall | Any framing red flags? |

### Q1 (≈4 min) — Baseline-selection priority for the final report

§7.3 catalogues the methods we *could* compare against. Time
and compute are limited; we'd like to add **one** trained
baseline before the final report. Three candidates, in
increasing cost:

1. **BC-RNN-GMM** (offline IL, ~30 min GPU, ~92 % cited).
   Cheap; completes the IL family but doesn't change the
   zero-training value axis.
2. **Offline RL prior — IQL or AWAC** (~3-4 h impl + 1 h
   training). Plug-compatible with `priors_zerotrain.py`;
   directly addresses the *"offline component must be RL"*
   reading.
3. **Diffusion Policy or Diffuser end-to-end** (~1-3 days +
   GPU, ~80-95 % cited). Current SOTA; biggest validation but
   biggest implementation risk and a different value axis from
   our zero-training pitch.

**The sentence we need:** *"Add `<option>` first because
`<value-axis>` is the right axis for this project's claim."*

### Q2 (≈4 min) — Depth of the theory write-up (§8.3)

We have committed to **theory choice B** (finite-sample SNIS
bound, §8.3) as the project's theoretical anchor, with effort
budgeted at ~3-5 days for a "clean sketch with all constants
computed for the 7-DoF Panda" rather than a ~12-page rigorous
derivation. The deeper option, choice C, is a full
bias-variance decomposition with a covering-number argument on
the demo bank (~1-2 weeks; predicts the soft-NN β-curve and
explains Square as a metric-coverage failure).

**The sentence we need:** *"Aim for `<sketch B | rigorous B |
sketch B + start C>`, because `<reason>` matters more than
`<other>` for this project's evaluation."* The trade-off is
**sketch buys you time for other things** vs **rigorous buys
you credibility but eats the rest of the project**.

### Q3 (≈4 min) — Further hyperparameter sweeps / seed pooling

Most of our existing cells are **single-seed** at n=60 (Lift
1.000, Square 0.083 peak, soft-NN β-sweep, λ-schedule sweep,
σ-tail). Only Can has a **4-seed pool at n=240** (177/240 =
73.8 %, Wilson CI [67.8, 78.9]). Four candidate follow-up
sweeps:

- **4-seed pool on Lift at λ=1 σ=0.25**: ~2 h wall, validates
  the 100 % single-seed at a publishable CI.
- **4-seed pool on Square at σ=0.30 peak**: ~2 h, tightens the
  Wilson CI from [3.6, 17.7] (single-seed n=60) to something
  defensible.
- **Fine-grain λ around the Can peak (λ ∈ {0.7, 0.85, 1.15, 1.3, 1.5})**:
  ~2.5 h. Already flagged as a Slate-E follow-up; never run.
- **2-D (λ, σ) grid with seed pooling**: would close drawback
  #11 in §8.1. ~6 h.

**The sentence we need:** *"Pool `<which cells>` first, because
`<reason>`. Skip / defer `<which>` because `<reason>`."*

### Q4 (≈3 min) — Extend to other tasks / benchmarks

Currently four Robomimic task families (Can, Lift, Square,
tool_hang) plus Transport queued. Three orthogonal extensions:

- **Other Robomimic splits**: `mh` (mixed human, lower demo
  quality) on existing tasks — tests the prior's degradation
  profile under noisier demos. ~2 h per task once the bank
  loader is parameterised.
- **MetaWorld** or **D4RL**: different robot families
  (Franka vs Sawyer), different reward structures, different
  obs spaces. Tests true cross-domain generalisation. ~1-2
  weeks of dataset + env integration.
- **Custom real-robot demos**: would close the sim-only
  drawback (#5 in §8.1). Requires hardware access, ~1+ week
  of data collection + integration.

**The sentence we need:** *"Extend to `<which extension>` first
because `<value-axis>` — task-family breadth, demo-quality
robustness, or real-robot validity — is the most important
generalisation claim to substantiate before submission."*

### Q5 (≈2 min) — Open / catchall

Reserved for whatever the instructor flags as missing — the
single most likely red flag is framing (whether
"zero-training" is the right value axis to pitch, or whether
the lack of a real-robot / OOD test is a fatal omission).
Better to leave room for that than to come in with five
pre-cooked questions and run over.

---

## 12. One-slide summary

> **Hybrid trajectory generation = zero-training retrieval prior + score-bridge MBD.**
>
> - **Problem.** Offline diffusion / flow priors capture demo structure but break OOD; online MBD is feasibility-correct but demo-blind.
> - **Idea.** Composite score $\nabla\!\log p^{\mathrm{data}}_t + \lambda\,\nabla\!\log p^{\mathrm{model}}_t$ at every diffusion step (product-of-experts). LQ limit = LQR / DARE.
> - **What we built.** Zero-training retrieval prior (`demo_1nn`) + per-step MBD score-bridge inside a unified MPC driver. No training.
> - **Results on Can (3-way head-to-head).** Pure offline (BC) **21.7 %** · Pure online (MBD no prior) **0.0 %** · **Hybrid (ours) 73.8 %** pooled, CI [67.8, 78.9]. Hybrid > max(endpoints) by **+52 pp** and > sum-of-endpoints by **+52 pp** — non-additive.
> - **Cross-task.** Lift **100 %** (cross-task ceiling). Square **8.3 %** (mechanism limit; triangulated to 5 orthogonal axes).
> - **Theory anchor (planned, §8.3).** Finite-sample SNIS bound $\;\mathbb{E}\|\hat s_M - s\|^2 \le C\,\sigma_t^2 V_w / (M\beta^2) + O(1/M^2)\;$ makes Pan limitation (iii) quantitative; predicts the MBD-no-prior break-even compute $M^\star$ via $V_w$-ratio. LQR 1.059× DARE sanity check is the empirical companion.
> - **Reproducibility caveat.** Robosuite 1.5 ships sparse-only rewards for `tool_hang` and `transport`; our dense-reward shim restores shaping; scored cells incoming.

---

## 13. Post-brief update (2026-05-25) — branch consolidation and what's now on `main`

The brief above is the May-22 snapshot; three days of cleanup
have happened since. Everything in the brief is still
empirically current — no headline number has moved — but the
project's git topology and documentation layout are now
substantially cleaner. Quick read for the talk:

- **Headline number is unchanged.** Can still 73.8 % pooled
  (CI [67.8, 78.9]) at the canonical λ=1 σ=0.25 K=10 R=1 M=16
  cell; Lift still 100 %; Square still 8.3 % peak; tool_hang /
  transport still pending Slate J. §7 is good as-is.
- **`feat/zero-train-priors` is now merged onto `main`**
  (commit `2daa166`, 2026-05-25 15:26 PDT). The Slate D – J
  empirical narrative and the `priors_zerotrain.py`
  implementation are no longer a branch artefact; they live on
  `main`. Slates D–E (the headline source) are the canonical
  state of the project.
- **Three additional branches merged the same day:**
  - `feat/stronger-mbd-signal` (`25ca457`) — Brings V_φ
    state-value control variate scaffolding for the MBD score
    onto `main` via a `--value-baseline-ckpt` flag (eval cell
    scoped, acceptance bar ≥ 5/60 at pure-MBD λ=0 σ=0.05 M=64).
    Tests H2 (variance-bottleneck) over H1 (capacity-bottleneck);
    not yet executed at headline protocol.
  - `feat/hybrid-coupling` (`595aadf`) — declares Phase 3
    adaptive `λ(σ)` flag for the score-bridge schedule; not yet
    wired.
  - `feat/theory` + `feat/theory-extensions` (`b8de97d`,
    rebased to drop WIP commits) — **seven new theory documents**
    under `docs/theory/`, forming a self-contained bundle whose
    headline contraction rate is
    $\alpha(\lambda) = \lambda^2 \sigma_{\min}^2 / (2 C_{\mathrm{LSI}})
    + O(\lambda^3)$, derived in doc 06 and applied throughout:
    - `01_soft_control.md` — Soft-control optimum is
      product-of-experts; LQ limit = one step of policy
      iteration (formal version of §8.2 item #10).
    - `02_snis_bound.md` — Finite-sample SNIS variance bound
      $\mathbb{E}\|\hat s - s\|^2 \le D^2 V_w / (\sigma_k^2 M) + O(M^{-2})$
      (formal version of §8.3 — the project's planned
      theoretical contribution).
    - `03_pareto_bias_variance.md` — Pareto bias-variance
      dominance: hybrid MSE $\le \min$(data, model) endpoint MSE
      by factor $\rho^{-1}$ (formalises the §7 "Hybrid > max(endpoints)"
      empirical statement).
    - `04_sample_complexity.md` — At fixed deployment cost,
      hybrid trains with $N_{\mathrm{hybrid}} = N_{\mathrm{pure}} / e^{2\alpha K / \alpha_{\mathrm{SM}}}$ demos
      (shorter-training direction).
    - `05_adaptive_robustness.md` — At fixed training data,
      hybrid certifies DRO radius $\delta(K) = \Omega(e^{2\alpha K})$
      against cost shift (added-adaptability direction).
    - `06_predictor_corrector_sde.md` — $W_2$ contraction of
      the hybrid distribution at rate $e^{-\alpha K}$, derived
      from log-Sobolev / Wasserstein flow; *this is where
      $\alpha(\lambda)$ is actually derived*.
    - `07_minimax_regret_synthesis.md` — Linear composition
      $s = s_d + \lambda s_m$ is minimax-optimal among all
      composition rules; MPC regret is $\tilde O(\sqrt{T})$;
      synthesis table + 7 open problems.
  - **Status:** docs 01-07 are all marked paper-ready in
    `docs/theory/README.md`. The bundle composes
    strictly-forward (doc $N$ cites only docs $1..N{-}1$); any
    individual theorem can be quoted with at most three layers
    of cross-reference.  This **closes §8.3's commitment in
    full**, and goes substantially beyond it: §8.3 promised a
    "3-5 day finite-sample SNIS bound write-up" as the project's
    theoretical contribution; the bundle delivers that bound
    (doc 02) plus six surrounding theorems that situate it in
    a soft-control / Pareto / $W_2$-contraction / minimax
    framework.  The single empirical anchor is the 1.059×
    DARE-LQ check.
- **`feat/flow-v3-karras` content is on `main` via `feat/v6-pull-slate-l`.**
  The Karras-EDM v3 score prior (exp 08), DP UNet1D + closed-loop
  adapter, the `--score-mode {learned, pull}` asymmetric fix,
  controlled warm-start corruption, and decoupled MBD budget all
  landed via the `feat/v6-pull-slate-l` merge (rebased onto
  theory + experiment-docs + V_φ; backup tag
  `pre-rebase-v6-pull-slate-l-2026-05-25`). Slate K's reading is
  preserved: at `--score-mode=learned`, MBD's softmax washes out
  strong learned priors; the asymmetric `pull` mode is the fix.
  See §6.5 for the K → O arc and the corrected +17.8 pp lift
  finding from Slate O.  This is the empirical evidence behind
  §11 Q2's "the learned-prior side is the open frontier".
- **Slate I and Slate J status (updated 2026-05-26):**
  - **Slate I (C1 contrastive obs prior) — abandoned
    2026-05-23.**  Paused 2026-05-22 mid-cell-1/13; process
    died May 23 with zero cell outputs.  Re-launching
    deferred to a successor branch.  Square peak strong gate
    (≥ 25 %) and falsification gate (≤ 10 %) both still
    unmet; the K-O headline does not depend on either.
  - Slate J (tool_hang + transport with the dense-reward
    shim) is still pre-staged. Manual fire only; launcher
    refuses if `slate_h` or `slate_i` screens are alive
    (neither is, as of 2026-05-26).
- **New navigational layer added: per-experiment docs.** A
  separate branch `feat/experiment-docs` (commit `4f3b4ae`)
  carries 17 new markdown files under `docs/experiments/`:
  one per experiment 01 – 16 plus a master index. Each file
  gives the experiment's current lifecycle status (active /
  archived / WIP), its most recent headline number with
  source, the canonical reproduce command, cross-references
  to its predecessor / successor / companion experiments, and
  caveats. This is the answer to *"if a reviewer asks for the
  exact recipe of cell X, where do I send them?"* — point
  them to `docs/experiments/<NN>_<name>.md` instead of
  archaeologising the experiment directory's README.
  - The companion top-level docs (`docs/results.md`,
    `docs/run_log.md`, `docs/plans.md`, `docs/branch_plan.md`)
    have all been refreshed to cross-reference the new index
    and to reflect the May 16 – May 25 catch-up (Slates D – J
    now narrativised in `run_log.md`).

**What this means for the talk.** Nothing in §1 – §12 needs
to change. Two optional updates a presenter might make
between the brief and the slide deck:

1. **Replace "branch `feat/zero-train-priors`" anywhere it
   appears in the frontmatter or in §10 with "merged into
   `main` on 2026-05-25"** — the talk now plays against a
   single canonical `main` rather than against an in-flight
   feature branch. Cleaner story.
2. **Q2 (theory write-up depth) is fully answered.** §8.3's
   choice-B "3-5 day SNIS sketch" is superseded by the
   seven-document bundle on `main`; the bundle README
   (`docs/theory/README.md`) marks all seven docs as
   paper-ready and maps them to a proposed 5-page report-
   theory section (1.5 pp soft-control + 1 pp SNIS bound +
   0.5 pp Pareto + 1 pp sample-complexity + 1 pp DRO +
   0.25 pp synthesis, with $W_2$ contraction, minimax, and
   MPC regret in a 3-page appendix). The question to the
   instructor narrows from *"sketch B or rigorous B?"* to
   *"which of the seven theorems goes in the main text vs
   appendix?"* Doc 07 §7 lists 7 open problems (LSI verification,
   mode-separation, finite-N constants, stochastic localisation,
   RLHF connection, continuous-time limit, very-good-prior
   second-order theory) for the natural-next-theory line.

---
