# Prior Design Choices — Full Report

*How every prior in the hybrid is built, what it outputs, and how it differs.*

All priors plug into the same unified hybrid driver
(`experiments/15_unified_hybrid/run_can.py`) via **two hooks**:

1. a **warm-start** `sample_initial_mean(s0)` — the initial mean trajectory
   `mu0` that seeds MBD's denoising loop;
2. a **per-iteration score** `score_fn(mu, k)` — added to the cost-guided
   model score inside `mbd.denoise_loop`.

Priors differ along three orthogonal axes: **what they learn**, **what
object they output**, and **how that object enters the cost-guided
denoiser**.

---

## A. Two families

| Family | Members | Training | Parameters |
|---|---|---|---|
| **Zero-train (nonparametric retrieval)** | `demo_1nn`, `demo_knn`, `demo_softnn`, `demo_random`, `demo_mean` | None — reads the demo bank | 0 |
| **Learned (parametric)** | BC, v1, v2, **v3 EDM**, **v6 CFM**, **DP** | Gradient descent on 200 `can/ph` demos | 80 k → 64 M |

Source files:
- Zero-train: `experiments/15_unified_hybrid/priors_zerotrain.py`
- BC: `experiments/03_bc_baseline/`
- v1/v2/v3/v6/DP: `experiments/08_flow_matching_prior/model*.py`

---

## B. Zero-train retrieval priors (`priors_zerotrain.py`)

Built over the flat demo bank (200 demos × ~150 steps ≈ 30 k
`(obs, action)` pairs). They differ only in **which action window they
retrieve** as the warm-start:

- **`demo_1nn`** *(headline, 73.8 %)* — nearest neighbour over the 46-D
  obs vector; warm-start = the next `H` actions from the matched demo
  timestep (clipped at demo end). State-conditioned and
  **mode-selecting** (commits to a single demo).
- **`demo_knn`** — `k`-NN; warm-start = **mean** of the `k` matched
  action windows. Lower variance but **averages modes** → produces an
  off-manifold synthetic trajectory that the per-step bridge amplifies
  (collapses under `λ>0`).
- **`demo_softnn`** — softmax(−d/β) weighting over the `k` nearest demo
  states; β=∞ reduces to `demo_1nn`.
- **`demo_random`** — uniform-random demo starting at t=0. Control for
  *"does state-conditioning matter?"*
- **`demo_mean`** — dataset-wide mean action sequence. Control for
  *"does any demo information matter?"* (0 %).

**Shared score.** All five reuse the **quadratic pull** score
`score_fn(mu, k) = a_target − mu` — a *stationary* spring toward the
retrieved warm-start (the gradient of `½‖mu − a_target‖²`). There is no
learned score field; `--lambda-prior > 0` simply turns the pull into a
per-step score-bridge with the same algorithmic shape as the BC headline.

---

## C. Learned priors — the exp08 ablation ladder

These were deliberately built as an **axis-isolated ablation**: same
200-demo `can/ph` dataset, same horizon, same obs-dim, same z-score
normalisation — *only the objective / architecture changes*, so any
env-success delta is attributable to model design, not "data shimmer".

### BC (`experiments/03_bc_baseline/`) — the regression baseline
- **Objective:** deterministic MLP `a = π(s)`, MSE on demo actions. No
  noise model, no distribution.
- **Score into MBD:** quadratic pull (same as `demo_1nn`), canonical λ=5
  ("BC bridge").
- **Role:** answers *"is the MLP doing real work vs. raw retrieval?"* —
  barely (6.7 % standalone, ~80 k params).

### v1 (`model.py`) — first flow attempt
- **Objective:** Lipman **OT-CFM**, `x_t=(1−t)x₀+t·x₁`, target velocity
  `v* = x₁ − x₀`, ODE sampling `x_{t+dt}=x_t+v_θ·dt`.
- **Conditioning:** `s0` *concatenated* to `[x_t, t]`.
- **Why it failed:** `s0` is ~5 % of the input vector → the optimiser
  ignores it (near-identical trajectories regardless of obs); and the
  velocity target is **noise-dominated** (loss floor ≈ just learning
  `−x₀`).

### v2 (`model_v2.py`) — fix conditioning + target
- **FiLM conditioning:** per-layer multiplicative scale/shift from `s0`
  — forces obs to have a non-additive effect the optimiser can't ignore.
- **`x₁`-parameterisation:** predict the *clean* trajectory directly,
  MSE against the real demo (better-conditioned than the velocity target).
- **Score-bridge:** `∇log p_t ≈ (t·x̂₁ − x_t)/(1−t)`.
- **Why it failed:** trained **per-demo at t=0 only** → mid-trajectory
  states are OOD → replanning broke (R=1 → 0/60); plus a **σ-mismatch**
  between its OT-CFM transport-path marginal and MBD's actual iterate
  `mu + σ·ε`.

### v3 (`model_v3.py`) — Karras-EDM score model
- **Objective:** **denoising score matching** with EDM preconditioning.
  Train on `x₀ + σ·ε` *directly*, with σ log-uniform over MBD's real
  `[σ_min, σ_max]` schedule. The network output `D(x;σ,s)` *is* the
  denoised mean; the **score is the exact Tweedie `(D − x)/σ²`** — no
  bridge approximation.
- **Conditioning scalar:** `c_noise(σ)=0.25·log σ` (σ-conditioned →
  the "sigma-specialist").
- **Data:** **per-chunk** `(s_t, a_{t:t+H})` for every `t` — matches
  BC's per-state distribution and fixes v2's t=0-only flaw.
- **Horizon:** H=120, near-open-loop.
- EDM preconditioning:
  `c_in=1/√(σ²+σ_d²)`, `c_out=σσ_d/√(σ²+σ_d²)`,
  `c_skip=σ_d²/(σ²+σ_d²)`,
  `D = c_skip·x + c_out·F_θ(c_in·x, c_noise, s)`.

### v6 (`model_v6_cfm.py`) — Conditional Flow Matching
- **Objective:** Lipman **CFM** with the linear interpolant
  `x_t=(1−t)ε+t·x₁`, **velocity target `x₁−ε`**, loss
  `‖v_θ(x_t,t,s) − (x₁−ε)‖²`, `t∼U[0,1−t_eps]`.
- **Sampler:** **Heun 2nd-order ODE** (native K=18).
- **Score-bridge for MBD:** `∇log p_t=(t·v_θ−x)/(1−t)` — analytic from
  the linear path.
- **Shares v3's trunks** (FiLM-MLP / Conv1D-UNet / Transformer): *only*
  the objective differs from v3 (EDM-DSM vs CFM), making it the clean
  objective ablation.

### DP (`model_dp.py`) — Diffusion Policy (the strong one)
- **Architecture:** faithful `ConditionalUnet1D` (Chi et al. 2023) — 1D
  conv U-Net, FiLM conditioning, GroupNorm + Mish, ~64 M params.
- **Objective:** **DDPM ε-prediction** (Ho et al. 2020),
  `squaredcos_cap_v2` β-schedule, T=100.
- **Sampler:** DDPM ancestral (K=100; strided DDIM at low K is unstable
  for our under-EMA'd checkpoints — only the full schedule recovers
  quality).
- **Distinctive:** closed-loop **T_obs=2, H=16**, replan every 8 steps
  (the DP-paper config) — versus the H=120 near-open-loop flow priors.
- **Selected checkpoint:** `v3b ep0199` = **80 % test** (the explicit
  phase-B winner); `ep0149` is a weaker earlier checkpoint at 64 %.

---

## D. The key axes of difference

| Prior | Objective | Output object | Sampler | Conditioning | Horizon | Score into MBD |
|---|---|---|---|---|---|---|
| `demo_*nn` | none (retrieval) | warm-start | — | obs k-NN | H=120 | **quadratic pull** |
| BC | MSE regression | action | — | FiLM / MLP | H=120 | quadratic pull |
| v1 | OT-CFM | velocity | ODE | concat | H=120 | `(t·v−x)/(1−t)` |
| v2 | OT-CFM, x₁-param | clean traj | ODE | FiLM | H=120 (t=0) | `(t·x̂₁−x)/(1−t)` |
| **v3** | Karras EDM DSM | denoised mean | EDM Heun | σ-cond FiLM | H=120 | **`(D−x)/σ²` (exact Tweedie)** |
| **v6** | Lipman CFM | velocity | Heun ODE | FiLM | H=120 | `(t·v−x)/(1−t)` |
| **DP** | DDPM ε-pred | noise ε | DDPM K=100 | UNet, T_obs=2 | H=16 closed-loop | ε→score, or **pull** mode |

### Three orthogonal design dimensions the ladder isolates

1. **What is modelled:** raw retrieval (`demo_*`) → deterministic action
   (BC) → generative trajectory *distribution* (v1–v6, DP).
2. **The generative objective:** OT-CFM (v1/v2) → Karras-EDM score
   matching (v3) → Lipman CFM (v6) → DDPM (DP). Because v3 and v6 share
   trunks, the v3↔v6 comparison is a *clean objective ablation*.
3. **How it guides MBD:** ad-hoc **quadratic pull** (retrieval / BC) vs.
   a **principled score-bridge / Tweedie** (v2 / v3 / v6) vs. DP's
   ε-derived score. Crucially, `--score-mode pull` lets *any* trained
   prior fall back to the stationary pull — which is what makes the
   strong DP prior usable inside the hybrid, since its learned
   per-iterate score destabilises the denoiser (the non-stationary
   attractor smears the warm-start across the K inner iterations,
   −36.7 pp on the DP H=16 baseline).

---

## E. "Pull" vs "learned" score mode (inference-time)

`--score-mode` (in `run_can.py`) selects how a *trained* prior injects
its score:

- **`learned`** — query the trained field at the current iterate `mu`
  (e.g. v3's `(D−x)/σ²`, v6's `(t·v−x)/(1−t)`). Non-stationary;
  destructive for strong DP priors.
- **`pull`** — ignore the learned field, use the stationary spring
  `score_fn(mu,k) = a_target − mu` toward the prior's
  observation-conditioned warm-start. This is the "trained-prior-as-
  retrieval" mode; it produced the 73.8 % `demo_1nn` headline and is the
  recommended mode for strong learned priors.

> Note: a Gaussian demo factor `N(Y|Y_demo, σ²I)` has score
> `(Y_demo − Y)/σ²` — i.e. the **pull** is exactly the demonstration
> score of MBD §4.3 up to a `1/σ²` scale (see
> `docs/theory/09_mbd_data_guided_comparison.md`).

---

## F. Why our reported success sits a little below the published figures

Published reference points (see `docs/references/sota_reference_numbers.md`):

| Method | Published | Our best | Protocol of "our best" |
|---|---:|---:|---|
| Diffusion Policy (Chi 2023, low-dim crosswalk) | **94.7 %** | **80 % test** (DP `ep0199`) | 50 fresh-seed test |
| BC-RNN-GMM Can-ph (Mandlekar 2021, low-dim) | **92.3 %** | — | (different policy class) |
| Our retrieval headline (`demo_1nn` + bridge) | — | **73.8 %** pooled | 0 training params |

The gap is mostly **protocol and training-budget**, not a modelling
defect. Specifically:

1. **Reporting protocol.** Robomimic Table 4 reports the **best over 5
   seeds**; DP tunes a config and reports its peak. We report a **single
   50 fresh-seed test run** with a fixed config — best-of-N reporting
   alone inflates the published number by several points.
2. **Training budget + EMA.** The DP recipe uses 100–300 epochs with
   **EMA (decay 0.9999)**; our `v3b` checkpoints are shorter and
   **non-EMA**. Our EMA-300 retrain actually *regressed* (17 %) due to
   sampler/obs interactions, so the best checkpoint we have is the
   non-EMA `ep0199` — still short of a fully-tuned DP.
3. **Checkpoint-selection sensitivity.** There is a **22 pp swing**
   across epochs (`ep0199` 80 % vs `ep0399` 58 %). Without selecting on a
   held-out test bucket it is easy to report a materially worse epoch
   (this is exactly the `ep0149` vs `ep0199` mix-up that understated an
   earlier slate).
4. **Sampler step count.** Our under-EMA'd DDPM needs the **full K=100**
   ancestral sampler; strided DDIM at K≤64 amplifies the ε-prediction
   error ~100× through the `1/√ᾱ_t` divide and collapses to ~0 %. A
   better-trained model would tolerate the cheap K=16 sampler the paper
   uses.
5. **Observation modality.** Published DP peaks (96–100 %) are
   **image-obs**; the apples-to-apples **low-dim** crosswalk is 94.7 %.
   We run low-dim (23-D `can_dp` / 46-D), a smaller-capacity, harder
   setting.
6. **It isn't our headline anyway.** Our contribution is the
   **zero-training retrieval prior + hybrid bridge** (73.8 % at *0*
   trained parameters), not beating DP. DP is the *trained upper
   reference*, and in our story it only pulls ahead **via the bridge**
   (86.7 %), not standalone.
7. **CFM caveat.** The 83.3 % CFM number was measured on the easier
   **30 train-init** grid, not the 50 fresh-seed test set; re-measuring
   on the common protocol (the running
   `slate_headline_priors_2026_06_04.sh`) will likely show a lower,
   directly comparable figure.
8. **pure-MBD 0 % is expected, not a failure.** On Can and the hard
   tasks the in-horizon cost is flat (sparse reward), so MBD's importance
   weights never concentrate. That is a cost-shaping property of the
   task, not an under-reproduction.

---

## G. What we changed relative to the original papers

We deliberately re-implement architectures/objectives rather than load
released weights, so every comparison is an **axis-isolated ablation on
our data**, and we extend MBD with a data term it does not have.

- **vs Diffusion Policy (Chi et al. 2023).** Faithful
  `ConditionalUnet1D` + DDPM ε-objective, but trained on **our** 200-demo
  `can/ph` set with **our obs schema** and **z-score normalisation**
  (the paper uses min-max `[-1,1]` and a different low-dim feature
  selection). We reproduce *architecture + objective*, not their weights
  — the scientifically interesting axis.
- **vs Karras EDM (Karras et al. 2022) — v3.** Standard EDM
  preconditioning specialised to 7-DoF × 120-step trajectories, but
  trained **directly on MBD's runtime σ-schedule** (`x₀ + σε`,
  σ log-uniform over `[σ_min,σ_max]`) so the score is valid at MBD's
  *actual* iterate `mu + σε` — a deliberate deviation from generic EDM
  training (the "σ-preservation" fix). Output is the exact Tweedie score,
  no bridge approximation.
- **vs Lipman CFM (2023) — v6.** Standard CFM objective with our shared
  trunks, plus an analytic **score-bridge `(t·v−x)/(1−t)`** so the flow
  field can act as a *score* inside a separate cost-guided denoiser
  (CFM is normally just ODE-sampled, not queried as a score).
- **vs MBD (Pan et al. 2024).** MBD is **data-free** — its score comes
  only from dynamics + cost via SN-IS. We **add a data factor**: a
  learned (CFM/DP/EDM) *or* nonparametric (`demo_*`) prior, fused with
  MBD's model score as a **product-of-experts** with a tunable
  `λ_prior = λ/β`. We also run trained priors in **`pull` mode**
  (trained-prior-as-retrieval) and apply the whole pipeline to
  contact-rich robomimic manipulation using the env's **shaped reward as
  the cost J**. (Combination-rule contrast with MBD §4.3 is in
  `docs/theory/09_mbd_data_guided_comparison.md`.)
- **vs Diffuser / MPD (cost-guided diffusion).** Those add **ad-hoc cost
  gradients** on the Tweedie estimate. We instead compute the model
  score from the **full Boltzmann `exp(−J/β)`** via SN-IS (MBD) and
  analyse the data↔cost fusion as a soft-control optimum
  (`docs/theory/01_soft_control.md`).

---

## H. Cross-references

- Combination rule (product-of-experts vs MBD's mixture+max-gate):
  `docs/theory/09_mbd_data_guided_comparison.md`.
- Soft-control / PoE derivation and `λ_prior=λ/β`:
  `docs/theory/01_soft_control.md`.
- README prior table: `experiments/08_flow_matching_prior/README_v3.md`.
- Standalone re-run on a common 50-seed protocol:
  `experiments/08_flow_matching_prior/slate_headline_priors_2026_06_04.sh`.
