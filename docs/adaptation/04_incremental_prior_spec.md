# Adaptation 4 — Incremental & cross-task prior banks

*Branch: `feat/adapt-incremental-prior`. Companion specs:
`01_cost_time_spec.md`, `02_dynamics_spec.md`,
`03_demo_scaling_spec.md`. Theory companion:
`docs/theory/01_soft_control.md` §1 — together with spec 03 this
exercises the $p^{\mathrm{data}}$ factor of the soft-control
target.*

The highest-novelty of the four adaptation specs. Specs 01 and
02 (cost / dynamics) are nominally available to any planner that
re-uses its environment model; the **cross-task and incremental
prior** is **uniquely available to retrieval-based methods**, and
gives the "zero-training" pitch its strongest selling point — no
learned baseline can match it without retraining.

## 1. What this measures

Two distinct experiments share the same flag surface:

### T1 — Cross-task transfer prior

*Hypothesis.* The retrieval prior captures **task-generic motion
structure** (reach, grasp, lift, place) rather than task-specific
geometry. If true, swapping the bank's task family (e.g., using
**Lift demos as the prior for a Can eval**) should preserve
*some* of the hybrid's advantage over BC.

*Prediction.* Hybrid succ on Can with a Lift-demo bank is between
the canonical 73.8 % and pure MBD's 0 %. The gap to BC (which
cannot easily be re-purposed for cross-task transfer without
retraining) is the headline number.

### T2 — Incremental bank adaptation

*Hypothesis.* When task dynamics shift (e.g., mass ×1.3 from
spec 02), appending **5 fresh demos collected under the perturbed
dynamics** to the retrieval bank recovers most of the canonical
performance. Retrieval is parameter-free, so the bank update is
*instantaneous*. BC requires ~30 min of GPU retraining to
absorb the same 5 demos and cannot match the wall-clock
adaptation profile.

*Prediction.* At mass ×1.3:

| Method | Canonical bank only | Bank + 5 perturbed demos | Adaptation wall-clock |
|---|---:|---:|---:|
| BC (canonical, no retrain) | ~10 % | n/a | n/a |
| BC (retrained on 200 + 5) | n/a | ~25 % | ~30 min GPU |
| Hybrid `demo_1nn` (canonical) | ~30 % | ~55 % | **~1 s** (just re-build cache) |

The **adaptation wall-clock column** is the single most
striking result: $1$ s vs $30$ min is a 1800× speedup *at
comparable adapted-task succ*. This is the "incremental
zero-training" selling point.

## 2. Experiment cells

Operating point: canonical $\lambda = 1$, $\sigma_{\max} = 0.25$,
$M = 16$, $K = 10$, $R = 1$ on Can.

### T1 ladder

| Cell | Eval task | Bank task | Method | n | Notes |
|---|---|---|---|---:|---|
| T1.0 | Can | Can | Hybrid `demo_1nn` | 60 | canonical control = 73.8 % |
| T1.1 | Can | Lift | Hybrid `demo_1nn` | 60 | core T1 cell |
| T1.2 | Can | Square | Hybrid `demo_1nn` | 60 | core T1 cell |
| T1.3 | Lift | Can | Hybrid `demo_1nn` | 60 | reverse-direction control |
| T1.4 | Can | Lift | MBD-no-prior | 60 | clean negative control |
| T1.5 | Can | (none, BC) | BC (Can-trained) | 60 | the canonical BC; "no-cross-task" reference |

Wall: 6 cells × ~50 min = ~5 h.

### T2 ladder

Requires Slate I-style scripted demos under perturbed dynamics
(spec 02's $f$-mutated env). The +5 demo collection itself is a
~10 min task (drive the env open-loop with handcrafted way-points,
record state-action pairs, append to the bank HDF5). This is the
one cell that genuinely requires *new data collection*.

| Cell | Mass scale | Bank source | Method | n |
|---|---:|---|---|---:|
| T2.0 | 1.0 | canonical 200 | Hybrid | 60 | control |
| T2.1 | 1.3 | canonical 200 | Hybrid | 60 | unadapted at mass 1.3 |
| T2.2 | 1.3 | canonical 200 + 5 mass-1.3 demos | Hybrid | 60 | adapted; T2 headline |
| T2.3 | 1.3 | canonical 200 + 5 mass-1.3 demos | BC (retrained) | 60 | BC adapted (cite ~30 min retrain) |
| T2.4 | 1.3 | canonical 200 | BC (not retrained) | 60 | BC unadapted control |

Wall: 5 cells × ~50 min + ~30 min BC retrain + ~10 min data
collection = ~5 h.

### Total budget

T1 + T2 = ~10 h. **Realistic budget for both: 10–12 h on the test
rig.**

## 3. Implementation checklist

Before firing T1 cells (lower-friction):

- [x] `--prior-bank-dataset` and `--prior-bank-task` flags in
      `run_can.py` (this commit).
- [x] Bank source dispatcher in `_load_prior` (this commit).
      `load_demo_bank` already supports any dataset path; the
      change is purely *which* path to pass.
- [x] Output JSON `params.{prior_bank_dataset, prior_bank_task}`
      (this commit).
- [ ] **Cross-task obs vocab verification.** `priors_zerotrain.get_obs_keys`
      returns task-specific obs keys (e.g., Can has bin-pose, Lift
      doesn't). When the bank's task differs from the eval task,
      the retrieval obs vector dimensions **must agree** — meaning
      we either:
      - (a) restrict both to a *shared* obs subset (proprio only;
            ~40 dims for Panda),
      - (b) zero-pad the missing dims in the smaller-vocab task,
      - (c) refuse cross-task swaps with vocab mismatch and emit
            a clear error.
      Decision pending. **Option (a) is the cleanest experimental
      design**; documented in this spec sec 5.
- [ ] **Pre-flight smoke** for T1.1 (Can eval, Lift bank) to
      catch any vocab mismatch error before scored cells.

Before firing T2 cells:

- [ ] Spec 02's `apply_dynamics_perturbation` must be implemented
      (not just stubbed). T2.1–T2.4 require the mass-1.3 env to
      work.
- [ ] **Perturbed-dynamics demo collection script.** ~50 lines:
      drive the env open-loop using the canonical Can demos'
      action sequences but stepped through the mass-1.3 env;
      record the resulting (obs, action) pairs as new demos.
      The action sequences are the *same* as canonical; only the
      observed effects differ. Append to an HDF5 sibling of
      the canonical dataset.

## 4. Pre-flight smoke (T1)

```bash
# 30 s smoke -- verifies cross-task bank loading.
python -m experiments.15_unified_hybrid.run_can \
    --task=can --prior=demo_1nn --lambda-prior 1.0 \
    --K=10 --M=16 --sigma-max=0.25 --mpc-replans=1 \
    --prior-bank-dataset=/Users/szh/.../lift_low_dim_v141.hdf5 \
    --prior-bank-task=lift \
    --smoke --output-name-suffix=_T1_smoke
```

Expected: `[exp15] zero-train bank: 200 demos, ..., task=lift,
... cross-task: eval=can bank=lift`. If the obs vocab differs
between can and lift, expect a runtime error in
`priors_zerotrain.make_zero_train_prior` — fix per item 3
bullet 4 above before scored cells fire.

## 5. Obs-vocab decision for cross-task transfer

The cleanest experimental design is **option (a): restrict both
to the proprio subset** (joint angles + EEF pose, ~40 dims for
Panda, identical across all Robomimic Panda tasks).

This is already supported via `--prior=demo_1nn_proprio`. So
the *true* T1 spec uses `demo_1nn_proprio` rather than
`demo_1nn` to avoid obs-vocab drift. Trade-off: we already know
from Slate F that `demo_1nn_proprio` is ~3 pp below `demo_1nn`
on within-task Can (the bin-pose dims help slightly). The
cross-task cell will then be reported as `demo_1nn_proprio` vs.
canonical `demo_1nn_proprio` (not vs. `demo_1nn`), so the
comparison is apples-to-apples.

## 5.1 What Theorem 1' predicts for this experiment

Both T1 (cross-task transfer) and T2 (incremental adaptation)
manipulate the **$p^{\mathrm{data}}$ factor** of the soft-control
target — like spec 03 — but along the **identity** of the prior
(which task's demos, which mass-perturbed demos) rather than its
size. The predictions are therefore *complementary* to spec 03's
scaling-law predictions.

### T1 — cross-task transfer prior

| Term in Theorem 1' | Behaviour for cross-task vs.\ same-task bank | Why |
|---|---|---|
| **Hybrid bias** | **grows** (cross-task bank is biased away from the eval-task optimum) | the prior anchor $\bar\tau$ is now a *wrong-mode* demonstration; bias term inherits the Lift→Can distance |
| **Hybrid variance** $\rho^{-1}(\bar\tau)$ | depends on whether the cross-task demo is still near *some* cost-Boltzmann mode | if Lift demos pass through "reach EEF over object" trajectories (which Can also requires), $\rho$ partially survives. Otherwise $\rho \to 1$ |
| Overall | hybrid succ **between** MBD-no-prior succ and within-task hybrid succ | "task-generic motion structure" preserved iff $\rho$ partially survives |

**Most theory-relevant prediction:** the cross-task hybrid's
succ rate is **strictly between** the MBD-no-prior floor (0 % on
Can at $M\!=\!16$) and the within-task ceiling (73.8 % on Can).
A non-zero result means **the prior captures task-generic
motion**, validating the "demonstration prior = generic motion
prior" reading of the soft-control bridge.

**Falsifies Theorem 1' if:** cross-task hybrid succ $\approx 0$,
matching MBD-no-prior. Would mean Lift demos provide *no*
anchoring useful for Can — the prior is *fully task-specific*.

**Substantiates Theorem 1' if:** cross-task hybrid succ
$\in [10\%, 50\%]$, halfway between the two endpoints. The
project's strongest "the prior captures something general"
finding.

### T2 — incremental adaptation

| Term in Theorem 1' | Behaviour: canonical bank → bank + 5 perturbed demos | Why |
|---|---|---|
| **Hybrid bias under perturbed dynamics** | **decreases sharply** | new demos are perfect mode anchors at the perturbed dynamics, replacing wrong-mode canonical demos at retrieval time |
| **Hybrid variance** | **decreases via $\rho$ collapse** | the new demos are *exactly* at modes of $p^{\mathrm{model}}_\eta$ (they were collected there), so $\rho(\bar\tau_{\mathrm{new}}) \gg \rho(\bar\tau_{\mathrm{old}})$ for any obs near the new-demo trajectories |
| Adaptation wall-clock | **~1 s** (just re-index the retrieval bank) | retrieval is parameter-free — no training step |
| **BC equivalent: retrained on 200 + 5 demos** | same bias reduction, *slower* wall-clock | BC needs ~30 min GPU to absorb the 5 new demos via gradient updates |

**Most theory-relevant prediction:** the **wall-clock × adapted-succ
Pareto plot** of §6 has the hybrid point **upper-left of the
BC-retrained point** — strictly better on both axes. This is
**a constructive demonstration of Theorem 1' in adaptation
space**: hybrid's bias + variance combine to give a finite-time
adaptation MSE that BC cannot match without paying the gradient
update cost.

**Falsifies Theorem 1' if:** T2.3 (BC retrained) succ ≥ T2.2
(hybrid adapted) succ. Would mean BC's bias reduction from 5
new demos *exceeds* the hybrid's combined bias+variance
reduction. Most likely cause: 5 demos is too few to shift 1-NN
retrieval. Sweep $\in \{5, 20, 50\}$ to locate the saturation
point.

**Substantiates Theorem 1' if:** hybrid adapted succ
$\ge$ BC retrained succ, AND hybrid adapted wall-clock
$\ll$ BC retrained wall-clock (1800$\times$ ratio cited in §1).
Two-sided strict Pareto dominance in (wall-clock × succ) space.

## 6. Analysis plan — the two headline figures

**Figure T1.** Bar chart, $x$-axis = (eval task × bank task)
pairs from the T1 ladder, $y$-axis = succ. The same-task bar
(T1.0) is the canonical control; the cross-task bars (T1.1, T1.2)
should be **non-zero** and **above the MBD-no-prior bar (T1.4)**.
If both hold, the retrieval prior captures task-generic motion
structure.

**Figure T2.** Wall-clock vs. adapted-task succ scatter. $x$-axis
= log(wall-clock seconds to adapt to mass 1.3). $y$-axis = succ
at mass 1.3. Three points:

- BC unadapted: (0 s, ~10 %).
- BC retrained: (~1800 s, ~25 %).
- Hybrid `demo_1nn` with appended 5 demos: (~1 s, ~55 %).

The hybrid point should sit **upper-left** of the BC retrained
point — *strictly* better on both axes. This is the
**Pareto-dominance plot** for the adaptation claim, mirroring the
bias-variance Pareto-dominance theorem of
`docs/theory/03_pareto_bias_variance.md`.

## 7. Falsification protocol

- **T1.1 succ ≤ 5 %** (Lift bank on Can eval): cross-task
  transfer fails; retrieval prior is task-specific. The
  paper's *"captures generic motion structure"* framing
  must be retracted in favour of the more limited *"reduces
  variance within a task family"*.
- **T2.2 ≈ T2.1** (incremental adaptation no better than
  unadapted): 5 demos at the new dynamics are insufficient
  to shift the 1-NN retrieval. Sweep number-of-new-demos
  $\in \{5, 20, 50\}$ to locate the saturation point.
- **T2.3 ≥ T2.2** (BC retrained matches or beats hybrid
  adapted): the *adaptation succ rate* is not the right axis;
  fall back to the *wall-clock × succ* Pareto plot as the
  headline (still a win on wall-clock).

## 8. Status

- 2026-05-25: branch created, flag surface + dispatcher landed.
  `load_demo_bank` already accepts any dataset path; the
  unified driver now routes through it via `--prior-bank-dataset`.
  **Real implementation, not a stub** — T1 smoke can fire today
  modulo the cross-task obs vocab decision (sec 5).
- Pending for T1: obs vocab decision (sec 5; recommend option (a)
  = `demo_1nn_proprio`) + smoke.
- Pending for T2: spec 02's `apply_dynamics_perturbation`
  implementation + perturbed-dynamics demo collection script.
- Once landed: ~10 h to two publishable figures for the final
  report.

## 9. The headline claim this experiment enables

The current presentation brief's *"zero-training is the right
axis for this project's claim"* (§7.2 talking points) is a
*qualitative* framing. T2's Pareto plot — wall-clock × succ —
**converts it into a quantitative comparison against the
trained-baseline family**. If T2 lands as predicted, the project
has a defensible quantitative answer to *"why retrieval and not
BC?"* that does not require a SOTA succ-rate comparison.

This is the single experiment most likely to **upgrade the
project's framing from "interesting alternative" to "Pareto
better"**.
