# Adaptation 2 — Dynamics adaptation: mass and friction perturbation

*Branch: `feat/adapt-dynamics`. Companion specs:
`docs/adaptation/01_cost_time_spec.md` (cost-time adaptation),
`docs/adaptation/03_demo_scaling_spec.md` (prior-bank scaling),
`docs/adaptation/04_incremental_prior_spec.md` (runtime bank
swap). Theory companion: `docs/theory/01_soft_control.md` §1 (the
$p^{\mathrm{model}}$ factor that this adaptation perturbs).*

This document specifies the **dynamics adaptation experiment** —
the medium-cost member of the three adaptation axes. Maps onto the
**$p^{\mathrm{model}}$** factor of the composite score's
soft-control target, but via the *simulator* $f$ rather than the
*cost* $J$. Both endpoints (BC and MBD) are nominally affected;
the hybrid's behaviour depends on whether the demonstration prior
remains useful as $f$ drifts.

## 1. What this measures

The hybrid's *adaptability* claim has two distinct readings:

- **Cost adaptation** (spec 01): runtime cost-function changes.
  The data prior is unchanged; only $J$ shifts. Tests whether the
  *online MBD factor* of the hybrid inherits MBD's automatic
  adaptation. **The data prior is a "fixed asset"** in this
  regime — it does the same thing regardless.
- **Dynamics adaptation** (this doc): runtime simulator changes.
  The data prior is still unchanged, but its demos are now
  *partially mis-specified* under the new dynamics (a grasp that
  worked at canonical mass may slip at 1.3× mass). The MBD factor
  re-plans correctly under the new $f$; the prior is a *partial
  liability*. **Tests whether the bridge $\lambda$ adapts.**

The theoretically interesting prediction (from
`docs/theory/01_soft_control.md` §3 — Toussaint 2009's
soft-control reading of the bridge) is that **the optimal $\lambda$
shifts as dynamics drift**:

- At canonical dynamics, $\lambda^\star \approx 1$ on Can (Slate E).
- As dynamics perturbation grows, demos become a worse mode hint,
  so $\lambda^\star \searrow 0$ — the bridge should *down-weight*
  the prior. This is a falsifiable prediction.

A 2-D $(\lambda, \text{perturbation})$ heatmap with the optimal
$\lambda^\star(\text{perturbation})$ ridge clearly visible is the
**single best figure** the dynamics-adaptation experiment can
produce.

## 2. Experiment cells

Operating point: canonical `demo_1nn`, $\sigma_{\max} = 0.25$,
$M = 16$, $K = 10$, $R = 1$ on Can.

### 2.1 Mass perturbation ladder (D1)

Sweep robot-link mass scale ∈ $\{0.7, 1.0, 1.3, 1.5\}$ at
$\lambda \in \{0.0, 0.5, 1.0, 2.0, 5.0\}$. Three methods (hybrid,
BC, MBD-no-prior).

| Cell | Mass scale | $\lambda$ grid | Notes |
|---|---:|---|---|
| D1.0 | 1.0 | $\{0, 0.5, 1, 2, 5\}$ | Control: reproduces Slate E E2 |
| D1.1 | 0.7 | $\{0, 0.5, 1, 2, 5\}$ | Lightweight arm; gripper overshoots |
| D1.2 | 1.3 | $\{0, 0.5, 1, 2, 5\}$ | 30 % heavier; slip-prone |
| D1.3 | 1.5 | $\{0, 0.5, 1, 2, 5\}$ | 50 % heavier; severe slip |

Per cell: 60 evaluations × 5 $\lambda$ values × 3 methods = 900
evaluations × 4 mass scales = **3600 evaluations total**.
Wall-clock: 3600 × 50 s / 10 parallel workers = ~5 h on the test
rig.

### 2.2 Friction perturbation ladder (D2)

Sweep object-table friction scale ∈ $\{0.5, 1.0, 2.0\}$ at the
same $\lambda$ grid. Three methods.

| Cell | Friction scale | $\lambda$ grid |
|---|---:|---|
| D2.0 | 1.0 | same as D1.0 (shared control) |
| D2.1 | 0.5 | slippery; harder grasp |
| D2.2 | 2.0 | tacky; easier grasp |

Per cell: 60 × 5 × 3 = 900 evaluations × 2 new perturbations
(D2.0 shared with D1.0) = 1800 evaluations = ~2.5 h.

### 2.3 Total budget

D1 + D2 ≈ 7.5 h on the test rig, single seed n=60 per cell. If
the headline figure (the $\lambda^\star$-ridge plot) shows a
clear monotone drift, pool 4 seeds at the canonical $\lambda^\star$
for each perturbation (+2 h). **Realistic total: 9–10 h, all in
one slate.**

## 3. Implementation checklist

Before firing any cell:

- [x] `--mass-scale` and `--friction-scale` flags in `run_can.py`.
- [x] Flags surfaced into output JSON `params.{mass_scale, friction_scale}`.
- [x] `env_perturbations.apply_dynamics_perturbation` **real impl**
      (commit `81bcac0`, 2026-05-25).
- [x] **Robot link body registry** filled 2026-05-25 by REPL probe:
      `("robot0_link", "robot0_right_hand", "gripper0_")` (single-arm
      tasks) and the dual-arm variant for `transport`. 15 bodies
      match per Panda env.
- [x] **Friction geom registry** filled, **re-typed to
      `_FRICTION_GEOM_PREFIXES`** (object-side only) because Can
      uses unnamed substrate collision geoms. See module docstring
      for the sub-linearity caveat.
- [x] **Mass mutation logic**: scales `body_mass` AND
      `body_inertia` by `mass_scale` (proportional inertia scaling
      = uniform-density-same-shape approximation).
- [x] **Friction mutation logic**: scales `geom_friction[gid, :3]`
      for object-side geoms; substrate side blended by MuJoCo's
      per-contact friction-mix (default solmix=0.5).
- [x] **`env.sim.forward()`** called once at the end of the helper.
- [x] **Smoke** (`/tmp/smoke_dyn_pert.py`, 4/4 pass):
      Can / Lift / Square all show correct mass & friction scaling;
      no-op returns silently.
- [ ] **Verify the rollout simulator and the eval env share `env.sim`**
      so the MBD planning rollouts also see the perturbed dynamics
      (otherwise the planner is solving the wrong problem). This
      is currently the case in `unified_loop.py` but worth
      confirming with a 2-line `assert env.sim is rollout_env.sim`
      print before scored cells fire.

## 4. Pre-flight smoke

```bash
# 5 min, 2 demos x 1 rollout, K=4 -- verifies mass/friction mutations
# wire through to env.sim.model and the rollout simulator.
python -m experiments.15_unified_hybrid.run_can \
    --task=can --prior=demo_1nn --lambda-prior 1.0 \
    --K=10 --M=16 --sigma-max=0.25 --mpc-replans=1 \
    --mass-scale 1.3 --friction-scale 0.5 \
    --smoke --output-name-suffix=_D_smoke
```

Should raise `NotImplementedError` from the stub (until registry
is filled). Then with real impl: verify the rollout's
`final_object_pose` differs visibly from the canonical baseline,
confirming the dynamics actually changed.

## 4.1 What Theorem 1' predicts for this experiment

The Pareto bias–variance theorem
(`docs/theory/03_pareto_bias_variance.md` eq. (11)) behaves
**differently** under dynamics perturbation than under cost
perturbation (spec 01 §5.1), because dynamics enters $J$
*implicitly* through the rollout simulator $f$ — which means
the proposal $q_k$'s **importance weights $w = \exp(-J/\beta)$
change** as $f$ drifts, and hence $V_w$ shifts.

**Specialised to dynamics perturbation, with $\eta$ the mass /
friction scale:**

| Term in Theorem 1' | Behaviour as $|\eta - 1|$ grows | Why |
|---|---|---|
| BC bias | **grows fastest** | demos collected at $\eta = 1$ are now generated by a different dynamic system; the MLP forward pass is fundamentally mis-specified |
| MBD-no-prior bias | constant $O(M^{-1})$ | SNIS is consistent for any $f$, $J$; bias depends on $M$ only |
| **Hybrid bias** | grows, but **bounded by min(BC bias, MBD bias)** | inherits MBD's automatic dynamics-re-planning through the cost factor (the simulator $f$ is recomputed under $\eta$) |
| MBD-no-prior variance | **grows** | unanchored $V_w$ inflates because demonstration-shaped trajectories are now *off the cost-Boltzmann support* under perturbed $f$ |
| **Hybrid variance** | **also grows** (in contrast to spec 01) | anchored $V_w(\bar\tau, \beta)$ *now depends on $\eta$* because the prior anchor $\bar\tau$ is no longer near a mode of $p^{\mathrm{model}}_{\eta}$ — Corollary 1's mode-anchoring argument weakens monotonically in $|\eta - 1|$ |

**The single most theory-relevant prediction is the bottom-right
cell: the hybrid's variance term grows with dynamics
perturbation**, in contrast to cost-time perturbation where it
is invariant. This makes the **two adaptation axes
distinguishable at the experiment level**:

- **Cost-time** (spec 01) widens the hybrid-vs-BC gap **via bias
  only**; hybrid SE stays flat.
- **Dynamics** (this spec) widens the gap **via bias and
  variance**; hybrid SE **grows visibly** with $|\eta - 1|$.

This also predicts the **optimal $\lambda^\star$ shift** in §1:
as $|\eta - 1|$ grows, $\rho(\bar\tau)$ shrinks toward 1 (the
prior anchor becomes a worse mode hint), so the variance gain
from the prior shrinks, so the optimal trade-off
$\lambda^\star = \lambda/\beta$ shifts toward 0 — exactly the
ridge plot of §5.

**Falsifies Theorem 1' if:** $\lambda^\star$ stays *constant*
across perturbation. Would mean the prior anchor maintains
its mode-coverage under dynamics drift — surprising for any
non-trivial perturbation. The most likely cause if observed:
the perturbation is **not actually changing the policy
landscape** (e.g., the gripper is already over-clamped, so
mass ×1.3 doesn't break grasps). Would force a sweep to
larger perturbations.

**Substantiates Theorem 1' if:** $\lambda^\star$ decreases
monotonically with $|\eta - 1|$, and the rate of decrease
matches the predicted $\rho^{-1}$ scaling from
`02_snis_bound.md` §4 closed-form (which gives an exponential
in $\|\mu_k - \tau^\star_{\eta}\|^2 / \beta$ from Corollary 1).

## 5. Analysis plan — the headline figure

**Plot.** $x$-axis: perturbation magnitude (mass scale or
friction scale, depending on panel). $y$-axis: optimal
$\lambda^\star$ that maximises success rate at that perturbation.
Two curves per panel: one for hybrid (the $\lambda^\star$-ridge),
one as the *predicted* curve from the soft-control reading
(monotone decrease in $\lambda^\star$ as perturbation grows).

**Secondary plot.** Same axes but $y$ = success rate at fixed
$\lambda = 1$ (the canonical operating point). Three curves: BC,
MBD-no-prior, hybrid. Predicted shape:

- BC: degrades monotonically with perturbation (memorised demos
  fail).
- MBD-no-prior: roughly invariant (its $f$ is recomputed).
- Hybrid: between the two; closer to MBD-no-prior at large
  perturbation, closer to BC at zero perturbation.

The **gap between hybrid and BC widens** with perturbation — the
same widening-gap story as cost-time adaptation (spec 01), but on
a different perturbation axis. Two independent widening-gap
plots from two independent perturbation types is a
**substantially stronger adaptation claim** than either one
alone.

## 6. Falsification protocol

The cleanest "the theory is wrong" outcomes:

- **$\lambda^\star$ does NOT shift with perturbation.** Indicates
  the demonstration prior remains an equally good mode hint even
  at 1.5× mass — either because the Robomimic demos are at the
  variance limit of the dynamics (the gripper grip is robust) or
  because the soft-control bias term is dominated by something
  other than dynamic correctness. Either way: the
  paper's "adaptability" framing needs revision.
- **MBD-no-prior degrades with perturbation.** Would mean the
  rollout simulator was NOT being mutated — a wiring bug, not a
  result.
- **Hybrid degrades faster than BC.** Would mean the bridge is
  pulling toward stale demos rather than letting MBD correct.
  Re-run at *small* $\lambda$ (e.g., $\lambda = 0.1$) to test
  whether the hybrid can recover MBD-no-prior's invariance.

## 7. Connection to spec 01

Specs 01 and 02 are **independent halves of the
$p^{\mathrm{model}}$-factor adaptation claim**. Together they
make a single story: the hybrid's online MBD factor *also*
provides robustness to whichever side of the model is
perturbed — cost or dynamics. The final report should present
both, with the headline figures stacked side-by-side.

## 8. Status

- 2026-05-25: branch created, flags + stubs + spec landed.
- Pending: link-body and friction-pair registries + ~15 lines
  of MuJoCo mutation logic + smoke cell. ~30 min of work.
  **Do NOT fire scored cells until that lands and the smoke
  passes.**
- Once landed: ~9 h to a publishable figure for the final
  report.
