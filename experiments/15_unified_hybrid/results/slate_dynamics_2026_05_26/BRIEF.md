# Slate Dynamics — D1' mass-ladder, minimal variant

Branch: `feat/adapt-dynamics`.
Spec: `docs/adaptation/02_dynamics_spec.md` (§2.1 D1, §4.1 Theorem 1' predictions).
Theory: `docs/theory/01_soft_control.md` §1 (the $p^{\mathrm{model}}$ factor;
this slate perturbs $f$, the simulator).
Companion: `slate_costtime_2026_05_25` (negative result on cost perturbation,
the other half of the $p^{\mathrm{model}}$ adaptation claim).

Launched 2026-05-26 ~12:19 PT, finished 2026-05-27 ~00:17 PT
across 4 attempts (the launcher was hardened into `_v2.sh` after
attempt 4 died at cell 6/12 from external SIGTERM).
12 / 12 cells landed. 720 total scored rollouts (60 per cell).

## Headline (one sentence)

**At canonical per-method operating points, hybrid / pure-MBD /
BC-bridge are all flat in success rate across ±50 % mass
perturbation on Can** — no method visibly degrades, the method
ordering is preserved at every mass, and the spec's central
prediction (BC degrades fastest, hybrid variance grows with
perturbation) is *not* observed in this minimal grid.

This is the **§6 falsification-bullet-1 outcome**: the Robomimic
Can demos appear to sit at a *variance limit* of the gripper
grasp, so a 1.5× mass shift does not break the canonical
operating point.  The "dynamics adaptability" framing in the
spec needs revision OR a stronger perturbation OR the full
$\lambda$-grid to be testable in this regime.

## Setup

- Task: PickPlaceCan (`robosuite v1.5`, ph dataset, 60 evaluations per cell).
- Perturbation: robot-link mass scale $\eta \in \{0.70, 1.00, 1.30, 1.50\}$
  via `--mass-scale`, applied uniformly to body masses (MuJoCo `body_mass`
  field), with the rollout simulator $f$ re-instantiated under $\eta$.
  Friction scale held fixed at 1.0.
- Methods (canonical operating points only, *not* a full $\lambda$-grid):

  | method | prior | $\lambda$ | $\sigma_{\max}$ | source |
  |---|---|---:|---:|---|
  | `hybrid_demo1nn` | `demo_1nn` | 1.0 | 0.25 | Slate E pooled peak (73.8 % @ $\eta=1$) |
  | `pure_mbd`       | `demo_1nn` | 0.0 | 0.25 | $\lambda=0$ zeros the prior; BC ckpt unused |
  | `bc_bridge`      | `bc`       | 5.0 | 0.25 | Slate E E2 BC peak (~37 % @ $\eta=1$) |

- Driver: `experiments/15_unified_hybrid/run_can.py --mass-scale $\eta`,
  $K=10$, $R=1$, $M=16$, $H=120$, 10 parallel workers, seed=0.
- Scope: **single canonical $\lambda$ per method**, mirroring the
  cost-time slate's 12-cell structure.  The full 5-$\lambda$ × 4-mass
  heatmap (D1 in the spec) is *deferred* — see §Follow-ups.

## Results

12 cells, 720 total trials, Wilson 95 % CI per cell (z=1.96).

| mass | hybrid (λ=1) | pure_MBD (λ=0) | bc_bridge (λ=5) | wall (h) |
|---:|---|---|---|---:|
| 0.70 | **65.0 %** [52.4, 75.8] (39/60) | 11.7 % [5.8, 22.2] (7/60)  | 30.0 % [19.9, 42.5] (18/60) | 1.65 |
| 1.00 | **68.3 %** [55.8, 78.7] (41/60) |  8.3 % [3.6, 18.1] (5/60)  | 35.0 % [24.2, 47.6] (21/60) | 1.66 |
| 1.30 | **60.0 %** [47.4, 71.4] (36/60) | 11.7 % [5.8, 22.2] (7/60)  | 38.3 % [27.1, 51.0] (23/60) | 1.50 |
| 1.50 | **66.7 %** [54.1, 77.3] (40/60) | 13.3 % [6.9, 24.2] (8/60)  | 35.0 % [24.2, 47.6] (21/60) | 1.81 |

Pooled across masses (n = 240 per method):

| method | k/n | succ | Wilson 95 % CI |
|---|---:|---:|---|
| `hybrid_demo1nn` | 156 / 240 | **65.0 %** | [58.8, 70.8] |
| `bc_bridge`      |  83 / 240 |   34.6 %  | [28.9, 40.8] |
| `pure_mbd`       |  27 / 240 |   11.2 %  | [ 7.8, 15.9] |

Slate E canonical-mass anchor for cross-check:
hybrid pooled (n=240, 4 seeds) = 73.8 % [67.8, 78.9].
This slate's $\eta=1$ hybrid (n=60, seed=0) = 68.3 % [55.8, 78.7] —
fully within the pooled CI, confirming no drift in the operating point.

## What the data shows

1. **Flatness across mass, per method.**  For every method the
   four per-mass CIs overlap pairwise.  No single method shows a
   monotone trend in mass.  Including the no-perturbation cell
   ($\eta = 1.00$), hybrid spans 60.0 % – 68.3 %, a range of
   8.3 pp that is well within the typical sampling noise for
   n=60 (CI half-widths are ~11 pp).

2. **Method ordering preserved at every mass.**
   `hybrid_demo1nn` > `bc_bridge` > `pure_mbd` at $\eta \in
   \{0.70, 1.00, 1.30, 1.50\}$, with the hybrid–BC gap consistently
   $\approx 30$ pp and the BC–MBD gap consistently $\approx 23$ pp.
   The composition wins by the same margin under ±50 % mass shift
   as it does at canonical.

3. **CI widths flat with perturbation.**  Hybrid CI half-widths are
   {12.0, 11.5, 12.0, 11.6} pp at $\eta \in \{0.70, 1.00, 1.30, 1.50\}$
   — no inflation.  The Theorem 1' prediction "hybrid variance term
   grows with dynamics perturbation" (spec §4.1, last row of the
   prediction table) is *not* observed in the response variable.

4. **Pure-MBD is bottlenecked low and flat.**  ~11 % across masses
   matches the Slate E E2 $\lambda = 0$ point (11.7 % at canonical).
   This is the "MBD-no-prior bias depends on $M$ only" prediction
   from the spec, but at $M=16$ the absolute number is low enough
   that we cannot resolve dynamics-driven variance changes against
   the noise floor.  A larger $M$ slate (e.g. $M = 64$) would be
   needed to actually test the MBD-invariance claim cleanly.

5. **The dead-cell pattern from attempts 1–4 was external, not
   methodological.**  All four attempts that died (`attempt1_dead_*`
   through `attempt4_dead_*`) were killed by SIGTERM from outside
   the run (likely user pulses for CPU during the day); the
   `_v2.sh` launcher's resume-on-existing-JSON logic is what got
   the final run through cleanly.  No model / driver bug is
   implicated.

## Connection to spec predictions

| spec §4.1 prediction | observed | verdict |
|---|---|---|
| BC bias grows fastest with $\|\eta - 1\|$ | BC flat at 30–38 % | **not observed** |
| MBD-no-prior bias constant | MBD flat at 8–13 % | *consistent* (but n is small and absolute level is low) |
| Hybrid bias bounded by min(BC, MBD) | hybrid always > both | trivially consistent (gap is large) |
| Hybrid variance grows with $\|\eta - 1\|$ | CI widths flat (~12 pp) | **not observed** |
| $\lambda^\star$ shifts toward 0 as $\|\eta - 1\|$ grows | not testable (single $\lambda$ per method) | **out of scope for this slate** |

The two predictions that *are* observable in a single-$\lambda$
slate (BC degradation, hybrid CI inflation) both go null.
This matches **spec §6 falsification bullet 1** verbatim:

> "$\lambda^\star$ does NOT shift with perturbation.  Indicates the
> demonstration prior remains an equally good mode hint even at
> 1.5× mass — either because the Robomimic demos are at the
> variance limit of the dynamics (the gripper grip is robust) or
> because the soft-control bias term is dominated by something
> other than dynamic correctness."

Note: we did *not* directly observe $\lambda^\star$ shifting (we
didn't run a $\lambda$-grid).  But the bullet's two *observable*
predictions (BC degradation and hybrid CI inflation, both of which
would have been visible at single $\lambda$ if they were going to
occur) both went null, which is a strong indirect signal.

## What this lets us claim (paper-grade)

- ✅ **Robustness on the canonical operating point**: the hybrid
  composition that wins at canonical dynamics also wins, by
  approximately the same margin, at ±50 % mass perturbation.
  This is a useful *deployment* claim — the operating point
  doesn't need retuning across this mass range.
- ✅ **Method ordering invariant under mass shift**: `hybrid > bc_bridge > pure_mbd`
  at all four mass scales.

## What this does NOT let us claim

- ❌ **"Hybrid adapts to dynamics"** in the soft-control /
  $\lambda^\star$-shift sense.  We cannot rule it in or out from
  this slate; we'd need the full D1 grid (5 $\lambda$ × 4 mass).
- ❌ **"Bridge $\lambda$ should down-weight under perturbation"**
  (spec §1 prediction). Untested here.
- ❌ **General "dynamics adaptation" claim**.  Mass is one axis
  of $f$-perturbation; friction (D2) and gripper-clearance / contact
  parameters are independent axes that may show different
  behaviour.  D2 is deferred.

Most importantly: **a NEGATIVE result on the spec's central
"adaptability" prediction is the most parsimonious reading**, and
the spec's own falsification protocol §6.1 calls this out as the
likely cause if observed — "the Robomimic demos are at the
variance limit of the dynamics."  The paper's adaptability arc
on the dynamics axis needs either (a) stronger perturbation
(mass 2.0, 3.0, or replace mass with friction or contact-stiffness),
or (b) the full $\lambda$-grid, or (c) reframing as a
*robustness* claim rather than an *adaptability* claim.

## Recommended follow-ups (NOT part of this branch)

1. **D1 full grid** (5 $\lambda$ × 4 mass × 3 methods = 60 cells,
   ~5 h).  Direct test of the $\lambda^\star$-ridge prediction.
   This is the headline figure the spec §5 describes.
2. **Stronger mass perturbation** (mass 2.0, 3.0, 5.0) to find
   where the gripper grasp actually starts breaking.  If the
   variance limit hypothesis is right, there *should* be an
   inflection point where BC collapses.
3. **D2 friction ladder** (friction $\in \{0.5, 1.0, 2.0\}$, same
   3 methods × single $\lambda$, 6 new cells, ~2.5 h).  Friction
   is a different $f$-perturbation; gripper grasp depends on it
   *much* more sensitively than on mass.  Most likely place to
   actually see a BC-degradation signal.
4. **MBD-no-prior with larger $M$** ($M = 64$ instead of $M = 16$)
   at this same mass ladder.  Would test the MBD-invariance
   prediction at a regime where MBD can actually solve the task.

These follow-ups should each be their own slate / branch, not
appended to this one.

## Status & files

- `AGGREGATE.json` — structured numbers (cells + pooled + CI).
- `unified_summary_{prior}_K-10_R-1_{tag}_m{070,100,130,150}.json` ×8
  — driver outputs (per-cell summary, includes wall_s, per_lambda).
- `unified_{prior}_lam-{0,1,5}.0_K-10_R-1_{tag}_m{070,100,130,150}.json` ×12
  — driver outputs (per-trial detail).
- `run.log` (21 MB) — full stdout/stderr of the v2 launcher's
  successful pass; **excluded from commit** (size).
- `attempt{1,2,3,4}_dead_*` — pre-success attempts that died to
  external SIGTERM; **excluded from commit** (debug artefact;
  superseded by the live cells).
- `slate_dynamics_2026_05_26_v2.sh` — hardened launcher (resume
  logic + per-cell rc tracking).  Commits with the slate.
