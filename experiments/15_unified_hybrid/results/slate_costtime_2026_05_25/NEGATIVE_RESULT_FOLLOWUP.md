# Slate CostTime — follow-up experiments

*Branch: `feat/adapt-cost-time`. Companion to `RESULT.md` (the
negative-result writeup) and `docs/adaptation/01_cost_time_spec.md`.*

The slate falsified the spec's primary prediction
("hybrid-vs-BC gap widens with $\|\Delta x\|$") and identified
a previously-unmodelled mechanism (**fixed-anchor bias**: the
retrieval prior's anchor $\bar\tau$ is implicitly pinned to the
canonical goal because Can/ph obs does not contain the bin pose).
This document enumerates the experiments that follow from the
finding, in priority order.

## Tier 1 — confirms the finding (do before any final-report claim)

### F1. Multi-seed pool at $\Delta x = 0.10$ m

Single-seed gives the $\Delta x = 0.05 \to \Delta x = 0.10$ drop
of 22 pp at $z = 2.50, p \approx 0.012$. Borderline. Pool 3
additional seeds (1, 2, 3) at $\Delta x = 0.10$ m for all three
methods.

| cell | seeds | n_total | wall |
|---|---|---:|---:|
| hybrid + pure-MBD at $\Delta x = 0.10$, seeds 1–3 | 3 | 360 | ~80 min |
| BC at $\Delta x = 0.10$, seeds 1–3 | 3 | 180 | ~80 min |

**Total wall: ~2.5 h, single invocation:**

```bash
python experiments/15_unified_hybrid/run_can.py \
  --task can --K 10 --M 16 --H 120 \
  --sigma-max 0.25 --sigma-min 0.05 \
  --n-demos 20 --n-rollouts 3 --n-workers 10 \
  --seed 1 2 3 \
  --prior demo_1nn --lambda-prior 0 1 \
  --goal-perturbation-xyz 0.10 0.0 0.0 \
  --output-dir experiments/15_unified_hybrid/results/slate_costtime_2026_05_25_followup
```

(Plus a corresponding BC invocation.)

**Decision rule:** pool's Wilson CI on the hybrid-vs-pure-MBD
gap at $\Delta x = 0.10$ m. If lower bound > +20 pp, the
single-seed signal is confirmed; final report can cite it
without hedging. If lower bound < 0 pp, the 50.0 % single-seed
hybrid was bad luck and the headline narrative needs re-evaluation.

## Tier 2 — tests the proposed mechanism (do if Tier 1 confirms)

### F2. Goal-conditioned retrieval

The fixed-anchor diagnosis predicts that a **goal-aware
retrieval query** restores the spec's invariance prediction.
Implementation: extend `demo_1nn` to use
$(\mathrm{obs}, J(\tau)\text{'s target pose})$ as the query
instead of just $\mathrm{obs}$. Concretely, augment the demo bank's
key with the bin pose at recording time, augment the query with
the *current* (perturbed) bin pose, and re-fit the 1-NN
distance metric. The retrieved demo then tracks the new goal.

Re-run the slate's 4 perturbation cells with this prior. Spec
prediction (revised): hybrid invariant in $\Delta x$ across all
4 cells, gap to BC *widens* monotonically.

**Wall:** ~6 h for the full slate replay.
**Implementation cost:** ~1 day of infra
(query augmentation + per-demo-bank re-keying + a regression test).

This is the *principled fix* to the negative result and is the
single most informative follow-up.

### F3. Pure-MBD comparison at higher $M$

The slate's pure-MBD cell at $\Delta x = 0.10$ m is at $M = 16$,
where the variance term dominates per Theorem 1 of
`02_snis_bound.md`. The "MBD inherits cost re-targeting"
mechanism predicts pure-MBD should *also* recover at higher $M$.
Re-run pure-MBD at $\Delta x = 0.10$ m with $M \in \{64, 256\}$.

**Wall:** ~3 h (single-seed, perturbed cell only).
**Decision rule:** if pure-MBD at $M = 256$ reaches > 50 % at
$\Delta x = 0.10$ m, the spec's "MBD retargets" picture survives
at large compute; the hybrid loses its variance-reduction
advantage but the cost-side mechanism is real.

## Tier 3 — broader sweeps (do for the eventual paper)

### F4. $-x$, $\pm y$ perturbation symmetry

The slate ran $+x$ only. Demos on Can are asymmetric in $y$ (the
robot reaches from a particular side). Run a 3-cell mini-sweep
with $\Delta y \in \{0.05, 0.10\}$ and $\Delta x = -0.10$ to
check whether the manifold-coverage limit is direction-symmetric.

**Wall:** ~6 h for the 3 cells $\times$ 3 methods.

### F5. Cross-task replication on Lift

Can has the obs-goal-blindness confound. Lift's task is
"height threshold reached" — the goal is *implicitly* the
top of the gripper's workspace, perturbation analogue is the
**threshold height itself**. A Lift cost-time slate would test
whether the fixed-anchor bias mechanism generalises beyond Can.

**Wall:** ~6 h.
**Implementation cost:** ~half day to wire
`apply_threshold_perturbation` into `env_perturbations.py`
(currently `lift→None`).

### F6. Cost-shift only (decouple env-shift)

The current slate is *env-shift*: the bin actually moves, so the
env's reward function reads from the moved bin. A *cost-shift
only* slate would leave the env unperturbed and apply the shift
**only inside the cost function**. This is the cleaner test of
"does MBD retarget the cost?" because it removes any
env-perturbation confounds (collision geometry, contact
dynamics, etc.). Should give a *cleaner* hybrid invariance
prediction.

**Wall:** ~6 h.
**Implementation cost:** ~1 day.

## Not pursued (after triage)

- **C2 obstacle injection from spec sec 2.** The 20 cm collapse
  result makes this less interesting; the manifold-coverage
  limit already binds before obstacle effects become detectable.
- **dx > 0.20 m cells.** Floor is solid; no information in
  $\Delta x = 0.30, 0.40$, etc.

## Ordering and budget

If the final-report deadline allows ~12 h of compute and
~2 days of infra, do **F1 (2.5 h) + F2 (6 h + 1 day infra)**.
That converts the negative result into a *positive* result for
the goal-conditioned retrieval extension, which is a stronger
final-report thesis than the original spec.

If the deadline allows ~6 h of compute and no infra, do **F1**
only. That confirms the negative finding without claiming
the principled fix yet.

If the deadline is tight, **the negative result as written in
`RESULT.md` is enough to ship.** The mechanism is identified,
the bound is located ($\Delta x \approx 0.10$ m on Can), and
the follow-up is queued.

---

*This file is the bridge between the negative result and the
post-final-report research queue. Items F1–F6 are not
gate-blockers for the final report.*
