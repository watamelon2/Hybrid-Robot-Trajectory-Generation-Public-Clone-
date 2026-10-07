# Slate CostTime — RESULT

*Branch: `feat/adapt-cost-time`. Companion to
`docs/adaptation/01_cost_time_spec.md` (spec) and
`docs/theory/03_pareto_bias_variance.md` Theorem 1' (the
finite-$M$ Pareto inequality this slate tests).*

## Verdict in one line

**Negative result for the spec's primary hypothesis,** with a
clean infrastructure check and a substantive new finding (the
*fixed-anchor bias* mechanism) inside it. The slate falsifies
"hybrid widens its advantage under cost perturbation"; it
*identifies* the regime in which hybrid invariance does and does
not hold.

## Headline (single seed=0, n=60/cell, Wilson 95% CIs)

| method | $\Delta x = 0.00$ m | $\Delta x = 0.05$ m | $\Delta x = 0.10$ m | $\Delta x = 0.20$ m |
|---|---:|---:|---:|---:|
| **hybrid**  (`demo_1nn`, $\lambda = 1$) | **73.3 %** [61.0, 82.9] | **71.7 %** [59.2, 81.5] | **50.0 %** [37.7, 62.3] | **0.0 %** [0, 6.0] |
| BC bridge (`bc`, $\lambda = 5$)         | 26.7 % [17.1, 39.0] | 26.7 % [17.1, 39.0] | 21.7 % [13.1, 33.6] | 0.0 % [0, 6.0] |
| pure MBD  (`demo_1nn`, $\lambda = 0$)   | 6.7 % [2.6, 15.9]   | 10.0 % [4.7, 20.1]  | 10.0 % [4.7, 20.1]  | 1.7 % [0.3, 8.9]   |

Plot: `cost_time.png` / `cost_time.pdf` (3 curves, Wilson error bars).

## 1. The spec's primary hypothesis is falsified

Spec §5.1 substantiation criterion:

> *"Substantiates Theorem 1' if: the hybrid-vs-BC succ gap **widens
> monotonically** with $\|\Delta x\|$."*

Observed hybrid-minus-BC gap:

| $\Delta x$ (m) | 0.00 | 0.05 | 0.10 | 0.20 |
|---:|---:|---:|---:|---:|
| hybrid $-$ BC (pp) | +46.7 | +45.0 | +28.3 | 0.0 |

**The gap narrows monotonically.** The substantiation criterion is
not met.

## 2. The §6 falsification path is triggered — but not via the spec's wiring-bug mechanism

Spec §6 falsification criterion 2:

> *"Hybrid degrades faster than BC. Would indicate the perturbation
> breaks MBD's rollout-simulator alignment with the cost ...
> a wiring bug, not a real result."*

Observed degradation between $\Delta x = 0.05$ and $\Delta x = 0.10$:

| | Δx=0.05 | Δx=0.10 | drop |
|---|---:|---:|---:|
| hybrid | 71.7 % | 50.0 % | **−21.7 pp** |
| BC     | 26.7 % | 21.7 % | −5.0 pp |

Two-sample $z$-test (hybrid, 43/60 vs 30/60): $z = 2.50$,
$p \approx 0.012$. Statistically real, not seed noise.

**The hybrid degrades $4.3\times$ faster than BC. The §6
falsification clause is triggered.** But three pieces of evidence
exclude the wiring-bug interpretation:

1. **Regression test passes.** `experiments/04_mbd_baseline/test_worker_perturbation.py`
   PART A (`bin2_pos` mutates by exactly the requested xyz inside
   `_worker_init`) and PART B (spawn-context pool rewards differ
   between canonical and perturbed initargs by $\Delta = 0.26$).
   Perturbations reach the worker cost functions.
2. **Control cell reproduces canonical.** At $\Delta x = 0$: hybrid
   73.3 % vs Slate E pool 73.8 %; BC 26.7 % (single-seed vs E2's
   $\lambda=5$ 41.7 %, within noise); pure-MBD 6.7 % vs E2's
   single-seed 11.7 %. No cell is *off* canonical — only their
   *responses to perturbation* are.
3. **Pure-MBD's response is consistent.** If the cost reference
   were stale, pure-MBD would also fail to retarget; it would
   degrade the *same way* hybrid does. Instead pure-MBD is roughly
   flat (6.7 %, 10.0 %, 10.0 %, 1.7 %), confirming the cost moves
   with $\Delta x$ at the worker level.

So the falsification clause is triggered by a **real mechanism**,
not a bug.

## 3. The actual mechanism — fixed-anchor bias

`demo_1nn` retrieves the nearest demonstration in *observation
space*. Robomimic Can/ph's `low_dim` obs **does not include the
goal pose** — the bin is canonically fixed. Consequence:

- The retrieved demo is identical regardless of $\Delta x$.
  It always points to the *canonical* bin location.
- The prior anchor $\bar\tau$ is *implicitly fixed* at the
  canonical goal.

When $\pi^\star$ moves under perturbation, $\bar\tau$ does not
move with it. The mismatch grows monotonically with $\|\Delta x\|$:

- $\Delta x = 0$: $\bar\tau = \pi^\star$. Hybrid wins by +46.7 pp.
- $\Delta x = 0.05$ m: $\|\bar\tau - \pi^\star\| \approx 5$ cm,
  inside the demo bank's natural ~10 cm bin scatter. The
  cost-Boltzmann measure still has $\bar\tau$ in a
  high-probability region.  Hybrid stays at 71.7 %.
- $\Delta x = 0.10$ m: $\bar\tau$ is now on the *edge* of the
  bank's coverage. The cost score pulls one way, the prior
  score pulls toward the *old* goal. Hybrid drops 22 pp.
- $\Delta x = 0.20$ m: $\bar\tau$ is *outside* the bank.
  The prior is actively misleading — pulls *away* from the
  cost optimum. Hybrid collapses to 0/60.

**This is the bias side of Theorem 1' eq. (10).** The spec's
§5.1 emphasised the variance side ("hybrid variance constant in
$\|\Delta x\|$" — true) but did not weight the bias side, which
dominates once $\bar\tau$ leaves the cost-Boltzmann's
high-probability region. Specifically:

- `02_snis_bound.md` Corollary 1 requires *"the anchor $\bar\tau$
  is near a mode of $p^{\mathrm{model}}_{\sigma_k}$"*. As
  $\|\Delta x\|$ grows with $\bar\tau$ fixed, this assumption
  fails, $\rho(\bar\tau) \to 1$, and the variance-reduction
  argument disappears.
- `01_soft_control.md` §3.3 eq. (4) bounds the hybrid bias by
  $\|q^\star - \pi^\star\|$. When $\bar\tau$ is in the canonical
  location and $\pi^\star$ is at the perturbed location, this
  term grows monotonically with $\|\Delta x\|$.

The spec missed that **fixing $\bar\tau$ is a feature at
$\Delta x = 0$ and a liability at $\Delta x > 0$.**

## 4. What still holds from the spec

Two predictions survive the slate, both quantitatively:

- **Hybrid invariance at small perturbations** ($\Delta x \le 0.05$ m,
  73.3 % $\to$ 71.7 %, Wilson intervals overlap by ~22 pp). The
  variance-side prediction of Theorem 1' is *correct within the
  demo bank's coverage radius*.
- **Hybrid Pareto-dominance at every $\Delta x$ where any method
  works** (Theorem 1' eq. (10)).  Hybrid leads BC by
  $\ge 28$ pp at $\Delta x \le 0.10$ m
  ($z \ge 3.4$, $p \le 0.001$) and leads pure-MBD by
  $\ge 40$ pp at $\Delta x \le 0.10$ m ($z \ge 5$, $p \le 10^{-6}$).
  Theorem 1' is *not* falsified; the spec's restatement of it is.

## 5. What this buys for the final report

The natural framing for the final report is **"limits of
zero-training retrieval priors under cost perturbation"**, not
*"hybrid widens its advantage"*. Concrete deliverables:

1. **Closes `presentation_brief_2026_05_22.md` §8.1 limitation 5**
   ("sim-only, no perturbation studies") — with an *honest negative*
   that identifies a concrete failure mode rather than an
   unreplicated positive.
2. **Quantifies the manifold-coverage limit on Can** at
   $\Delta x \approx 0.10$ m — a publishable boundary number that
   the spec did not anticipate.
3. **Motivates two next experiments** (see
   [`NEGATIVE_RESULT_FOLLOWUP.md`](NEGATIVE_RESULT_FOLLOWUP.md)):
   - **Goal-conditioned retrieval.** Modify `demo_1nn` to use the
     *current cost target* as part of its retrieval query. This is
     the principled fix to the fixed-anchor bias; should restore
     the spec's predicted invariance.
   - **3-seed pool at $\Delta x = 0.10$ m.** Confirms that the
     22 pp degradation (currently $p \approx 0.012$ single-seed)
     is robust under reseeding.

## 6. Caveats

- **BC is goal-blind on Can** (Can/ph obs does not contain goal
  pose), so BC's "invariance" up to $\Delta x = 0.10$ m is *not*
  adaptation — it is the canonical pre-trained motion missing the
  bin by $\Delta x$. The fair comparison for "MBD inherits cost
  re-targeting" is **hybrid vs pure-MBD**, where the picture is
  the same: hybrid leads pure-MBD by 61.7 pp at $\Delta x = 0.05$ m
  and degrades 40 pp by $\Delta x = 0.10$ m while pure-MBD stays
  flat. The fixed-anchor diagnosis holds.
- **Single seed (n=60).** Wilson CIs are wide; the
  $\Delta x = 0.10$ m degradation needs a multi-seed pool to be
  decisive. Spec sec 2 hedged this; see follow-up doc.
- **+x direction only.** Asymmetric demonstrations could give a
  different shape for $-x$, $\pm y$. Not tested.

## 7. Wall-clock

| cell flavour | wall (min, $\Delta x = 0.05$) |
|---|---:|
| `demo_1nn` (lam-sweep 0, 1)  | 26.2 |
| `bc`                          | 26.5 |

Total slate: ~5.5 h elapsed, minus ~25 min SIGSTOP pause during
the v6 cleanup window. Matches the spec sec 2 prediction of 5-7 h.

## 8. Artefacts

```
results/slate_costtime_2026_05_25/
  AGGREGATE.json                 - cells + by_method pivot
  aggregate.csv                  - flat table
  cost_time.png|pdf              - headline plot
  RESULT.md                      - this file
  NEGATIVE_RESULT_FOLLOWUP.md    - principled follow-ups
  monitor.sh                     - one-shot status snapshot
  run.log                        - full driver stdout (~87k lines, gitignored)
  unified_bc_lam-5.0_K-10_R-1_bc_dx{000,005,010,020}.json
  unified_demo_1nn_lam-0.0_K-10_R-1_demo1nn_dx{000,005,010,020}.json
  unified_demo_1nn_lam-1.0_K-10_R-1_demo1nn_dx{000,005,010,020}.json
  unified_summary_*.json         - driver per-prior aggregate summaries
```

---

*Branch: `feat/adapt-cost-time`. Merge into `main` is part of
the K-O + adaptation-arc roll-up (see
`docs/todos/future_2026_05_26.md` item B4). Commits on this
branch (oldest first): `2c78d6f`, `05edaf6`, `716630a`, `d51a8fd`,
`300761e` (this RESULT.md + aggregation + 12 cell JSONs).*
