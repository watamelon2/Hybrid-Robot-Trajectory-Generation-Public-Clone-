# MBD gated/tempered slate family -- findings (2026-06-05)

All runs: `demo_1nn` prior, Can unless noted, K=10, M=16, H=120,
MPC replans=1 chunk=60, n=30/cell (sparse n=10). Read any slate with
`python experiments/15_unified_hybrid/analyze_tempered_slate.py
--results-dir <dir>`.

## Headline (one line)
The **additive score-bridge** (`combine_mode=score`) is the operating
mode that works (Can 66.7-73.3%, Lift 100%); the **multiplicative
tempered/gated reweighting** family (`combine_mode=tempered`) is far
weaker at M=16 (3-20%), though *within* that family a **finite temper-p
does beat both the product (p=0) and the max-gate (p=inf) endpoints**.

## 1. sigma_demo calibration (Section D, COMPLETE)
The demo Gaussian must be scaled so `ldem-lmod ~ 0`. On Can that is
**sigma_demo = 1** (gate_frac 0.40). At the old default 0.25 the demo
term dwarfs the cost (ldem-lmod = -720, gate dead) -- which is why the
*first* slate's Section A was degenerate. `pick_calibrated_sigma.py`
returns 1 automatically. Calibration is **task-specific**: Lift balances
near sigma=1 (Dg1 ldem-lmod +3), Square needs sigma>16 (Dg1 -41),
tool_hang higher still.

## 2. Tempered-p rule (Section A, recalibrated @ sigma=1)
| p | mode | succ % | gate_frac |
|---:|---|---:|---:|
| 0 | product | 6.7 | 0.38 |
| **1** | **tempered** | **20.0** | 0.35 |
| 2 | tempered | 6.7 | 0.36 |
| 5 | tempered | 10.0 | 0.34 |
| inf | max-gate | 3.3 | 0.34 |

**Finite p=1 beats both endpoints** (20% vs 6.7% product vs 3.3%
max-gate) -- the doc 09 §6.2 prediction holds *qualitatively*. BUT every
tempered cell is far below the score-bridge product (66.7%). The
generalised-mean reweighting of only M=16 samples is high-variance; the
additive score gradient (applied to every sample) is the stronger
mechanism. **n=30 -> CI ~+/-15pp, so p=1>endpoints needs replication.**

## 3. sigma-mode (Section B): const 73.3% > decay 66.7% (legacy score path).

## 4. Off-distribution gate rescue (Section C @ sigma=1) -- NEGATIVE
| cell | succ % | gate_frac |
|---|---:|---:|
| mass2.5 product | 0.0 | 0.25 |
| mass2.5 max-gate | 0.0 | 0.30 |
| fric3.0 product | 3.3 | 0.35 |
| fric3.0 max-gate | 3.3 | 0.36 |

At the calibrated sigma=1 the **gate does NOT rescue** off-distribution:
both product and max-gate collapse and gate_frac stays ~0.3 (does not ->1).
The earlier "product collapses / gate holds" story was an artefact of
sigma=0.25. The gate only dominates (gate_frac->1) at large sigma_demo,
but large sigma kills on-distribution performance -- **no single static
sigma_demo both rescues off-dist and preserves on-dist** (motivates an
adaptive sigma_demo, follow-up below).

## 5. Warm-start vs persistent bridge (Slate #5)
| schedule | succ % |
|---|---:|
| pure warm-start (lambda=0) | 13.3 |
| **persistent (constant)** | **73.3** |
| linear anneal-out | 33.3 |
| cosine anneal-out | 40.0 |
| inverse-linear anneal-in | 20.0 |

The bridge must **persist across all denoising iters**: warm-start alone
is 13.3%; the constant bridge is 73.3%; every annealing schedule
degrades it. The score-bridge does continuous work, not just seeding.

## 6. Demo-quality gate-ceiling (Section DQ) -- STRONG, CLEAN
| band | gate % | product % | gap |
|---|---:|---:|---:|
| optimal | 10.0 | 53.3 | +43.3 |
| mid | 3.3 | 40.0 | +36.7 |
| worst | 0.0 | 0.0 | +0.0 |

**Product >= gate at every band; both fail on a bad demo.** The hard
max-gate throws away cost information the soft product keeps; the gap is
huge for good/mid demos and closes only when the demo is too poor to
help either rule. Cleanest support for "soft beats hard".

## 7. lambda x p operating surface (Section E @ sigma=1)
Best cell = lambda=0.5, p=1 (13.3%) -- again finite-p, *low* lambda.
Higher lambda=2 uniformly hurts. Interior-ish optimum but low absolute
(same M=16 variance ceiling as Section A).

## 8. Cross-task generality (Section G)
| task | score-ref | product@s1 | max-gate@s_best |
|---|---:|---:|---:|
| **Lift** | **100.0** | 80.0 | 83.3 (sigma=16) |
| Square | 0.0 | 0.0 | 3.3 |

- **Lift transfers cleanly**: score-ref 100%, product 80% > max-gate at
  sigma=1 (66.7%). The Can pattern holds.
- **Square is 0% across the board** (even score-ref) at H=120 -- a
  long-horizon contact-rich task the current config cannot solve. Not a
  combination-rule issue; a horizon/config issue (follow-up below).

## 9. Sparse-reward regime (Section S, tool_hang)
| cost | arm | succ % | mean reward | gate_frac |
|---|---|---:|---:|---:|
| raw (sparse) | product/gate/score | 0.0 | **0.000** | 0.000 |
| dense | product | 0.0 | 72.7 | 0.000 |
| dense | max-gate | 0.0 | 77.7 | 0.000 |
| dense | score | 0.0 | **112.5** | 0.000 |

- **Confirms the signal-density thesis (doc §7.1):** raw sparse -> mean
  reward 0 (flat J, no gradient); dense shaping -> reward 70-112 (signal
  returns). This is the core §7.1 claim, demonstrated.
- **Inconclusive on product-vs-gate** here: gate_frac=0 everywhere
  (sigma=1 is far too small for tool_hang, so the demo branch never
  fires) and 0% success (H=120 too short for tool_hang). Needs per-task
  sigma calibration + longer horizon (follow-up below).

## Open follow-ups (designed 2026-06-05, see slates)
- **T1 (M-scaling):** does the tempered family close the gap to the
  score-bridge as M grows {16,64,256}? Tests the "16-sample variance"
  explanation for §2/§7's weakness. -> `slate_mbd_tempered_Mscale`.
- **HARD (config rescue):** Square H={200,300}+replans, tool_hang
  per-task sigma{16,64,256}+H=300, so the hard tasks leave 0% and the
  cross-task / sparse comparisons become meaningful.
  -> `slate_mbd_hardtask_config`.
- **Replication:** re-run A2_cal_{p0,p1,pinf} at n=90 (seeds 0/1/2) to
  put a CI on "finite-p beats both endpoints".
- **Adaptive sigma_demo:** scale sigma_demo by detected distribution
  shift (model-branch likelihood drop) so the gate can rescue off-dist
  WITHOUT the on-dist penalty of a large static sigma (needs code).

---

# Follow-up results (2026-06-06, T1/T2/HARD slates COMPLETE)

## T1 -- M-scaling: tempered improves with M but NEVER catches the bridge
| M | score-bridge | tempered p=1 |
|---:|---:|---:|
| 16 | 66.7 | 6.7 |
| 64 | 66.7 | 10.0 |
| 256 | 63.3 | **30.0** |

At M=256 the tempered endpoints are product(p0) 33.3, p1 30.0,
max-gate(pinf) 20.0.

**Conclusions:**
- The **score-bridge is M-insensitive** (saturates at M=16: 66.7% with
  16 samples = 63.3% with 256). The additive gradient does not need a
  large sample budget.
- The **tempered family scales strongly with M** (6.7->10->30%),
  confirming the sample-variance hypothesis -- BUT even at 16x the
  budget it reaches only ~30-33%, still HALF the bridge. **Variance is
  part of the story, not all of it: a structural gap remains.** The
  score-bridge is both cheaper and better.
- At M=256 the ordering is product >= finite-p >= max-gate -- the §2
  finite-p advantage did NOT survive the larger budget.

## T2 -- replication n=90: "finite-p beats endpoints" does NOT replicate
| p | succ % (n=90) |
|---:|---:|
| 0 (product) | 12.2 |
| 1 | 6.7 |
| 2 | 8.9 |
| inf (max-gate) | 10.0 |

**The n=30 headline (p=1 = 20% > endpoints) was small-sample noise.** At
n=90 all tempered cells are statistically indistinguishable (~7-12%,
CI ~+/-7pp) and p=1 is in fact the *lowest*. **The doc 09 §6.2
"interior-optimum" prediction is NOT supported.** Within the tempered
family the combination-rule choice (product vs gate vs finite-p) does
not matter much; what matters is `combine_mode=score` vs `tempered`.

## HARD -- Square rescued by horizon; tool_hang still unsolved
| task | cell | succ % |
|---|---|---:|
| Square | score H=200 | 23.3 |
| Square | score H=300 | 23.3 |
| Square | product H=300 (sigma=16) | 3.3 |
| tool_hang | score/product/gate, H=300, sigma{16,64,256} | 0.0 (all) |

- **Square 0% WAS a horizon issue**: H=120 -> 0%, H>=200 -> 23.3% on the
  score path. Confirmed config, not combination-rule.
- **tool_hang stays 0%** at every config; gate_frac=0 even at sigma=256
  (the demo branch never engages) and the score path is also 0%. Needs
  more than horizon+sigma -- a value baseline / subgoal decomposition
  (doc §7.1 routes 2/4), not tuning.

## Net revised story
1. **Score-bridge dominates and is cheap** (M-insensitive, wins every
   solvable task: Can 66.7%, Lift 100%, Square 23.3%). This is the
   robust headline.
2. **Tempered/gated reweighting is variance-limited AND structurally
   weaker** -- it climbs with M but plateaus at ~half the bridge.
3. **No interior temper-p optimum** survives replication; the rule knob
   is a red herring vs the score-vs-tempered choice.
4. **Hard tasks**: Square is a horizon fix; tool_hang needs a dense
   value/subgoal signal, not parameter tuning.

---

# Follow-up-2 results (2026-06-07, DR/AD/DP3 slates COMPLETE)

## DR -- dense-revival on solvable Can/Lift: dense reward does NOT rescue tempered

| task | reward | product | max-gate | score |
|---|---|---:|---:|---:|
| Can | raw | 13.3 | 16.7 | **60.0** |
| Can | dense | 3.3 | 6.7 | **56.7** |
| Lift | raw | 73.3 | 83.3 | **100.0** |
| Lift | dense | 73.3 | 56.7 | **100.0** |

The score-bridge stays dominant under both reward choices. Dense reward
does not revive product/gate on Can, and it weakens max-gate on Lift. This
falsifies the optimistic "continuous dense reward may make tempered/gated
competitive on solvable tasks" reading. Dense signal is necessary for
hard sparse tasks, but it is not sufficient to close the structural
score-vs-tempered gap.

## AD -- adaptive sigma_demo: NEGATIVE

| perturbation | static product | adaptive product | adaptive max-gate |
|---|---:|---:|---:|
| canonical | **20.0** | 0.0 | 3.3 |
| friction 3.0 | **10.0** | 0.0 | 3.3 |
| mass 2.5 | 0.0 | 0.0 | 0.0 |

Adaptive sigma_demo collapses the demo branch too aggressively:
`ldem-lmod` becomes ~-10k to -13k and gate_frac drops near zero. The
off-distribution gate-rescue idea is not just unproven; this
implementation is actively worse than the calibrated static product on
canonical and friction-shifted Can. Do not spend more compute tuning
adaptive sigma unless the adaptation law is redesigned from first
principles.

## DP3 -- demo-phase pseudo-cost: dense progress without success

| task | arm | succ % | mean reward |
|---|---|---:|---:|
| Can | product | 6.7 | 580.8 |
| Can | max-gate | 3.3 | 595.0 |
| Can | score | **66.7** | **653.3** |
| tool_hang | product | 0.0 | 1001.2 |
| tool_hang | product sigma=16 | 0.0 | 1033.4 |
| tool_hang | max-gate | 0.0 | 1103.9 |
| tool_hang | max-gate sigma=16 | 0.0 | 874.2 |
| tool_hang | score | 0.0 | 1225.1 |
| tool_hang | score scale=30 | 0.0 | **3336.8** |

The demo-phase reward is a valid dense-progress signal -- rewards are
large and nonzero -- but it still produces 0% success on tool_hang across
score/product/gate and sigma/scale variants. This closes the "maybe
nearest-demo phase alone rescues sparse reward" route. tool_hang now
needs route 2/4: learned value or explicit subgoal decomposition, not
more scalar reward shaping or gate calibration.

## Final future-work triage after all follow-ups

1. **Keep / headline**: additive score-bridge with persistent data pull.
2. **Keep as negative appendix**: tempered/gated family; useful because
   it falsifies the MBD-mixture/max-gate path and supports PoE/score
   fusion.
3. **Stop**: temper-p sweeps, static/adaptive sigma_demo rescue,
   lambda-annealing, dense-revival for tempered, demo-phase-only
   tool_hang.
4. **Next Square direction**: score path at H>=200 plus better
   representation / coordinate frame / subgoal diagnostics.
5. **Next tool_hang direction**: learned value, phase-conditioned
   subgoals, or hierarchical cost; do not tune product/gate further.
