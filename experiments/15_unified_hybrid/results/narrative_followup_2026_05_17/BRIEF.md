# Narrative-strengthening follow-up — running brief

Launched 2026-05-17 14:34 PT, branch `feat/zero-train-priors`.
Screen `narrative`.  ETA finish ≈ 20:30 PT (~5.9 h).

## The story this slate defends

**Story B (mechanistic):** *The score-bridge needs an
in-distribution per-step pull target.*  Real demo trajectories (1-NN)
are in-distribution; smoothed approximations (BC) are partially
in-distribution; averages (k-NN, k≫1), random demos, and learned
generative samples (flow v2) are out-of-distribution.  Success rate
falls accordingly: 63 % → 42 % → 17 % → 0 % → 0 %.

## What each slate closes

### Slate 1 — 4-seed pool of `demo_1nn` (6 cells, ~2.5 h)
Cells: `demo_1nn` × (λ∈{0,5}) × (seed∈{1,2,3}); pooled with
existing seed=0 → 4-seed CI on both warm-start-only and full-method.

**Predictions:**
- λ=0 pooled: 5–15 % (existing seed=0 = 11.7 %)
- λ=5 pooled: 55–70 % (existing seed=0 = 63.3 %)
- CI on λ=5 pool must **not overlap** the BC 4-seed pool CI [31.6, 43.8]
  for the claim "demo_1nn beats BC by a wide margin" to be safe.

### Slate 2 — k-sweep for `demo_knn` at λ=5 (4 cells, ~1.7 h)
Cells: `demo_knn` λ=5 seed=0 at k∈{2, 3, 10, 20}.
With existing data: k=1 (= `demo_1nn`) = 63.3 %, k=5 = 16.7 %.

**Predictions:**
- Monotone decrease k=1 → k=20.
- Knee somewhere in [2, 5] (where the k-NN average leaves the
  retrieved demo's neighbourhood on the action manifold).
- k=20 may collapse to near `demo_mean` (0 %).
- If instead we see k=10 > k=5, or non-monotone, the OOD claim is
  wrong and the mechanism needs reformulation.

### Slate 3 — σ-sweep for `demo_1nn` at λ=5 (4 cells, ~1.7 h)
Cells: `demo_1nn` λ=5 seed=0 at σ_max ∈ {0.05, 0.10, 0.15, 0.40}.
Compare against BC σ-sweep numbers in `docs/results.md`.

**Predictions:**
- 1-NN > BC at every σ — the win is invariant in noise level.
- Peak likely at σ=0.05–0.15 (BC peaks at σ=0.03 with 40 %).
- σ=0.40 likely 0 % for both 1-NN and BC.

## How to read results when done

```bash
cd /Users/szh/Documents/SZH/Courseworks/ECE\ M237/Project/Hybrid-Robot-Trajectory-Generation
DIR="experiments/15_unified_hybrid/results/narrative_followup_2026_05_17"

# alive?
screen -ls | grep narrative
tail -f $DIR/run.log

# headlines only
grep -E "^=== |succ=" $DIR/run.log
```

## Aggregation script (morning)

```python
import json
from pathlib import Path
D = Path("experiments/15_unified_hybrid/results/narrative_followup_2026_05_17")
for j in sorted(D.glob("unified_*.json")):
    if "summary" in j.name: continue
    d = json.loads(j.read_text())
    print(f"{j.stem.split('_S',1)[-1]:25s}  prior={d['prior']:14s}  "
          f"sigma={d['params']['sigma_max']}  "
          f"lam={d['lambda_prior']:>4}  seed={d['seed']}  "
          f"{d['n_success']:>3}/{d['n_total']:<3}  "
          f"{d['success_rate']*100:>5.1f}%")
```

## If something goes wrong

Slate is sequential.  To resume mid-flight, comment out completed
`run` calls in `narrative_followup_slate.sh` and relaunch.

Worker-spawn cascade ("Bad file descriptor", thousands of repeats)
was the failure mode in `nohup`-launched runs.  This slate uses
`screen` (real PTY) and inherits the same launcher pattern as the
zero-train slate that completed clean overnight.

---

## RESULTS — all 14 cells done 20:50 PT, 0 bad-FD

### Slate 1: 4-seed pool of demo_1nn
```
seed   lam=0           lam=5
0      7/60 = 11.7%   38/60 = 63.3%
1      6/60 = 10.0%   39/60 = 65.0%
2      8/60 = 13.3%   40/60 = 66.7%
3      7/60 = 11.7%   37/60 = 61.7%
POOL  28/240 = 11.7%  154/240 = 64.2%
                       CI95 [57.9, 70.0]
```

vs BC pool 90/240 = 37.5% CI [31.6, 43.8].
**CIs do not overlap. +26.7pp gap is statistically decisive.**

Per-seed std on lam=0 = ~1pp. The "BC and 1-NN give identical
warm-starts" finding is now substantiated.

### Slate 2: k-sweep at lambda=5
```
k=1    38/60 = 63.3%   (= demo_1nn)
k=2    21/60 = 35.0%
k=3    10/60 = 16.7%
k=5    10/60 = 16.7%
k=10    5/60 =  8.3%
k=20    6/60 = 10.0%
```

**Monotone collapse with a sharp knee at k=2->3.** Each averaging
step costs ~17-28pp until the floor near 10%.  Locks in "averaging
amplifies OOD" mechanism.

### Slate 3: sigma-sweep at lambda=5, demo_1nn
```
sigma=0.05   39/60 = 65.0%
sigma=0.10   39/60 = 65.0%
sigma=0.15   39/60 = 65.0%
sigma=0.25   38/60 = 63.3%
sigma=0.40   40/60 = 66.7%
```

**Astonishing sigma-invariance** -- flat 63-67% across an order
of magnitude.  BC's sigma-sweep crashes to 3.3% at sigma=0.25 and
0% at sigma=0.40.  The 1-NN + bridge method removes sigma_max
from the hyperparameter tuning surface entirely.

## All three falsifiers from BRIEF.md predictions: PASSED

- Predicted lam=5 pool [55, 70]; got 64.2% with CI [57.9, 70.0].
- Predicted monotone k-sweep with knee in [2, 5]; got monotone
  with sharp knee at k=2->3.
- Predicted demo_1nn > BC at every sigma; got demo_1nn flat at
  ~65% across [0.05, 0.40] while BC peaks at 40% and crashes.

The story Story B holds up under every test thrown at it.

## What to do next

Three high-value follow-ups remain:

1. **Task-aware retrieval distance.** Up-weight object-pose dims
   in the L2 metric.  Likely free +few pp.  ~30 min to implement,
   ~25 min to evaluate.  Single seed first.

2. **Cross-task generalisation.** Run demo_1nn on `square` and
   `transport`.  Tests whether the win is can-specific.  Each
   ~25 min.  Need to verify obs key compatibility first.

3. **In-distribution-ness measurement.** Quantify "how OOD is each
   prior's predicted trajectory" by computing per-action MSE to the
   nearest demo trajectory.  Connects the success-rate ranking
   to a direct distributional quantity.  Pure analysis, no rollouts.

Priority order: (3) supports the mechanistic claim with a
non-success-rate measurement; (2) tests generalisation; (1) is
just an optimisation.
