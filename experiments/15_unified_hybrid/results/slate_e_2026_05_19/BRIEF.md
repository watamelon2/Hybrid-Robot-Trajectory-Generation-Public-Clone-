# Slate E -- Story-B headline-tightening (running brief)

Launched 2026-05-19 ~02:40 PT, branch `feat/zero-train-priors`.
Screen `slate_e`.  ETA finish approx 09:00 PT (~6.5 h).
Gate is 10:00 PT, so the slate has a 1 h buffer.

## The story this slate defends

After Slate D the new project peak is **demo_1nn lam=1
sigma=0.25 K=10 R=1 = 45/60 = 75.0 %** (single seed=0).
Slate D itself listed three follow-ups to make this number
defensible; Slate E closes the two that fit in one overnight
plus the missing BC companion that gives Story B its
"prior-specific tuning" hook.

## What each block closes

### E1 -- 4-seed pool at lambda=1 sigma=0.25 demo_1nn (3 cells, ~105 min)

Cells: `--prior=demo_1nn --lambda-prior=1 --seed in {1,2,3}`;
pooled with the existing seed=0 = 45/60 cell from
`slate_d_2026_05_18/`.

**Prediction:** pool in [60, 75] with 95 % CI lower bound
above 63 %.  Reasoning: the May-17 4-seed pool at lambda=5 gave
64.2 % (CI [57.9, 70.0]) with per-seed range [61.7, 66.7] -- a
4-pp spread on 60 rollouts.  At lambda=1 the seed=0 cell is
11.7 pp above the lambda=5 mean, so even with a 5-pp
under-shoot from seed-noise we should land at pool ~ 65 %.

**Acceptance gates:**

- Pool >= 65 %, CI lower >= 60 %  =>  the +11 pp lift from
  lambda=5 -> lambda=1 is *substantiated*, the new pooled
  headline is the slate E1 number.  Story B's falsifiable
  prediction holds: hybrid peak strictly above bridge-
  dominated plateau.
- Pool in [55, 65], CI overlaps the lambda=5 pool  =>  the
  May-18 single-seed 75 % was a lucky outlier; quote the
  May-17 64.2 % as the pooled headline and note that lambda=1
  doesn't differ from lambda=5 at the seed-pool level.
- Pool < 55 %  =>  Story-B's "synergy" claim collapses.  This
  would be a major narrative pivot; not predicted by the
  May-18 single-seed shape.

### E2 -- BC lambda-sweep companion at sigma=0.25 seed=0 (5 cells, ~175 min)

Cells: `--prior=bc --lambda-prior in {0.5, 1, 2, 10, 20}
--seed=0`.  Combines with existing BC seed=0 cells:
- lambda=0 = 7/60 = 11.7 % (warmstart_ablation C1)
- lambda=5 = 25/60 = 41.7 % (phase 2 K=10 R=1)

So after E2 we have a 7-point BC lambda-curve at the headline
operating point, mirroring D2's 7-point demo_1nn curve.

**Prediction:** BC peak in [35, 50] somewhere in lambda in
[2, 10].  Reasoning: BC's per-step pull target is a learned
smoothing of multiple demos, not a single demo trajectory.
Score-bridge mechanism needs higher per-step pull strength to
push the MBD samples toward BC's smoothed action than toward
1-NN's raw demo action.

**Acceptance gates:**

- BC peak at lambda >= 5, and BC(lambda=1) < 1nn(lambda=1)
  by >= 20 pp  =>  Story-B's "bridge tuning is prior-specific"
  claim is *substantiated*.  The single-seed lambda=1 = 75 %
  for 1nn vs lambda=5 = 42 % for BC is the cleanest evidence.
- BC peak also at lambda=1  =>  prior-specific tuning is
  *partially* substantiated; the lambda location agrees but
  the magnitudes differ.  Story B is preserved but the headline
  becomes "prior quality matters at every lambda" rather than
  "the optimal lambda is prior-dependent".
- BC peak at lambda=20, > 50 %  =>  unexpected; suggests BC's
  pull target may also be in-distribution enough to want
  stronger pulls, and the prior-specific story flips sign.

### E3 -- demo_1nn sigma-tail at lambda=1 seed=0 (3 cells, ~105 min)

Cells: `--prior=demo_1nn --lambda-prior=1 --sigma-max in
{0.05, 0.40, 1.00} --seed=0`.  Combines with the existing
slate_d cell at sigma=0.25 lambda=1 = 45/60.

The May-17/18 sigma-invariance claim (sigma in [0.05, 1.00] all
63-68 % at lambda=5) is the strongest robustness statement in
the project.  But it was substantiated at lambda=5, not at the
new lambda=1 optimum.  E3 probes three sigma values to check
whether the same shape transfers.

**Prediction:** all three cells land in [65, 80] (i.e. at or
above the lambda=5 plateau, with a possible upward tilt).
Reasoning: at lambda=1 the bridge is weaker, so the MBD softmax
has more room to discover good actions independently; if the
bridge-dominated lambda=5 regime was already sigma-invariant,
the lambda=1 regime should be at-least-as sigma-invariant.

**Acceptance gates:**

- All three cells >= 60 %  =>  sigma-invariance extends to the
  new lambda optimum.  Combined with E1's pool, the headline
  becomes "robust across a 20x sigma range at the new optimum".
- One or more cells drop to < 30 %  =>  found a sigma-cliff
  at the new lambda; the May-18 D3 sigma-tail was specific to
  lambda=5.  Worth quoting both shapes.

## How to read results when done

```bash
cd /Users/szh/Documents/SZH/Courseworks/ECE\ M237/Project/Hybrid-Robot-Trajectory-Generation
DIR="experiments/15_unified_hybrid/results/slate_e_2026_05_19"

# alive?
screen -ls | grep slate_e
tail -f $DIR/run.log

# headlines only
grep -E "^=== |succ=" $DIR/run.log

# aggregate
python - <<'PY'
import json
from pathlib import Path
D = Path("experiments/15_unified_hybrid/results/slate_e_2026_05_19")
for j in sorted(D.glob("unified_*.json")):
    if "summary" in j.name: continue
    d = json.loads(j.read_text())
    print(f"{j.stem.split('_E',1)[-1]:35s}  prior={d['prior']:14s}  "
          f"sigma={d['params']['sigma_max']}  "
          f"lam={d['lambda_prior']:>4}  seed={d['seed']}  "
          f"{d['n_success']:>3}/{d['n_total']:<3}  "
          f"{d['success_rate']*100:>5.1f}%")
PY
```

## If something goes wrong

Slate is sequential (E1 -> E2 -> E3).  To resume mid-flight,
comment out completed cells in `slate_e_2026_05_19.sh` and
relaunch.  Screen-based launcher avoids the worker-spawn
cascade seen with `nohup`.

The most likely failure mode is a per-cell hang (the BC
warm-start path occasionally blocks on a torch RNG state
issue).  Each cell has its own python invocation, so a hang
in one cell does not propagate; just `screen -X -S slate_e
kill` and relaunch from the next cell.

## Source

- Launcher: `experiments/15_unified_hybrid/slate_e_2026_05_19.sh`
- Output dir: `experiments/15_unified_hybrid/results/slate_e_2026_05_19/`
- Branch: `feat/zero-train-priors`, HEAD `3041466` at launch

---

## RESULTS -- all 11 cells done 07:31 PDT, runtime 4 h 53 min

Full numbers in `AGGREGATE.json`; this section gives the read-out.

### E1 -- 4-seed pool at lambda=1 sigma=0.25 demo_1nn  *(NEW POOLED HEADLINE)*

```
seed   succ           source
   0   45/60 = 75.0 % slate_d_2026_05_18 D2 (existing)
   1   46/60 = 76.7 % E1_1nn_lam1_seed1
   2   44/60 = 73.3 % E1_1nn_lam1_seed2
   3   42/60 = 70.0 % E1_1nn_lam1_seed3
POOL 177/240 = 73.8 % CI95 [67.8, 78.9]
```

**The 75.0 % single-seed peak is substantiated as a pooled
headline.** Per-seed range is 70.0 - 76.7 %, std ~2.7 pp; the
4-seed Wilson 95 % CI lower bound is **67.8 %**, which is
**above the bridge-dominated plateau (~63 %)**.

Story-B's falsifiable prediction holds decisively:

- Predicted gate "pool >= 65 %, CI lower >= 60 %"  ->  ACTUAL
  pool 73.8 %, CI lower 67.8 %.  Passes both gates by wide
  margins.
- The hybrid peak is **strictly above both endpoints** of the
  lambda-axis at this operating point (warm-start-only = 11.7 %
  at lambda=0; bridge-dominated plateau = 63-65 % at lambda=5).
  The +10.5 pp lift from lambda=5 -> lambda=1 is therefore
  **attributable to the score-bridge sweet-spot at moderate
  lambda**, not seed-noise.

vs the BC 4-seed pool at the *same* operating point but
lambda=5 (Job C, May 15): 90/240 = 37.5 %, CI [31.6, 43.8].

**Delta(hybrid 1-NN lambda=1) - (BC lambda=5) = +36.3 pp pooled.**
The 95 % CIs do not overlap (BC upper 43.8 % vs 1-NN lower
67.8 %; ~24 pp separation).

### E2 -- BC lambda-curve at sigma=0.25 seed=0  *(inverted-U confirmed, peak at lambda=5)*

| lambda | succ   | rate    | source                       |
|-------:|--------|---------|------------------------------|
|    0   |  7/60  | 11.7 %  | warmstart_ablation C1        |
|  0.5   | 13/60  | 21.7 %  | E2 (new)                     |
|    1   | 17/60  | 28.3 %  | E2 (new)                     |
|    2   | 22/60  | 36.7 %  | E2 (new)                     |
|  **5** | **25/60** | **41.7 %** | _archive phase2 K=10 R=1 |
|   10   | 23/60  | 38.3 %  | E2 (new)                     |
|   20   | 17/60  | 28.3 %  | E2 (new)                     |

**Clean inverted-U with BC peak at lambda=5 = 41.7 %.**
Prediction was "BC peak at lambda >= 5" -> CONFIRMED.

Direct apples-to-apples comparison with `demo_1nn` D2 curve at
the same K=10 R=1 sigma=0.25 seed=0:

| lambda | BC succ | BC rate | 1-NN succ | 1-NN rate | gap (1nn - bc) |
|-------:|---------|---------|-----------|-----------|---------------:|
|    0   |  7/60   | 11.7 %  |  7/60     | 11.7 %    |    +0.0 pp     |
|  0.5   | 13/60   | 21.7 %  | 19/60     | 31.7 %    |   +10.0 pp     |
|  **1** | 17/60   | 28.3 %  | **45/60** | **75.0 %**|  **+46.7 pp**  |
|    2   | 22/60   | 36.7 %  | 39/60     | 65.0 %    |   +28.3 pp     |
|  **5** | **25/60** | **41.7 %** | 38/60 | 63.3 %    |   +21.7 pp     |
|   10   | 23/60   | 38.3 %  | 39/60     | 65.0 %    |   +26.7 pp     |
|   20   | 17/60   | 28.3 %  | 38/60     | 63.3 %    |   +35.0 pp     |

**Bridge tuning is strongly prior-specific** -- the optimal
lambda is 1 for `demo_1nn` and 5 for BC.  The two priors agree
at lambda=0 (11.7 % both -- no bridge, just warm-start
provenance) and the gap opens once the bridge is on.  At
lambda=1, `demo_1nn` is **46.7 pp above** BC at the *same*
operating point: BC's smoothed-action targets cannot use a
weak bridge productively, while 1-NN's raw-demo targets thrive
under a weak bridge that leaves MBD room to explore.

The BC curve area is uniformly below the 1-NN curve, and the
1-NN peak is 33.3 pp above the BC peak at their respective
optimal lambdas (75.0 % vs 41.7 %).

### E3 -- demo_1nn sigma-tail at lambda=1 seed=0  *(sigma-invariance does NOT extend to lambda=1)*

| sigma  | succ   | rate    | source                  |
|-------:|--------|---------|-------------------------|
|  0.05  | 39/60  | 65.0 %  | E3 (new)                |
|  0.25  | **45/60** | **75.0 %** | slate_d D2 (existing) |
|  0.40  | 32/60  | 53.3 %  | E3 (new)                |
|  1.00  | 14/60  | 23.3 %  | E3 (new)                |

**sigma-invariance is lambda-specific.**  Prediction was "all
three cells in [65, 80]" -> PARTIALLY FAILED.  sigma=0.05 lands
in-range, but sigma=0.40 drops to 53.3 % and sigma=1.00
collapses to 23.3 % (vs 68.3 % at lambda=5).

Contrast with the lambda=5 sigma-tail (slate_d D3 + slate_3):

| sigma  | lam=1 rate | lam=5 rate | delta (lam1 - lam5) |
|-------:|------------|------------|--------------------:|
|  0.05  |    65.0 %  |    65.0 %  |         0.0 pp      |
|  0.25  |  **75.0 %**|    63.3 %  |       +11.7 pp      |
|  0.40  |    53.3 %  |    66.7 %  |       -13.4 pp      |
|  1.00  |    23.3 %  |    68.3 %  |       -45.0 pp      |

**The sigma-invariance claim must be quoted carefully.**  At
lambda=5 the cell is sigma-flat across [0.05, 1.00] (~63-68 %).
At lambda=1 the cell has a *peak* at sigma=0.25 = 75 % and
collapses at high sigma.

Mechanistic reading: at lambda=1 the bridge is weak, so MBD's
own exploration over the M=16 noise-perturbed samples carries
more weight.  When sigma is small (0.05), the samples are
clustered near the warm-start and MBD has little room to
improve -- success rate matches the bridge-dominated plateau
(65 %).  When sigma is large (1.00), the samples are scattered
and the weak bridge cannot anchor them; the cell collapses.
sigma=0.25 is the sweet spot: enough noise for MBD to explore,
small enough that the weak bridge can still pull samples back
toward the retrieved demo.

At lambda=5 the bridge is strong enough to anchor samples at
any noise level, so the cell is sigma-flat but caps lower
because strong bridges suppress MBD's productive exploration.

### What this changes for the paper

1. **Headline single-cell number is 73.8 % CI [67.8, 78.9]
   pooled (4 seeds).**  Up from 64.2 % (the May-17 lambda=5
   pool).  Method: `demo_1nn` retrieval + lambda=1 per-step
   score-bridge through the unified driver at K=10 R=1 sigma=0.25
   M=16.  Zero-training, no neural network in the loop.

2. **The unimodal-in-lambda hybrid claim now has its
   confirmation strength**:
   - 1-NN warm-start alone (lambda=0) pool: 11.7 % CI [8.2, 16.3]
   - 1-NN bridge plateau (lambda=5) pool: 64.2 % CI [57.9, 70.0]
   - 1-NN sweet spot (lambda=1) pool: **73.8 % CI [67.8, 78.9]**
   - The single-CI bounds overlap by ~2 pp (peak lower 67.8 vs
     plateau upper 70.0), but CI-overlap is a conservative test.
     The proper two-sample z-test on the two pooled proportions
     gives **diff = +9.6 pp, z = 2.28, p = 0.023**: the peak is
     significantly above the plateau at the 5 % level.  Story B's
     strongest form (peak strictly above bridge-dominated plateau)
     holds at p = 0.023.

3. **Bridge tuning is prior-specific (E2).**  BC peaks at
   lambda=5 = 41.7 %; 1-NN peaks at lambda=1 = 75.0 %.  The
   two priors agree exactly at lambda=0 (warm-start only) and
   diverge as soon as the bridge is on.  This rules out the
   reading "the bridge is doing all the work regardless of
   prior" -- prior quality controls *both* the peak height and
   the peak location.

4. **sigma-invariance must be quoted at lambda=5, not lambda=1
   (E3).**  The "flat across 20x sigma range" claim from May
   17-18 is true *at lambda=5* but does not transfer to the
   new lambda=1 optimum, where the cell is sigma-peaked at 0.25
   and collapses at sigma=1.00 (23.3 %).  This is mechanism
   evidence: at low lambda the cell behaves like a tuned MBD
   sweet-spot in sigma; at high lambda the bridge dominates and
   sigma matters less.

### Open follow-ups (in rough priority order)

1. **Fine-grain lambda around the new peak.**  Current lambda
   resolution {0, 0.5, 1, 2, 5, 10, 20} shows a sharp peak at 1.
   A 5-point sweep at lambda in {0.7, 0.85, 1.15, 1.3, 1.5}
   would resolve whether the peak is at lambda=1 exactly or
   somewhere nearby.  ~2.5 h at single-seed.
2. **4-seed pool at lambda=1 sigma=0.05** (the new sigma-pinned
   companion to E1).  E3 shows sigma=0.05 lambda=1 = 65 %
   single-seed; a 4-seed pool would substantiate sigma=0.05 as
   the safer-but-lower-peak operating point.  ~2 h.
3. **Cross-task: `demo_1nn` lambda-sweep on `square/ph`.**
   Same hybrid claim but on a different contact geometry.
   The lambda-sweep shape (peaked above both endpoints) should
   transfer if the mechanism is real, even if the peak location
   shifts.  Gated on the exp 13 dataset shim.
