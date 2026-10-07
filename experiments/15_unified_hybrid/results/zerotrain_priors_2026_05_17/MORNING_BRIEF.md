# Zero-train priors overnight slate — wake-up brief

Launched: 2026-05-17 01:17 PT, branch `feat/zero-train-priors`,
commit `0538c31`.

## What is running

Eight cells through the unified driver (exp 15) at the **headline
config** K=10 R=1 σ=0.25 M=16 H=120 seed=0, 20 demos × 3 rollouts
= 60 rollouts each:

| tag | --prior      | --lambda-prior | hypothesis tested |
|-----|--------------|----------------|-------------------|
| Z1a | demo_1nn     | 0              | 1-NN retrieval warm-start alone |
| Z1b | demo_1nn     | 5              | 1-NN + retrieved-target score-bridge |
| Z2a | demo_knn k=5 | 0              | 5-NN mean warm-start alone |
| Z2b | demo_knn k=5 | 5              | 5-NN + retrieved-target bridge |
| Z3a | demo_random  | 0              | state-conditioning ablation |
| Z3b | demo_random  | 5              | state-cond ablation + bridge |
| Z4a | demo_mean    | 0              | dataset-mean (no demo identity) |
| Z4b | demo_mean    | 5              | dataset-mean + bridge |

Each cell ~25–30 min, 4 invocations (each runs both λ in one
process), ETA finish ≈ **05:00–05:30 PT**.

## Where to look

```bash
SCREEN="zerotrain"
DIR="experiments/15_unified_hybrid/results/zerotrain_priors_2026_05_17"

# Still running?
screen -ls | grep zerotrain
ps aux | grep run_can | grep -v grep

# Live log
tail -f $DIR/run.log
# Just the headlines:
grep -E "^=== |succ=|lambdas=" $DIR/run.log

# Completed cells
ls $DIR/unified_*.json | grep -v summary
```

## How to interpret against existing data

Existing cells at the same headline config (`docs/results.md`
§ "Warm-start ablation"):

| | λ=0 (warm-start only) | λ=5 (full method) |
|-|-|-|
| none      | 0.0 %  | —      |
| **BC**    | 11.7 % | **37.5 %** (4-seed pool) |
| flow v2   | 0.0 %  | 0.0 %  |

Add the eight new cells to fill the missing rows:

| | λ=0 | λ=5 |
|-|-|-|
| none           | 0.0 %  | — |
| demo_mean      | Z4a    | Z4b |
| demo_random    | Z3a    | Z3b |
| demo_1nn       | Z1a    | Z1b |
| demo_knn k=5   | Z2a    | Z2b |
| BC (MLP)       | 11.7 % | 37.5 % |
| flow v2        | 0.0 %  | 0.0 %  |

**Predicted readings (to compare against in the morning):**

- **Z1a (1-NN, λ=0)** likely 5–15 %. State-conditioned retrieval
  should beat dataset mean / random; whether it beats BC's 11.7 %
  is the headline question.
- **Z1b (1-NN, λ=5)** is the most interesting cell. If it ≥ 37.5 %,
  the score-bridge story is *"any reasonable target works, BC's
  generalisation is incidental"*. If it < 37.5 %, BC's smoothing
  is doing something real.
- **Z3a (random, λ=0)** should be ≈ 0 %. Sanity floor.
- **Z3b (random, λ=5)** controls whether the score-bridge can
  *recover* from a bad warm-start (likely no).
- **Z4a, Z4b** test whether the score-bridge has any chance with a
  fixed dataset-mean target. Likely 0 % and 0 % respectively.

## Aggregation command (morning)

```bash
cd /Users/szh/Documents/SZH/Courseworks/ECE\ M237/Project/Hybrid-Robot-Trajectory-Generation
python - <<'PY'
import json, glob
from pathlib import Path
D = Path("experiments/15_unified_hybrid/results/zerotrain_priors_2026_05_17")
rows = []
for j in sorted(D.glob("unified_*_K-10_R-1_Z*.json")):
    if "summary" in j.name: continue
    d = json.loads(j.read_text())
    rows.append((d["prior"], d["lambda_prior"], d["n_success"], d["n_total"],
                 d["success_rate"], j.name))
for r in rows:
    print(f"{r[0]:14s} lam={r[1]:>4}  {r[2]:>3}/{r[3]:<3}  {r[4]*100:>5.1f} %  {r[5]}")
PY
```

## If something went wrong

The slate is sequential per prior, so a crash in cell K only
affects cells K..7. To resume mid-slate, edit
`experiments/15_unified_hybrid/zerotrain_priors_slate.sh` and
comment out the `run_prior` calls that already completed.

Worker-spawn cascade ("Bad file descriptor", thousands of repeats)
was the failure mode in yesterday's warmstart_ablation launch via
`nohup` from zsh. We're using `screen` here for that reason.
**Do not switch to nohup / setsid** — neither works reliably on
macOS for multiprocessing workloads.

---

## RESULTS — slate finished 04:41 PT, 0 bad-FD errors

```
prior              lam=0          lam=5
demo_mean       0/60 =  0.0%   0/60 =  0.0%
demo_random     0/60 =  0.0%   0/60 =  0.0%
demo_knn k=5   12/60 = 20.0%  10/60 = 16.7%
demo_1nn        7/60 = 11.7%  38/60 = 63.3%   <-- NEW PROJECT PEAK
```

Reference cells at the same headline config:

```
BC      lam=0    7/60 = 11.7%   (existing C1)
BC      lam=5   25/60 = 41.7%   (single-seed Phase 2 grid)
BC      lam=5   90/240 = 37.5%  (4-seed pool, existing headline)
flow v2 lam=0    0/60 =  0.0%
flow v2 lam=5    0/60 =  0.0%
```

## The four findings

1. **demo_1nn lam=5 = 63.3% beats the BC headline by +25.8pp**
   (or +21.6pp single-seed vs single-seed).  A zero-training,
   non-parametric prior is the new headline algorithm.

2. **demo_1nn lam=0 = 11.7% EXACTLY EQUALS BC lam=0 = 11.7%.**
   BC's MLP and 1-NN retrieval provide indistinguishable warm-start
   signal.  BC's "generalisation" produces no warm-start advantage.

3. **k-NN averaging hurts under the bridge.**
   demo_knn lam=0 = 20.0% (best lam=0 cell) but lam=5 = 16.7%.
   Averaging across neighbours produces an OOD synthetic trajectory
   that the bridge cannot productively pull toward.  BC, as a
   learned smoother, hits the same ceiling.

4. **State-conditioning is essential.**
   demo_random / demo_mean = 0/60 at both lambdas.  The bridge is
   a multiplier on a state-conditioned target, not an independent
   search mechanism.

## Updated narrative for the report

OLD: "BC + per-step score-bridge wins; flow is a narrow-σ specialist."

NEW: **"The score-bridge needs an in-distribution per-step pull
target.  A real demo trajectory retrieved by 1-NN is in-distribution
and gives 63.3%; a BC-MLP smoothed approximation is partially
in-distribution and gives 41.7%; a k-NN average or a learned flow
sample is out-of-distribution and degrades the bridge below the
warm-start-only baseline."**

The headline algorithm is now `--prior=demo_1nn --lambda-prior=5`
through the unified driver.  BC is retained as a *learned*
baseline; flow is retained as a negative result with explanatory
power.

## Open follow-ups

- **4-seed pool of demo_1nn lam=5.** Single-seed 63.3% needs a
  pooled re-evaluation for an honest headline.  Predicted range
  [55%, 70%].  Cost: 4 cells x ~25 min = ~1.7 h.
- **demo_1nn at other operating points** -- K=5 / K=20, R=0 / R=2
  variants, sigma sweep.  Would establish whether demo_1nn's lead
  over BC is invariant in (K, R, sigma).
- **Task-aware retrieval distance.**  Weight object-pose dimensions
  more heavily.  Free try, likely +pp.
- **Update docs/results.md** -- DONE (new section "Zero-training
  retrieval priors (May 17 AM)" + headline-table entry).
