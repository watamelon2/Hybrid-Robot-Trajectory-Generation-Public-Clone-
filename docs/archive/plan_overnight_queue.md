# Overnight queue plan (12 h budget)

**Created:** 2026-05-15, before launching `scripts/local/overnight_queue.sh`.

**Goal:** schedule every compute-ready experiment from branches A and C
so the M4 Max stays utilised for the next ~12 hours, then stops cleanly.

**Out of scope:** anything that requires new code (A2 `V_φ`, A4 reward
shaping, B2/B3/B4) — those need engineering work before they can be
queued. The B1.5 `v_theta` magnitude debug is also kept off the queue
because it's interactive inspection, not a batch.

## Pre-queue state (live at 06:20 UTC, 2026-05-15)

Three jobs running:

| Job | PID | Workers | Status |
|---|---|---|---|
| A1.2 main σ-sweep | 83692 | 6 | 3/7 done, σ=0.40 in progress |
| C Phase 1 regression | 83695 | 6 | 1/2 PASS, cell 2 in progress |
| A1.2 σ-sweep extension | 86355 | 2 | 0/3, σ=0.01 in progress |

ETAs: main sweep ~100 min, regression ~50 min, extension ~120 min.
**Queue starts when all three drain** (~2 h from launch).

## Stage 0: wait for the running jobs (~2 h, queue does no compute)

The runner polls all three PIDs at 60 s cadence. Once they all exit,
it advances to the acceptance gate.

**No retries:** if any PID dies before its expected work-product
appears, the runner records the discrepancy in `queue.log` and
proceeds anyway. The gate at Stage 0.5 will catch the only fatal case
(Phase 1 regression failure).

## Stage 0.5: Phase 1 acceptance gate

Reads `experiments/15_unified_hybrid/results/regression_summary.json`.

| Outcome | Action |
|---|---|
| Both cells PASS (within tolerance) | continue to Stage 1 |
| Either cell FAIL | **ABORT** with diagnostic; do not launch Phase 2 |
| Summary file missing | ABORT (treat as failed) |

A failed Phase 1 means the unified driver doesn't reproduce exp 07 +
exp 10 headlines, so its results from Phase 2 wouldn't be trustworthy.

## Stage 1: C Phase 2 K×R grid at λ=5 (~2 h)

The headline planned experiment. Uses the unified driver from
Branch C (Phase 1) on `feat/hybrid-coupling`.

| K | R | wall (est) |
|---|---|---|
| 5  | 0 | 8 min  |
| 5  | 1 | 12 min |
| 5  | 2 | 15 min |
| 10 | 0 | 13 min |
| 10 | 1 | 18 min |
| 10 | 2 | 22 min |
| 20 | 0 | 22 min |
| 20 | 1 | 30 min |
| 20 | 2 | 38 min |

**Total**: ~3 h (revised — K=20 cells dominate). Could trim to
K ∈ {5, 10} = 6 cells = ~90 min, but K=20 is the natural ceiling
we need to know about.

**Command template (per cell):**
```bash
python experiments/15_unified_hybrid/run_can.py \
  --n-demos 20 --n-rollouts 3 --seed 0 \
  --no-live-view --n-workers 12 \
  --prior bc --warm-start \
  --lambda-prior 5.0 \
  --K $K --mpc-replans $R \
  --output-dir experiments/15_unified_hybrid/results \
  --output-name-suffix "_phase2"
```

**Skip logic:** if
`experiments/15_unified_hybrid/results/unified_bc_lam-5.0_K-${K}_R-${R}_phase2.json`
exists, skip the cell.

**Acceptance:** primary readout is which (K, R) cell maximises
success. Secondary: success-vs-wall Pareto frontier.

## Stage 2: 3-seed pool of σ=0.05 (~50 min)

The current sweep gave σ=0.05 → 19/60 = 31.7 % at seed=0. To tighten
this from 95 % Wilson CI [20.6, 45.0] to something useful, run two
more seeds.

| Cell | Wall |
|---|---|
| σ=0.05, seed=1 | ~25 min |
| σ=0.05, seed=2 | ~25 min |

**Total:** ~50 min for 120 additional rollouts. Pooled n=180 gives a
much narrower CI.

**Command template (per seed):**
```bash
python experiments/05_hybrid_method/run_can.py \
  --n-demos 20 --n-rollouts 3 --seed $SEED \
  --no-live-view --n-workers 12 \
  --sigma-max 0.05 \
  --deterministic-bc-reset \
  --output-path experiments/09_sigma_sweep/results/sigma_0.05_seed${SEED}.json
```

**Skip logic:** if `sigma_0.05_seed${SEED}.json` exists, skip.

## Stage 3: λ-sweep on unified driver at K=10, R=2 (~80 min)

Re-runs exp 07's λ-sweep on the unified driver with the new BC and
MPC chunking. Tests whether the prior strength has a different
optimum once MPC is in the loop. Single command does all 6 λ in
one process (the unified driver supports `--lambda-prior` as a sweep
list).

```bash
python experiments/15_unified_hybrid/run_can.py \
  --n-demos 20 --n-rollouts 3 --seed 0 \
  --no-live-view --n-workers 12 \
  --prior bc --warm-start \
  --lambda-prior 0 1 2 5 10 20 \
  --K 10 --mpc-replans 2 \
  --output-dir experiments/15_unified_hybrid/results \
  --output-name-suffix "_lamsweep"
```

**Wall:** 6 cells × ~13 min ≈ 80 min.

**Skip logic:** per-λ — the unified driver checks and skips existing
`unified_bc_lam-${L}_K-10_R-2_lamsweep.json` files.

## Stage 4: 3-seed pool of the best Phase 2 cell (~50 min)

After Stage 1 completes, the runner identifies the (K*, R*) cell with
the highest success rate and queues seeds 1 and 2 on that cell.

```bash
python experiments/15_unified_hybrid/run_can.py \
  --n-demos 20 --n-rollouts 3 --seed $SEED \
  --no-live-view --n-workers 12 \
  --prior bc --warm-start \
  --lambda-prior 5.0 \
  --K $K_STAR --mpc-replans $R_STAR \
  --output-dir experiments/15_unified_hybrid/results \
  --output-name-suffix "_phase2_seed${SEED}"
```

**Wall:** 2 seeds × (8 to 38 min) ≈ 30 to 80 min depending on which
cell wins. Budget mid-estimate ~50 min.

## Stage 5: A1.3 particle ablation at σ=0.05 (~3.5 h)

Tests if MBD is sample-starved at the default M=16. Uses the exp 05
driver (Branch A's natural home) at the best σ.

| M | scale vs M=16 | wall |
|---|---|---|
| 16 (baseline) | 1× | already done as `sigma_0.05.json` from main sweep |
| 64  | 4×  | ~50 min  |
| 256 | 16× | ~3.5 h |
| 1024 | 64× | ~14 h ⚠ |

**Plan:** queue M ∈ {64, 256}, skip M=1024 (over budget). If
M=64 vs M=16 shows < 2 pp delta, also skip M=256 (sample count clearly
doesn't matter). If M=256 - M=16 > 5 pp, queue M=1024 in a future
session.

**Command template (per M):**
```bash
python experiments/05_hybrid_method/run_can.py \
  --n-demos 20 --n-rollouts 3 --seed 0 \
  --no-live-view --n-workers 12 \
  --sigma-max 0.05 --M $M \
  --deterministic-bc-reset \
  --output-path experiments/09_sigma_sweep/results/sigma_0.05_M${M}.json
```

## Stage 6 (slack, ~30 min): failure-taxonomy refresh

After all the above, re-run `experiments/06_failure_taxonomy/run.py`
across every sigma_*.json and unified_*.json so the bucketed failure
table reflects the freshest data.

## Total queue budget

| Stage | Wall (est) | Cumulative |
|---|---|---|
| 0 — wait for current jobs | 2 h | 2 h |
| 1 — Phase 2 K×R grid | 3 h | 5 h |
| 2 — 3-seed σ=0.05 | 50 min | 5 h 50 |
| 3 — λ-sweep on unified | 80 min | 7 h 10 |
| 4 — 3-seed best Phase 2 cell | 50 min | 8 h |
| 5 — A1.3 particle ablation M∈{64,256} | 3.5 h | 11.5 h |
| 6 — failure taxonomy refresh | 5 min | 11.5 h |

**Slack:** ~30 min for stragglers / I/O / log analysis. Fits in 12 h.

## What the queue does NOT cover (still TODO after the night)

- **B1.5 v_theta sample-magnitude debug** — needs human inspection
- **A2 V_φ training** — needs 3 days of new code
- **A4 reward shaping** — needs ~1 day of new code
- **C Phase 3-5** (adaptive λ, truncated K, replanning trigger) — need
  code work
- **B2/B3** (flow prior λ-sweep) — gated on B1.5 fix
- **Exp 13 Square task** — needs new BC checkpoint trained for square
- **M=1024 particle ablation** — too expensive for one night

## Interruption / resume semantics

- The queue script writes one log per stage and one master `queue.log`.
- Each stage writes its result JSONs as it completes. Killing the
  queue mid-stage loses only the in-progress cell, not the whole stage.
- Re-launching the queue script (`bash scripts/local/overnight_queue.sh`)
  picks up where it left off because every cell does a skip-if-exists
  check before launching.

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Phase 1 regression FAIL | low (cell 1 already PASS) | abort at Stage 0.5; investigate manually |
| Unified driver crashes on a specific (K, R) | medium | log + skip; queue continues |
| Memory pressure at M=256 | low (~6 GB extra per process) | workers=12 × ~700 MB each = ~8 GB; total 14 GB free should suffice |
| User wakes up early and wants to stop | n/a | Ctrl-C kills runner cleanly; partial results stay on disk |
| Stage 5 over-runs (~14 h instead of 3.5 h) | low | only M ∈ {64, 256}; even M=256 fits in 3.5 h budget |

## Launch command

```bash
nohup bash scripts/local/overnight_queue.sh \
  > logs/overnight_queue/stdout.log 2>&1 < /dev/null &
```

The runner writes its master log to `logs/overnight_queue/queue.log`
and per-stage logs to `logs/overnight_queue/stage_N_*.log`.
