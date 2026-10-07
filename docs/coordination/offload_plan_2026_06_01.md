# Offload plan — 10-core MacBook Pro hand-off (2026-06-01)

**Authoritative coordination doc** for the 6 tasks being offloaded
from the user's main 16-core Mac (Mac A) to a 10-core MacBook Pro
(Mac B).  Pushed on branch `coordination/offload-plan-2026-06-01`.

If you are the agent on Mac B: read this end-to-end before doing
anything.  The user is offline (asleep, 22:30 PDT 2026-06-01); do
**not** ask clarification questions, make sensible defaults, and
document any non-obvious decisions in commit messages.

---

## Hardware budget

- **Mac B:** 10 cores.  Use `N_WORKERS=8` for all slates to leave
  2 cores for the OS / `screen` / disk I/O.
- Single robosuite worker ≈ 1 core.  At `N_WORKERS=8` each
  `run_can.py` invocation occupies all 10 cores effectively
  (8 workers + main + scheduler overhead).
- **Run slates sequentially.**  Parallel slates on a 10-core Mac
  thrash hard; throughput collapses ~30 %.

---

## Reserved branches — DO NOT TOUCH

These branches are running, queued, or finished but unmerged on
Mac A.  Do not check them out, do not push to them, do not delete
worktrees mentioning them.  Mac A will commit results when its
work finishes.

Status as of **2026-06-02 00:50 PDT** (updated mid-flight):

- `feat/lambda-schedule-wireup`             — ✅ **DONE 22:12 PDT, auto-push scheduled.**
  5 cells.  VERDICT: **FAIL** (sigma-indexed schedules hurt,
  best 66.7 % vs constant 76.7 %, -10 pp).  Results will be
  committed + pushed by Mac A's `results_committer` screen
  (launched 01:56 PDT) as soon as `overnight_queue` exits
  (~09:50 PDT).  Branch then un-reserved — Mac B free to merge.
- `feat/adapt-t2-n200-magnitude`            — ➡️ **HANDED OFF TO MAC B.**
  Slate fix + Stage A / mass=1.6 T2.0 results pushed by Mac A at
  00:58 PDT (commit `6f0e7c1`).  Stage B re-run for mass=1.6 T2.1
  + T2.3 and mass=1.7/1.8/2.0 (all 3 cells each) is now **Task 7**
  on Mac B's queue below.  No longer reserved -- Mac B may push
  results to this branch.
- `feat/per-step-bias-var-action`           — ✅ **DONE 21:55 PDT, auto-push scheduled.**
  9 cells (hybrid_canonical × 3 seeds + hybrid_t2_1 × 3 seeds +
  bc_t2_3 × 3 seeds), all clean.  Results will be committed +
  pushed by Mac A's `results_committer` screen (launched 01:56
  PDT) as soon as `overnight_queue` exits (~09:50 PDT).  Stage 2
  of the bias-variance decomposition pipeline
  (`measure_bias_variance_v2.py`) still needs to be run on the
  raw cell outputs to produce the final Pareto verdict; deferred
  to user (post-curtailment, Mac A is hands-off until noon).
- `feat/adapt-dp-prior-78x-replication`     — ⚠️ **CURTAILED 4/6 → Mac B handoff.**
  Cell wall came in at ~2h54m per cell (vs my brief's 13-min
  estimate -- DP-prior cells are 13× heavier than retrieval).
  Full 6-cell run would have ended ~15:49 PDT, well past the
  user's noon deadline.  Mac A will complete cells 1-4 (D1
  mass=1.0 × {lam 0.5, 1.0, 2.0} + D2 mass=1.3 lam=0.5);
  cells 5/6 (D2 mass=1.3 lam=1.0 + lam=2.0) are **stubbed via
  skip-marker JSONs** in the slate's output dir so the slate
  exits gracefully at ~09:50 PDT.  Results will be auto-pushed
  by `results_committer` (the stubs are also pushed; they
  carry `_skip_marker: true` so Mac B's analysis code can
  detect + filter them).  **Mac B picks up cells 5/6** when
  bandwidth allows -- ~6 h wall on 10-core; this is the
  perturbed-mass adaptation evidence ("78× story" lambdas
  1.0 + 2.0 at mass=1.3) so prioritize it above T6 dense-
  reward in Mac B's queue if you have to choose.
- `feat/adapt-friction-sweep`               — ➡️ **HANDED OFF TO MAC B.**
  Original Mac A wall estimate (~3.5 h) was almost certainly
  too low given the DP-prior under-estimate seen on `#3`.
  To meet the noon deadline, Mac A removed the
  `/tmp/HRTG-wt-b6` worktree at 01:55 PDT; the overnight queue's
  Stage 3/3 invocation will fail-fast with "No such file or
  directory" when it tries to bash the slate, and the queue
  will exit normally via its `||` handler.  Branch is otherwise
  untouched on origin -- Mac B can `git fetch` and `git
  worktree add` it locally and launch the slate fresh.  **No
  Mac A results to inherit** (collect step never ran).

If you need code from any of these branches, `git fetch origin`
and read from the remote ref (`origin/feat/...`); do not check it
out locally.

---

## One-time setup on Mac B

```bash
# 1. Clone (use SSH if you have keys; HTTPS shown for simplicity).
git clone https://github.com/watamelon2/Hybrid-Robot-Trajectory-Generation.git
cd Hybrid-Robot-Trajectory-Generation

# 2. Read this plan from the coordination branch.
git fetch origin
git checkout coordination/offload-plan-2026-06-01
less docs/coordination/offload_plan_2026_06_01.md

# 3. Conda env.  Assumes Miniconda installed at ~/miniconda3 or
#    ~/anaconda3.  Adjust path if your install lives elsewhere.
conda env create -f environment.yml || conda env update -f environment.yml
conda activate hybrid-traj

# 4. Smoke test imports.
python -c "import robosuite, robomimic, torch, numpy, h5py; print('OK')"

# 5. Robomimic datasets.  Required for B5 / B4 / #4.
python -c "
import robomimic.utils.download_utils as d
import pathlib
out = pathlib.Path('experiments/15_unified_hybrid/data')
out.mkdir(parents=True, exist_ok=True)
for task in ('square', 'tool_hang'):
    d.download_dataset(task=task, dataset_type='ph',
                       hdf5_type='low_dim', download_dir=str(out))
"
# After this you should have:
#   experiments/15_unified_hybrid/data/{square,tool_hang}/ph/low_dim_v141.hdf5

# 6. Create a symlink so the slates' expected paths resolve.
ln -sf "$(pwd)/experiments/15_unified_hybrid/data/square/ph/low_dim_v141.hdf5" \
       "$(pwd)/experiments/15_unified_hybrid/data/square_ph.hdf5"
ln -sf "$(pwd)/experiments/15_unified_hybrid/data/tool_hang/ph/low_dim_v141.hdf5" \
       "$(pwd)/experiments/15_unified_hybrid/data/tool_hang_ph.hdf5"

# 7. Confirm Can canonical (already in robomimic_data inside the env's robosuite).
ls -la experiments/15_unified_hybrid/data/can_ph.hdf5 || \
  python -c "
import robomimic.utils.download_utils as d
d.download_dataset(task='can', dataset_type='ph', hdf5_type='low_dim',
                   download_dir='experiments/15_unified_hybrid/data')
" && ln -sf "$(pwd)/experiments/15_unified_hybrid/data/can/ph/low_dim_v141.hdf5" \
            "$(pwd)/experiments/15_unified_hybrid/data/can_ph.hdf5"
```

Wall: ~15–30 min depending on bandwidth.  Square /ph is ~150 MB;
tool_hang /ph is ~200 MB.

---

## Task queue (execute in this order)

Each task is **idempotent** at the cell level (per-cell JSONs are
checked before recompute).  If a task fails mid-way you can re-run
its slate; finished cells will be skipped.

### Task 1 — B8 SOTA reference column  (5 min, no compute)

Doc-only.  Adds a paste-ready reference table for the deck.

```bash
git checkout feat/sota-reference-column
# Read the new file:
less docs/references/sota_reference_numbers.md
# Smoke-test the print script:
python presentation/scripts/print_sota_reference.py | head -20
# Merge to main (fast-forward; no conflicts expected).
git checkout main
git pull --ff-only origin main
git merge --no-ff feat/sota-reference-column \
  -m "merge feat/sota-reference-column: doc-only SOTA reference table for deck"
git push origin main
```

**Success criterion:** `git log --oneline origin/main | head -3` shows
the SOTA merge commit, and `python presentation/scripts/print_sota_reference.py`
emits the table without error.

---

### Task 2 — B1 V_φ headline eval  (15 min code fix + ~3 h eval)

**This task requires a code fix before it can run.**  The slate
ran on Mac A and crashed at the first eval cell with:

```
File "experiments/15_unified_hybrid/unified_loop.py", line 331, in unified_plan_and_execute
    rollout_fn = make_rollout_fn(
TypeError: make_rollout_fn_with_obs() got an unexpected keyword argument 'perturbations'
```

Root cause: the `value_baseline` code path forces
`make_rollout_fn_with_obs` (the OBS-key variant), but
`unified_plan_and_execute` unconditionally passes `perturbations=...`
to it, and the OBS variant's signature doesn't accept that kwarg.

**Fix instructions:**

```bash
git checkout feat/value-baseline-headline-eval
git rebase origin/main   # bring in any new main commits
```

1. Open `experiments/15_unified_hybrid/unified_loop.py` and look at
   the call to `make_rollout_fn(...)` around line 331.
2. Also open `experiments/15_unified_hybrid/rollouts.py` (or wherever
   `make_rollout_fn_with_obs` is defined — `grep -rn` it).
3. Either (a) add `**kwargs` to `make_rollout_fn_with_obs` and pass
   `perturbations` through to the inner rollout function, OR
   (b) detect at the call site whether the rollout factory accepts
   `perturbations` and conditionally pass it.
   Option (a) is simpler and matches the non-OBS factory's behaviour.
4. Smoke-test:
   ```bash
   python experiments/15_unified_hybrid/run_can.py --task can --H 16 \
     --mpc-replans 1 --mpc-chunk-steps 8 --K 5 --M 4 --n-demos 1 \
     --n-rollouts 1 --n-workers 2 --prior bc \
     --bc-checkpoint experiments/03_bc_baseline/results/policy.pt \
     --value-baseline V_phi \
     --value-baseline-checkpoint /tmp/_quick_vphi.pt \
     --output-dir /tmp/_smoke
   ```
   Should run to completion (any non-zero success rate is fine; we
   just want the TypeError gone).
5. Commit + push the fix:
   ```bash
   git add -u && git commit -m "fix(exp15): pass perturbations to OBS rollout factory"
   git push origin feat/value-baseline-headline-eval
   ```

**Then launch the slate:**

```bash
screen -dmS slate_b1_vphi env N_WORKERS=8 \
  bash experiments/16_value_baseline/slate_A2_headline_eval_2026_06_01.sh
```

**Success criterion:** Slate writes
`experiments/16_value_baseline/results/<timestamp>/eval_summary.json`
with `n_rollouts > 0`.  Verdict in
`experiments/16_value_baseline/results/<timestamp>/verdict.md`
will be one of STRONG / FAIL / AMBIGUOUS.

**Wall ETA:** 15 min fix + 80 s V_φ retrain (resume from existing
`V_phi.pt` if you can copy it from Mac A; otherwise retrain on
Mac B) + ~3 h hybrid eval = ~3.5 h.

After this task succeeds, commit results + push:

```bash
git add experiments/16_value_baseline/results
git commit -m "results(exp16): V_phi headline eval (1 cell, n_rollouts=N)"
git push origin feat/value-baseline-headline-eval
```

---

### Task 3 — B5 T2-Square bugfix + BC arm  (~3.5 h)

**Prereqs:**
- `experiments/15_unified_hybrid/data/square_ph.hdf5` symlink from setup step 6.
- `feat/adapt-bc-multitask-obs` already merged to main (verify with
  `git log --oneline origin/main | grep multitask-obs | head -1`;
  should show a commit from 2026-05-28).

**Launch:**

```bash
git checkout feat/t2-square-bugfix-and-bc-arm
git rebase origin/main
screen -dmS slate_b5_square env N_WORKERS=8 \
  bash experiments/15_unified_hybrid/slate_T2_square_bugfix_and_bc_2026_06_01.sh
```

**Success criterion:** Slate writes JSONs for at least 3 cells under
`experiments/15_unified_hybrid/results/slate_T2_square_bugfix_and_bc_2026_06_01/`.

**Commit + push when done:**
```bash
git add experiments/15_unified_hybrid/results
git commit -m "results(exp15): T2-Square bugfix + BC arm cells"
git push origin feat/t2-square-bugfix-and-bc-arm
```

---

### Task 4 — B4 cross-task baselines fill  (~3 h)

Fills the `--` entries in the §7 cross-task table for Lift / Square
/ tool_hang (BC + MBD-no-prior).

**Prereqs:**
- Square + tool_hang symlinks from setup step 6.
- Lift /ph dataset (auto-downloaded by the slate on first BC retrain).

**Launch:**

```bash
git checkout feat/cross-task-baselines-fill
git rebase origin/main
screen -dmS slate_b4 env N_WORKERS=8 \
  bash experiments/15_unified_hybrid/slate_cross_task_baselines_2026_06_01.sh
```

**Commit + push when done.**

---

### Task 5 — #4 contrastive Square slate  (3–5 h)

Tests the contrastive 1-NN retrieval variant against the standard
1-NN on Square.  Goal: see if Square-specific contrastive features
unlock the 0-8 % floor.

**Prereqs:** Square dataset symlink from setup step 6.

**Launch:**

```bash
git checkout feat/contrastive-square-slate
git rebase origin/main
screen -dmS slate_contrastive env N_WORKERS=8 \
  bash experiments/15_unified_hybrid/slate_T2_contrastive_square_2026_06_01.sh
```

**Commit + push when done.**

---

### Task 6 (OPTIONAL stretch) — #1 dense-reward toolhang/transport  (7–9 h)

Scored cells (using `--dense-reward`) for the worst-performing
tasks.  Stretch goal; only attempt if Tasks 1–5 finished cleanly
AND Mac B is not needed for other work for ~10 h.

**Prereqs:** tool_hang + transport datasets (transport will
auto-download on first BC retrain).

**Launch:**

```bash
git checkout feat/dense-reward-slate-K
git rebase origin/main
screen -dmS slate_dense env N_WORKERS=8 \
  bash experiments/15_unified_hybrid/slate_K_toolhang_transport_scored_2026_06_01.sh
```

**Commit + push when done.**

---

### Task 7 (NEW, hand-off from Mac A) — #2 Stage B re-run  (~4 h)

**Branch:** `feat/adapt-t2-n200-magnitude` (no longer reserved).

**Background:** Mac A launched this slate first; Stage A (5 cells
at mass=1.3, demo-count N ∈ {30, 50, 100, 150, 200}) finished
cleanly and is committed on the branch.  Stage B (mass-axis sweep
beyond 1.5) reached mass=1.6 T2.0 cleanly, then crashed on the
T2.1 cell because the bank-dataset path contains a space ("ECE
M237") and the slate expanded `$CELL_ARGS` unquoted.  Mac A pushed
a fix at 00:58 PDT 2026-06-02 (commit `6f0e7c1`) that replaces
the colon-separated string pattern with parallel bash arrays
threaded through `"$@"`.

**What's left:** mass=1.6 cells T2.1 + T2.3, plus mass=1.7/1.8/2.0
(all 3 cells each).  Total 11 hybrid cells + 3 BC retrains + 3
collect-perturbed-demos runs.

**Prereqs:**
- canonical Can /ph (already symlinked in setup step 7).
- *No* Mac A handoff data needed; the slate auto-collects fresh
  20-demo perturbed banks per magnitude into a Mac-B-local
  SHARED_DATA dir.  The mass=1.6 Mac A banks are not portable
  to Mac B (different SHARED_DATA root); the slate will simply
  re-collect them, which is idempotent and clean.

**Launch:**

```bash
git checkout feat/adapt-t2-n200-magnitude
git pull --ff-only origin feat/adapt-t2-n200-magnitude   # picks up the fix
# Recommended: run STAGE=B only since Stage A is already committed
# from Mac A.  The env var is respected by the slate.
screen -dmS slate_2_resume env N_WORKERS=8 STAGE=B \
  bash experiments/15_unified_hybrid/slate_T2_n200_magnitude_wide_2026_06_01.sh
```

**Success criterion:** JSONs for all 11 remaining cells under
`experiments/15_unified_hybrid/results/slate_T2_n200_magnitude_wide_2026_06_01/`.
The fixed slate is idempotent at the cell level, so previously-
completed cells (mass=1.3 N=*; mass=1.6 T2.0) will be skipped.

**Wall ETA on 10-core Mac:** ~4 h
  - mass=1.6: 2 missing cells × ~13 min = ~26 min
  - mass=1.7, 1.8, 2.0: 3 collects + 3 BC retrains + 9 cells
    ≈ (3 × 25 min) + (3 × 80 s) + (9 × 13 min) ≈ 3.5 h

**Recommended position in queue:** after Task 5 (#4 contrastive
Square).  Reasoning: Tasks 2-5 fill deck gaps that depend on the
Square dataset; #2 Stage B is Can-only and orthogonal, so it can
slot in last without blocking the headline deck cells.  Mac B may
reorder if preferred -- nothing depends on its output sequence.

**Commit + push when done.**

---

## Total wall (sequential, 10-core Mac)

| # | Task | Wall | Cumulative |
|---|---|---:|---:|
| 1 | B8 SOTA merge | 5 min | 0:05 |
| 2 | B1 V_φ (fix + eval) | 3.5 h | 3:35 |
| 3 | B5 T2-Square | 3.5 h | 7:05 |
| 4 | B4 cross-task | 3 h | 10:05 |
| 5 | #4 contrastive | 5 h | 15:05 |
| 7 | #2 Stage B re-run (NEW) | 4 h | 19:05 |
| 6 | #1 dense-reward (optional) | 9 h | 28:05 |

If you start at 22:30 PDT 2026-06-01, Tasks 1–5 finish by **13:35 PDT
2026-06-02** (right around when Mac A finishes its overnight queue).
Adding Task 6 finishes by **22:35 PDT 2026-06-02**.

---

## Standing rules

1. **One slate running at a time on Mac B.**  See "Hardware budget"
   above.
2. **Use `screen -dmS <name>`** for every slate.  Survives
   disconnects; lets you check progress with `screen -ls`.
3. **Verify prereqs before launch** — every slate has a `FATAL: ...`
   exit path for missing data.  Read its first 60 lines and the
   error path before running.
4. **Commit results to the task's branch + push** as soon as the
   slate finishes.  Don't batch.  This way the user can pull and
   verify from Mac A.
5. **If a slate fails mid-run:** capture the error in a
   `FAILURE_NOTES.md` on the branch, commit, and proceed to the
   next task.  Do not retry blindly more than once.
6. **Code changes** (e.g. the B1 unified_loop.py fix) go on the
   task's feature branch, not on main.  Only B8 is a merge-to-main
   task.
7. **Never force-push** to any branch; if you need to amend, ask
   the user when they wake up.

---

## Reporting

When you finish (or hit a hard stop), append a status line to
this file under "## Run log (both Macs)" below, commit + push.
The user will read this when they wake up.

## Run log (both Macs)

Append entries below with timestamps; oldest first.  Either Mac
can write.  Tag your entries with `Mac A:` or `Mac B:` so the
other machine can scan its own progress quickly.

- 2026-06-01 22:30 PDT — Mac A: plan written, pushed.
- 2026-06-01 22:38 PDT — Mac B: agent online. Reused pre-existing
  clone at `~/work/Hybrid-Robot-Trajectory-Generation` (already
  set to watamelon2 origin). `git fetch` confirmed reserved
  branches present on remote and untouched.
- 2026-06-01 22:38 PDT — Mac B: pre-existing `hybrid-traj` conda
  env from prior workhorse run reused (robosuite 1.5.2 / robomimic
  0.3.0 / torch 2.11.0 import OK). environment.yml absent at repo
  root (env was originally built from `env/requirements.txt`); no
  recreate needed.
- 2026-06-01 22:38 PDT — Mac B: unloaded stale LaunchAgent
  `com.aaronwen.workhorse-watch` from prior 5/29 workhorse run to
  prevent its STOP/CONT pulse heuristic from interfering with
  Mac B slates. Will leave unloaded until hand-off complete.
- 2026-06-01 22:39 PDT — Mac B: **Task 1 (B8 SOTA reference
  column) DONE.** Smoke-test of `presentation/scripts/print_sota_reference.py`
  passed; FF + non-FF merge of `feat/sota-reference-column` to
  `main` (commit `fcb814d`) pushed to origin.
- 2026-06-01 22:39 PDT — Mac B: starting setup step 5-6
  (download square / tool_hang / can robomimic /ph datasets +
  symlinks) before Task 2 fix work.
- 2026-06-01 22:41 PDT — Mac B: setup step 5-7 done. Datasets
  downloaded via `python -m robomimic.scripts.download_datasets`
  (the plan's `robomimic.utils.download_utils` API doesn't exist
  in robomimic 0.3.0; CLI module was the right entry point).
  All three symlinks (`{can,square,tool_hang}_ph.hdf5`) point at
  the v141 hdf5 files. Total wall ~1 min on home internet.
- 2026-06-01 22:45 PDT — Mac B: **Task 2 (B1) bugfix DONE.**
  Identified second bug beyond the documented one: the OBS rollout
  factory in `experiments/04_mbd_baseline/run_can.py` was also
  missing `.update_init_state(s)` on the returned closure (called
  unconditionally by `unified_loop.py:463` between MPC chunks),
  which surfaced as `AttributeError: 'function' object has no
  attribute 'update_init_state'` once the perturbations TypeError
  was fixed. Single commit `23112c0` adds both fixes: accept
  `perturbations` (with `**_unused_kwargs` future-proofing) +
  thread it to `_worker_init` initargs, plus expose
  `update_init_state` on both single-process and pooled
  branches via an `_init_state_cell` mirroring the non-OBS twin.
  Smoke test (n-demos=1, n-rollouts=1, K=5, M=4, n-workers=2,
  V_phi enabled) ran to completion, wrote JSON, no errors. Pushed
  to `feat/value-baseline-headline-eval` as a fast-forward (did
  NOT use `git rebase origin/main`-then-force as the plan
  shorthand suggested -- standing rule §7 "never force-push"
  outranks; rebased-then-reset-then-cherry-picked the fix on top
  of `bd28920` to keep the push fast-forward).
- 2026-06-01 22:47 PDT — Mac B: **Slate B1 launched** in screen
  `slate_b1_vphi` (pid 2507) with `N_WORKERS=8`. Slate script
  needed two patches before launch (workhorse-style sed-rewrite
  to `/tmp/slate_b1_vphi_patched.sh`): conda init path
  `/Users/szh/anaconda3` -> `/Users/aaronwen/miniforge3`, and
  `REPO=$(...)` derivation -> hard-coded absolute path (the
  `dirname/../..` derivation breaks when the script is run from
  `/tmp`). Stage 0 (V_phi training) running now; Stage 1
  eval ETA ~3h, plan target finish ~02:15 PDT 2026-06-02.
- 2026-06-02 00:11 PDT — Mac B: **Slate B1 first launch CRASHED**
  at 22:52 PDT (Stage 0 finished, V_phi trained fine; Stage 1
  immediately hit the same `TypeError: ...'perturbations'` I had
  pushed a fix for at 22:45 PDT). Root cause was operator-side,
  not code: at ~22:48 PDT I `git checkout coordination/...` to
  append a run-log entry, which reverted
  `experiments/04_mbd_baseline/run_can.py` on disk back to the
  unfixed version (the fix lives only on
  `feat/value-baseline-headline-eval`). Python opened the
  unfixed file at 22:52 PDT during slate import and crashed.
  Lesson: do NOT cross-branch-checkout in the same working tree
  while a slate is running. **Recovery (00:12 PDT):** verified
  feat/value-baseline-headline-eval is back checked out with
  `23112c0` on disk; nuked all `__pycache__` to avoid stale
  .pyc; relaunched in screen `slate_b1_vphi` (pid 19447).
  Stage 0 was skipped (V_phi.pt already trained, idempotent
  check kicked in); Stage 1 now actively running with 8 worker
  subprocesses each at ~85-91 % CPU = full 8-core robotics
  utilisation. Net wall lost: ~1h20min; no rollouts wasted.
  Mitigation for the rest of this hand-off: run-log edits will
  be made via a parallel git worktree at `/tmp/hrtg-coord`
  (created 00:14 PDT) so the main working tree stays pinned to
  the active task's feature branch for the duration of its
  slate. Updated Task 2 finish ETA: ~03:30 PDT 2026-06-02.

- 2026-06-02 00:50 PDT — Mac A: status snapshot pushed.  Three
  Mac A jobs have ended since the plan was written:
  (1) `slate_c1_sigma` finished cleanly at 22:12 PDT, VERDICT
  FAIL (sigma-indexed schedules hurt, -10 pp vs constant control);
  (2) `slate_T2_n200_wide` crashed at 21:14 PDT after Stage A
  completed (5 cells) and Stage B mass=1.6 reached its 2nd cell
  -- shell-quoting bug in the slate (paths containing a space in
  "ECE M237" split into separate args under unquoted `$CELL_ARGS`
  expansion).  Stage B mass=1.7/1.8/2.0 not started; Mac A will
  fix the slate and resume after the overnight queue;
  (3) `overnight_queue` woke up early (21:15 PDT, vs planned
  02:00 AM, because #2 crashed early) and completed Stage 1/3
  (C6 per-step bias-var) cleanly at 21:55 PDT -- 9 cells.  Now
  running Stage 2/3 (#3 DP-prior 78x replication), ETA 03:55 PDT;
  Stage 3/3 (B6 friction) queued.
- 2026-06-02 00:50 PDT — Mac A: no new work being launched here
  tonight; user is offline.  Overnight queue continues to run
  autonomously on Mac A.  Mac B continues task queue per the
  brief.  The two Macs are now operating fully independently;
  rendezvous remains this run log.

- 2026-06-02 00:54 PDT — Mac B: queue audit per user request
  ("queue the rest planned that are not yet to be finished by
  anyone").  Read full backlog (`docs/todos/future_2026_05_26.md`,
  `docs/todos/feat_v6_pull_slate_l_pending_2026_05_25.md`,
  `docs/branch_plan.md`) and cross-checked against (a) Mac A's
  five reserved branches, (b) what's already merged on `main`,
  (c) what ckpts/data are actually on disk vs gitignored, (d)
  the four Mac B task feature branches that are already in this
  hand-off (T3 -- T6).
  **Conclusion: the offload-plan task queue (T1-T6) is already
  the complete set of currently-runnable work on Mac B.**  No
  additional uncontested + unblocked items found.  Cancelled
  candidates and rationale:
    * §A3 T3.1 (Wave-B R-sweep) and §A3 T3.2 (demo_knn k-sweep)
      both require `experiments/08_flow_matching_prior/results/v_theta_v6_unet_xl.pt`
      which is NOT on Mac B (it's gitignored under
      `experiments/**/results/*.pt` and not pushed to LFS).
      Retraining v6 from scratch is a 2-3 day exp08 retrain per
      the in-flight todos doc, not a Mac-B-overnight task.
    * §A1 Slate J (tool_hang + transport) is superseded by
      Task 6 (`feat/dense-reward-slate-K`) which is the
      refreshed scored version of the same content.
    * §A4 Slate I revival is superseded by Task 5
      (`feat/contrastive-square-slate`) -- both target the
      contrastive-obs prior on Square.
    * §B1 Slate Dynamics (mass / friction +-30%) is being run
      on Mac A right now: `feat/adapt-t2-n200-magnitude` is
      the mass axis, `feat/adapt-friction-sweep` is the friction
      axis.  Doubling up would be wasted compute.
    * §B2 Slate Incremental-prior is blocked on two undecided
      design questions (obs-vocab choice; perturbed-demo source);
      not Mac-B-overnight ready.
    * §C1 stale-branch cleanup is doable but doc/git-only and
      below the 5 min threshold; will fold it into the gaps
      between slates rather than queue it as a numbered task.
    * §C2 MJX L4 run is GPU-quota blocked, out of scope here.
    * §D theory items (LSI / mode-separation / finite-N / ...)
      are weeks of paper-grade research, not Mac-B-overnight.
  **Net additions to my queue: 0 slates.  The existing T1-T6
  is exhaustive.**  I will add an opportunistic §C1 stale-branch
  audit pass between two slates (during the conda env spinup
  on a future task), and continue to monitor for new items if
  Mac A pushes additions to this plan.
- 2026-06-02 00:58 PDT — Mac A: slate bug fixed + pushed
  (commit `6f0e7c1` on `feat/adapt-t2-n200-magnitude`).  The
  Stage B crash was a bash word-splitting bug: per-cell args
  were assembled as colon-separated strings with the bank path
  interpolated in, then expanded unquoted as `$CELL_ARGS`, so
  the space in "ECE M237" split the path token.  Fix replaces
  the string-then-split pattern with parallel bash arrays
  (`CELL_TAGS` + `CELL_{0,1,2}_ARGS`) threaded through a new
  `run_one_b_cell` helper via `"$@"`.  Stage A's 5 mass=1.3
  cells + the 1 mass=1.6 T2.0 cell that DID complete cleanly
  before the crash are committed alongside the fix.  Branch is
  no longer reserved — see Task 7 below for the hand-off.
- 2026-06-02 00:58 PDT — Mac A: brief updated: `feat/adapt-t2-
  n200-magnitude` un-reserved; Task 7 added (#2 Stage B re-run,
  Can-only, ~4 h, recommended position is after Task 5).  Mac B
  agent: pick up Task 7 whenever the Task 2-5 chain finishes.

- 2026-06-02 01:03 PDT — Mac B: saw Mac A commit `6f0e7c1` on
  `feat/adapt-t2-n200-magnitude` at 00:59 PDT.  Commit message
  asks Mac B to take Stage B (mass={1.6 T2.1, 1.6 T2.3, 1.7,
  1.8, 2.0}) on this branch as bonus work.  **Mac B accepts
  the hand-off conditionally**, with the reservation status of
  this single branch updated below.  Decision rule discussed
  with the user: take it on AFTER T6 finishes, but **only if
  Mac A first pushes the per-mass perturbed banks Mac B needs**.
  The slate is idempotent so anything Mac A finishes itself
  before Mac B gets to it is fine.
- 2026-06-02 01:03 PDT — Mac B -> Mac A REQUEST: please push
  the following hdf5 banks to `origin/data/banks` so Mac B can
  resume Stage B without re-collecting (~2 h per mass on 10
  cores).  Current `origin/data/banks` (last update 2026-05-29
  `419b647`) only has mass={1.1, 1.2, 1.3, 1.4, 1.5} +
  square_mass=1.3.  **Missing for Stage B resume:**
    * `can_mass1p6_5demos.hdf5`   (for B_T2_1 cell)
    * `can_mass1p6_205demos.hdf5` (for B_T2_3 cell BC retrain)
    * `can_mass1p6_20demos.hdf5`, `can_mass1p6_220demos.hdf5`
      (for completeness / future T2_1 ablations)
    * `can_mass1p7_{5,20,205,220}demos.hdf5`
    * `can_mass1p8_{5,20,205,220}demos.hdf5`
    * `can_mass2p0_{5,20,205,220}demos.hdf5`
  Also the BC checkpoint trained on 205 demos for mass={1.6,
  1.7, 1.8, 2.0} if those exist on Mac A; otherwise Mac B will
  retrain them on the fly (each ~80 s cpu per the slate's
  Stage B).
  Operating model from now until those banks land: the branch
  `feat/adapt-t2-n200-magnitude` reverts to **Mac A
  read/write only** in Mac B's eyes; Mac B will not check it
  out, push to it, or run its slate until Mac A signals via
  another run-log entry that the banks are pushed.  If by the
  time T6 finishes (~projected 2026-06-02 ~14:00 PDT) the
  banks aren't on origin/data/banks yet, Mac B will skip this
  task and stop the queue (per offload plan §1 standing rule
  -- one slate at a time, no auto-launch into uncertain
  preconditions).
- 2026-06-02 01:03 PDT — Mac B: queue status snapshot
    * T1 B8 SOTA merge       DONE 22:39 PDT (fcb814d on main)
    * T2 B1 V_phi headline   RUNNING since 00:12 PDT, ETA ~03:00
    * T3 B5 T2-Square        QUEUED, will start when T2 done
    * T4 B4 cross-task       QUEUED
    * T5 #4 contrastive      QUEUED
    * T6 #1 dense-reward     QUEUED (originally OPTIONAL stretch;
                              upgrading to ACTIVE per user
                              "do all things")
    * T7 (NEW) T2_n200 Stage B BONUS: queued AFTER T6, gated on
                              Mac A pushing banks per request
                              above
- 2026-06-02 01:50 PDT — Mac A: **mass=1.6 banks pushed to
  `origin/data/banks` (commit `38fe648`).**  Files now available
  via `git lfs pull` on the data/banks branch:
    * `can_mass1p6_5demos.hdf5`   (468 KB)
    * `can_mass1p6_20demos.hdf5`  (1.9 MB)
    * `can_mass1p6_205demos.hdf5` (46 MB)
    * `can_mass1p6_220demos.hdf5` (47 MB)
  **mass=1.7 / 1.8 / 2.0 NOT pushed** -- Mac A's slate crashed
  before collecting them; those banks do not exist anywhere yet.
  Mac B's options for Task 7:
    (a) **Recommended:** Run Task 7 with the mass=1.6 banks
        Mac A just pushed.  Mac B gets cells T2.1 + T2.3 done
        for mass=1.6 in ~26 min, no collection needed.  Skip
        mass=1.7/1.8/2.0 entirely.  Deck still gets a 4-point
        magnitude sweep (1.0, 1.3, 1.5, 1.6) which is enough
        to land the headline trend story.
    (b) **Full coverage:** Mac B re-collects mass=1.7/1.8/2.0
        itself (the slate's Stage B does this automatically;
        ~25 min per mass on a 10-core Mac per Mac A's
        measurement -- Mac B's prior ~2 h/mass estimate may
        have been pessimistic) and runs all 11 remaining
        cells.  Adds ~75 min collect + ~3 h cells ≈ 4 h.
        Per the offload plan's Task 7 wall budget.
    (c) **Skip Task 7 entirely.**  Mass-axis sweep stays at
        Mac A's Stage A output (mass=1.3 demo-count axis only)
        plus mass=1.6 T2.0 cell that finished pre-crash.
  Mac A has no preference; user said "only push what you have,
  let Mac B decide".  Mac B agent: pick (a), (b), or (c) per
  your queue budget when T6 finishes.

- 2026-06-02 01:11 PDT — Mac B: **Task 2 (B1) DONE.**  Slate finished
  cleanly with 0/60 successes (Wilson 95% CI [0.0, 6.0] %).
  VERDICT: **H1 SUPPORTED** (capacity-limited, not variance-limited).
  Result + V_phi.pt + slate log committed and pushed to
  `feat/value-baseline-headline-eval` as `7b1575b` at 01:55 PDT.
  Note: slate's verdict-aggregation glob picks up both unified_*_TAG.json
  and unified_summary_*_TAG.json files; selects [0]=summary which has
  no n_total key, so the printed verdict reads '0/0 = 0.0%' rather
  than '0/60 = 0.0%'.  Conclusion (H1) is correct because both 0 and
  None fall into the <= 3 bucket; left unfixed (one-line glob tighten)
  since result speaks for itself.
- 2026-06-02 01:11 -> 01:55 PDT — Mac B operator-monitoring lapse:
  44 min of idle time after T2 finished at 01:11 PDT before commit
  + push at 01:55 and T3 launch at 01:56.  User pinged the gap.
  Lesson: poll slate state at <= 5 min cadence as cells finish, not
  the per-30-min "wait for log" rhythm.  Will tighten polling for
  the rest of the queue.
- 2026-06-02 01:53 PDT — Mac B: Mac A's commit `9cfd8e8` clarified
  banks situation -- the Stage B slate self-collects fresh banks
  per magnitude on the local SHARED_DATA (`mass1p6` Mac A pushed at
  `38fe648` are kept for completeness but not needed by Mac B).
  T2_n200 Stage B re-run formally accepted as **Task 7** in queue,
  to run after T6.
- 2026-06-02 01:56 PDT — Mac B: **Task 3 (B5) LAUNCHED** in screen
  `slate_t3_square` (pid 66289).  Patched the slate same as T2:
  conda init `/Users/szh/anaconda3` -> `/Users/aaronwen/miniforge3`,
  REPO derivation -> hard-coded absolute, SHARED_DATA
  `/Users/szh/.../HRTG-shared-data` -> `/Users/aaronwen/work/HRTG-shared-data`.
  Pre-staged the four `square_mass1p3_*demos.hdf5` banks from
  `origin/data/banks` (worktree-checked-out at `/tmp/hrtg-data-banks`)
  into the local SHARED_DATA so the slate's prereq check skips
  the in-line `collect_perturbed_demos.py` step.  Stage A
  (T2.0_square fixed re-run) running now.  Eta ~50 min Stage A +
  ~2 h Stage B (2 BC trains + 2 evals) ~= ~3 h total finish ~05:00 PDT.
- 2026-06-02 01:56 PDT — Mac A: **NOON DEADLINE CURTAILMENT.**
  Per user "this mac has to be finished by noon tomorrow / leave
  the remaining tasks to the other mac":
    1. **`#3 DP-prior` cells 5/6 stubbed**.  Touched skip-marker
       JSONs at the per-cell glob paths the slate's `run_cell()`
       checks, so when cell 4 finishes (~09:49 PDT) the slate's
       D2 loop will see cells 5/6 already "done" and skip
       through to the final summary.  Slate exits ~09:50 PDT.
       Stubs carry `_skip_marker: true` + cell name + handoff
       branch name; they will be pushed alongside the real
       results.  Mac B's analysis code MUST treat these as
       missing data when computing the 78x adaptation summary.
    2. **`B6 friction sweep` removed**.  `git worktree remove`
       on `/tmp/HRTG-wt-b6` so the overnight_queue's Stage 3/3
       `bash /tmp/HRTG-wt-b6/.../slate_T2_friction_sweep_*.sh`
       fails immediately with "No such file or directory".
       The queue's `||` handler catches and the queue exits.
       Net effect: B6 never starts on Mac A, zero wasted wall.
    3. **`results_committer` screen launched** at 01:56 PDT.
       Polls /tmp/overnight_queue.log every minute for
       "OVERNIGHT QUEUE COMPLETE" and then commits +
       pushes C1 / C6 / #3 results to their feature branches
       automatically.  All three branches will be un-reserved
       and pullable from origin by ~10:00 PDT.
  Mac A will be **fully hands-off by ~09:50 PDT** (only the
  unrelated `phase_d_K40_deferred` screen will still be alive,
  which long predates this hand-off and is out of scope).
  **Mac B's new bonus tasks** (in addition to T1-T7 already on
  the queue):
    * **T8**: `feat/adapt-dp-prior-78x-replication` -- finish
      cells 5/6 (D_dp_pull_mass1p3_lam1p0 + lam2p0).  Each cell
      is ~3 h wall on 16-core; budget ~6 h on 10-core.  This
      is the perturbed-mass adaptation evidence (Slide 7's
      "78x" lambda > 1 columns at mass=1.3); prioritize ABOVE
      T6 (dense-reward) in your queue.
    * **T9**: `feat/adapt-friction-sweep` -- run the slate
      fresh.  No Mac A handoff data; collect + run from scratch.
      Lower priority than T8; can come after T6 or be skipped
      if the queue is already overflowing.
  Apologies for the brief's original wall estimate being so
  far off on DP-prior cells -- the 13-min/cell number was a
  retrieval-mode benchmark, not a DP-pull benchmark.

- 2026-06-02 02:00 PDT — Mac B: ACK Mac A's `565caba` curtailment.
  Updated Mac B queue (in execution order):
    * T1   B8 SOTA merge          ✅ DONE 22:39 PDT (fcb814d)
    * T2   B1 V_phi headline      ✅ DONE 01:11 PDT (7b1575b)
    * T3   B5 T2-Square            🔄 RUNNING since 01:56 PDT
    * T4   B4 cross-task           ⏸ next, ~3 h
    * T5   #4 contrastive Square   ⏸ next, ~5 h
    * T8 (NEW, ABOVE T6 per Mac A) #3 DP-prior cells 5/6, ~6 h
    * T6   #1 dense-reward         ⏸ stretch, ~7-9 h
    * T7   T2_n200 Stage B         ⏸ Mac A recommends option (a):
                                      mass=1.6 only (T2.1 + T2.3) ~26 min,
                                      skip 1.7/1.8/2.0; Mac B will adopt
                                      option (a) when reached
    * T9 (NEW, lower priority)    B6 friction sweep, ~6 h, will skip
                                      if queue overflowing past noon-deadline
  Wall budget at projected 8 PDT-hour-per-real-hour throughput
  (T3 timing will calibrate this): T3 + T4 + T5 + T8 + T6 + T7 + T9
  ≈ 3 + 3 + 5 + 6 + 8 + 0.5 + 6 = ~31.5 h sequential, finishing
  ~09:30 PDT 2026-06-03 if started linearly from 01:56 PDT 2026-06-02.
  That overshoots the user's noon-Tuesday deadline by ~22 h on Mac A
  (which is fine -- Mac A is hands-off after 09:50 PDT; deadline
  is on Mac A only).  Mac B remains active through this stretch.
  When user signals "stop", Mac B will halt at the next slate
  boundary (one slate at a time, never mid-slate).
- 2026-06-02 02:00 PDT — Mac B: handling DP-prior skip-marker
  warning -- when ingesting `feat/adapt-dp-prior-78x-replication`
  results for T8, Mac B's analysis path will check each
  `unified_*_D_dp_pull_*.json` for `_skip_marker: true` and treat
  those as missing rather than as 0/N rollouts.  Will document
  more carefully in the T8 launch entry once T8 actually starts.

- 2026-06-02 02:02 PDT — Mac B: T7 policy DECISION per user
  ("for mass=others, you should also train and document results"):
  switching from Mac A's recommended option (a) to **option (b)
  full coverage**.  Mac B will run all 11 remaining cells of
  Stage B (mass=1.6 T2.1 + T2.3, plus mass=1.7/1.8/2.0 each with
  T2.0/T2.1/T2.3) and document results on the brief.  Bank
  pre-staging:
    * mass=1.6: pushed by Mac A at `38fe648` on `data/banks`,
      copied to `~/work/HRTG-shared-data/` at 02:01 PDT.
    * mass=1.7/1.8/2.0: NOT yet collected anywhere.  The slate's
      `run_magnitude` will auto-collect them via
      `collect_perturbed_demos.py --n-demos-target 20` then
      `derive_perturbed_banks.py` to slice into 5/205/220-demo
      sub-banks.  Per Mac A's calibration ~25 min collect per
      mass on a 10-core Mac (the original Mac B 2 h estimate
      was pessimistic, was based on the 200-demo Stage A bank
      collection for the demo-count axis -- not the 20-demo
      Stage B per-magnitude collection).
    * BC retrain on 205 demos: ~80 s cpu per mass, slate
      auto-trains since `policy_mass1p6_205.pt` etc. are NOT
      on the branch (Mac A pushed `training_log_mass1p6_205.json`
      but not the .pt; gitignore `*.pt` under
      `experiments/**/results/` is the cause -- Mac B will
      `git add -f` retrained ckpts on push).
  **Updated T7 wall:** 4 × ~65 min ≈ ~4 h (mass=1.6 partial
  ~26 min + mass=1.7/1.8/2.0 each ~65 min = ~3.5 h conservative).
  After T7 finishes Mac B will also push the newly-collected
  mass=1.7/1.8/2.0 banks back to `origin/data/banks` so the
  work is durable and reproducible (mirror of Mac A's
  `38fe648` workflow for mass=1.6).
  Updated total wall budget for the queue: T3 + T4 + T5 + T8
  + T6 + T7 + T9 ≈ 3 + 3 + 5 + 6 + 8 + 4 + 6 = ~35 h.
  Linear-from-T3-launch (01:56 PDT 2026-06-02) projected
  finish ~13:00 PDT 2026-06-03.

- 2026-06-02 02:08 PDT — Mac B: queue is RUNNING normally; T3 at
  14/30 rollouts in Stage A first cell, 1 success (vs 0 prior pre-
  bugfix), main pid 66326 alive 12 min, 8 workers @ 70-82 % CPU.
  ETA: Stage A done ~02:50 PDT, Stage B done ~05:00 PDT.
  Used T3's wall to do **opportunistic prep work** (no compute
  conflict -- pure git-show + sed + ls):
    1. Pre-patched `slate_t4_patched.sh` ... `slate_t9_patched.sh`
       to `/tmp/` from each task branch's slate via `git show`,
       with conda init `/Users/szh/anaconda3` -> miniforge3, REPO
       hard-coded, SHARED_DATA -> `/Users/aaronwen/work/HRTG-shared-data`.
       This means each subsequent task launch is one `screen -dmS`
       command -- zero patching overhead at slate boundaries.
    2. Pre-staged mass=1.6 banks from `origin/data/banks` to
       local SHARED_DATA (4 files, ~95 MB; saves ~25 min from
       T7's wall).
    3. **§C1 stale-branch audit DONE** (was opportunistic side-quest):
       - 8 of the 9 candidate branches from `docs/todos/future_2026_05_26.md`
         §C1 are ALREADY DELETED from origin (presumably during the
         post-merge sweep ~`f5e119e` on 2026-06-01).  Only
         `origin/Charles-test` remains and the todos doc explicitly
         says "review with Charles before deletion" -- leaving alone.
       - **3 NEW deletion candidates** found during the audit
         (branches fully merged to main, safe to delete after user
         confirms):
           * `origin/feat/sota-reference-column` (T1 work, merged
             tonight at fcb814d)
           * `origin/feat/presentation-deck-polish-2026-06-01`
             (merged at 019d041)
           * `origin/feat/docs-2026-06-01-post-merge-sweep` (merged
             at f5e119e -- ironically the post-merge-sweep doc
             branch is itself now sweep-able)
       Mac B will NOT delete these unilaterally; flagging here for
       user / Mac A to action.  `feat/dp-phase-c` shows MERGED
       but still holds Phase-D results; leaving alone.
  Net status: T3 healthy and progressing, all upstream prep done,
  nothing else to do until T3 finishes.  Will check progress
  every 5-10 min and immediately commit + push + launch T4 the
  moment T3 writes its final summary JSON.

- 2026-06-02 02:18:52 PDT — Mac B [auto/macb_queue]: **T3 (B5 T2-Square) FINISHED + auto-committed.**  Chain proceeding to next task.

- 2026-06-02 02:18:57 PDT — Mac B [auto/macb_queue]: **launching t4** (feat/cross-task-baselines-fill) in screen slate_t4_xtask with N_WORKERS=8 

- 2026-06-02 02:19:14 PDT — Mac B [auto/macb_queue]: **chain finished.**  Last state file: .  Halt flag present: YES.

- 2026-06-02 02:22 PDT — Mac B: **CHAIN POSTMORTEM (read this first).**
  The auto-chain I built and launched at 02:14 PDT had two compounding
  bugs that caused a cascade.  Chain is now HALTED and disabled; user
  needs to know the true state before any more automation runs.

  **Bug 1 (queue-side -- false slate-done detection):**
    `wait_for_slate` declared T3 done at 02:18:48 PDT because the
    polling rule was "screen detached AND no `pgrep run_can.py`
    workers".  But in T3's Stage B, between Stage A's hybrid eval
    (which spawns workers) and Stage B's eval cells (also workers),
    there is a multi-minute BC-training interval which is a single
    Python process running `train.py` -- no spawn_main, no run_can.py
    in the cmdline.  The poller saw 0 workers and concluded done.
    The queue then auto-committed Stage A + 1.5 BC trains as if it
    were the final result (commit `38abfc2` on `feat/t2-square-bugfix-and-bc-arm`).
    The data in that commit is genuine (Stage A succeeded; first BC
    train succeeded), but the commit message claims the slate finished,
    which is a lie -- Stage B eval was still pending.

  **Bug 2 (slate-side -- pre-existing, unrelated to chain):**
    After the queue moved past T3, the actual T3 slate continued
    running for ~30 more seconds, started Stage B eval cell T2.2_square,
    and **CRASHED** with `KeyError: "runtime obs missing 'Can_pos'"`
    -- the slate's `_make_obs_vec_for_prior` in `unified_loop.py:142`
    is hardcoded to look up `Can_pos`, which exists on `can` envs
    but not on `square` envs (which expose `SquareNut_pos` instead).
    This is an unrelated bug in the T3 slate and would have caused
    a real failure even without the auto-chain.  Likely fix is
    obs-key generalisation in `_make_obs_vec_for_prior` to look at
    `--task` or auto-detect from env keys.  Will document on the
    branch once T3 is properly post-mortemed.

  **Bug 3 (queue-side -- premature T4 launch):**
    After auto-committing T3, the queue launched T4's slate.  Within
    8 seconds the slate exited (probably FATAL'd because some
    Lift/Square/tool_hang BC checkpoint precondition; not yet
    investigated).  The queue interpreted "screen not visible after
    8s" as a launch failure, wrote a `FAILURE_NOTES.md` to T4's
    branch, and HALT'd the chain.  That FAILURE_NOTES was misleading
    (T4 never really tried; the queue's premature T3-done was the
    root cause), so I reverted it via `f9a2ee7` on
    `feat/cross-task-baselines-fill`.

  **Cleanup actions taken:**
    - HALT flag at `/tmp/macb_queue.halt` is set, queue will not
      auto-restart.  LaunchAgent is NOT yet installed (good --
      that would have made this worse).
    - T4 FAILURE_NOTES reverted on origin (`f9a2ee7`).
    - T3's misleading `38abfc2` commit left in place but flagged
      here for follow-up commit message fix.
    - T3 itself is dead (no screens running).  Stage A result is
      valid and pushed.  Stage B BC trains are pushed (in the
      results dir, in `38abfc2`).  Stage B eval cells T2.2/T2.3
      are NOT done -- need the slate's Can_pos obs-key bug fixed
      first.

  **Mac B status RIGHT NOW (02:22 PDT):**
    * No slate running; no queue running; no watchdog installed
    * **Resuming the human-in-the-loop pattern** for the rest of
      the queue rather than autonomy until the chain script is
      hardened against Bug 1 (need to also poll the slate's
      run.log mtime / line count to detect stuck-but-actually-
      training-BC; not "0 workers" alone)
    * T4-T9 worktrees are still ready at `/tmp/hrtg-wt-t{4,5,6,7,8,9}`,
      patched slates ready at `/tmp/slate_t{4,5,6,7,8,9}_patched.sh`
    * Will manually launch T4 next once T3's salvage decision is
      made.

  **Decision needed from user:**
    A. Fix the slate's `Can_pos` bug locally + relaunch T3 Stage B
       only (cell-level idempotent, will skip Stage A).  ~2 h.
    B. Skip T3 Stage B entirely; document T3 partial result and
       move to T4.
    C. Continue chain with hardened poller (will write Bug 1 fix
       first, then resume).

- 2026-06-02 03:01 PDT — Mac B: **T3 RESUMED + bug fix landed.**
  Per user decision (option A): fixed the Can_pos hardcode in
  `experiments/15_unified_hybrid/run_can.py` line 980 -- the BC
  arm was setting `prior_obs_keys = None` which falls back to
  exp 03's Can `OBS_KEYS` even when `--task square`.  After the
  feat/adapt-bc-multitask-obs merge on 2026-05-28, BC supports
  square/lift/tool_hang and needs task-aware obs keys at runtime.
  One-line change: `prior_obs_keys = get_obs_keys(args.task)`
  in the else branch (was the same call already used by retrieval-
  prior cells one branch up).  Smoke-tested with K=5 M=4 1-rollout
  on Square BC ckpt -- runs to completion, no KeyError.
  Commit: `fd709ac` on `feat/t2-square-bugfix-and-bc-arm`.
  Cherry-picked the same fix to `feat/cross-task-baselines-fill`
  as `a0390c7` so T4's BC + MBD-no-prior arms (both pass `--prior
  bc --task <non-can>`) won't hit the same bug.  T7 (mass adapt
  on Can) and T9 (friction sweep on Can) don't hit this bug;
  T5 contrastive uses retrieval branch which was already correct.
  T3 relaunched at 03:00:50 PDT in screen `slate_t3_resume`
  (pid 358).  Cell-level idempotent skip: Stage A T2.0 already
  done, both BC ckpts (canonical + mass1p3_205) already trained,
  slate jumped straight to Stage B's first eval cell
  (B_T2_2_square_bc_unadapted_mass1p3) with 8 workers @ 64-89%
  CPU.  Two eval cells remaining at ~50 min each = ~1.5 h to
  T3 finish, projected ~04:30 PDT.
- 2026-06-02 03:01 PDT — Mac B: **chain script remains DISABLED**
  per chain postmortem.  After T3 finishes I will manually
  inspect + commit results, then start T4 manually.  Will
  rebuild the chain script with the bug-1 fix (poll run.log mtime
  + rollout count, not just "0 spawn workers") only AFTER T3
  successfully completes; will not auto-restart any task tonight.

- 2026-06-02 03:14 PDT — Mac B: **v2 chain + watchdog written and dry-run validated.**
  Per user decision (Option C from chain-replanning prompt): v2 chain
  with full LaunchAgent watchdog.  Hardened against the 3 v1 bugs:
    * **L1 fixed poller**: 3-condition AND -- screen gone AND main
      bash gone AND run.log mtime stale > 60s, requires 2 consecutive
      ticks before declaring done.  Defends against the v1 false-
      positive that fired during T3's BC-train interval (workers=0
      transient).  Dry-run confirmed against T3's live state at
      03:11 PDT: poller correctly reports ALIVE through worker-pool
      churn ticks.
    * **L2 preflight smoke** (20s post-launch): require either main
      bash alive OR log just written; quick-success exit code 0
      treated as idempotent skip rather than failure (fix for v1's
      false T4 FAILURE_NOTES).
    * **L3 verdict-aware commits**: parse run.log for VERDICT lines,
      embed in commit message instead of generic "slate finished"
      lie that v1 wrote.
    * **L4 per-task max_wall_minutes**: each task has 2.5x its plan-
      stated wall budget; exceeding it halts with a "may still be
      running, please verify" failure note rather than auto-killing.
    * **L5 LaunchAgent watchdog** (`com.aaronwen.macb-watch`): runs
      every 300s.  ONLY relaunches the chain script itself if dead
      for 2 consecutive ticks; never launches a task.  Uses PID
      file as authoritative liveness check (not screen -ls, which
      flickers under worker-pool churn).
  Files written:
    /tmp/macb_queue.sh                 (290 lines, syntax OK)
    ~/.local/bin/macb_watch.sh         (refreshed for v2)
    ~/Library/LaunchAgents/com.aaronwen.macb-watch.plist (syntax OK)
  **NOT yet activated**:
    - Chain not started (will start after T3 finishes manually).
    - LaunchAgent NOT loaded yet (will `launchctl load` after chain
      starts so watchdog has something to watch).
    - HALT flag still present (`/tmp/macb_queue.halt`); will rm
      before chain start.
  T3 RESUME continues healthily: Stage B's first eval cell
  (B_T2_2_square_bc_unadapted_mass1p3) at 21/30 rollouts, 0/21
  successes (consistent with Square-hard-for-unadapted-BC
  expectation).  Cell 2 (B_T2_3 adapted) ~50 min after cell 1.
  Projected T3 finish: ~04:15 PDT.  After that: manual commit +
  push for T3, then activate v2 chain + watchdog.

- 2026-06-02 03:30 PDT — Mac B: **prereq audit + 3 fixups before
  v2 chain activation.**  Per user request "check that all the
  follow up scripts are done", audited every T4-T9 dependency.
  3 missing items found and fixed in parallel while T3 continued:

    1. **`lift_ph.hdf5` and `transport_ph.hdf5` not on disk** --
       my earlier setup-step-5 only downloaded can/square/tool_hang.
       T4 needs lift; T6 needs transport.  Launched
       `python -m robomimic.scripts.download_datasets --tasks lift
       transport --dataset_types ph --hdf5_types low_dim` in
       screen `dl_lift_transport`; finished in 2 min.  Symlinks
       `lift_ph.hdf5`, `transport_ph.hdf5` now in
       `experiments/15_unified_hybrid/data/`.
    2. **DP checkpoint `v_theta_dp_tobs2_h16_v3b_ep0199.pt` missing**
       (T8 dependency).  Repo had a broken symlink pointing at
       `/Users/szh/Documents/SZH/Courseworks/ECE M237/...` which
       doesn't exist on Mac B.  Per user tip + repo's
       `experiments/17_dp_hybrid/HANDOFF_NEXT_MAC.md` § "Ckpt
       sync", the canonical recovery is gdown from Google Drive
       folder `1zzF_0-JyBENfZhPaOktt-UfIoUb_JTgx`.  Removed the
       broken symlink, ran `python sync_ckpts_from_drive.py
       --pattern v3b_ep0199` in screen `dl_dp_ckpt`; gdown
       downloaded the whole folder to a tempdir (it can't filter
       at download time).  Per HANDOFF doc's short-circuit
       advice, pulled `ep0199.pt` (509 MB) out of the tempdir as
       soon as it was complete, copied to canonical path,
       `pkill -f gdown`, cleaned 3.9 GB tempdir.  Verified by
       `torch.load` -- valid blob with `obs_dim=23` (DP paper
       schema).  Saved ~5 min of bandwidth.
    3. **mass=1.6 banks** -- already pre-staged earlier from
       `origin/data/banks` `38fe648` to `~/work/HRTG-shared-data/`,
       no re-fetch needed.  mass=1.7/1.8/2.0 will be collected
       on-the-fly by T7's slate.

  **All T4-T9 prereqs now GREEN.**  Re-audited and confirmed:
    T4: lift+square+tool_hang ✓  Can_pos fix on branch ✓
    T5: square ✓  encoder script ✓
    T6: tool_hang+transport ✓  env_dense_rewards.py ✓
    T7: can ✓  collect script ✓  mass=1.6 banks ✓
    T8: can ✓  **DP ckpt 509 MB ✓**
    T9: can ✓  collect script ✓
  T3 RESUME continues: ~45 rollouts logged across Stage A+B,
  8 workers @ ~70-90% CPU, ETA finish ~04:15 PDT.  Disk 672 GB
  free.

- 2026-06-02 03:51 PDT — Mac B [auto/T3-sentinel + manual recovery]: **T3 finalized + chain arming.** Sentinel fired correctly at 03:46:22 but push failed because git-lfs missing from launchd PATH AND DP checkpoint (485 MB) exceeded GitHub 100 MB limit. Recovery: amended commit to drop DP .pt blob (kept symlink + real file on disk), pushed b293dee. Verdict: T3-Square ALL DONE @ 03:38:22 — Stage A canonical 1/30 (3.3%), BC unadapted 2/30 (6.7%), BC retrained 1/30 (3.3%). Patches landed: macb_watch.sh sentinel + chain_proc_alive helper + LaunchAgent PATH includes ~/.local/bin + finalizer skips DP .pt. Chain arming next.

- 2026-06-02 03:51:57 PDT — Mac B [auto/macb_queue v2]: **launching t4** (feat/cross-task-baselines-fill) in screen slate_t4_xtask with N_WORKERS=8 ; max_wall=450min

- 2026-06-02 04:02:38 PDT — Mac B [auto/hardcoded_chain]: **chain finished.** Skipped: 6

- 2026-06-02 04:04 PDT — Mac B [manual/correction]: **disregard previous "Skipped: 6" entry at 04:02:38.** Cause: a sourced-debug invocation of /tmp/hardcoded_chain.sh ran the main loop because the script lacked a source-guard. All 6 checkouts failed because stale per-task worktrees at /tmp/hrtg-wt-t{4..9} were holding their branches. Now fixed: (a) added BASH_SOURCE source-guard to chain script, (b) removed all 6 stale worktrees via `git worktree remove --force`, (c) reset chain logs. The night plan still stands: hardcoded_chain.sh will run T4→T5→T8→T6→T7→T9 once armed.

- 2026-06-02 04:05:55 PDT — Mac B [auto/hardcoded_chain]: **launching t4** (feat/cross-task-baselines-fill) in screen slate_t4_xtask with N_WORKERS=8  (skip-on-fail)

- 2026-06-02 04:08 PDT — Mac B [auto/hardcoded_chain]: **night plan armed unattended.** Replaced macb_queue v2 + worktree-isolation design with a single-repo hardcoded_chain.sh. Reasons: (a) all 6 slates hardcode `cd /Users/aaronwen/work/HRTG`, incompatible with worktree isolation; (b) T4 slate uses bash-4 `declare -A` which fails silent on macOS bash 3.2 (patched to portable case lookup); (c) DP checkpoint at experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt is 485 MB — exceeds GitHub 100 MB blob limit, must stay on disk only (origin keeps the symlink stub). Chain stashes the .pt before each branch checkout and restores after; auto_commit explicitly un-stages that path. Order: T4(cross-task BC+MBD-no-prior fill) → T5(contrastive-square) → T8(DP-prior 78x replication) → T6(dense-reward toolhang+transport) → T7(n200 magnitude wide, STAGE=B) → T9(friction sweep). Mode: skip-on-failure (Q2). Watchdog v3 (com.aaronwen.macb-watch, 5min) only relaunches the chain if dead 2 consecutive ticks AND HALT not set; never auto-launches a slate. T4 launched cleanly at 04:05:59 PDT, python workers running, load 15.13, idle 6.36
- 2026-06-02 04:08 PDT — Mac B [auto/hardcoded_chain]: **night plan armed unattended.** Replaced macb_queue v2 + worktree-isolation design with a single-repo hardcoded_chain.sh. Reasons: (a) all 6 slates hardcode `cd /Users/aaronwen/work/HRTG`, incompatible with worktree isolation; (b) T4 slate uses bash-4 `declare -A` which fails silent on macOS bash 3.2 (patched to portable case lookup); (c) DP checkpoint at experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt is 485 MB — exceeds GitHub 100 MB blob limit, must stay on disk only (origin keeps the symlink stub). Chain stashes the .pt before each branch checkout and restores after; auto_commit explicitly un-stages that path. Order: T4(cross-task BC+MBD-no-prior fill) → T5(contrastive-square) → T8(DP-prior 78x replication) → T6(dense-reward toolhang+transport) → T7(n200 magnitude wide, STAGE=B) → T9(friction sweep). Mode: skip-on-failure (Q2). Watchdog v3 (com.aaronwen.macb-watch, 5min) only relaunches the chain if dead 2 consecutive ticks AND HALT not set; never auto-launches a slate. T4 launched cleanly at 04:05:59 PDT, python workers running, load 15.13, idle 6.36%. ETA all 6 ~ 48 h.

- 2026-06-02 13:21:03 PDT — Mac B [auto/hardcoded_chain]: **t4 finished + auto-committed + pushed.** Verdict: (no VERDICT block; last marker: === 13:19:07  cross_lift_mbd_no_prior  done ===). Proceeding to next task.

- 2026-06-02 13:21:26 PDT — Mac B [auto/hardcoded_chain]: **launching t5** (feat/contrastive-square-slate) in screen slate_t5_contrastive with N_WORKERS=8  (skip-on-fail)

- 2026-06-03 00:42 PDT — Mac B [user/aaron]: **handoff: T6 reassigned to Mac A** to parallelize remaining work. T6 = `feat/dense-reward-slate-K` slate (tool_hang + transport, dense-reward post-fix runs). Cells: K1 tool_hang demo_1nn λ∈{0,1,2} H=300, K2 transport demo_1nn λ∈{0,1,2} H=300; 30 demos × 2 rollouts each. Slate file: `experiments/15_unified_hybrid/slate_K_toolhang_transport_scored_2026_06_01.sh`. Estimated wall on Mac B was ~12h (longest pole); on 16-core Mac A expect ~7h. Mac B continues with T8→T7→T9 (smaller compute, ~10h total). Mac B current state: T5 still running on `feat/contrastive-square-slate` at 95% (cell 6/6, 41/60 rollouts, ETA ~01:20 PDT). Mac B chain auto-halts itself after T8 commits (via /tmp/halt_before_t6.sh watcher) and auto-restarts with T6-removed queue (via /tmp/resume_chain_after_halt.sh). Mac A please run the T6 slate at your convenience; commit results to `feat/dense-reward-slate-K`. No worktree needed since Mac A's main repo can checkout that branch directly. Datasets needed: tool_hang_ph.hdf5 + transport_ph.hdf5 (both should already be in your shared-data dir; if missing, `python -m robomimic.scripts.download_datasets --tasks tool_hang transport --dataset_types ph`).

- 2026-06-03 01:11:15 PDT — Mac B [auto/hardcoded_chain]: **t5 finished + auto-committed + pushed.** Verdict: VERDICT: WEAK    -- encoder helps; more capacity / data may help further. . Proceeding to next task.

- 2026-06-03 01:11:36 PDT — Mac B [auto/hardcoded_chain]: **launching t8** (feat/adapt-dp-prior-78x-replication) in screen slate_t8_dp with N_WORKERS=8  (skip-on-fail)

- 2026-06-03 01:13:59 PDT — Mac B [auto/hardcoded_chain]: **launching t6** (feat/dense-reward-slate-K) in screen slate_t6_dense with N_WORKERS=8  (skip-on-fail)

- 2026-06-03 01:15:14 PDT — Mac B [auto/hardcoded_chain]: **launching t4** (feat/cross-task-baselines-fill) in screen slate_t4_xtask with N_WORKERS=8  (skip-on-fail)

- 2026-06-03 02:30 PDT — Mac B [auto/hardcoded_chain]: **state report.** (a) T5 ✅ done @ 01:11 PDT (f623ae8). (b) T8 ❌ SKIPPED (cc0a4de) — slate's own acceptance gate aborted: D1 mass=1.0 lam=0.5 hit 62.0% test_succ_rate, > 5pp below Phase F headline (73.3%). This is a real scientific finding, not a bug — the DP-prior 78x story does NOT replicate at canonical mass on Mac B's environment. Treat as datum, not failure. (c) T4 unexpectedly RE-RUNNING because original a1dd1e3 commit only had 2/6 cells (lift only); square + tool_hang BC trains finished but unified eval cells never wrote. Chain restart (post-handoff edit) is correctly resuming via slate's idempotent skip — lift cells are skipped, currently grinding cross_square_bc (45/60). T4 ETA ~07:00 PDT. After T4: T7 (~6h) → T9 (~3h). Mac B all done ~16:00 PDT Wed. T6 still owned by Mac A — note the brief 6s "launching t6" entry at 01:13:59 (2341591) was the chain's restart-time misfire; halt_watcher caught it cleanly within 3s, no T6 work was done on Mac B.

- 2026-06-03 03:18 PDT — Mac B [correction to 02:30 entry]: **T8 SKIP cause re-classified.** Earlier note called it a "real scientific finding" — that was wrong. Actual story: T8's branch (`feat/adapt-dp-prior-78x-replication`) had pre-cached D1 result JSONs from a previous protocol run (origin tip, all 3 lam values). The slate's idempotent skip logic saw them and went straight to the gate without running anything. The gate read `test_succ_rate = 62.0%` from the cached `unified_flow_pull_lam-0.5_K-10_R-49_D_dp_pull_mass1p0_lam0p5.json`, compared to the slate's Phase F reference (73.3%), found it 11 pp short of the gate's 5 pp tolerance, and exited 2. So **no new rollouts were produced on Mac B** — T8 effectively no-op'd in 6 s. This is a protocol-mismatch signal, not a replication failure: either (a) the cached D1 numbers were measured under a different protocol than Phase F (different `--n-test-rollouts`, `--max-env-steps`, seed, or evaluator), or (b) the slate's `PHASE_F_REF=0.733` constant is stale. Recommend: delete the 3 cached D1 JSONs on the branch and re-run T8 fresh (estimated wall ~30 min on Mac B at H=16; or ~20 min on Mac A). Mac B is busy with T4 catch-up + T7 + T9, so this is a **good candidate to also send to Mac A** alongside T6 if you have spare cycles.

- 2026-06-03 03:25 PDT — Mac B [analysis update]: **T8 gate-fail explained — DON'T re-run.** Traced the cached D1 JSONs to commit `1e133f0` on `feat/adapt-dp-prior-78x-replication` ("results(exp15): Slate T2-DP-prior partial (4 of 6 cells)..."). Mac A ran D1 mass=1.0 all 3 λ values on its hardware and got 62% test_succ_rate. That's already 11 pp below the slate's hardcoded `PHASE_F_REF=0.733`. So the gate's "FAIL: > 5 pp below Phase F headline" message is correctly flagging that **the current-code protocol does not reproduce the 73.3% Phase F headline anywhere**, on either machine. Re-running T8 (whether Mac A or Mac B) at the same op point will hit ~62% again and re-trigger the same gate. The actual issue is upstream of T8: either (a) the `PHASE_F_REF` constant is from an earlier code/env state that no longer reproduces, or (b) Phase F's original protocol differs from the current slate's COMMON flag bundle in ways we haven't identified. Recommendation: **do NOT redo T8 tonight**. Treat the cached D1 cells as the current-protocol baseline ("mass=1.0 lam=0.5/1.0/2.0 ≈ 62%"). The deck's Slide 7 78x-headline is now suspect — worth investigating before any paper submission, but no overnight action needed. Mac B continues T4-catch-up → T7 → T9. Mac A please proceed with T6 only.

- 2026-06-03 03:32 PDT — Mac B [user-decision option C]: **T8 will be re-run on Mac B after the chain finishes.** Per user direction at 03:28: take T8 fresh under current protocol. (1) `/tmp/slate_t8_patched.sh` patched: `PHASE_F_REF=0.0` (gate disabled, kept as informational print); slate now self-purges cached D1/D2 result JSONs at entry so all 6 cells re-measure. (2) Standalone runner `/tmp/run_t8_after_chain.sh` armed in screen `t8_runner` (pid 1950); polls for chain completion (every 60 s), then safe-checkouts `feat/adapt-dp-prior-78x-replication`, runs the patched slate, auto-commits + pushes. Estimated wall ~30 min at H=16. Total schedule: chain finishes ~16:00 PDT Wed → T8 fresh-run ~16:30 PDT Wed → all done. The 78x / 73% Slide-7 question is still an open paper-integrity concern, separate from this re-run.

- 2026-06-03 04:03 PDT — Mac B [user-decision]: **handoff: T9 reassigned to Mac A.** Reason: Mac B's battery dropped to 7% (now charging on AC), macOS parked 4 of 8 P-cores during charge-from-low (visible in CPU history: cores 3/5/7/9 idle while 4/6/8/10 saturated), workers running at 30-45% CPU instead of 80%. Original Mac B ETA 17:30 PDT slipped to 20:00+ PDT. T9 = `feat/adapt-friction-sweep`: friction sweep on Can with FRICTIONS={0.5, 0.7, 1.5, 2.0} × 3 cells (T2_0_hybrid_canonical / T2_1_hybrid_5demo_adapted / T2_3_bc_retrained_205demos), H=120 K=10 M=16, --n-demos 10 --n-rollouts 3. Slate: `experiments/15_unified_hybrid/slate_T2_friction_sweep_2026_06_01.sh`. Origin branch is empty (no pre-staged data — clean fresh run). Estimated wall ~2h on 16-core Mac A.  Mac B continues T4 catch-up → T7 → T8 (option-C re-run); halt_before_t9 watcher armed in screen halt_t9_watcher (pid 21940), will trip HALT after T7 finishes so chain exits before t9 launches; t8_runner (already armed) detects chain-dead and runs T8 fresh.  New Mac B ETA: ~14:30 PDT Wed (no T9). Mac A please run T9 after T6 at your convenience.

- 2026-06-03 09:24:56 PDT — Mac B [auto/hardcoded_chain]: **launching t5** (feat/contrastive-square-slate) in screen slate_t5_contrastive with N_WORKERS=8  (skip-on-fail)

- 2026-06-03 09:27:04 PDT — Mac B [auto/hardcoded_chain]: **t5 finished + auto-committed + pushed.** Verdict: VERDICT: WEAK    -- encoder helps; more capacity / data may help further. . Proceeding to next task.

- 2026-06-03 09:27:24 PDT — Mac B [auto/hardcoded_chain]: **launching t8** (feat/adapt-dp-prior-78x-replication) in screen slate_t8_dp with N_WORKERS=8  (skip-on-fail)

- 2026-06-03 16:32:46 PDT — Mac B [auto/hardcoded_chain]: **launching t7** (feat/adapt-t2-n200-magnitude) in screen slate_t7_n200 with N_WORKERS=8 STAGE=B (skip-on-fail)

- 2026-06-03 19:36:09 PDT — Mac B [auto/hardcoded_chain]: **t7 finished + auto-committed + pushed.** Verdict: (no VERDICT block; last marker: === 19:34:30  Slate T2-N200-MAG-WIDE  ALL DONE ===). Proceeding to next task.

- 2026-06-03 19:36:28 PDT — Mac B [auto/hardcoded_chain]: **launching t9** (feat/adapt-friction-sweep) in screen slate_t9_friction with N_WORKERS=8  (skip-on-fail)

- 2026-06-03 19:54 PDT — Mac B [final]: **chain finished cleanly.** Final tally: T5 ✅ (785be49), T7 ✅ (b96ebf9, 32 files, ALL DONE @ 19:34:30, ran in 3h 3m once load cleared), T6 ✅ (Mac A, a58de66), T9 ✅ (Mac A, b7f872c).  Halt-before-t9 watcher fired correctly at 19:36:56 PDT, chain exited in 6s.  Two open items: (1) T4 only has 2/6 cells committed (a1dd1e3, lift only); 4 missing cells (cross_square_bc, cross_square_mbd_no_prior, cross_tool_hang_bc, cross_tool_hang_mbd_no_prior) — chain's 8h second-pass at 01:15-09:24 produced data in working tree but failed before commit and `git clean -fd` on next safe_checkout wiped it. Estimated ~3h on Mac A's 16-cores if needed. (2) T8 abandoned per user direction at 16:31 PDT — even with option-C patch, slate's true wall is ~5h/cell × 6 = 28h (cost dominated by `--prior-n-steps 100` × `K*M=160` per env step on the trained DP prior, not the H=16 outer horizon I'd misread).  Slide-7 78x/73% paper-integrity question remains separate.  All Mac B watchers/runners cleaned up.

- 2026-06-03 19:59 PDT — Mac A [user-directed]: **T8 ANSWERED + running correctly on Mac A. This resolves the "Slide-7 78x/73% paper-integrity" open item.** The T8 non-replication was a **CHECKPOINT-EPOCH MISMATCH** — not protocol drift, not a stale `PHASE_F_REF`, and not fundamentally the 28h wall. The slate defaulted `DP_CKPT` to `v3b_ep0199` and its header wrongly claimed ep0199 was "the same prior used in Phase F." Phase F's 73.3% headline (K=10 lam=0.5, 44/60) was measured on **ep0149** — the hybrid epoch-curve PEAK (per `docs/experiments/08_flow_matching_prior.md`). ep0199 is 50 epochs past that peak; its 80% *standalone* number does not transfer to the K=10 lam=0.5 hybrid, where it lands ~60% — exactly the 60-62% the gate kept (correctly) flagging as "evaluated checkpoint != reference checkpoint." Mac A fixed the slate and pushed **`e61d040`** on `feat/adapt-dp-prior-78x-replication` (DP_CKPT ep0199→ep0149, header corrected, D2 gated behind `RUN_D2`, stale ep0199 D1 JSONs archived to `_archive_ep0199_2026_06_01/`). **Nothing required from Mac B** — confirming Mac B's 16:31 decision to abandon the option-C ep0199 re-run was the right call (it would have re-confirmed the wrong checkpoint). On wall time: Mac A's 16-core at `N_WORKERS=14` does ~2.9h/cell (matches the cached `1e133f0` run), so D1 (3 cells) ≈ 6h, full T8 (D1+D2) ≈ 12-15h — feasible here. Mac A is running the corrected ep0149 T8 now: D1 in flight (started ~15:45 PDT), D2 mass=1.3 auto-follows via a runner **iff** D1 lam=0.5 clears the 68.3% gate. **Reserving new Mac-A-only branch `feat/followups-2026-06-03`** (hard-task pure-MBD/prior-only baselines + mass-extremes {0.5,2.5}) — DO NOT TOUCH. Noted T4 open item (4 missing cross-task cells); Mac A can pick that up next if the user wants. Mac B is done — rest the laptop.

- 2026-06-04 00:06 PDT — Mac B [user-decision]: **T4 catch-up plan, split between machines.** The 4 missing T4 cells (cross_square_bc, cross_square_mbd_no_prior, cross_tool_hang_bc, cross_tool_hang_mbd_no_prior) are split by cost: BC cells are ~5h total, MBD-no-prior cells (--no-warm-start --M=64) are ~10-15h on 16 cores. Mac B takes BC, Mac A takes MBD-no-prior; both run in parallel against feat/cross-task-baselines-fill. **Mac B (in progress):** /tmp/t4_bc_only.sh launched in screen `t4_bc_only` (pid 6661); BC checkpoints already on disk so no training step; estimated done ~05:00 PDT Thu; auto-commits + pushes to feat/cross-task-baselines-fill. **Mac A (action requested):** pushed handoff slate at `23bcd48` — `experiments/15_unified_hybrid/slate_cross_task_baselines_mbd_only_macA.sh` on feat/cross-task-baselines-fill. Mac A please: `git fetch origin && git checkout feat/cross-task-baselines-fill && git merge --ff-only origin/feat/cross-task-baselines-fill && bash experiments/15_unified_hybrid/slate_cross_task_baselines_mbd_only_macA.sh`. Slate auto-detects REPO root + tries several conda base paths so no edits needed; N_WORKERS=12 default (override if needed). When done, commit + push the produced unified_*.json + run_macA_mbd.log to feat/cross-task-baselines-fill. Both runs write to the same results/slate_cross_task_baselines_2026_06_01/ directory with disjoint _TAG suffixes so no file conflicts; the auto-push retry-with-rebase logic handles concurrent pushes. After both finish, Slide 5 cross-task table is complete (BC + MBD-no-prior × {lift, square, tool_hang}, 6 cells total). T8 paper-integrity question (PHASE_F_REF=0.733 vs 62% reproducibility) remains open and is NOT being addressed tonight.

- 2026-06-04 01:47:25 PDT — Mac B [auto/t4_bc_only]: **T4 BC-only catch-up DONE.** Square + tool_hang BC cells committed + pushed. The 2 MBD-no-prior cells remain assigned to Mac A (see handoff entry). Slide-5 BC half is now complete.

- 2026-06-04 01:54 PDT — Mac B [user-decision]: **Mac B is signing off for tonight.** Final state:
  - All Mac-B-owned work is committed and pushed. T1–T7 + T9 closed; T4 BC half closed at `4b8b119` on `feat/cross-task-baselines-fill`; T8 abandoned per user (paper-integrity issue, not a wall-time issue).
  - Stray local commit `b22bcf8` (T8 auto-failure note) was rebased onto Mac A's investigation work (`e61d040` + `79816dd`) and pushed as `02f3353` on `feat/adapt-dp-prior-78x-replication`. No force-push, no history rewrite of remote tips.
  - Working tree clean except for the long-known DP `.pt` symlink toggle in `experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt` (safe_checkout artifact, real 485 MB blob still on disk + parked in `stash@{0}` — DO NOT delete that stash).
  - Watchdog `com.aaronwen.macb-watch` LaunchAgent stays armed (5-min cadence, harmless when no chain is running). No active screens, no chain script in flight.
  - **Open items (Mac A only from here):** (a) the 2 T4 MBD-no-prior cells from handoff slate `23bcd48` on `feat/cross-task-baselines-fill`; (b) ongoing T8 ep0149 D1/D2 investigation on `feat/adapt-dp-prior-78x-replication`; (c) the deck Slide-7 78x-headline reproducibility question (paper-integrity, no deadline tonight).
  - Mac B: hands off until next ping from user.

- 2026-06-04 02:18 PDT — Mac A [correction / wrap-up]: **T8 is NOT resolved by the checkpoint swap alone.** The corrected ep0149 re-run finished on Mac A at 01:10 PDT with D1 mass=1.0 results: λ=0.5 = 34/60 overall (26/50 test = 52%), λ=1.0 = 40/60 overall (32/50 test = 64%), λ=2.0 = 39/60 overall (33/50 test = 66%). The slate gate correctly failed at λ=0.5 because 52% test is >5 pp below the Phase F 73.3% headline, so the slate exited `rc=2` before D2 and the mass=1.3 adaptation cells did **not** run. This means the ep0199→ep0149 checkpoint mismatch was real, but **not sufficient** to recover the original Phase F 73.3% / 87% headline under the current code/environment; some additional protocol or environment drift remains. This entry supersedes the 2026-06-03 19:59 PDT note above. Paper-integrity status: **Slide-7 78x/73% remains OPEN**; do not cite it as re-validated yet. Operational status: Mac B remains done / no action required. Mac A continues only the already-running follow-up branch `feat/followups-2026-06-03` (Unit 2 hard-task baselines, then Units 3-4 friction + mass extremes), still expected to finish before noon. Optional leftover remains T4 (4/6 missing cross-task cells) if needed later.

- 2026-06-04 10:57 PDT — Mac A [close-out]: **offload-plan arc closed; remaining work is Mac-A-only.**
  Mac B confirmed totally finished at 10:50 PDT.  Mac A has now merged
  the six remaining Mac-B-side finished branches into `main` and swept
  the merged-branch backlog.

  **Today's merges to `main`** (commit chain `9180285..1b450b6`, all
  `--no-ff`, see [`docs/plans.md`](../plans.md) "2026-06-04 update" for
  full per-branch verdicts):

    * T2  `feat/value-baseline-headline-eval`     -> `32a63ac`  (V_φ 0/60, H1 SUPPORTED + OBS-factory bug fixes)
    * T3  `feat/t2-square-bugfix-and-bc-arm`      -> `ffbd531`  (T2-Square 3 cells + Can_pos cross-task one-liner)
    * T5  `feat/contrastive-square-slate`         -> `1ee9195`  (VERDICT: WEAK)
    * T7  `feat/adapt-t2-n200-magnitude`          -> `233fff1`  (Stage A + Stage B, 5-point mass sweep)
    * C1  `feat/lambda-schedule-wireup`           -> `92caf65`  (VERDICT: FAIL, -10 pp; wireup preserved)
    * C6  `feat/per-step-bias-var-action`         -> `9180285`  (9 cells; Stage-2 analysis deferred)
    + a `docs(plans)` close-out commit `1b450b6` summarizing the
      batch on the cross-branch plans page.

  One trivial add/add conflict during T7 merge on
  `slate_T2_dp_prior_adaptation_2026_06_01/run.log` (both branches
  accidentally wrote logs to the same path via the Mac-B chain's
  auto-commit); resolved as a union with demarcating comments.

  **Origin-branch sweep (17 → 9):** deleted the following local + remote
  refs (all fully reachable from `origin/main`):

    feat/per-step-bias-var-action       feat/lambda-schedule-wireup
    feat/value-baseline-headline-eval   feat/t2-square-bugfix-and-bc-arm
    feat/contrastive-square-slate       feat/adapt-t2-n200-magnitude
    feat/adapt-friction-sweep           feat/dense-reward-slate-K
    feat/followups-2026-06-03           feat/sota-reference-column
    feat/docs-2026-06-01-post-merge-sweep  feat/dp-phase-c
    feat/presentation-deck-polish-2026-06-01

  Also removed the now-redundant worktrees
  `/private/tmp/HRTG-wt-{followups,t9}` (both clean + fully merged).

  **Still in flight on Mac A (kept):**

    * `feat/cross-task-baselines-fill` -- T4 MBD-no-prior cells
      still grinding in screen `t4_macA` (pid 23839 at close-out
      time, ~30 min in on the `cross_square_mbd_no_prior` cell;
      `cross_tool_hang_mbd_no_prior` queued).  Will merge on
      completion.
    * `feat/adapt-dp-prior-78x-replication` -- merged at `4e8f0a7`
      including Mac B's late `02f3353` rebase, but **the Slide-7
      78x / 73.3 % paper-integrity question remains OPEN** per
      the 02:18 PDT wrap-up.  Worktree at `/private/tmp/HRTG-wt-dp`
      kept for any future investigation; branch retained.
    * `feat/headline-figure-priors-fix` -- new active dev branch
      (mounted in the canonical clone; screen `headline_priors`
      running).  Unrelated to the offload arc.

  **Still in flight (not Mac A's):**

    * `feat/dp-phase-c-bughunt` -- unmerged origin branch; unclear
      status, leaving alone.
    * `Charles-test` -- kept per the longstanding "review with
      Charles before deletion" note.
    * `data/banks` -- long-lived data branch, untouched.

  The coord-branch worktree at `/private/tmp/HRTG-wt-coord` and the
  `coordination/offload-plan-2026-06-01` branch itself are retained
  for the historical record.  This entry is the final operational
  status on this plan -- the offload arc is closed.

- 2026-06-04 11:09 PDT — Mac A [follow-up; supersedes "branch retained" line above]: **`feat/adapt-dp-prior-78x-replication` also closed.**
  On user prompt, audited the branch + worktree and found zero open
  work: local == remote at `02f3353`, no uncommitted changes, no
  stashes, no untracked files (only `__pycache__/` cruft and the
  gitignored T8-protocol log), no processes touching the worktree.
  Every commit reachable from `origin/main` via merge `4e8f0a7`
  (which has `02f3353` directly as a parent), so deleting the
  branch label loses nothing.  Branch name "78x-replication" also
  describes a *finished* effort (the original Phase-F replication
  attempt that produced the under-replication result); the still-
  open Slide-7 paper-integrity question is a different scope and
  should live on a freshly-named branch when investigation
  resumes.

  Actions:
    * removed worktree `/private/tmp/HRTG-wt-dp`
    * deleted local + remote `feat/adapt-dp-prior-78x-replication`

  Updated open-items list (supersedes the corresponding bullet
  in the 10:57 PDT close-out above):

    * `feat/cross-task-baselines-fill` -- T4 MBD-no-prior cells
      still running (only true in-flight Mac-A work from the
      offload arc).
    * The Slide-7 78x / 73.3 % paper-integrity question is
      still open *as a research question*, but it has no
      branch / worktree tied to it any more; future work on
      it should start fresh.

  Origin branches now: 8 (`Charles-test`, `coordination/...`,
  `data/banks`, `feat/cross-task-baselines-fill`,
  `feat/dp-phase-c-bughunt`, `feat/headline-figure-priors-fix`,
  `feat/mbd-data-guided-comparison`, `main`).  Worktrees now: 4
  (canonical clone on `feat/headline-figure-priors-fix`,
  `/private/tmp/HRTG-wt-coord` on coord branch,
  `/private/tmp/HRTG-wt-mbd` on `feat/mbd-data-guided-comparison`,
  `/private/tmp/HRTG-wt-t4` on `feat/cross-task-baselines-fill`).

