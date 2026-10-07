# Two-machine parallel launch — incremental adaptation slates

> **Closure note (2026-06-01)**: the two-machine launch completed
> as planned.  All 7 adaptation slates landed and merged into
> `main` (the final one, `feat/adapt-pareto-analysis`, on
> 2026-06-01 as `aaefba2`).  See the "Closure (2026-06-01)"
> subsection of [`05_followup_plan.md`](05_followup_plan.md) for
> the full merge-status table.  The body below is preserved as a
> reference for any future multi-machine slate launches that want
> to reuse the same partition strategy.

Companion to `docs/adaptation/05_followup_plan.md`.  That doc lays
out *which* seven follow-up slates to run; this doc lays out *how to
split them across two Macs*.

The operational asymmetry: the **10-core Mac stays plugged in,
unattended, and runs the bulk** (~22-25 h sequential).  The 16-core
Mac is interactive and only runs **one ~7 h slate plus the post-hoc
analysis**.

## TL;DR partition

| Slate (`feat/...`)         | Mac      | `--n-workers` | Wall   | Why this side |
|----------------------------|----------|---------------|-------:|---------------|
| (1) adapt-t2-multiseed     | 16-core  | 10            | ~7 h   | one fixed-budget chunk on the interactive Mac; pure consumer |
| (7) adapt-pareto-analysis  | 16-core  | (post-hoc)    | ~1 d   | runs after every slate; CPU-light writing task |
| (3) adapt-t2-magnitude     | 10-core  | 8             | ~7-8 h | produces 4 new perturbed-mass banks; workhorse |
| (2) adapt-t2-democount     | 10-core  | 8             | ~4-5 h | consumer of canonical mass=1.3 bank |
| (6) adapt-t2-lambda        | 10-core  | 8             | ~4-5 h | consumer of canonical mass=1.3 bank |
| (4) adapt-t2-square hybrid | 10-core  | 8             | ~5-9 h | builds Square banks locally |
| (5) adapt-bc-multitask-obs | 10-core  | (CPU train)   | ~6 m   | tiny CPU train; first in workhorse queue |

**16-core Mac total:** ~7 h compute + ~1 day post-hoc analysis.
**10-core Mac total:** ~22-25 h sequential at `--n-workers 8`
(~25 % per-cell slowdown vs the 16-core from fewer cores).

The 10-core Mac is the **canonical collector going forward**:
every perturbed bank produced after Stage 2.0 lands there and gets
pushed to LFS for any future consumer.  The 16-core Mac only
*pre-seeds* the Can mass=1.3 bank during scaffolding (Stage 2.0)
so the workhorse can start immediately without an upstream collect.

The Phase-A DP eval continues on Colab (off-machine, no local CPU
load) — both Macs are fully free for adaptation.

## Why this split

1.  **(1) on the 16-core Mac** — the interactive Mac gets exactly
    one ~7 h slate.  Multiseed is a pure consumer (no bank
    production), so it makes no LFS pushes mid-run.
2.  **(3), (2), (6), (4), (5) on the 10-core Mac** — everything
    else queued sequentially on the workhorse.  (3) magnitude
    produces 4 new mass-scale banks; (4) Square produces 3 new
    Square banks; both push to LFS as they finish.
3.  **(5) bcmulti first in the workhorse queue** — it's 6 min of
    CPU train and must merge to `main` before any future (4)-BC
    arm fires.  Running it first lets it land while the rest of
    the queue is in flight.
4.  **(7) analysis on the 16-core Mac** — post-hoc, CPU-light,
    benefits from the interactive Mac's available cycles after
    multiseed wraps.

**CPU rule of thumb:** `--n-workers ≤ physical_cores − 2`.
16-core → 10.  10-core → 8.  Do **not** run two launchers
concurrently on the same Mac; each already saturates
`--n-workers` cores.

---

## Phase 0 — One-time setup

### A0. Git LFS install (both Macs)

```bash
brew install git-lfs
git lfs install          # writes LFS hooks into ~/.gitconfig
git lfs version          # sanity check
```

### B0. 10-core Mac fresh-machine bootstrap

> Run these steps on the 10-core Mac in order.  Skip any step already done.

```bash
# B0.1  Xcode CLT + Homebrew
xcode-select --install      # click through the dialog
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# B0.2  conda (miniforge), git, git-lfs
brew install --cask miniforge
brew install git git-lfs
git lfs install

# B0.3  Clone repo
PROJECT_ROOT="$HOME/Documents/HRTG"          # any path you like
mkdir -p "$PROJECT_ROOT" && cd "$PROJECT_ROOT"
git clone https://github.com/watamelon2/Hybrid-Robot-Trajectory-Generation.git
cd Hybrid-Robot-Trajectory-Generation

# B0.4  Conda env (matches Mac A's `hybrid-traj`)
conda env create -f env/environment.yml      # ~10-15 min
conda activate hybrid-traj

# B0.5  Robomimic canonical PH datasets (Can, Lift, Square)
#       Lands inside the conda env's site-packages, not the repo.
python -m robomimic.scripts.download_datasets \
    --tasks can lift square \
    --dataset_types ph \
    --hdf5_types low_dim

# B0.6  Sanity smoke (must succeed)
python -c "from experiments.common import default_dataset_path; \
           print('CAN dataset:', default_dataset_path('can'))"
python experiments/15_unified_hybrid/run_can.py \
    --task can --prior demo_1nn --lambda-prior 1 \
    --n-demos 2 --n-rollouts 1 --n-workers 2 \
    --K 4 --M 4 --H 60 --mpc-replans 1 --mpc-chunk-steps 30 \
    --output-dir /tmp/macb_smoke
```

Expected B0.6 wall: ~2-3 min.  If it ends with a JSON in
`/tmp/macb_smoke/` whose `success_rate` is a finite number, the env
is ready.

### AB0. HRTG-shared-data layout (both Macs)

Both Macs must have the bank directory at the same absolute path so
the slate scripts' `SHARED_DATA` default resolves identically:

```bash
# 16-core Mac: already exists at:
#   /Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/
#
# 10-core Mac: create the same parent path (or override SHARED_DATA per launch).
mkdir -p "$HOME/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data"
```

If the absolute path differs on the 10-core Mac (e.g., shorter
`~/Documents/HRTG/`), export `SHARED_DATA` before every launch:

```bash
export SHARED_DATA="$HOME/Documents/HRTG/HRTG-shared-data"
```

The slate scripts honour the env var (see the
`SHARED_DATA="${SHARED_DATA:-...}"` line at the top of every
`slate_incremental_*.sh`).

### N0. n-workers parameterization (already committed)

The four workhorse-branch launchers now honour an `N_WORKERS` env
var that defaults to `10`.  Set it to `8` on the 10-core Mac:

```bash
export N_WORKERS=8
```

This applies to slates (2), (3), (4), and (6).  See commits on
each `feat/adapt-t2-*` branch for the parameterization patch.

---

## Phase 1 — Bank-sync convention (Git LFS on `data/banks` branch)

### Why an orphan branch

The perturbed-demo banks (`.hdf5`, ~tens of MB each) shouldn't live
on any feature branch — they'd bloat history.  Instead we keep them
on a dedicated **orphan** branch `data/banks` whose only contents are:

```
.gitattributes      # tells LFS to track *.hdf5
README.md           # one-paragraph note about the branch
*.hdf5              # the banks themselves, LFS-pointered
```

`HRTG-shared-data/` on disk stays *outside* every worktree as today;
we only use a separate ephemeral checkout to push/pull banks.

### A1. 16-core Mac — initialise `data/banks` (one-time)

```bash
cd /tmp
git clone https://github.com/watamelon2/Hybrid-Robot-Trajectory-Generation.git lfs-banks
cd lfs-banks
# Create the orphan branch with no history
git checkout --orphan data/banks
git rm -rf .                          # clear the working tree
cat > .gitattributes <<'EOF'
*.hdf5 filter=lfs diff=lfs merge=lfs -text
EOF
cat > README.md <<'EOF'
# data/banks — perturbed-demo HDF5 banks (LFS)

This orphan branch carries the perturbed-mass demo banks used by the
incremental adaptation slates.  Files here are LFS pointers; the
actual blobs live on GitHub LFS storage.

Do NOT merge this branch into any code branch.  Pull/push from a
dedicated worktree or fresh clone (see
`docs/adaptation/06_two_machine_launch.md`).
EOF
git add .gitattributes README.md
git commit -m "data(banks): scaffold orphan LFS branch for perturbed-demo banks"
git push -u origin data/banks
```

### A1.1. 16-core Mac — bank push helper

Add this shell function to the 16-core Mac's `~/.zshrc` (or a script
in `scripts/`).  This Mac uses it once during scaffolding (Stage 2.0)
to push the canonical Can mass=1.3 banks; afterwards the workhorse
owns all pushes.

```bash
push_banks () {
  # Usage: push_banks file1.hdf5 [file2.hdf5 ...]
  local SHARED="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data"
  local LFS_DIR="/tmp/lfs-banks"
  if [ ! -d "$LFS_DIR/.git" ]; then
    git clone --branch data/banks --single-branch \
      https://github.com/watamelon2/Hybrid-Robot-Trajectory-Generation.git "$LFS_DIR"
  fi
  cd "$LFS_DIR"
  git checkout data/banks
  git pull --rebase
  for f in "$@"; do
    cp "$SHARED/$f" "$LFS_DIR/$f"
    git add "$f"
  done
  git commit -m "data(banks): push $* ($(date '+%Y-%m-%d %H:%M'))"
  git push
  cd - >/dev/null
}
```

After a slate finishes building a new bank, run e.g.:

```bash
push_banks can_mass1p3_205demos.hdf5 can_mass1p3_220demos.hdf5
```

### B1. 10-core Mac — bank pull + push helpers

The 10-core Mac needs **both** helpers since it's the workhorse and
the ongoing canonical collector.  Add both to its `~/.zshrc`.

```bash
pull_banks () {
  local SHARED="${SHARED_DATA:-$HOME/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data}"
  local LFS_DIR="/tmp/lfs-banks"
  if [ ! -d "$LFS_DIR/.git" ]; then
    git clone --branch data/banks --single-branch \
      https://github.com/watamelon2/Hybrid-Robot-Trajectory-Generation.git "$LFS_DIR"
  fi
  cd "$LFS_DIR"
  git checkout data/banks
  git pull --rebase
  git lfs pull                          # materialise the blobs
  mkdir -p "$SHARED"
  cp -n *.hdf5 "$SHARED/" 2>/dev/null || true   # -n = don't overwrite local
  cd - >/dev/null
  ls -lh "$SHARED"/*.hdf5
}

push_banks () {
  # Usage: push_banks file1.hdf5 [file2.hdf5 ...]
  local SHARED="${SHARED_DATA:-$HOME/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data}"
  local LFS_DIR="/tmp/lfs-banks"
  if [ ! -d "$LFS_DIR/.git" ]; then
    git clone --branch data/banks --single-branch \
      https://github.com/watamelon2/Hybrid-Robot-Trajectory-Generation.git "$LFS_DIR"
  fi
  cd "$LFS_DIR"
  git checkout data/banks
  git pull --rebase
  for f in "$@"; do
    cp "$SHARED/$f" "$LFS_DIR/$f"
    git add "$f"
  done
  git commit -m "data(banks): push $* ($(date '+%Y-%m-%d %H:%M'))"
  git push
  cd - >/dev/null
}
```

The 10-core Mac runs `pull_banks` once before kicking off its queue,
then `push_banks` after slates (3) and (4) finish their bank
production (see launch sequence below).

### Bank dependency map (what each slate consumes / produces)

| Slate           | Consumes                                  | Produces                                       |
|-----------------|-------------------------------------------|------------------------------------------------|
| (1) multiseed   | `can_mass1p3_{205,220}.hdf5`              | none new                                       |
| (3) magnitude   | (builds its own)                          | `can_mass{1p1,1p2,1p4,1p5}_{205,220}.hdf5`     |
| (2) democount   | `can_mass1p3_{201..204,210}.hdf5`         | derives those from the 20-demo source bank     |
| (6) lambda      | `can_mass1p3_205.hdf5`                    | none new                                       |
| (4) square hyb. | (builds its own)                          | `square_mass1p3_{5,205,220}.hdf5`              |
| (5) bcmulti     | canonical Lift/Square PH (from robomimic) | `policy_{lift,square}_{canonical,mass1p3}.pt`  |

Slates (2) and (6) only need the canonical `can_mass1p3_205.hdf5`
bank.  If the 10-core Mac has the upstream 20-demo collect bank
(`can_mass1p3_20demos.hdf5`), each slate launcher auto-derives the
slice variants locally via `derive_perturbed_banks.py`.  Pulling
only the 20-demo source from LFS is sufficient.

---

## Phase 2 — Launch sequence

### Stage 2.0 — Seed the Can mass=1.3 bank (16-core Mac, ~10 min)

The 16-core Mac's (1) multiseed slate and the 10-core Mac's (2)
democount + (6) lambda slates all need the canonical Can mass=1.3
bank.  Build it once on the 16-core Mac during scaffolding and push
to LFS so the workhorse can `pull_banks` immediately when it comes
online.

```bash
# === 16-core Mac ===
cd "/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-multiseed"
conda activate hybrid-traj

LOCAL="$PWD/experiments/15_unified_hybrid/data"
SHARED="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data"
mkdir -p "$LOCAL" "$SHARED"

python experiments/15_unified_hybrid/collect_perturbed_demos.py \
    --task can --mass-scale 1.3 --friction-scale 1.0 \
    --n-demos-target 20 \
    --output "$LOCAL/can_mass1p3_20demos.hdf5" \
    --combine-output "$LOCAL/can_mass1p3_220demos.hdf5" \
    --verbose
python experiments/15_unified_hybrid/derive_perturbed_banks.py \
    --perturbed-hdf5 "$LOCAL/can_mass1p3_20demos.hdf5" \
    --output-dir "$LOCAL"
cp "$LOCAL"/can_mass1p3_{20,5,205,220}demos.hdf5 "$SHARED/"

# Push to LFS so the 10-core Mac can pull
push_banks can_mass1p3_20demos.hdf5 \
           can_mass1p3_5demos.hdf5  \
           can_mass1p3_205demos.hdf5 \
           can_mass1p3_220demos.hdf5
```

This collect step lives **outside** any slate launcher so it doesn't
get duplicated across the (1)/(2)/(6) consumers.  Doing it once on
the scaffolding side is the single bottleneck before parallel
launch begins.

### Stage 2.1 — 16-core Mac launches its single slate

The 16-core Mac runs only (1) multiseed.  No queue needed.

```bash
# === 16-core Mac ===
cd "/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-multiseed"
screen -dmS slate_multiseed \
    bash experiments/15_unified_hybrid/slate_incremental_T2_multiseed_2026_05_28.sh
```

This runs ~7 h.  Per-cell JSONs land in
`experiments/15_unified_hybrid/results/slate_incremental_T2_multiseed_2026_05_28/seed_{1,2}/`.
When DONE, this Mac is idle until the workhorse finishes and (7)
analysis is ready to run.

### Stage 2.2 — 10-core Mac queues the bulk

Workhorse queue: (5) bcmulti first (6 min, lets it land on `main`
early), then (3) magnitude (heaviest, produces 4 banks), then
(2) democount, (6) lambda, (4) Square-hybrid.  Single screen,
sequential.

```bash
# === 10-core Mac ===
conda activate hybrid-traj
pull_banks                            # materialises Can mass=1.3 banks
export N_WORKERS=8                    # global override; honoured by all launchers

REPO="$HOME/Documents/HRTG/Hybrid-Robot-Trajectory-Generation"

cat > /tmp/workhorse_queue.sh <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
export N_WORKERS="${N_WORKERS:-8}"
REPO="$HOME/Documents/HRTG/Hybrid-Robot-Trajectory-Generation"
LFS_DIR="/tmp/lfs-banks"
SHARED="${SHARED_DATA:-$HOME/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data}"

push_banks_inline () {
  cd "$LFS_DIR" && git pull --rebase
  for f in "$@"; do cp "$SHARED/$f" "$LFS_DIR/$f"; git add "$f"; done
  git commit -m "data(banks): workhorse push $* ($(date '+%Y-%m-%d %H:%M'))"
  git push
  cd - >/dev/null
}

# --- (5) bcmulti --------------------------------------------------
cd "$REPO"
git fetch && git checkout feat/adapt-bc-multitask-obs
bash experiments/03_bc_baseline/slate_multitask_retrain_2026_05_28.sh
echo "[workhorse] (5) bcmulti DONE at $(date)"

# --- (3) magnitude ------------------------------------------------
cd "$REPO"
git checkout feat/adapt-t2-magnitude
bash experiments/15_unified_hybrid/slate_incremental_T2_magnitude_2026_05_28.sh
echo "[workhorse] (3) magnitude DONE at $(date)"
push_banks_inline can_mass1p1_205demos.hdf5 can_mass1p2_205demos.hdf5 \
                  can_mass1p4_205demos.hdf5 can_mass1p5_205demos.hdf5 \
                  can_mass1p1_220demos.hdf5 can_mass1p2_220demos.hdf5 \
                  can_mass1p4_220demos.hdf5 can_mass1p5_220demos.hdf5

# --- (2) democount ------------------------------------------------
cd "$REPO"
git checkout feat/adapt-t2-democount
bash experiments/15_unified_hybrid/slate_incremental_T2_democount_2026_05_28.sh
echo "[workhorse] (2) democount DONE at $(date)"

# --- (6) lambda ---------------------------------------------------
cd "$REPO"
git checkout feat/adapt-t2-lambda
bash experiments/15_unified_hybrid/slate_incremental_T2_lambda_2026_05_28.sh
echo "[workhorse] (6) lambda DONE at $(date)"

# --- (4) Square hybrid --------------------------------------------
cd "$REPO"
git checkout feat/adapt-t2-square
bash experiments/15_unified_hybrid/slate_incremental_T2_square_hybrid_2026_05_28.sh
echo "[workhorse] (4) Square hybrid DONE at $(date)"
push_banks_inline square_mass1p3_5demos.hdf5 \
                  square_mass1p3_205demos.hdf5 \
                  square_mass1p3_220demos.hdf5

echo "[workhorse] ALL DONE at $(date)"
EOF
chmod +x /tmp/workhorse_queue.sh
screen -dmS workhorse bash /tmp/workhorse_queue.sh
```

> **N_WORKERS env var.** All four affected launchers were
> parameterized in the n-workers-parameterization commit (see
> Phase 0 / N0).  Setting `N_WORKERS=8` once in the queue script
> propagates to every cell.

### Stage 2.3 — Live monitoring

```bash
# === 16-core Mac ===
screen -ls                                          # should show slate_multiseed
tail -f "/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-multiseed/experiments/15_unified_hybrid/results/slate_incremental_T2_multiseed_2026_05_28/run.log"

# === 10-core Mac ===
screen -ls                                          # should show workhorse
tail -f $(ls -td "$HOME/Documents/HRTG/Hybrid-Robot-Trajectory-Generation/experiments/15_unified_hybrid/results/slate_incremental_T2_*_2026_05_28"/run.log 2>/dev/null | head -1)

# Per-machine CPU / RAM sanity (do not let load average drift above
# physical cores by more than ~20 %)
top -l 1 | head -20
```

---

## Phase 3 — Aggregation, push, and merge

Each slate's launcher writes per-cell JSONs into
`experiments/15_unified_hybrid/results/<slate_name>/` and a top-level
`run.log`.  Per-slate aggregation runs locally on the machine that
produced the data:

```bash
# Same on either machine, in the worktree that ran the slate
python experiments/15_unified_hybrid/aggregate_incremental.py \
    --results-dir experiments/15_unified_hybrid/results/slate_incremental_T2_multiseed_2026_05_28 \
    --output docs/results_writeups/t2_multiseed_AGGREGATE.json
```

Each branch's `docs/results_writeups/<branch_slug>.md` stub gets
filled in with the pooled numbers + CIs at this point.

### Commit + push from each Mac

```bash
# 16-core Mac — one branch to push (the multiseed worktree)
cd "/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-multiseed"
git add -A
echo "adapt(multiseed): land results JSONs + writeup numbers" > /tmp/commit_msg.txt
git commit -F /tmp/commit_msg.txt
git push
```

```bash
# 10-core Mac — five branches to push (workhorse output)
cd "$HOME/Documents/HRTG/Hybrid-Robot-Trajectory-Generation"
for b in feat/adapt-bc-multitask-obs \
         feat/adapt-t2-magnitude    \
         feat/adapt-t2-democount    \
         feat/adapt-t2-lambda       \
         feat/adapt-t2-square; do
  git checkout "$b"
  git add -A
  printf 'adapt(%s): land results JSONs + writeup numbers\n' "$b" > /tmp/commit_msg.txt
  git commit -F /tmp/commit_msg.txt
  git push
done
```

### Merge order to `main` (run on the 16-core Mac only)

Per `docs/adaptation/05_followup_plan.md`'s merge feasibility note,
only step (5) is order-sensitive (must merge before any later (4)-BC
work).  Otherwise any order is fine.

```bash
cd "/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
git checkout main && git pull
for b in feat/adapt-bc-multitask-obs \
         feat/adapt-t2-multiseed \
         feat/adapt-t2-democount \
         feat/adapt-t2-magnitude \
         feat/adapt-t2-lambda \
         feat/adapt-t2-square; do
  git merge --no-ff "$b" -m "Merge $b into main"
done
git push
```

Then run the post-hoc analysis branch:

```bash
git checkout feat/adapt-pareto-analysis && git rebase main
python experiments/15_unified_hybrid/measure_bias_variance.py \
    --all-results-root experiments/15_unified_hybrid/results \
    --output docs/results_writeups/pareto_analysis.json
git add -A
echo "adapt(analysis): land (bias, var) measurements + Pareto figure" > /tmp/commit_msg.txt
git commit -F /tmp/commit_msg.txt
git push
# Then merge as the final integration:
git checkout main
git merge --no-ff feat/adapt-pareto-analysis -m "Merge feat/adapt-pareto-analysis into main"
git push
```

### Integration commit on `docs/results.md`

Each branch landed its numbers as a `docs/results_writeups/<slug>.md`
stub to avoid merge conflicts on the central writeup.  After all
seven merge, a single integration commit weaves them into the
running narrative:

```bash
cd "/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
# Edit docs/results.md by hand or via assistant to pull headlines
# from the writeup stubs.  One commit, on main:
echo "docs(results): weave seven follow-up slate headlines into running narrative" > /tmp/commit_msg.txt
git commit -F /tmp/commit_msg.txt -- docs/results.md
git push
```

---

## Failure modes and recovery

### LFS quota exhausted
GitHub free-tier LFS is 1 GB storage + 1 GB/month bandwidth.  Each
bank is ~30-100 MB; the full set of ~15 banks across Can mass
scales + Square fits in ~1 GB.  If you trip the quota:
- Switch to `git lfs prune` on Mac A to clean older bank revisions.
- Or fall back to rsync (option 1 in the earlier sync menu).

### One Mac crashes mid-slate
The slate launchers are idempotent on a per-cell basis: each cell
checks for an existing result JSON and skips it.  Just re-run the
launcher on the surviving Mac (after `git fetch && git pull`).
The BC retrain inside (1) multiseed has the same skip-if-exists
gate via `if [ ! -f "$RETRAIN_CKPT" ]` (see the launcher).

### CPU thrash (load average climbs past ~20)
Likely cause: a background process (Spotlight, Time Machine, Xcode)
fired.  Pause the slate:
```bash
pkill -STOP -f "run_can.py"
# wait for the offender to clear, then:
pkill -CONT -f "run_can.py"
```

### Bank checksum mismatch between Macs
Collects are seeded (`--mass-scale`, `--friction-scale`, and an
implicit seed in `collect_perturbed_demos.py`), so byte-identical
banks on both Macs are expected when produced independently.  If
diffs appear, the LFS pull is the canonical version — overwrite
the local bank from `pull_banks`.

### Phase-A DP eval (Colab) collides with adaptation
It doesn't — DP eval runs on Colab GPU, no local CPU contention.
The Phase-B retrain trigger commit on `feat/dp-hybrid-baseline` is
gated on Phase-A's RED verdict (separate workstream; see
`experiments/08_flow_matching_prior/results/overnight_2026_05_24/BRIEF.md`).

---

## Timeline summary

```
T+00:00   16-core  : Stage 2.0 -- Can mass=1.3 bank build       (~10 min)
T+00:10   16-core  : Stage 2.0 -- push_banks to LFS              (~2 min)
T+00:12   10-core  : pull_banks                                  (~1 min)
T+00:13   16-core  : Stage 2.1 -- launch (1) multiseed
T+00:13   10-core  : Stage 2.2 -- launch workhorse queue
                       → (5) bcmulti                            (~6 min)
                       → (3) magnitude                          (~7-8 h)
                       → (2) democount                          (~4-5 h)
                       → (6) lambda                             (~4-5 h)
                       → (4) Square hybrid                      (~5-9 h)
T+07:00   16-core  : (1) multiseed DONE
T+08:00   10-core  : (3) magnitude DONE ; push 8 mass-scale banks
T+13:00   10-core  : (2) democount + (6) lambda DONE
T+22:00   10-core  : (4) Square hybrid DONE ; push square banks
T+22:10   Both     : aggregate + push branches
T+22:30   16-core  : merge into main, start (7) analysis
T+46:30   16-core  : (7) analysis done (long writing task)
```

Net wall: ~22 h workhorse compute + ~1 day for the analysis writing
pass on the interactive Mac.  The 16-core Mac is idle from T+07:00
until the workhorse finishes — free for any unrelated work the user
wants to do interactively in that window.

---

## Pre-flight checklist

- [ ] 16-core and 10-core Macs both pass `git lfs version`
- [ ] 10-core Mac's B0 bootstrap smoke (B0.6) emits a non-NaN
      `success_rate` in the JSON
- [ ] `data/banks` orphan branch exists on `origin` and the 10-core
      Mac can `pull_banks` successfully
- [ ] `HRTG-shared-data/` resolves to the same path the slate
      launchers expect (or `SHARED_DATA` env var is exported on the
      10-core Mac)
- [ ] All seven feature branches are fetched on both Macs:
      ```bash
      git fetch && git branch -a | grep feat/adapt-
      ```
- [ ] The n-workers parameterization commits exist on the four
      workhorse branches (`feat/adapt-t2-magnitude`,
      `feat/adapt-t2-democount`, `feat/adapt-t2-lambda`,
      `feat/adapt-t2-square`):
      ```bash
      for b in magnitude democount lambda square; do
        git log --oneline origin/main..origin/feat/adapt-t2-$b | head -3
      done
      ```
- [ ] 16-core Mac worktrees are up to date with `origin`:
      ```bash
      for w in HRTG-multiseed HRTG-magnitude HRTG-bcmulti HRTG-democount HRTG-lambda HRTG-square HRTG-analysis; do
        echo "=== $w ==="
        (cd "/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/$w" && \
         git status --short && git log -1 --oneline)
      done
      ```
- [ ] Phase-A DP Colab eval is configured and running off-machine
      (no local CPU load expected)

Once the checklist is green:
1.  16-core Mac runs Stage 2.0 + Stage 2.1 (`slate_multiseed` screen).
2.  10-core Mac runs `pull_banks`, exports `N_WORKERS=8`, kicks off
    the `workhorse` screen via the queue script in Stage 2.2.
