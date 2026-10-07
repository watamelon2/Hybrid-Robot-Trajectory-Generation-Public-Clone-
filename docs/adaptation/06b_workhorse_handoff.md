# 10-core Mac workhorse handoff (agent-actionable)

> **Closure note (2026-06-01)**: the workhorse handoff executed
> as planned.  All 5 slates landed and were merged into `main`
> across 2026-05-28; the post-hoc `feat/adapt-pareto-analysis`
> branch closed the arc 2026-06-01 (`aaefba2`).  See
> [`05_followup_plan.md`](05_followup_plan.md) "Closure (2026-06-01)"
> for the full merge-status table.  This runbook is preserved as a
> template for future multi-slate workhorse handoffs.

This is a self-contained runbook for the agent on the **10-core Mac**
(the workhorse).  It assumes:

  * Fresh macOS install, Apple Silicon, 32 GB RAM, ~10 cores.
  * Internet + GitHub access (the repo is public; LFS is via HTTPS).
  * The 16-core Mac already pushed `data/banks` with 4 Can mass=1.3
    banks (commit `2c99a1b` on branch `data/banks`).
  * The 16-core Mac is currently running `slate_multiseed` (slate 1);
    your job is everything else.

**Goal:** run 5 slates sequentially (~22-25 h wall), then push 5
result branches.

```
(5) bcmulti      ~6  min   <- first, lets it merge into main early
(3) magnitude    ~7-8 h   <- builds 4 new perturbed-mass banks
(2) democount    ~4-5 h
(6) lambda       ~4-5 h
(4) Square hyb.  ~5-9 h   <- builds 3 new Square banks
                ===========
                ~22-25 h
```

## Constants the agent needs

| Variable        | Value                                                              |
|-----------------|--------------------------------------------------------------------|
| Repo URL        | `https://github.com/watamelon2/Hybrid-Robot-Trajectory-Generation.git` |
| Conda env name  | `hybrid-traj`                                                      |
| Conda env file  | `env/environment.yml`                                              |
| LFS branch      | `data/banks` (orphan branch; do not merge into code)               |
| Workers         | `N_WORKERS=8` (10-core - 2 spare for OS)                           |
| Screen name     | `workhorse`                                                        |

## Phase 0 — Machine bootstrap

> Skip any step that's already done.  Re-running is safe.

```bash
# 0.1  Xcode Command-Line Tools (interactive on first run)
xcode-select --install 2>/dev/null || true

# 0.2  Homebrew (no-op if already installed)
if ! command -v brew >/dev/null; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Add brew to PATH for this session (Apple Silicon default)
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# 0.3  conda (miniforge), git, git-lfs, screen
brew list --cask miniforge >/dev/null 2>&1 || brew install --cask miniforge
brew list git >/dev/null 2>&1 || brew install git
brew list git-lfs >/dev/null 2>&1 || brew install git-lfs
git lfs install

# 0.4  Initialise conda for the current shell (zsh)
CONDA_BASE="$(brew --prefix)/Caskroom/miniforge/base"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 0.5  Sanity
git --version
git lfs version
conda --version
echo "PHASE 0 DONE"
```

**Gate:** `git lfs version` prints `git-lfs/3.x.x`.

## Phase 1 — Workspace + dependencies

```bash
# 1.1  Project root + repo clone
export PROJECT_ROOT="$HOME/Documents/HRTG"
mkdir -p "$PROJECT_ROOT" && cd "$PROJECT_ROOT"
if [ ! -d Hybrid-Robot-Trajectory-Generation ]; then
  git clone https://github.com/watamelon2/Hybrid-Robot-Trajectory-Generation.git
fi
cd Hybrid-Robot-Trajectory-Generation

# 1.2  Conda env (matches the 16-core Mac's hybrid-traj exactly)
if ! conda env list | grep -q "^hybrid-traj "; then
  conda env create -f env/environment.yml      # ~10-15 min
fi
conda activate hybrid-traj

# 1.3  Robomimic canonical PH datasets (Can, Lift, Square)
#      Land inside the conda env's site-packages, NOT the repo.
python -m robomimic.scripts.download_datasets \
    --tasks can lift square \
    --dataset_types ph \
    --hdf5_types low_dim

# 1.4  SHARED_DATA location (workhorse-local, persists across slates)
export SHARED_DATA="$PROJECT_ROOT/HRTG-shared-data"
mkdir -p "$SHARED_DATA"
# Persist for future shells
grep -q 'SHARED_DATA=' "$HOME/.zshrc" 2>/dev/null || \
  echo "export SHARED_DATA=\"$PROJECT_ROOT/HRTG-shared-data\"" >> "$HOME/.zshrc"

# 1.5  Smoke test (~2-3 min).  MUST emit a finite success_rate.
mkdir -p /tmp/macb_smoke
python experiments/15_unified_hybrid/run_can.py \
    --task can --prior demo_1nn --lambda-prior 1 \
    --n-demos 2 --n-rollouts 1 --n-workers 2 \
    --K 4 --M 4 --H 60 --mpc-replans 1 --mpc-chunk-steps 30 \
    --output-dir /tmp/macb_smoke 2>&1 | tail -30

ls /tmp/macb_smoke/*.json | xargs -I{} python -c "
import json, sys
d = json.load(open('{}'))
print('smoke success_rate:', d.get('success_rate'))
assert isinstance(d.get('success_rate'), (int, float)), 'non-numeric success_rate'
"
echo "PHASE 1 DONE"
```

**Gate:** smoke JSON contains a numeric `success_rate` (any value).

## Phase 1.6 — Slate-file preflight (per-branch sanity)

> **Why this is here:** the slate `.sh` files exist **only on their
> respective `feat/adapt-*` branches**, not on `main`.  If you grep
> `main` for them you will find nothing — that is correct.  Verify
> instead against `origin/<feature-branch>` to confirm each slate
> script is reachable before launching the queue.

```bash
cd "$HOME/Documents/HRTG/Hybrid-Robot-Trajectory-Generation"
git fetch origin --quiet
MISSING=0
for B in feat/adapt-bc-multitask-obs feat/adapt-t2-magnitude \
         feat/adapt-t2-democount    feat/adapt-t2-lambda    \
         feat/adapt-t2-square; do
  case "$B" in
    *bc-multitask-obs) F=experiments/03_bc_baseline/slate_multitask_retrain_2026_05_28.sh ;;
    *magnitude)        F=experiments/15_unified_hybrid/slate_incremental_T2_magnitude_2026_05_28.sh ;;
    *democount)        F=experiments/15_unified_hybrid/slate_incremental_T2_democount_2026_05_28.sh ;;
    *lambda)           F=experiments/15_unified_hybrid/slate_incremental_T2_lambda_2026_05_28.sh ;;
    *square)           F=experiments/15_unified_hybrid/slate_incremental_T2_square_hybrid_2026_05_28.sh ;;
  esac
  if git cat-file -e "origin/$B:$F" 2>/dev/null; then
    echo "  [OK]      $B  $F"
  else
    echo "  [MISSING] $B  $F"
    MISSING=$((MISSING+1))
  fi
done
[ "$MISSING" -eq 0 ] && echo "PHASE 1.6 DONE -- all 5 slates reachable" \
                    || { echo "ABORT -- $MISSING slate(s) missing on origin"; exit 1; }
```

**Gate:** 5 lines starting with `[OK]`.

## Phase 2 — Bank sync (LFS pull)

```bash
# 2.1  Install pull_banks/push_banks helpers
cat > "$HOME/.local/bin/bank_helpers.sh" <<'EOF'
pull_banks () {
  local SHARED="${SHARED_DATA:?SHARED_DATA env var must be set}"
  local LFS_DIR="/tmp/lfs-banks"
  if [ ! -d "$LFS_DIR/.git" ]; then
    git clone --branch data/banks --single-branch \
      https://github.com/watamelon2/Hybrid-Robot-Trajectory-Generation.git "$LFS_DIR"
  fi
  ( cd "$LFS_DIR" && git checkout data/banks && git pull --rebase && git lfs pull )
  mkdir -p "$SHARED"
  cp -n "$LFS_DIR"/*.hdf5 "$SHARED/" 2>/dev/null || true
  ls -lh "$SHARED"/*.hdf5
}
push_banks () {
  local SHARED="${SHARED_DATA:?SHARED_DATA env var must be set}"
  local LFS_DIR="/tmp/lfs-banks"
  if [ ! -d "$LFS_DIR/.git" ]; then
    git clone --branch data/banks --single-branch \
      https://github.com/watamelon2/Hybrid-Robot-Trajectory-Generation.git "$LFS_DIR"
  fi
  ( cd "$LFS_DIR" && git checkout data/banks && git pull --rebase
    for f in "$@"; do cp "$SHARED/$f" "$LFS_DIR/$f"; git add "$f"; done
    git commit -m "data(banks): workhorse push $* ($(date '+%Y-%m-%d %H:%M'))"
    git push )
}
EOF
mkdir -p "$HOME/.local/bin"
grep -q 'bank_helpers.sh' "$HOME/.zshrc" 2>/dev/null || \
  echo "source $HOME/.local/bin/bank_helpers.sh" >> "$HOME/.zshrc"
source "$HOME/.local/bin/bank_helpers.sh"

# 2.2  Pull the 4 Can mass=1.3 banks pre-built by the 16-core Mac
pull_banks
```

**Gate:** `pull_banks` finishes with 4 hdf5 files listed in
`$SHARED_DATA/`:
  * `can_mass1p3_5demos.hdf5`    (~468 KB)
  * `can_mass1p3_20demos.hdf5`   (~1.9 MB)
  * `can_mass1p3_205demos.hdf5`  (~46 MB)
  * `can_mass1p3_220demos.hdf5`  (~47 MB)

If sizes look like a few hundred bytes each: LFS materialisation
failed — re-run `( cd /tmp/lfs-banks && git lfs pull )`.

## Phase 3 — Launch workhorse queue

The queue script runs 5 slates sequentially in one screen, pushes
new banks to LFS automatically after the bank-producing slates, and
logs everything to `/tmp/workhorse_*.log`.

```bash
# 3.1  Write the queue script
REPO="$HOME/Documents/HRTG/Hybrid-Robot-Trajectory-Generation"
cat > /tmp/workhorse_queue.sh <<'EOF'
#!/usr/bin/env bash
# Workhorse queue: 5 slates sequential, total ~22-25 h.
set -uo pipefail   # not -e: we want each slate to run even if a prior one tripped

export N_WORKERS="${N_WORKERS:-8}"
export SHARED_DATA="${SHARED_DATA:?SHARED_DATA must be set}"
REPO="${REPO:-$HOME/Documents/HRTG/Hybrid-Robot-Trajectory-Generation}"
LFS_DIR="/tmp/lfs-banks"
LOG="/tmp/workhorse_main.log"

source "$HOME/.local/bin/bank_helpers.sh"

step () {
  local TAG="$1"; shift
  echo "" | tee -a "$LOG"
  echo "========================================" | tee -a "$LOG"
  echo "[workhorse $(date '+%F %H:%M:%S')] >>> $TAG" | tee -a "$LOG"
  echo "========================================" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  echo "[workhorse $(date '+%F %H:%M:%S')] <<< $TAG  (exit $?)" | tee -a "$LOG"
}

# --- (5) bcmulti ---------------------------------------------------
step "(5) bcmulti -- 4 BC retrains, ~6 min" bash -c '
  cd "$REPO"
  git fetch
  git checkout feat/adapt-bc-multitask-obs
  git pull --ff-only
  bash experiments/03_bc_baseline/slate_multitask_retrain_2026_05_28.sh
'

# --- (3) magnitude -------------------------------------------------
step "(3) magnitude -- 4 mass-scales x 3 hybrid cells, ~7-8 h" bash -c '
  cd "$REPO"
  git checkout feat/adapt-t2-magnitude
  git pull --ff-only
  bash experiments/15_unified_hybrid/slate_incremental_T2_magnitude_2026_05_28.sh
'
step "(3.post) push 8 new mass-scale banks to LFS" bash -c '
  push_banks can_mass1p1_205demos.hdf5 can_mass1p2_205demos.hdf5 \
             can_mass1p4_205demos.hdf5 can_mass1p5_205demos.hdf5 \
             can_mass1p1_220demos.hdf5 can_mass1p2_220demos.hdf5 \
             can_mass1p4_220demos.hdf5 can_mass1p5_220demos.hdf5 \
             can_mass1p1_20demos.hdf5  can_mass1p2_20demos.hdf5 \
             can_mass1p4_20demos.hdf5  can_mass1p5_20demos.hdf5 \
             can_mass1p1_5demos.hdf5   can_mass1p2_5demos.hdf5 \
             can_mass1p4_5demos.hdf5   can_mass1p5_5demos.hdf5 \
    || echo "[push] partial -- some files may be missing; that is fine"
'

# --- (2) democount -------------------------------------------------
step "(2) democount -- 4 hybrid cells, ~4-5 h" bash -c '
  cd "$REPO"
  git checkout feat/adapt-t2-democount
  git pull --ff-only
  bash experiments/15_unified_hybrid/slate_incremental_T2_democount_2026_05_28.sh
'

# --- (6) lambda ----------------------------------------------------
step "(6) lambda -- 3 hybrid cells, ~4-5 h" bash -c '
  cd "$REPO"
  git checkout feat/adapt-t2-lambda
  git pull --ff-only
  bash experiments/15_unified_hybrid/slate_incremental_T2_lambda_2026_05_28.sh
'

# --- (4) Square hybrid ---------------------------------------------
step "(4) Square hybrid -- 3 hybrid cells + Square bank build, ~5-9 h" bash -c '
  cd "$REPO"
  git checkout feat/adapt-t2-square
  git pull --ff-only
  bash experiments/15_unified_hybrid/slate_incremental_T2_square_hybrid_2026_05_28.sh
'
step "(4.post) push 3 new Square banks to LFS" bash -c '
  push_banks square_mass1p3_5demos.hdf5  \
             square_mass1p3_20demos.hdf5 \
             square_mass1p3_205demos.hdf5 \
             square_mass1p3_220demos.hdf5 \
    || echo "[push] partial -- some files may be missing; that is fine"
'

echo "[workhorse $(date '+%F %H:%M:%S')] ALL DONE" | tee -a "$LOG"
EOF
chmod +x /tmp/workhorse_queue.sh

# 3.2  Launch in screen (detached)
screen -dmS workhorse bash -c '
  source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
  conda activate hybrid-traj
  export SHARED_DATA="$HOME/Documents/HRTG/HRTG-shared-data"
  export REPO="$HOME/Documents/HRTG/Hybrid-Robot-Trajectory-Generation"
  export N_WORKERS=8
  bash /tmp/workhorse_queue.sh
'

# 3.3  Confirm screen is alive
sleep 5
screen -ls | grep workhorse
echo "PHASE 3 LAUNCHED"
```

**Gate:** `screen -ls` shows `workhorse (Detached)`.  Within ~30 s,
`/tmp/workhorse_main.log` should exist and contain a `>>> (5)
bcmulti` line.

## Phase 4 — Watch loop

Run this **every ~10 min** while the queue is alive.  The agent
should react to any failed gate.

```bash
# 4.1  Liveness
screen -ls | grep -q workhorse || echo "ALERT: screen workhorse died"

# 4.2  Latest main log line
tail -5 /tmp/workhorse_main.log 2>/dev/null

# 4.3  Currently-running slate log (most recent run.log under results/)
LATEST_LOG=$(ls -td "$HOME/Documents/HRTG/Hybrid-Robot-Trajectory-Generation"/experiments/{15_unified_hybrid,03_bc_baseline}/results/*/run.log 2>/dev/null | head -1)
[ -n "$LATEST_LOG" ] && tail -10 "$LATEST_LOG"

# 4.4  CPU sanity (load avg should stay < 12; if > 16, intervene)
sysctl -n vm.loadavg

# 4.5  RAM sanity (compressor should not be slammed)
top -l 1 -s 0 | head -5

# 4.6  Per-slate completion count: how many JSONs landed so far
for d in "$HOME/Documents/HRTG/Hybrid-Robot-Trajectory-Generation"/experiments/15_unified_hybrid/results/slate_incremental_T2_*_2026_05_28; do
  [ -d "$d" ] || continue
  echo "$(basename $d) : $(ls $d/*.json 2>/dev/null | wc -l) JSONs"
done
```

**Intervention triggers (rare):**

| Symptom                                              | Action                                                           |
|------------------------------------------------------|------------------------------------------------------------------|
| Load avg > 16 sustained                              | `pkill -STOP -f run_can.py`; wait 30 s; `pkill -CONT -f run_can.py` |
| Screen `workhorse` dies mid-slate                    | Re-run `screen -dmS workhorse bash /tmp/workhorse_queue.sh`; the launchers are idempotent per cell (skip-if-exists) |
| LFS push fails (rate-limited / auth)                 | Retry once: `cd /tmp/lfs-banks && git push` — if still fails, leave banks local; the result JSONs are already on the workhorse filesystem |
| Disk fills (`df -h .`)                               | `cd /tmp/lfs-banks && git lfs prune` reclaims old LFS blob copies |
| Network drops mid-`pull_banks`                       | `cd /tmp/lfs-banks && git lfs pull` (idempotent)                 |

## Phase 5 — Finish: aggregate + push 5 branches

When `/tmp/workhorse_main.log` ends with `ALL DONE`:

```bash
cd "$HOME/Documents/HRTG/Hybrid-Robot-Trajectory-Generation"
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate hybrid-traj

# 5.1  Per-branch: aggregate result JSONs, commit, push
for spec in \
  "feat/adapt-bc-multitask-obs:experiments/03_bc_baseline/results" \
  "feat/adapt-t2-magnitude:experiments/15_unified_hybrid/results/slate_incremental_T2_magnitude_2026_05_28" \
  "feat/adapt-t2-democount:experiments/15_unified_hybrid/results/slate_incremental_T2_democount_2026_05_28" \
  "feat/adapt-t2-lambda:experiments/15_unified_hybrid/results/slate_incremental_T2_lambda_2026_05_28" \
  "feat/adapt-t2-square:experiments/15_unified_hybrid/results/slate_incremental_T2_square_hybrid_2026_05_28"; do
    B="${spec%%:*}"; R="${spec##*:}"
    git checkout "$B" && git pull --ff-only
    # Run aggregator if the slate has one (skip silently if not)
    if [ -d "$R" ]; then
      python experiments/15_unified_hybrid/aggregate_incremental.py \
          --results-dir "$R" --output "docs/results_writeups/$(basename $R)_AGGREGATE.json" \
          2>/dev/null || true
    fi
    git add -A
    printf 'adapt(%s): land results JSONs + writeup numbers\n' "$B" > /tmp/commit_msg.txt
    git commit -F /tmp/commit_msg.txt || echo "(nothing to commit on $B)"
    git push
done

echo "PHASE 5 DONE -- the 16-core Mac will handle merge-to-main."
```

**Gate:** `git log origin/feat/adapt-bc-multitask-obs --oneline | head`
shows a fresh `adapt(...): land results` commit; same for the
other 4 branches.

After this point, hand off back to the 16-core Mac for Phase 3 of
`docs/adaptation/06_two_machine_launch.md` (merge into `main`,
weave numbers into `docs/results.md`, run (7) pareto-analysis).

## Appendix — Quick reference

```bash
# Attach to the live workhorse screen
screen -r workhorse

# Detach without killing: Ctrl-A, then D

# Kill everything (emergency)
screen -X -S workhorse quit
pkill -f run_can.py
pkill -f workhorse_queue.sh

# Re-run the queue from where it left off (idempotent per cell)
screen -dmS workhorse bash /tmp/workhorse_queue.sh

# Check which slate is currently running
ps -ax -o etime,command | grep -E "slate_incremental|slate_multitask" | grep -v grep
```

## What the 16-core Mac is doing in parallel

Running slate (1) `feat/adapt-t2-multiseed` (~7 h, started ~2026-05-29
02:05 PT, screen `slate_multiseed`).  It does NOT touch any of the
five branches you're running; no merge contention.

After both Macs finish, the 16-core Mac is responsible for:
  * Merging 6 branches into `main` (your 5 + its 1).
  * Running slate (7) `feat/adapt-pareto-analysis` post-hoc.
  * Weaving headlines into `docs/results.md`.
