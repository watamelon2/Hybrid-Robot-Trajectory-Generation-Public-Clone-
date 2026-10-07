#!/usr/bin/env bash
# Overnight training + eval queue for Phases A, B, C of the
# DP-replication ladder.  Built 2026-05-24 ~01:00 PDT.
#
# Hardware target: local Apple M4 Max with MPS GPU.  All trainings
# run sequentially on the MPS device; standalone evals run in
# parallel as CPU-bound processes (env stepping is CPU-only) once
# the checkpoints land.
#
# Layout
# ------
#
#   Stage 1 (sequential, MPS):  4 trainings
#     v6_cfm_xl_h120                 ~2.2 h    Phase A1
#     v7_dp_tobs1_h120               ~2.2 h    Phase A2
#     v7_dp_tobs2_h120               ~2.2 h    Phase B
#     v7_dp_tobs2_h16                ~0.3 h    Phase C
#
#   Stage 2 (parallel x4, CPU):  4 standalone evals
#     each: 10 demos x 3 rollouts = 30 rollouts at ~15 s/rollout
#     ~ 15 min wall when run in parallel (4 procs sharing 16 cores)
#
#   Stage 3 (sequential, CPU):  3 MBD evals
#     each: 10 demos x 3 rollouts = 30 rollouts at ~30 s/rollout
#     ~ 15 min each = 45 min total
#
# Total wall ~7.5 hours (training) + 1 hour (eval) = 8.5 hours.
# Started at ~01:00, expected done by ~09:30 PDT.
#
# Output layout
# -------------
#
#   experiments/08_flow_matching_prior/results/
#     v_theta_v6_unet_xl.pt
#     v_theta_dp_tobs1_h120.pt
#     v_theta_dp_tobs2_h120.pt
#     v_theta_dp_tobs2_h16.pt
#     train_log_*.json   (one per training)
#
#   experiments/08_flow_matching_prior/results/overnight_2026_05_24/
#     standalone_v6_unet_xl/
#     standalone_dp_tobs1_h120/
#     standalone_dp_tobs2_h120/      <- with stateful obs buffer
#     standalone_dp_tobs2_h16/       <- closed-loop n_action_steps=8
#     mbd_v6_unet_xl/
#     mbd_dp_tobs1_h120/
#     mbd_dp_tobs2_h120/
#
# Idempotency
# -----------
#
# Each stage skips if its output marker exists.  Re-running after
# a crash will resume from the next un-done stage.
#
# Commits happen between every checkpoint produced and at end of
# Stage 2 / Stage 3 batches.

set -euo pipefail

REPO=/Users/szh/Documents/SZH/Courseworks/ECE\ M237/Project/Hybrid-Robot-Trajectory-Generation
PY=/Users/szh/anaconda3/envs/hybrid-traj/bin/python
EXP08="$REPO/experiments/08_flow_matching_prior"
RESULTS="$EXP08/results"
RUN_DIR="$RESULTS/overnight_2026_05_24"
mkdir -p "$RUN_DIR"

LOG="$RUN_DIR/queue.log"
# Detach stdin from the (potentially dying) controlling terminal so that
# subprocesses launched after the terminal closes don't die with
# "OSError: [Errno 9] Bad file descriptor" during python startup.
exec < /dev/null > >(tee -a "$LOG") 2>&1

# Pick up shared hyperparams.
EPOCHS=50
N_DEMOS=200
EVAL_DEMOS=10
EVAL_ROLLOUTS=3
DEVICE=mps

# Force python to flush stdout/stderr line-by-line so we can monitor
# the queue.log live during training.
export PYTHONUNBUFFERED=1

log() {
    # Portable timestamp for macOS bash 3.2 (no %(...)T printf format).
    printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

commit_if_changes() {
    local msg="$1"
    cd "$REPO"
    if [[ -n $(git status --porcelain experiments/08_flow_matching_prior/results) ]]; then
        git add experiments/08_flow_matching_prior/results || true
        git commit -m "$msg" --allow-empty --no-verify || true
        log "[git] commit: $msg"
    fi
}

# ---------------------------------------------------------------------------
# Stage 1: trainings (sequential on MPS)
# ---------------------------------------------------------------------------

run_training() {
    local name=$1
    local script=$2
    local ckpt=$3
    shift 3
    local args=("$@")

    if [[ -f "$ckpt" ]]; then
        log "[stage-1 $name] checkpoint already exists, skipping: $ckpt"
        return 0
    fi

    log "[stage-1 $name] STARTING  -> $ckpt"
    local t0=$(date +%s)
    "$PY" -u "$script" \
        --task can \
        --n-demos "$N_DEMOS" \
        --epochs "$EPOCHS" \
        --batch-size 256 \
        --device "$DEVICE" \
        --seed 0 \
        --print-every 1 \
        --output "$ckpt" \
        --log "${ckpt%.pt}.log.json" \
        "${args[@]}"
    local dt=$(( $(date +%s) - t0 ))
    log "[stage-1 $name] DONE in ${dt}s -> $ckpt"

    commit_if_changes "exp08: $name trained ($dt s on MPS) [overnight queue]"
}

# ----- v6 CFM XL (Phase A1) -----
run_training v6_cfm_xl \
    "$EXP08/train_v6.py" \
    "$RESULTS/v_theta_v6_unet_xl.pt" \
    --horizon 120 --lr 3e-4 --warmup-epochs 5 --weight-decay 1e-5 \
    --backbone conv1d_unet --hidden 1536 --n-blocks 4 --cond-hidden 256 \
    --t-eps 1e-3 --n-t-per-traj 1 --grad-clip 1.0

# ----- v7 DP T_obs=1 H=120 (Phase A2) -----
run_training v7_dp_tobs1_h120 \
    "$EXP08/train_dp.py" \
    "$RESULTS/v_theta_dp_tobs1_h120.pt" \
    --horizon 120 --lr 1e-4 --warmup-epochs 5 --weight-decay 1e-6 \
    --down-dims 256 512 1024 --n-obs-steps 1 --grad-clip 1.0

# ----- v7 DP T_obs=2 H=120 (Phase B) -----
run_training v7_dp_tobs2_h120 \
    "$EXP08/train_dp.py" \
    "$RESULTS/v_theta_dp_tobs2_h120.pt" \
    --horizon 120 --lr 1e-4 --warmup-epochs 5 --weight-decay 1e-6 \
    --down-dims 256 512 1024 --n-obs-steps 2 --grad-clip 1.0

# ----- v7 DP T_obs=2 H=16 (Phase C) -----
run_training v7_dp_tobs2_h16 \
    "$EXP08/train_dp.py" \
    "$RESULTS/v_theta_dp_tobs2_h16.pt" \
    --horizon 16 --lr 1e-4 --warmup-epochs 5 --weight-decay 1e-6 \
    --down-dims 256 512 1024 --n-obs-steps 2 --grad-clip 1.0

log "[stage-1] ALL TRAININGS DONE"

# ---------------------------------------------------------------------------
# Stage 2: standalone evals (parallel x4 on CPU)
# ---------------------------------------------------------------------------

run_standalone_bg() {
    local name=$1
    local ckpt=$2
    local n_action_steps=$3
    shift 3
    local extra=("$@")

    local out_dir="$RUN_DIR/standalone_${name}"
    mkdir -p "$out_dir"

    if [[ -f "$out_dir/standalone_dp.json" ]]; then
        log "[stage-2 $name] already done, skipping"
        return 0
    fi

    log "[stage-2 $name] BG START  ckpt=$(basename "$ckpt") n_action_steps=$n_action_steps"
    # ${extra[@]+...} is the POSIX-safe way to expand a possibly-empty array
    # under `set -u`; bare "${extra[@]}" would crash with "unbound variable"
    # when no extras were passed in (which is true for all current call sites).
    nohup "$PY" -u "$EXP08/standalone_dp.py" \
        --task can \
        --prior-checkpoint "$ckpt" \
        --n-demos "$EVAL_DEMOS" \
        --n-rollouts "$EVAL_ROLLOUTS" \
        --n-action-steps "$n_action_steps" \
        --max-env-steps 240 \
        --n-steps 18 \
        --device cpu \
        --output-dir "$out_dir" \
        ${extra[@]+"${extra[@]}"} \
        < /dev/null > "$out_dir/run.log" 2>&1 &
    echo $! > "$out_dir/pid"
}

log "[stage-2] launching 4 standalone evals in parallel"

run_standalone_bg v6_unet_xl       "$RESULTS/v_theta_v6_unet_xl.pt"      60
run_standalone_bg dp_tobs1_h120    "$RESULTS/v_theta_dp_tobs1_h120.pt"   60
run_standalone_bg dp_tobs2_h120    "$RESULTS/v_theta_dp_tobs2_h120.pt"   60
run_standalone_bg dp_tobs2_h16_cl  "$RESULTS/v_theta_dp_tobs2_h16.pt"     8

log "[stage-2] waiting for all 4 standalone evals to finish"
for d in v6_unet_xl dp_tobs1_h120 dp_tobs2_h120 dp_tobs2_h16_cl; do
    if [[ -f "$RUN_DIR/standalone_$d/pid" ]]; then
        wait "$(cat "$RUN_DIR/standalone_$d/pid")" || true
        log "[stage-2 $d] finished"
    fi
done

log "[stage-2] ALL STANDALONE EVALS DONE"
commit_if_changes "exp08: standalone evals (v6, dp_tobs1, dp_tobs2_h120, dp_tobs2_h16_cl) [overnight queue]"

# ---------------------------------------------------------------------------
# Stage 3: MBD evals (sequential because run_can.py uses worker pool internally)
# ---------------------------------------------------------------------------

run_mbd() {
    local name=$1
    local ckpt=$2

    local out_dir="$RUN_DIR/mbd_${name}"
    mkdir -p "$out_dir"

    if [[ -f "$out_dir/done" ]]; then
        log "[stage-3 $name] already done, skipping"
        return 0
    fi

    log "[stage-3 $name] STARTING  ckpt=$(basename "$ckpt")"
    local t0=$(date +%s)
    "$PY" -u "$REPO/experiments/15_unified_hybrid/run_can.py" \
        --task can \
        --prior flow \
        --prior-checkpoint "$ckpt" \
        --n-demos "$EVAL_DEMOS" \
        --n-rollouts "$EVAL_ROLLOUTS" \
        --K 10 \
        --mpc-replans 1 \
        --mpc-chunk-steps 60 \
        --lambda-prior 1.0 \
        --warm-start \
        --deterministic-prior-warmstart \
        --output-dir "$out_dir" \
        --output-name-suffix "_overnight_$name" \
        > "$out_dir/run.log" 2>&1 || true
    local dt=$(( $(date +%s) - t0 ))
    touch "$out_dir/done"
    log "[stage-3 $name] DONE in ${dt}s"
}

run_mbd v6_unet_xl     "$RESULTS/v_theta_v6_unet_xl.pt"
run_mbd dp_tobs1_h120  "$RESULTS/v_theta_dp_tobs1_h120.pt"
run_mbd dp_tobs2_h120  "$RESULTS/v_theta_dp_tobs2_h120.pt"

log "[stage-3] ALL MBD EVALS DONE"
commit_if_changes "exp08: MBD evals (v6, dp_tobs1, dp_tobs2_h120) [overnight queue]"

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

log "[summary] writing $RUN_DIR/SUMMARY.md"
"$PY" - <<'PYEOF' "$RUN_DIR"
import json, sys
from pathlib import Path

run_dir = Path(sys.argv[1])
rows = []

for d in sorted(run_dir.glob("standalone_*")):
    j = d / "standalone_dp.json"
    if not j.exists():
        continue
    blob = json.loads(j.read_text())
    rows.append((
        d.name,
        blob.get("n_succ", 0),
        blob.get("n", 0),
        100.0 * blob.get("succ_rate", 0.0),
        blob.get("wall_s", 0.0),
    ))

for d in sorted(run_dir.glob("mbd_*")):
    # find the unified_*.json that run_can.py writes
    js = list(d.glob("unified_*.json"))
    js = [j for j in js if "summary" not in j.name]
    if not js:
        continue
    blob = json.loads(js[0].read_text())
    rows.append((
        d.name,
        blob.get("n_success", 0),
        blob.get("n_total", 0),
        100.0 * blob.get("n_success", 0) / max(blob.get("n_total", 1), 1),
        blob.get("wall_s", 0.0) if "wall_s" in blob else 0.0,
    ))

lines = ["# Overnight queue summary (2026-05-24)", "", "| run | success | rate | wall (s) |", "|-----|---------|------|----------|"]
for name, succ, total, rate, wall in rows:
    lines.append(f"| `{name}` | {succ}/{total} | {rate:.1f}% | {wall:.0f} |")

(run_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print((run_dir / "SUMMARY.md").read_text())
PYEOF

commit_if_changes "exp08: overnight summary (2026-05-24) [overnight queue]"

log "ALL DONE."
