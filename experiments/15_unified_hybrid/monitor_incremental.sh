#!/usr/bin/env bash
# One-shot status snapshot for the incremental-prior slate work.
# Covers Phase-1 collect + T1 cross-task + T2 eval cells + T2.3 BC retrain.
#
# Usage: ./monitor_incremental.sh    (read-only; safe to call any time)
set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root

cyan="\033[36m"; green="\033[32m"; yellow="\033[33m"; red="\033[31m"; r="\033[0m"
hdr() { printf "${cyan}==> %s${r}\n" "$1"; }

hdr "wall clock"
date

hdr "tracked python procs (incremental work)"
ps -A -o pid,etime,cputime,%cpu,command | \
    awk '/run_can\.py.*HRTG-incremental|collect_perturbed_demos|HRTG-incremental.*03_bc_baseline\/train\.py/' \
    | grep -v awk || echo "  (none -- nothing running)"

hdr "load average"
uptime

# Stage 1: collect
hdr "Stage 1: perturbed-dynamics demo collection"
COLLECT_DIR="experiments/15_unified_hybrid/results/slate_incremental_2026_05_26"
DATA_DIR="experiments/15_unified_hybrid/data"
if [ -f "$DATA_DIR/can_mass1p3_220demos.hdf5" ] && [ -f "$DATA_DIR/can_mass1p3_205demos.hdf5" ]; then
    printf "  STATUS: ${green}done${r}\n"
    ls -lh "$DATA_DIR"/*.hdf5 2>/dev/null | sed 's/^/    /'
elif [ -f "$COLLECT_DIR/collect.log" ]; then
    printf "  STATUS: ${yellow}in progress${r}\n"
    grep -E '\[collect\]' "$COLLECT_DIR/collect.log" | tail -3 | sed 's/^/    /'
else
    printf "  STATUS: ${red}not started${r}\n"
fi

# Stage 2: T1 cross-task
hdr "Stage 2: T1 cross-task transfer (6 cells)"
T1_DIR="experiments/15_unified_hybrid/results/slate_incremental_T1_2026_05_26"
T1_LOG="$T1_DIR/run.log"
T1_TAGS=(
  "T1_0_can_can_control"
  "T1_1_can_lift"
  "T1_2_can_square"
  "T1_3_lift_can"
  "T1_4_can_no_prior"
  "T1_5_lift_lift_control"
)
if [ -f "$T1_LOG" ]; then
    printf "  %-32s %-10s %-10s\n" "cell" "trials" "rate"
    for tag in "${T1_TAGS[@]}"; do
        SUMMARY=$(ls "$T1_DIR"/unified_summary_*_"$tag".json 2>/dev/null | head -1)
        if [ -n "$SUMMARY" ] && [ -f "$SUMMARY" ]; then
            RATE=$(python3 -c "
import json
d = json.load(open('$SUMMARY'))
pl = d.get('per_lambda', [])
parts = [f\"L{p['lambda_prior']:.1f}={100*p['success_rate']:.1f}%\" for p in pl]
print(' '.join(parts) if parts else '-')
")
            printf "  ${green}%-32s${r} %-10s %s\n" "$tag" "DONE" "$RATE"
        else
            N=$(grep -c "_${tag}.*success=" "$T1_LOG" 2>/dev/null || echo 0)
            printf "  ${yellow}%-32s${r} %-10s %s\n" "$tag" "$N/30" "running/pending"
        fi
    done
else
    printf "  STATUS: ${red}not started${r}\n"
fi

# Stage 3: T2 eval (4 cells)
hdr "Stage 3: T2 eval cells (T2.0/T2.1/T2.2/T2.4)"
T2_DIR="experiments/15_unified_hybrid/results/slate_incremental_T2_2026_05_26"
T2_LOG="$T2_DIR/run.log"
T2_TAGS=(
  "T2_0_hybrid_canonical_bank_mass1p3"
  "T2_1_hybrid_5demo_adapted_mass1p3"
  "T2_2_bc_unadapted_mass1p3"
  "T2_4_hybrid_20demo_saturation_mass1p3"
)
if [ -f "$T2_LOG" ]; then
    printf "  %-44s %-10s %-10s\n" "cell" "trials" "rate"
    for tag in "${T2_TAGS[@]}"; do
        SUMMARY=$(ls "$T2_DIR"/unified_summary_*_"$tag".json 2>/dev/null | head -1)
        if [ -n "$SUMMARY" ] && [ -f "$SUMMARY" ]; then
            RATE=$(python3 -c "
import json
d = json.load(open('$SUMMARY'))
pl = d.get('per_lambda', [])
parts = [f\"L{p['lambda_prior']:.1f}={100*p['success_rate']:.1f}%\" for p in pl]
print(' '.join(parts) if parts else '-')
")
            printf "  ${green}%-44s${r} %-10s %s\n" "$tag" "DONE" "$RATE"
        else
            N=$(grep -c "_${tag}.*success=" "$T2_LOG" 2>/dev/null || echo 0)
            printf "  ${yellow}%-44s${r} %-10s %s\n" "$tag" "$N/30" "running/pending"
        fi
    done
else
    printf "  STATUS: ${red}not started${r}\n"
fi

# Stage 4: T2.3 BC retrain + eval
hdr "Stage 4: T2.3 BC retrained + eval"
T23_DIR="experiments/15_unified_hybrid/results/slate_incremental_T2_bc_retrain_2026_05_26"
RETRAIN_CKPT="experiments/03_bc_baseline/results/policy_mass1p3_205.pt"
if [ -f "$RETRAIN_CKPT" ]; then
    SUMMARY=$(ls "$T23_DIR"/unified_summary_*_T2_3_bc_retrained_mass1p3.json 2>/dev/null | head -1)
    if [ -n "$SUMMARY" ] && [ -f "$SUMMARY" ]; then
        RATE=$(python3 -c "
import json
d = json.load(open('$SUMMARY'))
pl = d.get('per_lambda', [])
print(' '.join([f\"L{p['lambda_prior']:.1f}={100*p['success_rate']:.1f}%\" for p in pl]) or '-')
")
        printf "  ${green}T2.3 BC retrained + eval: DONE${r}  %s\n" "$RATE"
    else
        printf "  ${yellow}BC retrained but eval pending${r}\n"
    fi
else
    if [ -f "$T23_DIR/run.log" ]; then
        printf "  ${yellow}BC retraining in progress${r}\n"
        tail -2 "$T23_DIR/run.log" | sed 's/^/    /'
    else
        printf "  ${red}not started${r}\n"
    fi
fi

hdr "screens (incremental)"
screen -ls 2>&1 | grep -E 'slate_incremental' || echo "  (no incremental screens)"
