#!/bin/bash
# Quick status monitor for the currently-running A1.4 sigma-sweep
# and A2 V_phi training launched from feat/stronger-mbd-signal.
#
# Usage:
#   bash scripts/local/monitor_a1_a2.sh
#
# Reports per-cell rollout counts, successes, ETAs, and the V_phi
# training log tail. Read-only; safe to run at any time.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

BOLD=$(tput bold 2>/dev/null || true)
DIM=$(tput dim 2>/dev/null || true)
GREEN=$(tput setaf 2 2>/dev/null || true)
RED=$(tput setaf 1 2>/dev/null || true)
CYAN=$(tput setaf 6 2>/dev/null || true)
RST=$(tput sgr0 2>/dev/null || true)

elapsed_seconds() {
    # Parse "MM:SS" or "HH:MM:SS" or "d-HH:MM:SS" from ps elapsed.
    awk -F: '{
        if (NF == 3) { print $1*3600+$2*60+$3 }
        else if (NF == 2) { print $1*60+$2 }
        else { print $1 }
    }' <<< "$1"
}

print_section() {
    printf "\n${BOLD}%s${RST}\n" "$1"
}

# ============================================================================
# A1.4 status
# ============================================================================
print_section "=== A1.4 -- sigma-sweep on unified driver (K=10 R=1 lambda=5) ==="
A1_DIR=logs/local_a1_4_sigmasweep
A1_RESULT_DIR=experiments/15_unified_hybrid/results

a1_total_done=0
a1_total_running=0
for s in 0.01 0.03 0.05 0.10; do
    pid_file="$A1_DIR/sigma${s}.pid"
    log_file="$A1_DIR/sigma${s}.log"
    out_file="$A1_RESULT_DIR/unified_bc_lam-5.0_K-10_R-1_sigmasweep_lam5_sigma${s}.json"

    pid=$(cat "$pid_file" 2>/dev/null || echo "")
    if [[ -f "$out_file" ]]; then
        # Already complete -- read final rate from the summary JSON.
        rate=$(python3 -c "
import json
d = json.load(open('$out_file'))
ns = d.get('n_success', 0)
nt = d.get('n_total', 60)
print(f'{ns}/{nt} = {100*ns/max(nt,1):.1f}%')" 2>/dev/null || echo "?")
        printf "  sigma=${BOLD}%s${RST}  ${GREEN}DONE${RST}  ${rate}\n" "$s"
        a1_total_done=$((a1_total_done + 1))
    elif [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        et=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')
        rollouts=$(grep -cE "demo_[0-9]+#[0-9]" "$log_file" 2>/dev/null || echo 0)
        successes=$(grep -cE "success=True" "$log_file" 2>/dev/null || echo 0)
        if [[ "$rollouts" -gt 0 ]]; then
            sec=$(elapsed_seconds "$et")
            avg=$((sec / rollouts))
            eta_min=$(( avg * (60 - rollouts) / 60 ))
            cur_rate=$(awk "BEGIN { printf \"%.1f%%\", 100*$successes/$rollouts }")
            printf "  sigma=${BOLD}%s${RST}  ${CYAN}RUN${RST}   %s/60  successes=%s (%s)  ETA=%smin\n" \
                "$s" "$rollouts" "$successes" "$cur_rate" "$eta_min"
        else
            printf "  sigma=${BOLD}%s${RST}  ${CYAN}RUN${RST}   warmup (et=%s)\n" "$s" "$et"
        fi
        a1_total_running=$((a1_total_running + 1))
    else
        printf "  sigma=${BOLD}%s${RST}  ${RED}DEAD${RST}  pid=%s no JSON yet\n" "$s" "$pid"
    fi
done
printf "  ${DIM}summary: %s done, %s running${RST}\n" "$a1_total_done" "$a1_total_running"

# ============================================================================
# A2 status
# ============================================================================
print_section "=== A2 -- V_phi training (Branch A Task A2) ==="
A2_DIR=logs/local_a2_vphi_train
A2_OUT=experiments/16_value_baseline/results/V_phi.pt
vpid=$(cat "$A2_DIR/pid" 2>/dev/null || echo "")
if [[ -f "$A2_OUT" ]]; then
    printf "  ${GREEN}V_phi.pt EXISTS${RST}  at %s\n" "$A2_OUT"
    if [[ -f "${A2_OUT%.pt}.meta.json" ]]; then
        meta=$(python3 -c "
import json
m = json.load(open('${A2_OUT%.pt}.meta.json'))
print(f\"    RMSE/|mean(R)| = {m.get('rmse_over_abs_mean_R','?'):.3f}\"
      f\"    BC success {m.get('n_success','?')}/{m.get('n_trajs','?')}\"
      f\"    final_loss={m.get('final_loss','?'):.4f}\")" 2>/dev/null || echo "    (meta unreadable)")
        echo "$meta"
    fi
elif [[ -n "$vpid" ]] && kill -0 "$vpid" 2>/dev/null; then
    et=$(ps -p "$vpid" -o etime= 2>/dev/null | tr -d ' ')
    rollouts=$(grep -cE "rollouts [0-9]+/" "$A2_DIR/train.log" 2>/dev/null || echo 0)
    last=$(grep -vE "robosuite (WARN|INFO)" "$A2_DIR/train.log" 2>/dev/null | tail -1)
    printf "  ${CYAN}V_phi training RUN${RST}  elapsed=%s  progress checkpoints=%s\n" \
        "$et" "$rollouts"
    printf "  ${DIM}%s${RST}\n" "$last"
else
    printf "  ${RED}V_phi training NOT RUNNING${RST}  pid=%s and no V_phi.pt yet\n" "$vpid"
    if [[ -f "$A2_DIR/train.log" ]]; then
        printf "  ${DIM}log tail:${RST}\n"
        grep -vE "robosuite (WARN|INFO)" "$A2_DIR/train.log" | tail -3 | sed 's/^/    /'
    fi
fi

# ============================================================================
# A2 eval status (only meaningful once V_phi.pt exists)
# ============================================================================
print_section "=== A2 -- pure-MBD + V_phi eval (the acceptance gate) ==="
A2_EVAL_OUT=experiments/16_value_baseline/results/_pure_mbd_with_vphi.json
A2_EVAL_OUT_ALT=experiments/16_value_baseline/results/unified_bc_lam-0.0_K-10_R-0_pure_mbd_with_vphi.json
if [[ -f "$A2_EVAL_OUT_ALT" ]]; then
    rate=$(python3 -c "
import json
d = json.load(open('$A2_EVAL_OUT_ALT'))
ns = d.get('n_success', 0)
nt = d.get('n_total', 60)
print(f'{ns}/{nt} = {100*ns/max(nt,1):.1f}%')" 2>/dev/null || echo "?")
    printf "  ${GREEN}A2 eval DONE${RST}  %s  (vs acceptance bar: >=5/60 supports H2)\n" "$rate"
elif [[ -f experiments/16_value_baseline/results/V_phi.pt ]]; then
    printf "  ${DIM}V_phi.pt ready; waiting for A1.4 to free up M4 Max cores before launching eval.${RST}\n"
    printf "  Launch command (run after A1.4 done):\n"
    printf "    ${BOLD}python experiments/15_unified_hybrid/run_can.py \\\\${RST}\n"
    printf "      --K 10 --mpc-replans 0 --lambda-prior 0 \\\\\n"
    printf "      --sigma-max 0.05 --M 64 --prior bc --no-warm-start \\\\\n"
    printf "      --value-baseline-ckpt experiments/16_value_baseline/results/V_phi.pt \\\\\n"
    printf "      --n-demos 20 --n-rollouts 3 --seed 0 --n-workers 2 \\\\\n"
    printf "      --output-dir experiments/16_value_baseline/results/ \\\\\n"
    printf "      --output-name-suffix \"_pure_mbd_with_vphi\"\n"
else
    printf "  ${DIM}waiting for V_phi.pt to land first${RST}\n"
fi

echo ""
