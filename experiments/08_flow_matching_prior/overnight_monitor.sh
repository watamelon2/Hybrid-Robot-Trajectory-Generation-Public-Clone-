#!/usr/bin/env bash
# Cheap status dump for the overnight queue.  Run anytime with
#   bash experiments/08_flow_matching_prior/overnight_monitor.sh
set -u

REPO=/Users/szh/Documents/SZH/Courseworks/ECE\ M237/Project/Hybrid-Robot-Trajectory-Generation
RUN_DIR="$REPO/experiments/08_flow_matching_prior/results/overnight_2026_05_24"

echo "============================================================"
echo " overnight queue status   $(date)"
echo "============================================================"

if [[ -f /tmp/overnight.pid ]]; then
    pid=$(cat /tmp/overnight.pid)
    if ps -p "$pid" > /dev/null 2>&1; then
        et=$(ps -o etime= -p "$pid" | tr -d ' ')
        echo "master script alive: pid=$pid elapsed=$et"
    else
        echo "master script (pid=$pid) NOT running"
    fi
else
    echo "no /tmp/overnight.pid file"
fi
echo

echo "--- running python processes ---"
ps -A -o pid,%cpu,%mem,etime,command \
    | grep -E "train_v6|train_dp|standalone_dp|run_can.py" \
    | grep -v grep || echo "(none)"
echo

echo "--- last 30 lines of queue.log ---"
tail -30 "$RUN_DIR/queue.log" 2>/dev/null || echo "(no log)"
echo

echo "--- stage-1 checkpoints ---"
ls -la "$REPO/experiments/08_flow_matching_prior/results/"v_theta_v6_unet_xl.pt \
       "$REPO/experiments/08_flow_matching_prior/results/"v_theta_dp_tobs1_h120.pt \
       "$REPO/experiments/08_flow_matching_prior/results/"v_theta_dp_tobs2_h120.pt \
       "$REPO/experiments/08_flow_matching_prior/results/"v_theta_dp_tobs2_h16.pt \
       2>/dev/null | awk '{print $5"  "$NF}'
echo

echo "--- stage-2 standalone evals ---"
for d in v6_unet_xl dp_tobs1_h120 dp_tobs2_h120 dp_tobs2_h16_cl; do
    f="$RUN_DIR/standalone_$d/standalone_dp.json"
    if [[ -f "$f" ]]; then
        succ=$(python3 -c "import json; j=json.load(open('$f')); print(f\"{j['n_succ']}/{j['n']}  rate={100*j['succ_rate']:.1f}%\")" 2>/dev/null)
        echo "  $d: $succ"
    elif [[ -f "$RUN_DIR/standalone_$d/run.log" ]]; then
        last=$(tail -1 "$RUN_DIR/standalone_$d/run.log" 2>/dev/null)
        echo "  $d: running -> $last"
    fi
done
echo

echo "--- stage-3 MBD evals ---"
for d in v6_unet_xl dp_tobs1_h120 dp_tobs2_h120; do
    if [[ -f "$RUN_DIR/mbd_$d/done" ]]; then
        echo "  $d: done"
    elif [[ -d "$RUN_DIR/mbd_$d" ]]; then
        last=$(tail -1 "$RUN_DIR/mbd_$d/run.log" 2>/dev/null)
        echo "  $d: running -> $last"
    fi
done
echo

echo "--- last git commits on this branch ---"
git -C "$REPO" log --oneline -5
