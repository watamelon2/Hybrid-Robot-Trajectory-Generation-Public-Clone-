#!/usr/bin/env bash
# Quick monitor for Slate G (May 21 overnight) + Slate H follow-up.
# Prints a snapshot; wrap in `watch -n 60` for a live view.
#
# Tracks seven screens:
#   slate_g           -- the 18-cell main slate (this directory)
#   slate_g_aux       -- 8-cell "crack Square" parallel companion
#   slate_h_watcher   -- polls for slate_g to die, then launches Slate H
#   slate_h           -- the 7-cell follow-up (auto-fires post-G)
#   slate_i_watcher   -- polls for slate_h to die, then launches Slate I
#   slate_i           -- the 13-cell C1 contrastive-prior slate
#   slate_j           -- 4-cell tool_hang+transport recovery (manual fire)
#
# Cell outputs land as unified_*.json next to this script (slate_g)
# or in ../slate_h_2026_05_21/ (slate_h).

set -u
# Deliberately no `pipefail` -- macOS `screen -ls` returns exit 1
# even when screens ARE listed (BSD quirk).

# ---- config ----
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/run.log"
AUX_DIR="$HERE/../slate_g_aux_2026_05_21"
AUX_LOG="$AUX_DIR/run.log"
SLATE_H_DIR="$HERE/../slate_h_2026_05_21"
SLATE_H_LOG="$SLATE_H_DIR/run.log"
SLATE_H_WATCH_LOG="$SLATE_H_DIR/watcher.log"
SLATE_I_DIR="$HERE/../slate_i_2026_05_22"
SLATE_I_LOG="$SLATE_I_DIR/run.log"
SLATE_I_WATCH_LOG="$SLATE_I_DIR/watcher.log"
SLATE_J_DIR="$HERE/../slate_j_2026_05_22"
SLATE_J_LOG="$SLATE_J_DIR/run.log"
EXPECTED_G=18   # G1=6 + G2=6 + G3=3 + G4=3
EXPECTED_AUX=8  # AUX1=4 sigma + AUX2=4 beta
EXPECTED_H=7    # H1=1 + H3=3 + H4=3   (H2 deferred)
EXPECTED_I=13   # I1=6 lam + I2=4 sigma + I3=3 lam (Can control)
EXPECTED_J=4    # J1=2 tool_hang lam + J2=2 transport lam (dense-reward recovery)

# ---- helpers ----
print_screen_status() {
  local name="$1"
  if screen -ls 2>/dev/null | grep -q "\.${name}[[:space:]]"; then
    local pid
    pid=$(screen -ls 2>/dev/null | grep "\.${name}[[:space:]]" \
          | awk -F'.' '{print $1}' | tr -d '[:space:]')
    printf "  screen %-18s RUNNING (pid %s)\n" "${name}:" "${pid}"
  else
    printf "  screen %-18s not running\n" "${name}:"
  fi
}

# ---- banner ----
echo "============================================================"
echo "  Slate G + H monitor   $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"
print_screen_status "slate_g"
print_screen_status "slate_g_aux"
print_screen_status "slate_h_watcher"
print_screen_status "slate_h"
print_screen_status "slate_i_watcher"
print_screen_status "slate_i"
print_screen_status "slate_j"

# ---- CPU load snapshot ----
LOAD=$(uptime | awk -F'load averages: ' '{print $2}')
CPU=$(top -l 1 -n 0 2>/dev/null | grep '^CPU usage:' | head -1)
printf "  load:               %s\n" "$LOAD"
printf "  %s\n" "$CPU"

# ---- Slate G section ----
echo ""
echo "------------------------------------------------------------"
echo "  Slate G  (18 cells: G1=Lift_lam_sweep G2=softnn_beta G3=schedule G4=square_acthist)"
echo "------------------------------------------------------------"
if [[ ! -f "$LOG" ]]; then
  echo "  run.log: NOT YET CREATED at $LOG"
else
  FIRST=$(head -1 "$LOG" | tr -d '\n')
  LAST_MOD=$(stat -f '%Sm' -t '%H:%M:%S' "$LOG" 2>/dev/null \
             || stat -c '%y' "$LOG" 2>/dev/null | cut -d. -f1)
  echo "  log start:   $FIRST"
  echo "  last write:  $LAST_MOD"

  N_DONE=$(ls -1 "$HERE"/unified_demo_*.json 2>/dev/null \
           | grep -v summary | wc -l | tr -d ' ')
  printf "  cells done:  %d / %d\n" "$N_DONE" "$EXPECTED_G"

  # In-flight cell progress: count [exp15] rollout lines since the
  # last "starting" marker.  Helps see "we're in G1 lam=2 at 47/60".
  IN_FLIGHT=$(awk '
    /^=== / && / starting ===/ { tag=$0; rollouts=0; next }
    /^\[exp15\] (prior|lift|can|square|tool_hang|transport).*reward=/ { rollouts++ }
    END {
      if (tag != "") {
        sub(/^=== /, "", tag); sub(/ starting ===$/, "", tag);
        gsub(/[[:space:]]+/, " ", tag);
        printf "  in-flight:   %s  (%d rollouts since start)\n", tag, rollouts;
      }
    }
  ' "$LOG")
  if [[ -n "$IN_FLIGHT" ]]; then
    echo "$IN_FLIGHT"
  fi

  # Per-cell wall-time + ETA from JSON wall_s fields.
  python3 - "$HERE" "$EXPECTED_G" "$LOG" <<'PY' || true
import glob, os, sys, json
from datetime import datetime, timedelta

here, expected, log_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
jsons = sorted(
    [f for f in glob.glob(os.path.join(here, "unified_demo_*.json"))
     if "summary" not in f],
    key=os.path.getmtime,
)
n_done = len(jsons)
if n_done == 0:
    print("  no cells finished yet; ETA from log timestamps below")
    sys.exit(0)
walls = []
for f in jsons:
    try:
        w = float(json.load(open(f)).get("wall_s", 0)) / 60.0
        if w > 0:
            walls.append(w)
    except Exception:
        pass
if not walls:
    sys.exit(0)
mean_min = sum(walls) / len(walls)
remaining = expected - n_done
print(f"  mean per-cell: {mean_min:5.1f} min   "
      f"(latest cell: {walls[-1]:.1f} min)")
if remaining > 0:
    finish = datetime.now() + timedelta(minutes=mean_min * remaining)
    print(f"  ETA remaining: {mean_min * remaining:5.1f} min  "
          f"(~{finish.strftime('%H:%M')} local)")
else:
    print("  Slate G: all cells done")
PY

  # ---- per-cell results table ----
  echo ""
  echo "  Slate G per-cell  (prior              lam   beta  sched      hist  succ    rate)"
  python3 - "$HERE" <<'PY' || true
import glob, os, sys, json
here = sys.argv[1]
files = [f for f in glob.glob(os.path.join(here, "unified_demo_*.json"))
         if "summary" not in f]
rows = []
for f in files:
    j = json.load(open(f))
    base = os.path.basename(f)
    p = j["params"]
    rs = [r["final_reward"] for r in j["per_rollout"]]
    mean_r = sum(rs) / len(rs) if rs else 0.0
    rows.append((
        os.path.getmtime(f),
        j["prior"], float(j["lambda_prior"]),
        p.get("retrieval_beta", 1.0),
        p.get("lambda_schedule", "constant"),
        p.get("action_history_len", 0),
        j["n_success"], j["n_total"], j["success_rate"] * 100,
        mean_r,
    ))
rows.sort(key=lambda r: r[0])
for _, prior, lam, beta, sched, hist, n_s, n_t, rate, mean_r in rows:
    print(f"  {prior:<18} {lam:>4.1f}  {beta:>4.2f}  {sched:<10} {hist:>3}  "
          f"{n_s:>3}/{n_t:<3} {rate:>5.1f}%  mean_r={mean_r:+6.2f}")
if not rows:
    print("  (none yet)")
PY

  # ---- recent activity ----
  echo ""
  echo "  recent [exp15] lines (last 4)"
  grep '^\[exp15\]' "$LOG" | tail -4 || echo "  (no [exp15] output yet)"
fi

# ---- Slate G-aux section ----
echo ""
echo "------------------------------------------------------------"
echo "  Slate G-aux  (8 cells: AUX1=Square_sigma_hires AUX2=Square_softnn_beta)"
echo "------------------------------------------------------------"
if [[ -f "$AUX_LOG" ]]; then
  AUX_FIRST=$(head -1 "$AUX_LOG" | tr -d '\n')
  AUX_LAST_MOD=$(stat -f '%Sm' -t '%H:%M:%S' "$AUX_LOG" 2>/dev/null \
                 || stat -c '%y' "$AUX_LOG" 2>/dev/null | cut -d. -f1)
  echo "  log start:   $AUX_FIRST"
  echo "  last write:  $AUX_LAST_MOD"
  N_DONE_AUX=$(ls -1 "$AUX_DIR"/unified_demo_*.json 2>/dev/null \
               | grep -v summary | wc -l | tr -d ' ')
  printf "  cells done:  %d / %d\n" "$N_DONE_AUX" "$EXPECTED_AUX"
  AUX_IN_FLIGHT=$(awk '
    /^=== / && / starting ===/ { tag=$0; rollouts=0; next }
    /^\[exp15\] (prior|lift|can|square|tool_hang|transport).*reward=/ { rollouts++ }
    END {
      if (tag != "") {
        sub(/^=== /, "", tag); sub(/ starting ===$/, "", tag);
        gsub(/[[:space:]]+/, " ", tag);
        printf "  in-flight:   %s  (%d rollouts since start)\n", tag, rollouts;
      }
    }
  ' "$AUX_LOG")
  if [[ -n "$AUX_IN_FLIGHT" ]]; then
    echo "$AUX_IN_FLIGHT"
  fi
  # Per-cell summary table (compact)
  python3 - "$AUX_DIR" <<'PY' || true
import glob, os, sys, json
here = sys.argv[1]
files = [f for f in glob.glob(os.path.join(here, "unified_demo_*.json"))
         if "summary" not in f]
rows = []
for f in files:
    j = json.load(open(f))
    p = j["params"]
    rs = [r["final_reward"] for r in j["per_rollout"]]
    rows.append((
        os.path.getmtime(f), j["prior"], float(j["lambda_prior"]),
        p.get("retrieval_beta", 1.0),
        float(j.get("sigma_max", p.get("sigma_max", 0.25))),
        j["n_success"], j["n_total"], j["success_rate"] * 100,
        sum(rs) / len(rs) if rs else 0.0,
    ))
rows.sort(key=lambda r: r[0])
for _, prior, lam, beta, sig, n_s, n_t, rate, mean_r in rows:
    print(f"  {prior:<18} lam={lam:.1f} beta={beta:.2f} sig={sig:.2f}  "
          f"{n_s:>3}/{n_t:<3} {rate:>5.1f}%  mean_r={mean_r:+6.2f}")
if not rows:
    print("  (none yet)")
PY
else
  echo "  Slate G-aux not started."
fi

# ---- Slate H section ----
echo ""
echo "------------------------------------------------------------"
echo "  Slate H  (7 cells: H1=tool_hang  H3=A1-on-Can  H4=A1-on-Lift  -- transport DEFERRED)"
echo "------------------------------------------------------------"
if [[ -f "$SLATE_H_LOG" ]] && [[ -s "$SLATE_H_LOG" ]]; then
  echo "  Slate H IS RUNNING."
  N_DONE_H=$(ls -1 "$SLATE_H_DIR"/unified_demo_*.json 2>/dev/null \
             | grep -v summary | wc -l | tr -d ' ')
  printf "  cells done:  %d / %d\n" "$N_DONE_H" "$EXPECTED_H"
  echo "  last 3 [exp15] lines:"
  grep '^\[exp15\]' "$SLATE_H_LOG" | tail -3 || echo "  (no [exp15] output yet)"
elif [[ -f "$SLATE_H_WATCH_LOG" ]]; then
  echo "  Slate H not started yet; watcher status:"
  tail -2 "$SLATE_H_WATCH_LOG"
else
  echo "  Slate H watcher log not found yet."
fi

# ---- Slate I section ----
echo ""
echo "------------------------------------------------------------"
echo "  Slate I  (13 cells: I1=C1-Square-lam I2=C1-Square-sigma I3=C1-Can-control)"
echo "------------------------------------------------------------"
if [[ -f "$SLATE_I_LOG" ]] && [[ -s "$SLATE_I_LOG" ]]; then
  echo "  Slate I IS RUNNING."
  N_DONE_I=$(ls -1 "$SLATE_I_DIR"/unified_demo_*.json 2>/dev/null \
             | grep -v summary | wc -l | tr -d ' ')
  printf "  cells done:  %d / %d\n" "$N_DONE_I" "$EXPECTED_I"
  echo "  last 3 [exp15] lines:"
  grep '^\[exp15\]' "$SLATE_I_LOG" | tail -3 || echo "  (no [exp15] output yet)"
elif [[ -f "$SLATE_I_WATCH_LOG" ]]; then
  echo "  Slate I not started yet; watcher status:"
  tail -2 "$SLATE_I_WATCH_LOG"
else
  echo "  Slate I watcher log not found yet."
fi

# ---- ALL DONE markers ----
echo ""
if grep -q 'ALL DONE' "$LOG" 2>/dev/null; then
  echo "  >>> Slate G: ALL DONE marker present in run.log."
fi
if [[ -f "$AUX_LOG" ]] && grep -q 'ALL DONE' "$AUX_LOG" 2>/dev/null; then
  echo "  >>> Slate G-aux: ALL DONE marker present in run.log."
fi
if [[ -f "$SLATE_H_LOG" ]] && grep -q 'ALL DONE' "$SLATE_H_LOG" 2>/dev/null; then
  echo "  >>> Slate H: ALL DONE marker present in run.log."
fi
if [[ -f "$SLATE_I_LOG" ]] && grep -q 'ALL DONE' "$SLATE_I_LOG" 2>/dev/null; then
  echo "  >>> Slate I: ALL DONE marker present in run.log."
fi

# ---- Slate J section (manual-fire, not auto-queued) ----
if [[ -f "$SLATE_J_LOG" ]] && [[ -s "$SLATE_J_LOG" ]]; then
  echo ""
  echo "------------------------------------------------------------"
  echo "  Slate J  (4 cells: J1=tool_hang-lam J2=transport-lam, dense-reward recovery)"
  echo "------------------------------------------------------------"
  echo "  Slate J IS RUNNING."
  N_DONE_J=$(ls -1 "$SLATE_J_DIR"/unified_demo_*.json 2>/dev/null \
             | grep -v summary | wc -l | tr -d ' ')
  printf "  cells done:  %d / %d\n" "$N_DONE_J" "$EXPECTED_J"
  echo "  last 3 [exp15] lines:"
  grep '^\[exp15\]' "$SLATE_J_LOG" | tail -3 || echo "  (no [exp15] output yet)"
  if grep -q 'ALL DONE' "$SLATE_J_LOG" 2>/dev/null; then
    echo "  >>> Slate J: ALL DONE marker present in run.log."
  fi
fi

echo "============================================================"
