#!/usr/bin/env bash
# Quick monitor for Slate F (May 20 overnight) + follow-up retry.
# Prints a snapshot; wrap in `watch -n 30` for a live view.
#
# Layout assumes the launcher logs to ./run.log next to this script
# and that the JSON cell outputs land in this same directory.  Also
# watches for the `slate_f_retry` screen (Square budget-retry that
# auto-launches when slate_f finishes).

set -u
# NOTE: deliberately no `pipefail`.  macOS `screen -ls` returns
# exit code 1 even when screens *are* listed (BSD quirk vs GNU
# screen), and pipefail would make the screen-detection branch
# always fall through to NOT FOUND.

# ---- config ----
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/run.log"
RETRY_DIR="$HERE/../slate_f_square_retry_2026_05_20"
RETRY_LOG="$RETRY_DIR/run.log"
RETRY_WATCH_LOG="$RETRY_DIR/watcher.log"
EXPECTED_CELLS=13       # F1=6 lambdas + F2=3 + F3=4

# ---- screen status ----
echo "============================================================"
echo "  Slate F monitor      $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"
for NAME in slate_f slate_f_retry; do
  if screen -ls 2>/dev/null | grep -q "\.${NAME}[[:space:]]"; then
    PID=$(screen -ls 2>/dev/null | grep "\.${NAME}[[:space:]]" | awk -F'.' '{print $1}' | tr -d '[:space:]')
    printf "  screen %-15s RUNNING (pid %s)\n" "${NAME}:" "${PID}"
  else
    printf "  screen %-15s not running\n" "${NAME}:"
  fi
done

# ---- log basics ----
if [[ ! -f "$LOG" ]]; then
  echo "  run.log: NOT YET CREATED at $LOG"
  exit 0
fi
FIRST=$(head -1 "$LOG" | tr -d '\n')
LAST_MOD=$(stat -f '%Sm' -t '%H:%M:%S' "$LOG" 2>/dev/null || stat -c '%y' "$LOG" 2>/dev/null | cut -d. -f1)
echo "  log start:   $FIRST"
echo "  last write:  $LAST_MOD"

# ---- cell progress + cell-based ETA ----
N_CELLS_DONE=$(ls -1 "$HERE"/unified_demo_*.json 2>/dev/null | grep -v summary | wc -l | tr -d ' ')
printf "  cells done:  %d / %d\n" "$N_CELLS_DONE" "$EXPECTED_CELLS"

# Cell-based ETA: use mtimes of completed JSONs to estimate mean
# per-cell wall time, then extrapolate.
python3 - "$HERE" "$EXPECTED_CELLS" "$LOG" <<'PY' || true
import glob, os, sys, time, json
from datetime import datetime, timedelta

here, expected, log_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
jsons = sorted(
    [f for f in glob.glob(os.path.join(here, "unified_demo_*.json"))
     if "summary" not in f],
    key=lambda p: os.path.getmtime(p),
)
n_done = len(jsons)
if n_done == 0:
    print("  no cells finished yet; ETA unknown")
    sys.exit(0)

# Per-cell wall-time from JSON itself (preferred; matches what
# run_can.py reports).  Fall back to mtime-deltas if missing.
walls = []
for f in jsons:
    try:
        w = float(json.load(open(f)).get("wall_s", 0)) / 60.0
        if w > 0:
            walls.append(w)
    except Exception:
        pass
if not walls:
    # Fallback: gaps between consecutive JSON mtimes.
    mtimes = [os.path.getmtime(f) for f in jsons]
    log_start_m = os.path.getmtime(log_path)
    walls = [
        (mtimes[i] - (mtimes[i - 1] if i > 0 else log_start_m)) / 60.0
        for i in range(len(mtimes))
    ]

mean_min = sum(walls) / len(walls)
remaining = expected - n_done
eta_min = mean_min * remaining
print(f"  mean per-cell: {mean_min:5.1f} min   "
      f"(latest cell: {walls[-1]:.1f} min)")
if remaining > 0:
    finish = datetime.now() + timedelta(minutes=eta_min)
    print(f"  ETA remaining: {eta_min:5.1f} min  "
          f"(~{finish.strftime('%H:%M')} {finish.tzname() or ''})")
else:
    print("  ETA: all cells done")
PY

# ---- per-cell results table (read JSONs) ----
echo "------------------------------------------------------------"
echo "  per-cell results  (lambda  prior            sigma  succ   mean_r  max_r)"
echo "------------------------------------------------------------"
python3 - "$HERE" <<'PY' || true
import glob, os, sys, json
here = sys.argv[1]
files = [f for f in glob.glob(os.path.join(here, "unified_demo_*.json"))
         if "summary" not in f]
rows = []
for f in files:
    j = json.load(open(f))
    base = os.path.basename(f)
    # Sigma is not in the JSON header; parse from filename suffix
    # (e.g., "_F2_square_1nn_lam1_sigma0.05") if present.
    sig = ""
    if "sigma" in base:
        try:
            sig = base.split("sigma")[1].split("_")[0].rstrip(".json")
        except Exception:
            sig = ""
    n_t = j["n_total"]
    n_s = j["n_success"]
    rate = j["success_rate"] * 100
    rs = [r["final_reward"] for r in j["per_rollout"]]
    mean_r = sum(rs) / len(rs)
    max_r = max(rs)
    rows.append((
        os.path.getmtime(f), j["lambda_prior"], j["prior"], sig,
        f"{n_s}/{n_t}", rate, mean_r, max_r,
    ))
rows.sort(key=lambda r: r[0])  # chronological
for _, lam, prior, sig, succ, rate, mean_r, max_r in rows:
    sig_str = sig if sig else "0.25"
    print(f"  {lam:>5.2f}  {prior:<18}  {sig_str:>5}  {succ:>6}  "
          f"{rate:>4.1f}%  {mean_r:>+7.2f}  {max_r:>+7.2f}")
if not rows:
    print("  (none yet)")
PY

# ---- recent activity ----
echo "------------------------------------------------------------"
echo "  recent [exp15] lines (last 5)"
echo "------------------------------------------------------------"
grep '^\[exp15\]' "$LOG" | tail -5 || echo "  (no [exp15] output yet)"

# ---- retry follow-up status (if any) ----
if [[ -f "$RETRY_WATCH_LOG" ]] || [[ -f "$RETRY_LOG" ]]; then
  echo "------------------------------------------------------------"
  echo "  square retry (auto-launches when slate_f finishes)"
  echo "------------------------------------------------------------"
  if [[ -f "$RETRY_LOG" ]] && [[ -s "$RETRY_LOG" ]]; then
    echo "  retry started; last 3 lines from its run.log:"
    grep '^\[exp15\]' "$RETRY_LOG" | tail -3 \
      || tail -3 "$RETRY_LOG"
  elif [[ -f "$RETRY_WATCH_LOG" ]]; then
    echo "  retry not started yet; last watcher line:"
    tail -1 "$RETRY_WATCH_LOG"
  fi
fi

# ---- final marker ----
if grep -q 'ALL DONE' "$LOG"; then
  echo "============================================================"
  echo "  ALL DONE marker present in slate_f run.log."
  echo "============================================================"
fi
