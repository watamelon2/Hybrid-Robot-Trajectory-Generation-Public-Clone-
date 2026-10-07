#!/usr/bin/env bash
# Pretty-print the current scoreboard for the overnight queue.
# Parses every standalone_dp.json and unified_summary_*.json that exists
# in results/overnight_2026_05_24/, including in-progress runs (best-effort
# by re-grepping run.log if no JSON yet).
#
# Usage:  bash experiments/08_flow_matching_prior/overnight_results.sh
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_DIR="$REPO/experiments/08_flow_matching_prior/results/overnight_2026_05_24"
PY=/Users/szh/anaconda3/envs/hybrid-traj/bin/python

if [[ ! -d "$RUN_DIR" ]]; then
    echo "no run dir: $RUN_DIR" >&2
    exit 1
fi

"$PY" - <<PYEOF "$RUN_DIR"
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])

def parse_runlog_partial(log_path: Path) -> tuple[int, int]:
    """Count rollouts done + successes from a still-streaming run.log."""
    if not log_path.exists():
        return 0, 0
    txt = log_path.read_text(errors="ignore")
    done = len(re.findall(r"demo_\d+#\d+", txt))
    succ = txt.count("success=True")
    return done, succ

print()
print("=" * 78)
print(f" overnight queue scoreboard   {run_dir.name}")
print("=" * 78)
print()
print("STANDALONE (prior only, no MBD inner search):")
print(f"  {'name':30s}  {'success':>14s}  {'wall':>8s}  source")
print(f"  {'-'*30}  {'-'*14}  {'-'*8}  {'-'*8}")

stage2_rows = []
for d in sorted(run_dir.glob("standalone_*")):
    j = d / "standalone_dp.json"
    if j.exists():
        b = json.loads(j.read_text())
        n_succ = int(b.get("n_succ", 0))
        n = int(b.get("n", 0))
        wall_s = float(b.get("wall_s", 0.0))
        rate = 100 * n_succ / max(n, 1)
        stage2_rows.append((d.name, n_succ, n, rate, wall_s, "json"))
    else:
        done, succ = parse_runlog_partial(d / "run.log")
        if done > 0:
            rate = 100 * succ / done
            stage2_rows.append((d.name, succ, done, rate, 0.0, "runlog"))

for name, n_succ, n, rate, wall_s, src in stage2_rows:
    label = name.replace("standalone_", "")
    rate_str = f"{n_succ:>2d}/{n:<2d} = {rate:>4.1f}%"
    wall_str = f"{wall_s/60:>6.1f}m" if wall_s > 0 else "      -"
    print(f"  {label:30s}  {rate_str:>14s}  {wall_str:>8s}  {src}")

print()
print("MBD (prior + score-bridge cost shaping, K=10, R=1, lambda=1):")
print(f"  {'name':30s}  {'success':>14s}  {'wall':>8s}  source")
print(f"  {'-'*30}  {'-'*14}  {'-'*8}  {'-'*8}")

stage3_rows = []
for d in sorted(run_dir.glob("mbd_*")):
    summaries = list(d.glob("unified_summary_*.json"))
    if summaries:
        b = json.loads(summaries[0].read_text())
        # run_can.py writes:  per_lambda: [{n_total, n_success, success_rate, wall_s}, ...]
        per_lam = b.get("per_lambda") or [b]
        row = per_lam[0] if isinstance(per_lam, list) and per_lam else b
        n_succ = int(row.get("n_success", row.get("n_succ", 0)))
        n = int(row.get("n_total", row.get("n", 0)))
        wall_s = float(row.get("wall_s", b.get("wall_s", 0.0)))
        rate = 100 * n_succ / max(n, 1)
        stage3_rows.append((d.name, n_succ, n, rate, wall_s, "json"))
    else:
        done, succ = parse_runlog_partial(d / "run.log")
        if done > 0:
            rate = 100 * succ / done
            stage3_rows.append((d.name, succ, done, rate, 0.0, "runlog"))

for name, n_succ, n, rate, wall_s, src in stage3_rows:
    label = name.replace("mbd_", "")
    rate_str = f"{n_succ:>2d}/{n:<2d} = {rate:>4.1f}%"
    wall_str = f"{wall_s/60:>6.1f}m" if wall_s > 0 else "      -"
    print(f"  {label:30s}  {rate_str:>14s}  {wall_str:>8s}  {src}")

print()
# Headline deltas if both standalone + mbd exist for the same prior
print("DELTAS (standalone - MBD):")
s2_map = {n.replace("standalone_", ""): rate for n, _, _, rate, _, _ in stage2_rows}
s3_map = {n.replace("mbd_", ""): rate for n, _, _, rate, _, _ in stage3_rows}
common = sorted(set(s2_map) & set(s3_map))
if not common:
    print("  (no overlapping priors yet)")
else:
    for k in common:
        d = s2_map[k] - s3_map[k]
        verdict = "MBD helps" if d < 0 else ("MBD hurts" if d > 0 else "MBD tie")
        print(f"  {k:30s}  standalone {s2_map[k]:>5.1f}%  -  MBD {s3_map[k]:>5.1f}%  =  {-d:>+6.1f}pp   ({verdict})")

print()
PYEOF
