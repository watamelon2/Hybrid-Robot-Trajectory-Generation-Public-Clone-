---
description: Launch a hardened long-running evaluation slate (robosuite/run_can.py) in screen
---

# Skill: launch-slate

Use to start a multi-cell overnight eval slate so it survives disconnects,
SIGTERMs from parallel agents, and partial reruns.

## When to use
- Any `experiments/15_unified_hybrid/slate_*.sh` or similar long run.

## Hardware sizing (Apple Silicon)
- One robosuite worker ≈ 1 core. Leave 2 cores for OS / screen / I/O.
  - 16-core Mac: `N_WORKERS=14`.  10-core Mac: `N_WORKERS=8`.
- **Run slates sequentially.** Parallel slates thrash; throughput collapses
  ~30%. The launcher should refuse to start if another slate `screen` is
  alive.
- **MPS rule:** never run two MPS (GPU) trainings at once on M-series —
  combined throughput drops to ~14% (no NVIDIA-MPS equivalent). CPU eval
  in parallel with one MPS training IS fine (different compute units).

## Launcher hardening (the "v2" pattern — port to every slate script)
- Do NOT use `set -e` (one bad cell shouldn't kill the whole slate).
- Trap SIGTERM so a parallel agent's `pkill -f run_can.py` cleanup doesn't
  abort the slate: `trap 'echo "ignoring SIGTERM"' TERM`.
- Skip-existing: if a cell's output JSON already exists, skip it (makes
  reruns idempotent after a crash).
- Avoid bash 3.2 `awk "BEGIN{printf ... $var}"` for timing math (it
  deadlocks reading /dev/tty). Use bash arithmetic or `bc` instead.

## Steps
1. Confirm no other slate is running: `screen -ls` (kill stragglers only
   intentionally).
2. Launch detached:
   ```bash
   screen -dmS slate_x bash experiments/15_unified_hybrid/slate_x_YYYY_MM_DD.sh
   ```
3. Monitor:
   ```bash
   bash experiments/15_unified_hybrid/results/<slate_dir>/monitor.sh
   tail -f experiments/15_unified_hybrid/results/<slate_dir>/run.log
   ```
4. Per-core check (install if needed): `htop` (per-core bars) or
   `asitop` (P/E-core split on Apple Silicon).

## Verify
- `run.log` shows cells advancing and an "ALL DONE" marker at the end.
- Result JSONs land under the slate's `results/<slate_dir>/`.
