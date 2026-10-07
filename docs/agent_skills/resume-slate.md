---
description: Resume a paused (SIGSTOP'd) evaluation slate and its multiprocessing workers
---

# Skill: resume-slate

Use to wake a slate that was paused (e.g. to free CPU for an event). A
paused slate is the parent `run_can.py` plus its multiprocessing workers
all in state `T+` (stopped).

## When to use
- A prior session SIGSTOP'd a slate (`pkill -STOP -f run_can.py`) and you
  need to continue it.

## Steps
1. Find the live process tree (PIDs change between sessions — always
   re-discover, don't trust an old PID):
   ```bash
   ps -A -o pid,ppid,stat,command | grep '[r]un_can.py'
   ```
2. Wake the parent, then its workers (use the parent PID from step 1):
   ```bash
   pkill -CONT -f "run_can.py"     # wakes parent
   pkill -CONT -P <parent_pid>     # wakes the workers + resource_tracker
   ```
3. Verify they are running (`S+`/`R`, not `T+`):
   ```bash
   ps -A -o pid,stat,command | grep '[r]un_can.py'
   ```
4. Re-attach monitoring:
   ```bash
   bash experiments/15_unified_hybrid/results/<slate_dir>/monitor.sh
   ```

## Gotchas
- Mid-cell progress before a pause is in-memory only; the in-flight cell
  resumes from where SIGCONT lands, not from a written checkpoint.
- If the slate's `screen` registered Dead, the watcher hand-off may have
  failed silently — relaunch with the launch-slate skill instead.
