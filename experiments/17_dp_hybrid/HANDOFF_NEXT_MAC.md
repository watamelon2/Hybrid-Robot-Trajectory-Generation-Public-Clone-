# Phase-C K-sweep handoff to the secondary Mac

**Branch**: `feat/dp-phase-c`
**Date**: 2026-05-29 → 2026-05-30 (Phase-D superseded)
**Wall budget**: ~6 hr unattended, single-machine CPU
**Goal**: Complete the K = 0 / 10 / 20 sweep with the corrected
`--mpc-replans 49` flag, write final results into the BRIEF, commit
+ push.

---

## ⚠️ 2026-05-30 update: Phase-D supersedes this handoff

The first run of `run_phase_c_ksweep.sh` (per this handoff) hit the
K=0 sanity gate at **0 / 50 = 0.0 %** test rate (vs the gate band
of 75 – 85 %).  Root cause: a chunk-boundary `env.reset()` in
`unified_loop.py:466-468` that wiped robosuite's controller state
at every chunk boundary.  Two-fix landing on commit `a58af13`:

1. `unified_loop.py:478` -- gate the chunk-boundary reset+restore
   on `params.K > 0` (so K=0 doesn't trigger spurious resets).
2. `run_can.py:555` -- change `--n-workers` default from 1 to 2
   (so MBD's perturbations run in subprocess workers that own
   forked envs, leaving the main env untouched at K > 0 too).

Full triage + fix proposal: `results/phase_d_2026_05_30/BRIEF.md`.
Empirical evidence (pre-fix smoke + post-fix smoke + audit + the
chunk-boundary env.reset() regression test):
`results/phase_d_2026_05_30/{BRIEF.md, AUDIT.md, smoke_post_fix/,
triage_n_workers/}` + `experiments/15_unified_hybrid/test_phase_d_reset_gate.py`.

**Post-fix K-sweep status (in-flight as of 2026-05-30 ~22:45 PDT
on the primary Mac)**: K=0 train rate at 6/7 = 86 % after 19 min,
in line with Phase-B standalone 80 %.  The sweep continues with
K=10 + K=20 if the K=0 test gate (50 fresh seeds) lands in
[75, 85] %.

The rest of this document still applies to the K-sweep procedure;
only the K=0 result interpretation changed (was BLOCKED on the
Phase-D bug; now expected to PASS the gate).

---

## TL;DR for the next agent

```bash
# 0. From the repo root, on the secondary Mac:
cd /path/to/HRTG-dp
git checkout feat/dp-phase-c
git pull --ff-only origin feat/dp-phase-c

# 1. Make sure the Phase-B ckpt blob is on the local FS (~486 MB).
#    See "Ckpt sync" section below if it isn't.

# 2. Activate the same conda env (or any env with the project deps).
conda activate hybrid-traj   # or whatever it's called on this Mac

# 3. Smoke (~30 sec) to verify the env + ckpt + obs schema.
bash experiments/17_dp_hybrid/run_phase_c_smoke.sh

# 4. Launch the K-sweep (~5-7 hr, runs unattended).
nohup bash experiments/17_dp_hybrid/run_phase_c_ksweep.sh \
    > experiments/17_dp_hybrid/results/phase_c_2026_05_29/launch.out 2>&1 &
echo "pid: $!"

# 5. Poll progress (see "Monitoring" section).

# 6. When the script exits (or all 3 K JSONs exist), run the
#    finalisation step (see "Finalisation").
```

---

## Context

### What's done

- **Phase-A** (Phase-3 ckpt re-eval under DP-aligned protocol) →
  GREEN at 26.7 % `ep0399` test rate; failed the GREEN threshold of
  50 %, so the **schema hypothesis** (`can` 46-D vs `can_dp` 23-D)
  became the focus.  See
  `experiments/17_dp_hybrid/results/phase_a_2026_05_28/BRIEF.md`.

- **Phase-B** (retrain DP prior on the 23-D `can_dp` schema with
  `H = 16`) → GREEN at **80 % test rate** on `ep0199` (40 / 50, 95 %
  CI ≈ 66.3 – 89.9 %).  Closes the schema hypothesis: **the 25 %
  ceiling of every pre-B run was an obs-schema bug, not a hybrid
  bug**.  See `experiments/17_dp_hybrid/results/phase_b_2026_05_29/`.

- **Phase-C plumbing**: the unified hybrid pipeline
  (`experiments/15_unified_hybrid/run_can.py` +
  `unified_loop.py`) was rewired to accept the Phase-B `can_dp`
  ckpt schema.  Two flags were added (`--n-test-rollouts` and
  `--test-start-seed`) to mirror Phase-A / Phase-B's 50-fresh-seed
  test bucket.  Pre-flight smoke is in
  `experiments/17_dp_hybrid/results/phase_c_2026_05_29/BRIEF.md`,
  "Pre-flight" section.

### What we just learned (the bug fix)

The first Phase-C K-sweep launch on the **primary Mac** (CPU,
2026-05-29 ~13:54 PDT) failed catastrophically: **0 / 60 = 0 %** at
K = 0, with `final_reward ≈ 0.04` and `steps_executed = 16` per
rollout.

Root cause: the BRIEF specified `--mpc-replans 0` to "match
standalone DP", but `unified_loop.py:228` reads
`n_chunks = mpc_replans + 1`, so `mpc_replans = 0` produces a
single-shot 1-chunk rollout that terminates after 16 steps,
**before the robot reaches the can**.  `--max-env-steps 400` is a
HARD CAP, never a target.

**Fix** (already committed to `feat/dp-phase-c`):
`--mpc-replans 49` → 50 chunks × 8 steps = 400 env steps with
re-planning every 8 actions, mirroring standalone-DP exactly.

Smoke at the fixed config (1 train + 1 test, `--mpc-replans 14`
→ 120-step rollouts) printed:

```
[exp15] --prior=flow ckpt task='can_dp'  obs_dim=23  (7 keys)
[exp15] prior=flow lam=0.5 R=14 K=0  1 train demo_0#0  reward=+3.67  success=False
[exp15] prior=flow lam=0.5 R=14 K=0  2 test  seed=100000  reward=+3.01  success=False
```

i.e. reward of 3.0+ (vs the bug's 0.04) = robot moving toward the
can.  The full 400-step K = 0 run was killed before completion on
the primary Mac due to slow CPU + the user's secondary-Mac handoff
decision.

### What needs to happen on this Mac

Run the full K-sweep (K = 0 / 10 / 20) at the fixed config and
record the results in
`experiments/17_dp_hybrid/results/phase_c_2026_05_29/BRIEF.md`'s
"Result" table.  Pre-registered decision rules (gate + sweep verdict)
are in that BRIEF's "Decision rule" section.

---

## Ckpt sync (only if `v_theta_dp_tobs2_h16_v3b_ep0199.pt` isn't local)

The Phase-B winner ckpt is **486 MB** and **not in git** (cost-
prohibitive to LFS).  The repo has a symlink at:

```
experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt
```

pointing to
`Hybrid-Robot-Trajectory-Generation/experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt`.
On the secondary Mac, this symlink target probably doesn't exist
yet.  Two options:

### Option A — gdown via Drive share (fastest if you have wifi)

1. Get the share URL from the user (the primary-Mac sync used
   `https://drive.google.com/drive/folders/1zzF_0-JyBENfZhPaOktt-UfIoUb_JTgx?usp=sharing`,
   set to "Anyone with the link · Viewer").  If that share is still
   active, reuse it; otherwise ask the user to re-share.

2. Run the sync helper (it auto-installs to the canonical destination):

   ```bash
   python experiments/08_flow_matching_prior/sync_ckpts_from_drive.py \
       --folder-url '<URL>' \
       --pattern v3b_ep0199
   ```

   Wall: ~5 – 10 min on wifi.  **Note**: the helper's `--pattern`
   filter is applied at copy time, not download time, so gdown will
   pull every `.pt` in the folder (~2.5 GB worth) into a tempdir
   first.  If you want to short-circuit, watch the tempdir and copy
   `v_theta_dp_tobs2_h16_v3b_ep0199.pt` out manually as soon as it
   reaches 486 MB, then `pkill -f gdown`.

### Option B — scp from the primary Mac

If both Macs are on the same network:

```bash
# On secondary Mac:
mkdir -p ~/Documents/SZH/Courseworks/ECE\ M237/Project/Hybrid-Robot-Trajectory-Generation/experiments/08_flow_matching_prior/results
scp <primary-mac-host>:'~/Documents/SZH/Courseworks/ECE\ M237/Project/Hybrid-Robot-Trajectory-Generation/experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt' \
    ~/Documents/SZH/Courseworks/ECE\ M237/Project/Hybrid-Robot-Trajectory-Generation/experiments/08_flow_matching_prior/results/
```

### Verify the symlink resolves

```bash
ls -laL experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt
# Expect: -rw-...  486M  <date>  ...
# NOT:    lrwx... (-> ...)  (= self-symlink, broken)
```

If it shows up as a symlink-to-symlink chain that resolves to
itself, **delete** the inner symlink in the main worktree
(`Hybrid-Robot-Trajectory-Generation/.../v3b_ep0199.pt`) and put
the actual blob there.  See "ckpt symlink design" at the bottom of
this doc for why.

---

## Step-by-step work plan

### 1. Smoke test (~30 sec)

Verifies: env builds, ckpt loads, obs schema matches (`can_dp`, 23-D),
the corrected `--mpc-replans` path produces non-trivial rewards.

```bash
bash experiments/17_dp_hybrid/run_phase_c_smoke.sh
```

Expected output (last 3 lines):

```
[exp15] --prior=flow ckpt task='can_dp'  obs_dim=23  (7 keys)
[exp15] prior=flow lam=0.5 R=14 K=0  1 train demo_0#0  reward=+3.<NN>  success=<bool>
[exp15] prior=flow lam=0.5 R=14 K=0  2 test  seed=100000  reward=+3.<NN>  success=<bool>
```

**STOP criteria**: any of

- `task='can'` (wrong schema; ckpt loaded the legacy path)
- `obs_dim=46` (wrong schema)
- reward < 1.0 (robot not moving — back to the bug regime)
- non-zero exit code

If any of these triggers, do NOT launch the full sweep.  Surface
the error to the user, attach the last 50 lines of stderr.

### 2. Full K-sweep launch (~5 – 7 hr, runs unattended)

```bash
mkdir -p experiments/17_dp_hybrid/results/phase_c_2026_05_29
nohup bash experiments/17_dp_hybrid/run_phase_c_ksweep.sh \
    > experiments/17_dp_hybrid/results/phase_c_2026_05_29/launch.out 2>&1 &
echo "ksweep pid: $!"
disown
```

The script:

- Runs **K = 0** first (~60 – 90 min on CPU).  When the K = 0 JSON
  lands, it computes the **sanity gate** (test rate ∈ [0.75, 0.85]).
  - **FAIL** → halts; no K > 0 runs.  See "Gate failure" section.
  - **PASS** → continues to K = 10 then K = 20.
- Is **idempotent**: any K with an existing
  `K{K}/unified_flow_lam-*.json` is skipped.  Safe to re-run after
  a crash / OS restart.
- Streams `[exp15]` lines to both `K{K}/run.log` and
  `results/phase_c_2026_05_29/ksweep.log`.

### 3. Monitoring (poll roughly every 30 min)

```bash
# How many rollouts completed so far?
for K in K0 K10 K20; do
    d=experiments/17_dp_hybrid/results/phase_c_2026_05_29/$K
    n=$(grep -cE '^\[exp15\].*reward=' "$d/run.log" 2>/dev/null || echo 0)
    echo "$K: $n / 60 rollouts"
done

# Is the process still alive?
pgrep -fl 'run_can|run_phase_c_ksweep' | head -3

# Last 15 log lines:
tail -15 experiments/17_dp_hybrid/results/phase_c_2026_05_29/ksweep.log
```

### 4. Gate decision (automatic, no user input needed)

The script computes the gate when K = 0 finishes.  Manual
verification:

```bash
python -c "
import json
d = json.load(open('experiments/17_dp_hybrid/results/phase_c_2026_05_29/K0/unified_flow_lam-0.5_K-0_R-49.json'))
ts = d['test_succ_rate']
print(f'K=0 test: {ts*100:.1f}%  ({d[\"test_n_succ\"]}/{d[\"test_n\"]})')
print(f'GATE:    {\"PASS\" if 0.75 <= ts <= 0.85 else \"FAIL\"}  (band: 75-85%)')
"
```

### 5. Finalisation (after all 3 K JSONs land OR gate failed)

```bash
# (a) Generate the result table.
python -c "
import json
from pathlib import Path
out = Path('experiments/17_dp_hybrid/results/phase_c_2026_05_29')
print('| K  | train (10) | test (50, DP) | combined | wall  |')
print('|----|-----------:|--------------:|---------:|------:|')
for k in [0, 10, 20]:
    jsons = sorted((out / f'K{k}').glob('unified_flow_lam-*.json'))
    if not jsons:
        print(f'| {k:>2} | N/A        | N/A           | N/A      | N/A   |')
        continue
    d = json.load(open(jsons[0]))
    print(f'| {k:>2} | '
          f'{d[\"train_n_succ\"]:>2}/{d[\"train_n\"]:<2} = {d[\"train_succ_rate\"]*100:.1f}% | '
          f'**{d[\"test_n_succ\"]:>2}/{d[\"test_n\"]:<2} = {d[\"test_succ_rate\"]*100:.1f}%** | '
          f'{d[\"n_success\"]:>2}/{d[\"n_total\"]:<2} = {d[\"success_rate\"]*100:.1f}% | '
          f'{d[\"wall_s\"]/60:.1f} min |')
"
```

Copy the printed table into the BRIEF's "Result" section (lines
~286 – 290 of
`experiments/17_dp_hybrid/results/phase_c_2026_05_29/BRIEF.md`,
replacing the `TODO`s).

```bash
# (b) Edit the BRIEF.
${EDITOR:-vim} experiments/17_dp_hybrid/results/phase_c_2026_05_29/BRIEF.md
```

Sections to update:
- `## Result` — paste the table.
- `## Verdict` — write one of:
  - **GREEN**: K > 0 > 80 % + 3 pp with CIs separated.
  - **YELLOW**: K > 0 ≈ 80 % ± 3 pp.
  - **RED**:    K > 0 < 80 % − 5 pp.
  - **FAIL**:   K = 0 outside [75, 85] %.

```bash
# (c) Commit + push.
git add experiments/17_dp_hybrid/results/phase_c_2026_05_29/BRIEF.md
git add experiments/17_dp_hybrid/results/phase_c_2026_05_29/K0/ \
        experiments/17_dp_hybrid/results/phase_c_2026_05_29/K10/ \
        experiments/17_dp_hybrid/results/phase_c_2026_05_29/K20/
git commit -m "exp17(dp): Phase-C K-sweep results -- <VERDICT>"
git push origin feat/dp-phase-c
```

Commit message body example:

```
K=0 sanity gate: <NN>% test rate (PASS / FAIL).
K=10: <NN>%
K=20: <NN>%

Verdict: <GREEN/YELLOW/RED/FAIL>.

Pre-registered next step: see BRIEF "Pre-registered next step".

Wall: K=0 <NN> min, K=10 <NN> min, K=20 <NN> min.  Total <NN> hr.
```

---

## Gate failure (K = 0 outside [75, 85] %)

If the gate fails, **do not run K = 10 / 20** (uninterpretable).
Diagnostic checklist (in priority order):

1. **Inspect the per-rollout JSON** for what went wrong:

   ```bash
   python -c "
   import json
   d = json.load(open('experiments/17_dp_hybrid/results/phase_c_2026_05_29/K0/unified_flow_lam-0.5_K-0_R-49.json'))
   pr = d['per_rollout']
   print(f'first rollout chunks: {len(pr[0][\"chunks\"])}')
   print(f'first rollout final_reward: {pr[0][\"final_reward\"]}')
   print(f'first rollout total steps: {sum(c[\"steps_executed\"] for c in pr[0][\"chunks\"])}')
   import statistics as st
   rewards = [r['final_reward'] for r in pr]
   print(f'reward summary: min={min(rewards):.2f} max={max(rewards):.2f} mean={st.mean(rewards):.2f}')
   "
   ```

   Each rollout should have:
   - `len(chunks) = 50` (the 50 chunks of 8 steps each)
   - `total steps = 400` (or fewer if `done=True` fired)
   - For successful rollouts: `final_reward` > 5 (lift completed)

2. **Compare to the K0_BAD_replans0_2026_05_29/ baseline** (the
   diagnostic preserved from the original bug):

   ```bash
   diff <(python -c "..." K0/...json) <(python -c "..." K0_BAD/...json)
   ```

   If new K0 looks identical to K0_BAD (same single-chunk 16-step
   behaviour) → `--mpc-replans 49` got dropped somewhere; check
   `K0/run.log` for the actual CLI args.

3. **Re-run the smoke** at very short horizon to isolate:

   ```bash
   bash experiments/17_dp_hybrid/run_phase_c_smoke.sh
   ```

   If smoke is OK but K = 0 is not, the bug is in the longer
   rollouts (e.g. cumulative obs drift, env-step counter overflow,
   prior chunk seeding).

4. **Re-eval Phase-B's standalone** to confirm the prior itself
   still hits 80 %:

   ```bash
   python -m experiments.08_flow_matching_prior.standalone_dp \
       --task can \
       --prior-checkpoint experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt \
       --n-demos 10 --n-rollouts 1 \
       --n-test-rollouts 10 --test-start-seed 100000 \
       --n-action-steps 8 --max-env-steps 400 \
       --n-steps 100 --sigma-min 0.05 --sigma-max 0.25 \
       --device cpu --seed 0 \
       --output-dir /tmp/phase_c_standalone_sanity
   ```

   At 10 fresh seeds, expect ~6 – 9 successes (matching Phase-B's
   8 / 10 sub-bucket, with sampling noise).  If standalone passes
   but unified fails, the divergence is in the unified pipeline
   wrapping (not the prior).

5. **Surface to the user**.  Include: the K0 JSON's
   `per_rollout[0]`, the gate command's `test_succ_rate`, and any
   stderr from `run.log`.  Do NOT proceed to K > 0 unless the user
   explicitly tells you to (e.g. they want to see the K > 0 results
   despite the failed gate, for diagnostic reasons).

---

## Wall-budget triage (if running long)

The 6 hr budget assumes ~lower-end per-rollout times.  If
monitoring shows the run is significantly slower than expected,
here's the priority order for what to drop:

| time check  | observed K = 0 wall | action                                  |
|-------------|-----:|-----------------------------------------|
| 30 min in   | K = 0 not done (~30 / 60 rollouts) | OK; let it run |
| 90 min in   | K = 0 not done | OK; budget allows up to 90 min for K = 0 |
| 120 min in  | K = 0 not done | **Pause**; either user weighs in or drop K = 20 |
| K = 0 done, K = 10 still running at 3 hr | OK; expected wall ~140 min |
| K = 0 done, K = 10 still running at 4 hr | **Pause**; drop K = 20 to fit budget |

To drop K = 20 without disturbing K = 10:

```bash
# Kill anything running and re-launch with only K=0 K=10.
pkill -f 'run_can|run_phase_c_ksweep'
K_VALUES='0 10' nohup bash experiments/17_dp_hybrid/run_phase_c_ksweep.sh \
    > experiments/17_dp_hybrid/results/phase_c_2026_05_29/launch.out 2>&1 &
disown
```

The script is idempotent so the existing K = 0 JSON is reused.

---

## What to commit + push

**Files that should change** as part of completing Phase-C:

- `experiments/17_dp_hybrid/results/phase_c_2026_05_29/BRIEF.md` —
  Result table + Verdict filled in.
- `experiments/17_dp_hybrid/results/phase_c_2026_05_29/K0/` — JSONs
  (one per lambda) + run.log + summary.txt.
- `experiments/17_dp_hybrid/results/phase_c_2026_05_29/K10/` — same.
- `experiments/17_dp_hybrid/results/phase_c_2026_05_29/K20/` — same
  (if K = 20 ran).
- `experiments/17_dp_hybrid/results/phase_c_2026_05_29/ksweep.log`
  — full log of the run.

**Files that should NOT be committed**:

- `experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt`
  — the actual blob (486 MB).  The symlink IS committed; the blob
  is local-only.  Verify `.gitignore` covers `*.pt` in that dir.
- `experiments/17_dp_hybrid/results/phase_c_2026_05_29/launch.out`
  — process stdout (already streamed to `ksweep.log`).

Push:

```bash
git push origin feat/dp-phase-c
```

If the push needs a PR to merge into `main`, the body should
reference this handoff doc + the BRIEF's Verdict section.

---

## Appendix: ckpt symlink design

The committed symlink at
`experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt`
has target

```
/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation/experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt
```

This works in **two worktrees** that share a parent dir:

- `HRTG-dp` (the dp-branch worktree) — symlink chains into the
  sibling `Hybrid-Robot-Trajectory-Generation` (the main-branch
  worktree).
- `Hybrid-Robot-Trajectory-Generation` — the actual blob sits here,
  uncommitted (`.gitignore`d via `*.pt`).

On a **fresh clone with only HRTG-dp**, the symlink target won't
exist.  In that case, put the blob at the symlink target path
(create the parent dirs first):

```bash
mkdir -p ~/Documents/SZH/Courseworks/ECE\ M237/Project/Hybrid-Robot-Trajectory-Generation/experiments/08_flow_matching_prior/results
cp /path/to/downloaded/v_theta_dp_tobs2_h16_v3b_ep0199.pt \
   ~/Documents/SZH/Courseworks/ECE\ M237/Project/Hybrid-Robot-Trajectory-Generation/experiments/08_flow_matching_prior/results/
```

(The committed symlink is a known wart.  A cleaner design would be
a `results/` cache outside the repo, but that's not what was
committed at Phase-B GREEN time, and respecting the existing
symlink avoids a destructive cleanup mid-Phase-C.)

---

## Contact / escalation

If anything is unclear or the gate fails in an unfamiliar way,
surface to the user with:

1. The exact failing command + last 30 lines of stderr.
2. The `K0/unified_flow_lam-0.5_K-0_R-49.json` (or whichever K
   failed) summary line: `test_succ_rate`, `train_succ_rate`,
   `wall_s`.
3. A guess at the root cause from the diagnostic checklist above.

Do not invent new flags, do not change `--lambda-prior`, do not
re-train the prior — Phase-C is **eval-only** on the Phase-B
winner.  Any wiring fix that requires editing
`run_can.py` / `unified_loop.py` / `adapter_dp.py` is a separate
phase and needs the user's explicit go-ahead.
