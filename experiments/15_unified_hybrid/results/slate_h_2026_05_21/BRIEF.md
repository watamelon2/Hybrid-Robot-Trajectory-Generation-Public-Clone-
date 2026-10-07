# Slate H -- tool_hang + A1 generalisation (May 21, queued)

Pre-staged at Slate G engineering time.  Launches after Slate G
finishes (the launcher refuses to start if a `slate_g` screen is
still alive).

Slate H exists because Slate G left two gaps.  Live smoke at
02:26 PDT (1 demo x 1 rollout, K=4 M=8 H=100) closed one of those
gaps partially -- both `tool_hang` and `transport` are wired and
run end-to-end -- but surfaced a blocker for `transport`.

1. **Tool_hang was added to `dataset.py` but never exercised at
   scored scale.**  Demos T~681 -> Slate G's H=120 would
   trivially fail.  H1 is a single-point cell at H=300 to answer
   "does it run, and does the mechanism produce non-zero
   reward".  Tool_hang has dense shaped reward, so partial
   progress shows even without full success.

   **Transport is DEFERRED from Slate H.**  The 02:26 smoke
   emitted `Warning! No dense reward currently implemented for
   this task. Forcing sparse rewards.` from robomimic.  With
   sparse reward and demos T~714, an H=300 cell at 20x2 = 40
   rollouts would near-certainly hit 0/40 (the planner only
   gets gradient at full task completion, and H=300 covers
   ~42 % of a typical demo).  Wall ~3.8 h is not justified by
   the signal.  Transport returns to the queue once a dense-
   reward shaping function lands (logged as Slate I prerequisite).

2. **A1 (action-history obs) only ran on Square in Slate G (G4).**
   That's the hypothesis-target task, but without a Can/Lift
   control we can't distinguish "A1 helps Square specifically"
   from "A1 hurts everywhere but Square is so bad anything is
   movement".  H3/H4 sweep `--action-history-len` on Can/Lift at
   the canonical operating point.

## Three tracks, 7 cells, ~4.6 h

All cells: K=10 R=1 M=16, seed=0, n-workers=10.

### H1 -- tool_hang end-to-end (~2 h, 1 cell)

`demo_1nn` lambda=1 sigma=0.25 on `tool_hang/ph`, H=300, 20 demos
x 2 rollouts = 40 rollouts.

Tool_hang demos average ~681 steps.  H=300 covers ~44 % of a
typical demo -- enough for the "pick up the tool" phase but not
the "hang on the rack" phase.  Smoke 02:26 PDT measured a 1-
rollout cost of 12.6 s at K=4 M=8 H=100; extrapolating linearly
to K=10 M=16 H=300 R=1 -> ~190 s/rollout x 40 = ~2 h.

Acceptance:

- **Strong (mechanism runs)**: wall < 3 h, mean reward >= 5,
  success > 0 (any non-zero).
- **Weak (runs but trivial)**: mean reward in [0, 5], success 0.
  Confirms the pipeline works but H=300 is too short for this
  task.
- **Falsification**: wall >> 3 h, crash, or mean reward < 0.
  Indicates a wiring bug (e.g. cost_fn not handling tool_hang
  reward, env hooks rejecting the action format).

### H2 -- transport (DEFERRED)

Robomimic does not implement a dense reward for transport.  At
H=300 with sparse rewards, the planner has no gradient signal
until full task completion (placing payload in target bin),
which a 700-step demo only achieves at T>600.  An H=300 scored
cell would near-certainly produce 0/40 and consume ~3.8 h wall.
Dropped from Slate H.

Follow-up engineering ticket: implement a dense reward shaping
function for transport (e.g. EEF-to-payload + payload-to-target
distances + binary subgoals for grasp / pick / place).  Once
landed, transport returns as a scored cell in Slate I.

### H3 -- A1 control on Can (~1.3 h, 3 cells)

`demo_1nn` lambda=1 sigma=0.25 seed=0 on `can/ph`, sweeping
`--action-history-len` in {1, 5, 10}.

Baseline at N=0: Can `demo_1nn` lambda=1 sigma=0.25 seed=0 hit
**45/60 = 75.0 %** (Slate D D2, replicated Slate F F3).  H3's
question: does A1 hurt this baseline?

Acceptance:

- **Strong (A1 is a no-op on Can)**: all three cells in
  [70 %, 80 %].  Confirms A1 is targeted (helps on contact-rich
  Square, neutral on pick-place Can).
- **Weak (small drop)**: cells in [60 %, 70 %].  Tolerable;
  A1 stays gated behind the explicit flag.
- **Falsification**: any cell <= 50 %.  A1 is generically
  harmful, not Square-specific helpful.  Would weaken the
  Square hypothesis test in G4 (if A1 hurts even Can's
  retrieval, the augmentation signal is noise).

### H4 -- A1 control on Lift (~1.3 h, 3 cells)

Same shape as H3, on `lift/ph`.  Baseline = Lift `demo_1nn`
lambda=1 cell from Slate G's G1 (will land before H launches).
Acceptance bands are set as +/- 5 pp of G1's lambda=1 cell.

If Lift baseline is low (<= 40 %, indicating Slate G's G1 falsified
mechanism transfer), H4 is interpreted descriptively rather than
gated: "does A1 move Lift?".

## What lands in `docs/results.md` after Slate H

- A new "Slate H (May 21)" section.
- H1/H2: 1-2 sentence "task runs end-to-end at H=300, mean reward
  X, success Y/40" entries.  If the mean reward is large enough
  to suggest a real signal, a Slate I would launch a full
  lambda-sweep at H=300+ on these tasks.
- H3/H4: 1-NN-vs-A1 ablation table; if all six cells are within
  the +/- 5 pp band, A1 is filed as "Square-specific, neutral
  elsewhere".  If A1 surprises (helps on Lift, say), it becomes
  a default-on flag.

## Risk register

- **Memory**: tool_hang at K=10 M=16 H=300 is a 16 x 300 x 7 =
  33 600-float perturbation tensor per iter x 10 iters = 336 K
  floats.  Negligible.  The bigger memory cost is the rollout-fn
  worker pool; each of the 10 workers holds a full env, ~150 MB.
  1.5 GB total -- fine on the 64 GB box.
- **Wall blowup**: tool_hang env step is more expensive than Can
  (longer sim, more bodies).  Smoke at H=100 K=4 M=8 gave
  12.6 s/rollout; linear scaling -> H1 ~190 s/rollout x 40 ~
  2.1 h.  If H1 exceeds 3 h wall, kill the slate and re-stage at
  H=200 (`screen -X -S slate_h kill`).  H3/H4 are stock Slate-G
  shape, no wall concern.
- **Cache invalidation**: H3/H4 build new action-history banks
  on first run (one per N value).  ~30-60 s each; not material.

## Source

- Launcher: `experiments/15_unified_hybrid/slate_h_2026_05_21.sh`
- Output dir: `experiments/15_unified_hybrid/results/slate_h_2026_05_21/`
- Precondition: Slate G done (launcher refuses if `slate_g`
  screen exists).
- Branch: `feat/zero-train-priors`.

## POSTMORTEM (added 2026-05-21 19:30 PDT, after H1 completed)

### H1 result: 0/40 success, mean reward 0.00, wall 49 min

Faster than the 2 h budget but **null on signal**.  Every one of
40 rollouts returned reward = 0.00 across the entire trajectory.

### Root cause

Investigated at 19:15 PDT mid-Slate-H: replayed `demo_0` (a
known-successful proficient-human demo) through the same env
with `reward_shaping = True` set; rewards were still 0.0 for
all 681 demo steps including the terminal success state.

Read robosuite source: `robosuite-1.5.x` `ToolHang.reward()`
(at `lib/python3.11/site-packages/robosuite/environments/
manipulation/tool_hang.py:217`) is **hard-coded sparse**:

```python
def reward(self, action=None):
    reward = 0.0
    if self._check_success():
        reward = 1.0
    return reward
```

The `reward_shaping` parameter is honored at construction but
**never consulted** in `reward()` -- the dense branch is
missing entirely.  Same applies to `TwoArmTransport.reward()`
which prints a literal "No dense reward currently implemented
for this task. Forcing sparse rewards" warning and silently
overrides `self.reward_shaping = False` (transport.py:251-255).

This is the same upstream gap that caused Slate H to defer
transport in the first place; tool_hang is in the same boat
and the slate's "tool_hang has dense shaped reward" prediction
was wrong.

### Fix (landed before Slate H finished)

`experiments/15_unified_hybrid/env_dense_rewards.py` -- monkey-
patches `env.step` to compute a staged distance + subgoal-
indicator shaping for tool_hang and transport.  See that
module's docstring for the per-task definitions and weight
ladders.  Activated automatically inside `run_can.py` `main()`
after `make_env_for_dataset(...)`.  No-op for can / lift /
square (which actually do have dense shaping upstream).

Synthetic + integration smokes passed:

- tool_hang demo-replay: reward range 0.00 -> 6.71 (max), mean
  2.58, 251 / 681 steps >= 5 (subgoal active).  Pre-fix: all
  zeros.
- transport demo-replay (open-loop, controller-drift expected):
  shaping computes correctly from obs; the demo doesn't actually
  reach success under our controller, but reward terms respond
  to distances as designed (synthetic obs test: r=22 at success,
  r=2 at contact-no-subgoals, r=0 when both objects are out of
  reach).
- 1-demo x 1-rollout integration smokes (K=4 M=8 H=80) through
  the full driver:
  * tool_hang: reward = +21.60.
  * transport: reward = +1.22.
  * can:       reward = +3.40  (no wrap; control passes through).

### Implication for the running Slate H

The fix is **not active for the in-flight slate_h** -- that
screen was launched before `env_dense_rewards.py` existed.  H1
was always going to return 0/40 once the upstream gap was
discovered; H3 and H4 are unaffected because they use Can / Lift
which have native dense shaping.  Slate H still produces useful
data on A1 (the Can / Lift controls).

### Implication for the next slate

A new Slate J re-tests tool_hang and adds transport at the
canonical operating point with the dense-reward fix active.
See `experiments/15_unified_hybrid/slate_j_2026_05_22.sh` for
the launcher and `results/slate_j_2026_05_22/BRIEF.md` for the
full acceptance gates.  Slate J is NOT auto-queued -- the user
fires it by hand once Slate H + Slate I have finished (the box
has 16 cores and J would contend with the running screens).
