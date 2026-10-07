# Slate J -- tool_hang + transport recovery with dense reward shaping

Pre-staged at Slate H postmortem time (2026-05-21 19:30 PDT).
Manual fire only -- no auto-watcher.

## Why this slate

Slate H H1 returned 0/40 success on tool_hang in 49 minutes.
Mid-slate investigation traced the root cause to **upstream
robosuite-1.5 ships sparse-only rewards** for both `ToolHang`
and `TwoArmTransport`:

- `ToolHang.reward()` is hard-coded to return 0 or 1 from
  `_check_success()`; the `reward_shaping` parameter is honored
  at `__init__` but never consulted in `reward()`.
- `TwoArmTransport.reward()` prints a literal "No dense reward
  currently implemented for this task. Forcing sparse rewards"
  warning and silently overrides `self.reward_shaping = False`.

Both files carry literal `# TODO!` comments where the dense
branch would live -- this is a documented upstream gap.

The fix landed at 2026-05-21 19:25 PDT in
`experiments/15_unified_hybrid/env_dense_rewards.py`:

- `dense_reward_tool_hang(obs)` -- 5-stage shaping:
  1. EEF-to-frame distance (always, weight 1.0)
  2. frame_is_assembled subgoal flag (+5)
  3. EEF-to-tool distance (post-assembly, weight 1.0)
  4. tool-to-frame-hook distance (post-assembly, weight 2.0)
  5. tool_on_frame success bonus (+10)

  Per-step max in pre-assembly phase = 1.0; post-assembly,
  pre-hung = 9.0; at success ~= 19.0.

- `dense_reward_transport(obs)` -- 5-stage shaping:
  1. gripper0_to_payload distance (always, weight 1.0)
  2. gripper1_to_trash distance (always, weight 1.0)
  3. payload_in_target_bin subgoal flag (+5)
  4. trash_in_trash_bin subgoal flag (+5)
  5. joint success bonus (+10)

  Per-step max in approach phase = 2.0; one subgoal = 7.0;
  both subgoals = 12.0; at success ~= 22.0.

- `wrap_env_with_dense_reward(env, task)` monkey-patches
  `env.step` to add the shaping after the inner step runs.
  Idempotent; no-op on can / lift / square (which have native
  dense shaping); applied automatically in `run_can.py main()`.

Smoke tests:

- tool_hang demo-replay: 661/681 steps have reward > 0.05,
  max 6.71, mean 2.58, 251 steps reached subgoal (>= 5).
- transport synthetic-obs: 0 -> 22 across the success ladder
  (approach < contact < one-subgoal < both < success).
- 1-demo x 1-rollout integration through the unified driver:
  tool_hang reward = +21.60, transport reward = +1.22, can
  (control, no wrap) = +3.40.  All three ran without error.

## Tracks

All cells: K=10 R=1 M=16 H=300 sigma=0.25 seed=0,
20 demos x 2 rollouts = 40 rollouts/cell, n-workers=10.

### J1 -- tool_hang lambda-sweep (~5-6 h, 2 cells)

`demo_1nn` lambda in {0, 1} on `tool_hang/ph`.  Mirrors the
H1 setup exactly but with the dense-reward wrapper active.

Acceptance:

- **STRONG**: at least one cell with mean reward >= 5 AND any
  non-zero success rate.  Validates that the dense shaping is
  doing useful work and that the score-bridge can leverage it.
- **WEAK**:   mean reward in [2, 5] at lambda=1, success 0.
  Shaping fires (early-phase signal) but H=300 is too short
  to reach hang state.  Recommend H=500 retry in a follow-on.
- **FAIL**:   mean reward < 2 at both lambdas.  Shaping
  weights mis-calibrated or prior uninformative on tool_hang.
  Falsification of "dense reward unlocks tool_hang for the
  hybrid".

### J2 -- transport lambda-sweep (~5-6 h, 2 cells)

`demo_1nn` lambda in {0, 1} on `transport/ph`.  First scored
transport cells in the project; tests cross-task transfer to
dual-arm at the same H=300 budget.

Acceptance:

- **STRONG**: at least one cell with mean reward >= 3.
  Validates dense shaping + dual-arm dispatch.
- **WEAK**:   mean reward in [1, 3].  Shaping fires for one
  subgoal but not both.  Expected if the single-arm-shaped
  prior cannot coordinate both arms.
- **FAIL**:   mean reward < 1 at both lambdas.  Dual-arm
  dispatch broken or demo_1nn retrieval not finding viable
  neighbours (transport bank: 200 demos, 700+ steps each,
  ~70-dim obs).

### Combined expectation

Realistic outcome based on robomimic's published baselines:

- BC on tool_hang (T=400): 0%.  Diffusion policy at T=1000+: ~80%.
- BC on transport: 6%.  Diffusion policy on transport: ~70%.

Our hybrid uses a zero-train retrieval prior, so the right
comparison is "BC + bridge" rather than "Diffusion + bridge".
Realistic expectation:

- tool_hang at H=300: 0-15 % success (Strong gate at 5+%).
- transport at H=300: 0-10 % success (Strong gate at 3+%).

Even mean reward >= 5 at lambda=1 on either task would be a
publishable cross-task result -- the first time the hybrid
mechanism has been validated outside the
{can, square, lift} family.

## Engineering posture

- **Wall**: 10-12 h sequential.  J2 starts after J1 ends.  No
  parallelism within the slate (the box has 16 cores; one
  10-worker slate at a time is the right size).
- **Precondition**: both Slate H and Slate I screens must
  be GONE.  The launcher checks this and refuses to start
  if either is still alive.
- **Fire by hand**:
  ```bash
  screen -dmS slate_j bash experiments/15_unified_hybrid/slate_j_2026_05_22.sh
  ```
- **Monitor**: the existing
  `results/slate_g_2026_05_21/monitor.sh` does not yet track
  slate_j.  Add it post-fire if running interactively, or
  just `tail -F results/slate_j_2026_05_22/run.log`.

## What happens after Slate J

Decision tree:

- **STRONG on both** : tool_hang + transport are scoreable;
  add to the standard sweep grid.  Next slate runs lambda-sweeps
  at finer resolution + C1 contrastive prior on both (mirrors
  Slate I I1 but for the new tasks).
- **STRONG on tool_hang, WEAK on transport** : transport needs
  longer horizon OR a coordinated dual-arm prior.  Defer
  transport to a follow-on with H=500 and possibly a per-arm
  retrieval.
- **WEAK on both** : H=300 is the bottleneck.  Re-run at H=500
  (~9 h/cell) with tighter budget (10 demos x 2 rollouts =
  20 rollouts/cell, 2 cells per task, ~7 h total).
- **FAIL on both** : the dense shaping needs work.  Re-calibrate
  weights based on demo-replay reward distributions; current
  weights were picked from analytical max-per-step intuition
  rather than empirical demo statistics.

## Source

- Launcher:        `experiments/15_unified_hybrid/slate_j_2026_05_22.sh`
- Shaping module:  `experiments/15_unified_hybrid/env_dense_rewards.py`
- Driver wiring:   `experiments/15_unified_hybrid/run_can.py main()`
  (wrap_env_with_dense_reward called after make_env_for_dataset)
- Output dir:      `experiments/15_unified_hybrid/results/slate_j_2026_05_22/`
- Branch:          `feat/zero-train-priors`.
