# Branch plan: `feat/stronger-mbd-signal` (Branch A)

Companion to [`post_consolidation_plan.md`](post_consolidation_plan.md)
on `main`. This doc specifies the engineering steps for *this*
branch only.

**Branch goal**: make MBD's per-denoising-step gradient more
informative so the hybrid's MBD contribution is more than the
+3 pp it currently adds on top of λ=∞ pure-BC particle-averaging.
The diagnostic is Exp 07 λ=0 (no BC anchor, pure MBD) = 0/60 —
on its own, MBD's rollout-cost gradient finds nothing.

**Dependency**: tasks A2/A3/A4 evaluate against the unified
driver from `feat/hybrid-coupling` Phase 1. A1 (σ-sweep) uses
the existing Exp 05 driver and can run *now*. A2 (V_φ training)
is pure model training and can also start *now*.

---

## Task A1: σ-max sweep

### Hypothesis

The cosine noise schedule's top σ (`--sigma-max`) controls how
much the MBD perturbations stray from the BC anchor. The old
sweep had σ ∈ {0.25, 0.5} only. With the new strong BC, the
optimal σ may be very different.

### Existing scaffolding

`experiments/09_sigma_sweep/run.sh` is wired and uses Exp 05's
driver (warm-start-only). Pre-set σ-values:
`0.05 0.10 0.25 0.40 0.50`. Plus a downstream
`06_failure_taxonomy/run.py` aggregator.

### Tasks

| #     | task                                                                  | files                                              |
|-------|-----------------------------------------------------------------------|----------------------------------------------------|
| A1.1  | Extend σ-list to `{0.05, 0.10, 0.25, 0.40, 0.50, 0.75, 1.00}`         | `experiments/09_sigma_sweep/run.sh`                |
| A1.2  | Run the sweep with the new BC at n=60                                 | (just `./run.sh`)                                  |
| A1.3  | Analyse: success vs σ curve                                            | `experiments/09_sigma_sweep/analyze.py` (NEW, mirrors `07_lambda_sweep/analyze.py`) |
| A1.4  | When Branch C Phase 1 lands: re-run on unified driver at λ=5           | new `experiments/09_sigma_sweep/run_unified.sh`    |

### Commands

```bash
# A1.2 — run on existing exp 05 driver (warm-start-only)
SIGMAS="0.05 0.10 0.25 0.40 0.50 0.75 1.00" \
    N_DEMOS=20 N_ROLLOUTS=3 SEED=0 \
    ./experiments/09_sigma_sweep/run.sh

# A1.3 — analyse
python experiments/09_sigma_sweep/analyze.py
```

### Acceptance criteria

- A1.2 produces 7 result JSONs in `experiments/09_sigma_sweep/results/`.
- A1.3 produces `sigma_curve.png` + `summary.md` with success
  rate per σ.
- **Pass**: at least one σ value beats Exp 10 K=10 R=2 (18.3 %)
  by ≥ 3 pp, OR rules out σ as the bottleneck (all values
  within ±2 pp).
- After Branch C lands, A1.4 repeats on the unified driver at
  λ-prior=5; the eval bar is then Exp 07 λ=5 (33.3 %).

### Effort: ~10 h compute (7 σ × 80 min), ~2 h analysis code.

---

## Task A2: learned `V_φ(s)` value baseline

### Hypothesis

MBD's denoising loop uses Monte-Carlo rollout cost — the variance
of this estimator scales with horizon-squared and is the
dominant source of noise in the MBD gradient. Replacing the
plain rollout cost with `rollout_cost − V_φ(s_terminal)` (a
control-variate) should dramatically reduce gradient variance,
freeing MBD to *actually plan* rather than chase noise.

### New experiment directory

```
experiments/15_value_baseline/
├── __init__.py
├── README.md
├── model.py            # V_phi: 2-layer MLP, low_dim_v141 -> R
├── dataset.py          # (s, a, r, s') extractor from 20-demo can/ph
├── train.py            # TD(0) training, ~30 epochs, AdamW
├── adapter.py          # make_value_score_fn(s_0) for the unified driver
├── lq_sanity.py        # 2-D regression sanity test (mirrors exp 08)
└── results/
    └── V_phi.pt        # trained checkpoint
```

(Numbering: 15 is taken by `feat/hybrid-coupling`'s unified
driver; rename to 16 once Branch C confirms its numbering.
Tracked as open question.)

### Architecture

```python
class VphiNet(nn.Module):
    """
    V_phi : R^obs_dim -> R   (scalar value of a state).
    Same obs encoding as exp 03 BC (OBS_KEYS, low_dim_v141, ~24-D).
    Hidden: [128, 128], Tanh, single linear head.
    Output is unnormalised; trained against bootstrapped
    discounted return targets.
    """
    def __init__(self, obs_dim: int, hidden=(128, 128)):
        ...
    def forward(self, s: torch.Tensor) -> torch.Tensor: ...
```

### Training

- **Loss**: MSE between V_φ(s_t) and the bootstrapped target
  `r_t + γ · V_φ(s_{t+1})` with γ=0.99.
- **Data**: 20-demo `can/ph` rollouts, ~23k transitions, sparse
  task reward (r=1 only on success).
- **Optimiser**: AdamW, lr=3e-4, weight decay 1e-4, 30 epochs.
- **Validation**: hold-out 4 demos, report final-state MSE.
- **Cost**: ~5 min CPU.

### Adapter integration

The unified driver from Branch C exposes `--value-baseline-ckpt`
as a no-op flag in Phase 1. When this branch's `adapter.py`
lands, the flag becomes wired:

```python
# in experiments/15_unified_hybrid/unified_loop.py (Branch C deliverable)
def make_cost_fn(env, *, shaped, value_baseline):
    base_cost = env_rollout_cost  # original
    if value_baseline is not None:
        def cost_fn(states, tau):
            term_state = states[:, -1]
            return base_cost(states, tau) - value_baseline.predict(term_state)
        return cost_fn
    return base_cost
```

### Tasks

| #     | task                                                                                | files                                            |
|-------|-------------------------------------------------------------------------------------|--------------------------------------------------|
| A2.1  | Scaffold `experiments/15_value_baseline/` (skeleton + READMEs)                       | new dir                                          |
| A2.2  | Implement `model.py` + `dataset.py` + `train.py`                                     | new                                              |
| A2.3  | Run training, produce `V_phi.pt`                                                     | (just `python -m experiments.15_value_baseline.train`) |
| A2.4  | Implement `adapter.py` exposing `make_value_score_fn`                                | new                                              |
| A2.5  | (Branch C dependency) Eval on unified driver with `--value-baseline-ckpt`            | unified driver                                   |

### Acceptance criteria

- A2.2-A2.3: `V_phi.pt` exists; training-curve plateaus
  monotonically; held-out MSE < 0.1 of variance(V_targets).
- A2.4: adapter unit test (LQ sanity) passes.
- A2.5 (post-Branch-C): unified driver with
  `--prior=bc --lambda-prior=0 --value-baseline-ckpt=…` (pure
  MBD with V-baseline, no BC anchor) reaches **≥ 5/60**. This
  would falsify the current "MBD alone is 0/60" diagnostic and
  validate the gradient-variance hypothesis.
- A2.5 stretch: V-baseline + BC anchor (λ=5) beats unified
  λ=5 baseline by ≥ 3 pp.

### Effort: ~3 days coding + ~5 min training + ~1.5 h eval compute (post-Branch-C).

---

## Task A3: particle-count ablation

### Hypothesis

The denoising loop averages over M particles per step (default
M=16). If MBD gradient is variance-limited, scaling M → 256 or
1024 should help. But if M=16 is already enough, this rules
out particle count as the bottleneck.

### Tasks

| #     | task                                                          | files            |
|-------|---------------------------------------------------------------|------------------|
| A3.1  | Sweep M ∈ {16, 64, 256, 1024} at λ=5 on the unified driver     | unified driver   |
| A3.2  | One-figure result `experiments/15_unified_hybrid/results/M_sweep.png` | new analyzer  |

### Commands (post-Branch-C)

```bash
for M in 16 64 256 1024; do
    python -m experiments.15_unified_hybrid.run_can \
        --prior=bc --lambda-prior=5 --K=10 --M=$M \
        --n-demos=20 --n-rollouts=3 --seed=0 \
        --output=experiments/15_unified_hybrid/results/M_${M}.json &
done
wait
```

### Acceptance criteria

- **Pass**: M=1024 vs M=16 gap > 5 pp ⇒ MBD is sample-starved
  at default settings, recommend bumping M.
- **Negative**: gap < 2 pp ⇒ M=16 is enough, don't bother
  paying the 64× compute cost.

### Effort: ~6 h compute (mostly M=1024), no new code.

---

## Task A4: reward shaping (optional, paper-appendix material)

### Hypothesis

Robomimic's `can/ph` reward is sparse (success only). Adding
dense intermediate rewards (object-EEF distance, lift-height,
gripper-contact bonus) should give MBD a much cleaner gradient.
The trade-off: shaped rewards risk "cost-hacking" trajectories
that maximise the shaped term without succeeding.

### Tasks

| #     | task                                                                          | files                                             |
|-------|-------------------------------------------------------------------------------|---------------------------------------------------|
| A4.1  | Add `shaped_cost_fn` to `experiments/04_mbd_baseline/mbd.py`                  | `04_mbd_baseline/mbd.py`                          |
| A4.2  | Wire `--shaped-cost` flag in unified driver (Branch C provides the no-op)     | unified driver                                    |
| A4.3  | Sweep `--shaped-cost {on, off}` × `--lambda-prior {0, 5}` at n=60             | unified driver                                    |

### Reward shape (concrete)

```python
def shaped_reward(env, state, action):
    base = env.task_reward(state)      # 0 / 1 sparse
    obj_pos = env.object_pos(state)
    eef_pos = env.eef_pos(state)
    dist = np.linalg.norm(obj_pos - eef_pos)
    lift = max(0, obj_pos[2] - 0.85)   # bonus for object above 85 cm
    grasp = env.is_grasped(state) * 0.1
    return base + 0.5 * np.exp(-2 * dist) + 0.3 * lift + grasp
```

### Acceptance criteria

- **Pass**: shaped-cost + λ=0 (pure MBD) lifts from 0/60 to
  ≥ 5/60.
- **Pass stretch**: shaped-cost + λ=5 beats unshaped λ=5 by
  ≥ 3 pp.
- **Failure mode to flag**: high success rate on shaped reward
  but trajectories don't *actually* pick-and-place (cost
  hacking). Catch this with a visual rollout check on 3
  randomly-sampled successes.

### Effort: ~1 day coding + ~3 h compute.

---

## Ordering for this branch

Phase A1 can begin immediately (uses existing Exp 05 driver +
scaffolded Exp 09 runner). Phase A2 training (A2.1-A2.4) can
also begin immediately (purely model code, no driver dep).

Phases A2.5, A3, A4 all depend on Branch C Phase 1.

Recommended sequencing:

```
T+0     : start A1 (σ-sweep run, runs in background ~10 h)
T+0     : start A2.1-A2.4 (V_φ scaffold + training, ~3 days)
T+0     : Branch C Phase 1 (~10-12 h focused, runs in foreground)
T+1 day : A2.5, A3 run on unified driver (~12 h compute combined)
T+2 day : A4 runs if A2/A3 motivate it
```

---

## Open questions for the user (Branch A specifically)

1. **A2 numbering**: I picked `experiments/15_value_baseline/`
   but Branch C has already claimed 15. Rename to 16, or
   collapse this into Branch C's unified-driver subdir as
   `experiments/15_unified_hybrid/value_baseline/`?
2. **A2 training data**: 20-demo can/ph (~23k transitions) is
   tiny. Should we also train on the full 200-demo slice and
   compare? The 20-demo restriction is a deliberate small-data
   choice for the BC baseline; the value function could
   reasonably use more data without breaking the framing.
3. **A4 reward shape**: the shape above is a guess. The
   robomimic API exposes which intermediate signals? Should
   verify before coding.

---

## Rollback plan

If A2 (V_φ baseline) fails to lift pure-MBD off 0/60, the
"MBD gradient is signal-limited" hypothesis is falsified.
Branch C's score-bridge formulation becomes the *only* way
MBD adds value, which is paper-worthy by itself (a strong
negative result on the standard MBD-with-rollout-cost setup).
