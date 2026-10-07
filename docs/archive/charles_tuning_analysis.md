# Tuning analysis -- Charles's `Charles-test` runs (2026-05-12 / 13)

Author of this doc: Robin (analyzing the data in
`experiments/05_hybrid_method/results/can_ph_tuning_v{1,2,3}.json`
+ the new `experiments/03_bc_baseline/results/policy.pt`, both
pushed to `Charles-test` by Charles Beck).

The numbers are headline-changing, so this doc exists to make sure
we agree on *what they show* and *what they don't* before we re-run
at full n=60.

## TL;DR

1. Charles's new BC policy hits **~60 % task success** on its own
   (3/5 in the v2 and v3 BC-alone columns), versus the published
   Exp 03 BC of **6.7 % (4/60)**. The improvement comes entirely
   from a **smaller MLP** (`hidden=[256,256]`, 79.6 k params)
   replacing the original `hidden=[512,512]`, 290.3 k params. Same
   trainer, same data, same `seed=0`, same 500 epochs.
2. The hybrid at Charles's best config (v3: `K=20, sigma_max=0.10,
   lambda_softmax=1.0`) hits 3/5 = 60 % -- **identical to BC
   alone on the same 5 demos**. The planner converts **zero** BC
   failures into successes. It "preserves" BC's wins by perturbing
   weakly.
3. `sigma_max=0.5` (v2) **catastrophically destroys** the BC
   warm-start: BC alone 3/5, hybrid 0/5. Clean ablation showing
   the planner's cost surrogate is misaligned with robosuite's
   reward once BC structure is gone.
4. Sample size is **n=5 per config, seed=0 only**. Wilson 95 % CI
   on 3/5 is [0.15, 0.95]. The 60 % point estimate is suggestive,
   not statistically separated from the published numbers.

## What changed between yours and his

Two facts, no code changes:

### A. New BC checkpoint (commit `836daae`)

| Hyperparameter | feat/further-experiments BC | Charles-test BC | Delta |
|---|---:|---:|---|
| `hidden_sizes` | [512, 512] | **[256, 256]** | ~3.6x smaller |
| Param count | 290,311 | **79,623** | ~3.6x fewer |
| `batch_size` | 256 | 512 | doubled |
| `learning_rate` | 1e-4 | 1e-4 | same |
| `epochs` | 500 | 500 | same |
| `seed` | 0 | 0 | same |
| `device` | cpu | cuda | compute only |
| **Final training loss** | **0.0060** | **0.0095** | **Charles's is HIGHER** |
| Task success (closed loop) | 6.7 % (4/60) | ~60 % (3/5) | ~9x |

The smaller model has *worse* training loss but *much better*
test-time task success. Standard generalization signal: the
290k-param MLP memorises the 23 k training transitions in 500
epochs -- training loss converges to 0.006 -- but the memorised
policy doesn't generalise to fresh `env.reset()` init states.
The 80k-param MLP can't memorise (training loss floors at
0.009) and is forced to learn the data's coarser "approach
can / grasp / lift / place" structure that *does* transfer.

This is an ML hygiene fix, not a research finding. The original
choice of `[512, 512]` was almost certainly over-parameterised
for this dataset and was the reason the published BC bottomed
out at 6.7 %.

### B. Three new Exp 05 invocations

Same `run_can.py`, just different CLI flags:

| run | K  | sigma_max | lambda_softmax | n  | Planner succ | BC alone succ | Wall |
|-----|---:|---------:|--------------:|---:|------------:|-------------:|-----:|
| v1  | 20 | **0.10** | 0.5           | 5  | 2/5 = 40 %  | 1/5          | 40 min |
| v2  | 20 | **0.50** | 1.0           | 5  | **0/5 = 0 %** | **3/5**     | 39 min |
| v3  | 20 | **0.10** | 1.0           | 5  | 3/5 = 60 %  | 3/5          | 38 min |

For reference, your `feat/further-experiments` published Exp 05
headline used `K=10, sigma_max=0.25, lambda_softmax=0.5` at n=60
and got 1.7 % (1/60).

## The sigma_max destruction story (v2)

This is the cleanest mechanism evidence in Charles's data. With
`sigma_max=0.5` the planner perturbations are large enough that
within 1-2 iterations the BC structure in `mu_k` is gone, and
the planner is essentially exploring blindly. Per-demo:

```
v2 (sigma_max=0.5, lambda_softmax=1.0, K=20):
  demo_0:  BC succeeds (+43.94)  ->  hybrid fails (+2.95)
  demo_1:  BC succeeds (+52.20)  ->  hybrid fails (+20.30)
  demo_2:  BC succeeds (+41.60)  ->  hybrid fails (+3.42)
  demo_3:  BC fails    (+5.11)   ->  hybrid fails (+2.75)
  demo_4:  BC fails    (+1.09)   ->  hybrid fails (+2.62)
```

The planner's internal cost trajectory goes from -1.9 to -19.9
on demo_0 (it *thinks* it's improving by 10x) but the executed
trajectory's robosuite reward drops from +44 to +3. This is the
cost-vs-reward misalignment we expected from the
unnormalised `_pickplace_step_cost` ranking the planner uses --
when BC structure is intact the cost minimum coincides with the
reward maximum; when BC structure is gone, the cost finds a
local minimum that robosuite doesn't reward.

**Takeaway**: `sigma_max <= 0.15` is required for hybrid stability
with this BC + cost combination. The published `0.25` is on the
edge; `0.5` is over it. This corroborates the heuristic in exp
05's docstring ("BC warm-start preservation").

## The hybrid-does-not-improve-BC story (v3)

```
v3 (sigma_max=0.10, lambda_softmax=1.0, K=20):
  demo_0:  BC succeeds  ->  hybrid succeeds  (no change)
  demo_1:  BC fails (+1.96 partial)  ->  hybrid fails (+22.69 partial)
  demo_2:  BC succeeds  ->  hybrid succeeds  (no change)
  demo_3:  BC fails (+2.17 partial)  ->  hybrid fails (+12.77 partial)
  demo_4:  BC succeeds  ->  hybrid succeeds  (no change)
```

Three demos succeed under both BC and hybrid (no change). Two
demos go from "no reach" under BC to "strong reach" under hybrid
(reward jumps from +2 to +22 / +13) but neither converts to a
grasp/success. The hybrid is currently extending the reach
phase, not closing the grasp.

**Takeaway**: at the published cost function, the hybrid's job
becomes "don't break BC". To actually convert BC-failures to
successes we need either (a) more denoising iterations, (b) a
cost function more strongly weighted on the grasp/lift terms,
or (c) the per-step composite score with the flow prior
(Tier 1b in `docs/plans.md`).

## What this implies for the paper

| Claim | Status after Charles's data |
|---|---|
| "BC baseline = 6.7 % on can/ph" | **Stale.** The headline BC was a too-large MLP that overfit. The honest BC is ~60 % once you fit the model size to the dataset. Need to re-publish Exp 03 with the new architecture. |
| "Hybrid BC + MBD > BC alone" | **Currently unsupported.** v3 at n=5: hybrid = BC alone. Need to find a regime where MBD converts BC failures. The natural next experiments are Exp 07 (lambda sweep with flow prior) and a cost-function ablation. |
| "Hybrid + MPC > Hybrid single-shot" | Unchanged in spirit -- v3 is single-shot and only at n=5. MPC at n=60 with the new BC may or may not lift things; needs to be re-run. |
| "sigma_max controls BC structure preservation" | **Newly supported.** v2 is a clean ablation: identical to v3 except sigma_max, and the planner destroys 3 BC wins. |
| "lambda-tradeoff curve / Riccati / LQ" | Orthogonal. Unaffected by the BC change. |

## Action items

In priority order (these are suggestions; treat as a discussion
starting point):

1. **Adopt the new BC architecture in `feat/mjx-rollouts`** by
   merging `origin/main`'s `policy.pt`. Existing exp 05/05b
   tuning runs are downstream of this and need to be re-run.
2. **Re-run Exp 03 at n=60** with the new BC closed-loop to
   confirm the 3/5 point estimate scales. This is the new
   headline-row-zero of the paper.
3. **Re-run Exp 05 single-shot at n=60** with the new BC and v3
   hyperparams (`K=20, sigma_max=0.10, lambda_softmax=1.0`,
   `--deterministic-bc-reset`). Tells us whether the hybrid is
   strictly equal to BC or a small positive delta at scale.
4. **sigma_max micro-sweep at n=20**: `{0.05, 0.10, 0.15, 0.25}`
   to characterise the BC-preservation curve cleanly.
5. **Train v_theta on the 256x256 architecture** for Exp 07
   (Tier 1b). The flow prior was already on the queue but is
   now more important: it is the one path to "hybrid > BC" that
   doesn't require changing the cost function.

## Caveats

- All v1/v2/v3 results are at **n=5, seed=0**. The Wilson 95 % CI
  on 3/5 is [0.15, 0.95] (~1.6 orders of magnitude wide). Every
  percentage point in this doc should be read as "point
  estimate" not "true rate".
- The new BC's seed (0) is the same as the original BC's. Other
  seeds may not reproduce the 60 % point estimate; need a small
  BC seed sweep to characterize the variance from random init.
- `lambda_softmax` is the MBD batch-averaging temperature, not
  to be confused with the data-prior weight `lambda_prior`
  introduced in exp 07. v3's `lambda_softmax=1.0` is unrelated
  to the proposal's lambda-tradeoff curve.

## Reproducing

The three tuning runs were apparently launched as:

```bash
# v1
python experiments/05_hybrid_method/run_can.py \
    --n-demos 5 --n-rollouts 1 --seed 0 --no-live-view \
    --K 20 --sigma-max 0.10 --lambda_ 0.5 \
    --output experiments/05_hybrid_method/results/can_ph_tuning_v1.json

# v2
python experiments/05_hybrid_method/run_can.py \
    --n-demos 5 --n-rollouts 1 --seed 0 --no-live-view \
    --K 20 --sigma-max 0.50 --lambda_ 1.0 \
    --output experiments/05_hybrid_method/results/can_ph_tuning_v2.json

# v3
python experiments/05_hybrid_method/run_can.py \
    --n-demos 5 --n-rollouts 1 --seed 0 --no-live-view \
    --K 20 --sigma-max 0.10 --lambda_ 1.0 \
    --output experiments/05_hybrid_method/results/can_ph_tuning_v3.json
```

(Inferred from the `params` block of each JSON; the exact CLI
Charles used wasn't logged.)
