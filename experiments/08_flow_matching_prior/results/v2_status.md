# v_theta v2 status — measurement landed 2026-05-16

**Branch**: `feat/flow-mjx`.
**Plan reference**: [`docs/plan_principled_score.md`](../../../docs/plan_principled_score.md) § Task B3, May-15 v2 + MJX + Colab L4 spec.

## What was run

### Training (Colab L4)

`v_theta_v2.pt` trained on Colab L4 via
`experiments/08_flow_matching_prior/train_v2.py`:

```
--task=can --n-demos=200 --horizon=120
--epochs=4000 --batch-size=128 --lr=3e-4 --warmup-epochs=100
--hidden=768 --n-blocks=4 --cond-hidden=256 --time-embed-dim=64
--n-t-per-traj=32 --device=cuda --amp
```

Wall: ~20 min. Output: `v_theta_v2.pt` (42 MB, **gitignored** per
`*.pt` rule; downloaded from Drive to local for evaluation).

The Colab run came after multiple rounds of notebook install
debugging — see the squashed `fix(04/colab): consolidate all
colab_flow_mjx.ipynb fixes` commit on `feat/flow-mjx` for the full
list of fixes. The end state of the notebook is "Run All" friendly
on a fresh Colab runtime; the training-cell output ended up being
the only Colab-only artefact we needed.

### Smoke test (local Mac, CPU)

Loaded `v_theta_v2.pt` and sampled 4 trajectories conditioned on
4 different `s0` from `can/ph` (fixed seed=0):

| metric                              | v1 result        | v2 result        | target          |
|-------------------------------------|------------------|------------------|-----------------|
| sample norm/step                    | ~0.0008          | **1.45-1.52**    | ~1.24 (real)    |
| d2real ratio (`‖samp - real‖ / ‖real‖`) | 1.58-1.98x   | **0.63-0.68x**   | < 1             |
| pairwise-across-demos L2 (fixed seed) | ~5.0           | **11.35**        | > 10            |

**Diagnostic verdict**: v2 architecture (`FiLM` conditioning on s0
+ x_1-parameterization) **fixes the structural bug** in v1 where
samples ignored `s0`. The v2 prior now produces per-demo
trajectories with the right magnitude, with substantial across-demo
diversity (matched to the corresponding `s_0`), and with samples
closer to the recorded demo than the demo's own norm.

### Full 60-plan rollout (local Mac, CPU multiprocess)

`experiments/04_mbd_baseline/run_can.py`:

```
--n-demos=20 --n-rollouts=3 --H=120 --M=16 --K=10 --seed=0
--prior=flow --prior-checkpoint=v_theta_v2.pt --lambda-prior=5.0
--warm-start --device=cpu --n-workers=10
```

- Wall: **781.5 s** (= 13 min) on M4 Max with 10 workers.
- **Success rate: 0/60 = 0.0 %**.
- Plans with cost descent: 35/60.
- Plans with cost ascent (final_best worse than init_best): 25/60.

### Output artefacts

- `experiments/04_mbd_baseline/results/can_flow_v2_local_2026_05_16.json`
- `experiments/04_mbd_baseline/results/can_flow_v2_local_2026_05_16.png`
- `experiments/04_mbd_baseline/results/can_flow_v2_local_2026_05_16.log`

## Headline

> **v2 architecture fixes the v1 structural bug but does NOT
> translate into task success.** 0/60 success rate, identical to
> v1's outcome despite v2's substantially better diagnostic
> signature.

This is a real, decisive measurement. The next move is **not** more
prior training -- it is investigating *why* a structurally correct
prior with the right per-demo conditioning still fails to drive
MBD to success. Plausible follow-ups, in rough priority order:

1. **λ_prior sensitivity sweep.** Same v_theta_v2 ckpt, sweep
   λ_prior ∈ {0.5, 1.0, 2.0, 5.0, 10.0, 20.0}. The Colab plan
   reused λ=5 from the v1 BC-prior tuning; v2 lives in a different
   regime. The 25/60 cost-ascent plans suggest λ_prior may be too
   *low* and MBD is denoising away from the prior's good initial
   guess.

2. **Open-loop replay sanity on v2 samples directly.** If
   v_theta_v2's samples themselves don't certify success when
   replayed open-loop (i.e. before any MBD denoising), then MBD
   can't be expected to fix it. The v2 prior may be learning
   trajectories that look right (norm, diversity) but don't close
   the manipulation loop. Recorded demos are closed-loop policy
   outputs and notoriously fragile to open-loop replay.

3. **K × M scaling.** K=10, M=16 was tuned for the v1 BC prior.
   Bigger K (e.g. 20-30) may be needed if v2's score-bridge is
   weaker than v1's BC-driven cross-entropy term.

4. **Hybrid (BC initial-mean + flow score-bridge).** Use the BC
   prior to warm-start mu_0 (known to give 33.3 % in exp 07) and
   only use v_theta_v2 for the score-bridge term. Separates
   warm-start from refinement.

Each of these is a single follow-up run; (1) is the cheapest and
most informative. Pencil it for the next session.
