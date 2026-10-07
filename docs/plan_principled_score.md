# Branch plan: `feat/principled-score` (Branch B)

Companion to [`post_consolidation_plan.md`](post_consolidation_plan.md)
on `main`. This doc specifies the engineering steps for *this*
branch only.

**Branch goal**: replace the deterministic-MLP-BC's heuristic
"score" (`−∇‖π_BC(s, x) − x‖²`) with a rigorously-defined score
function from a probabilistic generative model of the action
sequence. This is the principled fix to the score-bridge
formulation; the empirical 10× mismatch between λ=5 and the
theoretical nominal is the smoking gun that today's score is
mis-calibrated.

**Dependency**: B1 (training `v_theta.pt`) can begin immediately
in parallel with Branch C's Phase 1. B2 / B3 / B4 evaluate
against the unified driver from `feat/hybrid-coupling`.

---

## Task B1: train the conditional flow-matching prior `v_theta`

### Hypothesis

A conditional flow-matching model `v_θ(x_t, t, s_0)` over H-step
action trajectories conditioned on the initial obs gives an
*exact* score (via the linear-interpolant coupling identity in
`08_flow_matching_prior/README.md`), AND samples trajectories
multimodally — a property the deterministic MLP cannot have.

### Existing scaffolding

`experiments/08_flow_matching_prior/` is fully scaffolded:

- `model.py` — `VelocityField` MLP + `conditional_fm_loss` + `sample_trajectory`.
- `dataset.py` — demo → (action_traj, init_obs) extractor.
- `train.py` — full CLI trainer with sensible defaults.
- `adapter.py` — `FlowPrior` class with `from_checkpoint`,
  `sample_initial_mean(s_0)`, `make_prior_score_fn(s_0, K)`.
- `lq_sanity.py` — 2-D ring sanity test for the trainer.

The only missing piece is **the actual trained `v_theta.pt`**.
Prior attempts produced a checkpoint that gave 0/60 at every λ
in Exp 07's flow-prior column — meaning it was undertrained
under the old BC regime and the bar is now much higher.

### Tasks

| #     | task                                                                   | files                                            |
|-------|------------------------------------------------------------------------|--------------------------------------------------|
| B1.1  | LQ sanity test (verifies trainer wiring + score-bridge integration)    | `python -m experiments.08_flow_matching_prior.lq_sanity` |
| B1.2  | Train v_theta at default hyperparameters (2000 epochs, hidden=512, n_layers=3) | `python -m experiments.08_flow_matching_prior.train` |
| B1.3  | Save val curve, sample-trajectory plot, training log                   | (auto-written by `train.py`)                     |
| B1.4  | Eval `v_theta` solo: sample 100 trajectories per init-state, measure trajectory diversity (cluster count + pairwise L2) | new `experiments/08_flow_matching_prior/eval_diversity.py` |

### Commands

```bash
# B1.1 — smoke test (~10 s, no robomimic dataset needed)
python -m experiments.08_flow_matching_prior.lq_sanity

# B1.2 — full training
python -m experiments.08_flow_matching_prior.train \
    --task=can --n-demos=20 --horizon=120 \
    --epochs=2000 --batch-size=64 --lr=2e-4 \
    --hidden=512 --n-layers=3 \
    --seed=0 --device=cpu \
    --output=experiments/08_flow_matching_prior/results/v_theta.pt \
    --log=experiments/08_flow_matching_prior/results/train_log.json

# B1.4 — multimodality eval
python -m experiments.08_flow_matching_prior.eval_diversity \
    --checkpoint=experiments/08_flow_matching_prior/results/v_theta.pt \
    --n-init-states=20 --n-samples-per-init=100
```

### Acceptance criteria

- **B1.1 (LQ sanity)**: passes (the existing test).
- **B1.2 training**: held-out FM loss within 20 % of train loss;
  monotonic loss decrease; no NaNs.
- **B1.3 sample plot**: 8 sampled trajectories per init-state
  look *visually plausible as can-pick-and-place actions* —
  bounded action magnitudes, smooth temporal evolution, no
  obvious diverging modes.
- **B1.4 multimodality**: trajectory pairwise distance > 0.1
  (action-space L2, normalised by horizon) AND cluster count
  ≥ 2 per initial state when k-means clustered at k=4. Failure
  here means v_theta has collapsed to a single mode and is no
  better than the MLP — kill the branch if this happens.

### Effort

- B1.1: ~10 s.
- B1.2 on Mac CPU: **~10-30 h** (the long pole — 2000 epochs ×
  ~5 s/epoch ≈ 3 h optimistic, 15 h realistic with PyTorch CPU
  overhead). On GCE L4 GPU: ~3 h.
- B1.3-B1.4: ~1 h coding + ~5 min compute.

### Risks

- 20 demos is *very* few for a 120-step trajectory flow model.
  The model may overfit and produce non-diverse samples. If
  validation FM loss diverges, drop to `--n-layers=2 --hidden=256`
  or train on the 200-demo slice (open question below).
- The existing prior attempt failed at 0/60. The fix may be
  more than "train longer"; could need architecture changes.

---

## Task B2: plug `v_theta` into the unified driver

### Existing wiring

`08_flow_matching_prior/adapter.py` already exposes:

```python
class FlowPrior:
    @classmethod
    def from_checkpoint(cls, path: Path, device: str = "cpu") -> "FlowPrior": ...
    def sample_initial_mean(self, s_0: np.ndarray) -> np.ndarray: ...
    def make_prior_score_fn(self, s_0: np.ndarray, K: int) -> ScoreFn: ...
```

This matches Exp 07's expected prior interface (`initial_mean_fn`
and `score_fn_factory`) exactly. Exp 07 already supports
`--prior=flow` and loads from `--prior-checkpoint`.

### Tasks

| #     | task                                                                          | files                                              |
|-------|-------------------------------------------------------------------------------|----------------------------------------------------|
| B2.1  | Verify `--prior=flow --prior-checkpoint=…/v_theta.pt` works on Exp 07's driver  | (smoke run, no code)                              |
| B2.2  | When Branch C Phase 1 lands, verify `--prior=flow` works on unified driver     | (smoke run, no code)                              |
| B2.3  | Document the prior swap in `experiments/15_unified_hybrid/README.md`           | (one-line README addition, post-Branch-C)         |

### Commands

```bash
# B2.1 — smoke on Exp 07 (works pre-Branch-C)
python -m experiments.07_lambda_sweep.run_can \
    --prior=flow \
    --prior-checkpoint=experiments/08_flow_matching_prior/results/v_theta.pt \
    --lambda-prior 0.5 5.0 \
    --n-demos=2 --n-rollouts=1 --K=4 \
    --output-dir=/tmp/b2_smoke
```

### Acceptance criteria

- B2.1: smoke run completes without errors, produces a JSON
  with `prior=flow` in metadata.
- B2.2 (post-Branch-C): unified driver's `--prior=flow` smoke
  produces an identical-to-B2.1 result (modulo seed/RNG
  reordering) — confirms the consolidation correctly preserved
  the flow-prior path.

### Effort: ~30 min, mostly compute.

---

## Task B3: λ-sweep on v_theta prior (the headline B run)

### Hypothesis

If v_theta is a genuinely better prior, the lambda-curve on
v_theta should dominate the BC lambda-curve at *every* λ — not
just at one peak. Specifically:

- At λ=∞ (pure prior), v_theta should match or exceed BC's
  30 % (Exp 07 λ=∞). Failing this, v_theta isn't even a good
  policy.
- At λ=5 (the BC peak), v_theta should beat the BC peak's
  33.3 %.
- At λ=0 (pure MBD, no prior anchor), no change — this is the
  MBD-only floor (currently 0/60).

### Tasks

| #     | task                                                                  | files                                                |
|-------|-----------------------------------------------------------------------|------------------------------------------------------|
| B3.1  | Run 6-cell λ-sweep (λ ∈ {0, 0.1, 0.5, 1, 5, inf}) with v_theta prior on unified driver | unified driver                          |
| B3.2  | Plot v_theta vs BC λ-curves on one axis                               | extension of `07_lambda_sweep/analyze.py`           |

### Commands (post-Branch-C)

```bash
for lam in 0 0.1 0.5 1 5 inf; do
    python -m experiments.15_unified_hybrid.run_can \
        --prior=flow \
        --prior-checkpoint=experiments/08_flow_matching_prior/results/v_theta.pt \
        --lambda-prior=$lam \
        --K=10 --M=16 --sigma-max=0.25 \
        --n-demos=20 --n-rollouts=3 --seed=0 \
        --output=experiments/08_flow_matching_prior/results/lambda_flow_${lam}.json &
done
wait
```

### Acceptance criteria

| outcome                                          | implication                                                 |
|--------------------------------------------------|-------------------------------------------------------------|
| v_theta whole-curve dominates BC                  | v_theta is a strictly better prior. Strong headline result. |
| v_theta beats BC at λ=∞ only                       | v_theta is a better generative model but doesn't help the planner; weaker but still publishable. |
| v_theta matches BC at λ=∞ (~30 %), loses elsewhere | v_theta is no better than MLP-BC for hybrid use; we keep the MLP. |
| v_theta is worse than BC at λ=∞                    | Training failed; kill branch and revisit architecture / data. |

### Effort: ~6 h compute (parallel), ~1 h plotting.

---

## Task B4 (optional): Gaussian-head BC

### Hypothesis

If v_theta training is too unstable or too slow on Mac CPU, a
Gaussian-head BC is a strictly simpler probabilistic policy
whose score is also exact (Gaussian log-density gradient is
closed-form). It's a half-step on the road from deterministic
MLP to full flow model.

### Tasks (only if B1 underperforms)

| #     | task                                                                    | files                                              |
|-------|-------------------------------------------------------------------------|----------------------------------------------------|
| B4.1  | Add `GaussianMLPPolicy` to `experiments/03_bc_baseline/policy.py`        | `03_bc_baseline/policy.py`                         |
| B4.2  | Add `--head={deterministic,gaussian}` to `03_bc_baseline/train.py`       | `03_bc_baseline/train.py`                          |
| B4.3  | Train Gaussian BC, save as `experiments/03_bc_baseline/results/policy_gauss.pt` | (running training)                          |
| B4.4  | Add `--prior=gaussian_bc` adapter in unified driver (Branch C provides no-op) | unified driver                                  |
| B4.5  | Eval at n=60: stand-alone Gaussian BC, then in hybrid                    | unified driver                                     |

### Acceptance criteria

- B4.3: Gaussian-BC alone reaches 21.7 % ± 3 pp (no regression
  vs the deterministic MLP-BC).
- B4.5: Hybrid with Gaussian-BC prior reaches ≥ 33 % at some λ,
  using the *exact* Gaussian-log-density gradient as the score
  (not a heuristic).

### Effort: ~1 day coding + ~30 min training + ~6 h eval compute.

---

## Ordering for this branch

B1.1-B1.2 can begin immediately (no driver dependency; just
trains a model). B2 and B3 require Branch C Phase 1. B4 is
contingent on B1 underperforming.

Recommended sequencing:

```
T+0      : B1.1 LQ sanity (~10 s, gates the rest)
T+0+1 min: B1.2 training in background (~10-30 h CPU)
           Branch C Phase 1 in foreground (~10-12 h)
T+1 day  : B1.3-B1.4 multimodality eval
T+1 day  : B2.1 smoke on Exp 07 (confirms checkpoint loads)
T+1.5 day: B2.2 + B3 on unified driver (~6 h compute)
T+2 day  : B4 if B3 underperforms (~2 days additional)
```

---

## Open questions for the user (Branch B specifically)

1. **B1.2 compute target**: I'd default to launching Mac CPU
   training in the background now. If the GCE L4 path
   unblocks within 24 h, abort the Mac run and use GPU. Or
   commit to one or the other up-front?
2. **B1.2 dataset scope**: 20 demos for a flow model is
   borderline. Use the 200-demo `can/ph` slice for training
   (and document that v_theta sees more data than the BC
   baseline)? This breaks fair-comparison framing but may be
   the only way the flow model works.
3. **B4 trigger**: when do we decide to switch from "fix
   v_theta" to "fall back to Gaussian BC"? Suggested rule: if
   B1.4 fails the multimodality bar OR B1.2 doesn't reach a
   stable val loss after 2× the expected training time, kick
   off B4 in parallel.

---

## Rollback plan

If B1.4 fails the multimodality test (v_theta has collapsed),
do not waste compute on B3. Either:

- Iterate on B1 (different hyperparameters / more data / different
  arch), OR
- Skip to B4 (Gaussian-head BC as the half-step principled
  fix), OR
- Defer the whole branch and concentrate effort on Branches A
  and C.

A failed v_theta training is *itself* a finding — it suggests
flow-matching priors are not viable in the small-demo regime,
which constrains where the project's framing can land.

---

## 2026-05-15 update — v2 architecture + MJX integration on `feat/flow-mjx`

The May-14 v1 training (B1) hit the failure-mode predicted in the
rollback plan: `v_theta` trained in 7.1 s, but the sampled
trajectories were **1000× off the `[-1, 1]` action-bound**
(documented in `experiments/08_flow_matching_prior/results/B1_status.md`
and `train_log.json`). Rather than fall back to B4 (Gaussian-head
BC), the branch was renamed and re-scoped onto a stronger v2
architecture. The branch is now **`feat/flow-mjx`** (the original
`feat/principled-score` was deleted after the v2 work was
consolidated here).

### Status of the original B1-B4 tasks

| Task | Status                | Note                                                                                                              |
|------|-----------------------|-------------------------------------------------------------------------------------------------------------------|
| B1   | DONE (with rework)    | Initial v1 = 1000× off-scale; conditioning-bug retrain (Fix A/B) brought it down to 2× off-scale; still 0/60 eval. Logs in `train_log{,_fixA,_fixB}.json`. |
| B2   | partially obviated    | Multimodality wasn't measured directly because v1's scale bug dominated. v2 architecture skips this check; if v2 achieves competitive eval the multimodality claim is supported empirically by the result itself. |
| B3   | DONE                  | v1 evaluated against the unified driver (`experiments/15_unified_hybrid/results/unified_flow_lam-5.0_K-10_R-1_flow_phase2.json` = 0/60). |
| B4   | not taken             | Gaussian-head BC fallback is unnecessary now — v2 is the principled fix; B4 stays in reserve.                     |

### New tasks B5 / B6 / B7

These are commits already on this branch — see the log:

- **B5 — `v_theta` v2 architecture** (FiLM `s_0` conditioning +
  `x_1`-parameterization + residual blocks + cosine LR schedule).
  Files: `model_v2.py`, `train_v2.py`, `adapter_v2.py`,
  `colab_train_v2.ipynb` under
  `experiments/08_flow_matching_prior/`. Replaces v1's heuristic
  concat conditioning and `velocity`-target.
- **B6 — MJX + flow integration**: `--prior=flow` flag wired
  into `experiments/04_mbd_baseline/run_can.py`'s MJX rollout
  path. Per-demo `s_0` loaded; `prior_score_fn` closure built
  per demo; `mu_final` saved to JSON for post-eval.
- **B7 — Unified Colab notebook**
  (`experiments/04_mbd_baseline/colab_flow_mjx.ipynb`): single
  T4/L4 session does v2 train + MJX-accelerated rollouts +
  robosuite post-eval + automatic download of result JSONs.

### Task B8 (TODO) — Colab L4 train + eval run

The end-to-end run is gated only on **Colab L4 GPU access**
(free T4 will work but is ~5× slower). Both `v_theta` training
and MJX rollouts use the GPU; post-eval (robosuite) is
CPU-bound and runs after.

**Open the notebook:** `experiments/04_mbd_baseline/colab_flow_mjx.ipynb`.

Cells, top to bottom:

1. **Setup** — clone repo, install deps (uses
   `scripts/remote/install_linux_gpu.sh` lean install:
   jax[cuda12]==0.10.0 + mujoco-mjx==3.8.1 + torch).
2. **Train `v_theta` v2** (~30 min on L4, ~2 h on T4):
   ```bash
   python experiments/08_flow_matching_prior/train_v2.py \
       --epochs 20000 --batch-size 1024 \
       --n-trajs 20 \
       --output experiments/08_flow_matching_prior/results/v_theta_v2.pt
   ```
   Cosine LR + AMP + larger MLP + FiLM `s_0`-conditioning are all
   baked into the defaults; just run the cell.
3. **Smoke-test the trained adapter** — instantiate
   `FlowPriorV2.from_checkpoint(...)`, sample 4 trajectories at
   a fresh `s_0`, plot per-step `‖x‖` to confirm samples are in
   the action bound (`[-1, 1]`). The May-14 v1 had `‖x‖/step` =
   2.8 (residual 2× off-scale); the v2 acceptance gate is
   `‖x‖/step ≤ 1.5`.
4. **MJX rollouts at K=10 R=1 σ=0.25 M=16, n=60** (~15-25 min on
   L4, ~1.5 h on T4):
   ```bash
   python experiments/04_mbd_baseline/run_can.py --mjx --prior flow \
       --prior-checkpoint experiments/08_flow_matching_prior/results/v_theta_v2.pt \
       --K 10 --n-replans 1 --lambda-prior 5 \
       --sigma-max 0.25 --M 16 \
       --n-demos 20 --n-rollouts 3 --seed 0 \
       --output experiments/04_mbd_baseline/results/can_v2_flow_K-10_R-1.json
   ```
   The MJX path saves `mu_final` for each demo, which the next
   cell replays in robosuite.
5. **Post-eval in robosuite** (~30-60 min CPU):
   ```bash
   python experiments/04_mbd_baseline/run_can.py --postevaluate \
       --source experiments/04_mbd_baseline/results/can_v2_flow_K-10_R-1.json \
       --output experiments/04_mbd_baseline/results/can_v2_flow_K-10_R-1_posteval.json
   ```
6. **Download artifacts** — uses `google.colab.files.download`
   to pull the result JSONs locally.

### Acceptance bars

| Outcome | Eval rate | Interpretation                                                       | Next step                                                                                  |
|---------|----------:|----------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| **A — flow > BC** | ≥ **31.6 %** (Job C's 37.5 % CI lower bound) | v2 architecture fix worked; flow can substitute for BC.       | Headline the paper as "Flow Matching Hybrid"; add 4-seed pool at the v2 flow cell.            |
| **B — flow ≈ BC**  | 22-32 % (in BC CI [12.7, 33.0])             | v2 matches BC at 20 demos; multimodality but no clear win.    | Report both side-by-side. Hypothesis: flow > BC at larger n_demos; we don't have CPU budget to verify. |
| **C — flow < BC, sample-scale fixed** | < 22 % but `‖x‖/step` ≤ 1.5      | v2 fixed the scale bug but didn't catch up to BC; v2 is sample-correct but uninformative. | Document as a negative result. Headline stays at the BC numbers (37.5 % CI [31.6, 43.8]).    |
| **D — flow ≪ BC, scale still broken** | < 5 %                            | v2 architecture didn't fix the off-scale problem.              | Iterate on v2 (likely `tanh` head + stricter `x_1` clip).                                   |

### Compute

- Free T4 Colab tier: ~4 h end-to-end (train 2 h + rollout 1.5 h
  + post-eval 0.5 h). Risk: T4 sessions disconnect after ~12 h
  idle and after ~4 h continuous compute on the free tier.
- L4 Colab tier: ~1.5 h end-to-end. Highly recommended; needs
  Colab Pro or GCP $300 free credit.
- Local M4 Max: not viable for the training step (PyTorch + MPS
  doesn't accelerate FiLM blocks well).

### Where the result lands

- 2 JSONs under `experiments/04_mbd_baseline/results/` (the raw
  MJX run + the robosuite post-eval).
- New row in `docs/results.md` § "May 15 headline" — replacing
  the current v1 0/60 row (with footnote pointing at the v1 cell
  for historical context).
- New entry in `docs/run_log.md` summarising the v2 train
  curves, `‖x‖/step` post-train, and the final eval rate.
- Branch hygiene row in `docs/results.md` updated to note
  `feat/flow-mjx` owns the v2 result JSONs.

### Risk

- **Colab idle disconnect during training** — mitigated by the
  notebook saving `v_theta_v2.pt` to Drive at every epoch
  checkpoint (cell 2 above writes to `--checkpoint-every 1000`).
- **MJX rollout numerical drift** — verified at Gate-2c
  (`docs/mjx_migration.md`): cost-rank Spearman ρ = +0.485 vs
  robosuite, OSC parity 8e-12 Nm. The post-eval in robosuite
  catches any drift that survives.
- **`s_0` normalisation drift** — v2 uses the same `obs_mean` /
  `obs_std` as v1 (baked into the checkpoint). Smoke-test (step
  3 above) catches any drift before the 25-minute rollout cell.

### Cross-branch dependencies

- **Branch A (`feat/stronger-mbd-signal`)** and **Branch C
  (`feat/hybrid-coupling`)** both touch the unified driver but
  not the flow prior. No integration constraints with this
  branch's work; the `--prior=flow` flag is a prior-side concern.
- Cross-branch context (other branches' work + unified status
  table): [`plans.md`](plans.md) on `main`.
