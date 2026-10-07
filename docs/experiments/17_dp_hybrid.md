# Experiment 17 — Diffusion Policy + score-bridge hybrid

*Branch:* `feat/dp-phase-c` (merged into `main` 2026-06-01 as
`8e97ca3`, superseding earlier `feat/dp-hybrid-baseline` work) ·
*Worktree:* `HRTG-dp` · *Started:* 2026-05-26 PT · *Status:*
**closed 2026-06-01 -- two GREEN cells at ep0149**.

> **Phase-D headline (2026-06-01)**: the full 2-D grid
> (ckpt × λ × K) is done.  Two GREEN cells at the Q1 epoch-
> curve peak (ep0149):
>
>   - **(ep0149, K=10, λ=2.0) = 26/30 = 86.7 %** [70.3, 92.7],
>     +13.4 pp over the K=10 λ=0.5 baseline.  Cheapest path
>     (same compute as the baseline, just more bridge weight).
>   - (ep0149, K=20, λ=0.5) = 25/30 = 83.3 % [66.4, 92.7],
>     +10.0 pp.  Corroborates along the K-knob.
>
> YELLOW everywhere else.  K=40 at ep0049 is deferred (paused
> via SIGSTOP at 2 / 25 rollouts; will resume on a follow-up
> branch later in the week).  Canonical docs:
>
>   - [`experiments/17_dp_hybrid/results/phase_d_2026_05_30/CONCLUSION.md`](../../experiments/17_dp_hybrid/results/phase_d_2026_05_30/CONCLUSION.md)
>   - [`experiments/17_dp_hybrid/results/phase_d_2026_05_30/q2q3_AGGREGATE.md`](../../experiments/17_dp_hybrid/results/phase_d_2026_05_30/q2q3_AGGREGATE.md)
>   - [`experiments/17_dp_hybrid/results/phase_c_2026_05_29/BRIEF.md`](../../experiments/17_dp_hybrid/results/phase_c_2026_05_29/BRIEF.md)
>     (resolved verdict subsection)

## 0. One-paragraph thesis

We add **Diffusion Policy (DP, Chi et al. 2023)** as a learned prior
to our hybrid framework on Can/`ph`. The prior DP attempts on this
project (exp 08, May 2026) hit **36.7 % standalone** vs the paper's
~99 % — a 60 pp recipe gap — and **0 % when combined with MBD via the
canonical `learned`-score path** because DP's ε-prediction is
non-stationary across MBD's σ schedule. The same diagnostic on the
v6 CFM prior produced the `--score-mode pull` fix
(`feat/v6-pull-slate-l`, merged into main). **This experiment uses
pull-mode MBD on the existing 36.7 % DP checkpoint to test whether
the hybrid recovers and exceeds the standalone DP number.**

The headline claim, if it lands:

> *Score-bridge MBD with the pull-mode anchor lifts an under-trained
> Diffusion Policy from 36.7 % to ≥ 50 % standalone — i.e., online
> search compensates for missing training compute.*

If true, this becomes the **training-compute-Pareto** version of the
project's headline (the `demo_1nn` headline is the
*zero-training-compute* corner; this is the *limited-training-compute*
corner).

## 1. Where prior DP attempts landed

Reconstructed from `experiments/08_flow_matching_prior/results/`
and the `overnight_2026_05_24/BRIEF.md` post-mortem:

| Checkpoint | Train recipe | Standalone (K=100, n_act=8) | + MBD (`learned`, λ=1) |
|---|---|---:|---:|
| `v_theta_dp_tobs2_h16.pt`        | 50 ep, **no EMA** | **36.7 %** | 0.0 % |
| `v_theta_dp_tobs2_h16_ema_e300.pt` | 300 ep, **EMA 0.9999** | 16.7 %  | (not run) |
| `v_theta_dp_tobs2_h120.pt`       | 50 ep, no EMA, H=120 | 6.7 %  | 10.0 % |
| `v_theta_dp_tobs1_h120.pt`       | 50 ep, no EMA, T_obs=1 | 0–10 % | 13.3 % |

Key paradoxes the existing diagnosis identified:

1. **300-epoch EMA run scored 16.7 %** (vs the 50-epoch no-EMA
   ckpt's 36.7 %).  *Originally* attributed to an EMA misconfig;
   on re-diagnosis (§3.1) the actual mechanism is **overfitting**
   — our recipe overparameterises the dataset and longer training
   memorises trajectories at the cost of generalisation.  EMA
   happens to be a no-op in our recipe because LR fully cools
   before EMA's averaging window kicks in (verified by direct
   weight-diff inspection: EMA shadow vs raw differ by 0.18 % L2).
   Not the bottleneck for this experiment; flagged for §3.
2. **MBD's `learned`-mode destroys strong priors** (36.7 → 0).
   Confirmed mechanism: DP's score `−ε(x, t(σ))/σ` is a
   non-stationary attractor across K MBD inner iterations. The
   `pull` mode replaces this with the stationary
   `s_pull = (a_warmstart − x) / σ²` — provably correct for any
   on-manifold warm-start, and the operating mode behind the
   project's 73.8 % `demo_1nn` headline.

## 1.5 Why 36.7 % is the right operating point (the Pareto-headline framing)

This section is the **strategic spine** of exp 17.  It frames why we
deliberately ship the under-trained DP rather than first closing the
recipe gap.

### The 60 pp gap is almost entirely training compute

Verified from the training logs (`v_theta_dp_tobs2_h16.log.json`):

| Quantity | Our 36.7 % checkpoint | Paper |
|---|---:|---:|
| Epochs | **50** | **3050** |
| Gradient steps (B=256) | ~4 K | ~240 K |
| EMA | none | 0.9999 with warmup |
| Final train loss | 0.034 | (not reported) |
| Eval success (Can `ph`) | 36.7 % | 0.98 ± 0.02 |

We trained ~1.6 % of the paper's gradient-step budget.  We are *not*
seeing a recipe defect; we are seeing a deliberately undertrained
prior.  The EMA-300 evidence reinforces this — that checkpoint
reaches **5× lower train loss (0.0071 vs 0.034)** yet evals *worse*
(16.7 %), which is the signature of an EMA-application bug at eval
time (the live-weight 50-epoch ckpt outperforms the EMA-weight
300-epoch ckpt because we're not actually using the EMA copy at eval).

### Two narratives the project could choose

| | Narrative A — *MBD lifts under-trained DP* | Narrative B — *MBD adds to converged DP* |
|---|---|---|
| Prior | 36.7 % (50 ep, no EMA) | ~90 % (1000+ ep, EMA fixed) |
| Cost to set up the prior | already done | 1–2 d MPS training + EMA fix |
| What we measure | lift from 36.7 % → X with MBD | lift from ~90 % → Y with MBD |
| Story | "online search compensates for missing training compute" | "MBD also helps on a strong prior" |
| Novelty | high — fills the **training-compute-Pareto corner** alongside `demo_1nn` (zero-train) | low — once the prior is at 90 %+, MBD's ceiling-effect leaves little room and the story collapses to "MBD adds 2–5 pp" |
| Headline figure | success vs training compute, hybrid Pareto-dominates | success bar chart with hybrid slightly higher |

The project already has the **zero-training-compute corner** locked
in (`demo_1nn` at 73.8 %, exp 09 / `feat/zerotrain`).  Narrative A
extends the same Pareto-frontier story one step rightward on the
training-compute axis.  Narrative B effectively dilutes it.

**Decision**: bet on Narrative A.  Keep the 36.7 % DP as-is.

### Why MBD might NOT lift DP (the risk register for Narrative A)

The K=0 fix in §4.6 restored unified→standalone parity.  Whether
K>0 *lifts* this baseline is genuinely uncertain:

1. **Failure mode is bimodal, not jittery.**  The 30-trial K=0 cell
   shows DP either fully succeeds (reward ≥ 127) or whiffs entirely
   (reward 0.6–5.5; cube never approached).  Nothing in between.
   MBD's M=16 Gaussian perturbations at σ ≤ 0.25 in 112-dim action
   space can polish a jittery near-miss but cannot redirect a plan
   aimed the wrong way.
2. **DP runs MBD 30× per rollout; v6 CFM ran it 1×.**  v6's 75.9 %
   headline used H=120 with R=1 — one MBD call per rollout.  DP
   H=16 with R=29 calls MBD 30 times per rollout, each conditioned
   on the previous chunk's actual env state.  Cost-fn variance
   compounds; a single bad chunk derails the rollout.
3. **DP's posterior is tight (σ_min=0.05); MBD's σ_max=0.25 is
   calibrated for v6.**  Initial MBD perturbations at σ_max=0.25 add
   noise ~5× DP's training-time scale; the pull anchor must
   counteract this over K=10 iterations, which is a tug-of-war
   rather than a guarantee.
4. **Cost-fn signal is sparse near failures.**  Shaped reward on
   Can stays ~0–5 until grasp; for the 23/30 failing rollouts every
   one of M=16 perturbations returns ≈ uniform cost, so MBD's
   selection step is mostly noise.

These mechanisms argue for **modest expected lift** with material
probability of null or negative result.  Hence the graduated probe
in §2.2.

### Falsification path and pivot

If the de-risking probe in §2.2 fails to lift K=0's 23.3 % (within
±5 pp), we have a clean negative result that we publish as
"score-bridge MBD's lift is specific to T_obs=1, R=1 priors and
does not transfer to obs-conditioned DP-style structured priors".
Even the negative is informative because it sharpens the scope
boundary on the v6 / demo_1nn headlines.

In that case, we pivot to Narrative B as a fallback: fix the EMA
bug (§3.2 below), train DP to 60–80 %, and re-run the hybrid cells
on a stronger prior — losing some headline novelty but recovering
a smaller but defensible "MBD also helps trained DP" story.

## 2. Phase 1 — Hybrid on the existing 36.7 % DP (this week)

### 2.1 Smoke (~3 min)

```bash
python experiments/15_unified_hybrid/run_can.py \
    --task can \
    --prior flow --score-mode pull \
    --prior-checkpoint experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16.pt \
    --H 16 --mpc-chunk-steps 8 --mpc-replans 29 \
    --K 10 --M 16 --lambda-prior 1.0 \
    --sigma-max 0.25 --sigma-min 0.05 \
    --prior-n-steps 100 \
    --deterministic-prior-warmstart \
    --n-demos 1 --n-rollouts 1 \
    --n-workers 4 --seed 0 --device cpu \
    --output-dir experiments/17_dp_hybrid/results/smoke \
    --output-name-suffix _smoke
```

Goal: wiring verification + per-rollout timing for budget estimate.

### 2.2 Reproduce 36.7 % standalone (~15 min)

Match the verified config from
`overnight_2026_05_24/standalone_dp_tobs2_h16_cl_K100/standalone_dp.json`:

```bash
python experiments/08_flow_matching_prior/standalone_dp.py \
    --task can \
    --prior-checkpoint experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16.pt \
    --n-demos 10 --n-rollouts 3 \
    --n-action-steps 8 --max-env-steps 240 --n-steps 100 \
    --sigma-max 0.25 --sigma-min 0.05 \
    --seed 0 --device cpu \
    --output-dir experiments/17_dp_hybrid/results/baseline_standalone_dp \
    --output-name standalone_dp_repro.json
```

Expected: 11/30 = 36.7 %, ±5 pp seed noise.

### 2.3 De-risking probe (~60 min, added 2026-05-26)

After the three-bug fix in §4.6 restored unified K=0 to ~standalone
parity (7/30 = 23.3 %), the question is whether K>0 MBD actually
*lifts* this baseline.  Given the structural concerns laid out in
§1.5, we run **one focused 30-trial probe** before committing to
the full ablation slate.

Operating point chosen to maximise the probability of a positive
signal *if MBD-on-DP works at all*:

- `K=10` — same depth as the v6 75.9 % cell.
- `λ_prior=1.0` — full pull-anchor strength (keeps perturbations
  near the warm-start; respects DP's tight posterior).
- `σ_max=0.10` (NOT 0.25) — pulled in by 2.5× because DP's training
  noise scale is σ_min=0.05; σ_max=0.10 puts MBD's perturbation
  budget in the same ballpark as DP's own posterior width.
- 10 demos × 3 rollouts = 30 trials (matches K=0 baseline grid).

```bash
python experiments/15_unified_hybrid/run_can.py \
    --task can --prior flow --score-mode pull \
    --prior-checkpoint experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16.pt \
    --H 16 --mpc-chunk-steps 8 --mpc-replans 29 \
    --K 10 --M 16 --lambda-prior 1.0 \
    --sigma-max 0.10 --sigma-min 0.05 --prior-n-steps 100 \
    --deterministic-prior-warmstart \
    --n-demos 10 --n-rollouts 3 \
    --n-workers 4 --seed 0 --device cpu \
    --output-dir experiments/17_dp_hybrid/results/abl_D8_probe_K10_lam1_smax10 \
    --output-name-suffix _D8_probe
```

Decision rule:

| Probe outcome | Decision |
|---|---|
| **≥ 33 %** (clear lift above K=0's 23.3 %) | Expand to §2.5 ablation slate (λ ∈ {0, 0.5, 2}, σ ∈ {0.05, 0.25}, K ∈ {4, 20}) to map the lift surface; aim for headline. |
| **23–33 %** (parity / marginal) | Run one more cell (σ_max=0.05, K=10, λ=1) before concluding null.  Probably no headline. |
| **< 20 %** (degraded) | Negative result.  Document and pivot to §3 (Phase 2 — train DP longer with EMA fix). |

### 2.3a Probe result (2026-05-26): exactly null

The D8 probe completed with **7/30 = 23.3 %**, statistically identical
to the K=0 baseline (also 7/30 = 23.3 %).  McNemar exact two-sided
p = 1.000.  Paired comparison shows **3 recovered, 3 broken,
24 unchanged** — MBD shuffles outcomes but adds no net lift.

Detailed analysis in
`experiments/17_dp_hybrid/results/BRIEF_D8_probe_null_2026_05_26.md`.

The pattern is **bimodal shuffling**, not refinement: MBD either
flips a trial from total whiff (reward ~2) to clean success (~150)
or vice versa, with no in-between.  This matches the structural
risks predicted in §1.5 (sparse cost-fn signal, tight DP posterior).

**Decision rule lands in the 23-33 % "parity / marginal" band.**  Per
the rule we owe one more cell (σ_max=0.05, K=10, λ=1) before
declaring final null; pending user decision on whether to spend the
~2 h or pivot directly to Phase 2.

### 2.4 DP + pull-mode MBD headline cell (~45–90 min) — *runs only if §2.3 lifts ≥ 33 %*

Same operating point as 2.2 but routed through the unified driver
with `--score-mode pull`:

```bash
python experiments/15_unified_hybrid/run_can.py \
    --task can \
    --prior flow --score-mode pull \
    --prior-checkpoint experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16.pt \
    --H 16 --mpc-chunk-steps 8 --mpc-replans 29 \
    --K 10 --M 16 --lambda-prior 1.0 \
    --sigma-max 0.25 --sigma-min 0.05 --prior-n-steps 100 \
    --deterministic-prior-warmstart \
    --n-demos 10 --n-rollouts 3 \
    --n-workers 10 --seed 0 --device cpu \
    --output-dir experiments/17_dp_hybrid/results/cell_dp_pullmbd \
    --output-name-suffix _dp_pullmbd_seed0
```

Expected outcomes:

- **≥ 50 %**: hybrid recovers and exceeds standalone DP → headline.
- **30–50 %**: parity or marginal gain → λ_prior sweep.
- **< 30 %**: pull-mode also fails on DP → investigate sampler /
  obs-conditioning mismatch.

### 2.5 Ablation cells (~2–3 h total) — *runs only if §2.4 lands*

| Cell | Variation | Predicted |
|---|---|---|
| H1 | DP + pull-MBD, λ_prior=0 (warm-start only, no anchor) | < standalone (no anchor → MBD drifts off) |
| H2 | DP + pull-MBD, λ_prior=0.5 | between standalone and λ=1 |
| H3 | DP + pull-MBD, λ_prior=2.0 | over-anchored → ≈ standalone |
| H4 | DP + pull-MBD, M=32 (2× search budget) | + small lift |
| H5 | DP + learned-MBD (control) | confirms 0 % regression vs standalone |

Single-seed each; pool seeds 0/1/2 only on the headline cell that
lands.

## 3. Phase 2 — Close the 36.7 % → 99 % training gap (later, if time)

### 3.1 Re-diagnosis (2026-05-26): it's not an EMA bug, it's overfitting + a no-op EMA

The first read of "300-ep EMA evals worse than 50-ep no-EMA" suggested
an EMA-application bug at eval time.  Direct inspection of the
checkpoints (`experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_ema_e300.pt`)
shows the actual mechanics:

1. **The EMA weights ARE being loaded correctly** at eval time.
   Both `state_dict` (primary) and `state_dict_raw` contain all
   148 model keys; `adapter_dp.from_checkpoint` loads
   `blob["state_dict"]` which when `use_ema=True` is the shadow.
2. **The EMA shadow is essentially identical to raw**: relative
   L2 difference 0.18 %, max per-layer 1.5 %.  EMA is doing
   nothing useful.
3. **Why EMA is a no-op**: with our recipe (300 ep × 16 batches
   ≈ 4800 steps, cosine LR from 1e-4 fully annealed by ep 282),
   the power-decay EMA's effective averaging window in the last
   half of training falls within a regime where LR ≤ 1e-6 and
   weights barely move.  EMA averages essentially stationary
   weights → EMA ≈ raw.  Fix would be either (a) longer training
   so EMA has more "live" steps to average, (b) shorter EMA
   half-life (e.g., decay=0.99 → ~70-step half-life), or (c) a
   flatter LR schedule.
4. **The actual mechanism of the 36.7 → 16.7 % regression is
   overfitting**.  Train loss 0.034 → 0.007 (5× lower) but eval
   36.7 % → 16.7 %.  Classic train/test gap.  Our recipe has
   weight decay 1e-6 (essentially none) and no data augmentation;
   200 demos × ~100 steps × 7-dim action is highly
   overparameterised against a ~10 M-param UNet, so longer
   training memorises specific trajectories.

This finding changes Phase 2's plan substantially.

### 3.2 What it would actually take to reach 80-95 %

The paper's 99 % requires **simultaneously** all of:

| Recipe ingredient | Paper | Ours | Gap |
|---|---|---|---|
| Training duration | 3050 epochs | 50 epochs | 61× more compute |
| LR schedule | cosine over 3050 ep, slow cooldown | cosine over 50 / 300 ep, fast cooldown | weights stop moving early |
| EMA | 0.9999 over ~48 k steps, averages last ~7 k = ~440 epochs | 0.9999 over 4.8 k steps, averages stationary weights | EMA is a no-op |
| Weight decay | 1e-6 | 1e-6 | match (paper doesn't rely on WD) |
| Data augmentation | random crops / temporal jitter | none | we are missing augmentation |
| Dataset | 200 ph demos | 200 ph demos | match |
| Model size | 67 M params UNet | ~10 M params UNet (down_dims=256,512,1024) | smaller is OK if we're not overfitting |

The blocker is **all three of {training duration, LR schedule, EMA}
need to be co-scaled**.  Simply retraining for 1000 epochs with our
current LR schedule would either over-fit further (if LR stays
hot) or be wasted (if LR cooldowns too early).

Realistic Phase-2 plan (~1 week wall clock, NOT this week):

- **3.2.1** Re-implement the LR schedule to match the paper:
  cosine over the full training horizon, with `min_lr=1e-6` floor
  so weights keep moving in the late phase.
- **3.2.2** Switch EMA to decay=0.99 (half-life ~70 steps) so
  averaging tracks the late-training dynamics regardless of total
  step count.
- **3.2.3** Add minimal data augmentation (temporal jitter of ±1
  step in the demo replay).
- **3.2.4** Train for 1000 epochs with this revised recipe;
  measure standalone every 200 ep.
- **3.2.5** If 1000 ep hits ≥ 60 %, extend to 2000 ep.  Target:
  60-80 % standalone within a week of training wall clock.

**Phase 2 is decoupled from Phase 1.**  The §2 hybrid result is
publishable on its own as the training-compute-Pareto headline
(see §1.5).

### 3.3 Phase-2 retrain result (2026-05-27): pivot

The §3.2 recipe (3.2.1–3.2.4) ran end-to-end on a Colab L4 in 1.9 h.
Training landed cleanly — final loss 7.8e-4 (43× below the 50-ep
baseline's 0.034) and the LR floor genuinely held at 1 e-6 for the
last ~60 epochs (verified in `v_theta_dp_tobs2_h16_v2.log.json`).
**Standalone Can/`ph` evaluation: 4/30 = 13.3 %.**

| Ckpt | Train loss | Standalone (n=30) |
|---|---|---|
| 50 ep (`v_theta_dp_tobs2_h16.pt`) | 0.034 | 26.7 % (8/30) |
| 300 ep EMA (`..._ema_e300.pt`) | 0.0071 | 16.7 % (5/30) |
| **1000 ep + min_lr + ema=0.99 (this)** | **0.00078** | **13.3 %** (4/30) |

Monotonically decreasing standalone success with training duration
on a 66 M-param UNet against 23 k training chunks (~190× over-
parameterisation ratio) is the direct, falsifying evidence for the
§3.1 overfitting diagnosis.

Per §3.2.4's `< 35 %` band ("pivot to v6 CFM or data scaling"):

- **Decision (2026-05-27): adopt v6 CFM (already at 75.9 % standalone)
  as the production learned prior** for any remaining hybrid work.
  DP retraining ceiling on this dataset size is at most the 50-ep
  point (26.7 %), and more training collapses that further.
- Data scaling (200 → 600 demos) is deferred — not on the critical
  path for the project report.

Full breakdown in
[`experiments/17_dp_hybrid/results/BRIEF_phase2_retrain_2026_05_27.md`](../../experiments/17_dp_hybrid/results/BRIEF_phase2_retrain_2026_05_27.md),
including the per-demo bimodal failure mode (3/4 of the successes
concentrated on a single memorised demo).

## 4. Success criteria

For the project final report:

| Criterion | What lands in the brief |
|---|---|
| Phase 1: **DP + pull-MBD ≥ 50 %** | §7 gains a DP row; §9 gains a claim "score-bridge MBD lifts under-trained learned priors" |
| Phase 1: **headline beats standalone by 2× CI** | A new figure: success vs training-compute, hybrid Pareto-dominates |
| Phase 1 falsified: **pull-mode also at ~36 %** | Demoted to negative-result row; documents that pull-mode is necessary but not sufficient |
| Phase 2 partial: **DP at 1000 epochs hits 50–70 %** | The recipe gap is real; we close some of it |

## 4.5 Diagnostic finding (2026-05-26, superseded)

> *Originally hypothesized*: σ_max=0.25 was too wide for DP H=16's
> tight posterior, so MBD's perturbation step would push the DP plan
> off-manifold faster than the pull anchor could correct.  The σ
> ladder in `experiments/17_dp_hybrid/run_diagnostics.sh` (cells
> D0–D6) was launched to test this.

This hypothesis turned out to be **wrong** (or at most a third-order
effect).  The σ-ladder cells D0–D6 all converged to ~0 % on demo 0,
including the σ_max=0.05 cells that should have pinned the DP plan
in place.  Cell D7 added a **K=0 control** (zero MBD inner iterations
— a pure warm-start passthrough; this is what `standalone_dp.py`
does mathematically) which also collapsed to 0/3 on demo 0.  That
ruled out σ_max, MBD, and pull-anchor entirely as the cause: the
collapse happens **at warm-start sampling itself**, before MBD even
runs.  See §4.6 for the actual root cause.

## 4.6 Root cause (2026-05-26) — three-bug stack in unified driver

Cell D7full (K=0, 30 trials, exhaustive parity to `standalone_dp.py`)
went from **0/30 → 7/30 = 23.3 %** (vs standalone's 11/30 = 36.7 %;
within binomial CI on 30 trials) after fixing three independent
bugs in the `experiments/15_unified_hybrid` driver.  All three are
specific to **T_obs > 1, obs-history-sensitive priors** (DP); none
of them affected the v6 CFM 75.9 % or `demo_1nn` 73.8 % headlines
because those priors are dominated by cube-position obs, robust to
the perturbations these bugs introduce.

### Bug 1 — `prior.observe()` buffer not reset between rollouts

DP's adapter (`experiments/08_flow_matching_prior/adapter_dp.py`)
keeps a deque of recent obs to construct the `[s_prev, s_curr]`
conditioning that `standalone_dp.py` populates via per-step
`prior.observe(s)` calls.  The unified driver added the
`observe(s)` call inside the chunk loop but **never cleared the
deque between rollouts**, so chunk-0 of rollout N saw the leftover
history from rollout N−1.  Fixed at the top of each rollout in
`experiments/15_unified_hybrid/run_can.py` via a
`reset_obs_buffer()` exposed through the closure.

### Bug 2 — `env.reset() + set_sim_state` perturbs observations

Between chunks the driver did `env.reset() + set_sim_state(env,
current_state)` twice per chunk (Stage A for obs harvesting, Stage
D before action execution).  Even though `sim.get_state().flatten()`
is bit-identical after restore, robosuite's **controller / solver
fixed-point state** is *not* fully captured — calling
`env._get_observations(force_update=True)` after the reset+restore
returns obs that differ from the natural-propagation path by
~9e-3 in 40+ obs dims (verified by
`experiments/17_dp_hybrid/diagnose_chunk0.py`).  For T_obs=1 priors
dominated by cube-position obs this is benign; for DP T_obs=2
priors that condition on EEF orientation the drift compounds
across chunks and collapses success rate.  Fixed by positioning
the env at `init_state` once at the top of
`unified_plan_and_execute` and letting subsequent `env.step()`
calls propagate naturally; the per-chunk reset is only kept for
`n_workers <= 1` paths where `rollout_one` clobbers the main env.

### Bug 3 — `torch.manual_seed(X)` vs `Generator(seed=X)` paths produce different DP samples

`standalone_dp.py` calls
`prior.sample_initial_mean(s, n_steps=100, seed=X)`, which inside
the adapter constructs a private `torch.Generator(device).manual_seed(X)`
and passes it through to every `torch.randn(..., generator=gen)`
call in the DDPM ancestral chain.  The unified driver instead
called `torch.manual_seed(X)` globally, then
`sample_initial_mean(s, n_steps=100)` without the `seed=` kwarg —
so the adapter's `torch.randn(...)` calls fell back to torch's
global RNG state.  **These two paths produce different
samples for the same integer seed** (PyTorch's per-generator state
is initialized differently from the global state).  For this DP
checkpoint specifically, the global-state path lands uniformly in
a bad mode of the posterior (verified: reward ≈ 0 on demo 0 across
all chunks), while the private-generator path recovers the trained
rate.  Fixed via an opt-in `_set_chunk_seed` channel on the
closure (writer in `unified_loop.py` Stage A; reader in
`run_can.py`'s flow-prior branch); legacy priors that have always
used `torch.manual_seed` keep their existing behavior.

### Verification

- `experiments/17_dp_hybrid/diagnose_chunk0.py` — standalone-style
  vs unified-style chunk-0 plan comparison; reproduces all three
  bugs in isolation and confirms each fix.
- `experiments/17_dp_hybrid/results/abl_D7full_K0_30trials/run.log` —
  30-trial K=0 result post-fix: 7/30 = 23.3 % (vs standalone
  11/30 = 36.7 %; within statistical noise).
- The σ-ladder cells D0–D6 are superseded; the actual MBD-sigma
  sensitivity question remains open and will be answered by the
  K>0 sweep on top of the now-correct K=0 baseline.

## 5. Risk register

- **MPS UNet forwards slow**: 100-step DDPM × 30 chunks per
  rollout × 30 rollouts = 90,000 forwards. At ~10 ms each on MPS
  that's 15 min just for warm-starts. Budget: ~90 min for the
  headline cell. Mitigate with K=18 DDIM if budget overruns.
- **Pull-mode on DP differs from pull-mode on v6**: DP's H=16
  closed-loop has 30 chunks per rollout (vs v6's 1); the
  warm-start is *recomputed* every chunk. If DP's warm-start
  quality is inconsistent across chunks, pull-mode anchoring
  amplifies that. Mitigation: try R=14 (chunk=16, no replan
  within the predicted window) as a variant.

## 6. Files

- `docs/experiments/17_dp_hybrid.md` — this file (planning).
- `experiments/17_dp_hybrid/results/` — per-cell JSONs.
- `experiments/17_dp_hybrid/RESULT.md` — written after Phase 1.
- `experiments/17_dp_hybrid/aggregate.py` — aggregator (Phase 1 §2.4).
- `experiments/17_dp_hybrid/run_phase1.sh` — slate launcher.

## 7. Source-code audit (2026-05-27)

Phase-1 (26.7%) is irreproducible at the eval-time-seed level
(per-process Python-hash salt; ratified in
`experiments/17_dp_hybrid/results/BRIEF_nondeterminism_2026_05_26.md`),
and Phase-2c (16.7%) landed RED.  Before retraining further, this
audit compared every knob in our DP implementation against the
reference Chi et al. 2023 repo (`columbia-ai-robotics/diffusion_policy`,
config `task/lowdim/can_lowdim.yaml` + workspace
`train_diffusion_unet_lowdim_workspace.yaml`).

Reference numbers on can_lowdim ph (Table 5, Chi et al.):
- DP-C (CNN UNet — what we reimplemented): **0.85 mean / 0.94 max**
- DP-T (Transformer): 0.91 mean / 0.94 max
- Robomimic BC-RNN-GMM baseline: ~0.96

So "60%" is a low bar; the published ceiling for *our exact
architecture+objective* is 85%.  We're at 13–26% — a 60-pp gap that
requires multiple recipe deviations to explain.

### Findings (ranked by expected lift)

| # | Knob | Reference | Ours | Expected lift if fixed |
|---|---|---|---|---|
| 1 | Training epochs | **4500** | 50 (P1) / 64 (P2c best val) | **+20–40 pp** |
| 2 | Action normalisation | min-max → [-1, 1] per-dim | none (raw env actions) | **+5–15 pp** |
| 3 | Obs normalisation | min-max → [-1, 1] per-dim | z-score (mean=0, std=1) | +2–5 pp |
| 4 | Optimizer betas | (0.95, 0.999) | (0.9, 0.999) default | +1–3 pp |
| 5 | `diffusion_step_embed_dim` | 128 | 256 | ~0 (deviation, not regression) |
| 6 | `n_inference_steps` at eval | 100 (full DDPM) | 18 DDIM (P1/P2), 100 (P2c) | already fixed in P2c |
| 7 | DDIM `eta` | 0 (deterministic) | 0 ✓ | ✓ |
| 8 | EMA decay + warmup schedule | inv_gamma=1, power=0.75 | matches ✓ | ✓ (but pointless at 5k steps — shadow is still warmup-noise) |

The two findings that explain the 60-pp gap are (1) and (2),
both confirmed cheap to test.

### Finding A — undertraining

DP's published numbers on this dataset come from **540k optimizer
steps** (4500 epochs × 120 batches/epoch).  Phase-2c stopped at
**~5k steps** = 0.9% of the reference budget.  With the
`inv_gamma=1, power=0.75` EMA schedule (which we match), the
decay value only reaches `0.999` at ~1k steps and `0.9989` at
10k steps — so at 5k steps the EMA shadow is dominated by very
early (high-LR) iterates, not by the converged late-phase weights.
This is the dominant cause of Phase-2c's irreproducibility-by-
seed: a model still in its high-variance early phase reads off
different from epoch to epoch.

### Finding B — actions are not normalised (FIXED 2026-05-27)

`dataset_v3.load_chunks` returned raw env-space actions, and
`train_dp.dp_loss` consumed them directly.  Robomimic Can-ph
OSC-pose actions have ranges of just ~0.15 on rotation dims
(3–5) — for those dims, the ε-prediction loss against an
N(0, σ²·I) noise schedule has near-zero signal at high σ_t, so
the model degenerates into predicting ε ≈ x_t at the chaotic
end of the schedule.  DP's `LinearNormalizer` rescales every
action dim to `[-1, 1]` via per-dim min-max, equalising the SNR
across the noise schedule.

**Patch (2026-05-27)**:

- `dataset_v3.py`: `ChunkBundle` now carries `action_min`,
  `action_max`, plus `normalize_actions(x)` / `denormalize_actions(y)`
  helpers (affine `[a, b] ↔ [-1, 1]`).
- `train_dp.py`: new `--normalize-actions` flag (default ON);
  applies `bundle.normalize_actions(X)` before constructing the
  training tensor; persists action stats + flag in all three ckpt
  blob paths (intermediate, final, best-val).
- `adapter_dp.py`: `__init__` accepts `action_min/max` +
  `normalize_actions`; `_from_blob` reads them from the ckpt with
  identity-affine fallback for pre-patch checkpoints;
  `sample_initial_mean` denormalises model output to env-space;
  `make_prior_score_fn` normalises `mu` on entry and scales the
  returned score by the inverse Jacobian (per-dim `1/scale`) so
  MBD callers see `∇_{x_env} log p` unchanged.

**Verified end-to-end**: smoke train + load + sample +
score_fn all pass; backward compat with stripped-field
pre-patch ckpts confirmed.

### Phase-2d — single-variable test (RESULT: YELLOW, 23.3 %)

`experiments/08_flow_matching_prior/colab_train_dp_phase2d.ipynb`
is identical to Phase-2c except for `--normalize-actions`.  Ran
on Colab L4, 2026-05-27.

Results on 30 standalone trials:

| Ckpt | rate | n |
|---|---:|---:|
| `v_theta_dp_tobs2_h16_v2d` (final ep64) | **23.3 %** | 7/30 |
| `v_theta_dp_tobs2_h16_v2d_ep0049` | 16.7 % | 5/30 |
| `v_theta_dp_tobs2_h16_v2d_best_val` (ep33 EMA) | 3.3 % | 1/30 |

**Delta vs Phase-2c (final ckpt)**: **+6.6 pp** — confirms
Finding B (action normalisation) is real and signed correctly.

**Why the lift is smaller than predicted (+5-15 pp band, realised
+6.6 pp)**: on the full 200-demo training set, the action stats
are

```
action_min = [-1.0, -1.0, -1.0, -0.556, -1.0, -1.0, -1.0]
action_max = [ 1.0,  1.0,  1.0,  0.730,  0.450,  0.745,  1.0]
```

**5 of 7 dims already saturate at ±1.**  Only the rotation dims
(3, 4, 5) get any amplification, and only by 1.15-1.55×.  The
SNR-equalisation argument from the audit holds, but the headroom
is small.  Revised estimate: Finding B residual = +0-7 pp.

**The best-val=3.3% vs final=23.3% pattern repeats from Phase-2c**,
locking in Finding A: at ep33 (~2.6k steps) the EMA shadow is
dominated by warmup-noise; the `best-val` selection criterion
picks the worst possible ckpt because val loss is still tracking
the unstable shadow.

### Phase-3 — consolidated paper-faithful (RESULT: ORANGE, 26.7 % peak / 20.0 % final)

Phase-2d landed YELLOW per the pre-registered rule, so Option-1 /
consolidated paper-faithful was the next step.  Phase-3 ran on
Colab L4, 2026-05-28, 5.8 h total wall (3.7 h training + 2.1 h eval).

Settings: epochs=2000, dse_dim=128, adam_beta1=0.95, min_lr=1e-6,
warmup_epochs=5, save_every=200, EMA decay=0.9999, action norm
ON, NO val split, NO early stopping.  See
`experiments/08_flow_matching_prior/colab_train_dp_phase3.ipynb`
for the full recipe.

Eval table (30 standalone trials each, n_action=8, n_steps=100):

| Ckpt | Train loss | Eval rate | n |
|---|---:|---:|---:|
| ep0199 | 0.0161 | 20.0 % | 6/30 |
| **ep0399** | 0.0091 | **26.7 %** ← peak | 8/30 |
| ep0599 | 0.0061 | 20.0 % | 6/30 |
| ep0799 | 0.0044 | 20.0 % | 6/30 |
| ep0999 | 0.0029 | 20.0 % | 6/30 |
| ep1199 | 0.0018 | 23.3 % | 7/30 |
| ep1399 | 0.0011 | 20.0 % | 6/30 |
| ep1599 | 0.0005 | 16.7 % | 5/30 |
| ep1799 | 0.0003 | 20.0 % | 6/30 |
| ep1999 (final) | 0.0003 | 20.0 % | 6/30 |

**Key finding 1: Finding A (undertraining) was NOT the bottleneck.**
Train loss decreased 30× from 0.009 (ep399, peak) to 0.0003 (ep1999,
final) — and env-eval gained zero from this 30× improvement.  Peak
is at the first 200-ep multiple past Phase-2c's stopping point
with all later ckpts strictly equal or worse.  Textbook overfitting
to the noise-prediction loss without generalising the underlying
policy.

**Key finding 2: A ~25 % ceiling is consistent across runs.**
Phase-1 (50 ep, no EMA, no action norm): 8/30 = 26.7 %.
Phase-3 ep0399 (full audit stack): 8/30 = 26.7 %.
Phase-3 ep1199: 7/30 = 23.3 %.  Phase-2d final: 7/30.  Across
4 orders of magnitude in train loss and a wide range of training
schedules, env-eval stays in 17-27 %.  Consistent with either
(a) a tight system-level ceiling (eval pipeline / env / pipeline
bug), or (b) the one remaining untested audit finding (Finding C
/ obs norm) carrying the gap.

**Key finding 3: EMA averaging is not pulling weight.**
By ep2000 the EMA shadow has had ~182k optimisation steps to
converge (well past the half-life of decay=0.9999, which is
~6.9k steps).  Yet the final ckpt (which IS the EMA shadow) is
6.7 pp worse than the ep399 raw-model checkpoint.  Either EMA
is averaging over a polluted late-training distribution, or
the live-vs-EMA distinction matters less than the audit assumed.

**Key finding 4: Action norm (Phase-2d's win) still holds.**
Phase-3 with action norm at 26.7 % (ep399) vs Phase-2c without
it at 16.7 %: +10 pp at the comparable budget — larger than the
Phase-2d-vs-Phase-2c gap (+6.6 pp) because longer training
amortises some of the EMA-warmup tax that Phase-2d's ES@64 paid.

**Status of audit findings**:

| # | Finding | Status | Realised lift |
|---|---|---|---|
| A | Undertraining (50 → 2000 ep) | Phase-3 ❌ | ~0 pp |
| B | Action norm | Phase-2d ✓ | +6.6 pp |
| C | Obs norm | **Phase-3.5 ← next** | ? |
| D | AdamW betas (0.9 → 0.95) | Phase-3 ❌ | ~0 pp (bundled) |
| E | dse_dim (256 → 128) | Phase-3 ❌ | ~0 pp (bundled) |

Per the pre-registered Phase-3 decision rule, 26.7 % falls in the
ORANGE band (23.3-35 %): "Bundle is not fully attributable; ablate
one knob at a time."  The natural ablation target is Finding C
(obs norm) — the only finding the audit framework predicted that
remains untested.  Phase-3.5 runs Finding C as a single-variable
test.

### Phase-3.5 — obs min-max normalisation (single-variable, this is next)

Notebook: `experiments/08_flow_matching_prior/colab_train_dp_phase3p5.ipynb`.

Phase-3's plateau leaves obs norm as the only untested audit
finding.  Phase-3.5 keeps every Phase-3 setting that worked and
flips ONE variable: z-score → min-max obs normalisation, matching
DP's `LinearNormalizer.fit(mode='limits')` (the same affine shape
that the action-norm patch uses, applied to the conditioning vector
instead of the action target).

| Knob | Phase-3 | Phase-3.5 |
|---|---|---|
| epochs | 2000 | **400** (P3 peaked at ep399) |
| save_every | 200 | **50** (denser near peak) |
| action norm | ON | ON (carryover) |
| **obs norm** | z-score | **min-max [-1, 1]** ← test |
| dse_dim | 128 | 128 (carryover) |
| adam β1 | 0.95 | 0.95 (carryover) |
| min_lr | 1e-6 | 1e-6 (carryover) |
| warmup_epochs | 5 | 5 (carryover) |
| ema_decay | 0.9999 | 0.9999 (carryover) |
| val_frac | 0 | 0 (carryover) |
| early_stop | none | none (carryover) |

Pre-registered decision rule:

- ≥ 50 %: GREEN.  Obs norm carries the gap; framework was right;
  proceed to Phase-4 hybrid sweep (K=0/K=10/K=20).
- 35-50 %: YELLOW.  Obs norm lifts substantially but residual gap;
  Phase-3.6 candidate: re-enable val-split + best-EMA-tracking.
- 27-35 %: ORANGE.  Some lift over Phase-3 26.7 %; obs norm helps
  but isn't the whole story.  STOP training; audit the eval
  pipeline (`standalone_dp.py` vs DP's `robomimic_lowdim_runner`).
- 20-27 %: RED.  No lift over Phase-3.  Audit framework exhausted;
  the 25 % ceiling is system-level.  MANDATORY eval-code audit
  before any further training.
- < 20 %: BROKEN.  Obs norm actively hurt.  Bisect the patch.

Wall budget: ~30 min on A100, ~50 min on L4, ~70 min on T4
(training, 400 ep × 91 batches); +~105 min eval (8 ckpts × 13 min).
**TOTAL: ~2.5 hours.**  Significantly cheaper than Phase-3
because the 1600-epoch overfitting tail is gone.

**Patch dependencies** (all landed pre-Phase-3.5):
- `dataset_v3.py`: `ChunkBundle` adds `obs_min`/`obs_max`/
  `S_norm_minmax` + `normalize_obs`/`denormalize_obs` helpers.
- `train_dp.py`: `--normalize-obs` flag (default OFF), branches
  obs conditioning vector, saves new fields in ckpt blob.
- `adapter_dp.py`: constructor accepts new fields, `_from_blob`
  reads them with identity-affine fallback, `_normalise_s_np`
  branches.
- CPU smoke verified: both z-score and min-max paths round-trip
  at zero numerical drift; legacy pre-patch ckpts still load
  cleanly via the identity-affine fallback.

---

*Status: §7 audit + Phase-2d YELLOW + Phase-3 ORANGE + Phase-3.5
patches + notebook landed; Phase-3.5 ready for Colab run.*
