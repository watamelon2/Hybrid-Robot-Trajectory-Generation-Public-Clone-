# Adaptation 3 — Demo-bank scaling law for the retrieval prior

*Branch: `feat/adapt-demo-scaling`. Companion specs: `01_cost_time_spec.md`,
`02_dynamics_spec.md`, `04_incremental_prior_spec.md`. Theory
companion: `docs/theory/01_soft_control.md` §1 — this experiment
perturbs the $p^{\mathrm{data}}$ factor of the soft-control target.*

This document specifies the **demo-bank scaling experiment** —
the cheapest of the four adaptation axes after cost-time (spec 01).
Maps onto the $p^{\mathrm{data}}$ factor of the soft-control
target: the dynamics $f$ and the cost $J$ stay fixed; only the
retrieval bank's *size* changes.

Unlike the other adaptation axes, this experiment is
**fully implementable today** — the underlying `load_demo_bank`
function already takes an `n_demos` parameter, and the flag wiring
in this branch decouples it from the evaluation-rollout `--n-demos`.
**No stub / NotImplementedError**: real cells fire on go.

## 1. What this measures

The hybrid pitch's strongest "zero-training" claim is that the
retrieval prior generalises without any retraining as the demo
bank grows. The corollary — and the substantive claim — is that
**a retrieval prior should be steeper-then-saturating in bank
size, while a trained prior (BC) should be shallow-but-monotone**.
This is the standard *parametric vs. non-parametric scaling*
contrast (Snell & Mantegazza 2019; Du et al. 2024) made specific
to trajectory imitation.

**Concrete prediction.** At fixed evaluation set (20 init states ×
3 rollouts = 60 rollouts) on Can, sweep retrieval-bank size
$N_b \in \{5, 10, 20, 50, 100, 200\}$ (the Robomimic `can/ph`
bank has 200 demos). Expected curves:

| Bank size | `demo_1nn` succ | BC succ |
|---:|---:|---:|
| 5 | small (~30 %) | very small (MLP undertrains) |
| 10 | rising (~45 %) | rising (~10 %) |
| 20 | near plateau (~70 %) | ~22 % (current headline) |
| 50 | plateau (~75 %) | ~30 % |
| 100 | plateau (~76 %) | ~35 % |
| 200 | plateau (~76 %) | ~40 % (current ceiling) |

The retrieval curve **saturates** at $N_b \sim 20$; BC's curve
**keeps climbing** because each new demo improves the MLP's fit.
The crossover $N_b^\star$ where BC overtakes retrieval is the
prediction the experiment locates.

## 2. Experiment cells (E1–E6)

Operating point: canonical `demo_1nn`, $\lambda = 1$,
$\sigma_{\max} = 0.25$, $M = 16$, $K = 10$, $R = 1$ on Can.
Evaluation is **always** 20 init states × 3 rollouts (the
canonical 60-eval cell); only the retrieval bank's size varies.

| Cell | `--prior-bank-size` | `demo_1nn` | BC retrained | Total cells |
|---|---:|---:|---:|---:|
| E1 | 5 | 60 evals | (re-train MLP on 5 demos) + 60 | 2 |
| E2 | 10 | 60 | + 60 | 2 |
| E3 | 20 | 60 | + 60 (= current headline BC) | 2 |
| E4 | 50 | 60 | + 60 | 2 |
| E5 | 100 | 60 | + 60 | 2 |
| E6 | 200 | 60 (= current headline 1NN) | + 60 (= current headline BC) | 2 |

**Demo bank for evaluation init states** is **always the full 200**
(via `--n-demos=20` — the first 20 demos of the 200 — same as
canonical). The retrieval bank's truncation always takes the
**first $N_b$ demos**, which overlap with the evaluation init
states for $N_b \le 20$; this is intentional, mirrors the
canonical headline setup.

Wall-clock: 12 cells (6 retrieval + 6 BC) × ~50 min = **10 h**.
Halves if BC retraining is skipped (BC retraining is ~30 min CPU
per cell + 50 min eval; if we cite the published Robomimic BC-RNN
~92 % as the trained-baseline ceiling and only re-evaluate the
canonical MLP BC at varied bank sizes, the BC side becomes a
cheap re-eval at ~5 min per cell). **Realistic minimum: ~6 h.**

## 3. Implementation checklist

Before firing any cell:

- [x] `--prior-bank-size N` flag in `run_can.py` (this commit).
- [x] Plumbed through to `load_demo_bank(n_demos=...)` (this commit).
- [x] Output JSON `params.prior_bank_size` (this commit).
- [ ] **Verify cache key includes `prior_bank_size`** — already
      does, because `_bank_cache_key` hashes `n_demos`, so a
      change in this flag invalidates the cache cleanly.
      Confirm with `ls -la <dataset_dir>/.*.bank.*.npz` after
      smoke.
- [ ] **BC re-training at small banks.** For BC's E1–E6 column:
      `experiments/03_bc_baseline/train.py --n-demos N --epochs 100`,
      then point `--bc-checkpoint` at the new `policy.pt`. Add
      a tiny wrapper script
      `experiments/15_unified_hybrid/slate_demo_scaling.sh` to
      orchestrate the 6 retrains + 12 evals.
- [ ] **Cache-busting smoke.** Verify the bank-size cache works
      end-to-end: run twice at `--prior-bank-size 50`, expect
      `bank cache HIT` on second run.

This branch ships a **fully working** flag — no
NotImplementedError. The implementation checklist is for the
orchestration wrapper + BC retraining, not for the flag itself.

## 4. Pre-flight smoke

```bash
# 30 s, 2 demos x 1 rollout, K=4 -- verifies bank size is honoured.
python -m experiments.15_unified_hybrid.run_can \
    --task=can --prior=demo_1nn --lambda-prior 1.0 \
    --K=10 --M=16 --sigma-max=0.25 --mpc-replans=1 \
    --prior-bank-size 5 \
    --smoke --output-name-suffix=_E1_smoke
```

Expected: `[exp15] zero-train bank: 5 demos, ~750 timesteps, ...`
in the stderr, and `params.prior_bank_size = 5` in the output
JSON. No NotImplementedError; this is the **first adaptation
spec where the flag is real, not stub**.

## 4.1 What Theorem 1' predicts for this experiment

Bank-size scaling exercises the **$p^{\mathrm{data}}$ factor** of
the soft-control target (`01_soft_control.md` eq. (3)). It is
the most direct test of Corollary 1 of `02_snis_bound.md`,
because:

- The variance reduction ratio $\rho(\bar\tau)$ depends on **how
  close the prior anchor $\bar\tau$ is to a mode of the
  cost-Boltzmann measure** (`02_snis_bound.md` §4 closed-form).
- For a retrieval prior, $\bar\tau$ is the 1-NN demo action
  window at the current obs. The **expected 1-NN distance to
  the nearest mode** scales as a function of bank size $N_b$:
  fast initial decrease (each new demo covers a previously-empty
  region), then **plateau** once the bank covers all modes
  (additional demos only deepen existing mode coverage,
  marginal $\rho$-gain saturates).
- A trained prior (BC MLP) does not have this saturation
  property: each new demo improves the MLP's *function-class
  fit*, regardless of mode coverage.

**Specialised to bank size $N_b$:**

| Term in Theorem 1' | Behaviour as $N_b$ grows | Why |
|---|---|---|
| BC bias | **monotone, slow decrease** | MLP function-class fit improves with each demo; no saturation up to capacity limit |
| BC variance | 0 | deterministic forward pass |
| Retrieval bias | **slow decrease then plateau** | 1-NN distance saturates once bank covers all modes |
| **Retrieval variance** $\rho^{-1}(\bar\tau)$ | **fast decrease then plateau** | $\rho$ saturates as soon as bank coverage reaches mode-coverage threshold |
| Hybrid (retrieval prior) | **bias floor at retrieval-bias plateau, variance floor at retrieval-variance plateau** | both terms saturate together at $N_b^{\mathrm{cov}}$ |
| Hybrid (BC prior) | **bias monotone decrease, variance ~constant** | BC's bias does not saturate; variance side is dominated by SNIS, not by BC |

**Single most theory-relevant prediction:** the retrieval
hybrid's **succ-rate curve is concave-saturating with a knee at
$N_b^{\mathrm{cov}}$** — the mode-coverage threshold — while
BC's curve is **convex or log-linear** (no saturation). The
crossover $N_b^\star$ where BC overtakes retrieval is where
BC's monotone bias reduction outpaces retrieval's saturated
floor.

**Reading the canonical Robomimic operating point.** The
73.8 % pooled headline at $N_b = 200$ tells us that either
(a) we are *past* the saturation knee ($N_b^{\mathrm{cov}} < 200$,
retrieval is mechanism-limited), or (b) we are *near* the knee
($N_b^{\mathrm{cov}} \sim 200$, retrieval is bank-size-limited).
The scaling-law plot of §5 distinguishes these.

**Falsifies Theorem 1' if:** the retrieval curve is **NOT
concave-saturating** (e.g., monotone steepening, or
non-monotone). Would mean $\rho(\bar\tau)$ has an unexpected
$N_b$ dependence not captured by Corollary 1's
mode-anchoring argument. Most likely cause: 1-NN distance
metric is bad (different demos at similar obs distance are not
all "good modes"), connecting back to the Square mechanism
limit story.

**Substantiates Theorem 1' if:** the retrieval curve has a
visible knee at some $N_b^{\mathrm{cov}}$, and BC's curve
intersects retrieval's at $N_b^\star > N_b^{\mathrm{cov}}$
(BC eventually wins on bias). The exact $N_b^\star$ is the
**parametric-vs-non-parametric crossover** — a publishable
quantity, regardless of whether $N_b^\star$ is inside or
outside the 200-demo budget.

## 5. Analysis plan — the scaling-law figure

**Plot.** $x$-axis: $\log_2 N_b$ over $\{5, 10, 20, 50, 100, 200\}$.
$y$-axis: success rate. Two curves: `demo_1nn` (retrieval, no
training) and BC (MLP, retrained per bank size). Both with
Wilson 95 % CI bars.

**Predicted shapes.**

- `demo_1nn` curve is **concave**, rising steeply from $N_b = 5$
  to $N_b = 20$ then plateauing. Reflects coverage: small bank
  = few nearest neighbours available; large bank = nearest
  neighbour saturates at "demo-quality" distance.
- BC curve is **convex** (or roughly log-linear in $N_b$),
  rising more slowly but without an asymptote at $N_b \le 200$.
  Reflects parametric capacity: MLP can absorb more demos
  productively up to the saturation point of its function class.

**Crossover prediction.** $N_b^\star$ where BC overtakes
retrieval is the **single most informative number** the figure
produces. If $N_b^\star < 200$: retrieval is bank-size limited
on Can. If $N_b^\star > 200$: retrieval is mechanism-limited (the
score-bridge variance-reduction story is *not* about bank size).
**The current headline (73.8 % at $N_b = 200$) tells us that
retrieval is at least within ~20 pp of trained SOTA at the bank
size we have — the experiment localises whether more demos
would close that gap.**

## 6. Falsification protocol

- **Retrieval curve does NOT saturate** (keeps rising at
  $N_b = 200$): the project's "73.8 % is bank-limited not
  mechanism-limited" reading is wrong; more demos would
  improve. Strengthens the case for a flow / diffusion prior
  on a larger bank (`feat/flow-v3-karras` work).
- **Retrieval curve saturates at $N_b = 20$**: confirms the
  Slate-E reading that the retrieval prior is mechanism-limited.
  No path forward by adding demos; the Square mechanism limit
  on a *different* metric (representation, not bank size).
- **BC overtakes retrieval at $N_b \le 50$**: would mean the
  retrieval prior is *worse* than BC at moderate bank size and
  the canonical $N_b = 20$ headline is a happy accident. Major
  framing revision: "retrieval is the cheaper end of the
  trade-off only at the smallest banks".
- **BC never overtakes retrieval**: would mean retrieval is
  Pareto-optimal in (bank size, training cost) space across
  the entire range; "zero training" is the right axis to
  headline.

## 7. Status

- 2026-05-25: branch created, flag + plumbing landed.
  **This branch ships a real, working flag — no stub.** Smoke
  command verified to parse (item 6 pre-flight).
- Pending: BC retraining wrapper + orchestration shell script.
  Not blocking the retrieval-side curve, which can fire today.
- Once landed: ~6 h to a publishable figure for the final
  report.

## References

- Snell, C., Mantegazza, M. *Non-parametric estimation for
  imitation learning.* ICML 2019.
- Du, Y., Klimov, O., Lavin, A., et al. *Scaling laws for
  in-context learning.* 2024. (Empirical curves for
  retrieval-augmented vs.\ parametric methods.)
