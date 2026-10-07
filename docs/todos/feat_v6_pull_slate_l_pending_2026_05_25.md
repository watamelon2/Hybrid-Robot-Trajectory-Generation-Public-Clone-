# Pending follow-ups on `feat/v6-pull-slate-l`

Date opened: 2026-05-25 22:15 PDT.  Updated 22:25 (Slate O landed
and forced a major narrative revision: MBD-pull adds +17.8 pp
over v6 standalone, the earlier Slate K-M "parity" claim was a
measurement artefact -- see `docs/results.md` § "Slate O").

This file tracks **deferred** work that the Slate K-O narrative
does not require but would extend it.  The branch is publishable
as-is on the K + L + M + N + O story.  The items here are scoped
for a successor branch `feat/v6-pull-extended` or for a focused
future session.

---

## Completed (committed on `feat/v6-pull-slate-l`)

- [x] **Slate O** -- pure v6 standalone, 3 seeds (σ=1e-4, K=1, M=1, λ=0).
      Closed 22:19:05.  Headline: 3-seed pool 57/90 = **63.3 %**,
      CI [53.0, 72.6].  v6+MBD-pull 3-seed pool 81.1 % is **+17.8 pp
      higher with non-overlapping CIs**.  **Forced major revision of
      the Slate K-M "parity" narrative**: MBD-pull adds real lift
      over v6 standalone, not parity.  See `docs/results.md`
      § "Slate O" for the full analysis.

- [x] **Theory note** at
      `docs/theory/manifold_restoration_2026_05_25.md` (SNIS
      posterior-collapse argument).  Drafted and updated §3.4 with
      the corrected R1 small-lift classification post Slate O.

- [x] **Presentation brief refresh** at
      `docs/progress_reports/presentation_brief_2026_05_22.md`
      § 6.5 covers K, L, M, N, O findings with the corrected
      +17.8 pp lift narrative.

---

## Deferred — Tier 3 (medium eval cost, ~3 h each)

These are *follow-up* probes that would tighten the K-N
narrative but do not change its direction.  Each is a
self-contained slate that can run unattended overnight.

### T3.1 — Wave B R-sweep

**Question.**  Wave B (Slate L) measured a single point: v6+pull
at R=4 chunk=24 → 46.7 %.  Combined with K3 at R=0 → 83.3 %, the
R-curve is a 2-point line.  Filling in R ∈ {1, 2, 3} resolves
two competing hypotheses:

- **Smooth degradation.**  Each replan adds a fixed off-distribution
  cost.  Predicts roughly: R=1 → 75 %, R=2 → 65 %, R=3 → 55 %.
- **Cliff.**  v6 collapses after the *first* off-aligned replan;
  R ≥ 1 looks similar to R=4.  Predicts: R=1 → 50 %, R=2,3 ≈ 47 %.

Either result is publishable; the difference matters for closed-loop
deployment recommendations.

**Cells.**  3 cells × 30 trials × ~30 min each ≈ 1.5 h.

```bash
# config skeleton (chunk=24, lam=1, pull, K=10 M=16 sigma=0.25):
for R in 1 2 3; do
    python experiments/15_unified_hybrid/run_can.py \
        --task can --prior flow \
        --prior-checkpoint $V6_CKPT \
        --n-demos 10 --n-rollouts 3 \
        --H 120 --K 10 --M 16 \
        --mpc-replans "$R" --mpc-chunk-steps 24 \
        --max-env-steps 240 \
        --warm-start --deterministic-prior-warmstart \
        --sigma-max 0.25 --sigma-min 0.05 \
        --prior-n-steps 18 --lambda-prior 1.0 \
        --score-mode pull --n-workers 4 --device cpu \
        --seed 0 --output-name-suffix "_waveB_R${R}_lam1"
done
```

### T3.2 — demo_knn k-sweep

**Question.**  Slate M measured k=1 (full rescue, +62 pp) and
k=5 (partial rescue, +20 pp).  The lift-curve in `results.md`
predicts a monotone decay in MBD-lift as k grows.  Need k=2, 3
to fit the curve.

**Cells.**  4 cells × 30 trials × ~25 min for the +MBD cells
+ ~5 min for the standalone reference cells ≈ 1.5 h total.

| cell                       | succ_pred (lift-curve fit) |
|----------------------------|---------------------------:|
| demo_knn k=2 standalone    | ~8 %                       |
| demo_knn k=2 + MBD canonical | ~55 %                    |
| demo_knn k=3 standalone    | ~5 %                       |
| demo_knn k=3 + MBD canonical | ~40 %                    |

Predictions assume the lift decays roughly geometrically between
k=1 (73.8 %) and k=5 (23.3 %).  Falsification: any cell that
deviates from a monotone interpolation by > 10 pp.

```bash
for K_NN in 2 3; do
    # standalone
    python ... --prior demo_knn --knn-k "$K_NN" \
        --K 1 --M 1 --lambda-prior 0.0 --sigma-max 0.0001 \
        --output-name-suffix "_slateExt_knn${K_NN}_standalone"
    # +MBD canonical
    python ... --prior demo_knn --knn-k "$K_NN" \
        --K 10 --M 16 --lambda-prior 1.0 --sigma-max 0.25 \
        --score-mode pull \
        --output-name-suffix "_slateExt_knn${K_NN}_mbd"
done
```

---

## Deferred — Tier 4 (sanity audits, low value but cheap)

### T4.1 — K3 reproducibility audit

**Question.**  K3 seed=0 was measured at 25/30 = 83.3 %.  The Slate
M multi-seed pool (M1 seed=1 = 24/30, M2 seed=2 = 24/30) suggests
seed=0 was the high end of a ±3 pp band; the *expected* re-run of
K3 seed=0 should also land at 25/30.

**Cell.**  1 cell × 30 trials × ~30 min.  Same config as K3.

**Decision rule.**
- Lands at 24-26/30 → confirms reproducibility, no action.
- Lands < 22/30 or > 27/30 → hidden state somewhere; investigate
  RNG, JSON metadata, deterministic-warmstart logic.

Low value because Slate M already provides multi-seed evidence;
do this only if a referee asks or a downstream result is suspect.

---

## Out of scope for this branch (will go to `feat/v6-pull-extended`)

- Cross-task validation: re-run the lift-curve on Lift / Square /
  ToolHang.  Slate G1 measured Can→Lift λ-sweep, gives a partial
  precedent.  Estimated: 5 priors × 3 tasks × 30 trials ≈ 8 h.
- Higher-strength prior comparison: train a v7 CFM with more capacity
  or more data; does v7+pull-MBD parity hold or does the ceiling
  finally move?  This is a 2-3 day exp08 retrain + new Slate K-N
  evaluation arc.
- DP retrain to recover the 84 % published baseline: the DP-pull
  cells (Slate K6) collapsed to 0-3 % because the standalone DP
  ceiling stayed at 36.7 %.  A clean DP at 84 % would either
  validate the manifold-restoration framework cross-prior or
  falsify the parity-vs-rank-and-pick decomposition.

---

## Branch close checklist

When Tier-1 work lands tonight, this branch is in publishable
shape.  Before any merge:

1. [ ] Slate O lands; standalone pool number is in `results.md`.
2. [ ] `docs/theory/manifold_restoration_2026_05_25.md` exists.
3. [ ] `docs/progress_reports/presentation_brief_2026_05_22.md`
       has been refreshed.
4. [ ] All four documents (`results.md`, theory note, brief,
       this todos doc) committed.
5. [ ] User's `slate_demoscale_2026_05_25/`, `docs/adaptation/`,
       and `unified_bc_*.json` are still untouched (they belong
       on other branches).
6. [ ] `feat/v6-pull-slate-l` is NOT merged to `main` or any
       integration branch -- merge is a separate, user-driven
       decision.
