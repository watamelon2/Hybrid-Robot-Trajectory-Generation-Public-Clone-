# Experiment 17 — DP-prior + MBD Hybrid

The canonical hybrid experiment used in the Week-8 headline.
A trained Diffusion Policy (DP-v3b, CFM action-chunk objective,
`T_obs=2`, 350-epoch curve) supplies the learned prior; MBD adds
the analytical-cost branch via the per-step composite-score
update.  Built on top of `experiments/15_unified_hybrid/`
(the shared composite-score plumbing).

## Headline

Two GREEN cells at the **moderately-trained** checkpoint `ep0149`
(Q1 epoch-curve peak) on `PickPlaceCan/ph`:

- **`(ep0149, K=10, λ=2.0) = 26 / 30 = 86.7 %`** (Wilson CI
  `[70.3, 94.7]`) — **+13.4 pp** over the same prior at the
  canonical `K=10, λ=0.5` baseline, at the *same* compute cost.
  Phase-F result.
- `(ep0149, K=20, λ=0.5) = 25 / 30 = 83.3 %` (Wilson CI
  `[66.4, 92.7]`) — +10.0 pp via 2× the K-budget.  Phase-E result.

Neither knob lifts the weak prior (`ep0049` stays at 0.42–0.60
across the full Q2 + Q3 grid) nor the over-trained one
(`ep0199` stays at 70–73 % across all of Phase E + F).

The **mechanism reading**: MBD lifts iff *(a)* the prior's score
is informative at the operating noise level **and** *(b)* the
prior has not saturated the task.  Both conditions; neither
alone is enough.

## Phase plan

| Phase | What it sweeps | n / cell | Status |
|---|---|---:|---|
| Q1 (epoch curve)        | `ckpt ∈ {49, 99, 149, 199, 249, 299, 349}` at `K=10, λ=0.5` | 60 | done |
| Q2 (λ-sweep at weak)    | `λ ∈ {0.1, 0.25, 1.0, 2.0}` at `ckpt=49, K=10`              | 40 | done |
| Q3 (K-sweep at weak)    | `K ∈ {5, 20}` at `ckpt=49, λ=0.5`                            | 40 | done |
| Phase E (K=20 at peak)  | `K=20` at `ckpt ∈ {149, 199}, λ=0.5`                         | 30 | done; ep0149 GREEN |
| Phase F (λ-extreme)     | `λ ∈ {0.1, 2.0}` at `ckpt ∈ {149, 199}, K=10`                | 30 | done; ep0149/λ=2.0 GREEN |
| Phase E K=40            | `K=40` at `ckpt=49`                                          | 30 | deferred (long-pole compute) |

Full numbers + the 2-D grid: see
[`results/phase_d_2026_05_30/CONCLUSION.md`](results/phase_d_2026_05_30/CONCLUSION.md)
and [`results/phase_d_2026_05_30/q2q3_AGGREGATE.json`](results/phase_d_2026_05_30/q2q3_AGGREGATE.json).

## Layout

```
17_dp_hybrid/
├── README.md                            # this file
├── HANDOFF_NEXT_MAC.md                  # operational handoff to the next-Mac worker
├── aggregate.py                         # builds the Phase-D AGGREGATE.json files
├── diagnose_chunk0.py                   # chunk-zero divergence audit
├── monitor.sh                           # screen+tail wrapper for the chained runs
├── run_phase1_hybrid.sh, run_phase1_standalone.sh
├── run_phase_c_ksweep.sh, run_phase_c_smoke.sh
├── run_diagnostics.sh
└── results/
    ├── phase_d_2026_05_30/              # the canonical Phase-D grid
    │   ├── CONCLUSION.md                # Phase-D summary, two GREEN cells highlighted
    │   ├── q1_AGGREGATE.json            # epoch-curve at K=10, λ=0.5
    │   ├── q2q3_AGGREGATE.json          # full 2-D grid (Q2 + Q3 + E + F merged)
    │   ├── epoch_curve_local/           # raw per-checkpoint run logs
    │   └── ...
    └── phase_c_2026_05_29/              # earlier diagnostic phase (OSC-reset bughunt)
```

## Source data underpinning the Week-8 presentation

The Phase-D grid is the source for the GREEN-cell figures on
slides 5 / 6 / 5b of the Week-8 deck.  See
[`presentation/INDEX.md`](../../presentation/INDEX.md) for the
file-by-file mapping.

## Related experiments

- [`08_flow_matching_prior/`](../08_flow_matching_prior/) — owns
  the DP-v3b training (action-chunk CFM); the checkpoint loaded
  here is `v_theta_dp_tobs2_h16_v3b_ep0149.pt` from that folder.
- [`15_unified_hybrid/`](../15_unified_hybrid/) — owns the
  composite-score `unified_plan_and_execute` that this experiment
  invokes with `--prior=flow --prior-checkpoint=...`.

## Lessons captured here

The **OSC-reset bug** (`unified_loop.py:478`) was discovered on
this experiment's `feat/dp-phase-c-bughunt` branch.  Before the
fix, the chunk-boundary `env.reset() + set_sim_state()` was
firing even when `K = 0`, wiping robosuite's OSC integrator
state and silently zeroing the pure-MBD sanity gate (0 / 50).
One-line gate: `if params.K > 0:`.  Post-fix sanity gate:
8 / 10 = 80 %.  See
[`results/phase_c_2026_05_29/BRIEF.md`](results/phase_c_2026_05_29/BRIEF.md)
for the full bughunt timeline.
