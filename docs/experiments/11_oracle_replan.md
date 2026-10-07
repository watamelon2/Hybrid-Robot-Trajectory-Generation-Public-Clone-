# Experiment 11 — Oracle replan from strong-reach states

**Lifecycle:** **scaffolded**. End-to-end script ships; needs the
exp-05 single-shot JSON as input to pick rollouts from.

**In-tree deep doc:**
[`experiments/11_oracle_replan/README.md`](../../experiments/11_oracle_replan/README.md).

## Question

Exp 05b reports a strong-reach → success conversion of **14.8 %**
with one MPC replan, vs **3.1 %** for the single-shot variant. Two
competing explanations:

1. **Mechanism** — replanning at `t = H/2` gives the planner a
   near-grasp env state to refine against; the second plan is
   intrinsically better-positioned.
2. **Variance** — two independent draws of the perturbation
   lottery is just twice as many tickets.

This experiment isolates explanation (1): replan **only** the
strong-reach rollouts (already at the near-grasp state), **only**
once, from `t = H/2`.

## How to read the result

| Observed conversion rate | Interpretation                       |
|--------------------------|--------------------------------------|
| ≈ **14.8 %**             | Mechanism claim is **strong**: the MPC boost comes from the better second-plan starting state, not the lottery. |
| ≈ **3.1 %**              | Mechanism claim is **weak**: variance dominates, and exp 05b's 14.8 % is mostly luck. |
| ≫ **14.8 %**             | The BC + MPC schedule under-uses the second plan; tune `K` higher in the second chunk. |

## Reproduce

```bash
# Default: take strong-reach rollouts from the published
# experiments/05_hybrid_method/results/can_ph.json.
python experiments/11_oracle_replan/run.py

# Faster (fewer demos).
python experiments/11_oracle_replan/run.py --n-rollouts 1
```

Output: `experiments/11_oracle_replan/results/oracle.json`.

## Caveat — deterministic BC reset

The first chunk's env state is reproduced *deterministically* by
re-running the same BC warm-start + MBD denoising loop with
`--deterministic-bc-reset` (the flag that resolves the BC-alone
RNG anomaly noted in [`05_hybrid_method.md`](05_hybrid_method.md)).
If the published JSON was generated without that flag, the
snapshot at `t = H/2` may differ slightly from what produced the
original strong-reach reward. For the mechanism question this is
fine — "near-grasp env state at `t = H/2`" is the salient
property, not bit-equivalence.

## Files

| Path | Role |
|------|------|
| [`experiments/11_oracle_replan/run.py`](../../experiments/11_oracle_replan/run.py) | End-to-end script: load source JSON, pick strong-reach rollouts, re-execute, replan at H/2, score. |
| `experiments/11_oracle_replan/results/oracle.json` | Output. |

## Cross-references

- **Mechanism in question**:
  [`05_hybrid_method.md`](05_hybrid_method.md) Ablation 05b.
- **Generalised in**:
  [`15_unified_hybrid.md`](15_unified_hybrid.md) — Slate H's
  H3/H4 N=1/N=5 cells re-validate the MPC mechanism on
  `can/ph` and `lift/ph` at the canonical driver.

## Status / open questions

- [x] Script ships.
- [ ] Run the end-to-end measurement and commit the JSON.
- [ ] If the result is in the "mechanism strong" regime (~14.8 %),
      note in the Branch-C roadmap. If it's "variance dominates"
      (~3.1 %), revise the MPC paragraph in
      [`docs/results.md`](../results.md).
- [ ] Optional follow-up: try a higher-K second plan and see if
      conversion saturates.
