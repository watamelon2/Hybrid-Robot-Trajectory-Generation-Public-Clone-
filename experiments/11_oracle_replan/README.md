# Experiment 11 — Oracle replan from strong-reach states

**Status:** _scaffolded._ End-to-end script ships in this branch;
needs the existing Exp-05 single-shot JSON to pick rollouts from.

## Question

Exp 05b reports a strong-reach -> success conversion of
**14.8 %** with one MPC replan, vs **3.1 %** for the
single-shot variant. Two competing explanations:

1. **Mechanism**: replanning at `t=H/2` gives the planner a
   near-grasp env state to refine against; the second plan is
   intrinsically better-positioned.
2. **Variance**: two independent draws of the perturbation
   lottery is just twice as many tickets.

This script isolates explanation (1): we *only* replan the
strong-reach rollouts, *only* once, *only* from `t=H/2`.

## Reading the result

| outcome                                       | interpretation                       |
| --------------------------------------------- | ------------------------------------ |
| oracle conversion ≈ 14.8 % (Exp 05b)          | mechanism claim is **strong**        |
| oracle conversion ≈ 3.1 % (Exp 05 single-shot) | mechanism claim is **weak** -- variance dominates |
| oracle conversion >> 14.8 %                   | the BC + MPC schedule under-uses the second plan; tune `K` higher in the second chunk |

## Running

```bash
# Default: take strong-reach rollouts from the published
# experiments/05_hybrid_method/results/can_ph.json.
python experiments/11_oracle_replan/run.py

# A faster version that only attempts a few demos:
python experiments/11_oracle_replan/run.py --n-rollouts 1
```

Output: `results/oracle.json`.

## Caveat

The first chunk's env state is reproduced *deterministically* by
re-running the same BC warm-start + MBD denoising loop with
`--deterministic-bc-reset`. If the published JSON was generated
without that flag, the snapshot at `t=H/2` may differ slightly
from what produced the original strong-reach reward. For the
mechanism question this is fine -- "near-grasp env state at
`t=H/2` from a strong-reach trajectory" is the salient property,
not bit-equivalence to the original run.
