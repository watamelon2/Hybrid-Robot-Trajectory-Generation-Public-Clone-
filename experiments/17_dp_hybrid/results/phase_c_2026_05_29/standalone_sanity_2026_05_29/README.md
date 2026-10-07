# Standalone sanity re-eval (Phase-B `v3b_ep0199` on the secondary Mac)

**Date**: 2026-05-29
**Reason**: Phase-C K-sweep gate failed (K=0 test rate = 0/50 = 0.0%,
band [75, 85]%).  Per HANDOFF_NEXT_MAC.md "Gate failure" diagnostic
step 4, re-evaluate the Phase-B standalone driver on this Mac with
the same ckpt to isolate whether the prior degraded or the unified-
pipeline wrapping is at fault.

## Command

```bash
python -m experiments.08_flow_matching_prior.standalone_dp \
    --task can \
    --prior-checkpoint experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt \
    --n-demos 10 --n-rollouts 1 \
    --n-test-rollouts 10 --test-start-seed 100000 \
    --n-action-steps 8 --max-env-steps 400 \
    --n-steps 100 --sigma-min 0.05 --sigma-max 0.25 \
    --device cpu --seed 0 \
    --output-dir /tmp/phase_c_standalone_sanity
```

(Wall: 20.4 min CPU.)

## Result

| metric | this Mac (2026-05-29) | Phase-B reference (50-seed bucket) |
|---|---|---|
| train (10 legacy demos) | 7/10 = **70.0%** | n/a |
| test (10 fresh seeds 100000-100009) | 7/10 = **70.0%** | 8/10 sub-bucket of 40/50 = 80% |
| total (20) | 14/20 = **70.0%** | n/a |
| typical successful-rollout steps | 119 - 162 (early exit on lift) | n/a |
| mean reward | 22.3 | n/a |

**Verdict (this run)**: PASS -- within the runbook's predicted "~6-9
/ 10" band for fresh-seed sub-buckets at this size.  Confirms the
prior is healthy on this Mac.  See `../BRIEF.md` "Verdict" section
for the implications for Phase-C.

## Apples-to-apples summary

| pipeline | task / schema | ckpt | test seeds | test_succ | wall |
|---|---|---|---|---|---|
| Phase-B standalone (today, this Mac) | `can_dp` (23-D) | `v3b_ep0199` | 100000-100009 | **7/10 = 70.0%** | 20.4 min |
| Phase-C unified K=0 (today, this Mac) | `can_dp` (23-D) | `v3b_ep0199` | 100000-100049 | **0/50 = 0.0%** | 108.0 min |

Identical schema, identical ckpt, identical first 10 fresh seeds.
Only the wrapping changes.  The 70 percentage-point gap is in the
unified-pipeline wrapping, not the prior.

## Files

- `standalone_dp.json`: per-rollout JSON dump from the standalone driver.
- `run.log`: full stdout/stderr.
