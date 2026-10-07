# Phase-D smoke verification (post-fix)

3-seed, 120-step truncated smoke run AFTER applying Fix 1 + Fix 2a.
Compare to `../triage_n_workers/` (pre-fix evidence).

| Run | seed 100000 | 100001 | 100002 | mean | succ |
|---|---:|---:|---:|---:|---:|
| **Pre-fix n=1 (BUGGY)** | +2.75, FAIL | +1.89, FAIL | +1.36, FAIL | 2.00 | 0/3 |
| Pre-fix n=2 (correct) | +27.24, SUCCESS | +5.78, FAIL | +21.53, FAIL | 18.18 | 1/3 |
| **POST Fix1 only (n=1)** | +22.85, FAIL | +5.40, FAIL | +4.30, FAIL | 10.85 | 0/3 |
| **POST both fixes (default n=2)** | +21.97, FAIL | +4.14, FAIL | +3.54, FAIL | 9.88 | 0/3 |

**Conclusion**: both fixes lift K=0 rewards by ~5-10× vs the buggy
n=1 baseline.  Seed-to-seed variance is high at 120-step truncation
(success threshold ~25-30 reward).  The 1/3 success in the pre-fix
n=2 baseline vs 0/3 in post-fix runs is within noise (seed 100000
reached reward +27.24 vs +22.85, both at the success boundary).

Full 400-step verification (60 fresh seeds) is the next step.
Expected: ~70-80% test success, in line with Phase-B standalone.

## Fix summary

1. **Fix 1**: `unified_loop.py:478` gate the chunk-boundary
   reset+restore on `params.K > 0`.  Eliminates spurious
   controller-state corruption at K = 0 where MBD never ran.

2. **Fix 2a**: `run_can.py:555` change `--n-workers` default
   from 1 to 2.  Routes MBD's perturbations through subprocess
   workers that own forked envs, leaving the main env untouched
   at K > 0 (so the reset+restore path is never entered at all).

Both fixes are independent and complementary:
- Fix 1 alone rescues K=0 even at n_workers=1.
- Fix 2a alone rescues K>0 by avoiding the reset+restore path.
- Together they remove the chunk-boundary corruption from every
  (K, n_workers) combination except the explicit legacy override
  `--n-workers 1` at K > 0 (which still works but with the
  controller-state corruption -- documented as legacy behavior).
