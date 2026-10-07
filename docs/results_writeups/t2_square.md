# T2 Square cross-task ladder (writeup stub)

**Branch:**  `feat/adapt-t2-square`
**Worktree:** `HRTG-square`
**Slates:**
  - `experiments/15_unified_hybrid/slate_incremental_T2_square_hybrid_2026_05_28.sh`  (hybrid arm)
  - `slate_incremental_T2_square_bc_*.sh` (BC arm, **blocked on feat/adapt-bc-multitask-obs**)
**Status:**  results landed 2026-05-29; aggregation done; NULL RESULT diagnosed.

## Motivation

Lift was the obvious cross-task probe candidate, but the
extreme-perturbation sweep (`probe_lift_extreme_2026_05_28.sh`)
showed Lift's hybrid saturates at 90-100 % under every available
dynamics knob -- Lift cannot be pushed into the bite zone with
the current perturbation registry.  Lift is therefore a structural
null, not a confirmation.

Square is the better cross-task probe: per Slate G it is
mechanism-limited (tight nut-on-peg clearance) and demo-sensitive.
A perturbation that drifts the EEF by O(cm) makes the insertion
fail; a 5-demo bank that recalibrates the approach trajectory
should help.  Predicted outcome -- if T2 transfers cleanly:

- T2.0_square ~ 30-60 %  (degraded but not zero -- Square hybrid
  works at canonical mass=1.0 per Slate G).
- T2.1_square ~ 70-85 %  (5 demos restore the insertion approach).
- T2.4_square ~ 70-85 %  (saturation; T2.1 already enough).
- BC unadapted ~ 10-20 %, BC retrained ~ 0 % (mirror Can headline).

## Cells

Hybrid arm (this slate):

| Cell | Bank | n_trials |
|---|---|---:|
| T2.0_square | canonical 200 Square | 30 |
| T2.1_square | 200 + 5 mass=1.3 Square | 30 |
| T2.4_square | 200 + 20 mass=1.3 Square | 30 |

BC arm (companion slate after step 5):

| Cell | Description | n_trials |
|---|---|---:|
| T2.2_square | BC unadapted (Square obs schema) | 30 |
| T2.3_square | BC retrained on 205 Square mass=1.3 | 30 |

## To fill in after slate completes

### Square T2 table

| Cell | bank arg used | succ | Wilson 95 % CI |
|---|---|---:|---:|
| T2.0_square | None (slate bug) | 1/30 = 3.3% | [0.6, 16.7] |
| T2.1_square | square_mass1p3_205demos.hdf5 | 1/30 = 3.3% | [0.6, 16.7] |
| T2.4_square | square_mass1p3_220demos.hdf5 | 0/30 = 0.0% | [0.0, 11.4] |

### Diagnostic narrative (computed 2026-05-29)

**Two distinct failure modes confirmed:**

1. **T2.0_square slate bug.**  The launcher run `experiments/15_unified_hybrid/slate_incremental_T2_square_hybrid_2026_05_28.sh` for the T2.0 cell does NOT pass `--prior-bank-dataset`:

   ```bash
   # T2.0_square: hybrid + canonical 200 Square bank @ mass=1.3.
   run "T2_0_square_hybrid_canonical_bank_mass1p3" \
     --prior demo_1nn --lambda-prior 1
   ```

   At runtime, `prior_bank_dataset = None` was recorded in the rollout JSON.  The demo_1nn prior fell back to a default/empty bank; the resulting 1/30 = 3.3% is **not a valid measurement** of T2.0_square (canonical 200 Square bank).  Future re-run needs the missing `--prior-bank-dataset "$REPO/.../square_canonical_200.hdf5"` arg.

2. **T2.1_square / T2.4_square ran with the right banks but the upstream Square hybrid pipeline is broken.**  Both cells loaded the workhorse-built banks (`square_mass1p3_205demos.hdf5` / `square_mass1p3_220demos.hdf5`), produced 1/30 = 3.3% / 0/30 = 0.0% success respectively.  These are consistent with the canonical-mass Square hybrid floor that Slate G (2026-05-21) already measured at 0-8 % under every perturbation knob (sigma sweep, soft-NN beta sweep, action history).  The mass=1.3 perturbation is therefore **not the killer here** -- the Square hybrid pipeline was already broken at canonical mass=1.0.

**Net verdict.**  T2 transfer to Square cannot be demonstrated until the upstream Square hybrid pipeline is fixed (suspect obs schema / coordinate-frame issue, similar to the DP Phase-A finding for Can at the legacy 46-D `can` schema).  Future work, not blocking the Can T2 headline.

### Cross-task confirmation table (Can vs Square)

| Task   | T2.0 succ | T2.1 succ | Gap (pp) |
|---|---:|---:|---:|
| Can    | 13.3 %    | 73.3 %    | +60.0    |
| Square | 3.3 %¹    | 3.3 %     | +0.0 (NULL/blocked) |

¹ T2.0_square = 1/30 = 3.3% is **invalid** (slate bug: `--prior-bank-dataset`
was not passed, so `demo_1nn` fell back to an empty bank). T2.1_square = 1/30 =
3.3% and T2.4_square = 0/30 = 0.0% ran with the correct banks but sit at the
canonical-mass Square hybrid floor (Slate G measured 0-8% under every knob).
**Net:** T2 transfer to Square is a NULL/blocked result — it cannot be
demonstrated until the upstream Square hybrid pipeline (obs-schema /
coordinate-frame) is fixed. This is superseded as the Square story by the later
**Square-DP** task-specific prior (66.7% standalone, 70.0% hybrid); see
`square_dp_prior.md`.

### Predicted outcomes

If T2 transfers: Square shows a positive T2.1 - T2.0 gap (any
magnitude); the gap need not equal Can's 60 pp but should be
non-trivial (> 20 pp) and the CIs should not overlap T2.0's CI.

### Failure mode

If T2.1_square ~ T2.0_square (no adaptation gain): Square's
mass=1.3 perturbation doesn't disrupt the canonical hybrid enough
to demonstrate T2 (Lift-like null).  Falls back to: probe
larger Square perturbations OR Square goal-perturbation.

### Integration into docs/results.md

The "Lift cross-task perturbed adaptation" subsection (currently
ending with "future work" pointing at Square) gets either:

(a) replaced by a "Cross-task confirmation on Square" subsection
    reporting the T2.1_square gap, OR
(b) supplemented with a "Square also null" paragraph if the
    perturbation doesn't bite Square either.
