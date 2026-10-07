# MJX Gate-4 closeout note

**Status:** Gates 1-3 complete; Gate-4 Colab GPU measurement complete.  
**Primary answer:** MJX is working as an opt-in planning backend with a measured Colab L4 speedup, but it should remain an engineering-appendix path rather than being promoted to the canonical robosuite-eval path.

## What is already working

- Simplified PickPlaceCan MJX scene validates and rolls out.
- JAX OSC controller is implemented and parity-tested.
- OSC math parity vs robosuite reference: `8.03e-12 Nm`.
- MJX cost rank correlation vs robosuite shaped reward: `rho = +0.485` on the relaxed Gate-2 criterion.
- `experiments/04_mbd_baseline/run_can.py --mjx` runs end-to-end and writes JSON on Mac CPU.
- GPU fail-fast smoke exists: `experiments/04_mbd_baseline/test_mjx_gpu_smoke.py`.
- Colab Gate-4 notebook exists: `experiments/04_mbd_baseline/colab_mjx_gate4_2026_06_07.ipynb`.
- Colab L4 Gate-4 run completed on 2026-06-07: 60 rollouts, notebook wall `1310.1 s` (`21.8 min`), JSON `total_wall_s = 1304.711483141`, mean/median plan time `14.733 / 14.731 s`, speedup `13.80x` versus the nominal 5 h Mac/robosuite CPU baseline.
- GCE L4 path exists but was quota-blocked.

## Gate-4 outcome

The actual GPU run satisfied the execution requirements:

1. JAX backend must be `gpu`.
2. `test_mjx_gpu_smoke.py` must pass.
3. `run_can.py --mjx` at H=120, M=16, K=10, 20×3 plans must produce `can_mjx.json`.
4. The measured speedup must be recorded against the Mac/robosuite CPU baseline.

However, the measured `13.80x` speedup is below the prior promotion gates (`20x` for this note, `30x` in `docs/mjx_migration.md`). Cite MJX as **implemented, GPU-measured, and useful for engineering iteration**, not as the canonical headline evaluation backend.

## Completed closeout path

Colab was used because it was not GCE-quota blocked:

1. Open `experiments/04_mbd_baseline/colab_mjx_gate4_2026_06_07.ipynb`.
2. Select GPU runtime.
3. Run the notebook top-to-bottom.
4. Save `/content/can_mjx.json` and the smoke output.
5. Update `docs/mjx_migration.md` Gate-4 row with actual wall time and speedup.

## Decision gates

- **Promote MJX acceleration claim:** GPU full eval is at least 20× faster than Mac/robosuite CPU and the cost curves remain qualitatively sane.
- **Keep as engineering appendix:** GPU runs but speedup is under 20× or simplified-scene behavior diverges. This is the current outcome.
- **Do not cite:** GPU smoke fails or silently falls back to CPU.
