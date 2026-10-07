# Experiment 12 — Action-basis MBD perturbations

**Lifecycle:** **scaffolded**. `basis.py` ships an end-to-end
DCT-II-projected variant of `denoise_loop`; the can-task
evaluation script is left as a small wrapper (one will land here
once the upstream λ-sweep + σ-sweep settle).

**In-tree deep doc:**
[`experiments/12_action_basis_mbd/README.md`](../../experiments/12_action_basis_mbd/README.md).

## Hypothesis

From [`05_hybrid_method.md`](05_hybrid_method.md):

> the planner's `M=16` cosine-noise perturbations rarely close
> the gap on this high-frequency contact event.

Default MBD samples `eps[t] ~ N(0, σ²)` per timestep, so the
`M=16` budget is spread across an `H · A`-dim hyperball where
the vast majority of mass is high-frequency jitter that the
robosuite OSC controller filters out anyway.

**Idea:** re-parameterise the noise as `eps = eps_basis @ B` for
a low-frequency basis `B` of size `n_modes × H`. This
concentrates the same `M` samples on coherent multi-step shifts
rather than per-step jitter. The DCT-II basis is the natural
choice (orthonormal, smoothest n_modes).

## Wiring

`basis.denoise_loop_basis` is a drop-in replacement for
[`mbd.denoise_loop`](../../experiments/04_mbd_baseline/mbd.py):

```python
from experiments['12_action_basis_mbd'].basis import denoise_loop_basis
# (importlib is required because the dir name starts with a digit)

mu, log = denoise_loop_basis(
    initial_mean, rollout_fn, cost_fn, params, rng,
    n_modes=8,
)
```

The end-to-end runner is left out of this branch on purpose: it
is a copy of `experiments/05_hybrid_method/run_can.py` with one
function call swapped, and a small CLI flag for `--n-modes`. Add
once the upstream validation (λ sweep, σ sweep) is done so we
don't multiply the parameter space prematurely.

## Suggested first sweep

`n_modes ∈ {4, 8, 16, 32, 120}` (120 == "no projection", reduces
to the default exp-05 protocol).

Pair with `--prior flow --lambda-prior 0.5` (exp 07) so the basis
ablation runs on top of the strongest prior. The exp 05b
**14.8 % MPC strong-reach → success** number is the right
yardstick.

## Math note

The DCT-II basis used here has the property
`B B^T = I_{n_modes}` and `B^T B ≠ I_H` (rank-`n_modes`
projection). Sampling `eps_basis ~ N(0, I)` gives
`eps = eps_basis @ B` with per-step variance proportional to
`Σ_m B[m, t]^2`, which we rescale so the *average* per-step
std matches `σ_k`. The schedule contract therefore holds in
expectation, not pointwise.

## Files

| Path | Role |
|------|------|
| [`experiments/12_action_basis_mbd/basis.py`](../../experiments/12_action_basis_mbd/basis.py) | DCT-II basis + `denoise_loop_basis` wrapper. |
| `experiments/12_action_basis_mbd/run_can.py` | (planned) thin wrapper around `05_hybrid_method/run_can.py` with `--n-modes`. |

## Cross-references

- **Predecessors**:
  [`04_mbd_baseline.md`](04_mbd_baseline.md) (the MBD loop being
  re-parameterised),
  [`05_hybrid_method.md`](05_hybrid_method.md) (the high-frequency
  contact-event narrative this experiment attacks).
- **Likely consumer**:
  [`15_unified_hybrid.md`](15_unified_hybrid.md) — once the
  ablation lands, the unified driver should grow `--n-modes`
  too, mirroring how σ_max and λ_prior were promoted from exp
  05/07 to exp 15.

## Status / open questions

- [x] `basis.py` ships and is unit-testable.
- [ ] Land the can-task wrapper (`run_can.py`) and run the
      `n_modes ∈ {4, 8, 16, 32, 120}` sweep.
- [ ] Decide whether the per-step variance compensation is the
      right one. Alternative is L2-rescale per draw — both
      should be tried.
- [ ] If it helps, lift the basis flag into exp 15.
