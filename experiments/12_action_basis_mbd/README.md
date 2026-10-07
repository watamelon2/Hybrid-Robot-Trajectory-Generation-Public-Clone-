# Experiment 12 — Action-basis MBD perturbations

**Status:** _scaffolded._ `basis.py` ships an end-to-end DCT-II
projected variant of `denoise_loop`; the can-task evaluation is
left as a small wrapper script (one will land here once the
preceding wiring is validated).

## Hypothesis

The Exp 05 README:

> the planner's `M=16` cosine-noise perturbations rarely close
> the gap on this high-frequency contact event.

Default MBD samples `eps[t] ~ N(0, σ²)` per timestep, so the
`M=16` budget is spread across an `H · A`-dim hyperball where
the vast majority of mass is high-frequency jitter that the
controller filters out anyway. Re-parameterising the noise as
`eps = eps_basis @ B` for a low-frequency basis `B` of size
`n_modes × H` concentrates the same `M` samples on coherent
multi-step shifts.

## Wiring it in

`basis.denoise_loop_basis` is a drop-in replacement for
`mbd.denoise_loop`. To run the same can-task eval you can:

```python
from experiments['12_action_basis_mbd'].basis import denoise_loop_basis
# or via importlib because the dir name starts with a digit.

mu, log = denoise_loop_basis(
    initial_mean, rollout_fn, cost_fn, params, rng,
    n_modes=8,
)
```

The end-to-end runner is left out of this branch on purpose: it
is a copy of `experiments/05_hybrid_method/run_can.py` with one
function call swapped, and a small CLI flag for `--n-modes`. Add
once the upstream validation (lambda sweep, sigma sweep) is
done so we don't multiply the parameter space prematurely.

## Suggested first sweep

```
n_modes in {4, 8, 16, 32, H=120}     # 120 == "no projection"
```

Pair with `--prior flow --lambda-prior 0.5` from exp 07 so the
basis ablation is on top of the strongest prior. The Exp 05
14.8 % MPC SR->success number is the appropriate yardstick.

## Math note

The DCT-II basis used here has the property
`B B^T = I_{n_modes}` and `B^T B != I_H` (rank-`n_modes`
projection). Sampling `eps_basis ~ N(0, I)` gives an
`eps = eps_basis @ B` with per-step variance proportional to
`sum_m B[m, t]^2`, which we rescale so the *average* per-step
std matches `σ_k`. The schedule contract therefore holds in
expectation, not pointwise.
