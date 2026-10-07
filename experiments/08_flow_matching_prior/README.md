# Experiment 08 — Flow-matching prior `v_θ`

**Status:** _scaffolded._ Trainer + adapter + LQ sanity ship in this
branch; the can-task evaluation rows are filled in by exp 07's
λ-sweep runner once `v_theta.pt` exists.

This is **Tier 1a** in `docs/plans.md` -- the single highest-leverage
change. Replaces exp 03's deterministic MLP-BC stand-in with a
*conditional flow-matching* velocity field
`v_θ(x_t, t, s_0)` over H-step trajectories conditioned on the demo's
initial obs.

## What's in this directory

| file              | role                                                         |
| ----------------- | ------------------------------------------------------------ |
| `model.py`        | `VelocityField` MLP + conditional-FM loss + ODE sampler.     |
| `dataset.py`      | Demo -> `(action_traj, init_obs)` extractor (reuses exp 03's `OBS_KEYS`). |
| `train.py`        | CLI trainer; saves `v_theta.pt` + `train_log.json`.          |
| `adapter.py`      | `FlowPrior` wrapper: `sample_initial_mean(s0)` + `make_prior_score_fn(s0, K)`. |
| `lq_sanity.py`    | 2-D ring sanity for the trainer + composite-update bridge.   |

## Math sketch

Conditional flow-matching with the linear-interpolant coupling
(Lipman et al., ICLR 2023):

```
x0 ~ N(0, I);  x1 ~ p_data;  t ~ U(0, 1)
x_t   = (1 - t) x0 + t x1
v_t*  = x1 - x0
L     = E || v_θ(x_t, t, s0) - v_t* ||^2
```

For MBD's per-step composite update we expose `v_θ` as an
approximate score:

```
grad log p_t(x) ≈ (t v_θ(x, t, s0) - x) / (1 - t)^2
```

The `(k -> t)` schedule inside `make_prior_score_fn` is linear in
the MBD iteration index, mirroring the existing cosine `σ`
schedule.

## Reproducing

End-to-end smoke test (≈10 s, CPU, no robomimic dataset):

```bash
python experiments/08_flow_matching_prior/lq_sanity.py
```

The headline trainer (≈few minutes on CPU for the 20-demo slice;
faster on GPU):

```bash
python experiments/08_flow_matching_prior/train.py \
    --task can --n-demos 20 --horizon 120 --epochs 2000
```

Wiring check before the long run:

```bash
python experiments/08_flow_matching_prior/train.py --smoke
```

The resulting `results/v_theta.pt` is consumed by exp 07's
λ-sweep runner via the `--prior-checkpoint` flag.
