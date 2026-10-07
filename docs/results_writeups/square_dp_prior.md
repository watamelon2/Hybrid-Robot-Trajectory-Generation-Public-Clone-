# Square-specific Diffusion Policy prior follow-up

**Slate:** `experiments/08_flow_matching_prior/slate_square_dp_prior_2026_06_07.sh`
**Status:** completed 2026-06-08.
**Primary question:** was Square's low learned-prior performance caused by the score-bridge mechanism itself, or by using a Can-centric observation/prior schema on a contact-rich insertion task?

**Primary answer:** Square improves sharply once the DP prior is trained on a Square-specific `square_dp` observation schema. The standalone prior reaches 20/30 = 66.7%, and the hybrid reaches 21/30 = 70.0% at both `lambda=0.5` and `lambda=1.0`, versus 16/30 = 53.3% for the `lambda=0` warm-start/MBD-only control under the same short-horizon DP setup.

## Motivation

The June-07 Square horizon diagnostic showed that zero-training retrieval can reach 50.0% only at the long-horizon, strong-bridge setting `H=300, lambda=2`. That left a representation question: Square might need task-specific object-relative features rather than more product/gate or tempering variants. This follow-up trained a DP prior directly on Square with the DP-style compact observation schema.

## Implementation

The follow-up adds a `square_dp` schema analogous to `can_dp`:

| Key | Dim | Source |
|---|---:|---|
| `SquareNut_pos` | 3 | dataset `obs/object[:, 0:3]` |
| `SquareNut_quat` | 4 | dataset `obs/object[:, 3:7]` |
| `robot0_eef_pos` | 3 | native obs |
| `robot0_eef_quat` | 4 | native obs |
| `SquareNut_to_eef_pos_synth` | 3 | runtime/dataset synthetic relative position |
| `SquareNut_to_eef_quat_synth` | 4 | runtime/dataset synthetic relative quaternion |
| `robot0_gripper_qpos` | 2 | native obs |

Total conditioning dimension: 23. The dataset path resolver maps `*_dp` task aliases back to the underlying Robomimic task directory, so `square_dp` uses `datasets/square/ph/low_dim_v141.hdf5`.

## Training and evaluation protocol

| Stage | Setting |
|---|---|
| Training | `task=square_dp`, 200 demos, `H=16`, `T_obs=2`, 200 epochs |
| Normalisation | action min-max ON, obs min-max ON |
| Checkpoint | `v_theta_dp_square_tobs2_h16_ep0199.pt` |
| Standalone eval | 30 test seeds, `max_env_steps=400`, `n_action_steps=8` |
| Hybrid eval | `task=square`, `H=16`, chunk=8, `R=49`, `K=10`, `M=16`, `sigma_max=0.10`, `sigma_min=0.05` |
| Test seeds | `100000` through `100029` |

The final training loss was 0.0155 after 8563.9 s on the MPS device.

## Results

| Method | Success | Rate | Wall time | Output |
|---|---:|---:|---:|---|
| Standalone `square_dp` DP | 20/30 | 66.7% | 1249.1 s | `standalone_square_dp_test30.json` |
| Hybrid `lambda=0.0` | 16/30 | 53.3% | 8070.3 s | `unified_flow_pull_lam-0.0_K-10_R-49_square_dp_pull_test30.json` |
| Hybrid `lambda=0.5` | 21/30 | **70.0%** | 5544.3 s | `unified_flow_pull_lam-0.5_K-10_R-49_square_dp_pull_test30.json` |
| Hybrid `lambda=1.0` | 21/30 | **70.0%** | 5866.4 s | `unified_flow_pull_lam-1.0_K-10_R-49_square_dp_pull_test30.json` |
| Hybrid `lambda=2.0` | 19/30 | 63.3% | 5863.4 s | `unified_flow_pull_lam-2.0_K-10_R-49_square_dp_pull_test30.json` |

Summary file: `experiments/08_flow_matching_prior/results/square_dp_prior_2026_06_07/unified_summary_flow_pull_K-10_R-49_square_dp_pull_test30.json`.

## Analysis

- **The Square learned-prior issue is representation-specific.** The `square_dp` prior reaches 66.7% standalone, so the previous low Square learned-prior cells were not evidence that DP-style priors cannot solve Square.
- **MBD adds a small but positive lift at moderate bridge weight.** The best hybrid cells improve from 66.7% standalone to 70.0%, and from 53.3% at `lambda=0` to 70.0% with `lambda in {0.5, 1.0}`.
- **Over-anchoring is visible.** `lambda=2.0` drops to 63.3%, below the `lambda=0.5/1.0` peak and below standalone.
- **This supersedes the urgent Square-replication plan.** The earlier `H=300, lambda=2` zero-training result remains useful as a long-horizon retrieval diagnostic, but the stronger path for Square is now a task-specific DP prior plus moderate pull-mode hybridisation.

## Roadmap consequence

Near-term Square work should shift from asking whether Square is solvable to improving robustness and diagnosing residual failures:

1. add failure-phase labels for the 9 failures at the peak `lambda=0.5/1.0`,
2. test object-relative / peg-relative insertion features if failure phases cluster near alignment,
3. only then consider longer-horizon hybrid variants or subgoal decomposition.

Do not prioritize more static product/gate, adaptive sigma, or temper-p sweeps for Square; the stronger evidence points to task representation and phase structure.
