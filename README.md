# Hybrid Robot Trajectory Generation

ECE M237 course project.  We study a hybrid trajectory generator
that combines a **learned demonstration prior** (behaviour cloning,
flow matching, zero-training nearest-neighbour retrieval, or a
trained diffusion policy) with an **online model-based correction**
(denoising diffusion guided by `∇ log p_model` from a hand-designed
cost).  The two scores share a single denoising loop with a
noise-schedule-dependent blend `λ(σ_t)`.  Benchmarks are the five
`robomimic` proficient-human tasks (`lift`, `can`, `square`,
`tool_hang`, `transport`) running in `robosuite`.

## Idea in one paragraph

Pure behaviour cloning compounds errors out-of-distribution; pure
model-based diffusion (MBD) is slow and chronically falls into
local minima of the rollout cost.  Mixing the two — adding a
Gaussian pull toward a *learned* prior at every MBD iteration, with
a noise-schedule-dependent weight — should (in principle) give a
fast inner loop with the prior's coverage and the model's
task-specific correctness.  The project's goal is to (a) measure
the binary-success gap that the bridge buys over each component
alone on `PickPlaceCan` and the four sibling tasks, and (b)
identify *when* the bridge helps as a function of prior quality,
MBD budget `K`, and the bridge weight `λ`.

## Status

Headline numbers as of 2026-06-01 (post-Phase-F, all `n >= 30`,
Wilson 95 % CIs).  Source: [`docs/results.md`](docs/results.md) +
the per-task `AGGREGATE.json` files under `experiments/*/results/`.

### Best result per task on robomimic `ph`

| Task        | Best method | Success | Wilson 95 % CI | n |
|---|---|---:|---|---:|
| `lift`      | retrieval (1-NN) + MBD             | **100.0 %** (60/60) | [94.0, 100.0] |  60 |
| `can`/ph    | trained DP-v3b prior + MBD, λ=2.0  | **86.7 %**  (26/30) | [70.3,  94.7] |  30 |
| `can`/ph    | retrieval (1-NN) + MBD, 4-seed pool | 73.8 %   (177/240) | [68.0,  78.9] | 240 |
| `square`    | retrieval (1-NN) + MBD             |   8.3 %    (5/60)  | [ 3.6,  18.1] |  60 (representation-limited) |
| `tool_hang` | retrieval (1-NN) + MBD             |   0.0 %    (0/40)  | [ 0.0,   8.8] |  40 (sparse-reward bug, patched) |

The headline lift on `can` is `(ep0149, K=10, λ=2.0) = 86.7 %`
— **+13.4 pp** over the same prior at the canonical `λ=0.5`
baseline, at the *same* compute.  A second GREEN cell at
`(ep0149, K=20, λ=0.5) = 83.3 %` (25/30) confirms the lift along
an independent knob.  The mechanism reading: MBD helps **iff**
(a) the prior's score is informative at the operating noise level,
**and** (b) the prior has not saturated the task.  Demonstrated by
the full `ckpt × K × λ` Phase-D grid.

### Experiment index (current generation)

| #  | Experiment                                                                 | Status | Headline |
|----|----------------------------------------------------------------------------|--------|----------|
| 01 | [Demo replay](experiments/01_demo_replay/README.md)                        | done   | upper bound for `can` |
| 02 | [Random-action baseline](experiments/02_random_baseline/README.md)         | done   | 0 / 60 lower bound |
| 03 | [Behaviour cloning](experiments/03_bc_baseline/README.md)                  | done   | 6.7 % on `can` (canonical `[256, 256]` MLP, 72 s training) |
| 04 | [Pure MBD](experiments/04_mbd_baseline/README.md)                          | done   | 0 / 60 on `can` (cost-only, no prior) |
| 05 / 07 / 10 | [BC-prior hybrid + λ-sweep](experiments/07_lambda_sweep/README.md) | superseded | absorbed into the unified pipeline (exp 15) |
| 08 | [Flow-matching + DP-v3b prior](experiments/08_flow_matching_prior/README.md) | done | flow-matching dominated; DP-v3b (CFM, 17 min GPU) is the strong prior |
| 15 | [Unified hybrid (composite score)](experiments/15_unified_hybrid/README.md) | done | the canonical hybrid pipeline used in all current results |
| 16 | [Value-baseline retrieval](experiments/16_value_baseline/README.md)        | done   | bank-size scaling law on `can` (slate `DemoScale_2026_05_25`) |
| 17 | [DP-prior + MBD hybrid](experiments/17_dp_hybrid/README.md)                | done   | Phase-D `ckpt × K × λ` grid, **two GREEN cells** at `ep0149` |

### Week-8 deliverables

The full slide deck for the Week-8 presentation, plus all source
scripts and pre-rendered PDFs / MP4s / PNG previews, lives in
[`presentation/`](presentation/).  Start from
[`presentation/DECK_GUIDE.md`](presentation/DECK_GUIDE.md) — it is
a self-contained, slide-by-slide spec with inline figure previews
and relative links to every asset.  See
[`presentation/README.md`](presentation/README.md) for an overview.

## Repository layout

```
.
├── docs/                                 # project-wide documentation
│   ├── results.md                        # consolidated headline numbers + plots
│   ├── plans.md                          # project roadmap, tier-prioritised
│   ├── run_log.md                        # append-only diary of overnight sessions
│   ├── progress_reports/                 # week{1..7}.tex + IEEEtran class files
│   ├── experiments/                      # one design doc per experiment NN
│   ├── adaptation/                       # adaptation slate specs + handoffs
│   ├── theory/                           # asymmetric-robustness, Pareto bias-variance, ...
│   ├── mjx_migration.md                  # MJX GPU rollout migration runbook
│   ├── remote_run.md                     # Colab Pro + GCE execution guide
│   ├── references/                       # upstream-library / model links
│   ├── papers/                           # canonical references.bib (PDFs gitignored)
│   ├── literature_review/                # literature_review.tex (long form)
│   └── topic_introduction/               # topic_introduction_and_literature_review.tex
├── presentation/                         # Week-8 talk: slide spec + figures + videos
│   ├── DECK_GUIDE.md                     # ← OPEN THIS to build the deck
│   ├── SLIDE_SPEC.md, OUTLINE.md, INDEX.md
│   ├── figures/  (12 PDFs + 11 PNG previews under preview/)
│   ├── videos/   (11 MP4s)
│   └── scripts/  (5 active + 2 defunct build scripts)
├── env/                                  # frozen conda + pip lockfiles
├── experiments/
│   ├── common.py                         # shared sim utilities (env build, v1.4→1.5 OSC shim, ...)
│   ├── 01_demo_replay/                   # open-loop demo replay (upper bound)
│   ├── 02_random_baseline/               # random-action lower bound
│   ├── 03_bc_baseline/                   # MLP-MSE BC baseline ([256, 256] canonical)
│   ├── 04_mbd_baseline/                  # pure MBD + MJX rollout (—mjx flag)
│   ├── 05_hybrid_method/                 # exp-05 warm-start hybrid (superseded by 15)
│   ├── 06_failure_taxonomy/              # rollout-bucket auto-labeller
│   ├── 07_lambda_sweep/                  # λ-prior sweep (BC + flow priors, superseded by 15)
│   ├── 08_flow_matching_prior/           # v_theta train + DP-v3b CFM checkpoints
│   ├── 09_sigma_sweep/, 10_kxreplans_grid/, 11_oracle_replan/, 12_action_basis_mbd/, 13_square_task/, 14_seed_sweep/   # earlier ablations
│   ├── 15_unified_hybrid/                # canonical composite-score pipeline (current hybrid)
│   ├── 16_value_baseline/                # bank-size scaling-law slate
│   └── 17_dp_hybrid/                     # DP-prior + MBD; Phase-D grid + two GREEN cells
├── notebooks/                            # local + Colab runner notebooks
├── scripts/
│   ├── build_mujoco_docs.sh, check_env.py
│   ├── gce/                              # GCE fleet runners
│   ├── local/                            # local-Mac parallel runners + monitors
│   └── remote/                           # install_linux*.sh for GCE / Colab
└── README.md
```

## Setup

The reference environment is conda + pip, Python 3.11, on macOS / Linux.

```bash
# 1. create the conda env (named `hybrid-traj`)
conda env create -f env/environment-from-history.yml
conda activate hybrid-traj

# 2. install the pinned pip-only packages (jax, robomimic, robosuite, ...)
pip install -r env/requirements-explicit.txt

# 3. smoke-test the MuJoCo bindings
python scripts/check_env.py
```

Robomimic does not ship the dataset; download the proficient-human
`can` split once into the conda env's `site-packages`:

```bash
python -m robomimic.scripts.download_datasets \
    --tasks can --hdf5_types low_dim
```

Pinned constraints worth knowing about:

- `mujoco >= 3.2` (we standardise on the new `mujoco` Python bindings; legacy `mujoco_py` is **not** used).
- `robosuite == 1.5.2`. The dataset's `controller_configs` is in v1.4 flat-OSC format; `experiments/common.py` lifts it into v1.5's BASIC composite-controller schema.
- `numpy < 2` (`mink` dependency) which forces `jax < 0.5`.

If `pip check` complains about `mink ↔ numpy`, downgrade numpy
explicitly (`pip install 'numpy<2'`) and reinstall `mink`.

## Documentation

### Top of the funnel

- [`docs/results.md`](docs/results.md) — **consolidated headline
  results** with plots and per-experiment numbers (start here).
- [`docs/plans.md`](docs/plans.md) — project roadmap, tier-
  prioritised follow-up list.
- [`docs/run_log.md`](docs/run_log.md) — append-only diary of
  overnight execution sessions.
- [`presentation/DECK_GUIDE.md`](presentation/DECK_GUIDE.md) —
  the Week-8 slide deck, self-contained.

### Per-experiment

- [`docs/experiments/`](docs/experiments/) — one design doc per
  experiment `NN`, with rationale, ablations, and the
  observed numbers.
- [`docs/adaptation/`](docs/adaptation/) — adaptation slate specs
  (cost-time, dynamics, incremental prior, demo-count) and the
  two-machine launch handoffs.
- [`docs/theory/`](docs/theory/) — asymmetric-robustness model,
  Pareto bias-variance theory, scaling-law derivations.

### Operational

- [`docs/mjx_migration.md`](docs/mjx_migration.md) — MJX GPU
  rollout migration plan + gate-by-gate runbook.
- [`docs/remote_run.md`](docs/remote_run.md) — Colab Pro +
  Google Compute Engine execution guide.
- [`docs/MuJoCo.md`](docs/MuJoCo.md) — focused MuJoCo reference
  for this project (quickstart, sim loop, rendering, troubleshooting).
- [`docs/references/README.md`](docs/references/README.md) —
  upstream links for MuJoCo, `dm_control`, Menagerie, robosuite,
  robomimic, and MBD.

### Long-form

- [`docs/papers/README.md`](docs/papers/README.md) — paper
  library, canonical `references.bib`, Zotero workflow.
- [`docs/literature_review/literature_review.tex`](docs/literature_review/literature_review.tex) — long-form literature review.
- [`docs/topic_introduction/topic_introduction_and_literature_review.tex`](docs/topic_introduction/topic_introduction_and_literature_review.tex) — topic introduction + short literature review.
- [`docs/progress_reports/`](docs/progress_reports/) — weekly
  progress reports (`week{1..7}.tex`).
- [`notebooks/mujoco_official/`](notebooks/mujoco_official/) —
  the seven official MuJoCo tutorial notebooks, checked in for
  teammate ease-of-use.

## Running an experiment

Each experiment is self-contained and pulls shared sim primitives from
`experiments/common.py`. Reproducing the headline numbers in the
status table:

```bash
# 01 — demo replay (upper bound, ~1 min)
python experiments/01_demo_replay/replay.py        --task can --n-demos 20
# 02 — random actions (lower bound, ~1 min)
python experiments/02_random_baseline/random_actions.py \
                                                   --task can --n-demos 20 --n-rollouts 3
# 03 — behaviour cloning (train + eval, ~4 min). Default --hidden 256
#      matches the canonical checkpoint; older runs used --hidden 512.
python experiments/03_bc_baseline/train.py         --task can --epochs 500
python experiments/03_bc_baseline/eval.py          --task can --n-demos 20 --n-rollouts 3
# 04 — online MBD (no prior); full 20 x 3 eval (~78 min)
python experiments/04_mbd_baseline/run_can.py      --n-demos 20 --n-rollouts 3 \
                                                   --no-live-view --seed 0
# 05 — hybrid BC prior + MBD + 1 MPC replan; full 20 x 3 eval (~3.7 h)
python experiments/05_hybrid_method/run_can.py     --n-demos 20 --n-rollouts 3 \
                                                   --seed 0 --no-live-view \
                                                   --mpc-replans 1 --mpc-chunk-steps 60 \
                                                   --deterministic-bc-reset \
                                                   --output experiments/05_hybrid_method/results/can_ph_mpc.json
# 07 — λ-prior sweep (BC prior, full 6 lambdas in parallel, ~1-2 h on M4 Max)
bash scripts/local/run_lambda_bc_full.sh
# 10 — K × R pareto grid (9 cells in parallel, ~5-15 h on M4 Max)
bash scripts/local/run_pareto_full.sh
# Or chain 10 → 07 with overnight automation (waits, runs analyzers, marks done):
nohup bash scripts/local/chain_bc_swap_rerun.sh <9 exp10 PIDs> &
```

Outputs land in `experiments/NN_*/results/*.json`. See each
experiment's `README.md` for the full CLI, observed numbers, ablations,
and the caveats (in particular the robosuite 1.4 → 1.5 observation drift
documented in `experiments/03_bc_baseline/README.md`).

## Branching convention

As of 2026-06-01, all major work lives on `main`.  Recent merges
(in reverse chronological order):

- `feat/presentation-week8` — the Week-8 deck (`presentation/`
  folder).  Merged as `bb942d4`.
- `feat/docs-2026-06-01-post-merge-sweep` — doc sweep waves
  1–3 closure headers.  Merged as `89b1d78`.
- `feat/adapt-pareto-analysis` — proxy (bias², variance) Pareto-
  dominance verification + asymmetric-robustness theory doc.
- `feat/dp-phase-c` — DP Phase-C + Phase-D with two GREEN cells
  at the Q1-peak prior (`8e97ca3`).

Per-experiment progression is readable via commit subjects of
the form `feat(expNN): …` / `results(expNN): …` / `dp(phase-X): …`.
See [`docs/run_log.md`](docs/run_log.md) for the full chronology.
Retired topic branches are kept on origin as historical record;
[`docs/branch_plan.md`](docs/branch_plan.md) tracks active and
retired branches.

## References

Key papers the project builds on:

- Pan et al., *Model-Based Diffusion for Trajectory Optimization* (NeurIPS 2024). Upstream reference impl: <https://github.com/LeCAR-Lab/model-based-diffusion>.
- Janner et al., *Planning with Diffusion for Flexible Behavior Synthesis* (ICML 2022).
- Lipman et al., *Flow Matching for Generative Modeling* (ICLR 2023).
- Mandlekar et al., *What Matters in Learning from Offline Human Demonstrations for Robot Manipulation* (CoRL 2021) — the `Robomimic` benchmark and dataset format.

Upstream library and model links live in [`docs/references/README.md`](docs/references/README.md).