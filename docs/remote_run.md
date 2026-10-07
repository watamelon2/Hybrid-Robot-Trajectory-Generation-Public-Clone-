# Remote execution: Colab Pro + Google Compute Engine

Two complementary paths because the workload is split:

- **Colab Pro** — interactive / GPU-friendly. Best for the
  flow-matching trainer (exp 08), all smoke tests, the post-hoc
  analyses (exp 06), and one-off Exp 05 runs you want to watch.
- **GCE (spot VM)** — fire-and-forget for the long sequential
  CPU sweeps (exp 14 seed pool ~19 h, exp 09/10 sweeps ~7-10 h
  each). Auto-uploads JSON results to a GCS bucket and
  self-shuts-down.

All scaffolding lives in this branch; nothing references your
local machine. Both paths clone
`feat/further-experiments` from GitHub directly.

## Colab Pro

1. Open <https://colab.research.google.com/github/watamelon2/Hybrid-Robot-Trajectory-Generation/blob/feat/further-experiments/notebooks/remote/colab_runner.ipynb>.
2. Set runtime type to **CPU** (default) for everything *except*
   `RUN_08_TRAIN_VTHETA`, where a T4 / L4 GPU shaves the trainer
   from ~1 h to a few minutes.
3. Run cells top-to-bottom. The notebook:
   - mounts Drive at `/content/drive/MyDrive/hybrid-traj`,
   - clones the branch, runs `scripts/remote/install_linux.sh`,
   - runs whatever `RUN_*` flags you toggled,
   - syncs every `experiments/*/results/**` file into Drive.
4. Optional: provide a `GH_TOKEN` Colab secret to let the last
   cell push the result JSONs back to the branch.

The notebook auto-reuses an already-cloned repo on warm restarts,
so re-running cells 1-3 after a runtime reset is a few seconds.

### What runs cleanly on Colab today

| RUN flag                | wall (CPU)     | wall (GPU)  | notes                                |
| ----------------------- | -------------: | ----------: | ------------------------------------ |
| `RUN_06_TAXONOMY`       | <1 s           | n/a         | post-hoc, no env                     |
| `RUN_08_LQ_SANITY`      | ~10 s          | n/a         | torch only                           |
| `RUN_08_TRAIN_VTHETA`   | ~30 min full   | ~3-5 min    | smoke ~30 s either way               |
| `RUN_07_LAMBDA_BC`      | ~3 min smoke   | n/a         | full sweep is GCE territory          |
| `RUN_07_LAMBDA_FLOW`    | depends on 08  | n/a         | needs `v_theta.pt` first             |
| `RUN_05_HYBRID`         | ~80 min full   | n/a         | sequential robosuite sim             |
| `RUN_05B_HYBRID_MPC`    | ~3.7 h full    | n/a         | exceeds free Colab session length    |
| `RUN_11_ORACLE_REPLAN`  | ~10-15 min     | n/a         | depends on Exp 05 JSON in repo       |
| `RUN_14_SEED_SWEEP`     | ~19 h          | n/a         | **don't run on Colab; use GCE**      |

## Google Compute Engine

One-time setup:

```bash
# 1. authenticate gcloud locally if you haven't.
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. create a GCS bucket for result uploads (one-time).
gsutil mb -l us-central1 gs://your-hybrid-traj-results

# 3. (optional) verify the launcher is on PATH:
ls scripts/gce/launch.sh
```

Launch a sweep:

```bash
PROJECT=YOUR_PROJECT_ID \
BUCKET=gs://your-hybrid-traj-results \
    ./scripts/gce/launch.sh seed-sweep
```

Available experiments (first arg to `launch.sh`):

| name             | spot wall | spot cost (us-central1) |
| ---------------- | --------: | ----------------------: |
| `taxonomy`       | <1 min    | < $0.01                 |
| `train-vtheta`   | ~1 h CPU  | ~$0.03                  |
| `exp04`          | ~80 min   | ~$0.04                  |
| `exp05`          | ~80 min   | ~$0.04                  |
| `exp05b`         | ~3.7 h    | ~$0.10                  |
| `sigma-sweep`    | ~7 h      | ~$0.20                  |
| `kxreplans`      | ~10 h     | ~$0.30                  |
| `lambda-sweep`   | ~8 h      | ~$0.20                  |
| `seed-sweep`     | ~19 h     | ~$0.50                  |

Default machine: `e2-standard-8` spot (8 vCPU, 32 GB RAM).
Override via env vars:

```bash
MACHINE_TYPE=e2-standard-16 ZONE=us-west1-b ./scripts/gce/launch.sh exp05b
```

Watch a running VM:

```bash
gcloud compute instances tail-serial-port-output \
    hybrid-traj-seed-sweep-... --zone us-central1-a
```

Pull results when done (`gsutil` will print the exact URL on
launch):

```bash
gsutil -m cp -r gs://your-hybrid-traj-results/runs/<INSTANCE>/ ./gce-results/
```

Cancel early:

```bash
gcloud compute instances delete <INSTANCE> --zone us-central1-a --quiet
```

## Running it from another machine

The repo stays private; VMs fetch a tarball from GCS instead of
git-cloning. You only need gcloud + the scripts on the machine
that *launches* VMs.

```bash
# 1. install gcloud (if missing)
curl -fsSL -o /tmp/gcloud.tar.gz \
    https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar xf /tmp/gcloud.tar.gz -C ~ && rm /tmp/gcloud.tar.gz
~/google-cloud-sdk/install.sh --quiet --command-completion=true \
    --path-update=true --rc-path=$HOME/.zshrc
source ~/.zshrc
gcloud components install beta --quiet

# 2. authenticate to the same project
gcloud auth login
gcloud config set project hybrid-traj-robin-2026

# 3. clone the branch (one-time; or just copy the scripts/ dir)
git clone https://github.com/watamelon2/Hybrid-Robot-Trajectory-Generation.git
cd Hybrid-Robot-Trajectory-Generation
git checkout feat/further-experiments

# 4. upload the repo tarball to GCS (repeat after any code change)
export PROJECT=hybrid-traj-robin-2026
export BUCKET=gs://hybrid-traj-robin-2026-results
./scripts/gce/upload_repo.sh

# 5. fire the plan (see "Recommended phased run" below)
```

## Recommended phased run

Defaults assume the project + bucket already exist (one-time setup
above). All commands are non-blocking; each `&` block returns once
the VMs are *submitted*, not finished. Use
`gcloud compute instances list --filter="name~'^ht-'"` to watch.

```bash
export PROJECT=hybrid-traj-robin-2026
export BUCKET=gs://hybrid-traj-robin-2026-results
```

### Phase 0 — pipeline smoke (~5 min, ~$0.02)

```bash
./scripts/gce/launch.sh taxonomy
```

Wait for the auto-shutdown (≈5 min). Confirm the result tarball
appears under `$BUCKET/runs/hybrid-traj-taxonomy-*/`. Only proceed
if this round-trips cleanly.

### Phase 1 — train v_theta + Exp 04 sanity in parallel (~80 min wall, ~$0.50)

```bash
# Quick smoke: 5 demos × 1 rollout ≈ 1 h each on CPU
./scripts/gce/launch.sh train-vtheta &   # ~1 h on CPU; saves to $BUCKET/v_theta/v_theta.pt
./scripts/gce/launch.sh exp04 &          # ~1 h; verifies the new mbd hook
wait

# For full 20-demo × 3-rollout runs, override the defaults:
N_DEMOS=20 N_ROLLOUTS=3 ./scripts/gce/launch.sh exp04   # ~10 h on CPU!
```

### Phase 2 — λ-sweep (the headline figure) (~80 min wall, ~$3)

Six VMs in parallel, one per λ. Each fetches `v_theta.pt` from the
stable `$BUCKET/v_theta/` path before running.

```bash
./scripts/gce/fanout.sh lambda-sweep
```

### Phase 3 — 3-seed pool (Tier 2c) (~3.7 h wall, ~$9)

Nine VMs (3 methods × 3 seeds), `--deterministic-bc-reset` on:

```bash
./scripts/gce/fanout.sh seed-sweep
```

After they all finish, pool locally:

```bash
mkdir -p experiments/14_seed_sweep/results
gsutil -m cp 'gs://hybrid-traj-robin-2026-results/runs/ht-0*/**/*.json' \
    experiments/14_seed_sweep/results/
python3 experiments/14_seed_sweep/pool.py \
    --in experiments/14_seed_sweep/results \
    --out experiments/14_seed_sweep/results/summary.json
```

### Phase 4 — diagnostics in parallel (~10 h wall, ~$15)

```bash
./scripts/gce/launch.sh sigma-sweep        &
./scripts/gce/fanout.sh  pareto            &   # K x replans grid, 9 cells
./scripts/gce/launch.sh  custom            &   # oracle replan; ~30 min
wait
```

For `oracle-replan`, set:
```bash
NAME=ht-oracle-$(date +%s) \
CUSTOM_CMD='python experiments/11_oracle_replan/run.py --n-rollouts 3' \
    ./scripts/gce/launch.sh custom
```

## Caveats

- `robomimic.scripts.download_datasets` puts the HDF5 inside the
  installed package; on every fresh VM/Colab runtime this is a
  ~50 MB download triggered by `scripts/remote/install_linux.sh`.
  Set `SKIP_DATASET=1` if you've already pre-staged the dataset
  into the working dir (e.g. via Drive).
- The scripts pin `numpy<2` and `jax<0.5` per the existing repo
  constraint (`mink` ↔ `numpy`).
- GCE spot VMs can be preempted. The startup script is idempotent
  so re-launching with the same `NAME` resumes the run from the
  last successful clone, but **mid-experiment progress is lost**.
  For long sweeps, prefer non-spot (`PREEMPTIBLE=0`) or break
  the sweep into shorter cells and submit each separately.
- The startup script auto-shuts-down by default. Pass
  `shutdown-when-done=0` via `--metadata` if you want to SSH in
  and inspect the VM after the run.
