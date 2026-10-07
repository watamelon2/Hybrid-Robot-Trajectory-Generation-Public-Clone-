# Phase-D Q2 + Q3 on GCP

Run the λ-sweep + K-sweep at `ep0049` on a single GCP CPU VM, all 7
combos in parallel. Wall ≈ **8-12 hours**, total cost ≈ **$13** at
`c3-standard-22` on-demand pricing (`$1.05/hr`).

This bypasses Colab's 24-hour session limit and uses far less wall
than running on a local Mac with 4-parallel CPU contention.

## What this runs

| run tag | K | λ_prior | story |
|---|---:|---:|---|
| `q2_lam0p1`  | 10 | 0.1  | Q2 — light prior bias |
| `q2_lam0p25` | 10 | 0.25 | Q2 |
| `q2_lam1p0`  | 10 | 1.0  | Q2 |
| `q2_lam2p0`  | 10 | 2.0  | Q2 — heavy prior bias |
| `q3_K5`      | 5  | 0.5  | Q3 — shallow MBD |
| `q3_K20`     | 20 | 0.5  | Q3 — deeper MBD |
| `q3_K40`     | 40 | 0.5  | Q3 — deepest MBD |

Each produces a JSON with `train_n_succ` / `test_n_succ` so we can
fill in the Phase-D λ-curve and K-curve plots.

## One-time prerequisites

1. `gcloud` CLI installed locally and authenticated:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```
2. Compute Engine API enabled in your project.
3. (Optional) A GCS bucket if you want to push results there. Otherwise
   we just `scp` results back to your Mac.

## Step 1: create the VM

```bash
# Pick a zone with c3 availability:
ZONE="us-central1-a"
VM_NAME="phase-d-q2q3"

gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --machine-type=c3-standard-22 \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --boot-disk-size=30GB \
    --boot-disk-type=pd-balanced \
    --scopes=compute-rw \
    --metadata=enable-oslogin=TRUE
```

The `--scopes=compute-rw` lets the VM call the Compute Engine API to
stop itself when the run finishes (`--auto-stop` flag).

After ~30 sec the VM is ready. Verify:

```bash
gcloud compute instances list --filter="name=$VM_NAME"
```

## Step 2: copy ckpt + dataset to the VM

The ckpt is ~509 MB; the dataset is ~30 MB.

```bash
# Adjust these paths to where you have the files locally:
CKPT_LOCAL="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-dp/experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0049.pt"
DATA_LOCAL="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-dp/datasets/can/ph/low_dim_v141.hdf5"

gcloud compute scp "$CKPT_LOCAL" "$VM_NAME":~ --zone="$ZONE"
gcloud compute scp "$DATA_LOCAL" "$VM_NAME":~ --zone="$ZONE"
```

(If you don't have `low_dim_v141.hdf5` locally, fetch it via the
robomimic dataset download or copy from Drive — same file as what
the Phase-B notebook used.)

## Step 3: bootstrap the VM (install deps + clone repo)

SSH in and run the bootstrap script straight from GitHub:

```bash
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
    curl -sSL https://raw.githubusercontent.com/watamelon2/Hybrid-Robot-Trajectory-Generation/feat/dp-phase-c/scripts/cloud/gcp_bootstrap.sh | bash
"
```

This installs:
- Debian build essentials + MuJoCo's OpenGL deps
- Miniconda to `~/miniconda3`
- A `hybrid-traj` Python 3.11 env
- `torch` (CPU), `numpy==2.0.2`, `robosuite==1.5.2`, `mujoco==3.8.1`, `robomimic==0.3.0`
- The repo at `~/HRTG-dp` on `feat/dp-phase-c`

Wall: **~5-8 minutes**. The script is idempotent so re-running just
verifies the install is correct.

## Step 4: run all 7 evals in parallel (with auto-stop)

In a detached `screen` session so it survives SSH disconnects:

```bash
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
    screen -dmS phase_d_q2q3 bash -c '
        cd ~/HRTG-dp &&
        bash scripts/cloud/run_phase_d_q2q3.sh \
            --ckpt ~/v_theta_dp_tobs2_h16_v3b_ep0049.pt \
            --dataset ~/low_dim_v141.hdf5 \
            --auto-stop 2>&1 | tee ~/run.log
    '
"
```

Monitor progress (from your Mac):

```bash
# top-level master log
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="tail -20 ~/run.log"

# per-run log (e.g. q3_K40)
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
    tail -20 ~/phase_d_q2q3_results/q3_K40/run.log
"

# all live processes on the VM
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
    ps -A -o pid,etime,command | grep run_can | grep -v grep
"
```

When all 7 finish, the script runs the aggregation cell (prints a
table), then calls the Compute Engine API to **stop** the VM (charges
drop to ~\$0.04/GB/month for the boot disk only).

## Step 5: pull results back to local Mac

```bash
LOCAL_OUT="experiments/17_dp_hybrid/results/phase_d_2026_05_30/gcp_q2q3"
mkdir -p "$LOCAL_OUT"

# Recursive scp of the whole output dir:
gcloud compute scp --recurse \
    "$VM_NAME":~/phase_d_q2q3_results/* \
    "$LOCAL_OUT/" \
    --zone="$ZONE"
```

You should now have 7 sub-dirs under `$LOCAL_OUT/`, each with a
`unified_flow_lam-X_K-Y_R-49.json` and a `run.log`.

## Step 6: tear down (delete the VM)

After confirming your results are downloaded:

```bash
gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
```

Final cost on `c3-standard-22` (Sapphire Rapids, ~\$1.05/hr):
- 10-12 hr compute → **~\$11-13**
- Boot disk 30 GB × ~10 hr → **~\$0.05**
- Egress (a few KB of JSONs) → \$0

## Customizing the run

The `run_phase_d_q2q3.sh` script accepts:

| flag | default | purpose |
|---|---|---|
| `--ckpt PATH` | (required) | DP ckpt to use as prior (ep0049) |
| `--dataset PATH` | (required) | robomimic Can/ph HDF5 |
| `--output-dir DIR` | `~/phase_d_q2q3_results` | per-run sub-dirs land here |
| `--auto-stop` | off | stop VM via Compute API when done |
| `--runs-filter STR` | (none) | substring match on run tag; e.g. `--runs-filter q3_K40` runs only that one |
| `--dry-run` | off | print planned runs without executing |

### Examples

Run only the K-sweep:
```bash
bash scripts/cloud/run_phase_d_q2q3.sh \
    --ckpt ~/v_theta_dp_tobs2_h16_v3b_ep0049.pt \
    --dataset ~/low_dim_v141.hdf5 \
    --runs-filter q3
```

Smaller machine, run Q2 and Q3 sequentially:
```bash
# tab 1: run Q2
bash scripts/cloud/run_phase_d_q2q3.sh --runs-filter q2 ...

# tab 2: run Q3 separately
bash scripts/cloud/run_phase_d_q2q3.sh --runs-filter q3 ...
```

## Troubleshooting

**"sudo apt-get not found"** — you're on a non-Debian image. Use
Ubuntu 22.04 LTS or stick with `--image-family=debian-12`.

**MuJoCo render errors** — we install `libgl1-mesa-glx`, `libosmesa6`,
etc. in the bootstrap. If you still hit issues, MuJoCo defaults to
`MUJOCO_GL=egl` on headless boxes. Try `export MUJOCO_GL=osmesa`
before running.

**Out of memory / OOM-killer** — `c3-standard-22` has 88 GB RAM,
plenty for 7 parallel runs. If you switch to `c3-highcpu-22` (~44 GB
RAM) you might want to drop to 4-parallel.

**Auto-stop didn't fire** — the VM needs `--scopes=compute-rw` (or
`cloud-platform`) at creation. Re-create the VM with the scope, or
manually `gcloud compute instances stop $VM_NAME` after the run.

**Want even cheaper?** Use `--provisioning-model=SPOT` at VM create
time for ~70% discount (preemptible). The script is idempotent so a
preemption mid-run just resumes when you create a new VM.
