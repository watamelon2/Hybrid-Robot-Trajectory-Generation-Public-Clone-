#!/usr/bin/env bash
# Fan one logical experiment out across many VMs that run in parallel.
# Each VM runs one cell of the grid via launch.sh's `custom` mode,
# uploads its JSON to gs://$BUCKET/runs/<INSTANCE>/<TS>/, then
# self-shuts-down.
#
# Usage:
#   PROJECT=... BUCKET=gs://... ./scripts/gce/fanout.sh <pattern>
#
# Patterns:
#   lambda-sweep   6 VMs, one per lambda in {0, 0.1, 0.5, 1.0, 5.0, inf}
#                   (requires VTHETA_GCS env var; default
#                    $BUCKET/v_theta/v_theta.pt)
#   seed-sweep     9 VMs (3 methods x 3 seeds), with
#                   --deterministic-bc-reset on exp05/exp05b
#   pareto         9 VMs for the K x replans grid (exp 05)
#
# Env vars (all optional):
#   ZONE           default us-central1-a
#   MACHINE_TYPE   default e2-standard-8
#   PREEMPTIBLE    default 0 (set to 1 once you have spot quota)
#   N_DEMOS        default 20
#   N_ROLLOUTS     default 3
#   SEED           default 0 (only relevant for non-seed-sweep patterns)
#   PRIOR          default flow ('bc' is also valid, falls back to
#                  exp 03's MLP-BC checkpoint)
#   DRY_RUN        =1 prints the planned launches without firing
#
# To watch all of them at once after launch:
#   gcloud compute instances list --filter="name~'^hybrid-traj-'" \
#       --format='value(name,zone,status)'

set -euo pipefail

PATTERN="${1:-}"
if [[ -z "$PATTERN" ]]; then
    sed -n '2,30p' "$0"; exit 2
fi

PROJECT="${PROJECT:?set PROJECT to your GCP project ID}"
BUCKET="${BUCKET:?set BUCKET to gs://your-bucket-name (no trailing /)}"
ZONE="${ZONE:-us-central1-a}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-4}"
PREEMPTIBLE="${PREEMPTIBLE:-0}"
N_DEMOS="${N_DEMOS:-5}"
N_ROLLOUTS="${N_ROLLOUTS:-1}"
N_WORKERS="${N_WORKERS:-4}"
SEED="${SEED:-0}"
PRIOR="${PRIOR:-flow}"
VTHETA_GCS="${VTHETA_GCS:-${BUCKET}/v_theta/v_theta.pt}"
DRY_RUN="${DRY_RUN:-0}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LAUNCH="$REPO_ROOT/scripts/gce/launch.sh"

submit() {
    local name="$1"
    local cmd="$2"
    local extra_env="${3:-}"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry] NAME=$name CUSTOM_CMD=<...> $extra_env $LAUNCH custom"
        return 0
    fi
    env NAME="$name" \
        PROJECT="$PROJECT" BUCKET="$BUCKET" \
        ZONE="$ZONE" MACHINE_TYPE="$MACHINE_TYPE" \
        PREEMPTIBLE="$PREEMPTIBLE" \
        $extra_env \
        CUSTOM_CMD="$cmd" \
        "$LAUNCH" custom &
}

case "$PATTERN" in
    lambda-sweep)
        # Six VMs, one per lambda, all using the flow prior.
        # Each writes results/lambda_<value>.json which gets sucked
        # into the per-instance GCS upload at end.
        for lam in 0 0.1 0.5 1.0 5.0 inf; do
            slug=$(echo "$lam" | tr '._' '--')
            cmd="python experiments/07_lambda_sweep/run_can.py \
  --prior $PRIOR \
  --prior-checkpoint experiments/08_flow_matching_prior/results/v_theta.pt \
  --lambda-prior $lam \
  --n-demos $N_DEMOS --n-rollouts $N_ROLLOUTS --seed $SEED --no-live-view \
  --n-workers $N_WORKERS"
            submit "ht-lam-${slug}-$(date +%s)" "$cmd" \
                "VTHETA_GCS=$VTHETA_GCS"
        done
        ;;

    seed-sweep)
        # Three methods x three seeds = nine VMs. Deterministic BC reset
        # is on for exp05/exp05b so the BC-alone arm is bug-free.
        for s in 0 1 2; do
            submit "ht-04-s${s}-$(date +%s)" \
"python experiments/04_mbd_baseline/run_can.py \
  --n-demos $N_DEMOS --n-rollouts $N_ROLLOUTS --seed $s --no-live-view \
  --n-workers $N_WORKERS \
  --output experiments/04_mbd_baseline/results/can_ph_seed${s}.json \
  --video-dir ''"
            submit "ht-05-s${s}-$(date +%s)" \
"python experiments/05_hybrid_method/run_can.py \
  --n-demos $N_DEMOS --n-rollouts $N_ROLLOUTS --seed $s --no-live-view \
  --deterministic-bc-reset \
  --n-workers $N_WORKERS \
  --output experiments/05_hybrid_method/results/can_ph_seed${s}.json \
  --video-dir ''"
            submit "ht-05b-s${s}-$(date +%s)" \
"python experiments/05_hybrid_method/run_can.py \
  --n-demos $N_DEMOS --n-rollouts $N_ROLLOUTS --seed $s --no-live-view \
  --mpc-replans 1 --mpc-chunk-steps 60 --deterministic-bc-reset \
  --n-workers $N_WORKERS \
  --output experiments/05_hybrid_method/results/can_ph_mpc_seed${s}.json \
  --video-dir ''"
        done
        ;;

    pareto)
        # K x replans grid, fanned out as nine cells.
        for K in 5 10 20; do
            for R in 0 1 2; do
                submit "ht-K${K}-R${R}-$(date +%s)" \
"python experiments/05_hybrid_method/run_can.py \
  --n-demos $N_DEMOS --n-rollouts $N_ROLLOUTS --seed $SEED --no-live-view \
  --K $K --sigma-max 0.25 --mpc-replans $R \
  --deterministic-bc-reset \
  --n-workers $N_WORKERS \
  --output experiments/10_kxreplans_grid/results/K${K}_R${R}.json \
  --video-dir ''"
            done
        done
        ;;

    *)
        echo "unknown pattern: $PATTERN"
        exit 2
        ;;
esac

wait
echo "[fanout] all VMs submitted (pattern=$PATTERN)"
echo "[fanout] watch:  gcloud compute instances list --filter=\"name~'^ht-'\" --format='value(name,zone,status)'"
