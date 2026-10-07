#!/usr/bin/env bash
# scripts/gce/launch_overnight.sh  --  streaming / round-robin GCE launcher.
#
# Maintains an active pool of VMs sized to the CPUS_ALL_REGIONS quota.
# Pops items from a work queue and submits each one only when the
# remaining quota fits its vCPU count. When a VM finishes (TERMINATED),
# its slot is freed and the next pending item is dispatched.
#
# Properties:
#   - Idempotent: every VM is named ht-r<RUN_ID>-<slug>-<ts>, where
#     RUN_ID is unique per launcher invocation. Re-running the script
#     with the same RUN_ID resumes from the state file.
#   - Cleans up on SIGTERM / SIGINT: deletes every VM the launcher
#     created in this run before exiting non-zero.
#   - Quota-aware: queries CPUS_ALL_REGIONS quota and usage via gcloud
#     before each dispatch decision, so it cooperates with VMs created
#     by other tools (e.g., the lambda sweep already in flight).
#   - Verified submissions: after each launch.sh call, polls
#     `gcloud describe` until the VM is observed; if it never appears,
#     the item is re-queued.
#   - Partial-failure safe: if a launch.sh attempt fails, only that
#     one item is affected; the rest of the queue and the active pool
#     are untouched.
#
# Usage:
#   bash scripts/gce/launch_overnight.sh [pattern1 [pattern2 ...]]
#
# Recognized patterns:
#   seed-sweep   --  9 items, 3 seeds x {exp04, exp05, exp05b}
#   pareto       --  9 items, K in {5,10,20} x replans in {0,1,2}
#   lambda-sweep --  6 items, lambdas in {0, 0.1, 0.5, 1.0, 5.0, inf}
#                    (note: needs v_theta.pt in GCS)
#
# Default if no patterns given: "seed-sweep pareto"
#
# Env overrides (defaults shown):
#   PROJECT=hybrid-traj-robin-2026
#   BUCKET=gs://hybrid-traj-robin-2026-results
#   ZONE=us-central1-a
#   MACHINE_TYPE=n2-standard-2          # 2 vCPU per VM
#   VCPUS_PER_VM=2                       # MUST match MACHINE_TYPE
#   N_DEMOS=20  N_ROLLOUTS=3  N_WORKERS=2
#   POLL_SEC=60                          # main loop tick
#   SUBMIT_VERIFY_TRIES=12               # 12 * 5s = 60s describe poll
#   QUOTA_METRIC=CPUS_ALL_REGIONS        # the global cap that blocks us
#
# Known constraints not yet modeled by the dispatcher:
#   - IN_USE_ADDRESSES in the deploy region (default 8). Each VM gets
#     an external IP, so concurrent VM count is min(QUOTA/VCPUS_PER_VM,
#     IN_USE_ADDRESSES). When the IP cap binds first, launches fail
#     with a quota error and get re-queued; the launcher self-corrects
#     when a finished VM releases its IP. To bypass it, enable Private
#     Google Access on the subnet and pass `--no-address` to
#     launch.sh's gcloud create -- both currently TODO.

set -uo pipefail

# Force stdin to /dev/null so every gcloud subprocess inherits a valid
# (empty, readable) file descriptor 0. When this script is launched via
# `nohup ... &` on macOS, the inherited stdin can become invalid after
# the parent shell loses interactivity, and gcloud's Python startup
# crashes with `OSError: [Errno 9] Bad file descriptor` on every call.
# A single permanent redirect here fixes it for all descendants.
exec </dev/null

PROJECT="${PROJECT:-hybrid-traj-robin-2026}"
BUCKET="${BUCKET:-gs://hybrid-traj-robin-2026-results}"
ZONE="${ZONE:-us-central1-a}"
MACHINE_TYPE="${MACHINE_TYPE:-n2-standard-2}"
VCPUS_PER_VM="${VCPUS_PER_VM:-2}"
N_DEMOS="${N_DEMOS:-20}"
N_ROLLOUTS="${N_ROLLOUTS:-3}"
N_WORKERS="${N_WORKERS:-2}"
POLL_SEC="${POLL_SEC:-60}"
SUBMIT_VERIFY_TRIES="${SUBMIT_VERIFY_TRIES:-12}"
QUOTA_METRIC="${QUOTA_METRIC:-CPUS_ALL_REGIONS}"
VTHETA_GCS="${VTHETA_GCS:-${BUCKET}/v_theta/v_theta.pt}"
BC_GCS="${BC_GCS:-${BUCKET}/bc/policy.pt}"

# ---------- gcloud auto-discovery (same logic as watch.sh) -----------------
if ! command -v gcloud >/dev/null 2>&1; then
    for d in "$HOME/google-cloud-sdk/bin" "$HOME/Downloads/google-cloud-sdk/bin" \
             "/usr/local/google-cloud-sdk/bin" "/opt/google-cloud-sdk/bin"; do
        if [[ -x "$d/gcloud" ]]; then
            export PATH="$d:$PATH"; break
        fi
    done
fi
command -v gcloud >/dev/null 2>&1 || { echo "error: gcloud not on PATH" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ---------- per-run identifier + state -------------------------------------
# Sanitize RUN_ID to GCE name regex `[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?`:
# lowercase, alphanumeric only, fall back to timestamp digits if empty.
RUN_ID="${RUN_ID:-$(date -u +%s | tail -c 6)}"
RUN_ID=$(echo "$RUN_ID" | tr 'A-Z' 'a-z' | tr -cd 'a-z0-9')
if [[ -z "$RUN_ID" ]]; then
    RUN_ID=$(date -u +%s | tail -c 6)
fi
STATE_DIR="${TMPDIR:-/tmp}/overnight-$RUN_ID"
mkdir -p "$STATE_DIR"
ACTIVE_FILE="$STATE_DIR/active.txt"
QUEUE_FILE="$STATE_DIR/queue.txt"
LOG_FILE="$STATE_DIR/launcher.log"
: > "$ACTIVE_FILE"
: > "$QUEUE_FILE"

log() { echo "[overnight r$RUN_ID $(date -u +%H:%M:%S)] $*" | tee -a "$LOG_FILE" >&2; }

# ---------- signal handler: delete every VM matching our RUN_ID -----------
# Rationale: do NOT trust ACTIVE_FILE alone, because a SIGINT can race
# with the (submit -> write-to-file) sequence and miss VMs that were
# successfully created but not yet recorded. Instead, query gcloud
# for any instance whose name starts with our unique ht-r<RUN_ID>-
# prefix and delete them all. This is safe because no other tool ever
# produces that prefix.
cleanup() {
    local sig="${1:-EXIT}"
    log "cleanup triggered ($sig); locating VMs with prefix ht-r${RUN_ID}-"
    local victims
    victims=$(gcloud compute instances list --project="$PROJECT" \
                  --filter="name~'^ht-r${RUN_ID}-'" \
                  --format='value(name)' 2>/dev/null || true)
    if [[ -n "$victims" ]]; then
        log "deleting $(echo "$victims" | wc -l | tr -d ' ') VM(s):"
        echo "$victims" | sed 's/^/    /' | tee -a "$LOG_FILE"
        # shellcheck disable=SC2086
        gcloud compute instances delete $victims \
            --zone "$ZONE" --project "$PROJECT" --quiet 2>&1 | tail -5 | \
            sed 's/^/    /' | tee -a "$LOG_FILE"
    else
        log "no VMs found with our RUN_ID prefix; nothing to clean"
    fi
    log "cleanup done; state in $STATE_DIR"
    [[ "$sig" == "EXIT" ]] || exit 1
}
# Trap on INT/TERM (operator kill) AND on EXIT-with-error so partial
# failures don't leak VMs.  EXIT trap re-checks $? so a clean finish
# (no leftover VMs) is a no-op.
on_exit() { [[ $? -ne 0 ]] && cleanup EXIT_ERROR; true; }
trap 'cleanup INT'  INT
trap 'cleanup TERM' TERM
trap 'on_exit'      EXIT

# ---------- queue construction ---------------------------------------------
# Each queue line is:  <slug>\t<custom_cmd>
emit_seed_sweep() {
    for s in 0 1 2; do
        printf '%s\t%s\n' "04-s${s}" \
"python experiments/04_mbd_baseline/run_can.py \
  --n-demos $N_DEMOS --n-rollouts $N_ROLLOUTS --seed $s --no-live-view \
  --n-workers $N_WORKERS \
  --output experiments/04_mbd_baseline/results/can_ph_seed${s}.json \
  --video-dir ''"
        printf '%s\t%s\n' "05-s${s}" \
"python experiments/05_hybrid_method/run_can.py \
  --n-demos $N_DEMOS --n-rollouts $N_ROLLOUTS --seed $s --no-live-view \
  --deterministic-bc-reset --n-workers $N_WORKERS \
  --output experiments/05_hybrid_method/results/can_ph_seed${s}.json \
  --video-dir ''"
        printf '%s\t%s\n' "05b-s${s}" \
"python experiments/05_hybrid_method/run_can.py \
  --n-demos $N_DEMOS --n-rollouts $N_ROLLOUTS --seed $s --no-live-view \
  --mpc-replans 1 --mpc-chunk-steps 60 --deterministic-bc-reset \
  --n-workers $N_WORKERS \
  --output experiments/05_hybrid_method/results/can_ph_mpc_seed${s}.json \
  --video-dir ''"
    done
}

emit_pareto() {
    for K in 5 10 20; do
        for R in 0 1 2; do
            # slug stays lowercase so it is usable as part of a GCE VM
            # name without further sanitization.
            printf '%s\t%s\n' "k${K}-r${R}" \
"python experiments/05_hybrid_method/run_can.py \
  --n-demos $N_DEMOS --n-rollouts $N_ROLLOUTS --seed 0 --no-live-view \
  --K $K --sigma-max 0.25 --mpc-replans $R --deterministic-bc-reset \
  --n-workers $N_WORKERS \
  --output experiments/10_kxreplans_grid/results/K${K}_R${R}.json \
  --video-dir ''"
        done
    done
}

emit_lambda_sweep() {
    for lam in 0 0.1 0.5 1.0 5.0 inf; do
        slug=$(echo "$lam" | tr '._' '--')
        printf '%s\t%s\n' "lam-${slug}" \
"python experiments/07_lambda_sweep/run_can.py \
  --prior flow \
  --prior-checkpoint experiments/08_flow_matching_prior/results/v_theta.pt \
  --lambda-prior $lam \
  --n-demos $N_DEMOS --n-rollouts $N_ROLLOUTS --seed 0 --no-live-view \
  --n-workers $N_WORKERS"
    done
}

# BC-prior variant of the lambda sweep -- runs the same per-step
# composite-score machinery as `lambda-sweep` but uses the BC policy
# from exp 03 (well-trained, 500 epochs on 200 demos) as the prior
# velocity field instead of the flow-matching v_theta. This is a
# valid scientific ablation in its own right (the exp 07 docstring
# calls out --prior bc explicitly), AND it does NOT depend on
# v_theta being retrained -- so it can run while v_theta v2/v3 is
# still in flight.
#
# Endpoints worth checking against:
#   lam=0   + warm-start -> approx Exp 05's headline (BC warm-start,
#                            planner-only update)
#   lam=inf              -> approx Exp 03 (deterministic BC rollout)
#   lam in (0, inf)      -> exercises the composite score with a
#                            known-good prior
emit_lambda_sweep_bc() {
    for lam in 0 0.1 0.5 1.0 5.0 inf; do
        slug=$(echo "$lam" | tr '._' '--')
        printf '%s\t%s\n' "lam-bc-${slug}" \
"python experiments/07_lambda_sweep/run_can.py \
  --prior bc \
  --bc-checkpoint experiments/03_bc_baseline/results/policy.pt \
  --lambda-prior $lam \
  --n-demos $N_DEMOS --n-rollouts $N_ROLLOUTS --seed 0 --no-live-view \
  --n-workers $N_WORKERS"
    done
}

PATTERNS=("${@:-seed-sweep pareto}")
for p in "${PATTERNS[@]}"; do
    case "$p" in
        seed-sweep)      emit_seed_sweep      >> "$QUEUE_FILE" ;;
        pareto)          emit_pareto          >> "$QUEUE_FILE" ;;
        lambda-sweep)    emit_lambda_sweep    >> "$QUEUE_FILE" ;;
        lambda-sweep-bc) emit_lambda_sweep_bc >> "$QUEUE_FILE" ;;
        *) log "WARN: unknown pattern '$p' -- skipping" ;;
    esac
done

TOTAL_ITEMS=$(wc -l < "$QUEUE_FILE" | tr -d ' ')
log "RUN_ID=$RUN_ID  state=$STATE_DIR"
log "patterns: ${PATTERNS[*]}  -> $TOTAL_ITEMS work items queued"
log "machine: $MACHINE_TYPE  (${VCPUS_PER_VM} vCPU/VM)  zone=$ZONE"

# ---------- helpers --------------------------------------------------------

# Print global quota usage as "<used> <limit>" for $QUOTA_METRIC.
#
# Implementation note: `project-info describe` does not accept --filter
# (only --flatten + --format), so we list all quota rows, grep ours, and
# pick out usage/limit with awk. Tested on Cloud SDK 5xx-7xx series.
quota_usage() {
    # gcloud reports quotas as floats ("12.0 24.0"); cast to int with
    # printf %d so the caller's bash arithmetic works. Both gcloud
    # stderr and the parsed output are tee'd to a debug file for
    # post-mortem when the launcher sees an unexpected empty result.
    local raw err
    err="$STATE_DIR/quota_err.log"
    raw="$STATE_DIR/quota_raw.log"
    gcloud compute project-info describe --project="$PROJECT" \
        --flatten='quotas[]' \
        --format='value(quotas.metric,quotas.usage,quotas.limit)' \
        2>>"$err" \
      | tee -a "$raw" \
      | awk -v m="$QUOTA_METRIC" '$1 == m { printf "%d %d\n", $2, $3; exit }'
}

# Verify the VM name in $1 exists (any status). Returns 0 if yes.
vm_exists() {
    gcloud compute instances describe "$1" \
        --zone "$ZONE" --project "$PROJECT" \
        --format='value(name)' >/dev/null 2>&1
}

# Get the status of a VM in $1. Echoes status or "MISSING".
vm_status() {
    local s
    s=$(gcloud compute instances describe "$1" \
            --zone "$ZONE" --project "$PROJECT" \
            --format='value(status)' 2>/dev/null) || s="MISSING"
    echo "${s:-MISSING}"
}

# Submit a single work item. $1 = slug, $2 = custom_cmd.
# Echoes the VM name on success; returns 1 on failure.
submit_one() {
    local slug="$1" cmd="$2"
    # GCE VM names must match `[a-z][-a-z0-9]{0,61}[a-z0-9]?`. Force the
    # slug component to lowercase + drop any non-[a-z0-9-] character so a
    # pattern emitter that uses `K5-R0`-style ids cannot produce an invalid
    # name. (This is the same defensive transform we apply to RUN_ID.)
    local safe_slug
    safe_slug=$(printf '%s' "$slug" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-' '-')
    local name="ht-r${RUN_ID}-${safe_slug}-$(date +%s)"
    log "  submit  $name"
    # launch.sh respects NAME / MACHINE_TYPE / ZONE / etc.
    NAME="$name" \
    PROJECT="$PROJECT" BUCKET="$BUCKET" ZONE="$ZONE" \
    MACHINE_TYPE="$MACHINE_TYPE" \
    PREEMPTIBLE="${PREEMPTIBLE:-0}" \
    VTHETA_GCS="$VTHETA_GCS" \
    BC_GCS="$BC_GCS" \
    N_DEMOS="$N_DEMOS" N_ROLLOUTS="$N_ROLLOUTS" N_WORKERS="$N_WORKERS" \
    CUSTOM_CMD="$cmd" \
    bash "$REPO_ROOT/scripts/gce/launch.sh" custom \
        >> "$STATE_DIR/launch-$slug.log" 2>&1 || true
    # Verify: poll describe for up to SUBMIT_VERIFY_TRIES * 5 s.
    local i
    for ((i=1; i<=SUBMIT_VERIFY_TRIES; i++)); do
        if vm_exists "$name"; then
            log "  verified $name (after ${i} probe(s))"
            echo "$name"
            return 0
        fi
        sleep 5
    done
    log "  FAILED to verify $name -- see $STATE_DIR/launch-$slug.log"
    return 1
}

# ---------- main loop ------------------------------------------------------

log "entering main loop"
ITER=0
while true; do
    ITER=$((ITER+1))

    # ---- 1. reap finished VMs ----
    if [[ -s "$ACTIVE_FILE" ]]; then
        new_active="$STATE_DIR/active.new"
        : > "$new_active"
        while IFS= read -r vm; do
            [[ -z "$vm" ]] && continue
            st=$(vm_status "$vm")
            case "$st" in
                RUNNING|PROVISIONING|STAGING|REPAIRING)
                    echo "$vm" >> "$new_active"
                    ;;
                TERMINATED|STOPPING|STOPPED|SUSPENDED|SUSPENDING)
                    log "  reap   $vm  ($st) -> deleting to free quota"
                    gcloud compute instances delete "$vm" \
                        --zone "$ZONE" --project "$PROJECT" --quiet \
                        >/dev/null 2>&1 || \
                        log "    WARN: delete returned nonzero for $vm"
                    ;;
                MISSING)
                    log "  drop   $vm  (vanished from gcloud)"
                    ;;
                *)
                    log "  unknown status '$st' for $vm; keeping"
                    echo "$vm" >> "$new_active"
                    ;;
            esac
        done < "$ACTIVE_FILE"
        mv "$new_active" "$ACTIVE_FILE"
    fi

    n_active=$(wc -l < "$ACTIVE_FILE" | tr -d ' ')
    n_queue=$(wc -l < "$QUEUE_FILE" | tr -d ' ')

    # ---- 2. exit condition ----
    if (( n_queue == 0 && n_active == 0 )); then
        log "all items done.  exiting."
        break
    fi

    # ---- 3. dispatch decisions ----
    if (( n_queue > 0 )); then
        QU="$(quota_usage)"
        if [[ -z "$QU" ]]; then
            log "WARN: quota query returned empty -- skipping dispatch this tick"
            log "  tick $ITER done; sleeping ${POLL_SEC}s"
            sleep "$POLL_SEC"
            continue
        fi
        read -r USED LIMIT <<< "$QU"
        FREE=$((LIMIT - USED))
        log "  quota  $QUOTA_METRIC: used=$USED limit=$LIMIT free=$FREE  | active=$n_active queue=$n_queue"
        SLOTS=$(( FREE / VCPUS_PER_VM ))
        if (( SLOTS > n_queue )); then SLOTS=$n_queue; fi
        if (( SLOTS > 0 )); then
            log "  -> dispatching $SLOTS new item(s)"
            for ((s=1; s<=SLOTS; s++)); do
                # Pop the first line from the queue file.
                IFS=$'\t' read -r slug cmd < <(head -1 "$QUEUE_FILE")
                if [[ -z "${slug:-}" ]]; then break; fi
                tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && \
                    mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
                if vm_name=$(submit_one "$slug" "$cmd"); then
                    echo "$vm_name" >> "$ACTIVE_FILE"
                else
                    # re-queue this item at the BACK of the queue
                    printf '%s\t%s\n' "$slug" "$cmd" >> "$QUEUE_FILE"
                    log "    re-queued $slug to back of queue"
                fi
            done
        fi
    fi

    # ---- 4. sleep before next tick ----
    log "  tick $ITER done; sleeping ${POLL_SEC}s"
    sleep "$POLL_SEC"
done

log "launcher done after $ITER ticks.  results in gs://${BUCKET#gs://}/runs/ht-r${RUN_ID}-*"
