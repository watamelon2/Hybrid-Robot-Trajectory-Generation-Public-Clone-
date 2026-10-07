#!/usr/bin/env bash
# watchdog.sh -- conservative hourly babysitter for the overnight launcher.
#
# Designed to run alongside an active launch_overnight.sh process. Every
# CHECK_INTERVAL seconds (default 3600 = 1 h) it:
#
#   1. Snapshots fleet, quota, queue, active state to a CSV-ish log line.
#   2. Inspects each RUNNING VM's serial console for known failure
#      signatures (Traceback / ArgumentError / FileNotFoundError / etc).
#      If a VM matches AND has been running for more than $MIN_AGE_SEC,
#      `gcloud delete`s it. This frees its IP + CPU quota slot for the
#      launcher to fill on its next tick.
#   3. Detects "stuck on apt" failures (no `[startup] running custom`
#      line after 10 min uptime) -- often means apt mirror is unreachable.
#      Also auto-deletes those.
#   4. Logs everything; never restarts the launcher (too risky), never
#      modifies queue.txt (caller has to verify the unknown unknowns).
#
# Designed to be safe: only kills VMs whose serial console gives strong
# evidence of an unrecoverable failure. Won't touch healthy VMs even if
# they look idle (e.g. mid-rollout python that hasn't printed in a while).
#
# Env overrides:
#   PROJECT          (default hybrid-traj-robin-2026)
#   ZONE             (default us-central1-a)
#   NAME_FILTER      (default 'ht-r')          regex prefix of managed VMs
#   CHECK_INTERVAL   (default 3600)            seconds between checks
#   MIN_AGE_SEC      (default 180)             ignore VMs younger than this
#   LOG_FILE         (default /tmp/watchdog-<ts>.log)

set -uo pipefail
exec </dev/null   # macOS nohup workaround (same fix as launch_overnight.sh)

PROJECT="${PROJECT:-hybrid-traj-robin-2026}"
ZONE="${ZONE:-us-central1-a}"
NAME_FILTER="${NAME_FILTER:-ht-r}"
CHECK_INTERVAL="${CHECK_INTERVAL:-3600}"
MIN_AGE_SEC="${MIN_AGE_SEC:-180}"
LOG_FILE="${LOG_FILE:-/tmp/watchdog-$(date -u +%s).log}"

# Discover gcloud the same way watch.sh does.
if ! command -v gcloud >/dev/null 2>&1; then
    for d in "$HOME/google-cloud-sdk/bin" "$HOME/Downloads/google-cloud-sdk/bin" \
             "/usr/local/google-cloud-sdk/bin" "/opt/google-cloud-sdk/bin"; do
        if [[ -x "$d/gcloud" ]]; then
            export PATH="$d:$PATH"
            break
        fi
    done
fi
command -v gcloud >/dev/null 2>&1 || { echo "watchdog: gcloud not on PATH" >&2; exit 1; }

# Patterns that indicate the experiment crashed and the VM is just
# waiting for its auto-shutdown timer. Conservative: only match strong
# Python-exception signatures, not generic "error" / "warning" strings.
CRASH_PATTERNS='Traceback \(most recent call last\)|FileNotFoundError|argparse\.ArgumentError|ImportError|ModuleNotFoundError|RuntimeError: CUDA|Killed$'

log() {
    local msg
    msg="[watchdog $(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"
    echo "$msg" | tee -a "$LOG_FILE" >&2
}

snapshot() {
    # Headline counts (one line per category).
    local n_running n_terminated n_total
    n_running=$(gcloud compute instances list --project="$PROJECT" \
                    --filter="status=RUNNING AND name~'^${NAME_FILTER}'" \
                    --format='value(name)' 2>/dev/null | wc -l | tr -d ' ')
    n_terminated=$(gcloud compute instances list --project="$PROJECT" \
                    --filter="status=TERMINATED AND name~'^${NAME_FILTER}'" \
                    --format='value(name)' 2>/dev/null | wc -l | tr -d ' ')
    n_total=$((n_running + n_terminated))

    local cpu_used cpu_limit ip_used ip_limit
    read -r cpu_used cpu_limit < <(gcloud compute project-info describe \
            --project="$PROJECT" --flatten='quotas[]' \
            --format='value(quotas.metric,quotas.usage,quotas.limit)' 2>/dev/null \
        | awk '$1 == "CPUS_ALL_REGIONS" { printf "%d %d", $2, $3; exit }')
    read -r ip_used ip_limit < <(gcloud compute regions describe us-central1 \
            --project="$PROJECT" --flatten='quotas[]' \
            --format='value(quotas.metric,quotas.usage,quotas.limit)' 2>/dev/null \
        | awk '$1 == "IN_USE_ADDRESSES" { printf "%d %d", $2, $3; exit }')

    log "snapshot: running=$n_running terminated=$n_terminated cpu=$cpu_used/$cpu_limit ip=$ip_used/$ip_limit"
}

inspect_and_cull() {
    local now killed=0 vm_list age_file
    now=$(date +%s)
    vm_list=$(mktemp)
    age_file=$(mktemp)
    trap "rm -f '$vm_list' '$age_file'" RETURN

    # Get all RUNNING managed VMs + their creation timestamps.
    gcloud compute instances list --project="$PROJECT" \
        --filter="status=RUNNING AND name~'^${NAME_FILTER}'" \
        --format='value(name,creationTimestamp)' 2>/dev/null > "$vm_list" || return 0

    while IFS=$'\t' read -r name created; do
        [[ -z "$name" ]] && continue
        local age created_ts
        created_ts=$(date -j -u -f '%Y-%m-%dT%H:%M:%S.%3N%z' "${created%-*}+0000" +%s 2>/dev/null \
                 || date -d "$created" +%s 2>/dev/null) || created_ts=$now
        age=$((now - created_ts))
        (( age < MIN_AGE_SEC )) && continue

        # Pull the last 200 lines of the serial console (cheaper than full dump).
        local serial last_match age_min
        age_min=$((age / 60))
        serial=$(gcloud compute instances get-serial-port-output "$name" \
                    --project="$PROJECT" --zone="$ZONE" 2>/dev/null | tail -n 300)
        last_match=$(printf '%s\n' "$serial" \
                    | grep -oE "$CRASH_PATTERNS" | tail -1)

        if [[ -n "$last_match" ]]; then
            log "  CULL  $name age=${age_min}m -- matched crash pattern: '$last_match'"
            gcloud compute instances delete "$name" \
                --project="$PROJECT" --zone="$ZONE" --quiet >/dev/null 2>&1 \
                && killed=$((killed + 1)) \
                || log "    WARN: delete returned nonzero for $name"
            continue
        fi

        # Stuck-on-apt heuristic: > 10 min old AND no `[startup] running custom` line.
        if (( age > 600 )); then
            if ! printf '%s\n' "$serial" | grep -q '\[startup\] running custom'; then
                log "  CULL  $name age=${age_min}m -- no '[startup] running custom' after >10 min (apt stuck?)"
                gcloud compute instances delete "$name" \
                    --project="$PROJECT" --zone="$ZONE" --quiet >/dev/null 2>&1 \
                    && killed=$((killed + 1)) \
                    || log "    WARN: delete returned nonzero for $name"
            fi
        fi
    done < "$vm_list"

    (( killed > 0 )) && log "  cull pass killed $killed VM(s)"
    return 0
}

log "starting; pid=$$ interval=${CHECK_INTERVAL}s project=$PROJECT log=$LOG_FILE"
log "name filter: ^${NAME_FILTER}*"
log "crash patterns: $CRASH_PATTERNS"
trap 'log "SIGTERM/SIGINT received -- exiting cleanly"; exit 0' INT TERM

while true; do
    snapshot
    inspect_and_cull
    sleep "$CHECK_INTERVAL"
done
