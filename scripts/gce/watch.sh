#!/usr/bin/env bash
# scripts/gce/watch.sh -- one-glance status board for the active GCE fleet.
#
# Lists every VM whose name matches $NAME_FILTER (default: ht-) along with
#   - status (RUNNING / TERMINATED / ...)
#   - age   (HHhMMm since creation)
#   - whether results have been uploaded to GCS yet
#   - latest progress line from the serial console
#
# Usage:
#   bash scripts/gce/watch.sh                          # one shot
#   watch -n 30 'bash scripts/gce/watch.sh'            # live refresh
#
# Env overrides (defaults shown):
#   PROJECT="hybrid-traj-robin-2026"
#   BUCKET="gs://hybrid-traj-robin-2026-results"
#   ZONE="us-central1-a"
#   NAME_FILTER="ht-"     # gcloud filter pattern (name~"^${NAME_FILTER}")
#
# Designed to be safe to call from any CWD and from a shell where
# gcloud's active project is different from $PROJECT -- we always pass
# --project= explicitly.

set -euo pipefail

PROJECT="${PROJECT:-hybrid-traj-robin-2026}"
BUCKET="${BUCKET:-gs://hybrid-traj-robin-2026-results}"
ZONE="${ZONE:-us-central1-a}"
NAME_FILTER="${NAME_FILTER:-ht-}"

# Try to find gcloud / gsutil in a few common places.
if ! command -v gcloud >/dev/null 2>&1; then
    for d in "$HOME/google-cloud-sdk/bin" "$HOME/Downloads/google-cloud-sdk/bin" \
             "/usr/local/google-cloud-sdk/bin" "/opt/google-cloud-sdk/bin"; do
        if [[ -x "$d/gcloud" ]]; then
            export PATH="$d:$PATH"
            break
        fi
    done
fi
if ! command -v gcloud >/dev/null 2>&1; then
    echo "error: gcloud not on PATH" >&2; exit 1
fi

now_unix=$(date +%s)

# Fetch all VMs matching the filter in one round-trip, into a temp file.
# (Avoid bash 4 `mapfile` so this works on macOS's stock bash 3.2.)
VM_LIST=$(mktemp -t gce_watch.XXXXXX)
trap 'rm -f "$VM_LIST"' EXIT
gcloud compute instances list \
    --project="$PROJECT" \
    --filter="name~'^${NAME_FILTER}'" \
    --format='value(name,status,creationTimestamp)' 2>/dev/null > "$VM_LIST"

if [[ ! -s "$VM_LIST" ]]; then
    echo "[watch] no instances matching '^${NAME_FILTER}' in $PROJECT"
    exit 0
fi

printf '%-32s %-12s %-10s %-12s %s\n' "NAME" "STATUS" "AGE" "GCS_RESULTS" "LAST_LOG"
printf '%-32s %-12s %-10s %-12s %s\n' "----" "------" "---" "-----------" "--------"

while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    name=$(echo "$line"  | awk '{print $1}')
    status=$(echo "$line" | awk '{print $2}')
    created=$(echo "$line" | awk '{print $3}')

    # Age (mac date vs gnu date both accept ISO 8601 via -d / -j; pick one).
    if created_unix=$(date -j -f "%Y-%m-%dT%H:%M:%S" "${created%.*}" "+%s" 2>/dev/null); then
        :
    else
        created_unix=$(date -d "$created" +%s 2>/dev/null || echo "$now_unix")
    fi
    age_s=$(( now_unix - created_unix ))
    age=$(printf '%dh%02dm' $((age_s/3600)) $(((age_s%3600)/60)))

    # Has this VM uploaded a summary.json yet?
    gcs_status="-"
    rundir=$(gsutil ls "${BUCKET}/runs/${name}/" 2>/dev/null | head -1 || true)
    if [[ -n "$rundir" ]]; then
        if gsutil -q stat "${rundir}results/summary.json" 2>/dev/null; then
            gcs_status="summary"
        elif gsutil -q stat "${rundir}results.tar.gz" 2>/dev/null; then
            gcs_status="tar"
        else
            gcs_status="pending"
        fi
    fi

    # One-line progress hint from the serial console (cheap call).
    last_log="-"
    if [[ "$status" == "RUNNING" ]]; then
        # Pull the serial output, drop the syslog/kernel prefixes (anything
        # up to and including ' startup-script: '), keep our own [startup]
        # tag plus per-rollout / lambda / exception lines, and show the
        # most recent match truncated to 60 chars.
        #
        # The `|| true` is mandatory: grep returns 1 when no line matches
        # the pattern (typical for freshly-booted VMs whose serial output
        # hasn't reached any tagged line yet). Combined with `set -e +
        # pipefail` that 1 would terminate the whole watcher mid-table,
        # which was the "watch sometimes drops unfinished VMs" bug.
        last_log=$(gcloud compute instances get-serial-port-output "$name" \
                       --project="$PROJECT" --zone="$ZONE" 2>/dev/null \
                   | sed -E 's/.*startup-script: //' \
                   | grep -E '\[startup\]|\[rollout|lambda_prior|Traceback|Error' \
                   | tail -1 \
                   | cut -c1-60) || true
        last_log=${last_log:-"(no tag yet)"}
    fi

    printf '%-32s %-12s %-10s %-12s %s\n' "$name" "$status" "$age" "$gcs_status" "$last_log"
done < "$VM_LIST"

echo
echo "[watch] project=$PROJECT  bucket=$BUCKET  filter=^${NAME_FILTER}"
echo "[watch] re-run:  bash scripts/gce/watch.sh"
echo "[watch] live:    watch -n 30 'bash scripts/gce/watch.sh'"
