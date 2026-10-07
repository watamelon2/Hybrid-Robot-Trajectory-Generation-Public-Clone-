#!/usr/bin/env bash
# Create a tarball of the current repo checkout and upload it to GCS.
# Run this once per branch update before launching VMs.
#
# Usage:
#   PROJECT=... BUCKET=gs://... ./scripts/gce/upload_repo.sh
#
# The tarball lands at $BUCKET/repo/repo.tar.gz and is fetched by
# the VM startup script instead of a git clone, so the repo can
# stay private.

set -euo pipefail

PROJECT="${PROJECT:?set PROJECT to your GCP project ID}"
BUCKET="${BUCKET:?set BUCKET to gs://your-bucket-name (no trailing /)}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BRANCH="$(git -C "$REPO_ROOT" branch --show-current | tr '/' '-')"
COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"

TARBALL="/tmp/hybrid-traj-${BRANCH}-${COMMIT}.tar.gz"

echo "[upload] branch=$BRANCH commit=$COMMIT"
echo "[upload] creating tarball ..."

git -C "$REPO_ROOT" archive --format=tar.gz \
    --prefix=repo/ \
    -o "$TARBALL" \
    HEAD

echo "[upload] uploading to $BUCKET/repo/repo.tar.gz ..."
gsutil cp "$TARBALL" "$BUCKET/repo/repo.tar.gz"

# Also write a version file so VMs can log what they're running.
echo "$BRANCH $COMMIT $(date -u -Iseconds)" > /tmp/repo-version.txt
gsutil cp /tmp/repo-version.txt "$BUCKET/repo/version.txt"

rm -f "$TARBALL" /tmp/repo-version.txt
echo "[upload] done — $BUCKET/repo/repo.tar.gz ($COMMIT)"
