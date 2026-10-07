#!/usr/bin/env bash
# ============================================================================
#  download_missing.sh -- Fetch any paper PDFs that we have a `references.bib`
#  entry for but no local file in `docs/papers/` yet.
#
#  Strategy:
#    1. Parse `references.bib` (next to this script).
#    2. For each @entry that has BOTH a `file = {...}` and an `eprint = {...}`
#       field, download from   https://arxiv.org/pdf/<eprint>.pdf
#       into `docs/papers/<basename of file>`.
#    3. Skip silently if the target already exists.
#    4. At the end, print:
#         (a) entries we successfully downloaded,
#         (b) entries that we already had on disk,
#         (c) entries that have a `file = {...}` target but no `eprint`
#             (you have to fetch these manually -- usually paywalled
#             journals, Nature, books, etc.),
#         (d) entries with neither a file nor an eprint (citations only,
#             e.g. textbooks).
#
#  Usage:  bash docs/papers/download_missing.sh           # from repo root
#          bash download_missing.sh                       # from docs/papers/
#
#  Requires: curl, awk, bash 4+.
# ============================================================================
set -euo pipefail

# Resolve script directory so the script works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIB="${SCRIPT_DIR}/references.bib"
OUT_DIR="${SCRIPT_DIR}"

if [[ ! -f "$BIB" ]]; then
  echo "error: cannot find $BIB" >&2
  exit 1
fi

# Parse references.bib into TSV: key<TAB>filename<TAB>eprint
parse_bib() {
  awk '
    /^@/ {
      key=$0
      sub(/^@[a-zA-Z]+\{/, "", key)
      sub(/,$/, "", key)
      fn=""; ax=""
      next
    }
    /^[[:space:]]*file[[:space:]]*=/ {
      if (match($0, /\{[^}]+\}/)) {
        s = substr($0, RSTART+1, RLENGTH-2)
        n = split(s, a, "/")
        fn = a[n]
      }
      next
    }
    /^[[:space:]]*eprint[[:space:]]*=/ {
      if (match($0, /\{[^}]+\}/)) {
        ax = substr($0, RSTART+1, RLENGTH-2)
      }
      next
    }
    /^\}[[:space:]]*$/ {
      if (key != "") printf "%s\t%s\t%s\n", key, fn, ax
      key=""; fn=""; ax=""
    }
  ' "$BIB"
}

# Try to fetch a single arXiv paper.
download_one() {
  local key="$1" filename="$2" eprint="$3"
  local target="${OUT_DIR}/${filename}"
  local url="https://arxiv.org/pdf/${eprint}.pdf"

  if [[ -s "$target" ]]; then
    echo "  [skip] $filename already exists"
    return 2  # already had
  fi

  echo "  [get ] $filename  <-  $url"
  if curl --fail --silent --show-error --location \
          --user-agent "Mozilla/5.0 (download_missing.sh)" \
          -o "$target.part" "$url"; then
    mv "$target.part" "$target"
    return 0  # downloaded
  else
    rm -f "$target.part"
    echo "  [FAIL] could not download $filename from $url" >&2
    return 1  # failed
  fi
}

declare -a downloaded=()
declare -a already=()
declare -a manual=()
declare -a citation_only=()
declare -a failed=()

while IFS=$'\t' read -r key filename eprint; do
  if [[ -z "$filename" && -z "$eprint" ]]; then
    citation_only+=("$key")
    continue
  fi
  if [[ -z "$filename" ]]; then
    # An eprint with no expected local filename -- unusual; skip.
    citation_only+=("$key (eprint=$eprint, no file)")
    continue
  fi
  if [[ -z "$eprint" ]]; then
    manual+=("$key -> $filename")
    continue
  fi

  if download_one "$key" "$filename" "$eprint"; then
    downloaded+=("$key -> $filename ($eprint)")
  else
    rc=$?
    if [[ $rc -eq 2 ]]; then
      already+=("$key -> $filename")
    else
      failed+=("$key -> $filename ($eprint)")
    fi
  fi
done < <(parse_bib)

print_section() {
  local title="$1"; shift
  local items=("$@")
  echo
  echo "=== $title (${#items[@]}) ==="
  if [[ ${#items[@]} -eq 0 ]]; then
    echo "  (none)"
  else
    printf '  %s\n' "${items[@]}"
  fi
}

print_section "downloaded this run"             "${downloaded[@]+"${downloaded[@]}"}"
print_section "already on disk"                 "${already[@]+"${already[@]}"}"
print_section "failed to download"              "${failed[@]+"${failed[@]}"}"
print_section "needs manual download (no eprint in bib)"  "${manual[@]+"${manual[@]}"}"
print_section "citation-only entries (books / textbooks / no PDF expected)"  "${citation_only[@]+"${citation_only[@]}"}"

echo
echo "Tip: for the 'manual' entries, the easiest path is Zotero's"
echo "     'Find Available PDF' (right-click the entry). Add the arXiv"
echo "     identifier to references.bib as  eprint = {NNNN.NNNNN}  to"
echo "     pick it up automatically next time."
