#!/bin/bash
# Phase-D aggregator -- merges Q1 epoch curve + Q2/Q3 local + (optional) GCP +
# Phase E + F into a single (ckpt, lam, K) Wilson-CI table.
#
# Reads:
#   experiments/17_dp_hybrid/results/phase_d_2026_05_30/
#     epoch_curve_local/{ep0049,ep0099,...,ep0349}/   <- Q1 (K=10 lam=0.5)
#     q2q3_local/
#       q3_K5/, q3_K20/                               <- Q3 (ep0049)
#       q2_lam0p1/, q2_lam0p25/, q2_lam1p0/, q2_lam2p0/  <- Q2 (ep0049)
#       E_K40_ep0049/, E_K20_ep0149/, E_K20_ep0199/   <- Phase E (K-sweep across ckpts)
#       F_lam0p1_ep0149/, F_lam2p0_ep0149/, F_lam0p1_ep0199/, F_lam2p0_ep0199/  <- Phase F
#     q2q3_gcp/  (optional, if SCPed from the VM)
#       q2_lam*/, q3_K5/, q3_K20/                     <- GCP V2 (ep0049, seed=100000)
#
# Writes:
#   q2q3_AGGREGATE.json
#   q2q3_AGGREGATE.md   (multi-section grid keyed by ckpt x (lam, K))
#
# Cell-dir naming convention used to recover the ckpt:
#   - Q2/Q3 cells without ep token  -> ckpt = ep0049 (Q2/Q3 sweep at the weak prior)
#   - Q1 cells (epoch_curve_local/ep0XXX/)  -> ckpt = ep0XXX
#   - Phase E/F cells (tag_..._ep0XXX/)     -> ckpt = ep0XXX (parsed from trailing token)
#
# Usage:
#   bash scripts/local/aggregate_q2q3.sh
#
# Idempotent; safe to re-run as cells land.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PHASE_D_ROOT="$REPO_ROOT/experiments/17_dp_hybrid/results/phase_d_2026_05_30"
PYTHON="${PYTHON:-/Users/szh/anaconda3/envs/hybrid-traj/bin/python}"

"$PYTHON" - <<PY
import json, math, os, re, glob
from collections import defaultdict

PHASE_D_ROOT = "$PHASE_D_ROOT"
Q1_DIR       = os.path.join(PHASE_D_ROOT, "epoch_curve_local")
LOCAL_DIR    = os.path.join(PHASE_D_ROOT, "q2q3_local")
GCP_DIR      = os.path.join(PHASE_D_ROOT, "q2q3_gcp")
OUT_JSON     = os.path.join(PHASE_D_ROOT, "q2q3_AGGREGATE.json")
OUT_MD       = os.path.join(PHASE_D_ROOT, "q2q3_AGGREGATE.md")

EP_DEFAULT = "ep0049"

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z*z / n
    center = (p + z*z / (2*n)) / denom
    half = (z / denom) * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (max(0.0, center - half), p, min(1.0, center + half))

def load_summary(path):
    """Return list of (lam, K, n_test_success, n_test_total).

    The unified_summary_*.json schema (Phase-D) puts K (and R) at the top level
    of the dict and packs one or more (lam, n_success, n_total) tuples inside
    ``per_lambda``.  We inherit top-level K when per_lambda cells don't carry one.
    """
    if not os.path.isfile(path):
        return []
    d = json.load(open(path))
    cells = d.get("per_lambda", [d])
    top_K = d.get("K")
    out = []
    for c in cells:
        lam = c.get("lambda_prior")
        K = c.get("K", top_K)
        n_succ = c.get("n_test_success", c.get("n_success", 0))
        n_tot  = c.get("n_test_total",   c.get("n_total",   0))
        out.append((lam, K, n_succ, n_tot))
    return out

EP_RX = re.compile(r"ep(\d{4})")

def parse_ckpt_from_cell(cell_name, default=EP_DEFAULT):
    """Cell dir name -> ckpt token like 'ep0149'.  Defaults to ep0049 if no ep token."""
    m = EP_RX.search(cell_name)
    if m:
        return "ep" + m.group(1)
    return default

# Bucket key:  (ckpt, lam, K).
buckets = defaultdict(lambda: [0, 0, []])   # k, n, sources

# Q1: epoch_curve_local/epXXXX/
if os.path.isdir(Q1_DIR):
    for cell in sorted(os.listdir(Q1_DIR)):
        cell_path = os.path.join(Q1_DIR, cell)
        if not os.path.isdir(cell_path):
            continue
        ckpt = parse_ckpt_from_cell(cell)
        for js in glob.glob(os.path.join(cell_path, "unified_summary_*.json")):
            for lam, K, k, n in load_summary(js):
                if lam is None or K is None:
                    continue
                key = (ckpt, float(lam), int(K))
                buckets[key][0] += int(k)
                buckets[key][1] += int(n)
                buckets[key][2].append("Q1:" + cell)

# Q2/Q3 + Phase E/F: q2q3_local/<cell>/
for src_dir, src_tag in [(LOCAL_DIR, "local"), (GCP_DIR, "gcp")]:
    if not os.path.isdir(src_dir):
        continue
    for cell in sorted(os.listdir(src_dir)):
        cell_path = os.path.join(src_dir, cell)
        if not os.path.isdir(cell_path):
            continue
        ckpt = parse_ckpt_from_cell(cell, default=EP_DEFAULT)
        for js in glob.glob(os.path.join(cell_path, "unified_summary_*.json")):
            for lam, K, k, n in load_summary(js):
                if lam is None or K is None:
                    continue
                key = (ckpt, float(lam), int(K))
                buckets[key][0] += int(k)
                buckets[key][1] += int(n)
                buckets[key][2].append("{}:{}".format(src_tag, cell))

# Build aggregate JSON
out_cells = []
for (ckpt, lam, K), (k, n, sources) in sorted(buckets.items()):
    lo, p, hi = wilson(k, n)
    out_cells.append({
        "ckpt": ckpt,
        "lambda_prior": lam,
        "K": K,
        "n_test_success": k,
        "n_test_total": n,
        "p_hat": p,
        "wilson95_lo": lo,
        "wilson95_hi": hi,
        "sources": sources,
    })

out = {
    "metadata": {
        "phase": "Phase-D Q1 + Q2/Q3 + E/F aggregate",
        "source_dirs": [Q1_DIR, LOCAL_DIR, GCP_DIR],
        "note": "k/n test rates pooled across disjoint seed bands (Q1=100k, local-Q2Q3=200k, local-E/F=300k, GCP=100k); same seed band on same cell only.",
    },
    "cells": out_cells,
}
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)

# --- Markdown: 4 sections ---
def fmt_row(c):
    return "| {ckpt} | {lam:.2f} | {K:>3} | {k}/{n} | {p:.3f} | [{lo:.3f}, {hi:.3f}] | {src} |".format(
        ckpt=c["ckpt"], lam=c["lambda_prior"], K=c["K"],
        k=c["n_test_success"], n=c["n_test_total"],
        p=c["p_hat"], lo=c["wilson95_lo"], hi=c["wilson95_hi"],
        src=", ".join(c["sources"][:3]) + ("" if len(c["sources"]) <= 3 else " +" + str(len(c["sources"])-3)),
    )

# 1. Q1 epoch curve (K=10 lam=0.5 across ep0049..ep0349)
q1_cells = [c for c in out_cells if c["K"] == 10 and abs(c["lambda_prior"] - 0.5) < 1e-6 and c["ckpt"].startswith("ep")]

# 2. Q2 lambda-sweep at K=10 ep0049 + Phase F at ep0149/ep0199
lam_sweep_cells = [c for c in out_cells if c["K"] == 10]

# 3. Q3 K-sweep at lam=0.5 ep0049 + Phase E at ep0149/ep0199
K_sweep_cells = [c for c in out_cells if abs(c["lambda_prior"] - 0.5) < 1e-6]

md = [
    "# Phase-D full aggregate (Q1 + Q2/Q3 + Phase E/F)",
    "",
    "Auto-generated by ``scripts/local/aggregate_q2q3.sh``.",
    "",
    "Sources walked:",
    "  - ``epoch_curve_local/ep0XXX/``     (Q1 K=10 lam=0.5 epoch curve, n=50/ckpt)",
    "  - ``q2q3_local/<cell>/``            (Q2/Q3 local @ ep0049 + Phase E/F across ckpts)",
    "  - ``q2q3_gcp/<cell>/``              (optional; SCPed from GCP if pooled)",
    "",
    "Seed-band layout (disjoint, safe to pool within same cell):",
    "  Q1=100000+, local-Q2/Q3=200000+, local-E/F=300000+, GCP=100000+.",
    "",
    "## 1. Q1 epoch curve (K=10, lam=0.5)",
    "",
    "| ckpt | lam | K | k / n | p_hat | 95 % Wilson CI | sources |",
    "|-----|----:|--:|------:|------:|---------------:|---------|",
]
for c in sorted(q1_cells, key=lambda c: c["ckpt"]):
    md.append(fmt_row(c))

md += [
    "",
    "## 2. lambda-sweep at K=10 (Q2 + Phase F)",
    "",
    "| ckpt | lam | K | k / n | p_hat | 95 % Wilson CI | sources |",
    "|-----|----:|--:|------:|------:|---------------:|---------|",
]
for c in sorted(lam_sweep_cells, key=lambda c: (c["ckpt"], c["lambda_prior"])):
    md.append(fmt_row(c))

md += [
    "",
    "## 3. K-sweep at lam=0.5 (Q1 + Q3 + Phase E)",
    "",
    "| ckpt | lam | K | k / n | p_hat | 95 % Wilson CI | sources |",
    "|-----|----:|--:|------:|------:|---------------:|---------|",
]
for c in sorted(K_sweep_cells, key=lambda c: (c["ckpt"], c["K"])):
    md.append(fmt_row(c))

# 4. Full 2-D grid (ckpt -> K=5/10/20/40 at lam=0.5; ckpt -> lam=0.1/0.25/0.5/1.0/2.0 at K=10)
ckpts_seen = sorted({c["ckpt"] for c in out_cells})

md += [
    "",
    "## 4. 2-D grid view",
    "",
    "Rows = ckpt.  Each cell shows p_hat (k/n).  Empty = not run.",
    "",
    "### K-sweep at lam=0.5",
    "",
    "| ckpt \\\\ K | 5 | 10 | 20 | 40 |",
    "|-----------|--:|---:|---:|---:|",
]
for ckpt in ckpts_seen:
    row = ["| " + ckpt]
    for K_val in [5, 10, 20, 40]:
        match = next((c for c in K_sweep_cells if c["ckpt"] == ckpt and c["K"] == K_val), None)
        if match:
            row.append("{:.2f} ({}/{})".format(match["p_hat"], match["n_test_success"], match["n_test_total"]))
        else:
            row.append("--")
    md.append(" | ".join(row) + " |")

md += [
    "",
    "### lambda-sweep at K=10",
    "",
    "| ckpt \\\\ lam | 0.1 | 0.25 | 0.5 | 1.0 | 2.0 |",
    "|-------------|----:|-----:|----:|----:|----:|",
]
for ckpt in ckpts_seen:
    row = ["| " + ckpt]
    for lam_val in [0.1, 0.25, 0.5, 1.0, 2.0]:
        match = next((c for c in lam_sweep_cells if c["ckpt"] == ckpt and abs(c["lambda_prior"] - lam_val) < 1e-6), None)
        if match:
            row.append("{:.2f} ({}/{})".format(match["p_hat"], match["n_test_success"], match["n_test_total"]))
        else:
            row.append("--")
    md.append(" | ".join(row) + " |")

md += [
    "",
    "## Reference baselines",
    "",
    "| comparator | k / n | rate |",
    "|---|---:|---:|",
    "| Phase-B standalone DP at ep0199 (GCP T4, DP paper apples-to-apples) | 40/50 | 80.0 % |",
    "| Phase-B re-eval at ep0199 (this Mac) | 7/10 | 70.0 % |",
    "| Phase-D Q1 K=10 lam=0.5 at ep0049 (weak-prior baseline, this Mac) | 27/50 | **54.0 %** |",
    "",
    "## Reading rule (pre-registered)",
    "",
    "Applied per cell against the SAME-ckpt Q1 K=10 lam=0.5 baseline (or against",
    "ep0049 = 54 % when no Q1 baseline exists for that ckpt):",
    "",
    "- **GREEN** :  p_hat - wilson_lo  >  Q1_baseline_for_ckpt + 6 pp",
    "- **YELLOW**:  CI overlaps the Q1 baseline (default reading)",
    "- **RED**   :  p_hat + wilson_hi  <  Q1_baseline_for_ckpt - 6 pp",
    "",
    "Sweet spot for the paper headline: GREEN cell at the weakest prior (ep0049)",
    "with the smallest K and most moderate lam, since that minimises MBD compute",
    "while delivering the largest lift over the under-trained DP baseline.",
    "",
]

with open(OUT_MD, "w") as f:
    f.write("\n".join(md))

print(f"[aggregate] wrote {OUT_JSON}")
print(f"[aggregate] wrote {OUT_MD}")
print(f"[aggregate] {len(out_cells)} cells; ckpts seen: {ckpts_seen}")
PY
