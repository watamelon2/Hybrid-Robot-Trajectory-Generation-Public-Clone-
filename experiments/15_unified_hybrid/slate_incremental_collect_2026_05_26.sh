#!/usr/bin/env bash
# Slate Incremental -- Phase-1 prereq: collect perturbed-dynamics demos.
#
# Runs experiments/15_unified_hybrid/collect_perturbed_demos.py to
# build the perturbed-dynamics bank(s) that T2 cells consume.  Two
# outputs land:
#
#   1. data/can_mass1p3_5demos.hdf5      -- 5 successful mass=1.3 demos
#   2. data/can_mass1p3_205demos.hdf5    -- canonical 200 + new 5 (T2.1 bank)
#   3. data/can_mass1p3_220demos.hdf5    -- canonical 200 + new 20 (T2.4 bank)
#
# Also harvests 20 demos opportunistically (for the T2.4 saturation
# probe).  Single replay-pass over all 200 canonical demos under
# mass=1.3 dynamics; whichever ones still succeed get kept.
#
# Spec:        docs/adaptation/04_incremental_prior_spec.md sec 3
# Branch:      feat/adapt-incremental-prior
# Worktree:    HRTG-incremental
# Wall-clock:  ~10 min (200 replays * ~3 s each on 1 worker).

set -euo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-incremental"
DATA_DIR="$REPO/experiments/15_unified_hybrid/data"
LOG_DIR="$REPO/experiments/15_unified_hybrid/results/slate_incremental_2026_05_26"
LOG="$LOG_DIR/collect.log"
mkdir -p "$DATA_DIR" "$LOG_DIR"
cd "$REPO"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

date | tee "$LOG"
echo "Slate Incremental - Phase-1 prereq: perturbed-dynamics demo collection" | tee -a "$LOG"
echo "Spec: docs/adaptation/04_incremental_prior_spec.md sec 3" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ---- Stage 1: harvest 20 demos at mass=1.3 (covers T2.1 and T2.4) ----
echo "=== $(date '+%H:%M:%S')  collecting 20 mass=1.3 demos  ===" | tee -a "$LOG"
python "$REPO/experiments/15_unified_hybrid/collect_perturbed_demos.py" \
    --task can \
    --mass-scale 1.3 \
    --friction-scale 1.0 \
    --n-demos-target 20 \
    --output "$DATA_DIR/can_mass1p3_20demos.hdf5" \
    --combine-output "$DATA_DIR/can_mass1p3_220demos.hdf5" \
    --verbose 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ---- Stage 2: derive the 5-demo + 205-demo banks from the 20-demo HDF5 ----
# Cheaper than re-running replay: take the first 5 demos out of the
# 20-demo file and produce the 5-demo / 205-demo HDFs.
echo "=== $(date '+%H:%M:%S')  deriving 5-demo / 205-demo banks  ===" | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
import h5py, json
from pathlib import Path
import robomimic
src = Path("/Users/szh/anaconda3/envs/hybrid-traj/lib/python3.11/site-packages/datasets/can/ph/low_dim_v141.hdf5")
new = Path("__DATA_DIR__/can_mass1p3_20demos.hdf5".replace(
    "__DATA_DIR__",
    "experiments/15_unified_hybrid/data",
))

# 5-demo perturbed-only HDF5 (first 5 of the 20)
out5 = new.parent / "can_mass1p3_5demos.hdf5"
if out5.exists():
    out5.unlink()
with h5py.File(str(new), "r") as src_p, h5py.File(str(out5), "w") as dst:
    g = dst.create_group("data")
    for k, v in src_p["data"].attrs.items():
        g.attrs[k] = v
    src_demos = sorted(src_p["data"].keys(), key=lambda k: int(k.split("_")[1]))[:5]
    total = 0
    for i, name in enumerate(src_demos):
        src_p.copy(src_p["data"][name], g, name=f"demo_{i}")
        if "num_samples" in g[f"demo_{i}"].attrs:
            total += int(g[f"demo_{i}"].attrs["num_samples"])
    g.attrs["total"] = total
print(f"wrote {out5}  ({len(src_demos)} demos, {total} steps)")

# 205-demo combined HDF5 (canonical 200 + first 5 perturbed)
out205 = new.parent / "can_mass1p3_205demos.hdf5"
if out205.exists():
    out205.unlink()
with h5py.File(str(src), "r") as canon, h5py.File(str(new), "r") as new_p, h5py.File(str(out205), "w") as dst:
    g = dst.create_group("data")
    for k, v in canon["data"].attrs.items():
        g.attrs[k] = v
    total = 0
    # canonical 200
    canon_demos = sorted(canon["data"].keys(), key=lambda k: int(k.split("_")[1]))
    for i, name in enumerate(canon_demos):
        canon.copy(canon["data"][name], g, name=f"demo_{i}")
        if "num_samples" in g[f"demo_{i}"].attrs:
            total += int(g[f"demo_{i}"].attrs["num_samples"])
    # first 5 perturbed
    new_demos = sorted(new_p["data"].keys(), key=lambda k: int(k.split("_")[1]))[:5]
    for i, name in enumerate(new_demos):
        dst_name = f"demo_{len(canon_demos) + i}"
        new_p.copy(new_p["data"][name], g, name=dst_name)
        if "num_samples" in g[dst_name].attrs:
            total += int(g[dst_name].attrs["num_samples"])
    g.attrs["total"] = total
    g.attrs["incremental_mass_scale"] = 1.3
    g.attrs["incremental_friction_scale"] = 1.0
    g.attrs["n_canonical_demos"] = len(canon_demos)
    g.attrs["n_perturbed_demos"] = 5
print(f"wrote {out205}  ({len(canon_demos) + 5} demos, {total} steps)")
PY

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  ALL DONE  ===" | tee -a "$LOG"
echo "Banks written:" | tee -a "$LOG"
ls -lh "$DATA_DIR/"*.hdf5 | tee -a "$LOG"
