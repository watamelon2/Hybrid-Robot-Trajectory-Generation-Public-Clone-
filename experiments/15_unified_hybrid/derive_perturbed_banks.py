"""Derive 5-demo and (canonical-N + 5)-demo banks from a 20-demo
perturbed-dynamics collection.

Task-agnostic: works for can / lift / square (anything with a canonical
robomimic ``low_dim_v141.hdf5`` reachable via
``experiments.common.default_dataset_path``).

The Can equivalent of this logic was inline (heredoc) in
``slate_incremental_collect_2026_05_26.sh``; promoting it to a stand-
alone script avoids the heredoc-in-bash pattern when adding Lift /
Square cross-task slates.

Usage::

    python derive_perturbed_banks.py \\
        --task lift \\
        --perturbed-20 data/lift_mass1p3_20demos.hdf5 \\
        --out-5 data/lift_mass1p3_5demos.hdf5 \\
        --out-205 data/lift_mass1p3_205demos.hdf5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py


_THIS_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _THIS_DIR.parent
for _p in (_THIS_DIR, _EXPERIMENTS_DIR):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from common import default_dataset_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", type=str, required=True,
                   choices=("can", "lift", "square"))
    p.add_argument("--perturbed-20", type=Path, required=True,
                   help="20-demo perturbed HDF5 written by "
                        "collect_perturbed_demos.py.")
    p.add_argument("--out-5", type=Path, required=True,
                   help="Output 5-demo perturbed HDF5 (first 5 of the 20).")
    p.add_argument("--out-205", type=Path, required=True,
                   help="Output combined HDF5 (canonical N + first M "
                        "perturbed; uses the task's canonical dataset).")
    p.add_argument("--n-canonical", type=int, default=200,
                   help="How many canonical demos to include. Default 200.")
    p.add_argument("--n-perturbed", type=int, default=5,
                   help="How many perturbed demos to append to --out-205. "
                        "Default 5 (matches the historical T2.1 bank).  "
                        "Use values in [1, len(perturbed_20)] for the "
                        "demo-count saturation sweep (feat/adapt-t2-democount).")
    return p.parse_args()


def _write_subset(src_path: Path, dst_path: Path, n: int) -> int:
    """Copy the first ``n`` demos from ``src_path`` to ``dst_path``."""
    if dst_path.exists():
        dst_path.unlink()
    with h5py.File(str(src_path), "r") as src, \
         h5py.File(str(dst_path), "w") as dst:
        g = dst.create_group("data")
        for k, v in src["data"].attrs.items():
            g.attrs[k] = v
        names = sorted(src["data"].keys(),
                       key=lambda k: int(k.split("_")[1]))[:n]
        total = 0
        for i, name in enumerate(names):
            src.copy(src["data"][name], g, name=f"demo_{i}")
            if "num_samples" in g[f"demo_{i}"].attrs:
                total += int(g[f"demo_{i}"].attrs["num_samples"])
        g.attrs["total"] = total
        return len(names)


def _write_combined(
    canon_path: Path, perturbed_path: Path, dst_path: Path,
    n_canon: int, n_perturbed: int,
) -> tuple[int, int]:
    """Build the combined HDF5 (canonical N + first M perturbed)."""
    if dst_path.exists():
        dst_path.unlink()
    with h5py.File(str(canon_path), "r") as canon, \
         h5py.File(str(perturbed_path), "r") as new_p, \
         h5py.File(str(dst_path), "w") as dst:
        g = dst.create_group("data")
        for k, v in canon["data"].attrs.items():
            g.attrs[k] = v
        canon_names = sorted(canon["data"].keys(),
                             key=lambda k: int(k.split("_")[1]))[:n_canon]
        new_names = sorted(new_p["data"].keys(),
                           key=lambda k: int(k.split("_")[1]))[:n_perturbed]
        total = 0
        for i, name in enumerate(canon_names):
            canon.copy(canon["data"][name], g, name=f"demo_{i}")
            if "num_samples" in g[f"demo_{i}"].attrs:
                total += int(g[f"demo_{i}"].attrs["num_samples"])
        for i, name in enumerate(new_names):
            dst_name = f"demo_{len(canon_names) + i}"
            new_p.copy(new_p["data"][name], g, name=dst_name)
            if "num_samples" in g[dst_name].attrs:
                total += int(g[dst_name].attrs["num_samples"])
        g.attrs["total"] = total
        # Mirror the per-collection attrs that slate_incremental_collect
        # wrote on the Can combined HDF5 (so downstream loaders can
        # introspect provenance).
        for k in ("mass_scale", "friction_scale"):
            if k in new_p["data"].attrs:
                g.attrs[f"incremental_{k}"] = new_p["data"].attrs[k]
        g.attrs["n_canonical_demos"] = len(canon_names)
        g.attrs["n_perturbed_demos"] = len(new_names)
        return len(canon_names), len(new_names)


def main() -> int:
    args = parse_args()
    if not args.perturbed_20.exists():
        print(f"[derive] FATAL: {args.perturbed_20} not found", file=sys.stderr)
        return 1
    canon_path = default_dataset_path(args.task)
    if not canon_path.exists():
        print(f"[derive] FATAL: canonical {args.task} dataset missing: "
              f"{canon_path}", file=sys.stderr)
        return 1

    # 5-demo bank (first 5 of the perturbed-20).
    n5 = _write_subset(args.perturbed_20, args.out_5, n=5)
    print(f"[derive] wrote {args.out_5}  ({n5} demos)")

    # Combined bank: canonical N + first --n-perturbed perturbed.
    nc, np_ = _write_combined(
        canon_path, args.perturbed_20, args.out_205,
        n_canon=args.n_canonical, n_perturbed=args.n_perturbed,
    )
    print(f"[derive] wrote {args.out_205}  ({nc} canonical + {np_} perturbed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
