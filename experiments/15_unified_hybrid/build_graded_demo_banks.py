"""Build graded-quality demo banks for the #6 gate-ceiling experiment.

docs/theory/09_mbd_data_guided_comparison.md §6b-#6 asks for demo pools
at graded quality (optimal / +X% cost) to test the prediction that the
hard-gate (§6.1, ``--temper-p inf``) **plateaus at the demo's own cost**
while the best-lambda **product exceeds it** -- the cleanest single plot
for "persistent fusion beats warm-start".

Cost metric
-----------
We grade by **demo trajectory length** (``num_samples``), a rollout-free
surrogate for control cost: a longer expert trajectory reached the goal
less efficiently, so it carries higher accumulated cost.  This matches
the gate's own ``ℓ_dem`` term, which our ``denoise_loop`` derives by
rolling out the demo target (a longer/less-efficient demo -> higher
``demo_cost`` -> lower ℓ_dem -> lower gate ceiling).  (Summed reward is
useless here -- it is sparse/flat at 5-6 across the Can demos.)

The 200 canonical Can demos span length 82..151 (median 115), a ~27%
best-vs-worst tercile spread -- enough for an optimal-vs-degraded
comparison without synthesising fake demos.

Output
------
One HDF5 per band (robomimic layout, demos renumbered ``demo_0..``,
``data`` attrs copied so the zero-train bank loader consumes them
verbatim -- same writer contract as ``derive_perturbed_banks.py``), plus
a ``<prefix>_manifest.json`` with each band's count + length stats +
``cost_pct_vs_best`` (median-length ratio, the band's nominal "+X%").

Usage::

    python build_graded_demo_banks.py --task can --out-dir data \\
        --n-bands 3
    # -> data/can_quality_optimal.hdf5, _mid.hdf5, _worst.hdf5, _manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _THIS_DIR.parent
for _p in (_THIS_DIR, _EXPERIMENTS_DIR):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from common import default_dataset_path  # noqa: E402

# Band labels for the common --n-bands choices.
_BAND_NAMES = {
    2: ["optimal", "worst"],
    3: ["optimal", "mid", "worst"],
    4: ["optimal", "q2", "q3", "worst"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", type=str, default="can",
                   choices=("can", "lift", "square"))
    p.add_argument("--source", type=Path, default=None,
                   help="Source HDF5 (default: the task's canonical "
                        "robomimic low_dim dataset).")
    p.add_argument("--out-dir", type=Path, default=_THIS_DIR / "data")
    p.add_argument("--prefix", type=str, default=None,
                   help="Output filename prefix (default '<task>_quality').")
    p.add_argument("--n-bands", type=int, default=3, choices=(2, 3, 4),
                   help="Number of equal-count length bands. Default 3.")
    p.add_argument("--per-band-n", type=int, default=0,
                   help="Cap demos per band (0 = use the whole band). "
                        "When >0, takes the band's most-representative "
                        "demos (closest to the band median length).")
    return p.parse_args()


def _demo_lengths(src: h5py.File) -> tuple[list[str], np.ndarray]:
    names = sorted(src["data"].keys(), key=lambda k: int(k.split("_")[1]))
    lens = []
    for n in names:
        g = src["data"][n]
        if "num_samples" in g.attrs:
            lens.append(int(g.attrs["num_samples"]))
        else:
            lens.append(int(g["actions"].shape[0]))
    return names, np.asarray(lens, dtype=np.int64)


def _write_band(
    src: h5py.File, dst_path: Path, demo_names: list[str],
) -> int:
    if dst_path.exists():
        dst_path.unlink()
    with h5py.File(str(dst_path), "w") as dst:
        g = dst.create_group("data")
        for k, v in src["data"].attrs.items():
            g.attrs[k] = v
        total = 0
        for i, name in enumerate(demo_names):
            src.copy(src["data"][name], g, name=f"demo_{i}")
            gi = g[f"demo_{i}"]
            if "num_samples" in gi.attrs:
                total += int(gi.attrs["num_samples"])
        g.attrs["total"] = total
        return len(demo_names)


def main() -> int:
    args = parse_args()
    source = args.source or default_dataset_path(args.task)
    if not Path(source).exists():
        print(f"[graded] FATAL: source dataset missing: {source}",
              file=sys.stderr)
        return 1
    prefix = args.prefix or f"{args.task}_quality"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    band_names = _BAND_NAMES[args.n_bands]

    with h5py.File(str(source), "r") as src:
        names, lens = _demo_lengths(src)
        order = np.argsort(lens, kind="stable")  # ascending length = best
        bands = np.array_split(order, args.n_bands)

        best_med = float(np.median(lens[bands[0]]))
        manifest = {
            "task": args.task,
            "source": str(source),
            "cost_metric": "trajectory_length_num_samples",
            "n_bands": args.n_bands,
            "bands": [],
        }
        for label, idx in zip(band_names, bands):
            idx = np.asarray(idx)
            if args.per_band_n and args.per_band_n < len(idx):
                med = np.median(lens[idx])
                pick = idx[np.argsort(np.abs(lens[idx] - med))][:args.per_band_n]
                idx = np.sort(pick)
            band_demo_names = [names[i] for i in idx]
            dst = args.out_dir / f"{prefix}_{label}.hdf5"
            n = _write_band(src, dst, band_demo_names)
            blens = lens[idx]
            med = float(np.median(blens))
            entry = {
                "label": label,
                "path": str(dst),
                "n_demos": n,
                "len_min": int(blens.min()),
                "len_median": med,
                "len_mean": round(float(blens.mean()), 1),
                "len_max": int(blens.max()),
                "cost_pct_vs_best": round((med / best_med - 1.0) * 100, 1),
            }
            manifest["bands"].append(entry)
            print(f"[graded] wrote {dst.name}  n={n}  "
                  f"len med={med:.0f}  (+{entry['cost_pct_vs_best']}% vs best)")

    man_path = args.out_dir / f"{prefix}_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"[graded] manifest -> {man_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
