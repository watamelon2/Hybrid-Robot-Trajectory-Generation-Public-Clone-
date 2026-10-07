"""Post-hoc consolidated Pareto / stability analysis.

Runs on ``main`` after the 6-branch merge cascade.  Reads the
already-landed per-cell JSONs from the four T2 sweeps (multiseed,
magnitude, democount, lambda) and the original T2 ladder, and
emits:

  1. ``results/pareto_master_2026_05_29/master_cells.csv``
     -- one row per measured cell with
        (slate, cell_tag, prior, n_perturbed, mass_scale, lambda_prior,
         succ, wilson_lo, wilson_hi, wall_s_per_cell, source_json).

  2. ``results/pareto_master_2026_05_29/pareto_front.csv``
     -- the subset of cells lying on the upper-left Pareto front
        in (adaptation_wall_s, success_rate) space.

  3. ``results/pareto_master_2026_05_29/headline.md``
     -- one-paragraph summary of T2.1 stability across the
        4 perturbation knobs (seed, mass, n_perturbed, lambda).

This is a "stability envelope" analysis: the T2.1 cell sits at
(adapt_wall=1s, succ ~ 70%) and we check whether the cell stays
inside its own Wilson CI as we vary each knob.  No new compute is
done -- everything is pulled from the result JSONs landed by the
five adapt branches.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

THIS = Path(__file__).resolve().parent
RESULTS = THIS / "results"
OUT_DIR = RESULTS / "pareto_master_2026_05_29"

# Slate roots we walk
SLATE_DIRS = [
    RESULTS / "slate_incremental_T2_2026_05_26",
    RESULTS / "slate_incremental_T2_multiseed_2026_05_28",
    RESULTS / "slate_incremental_T2_magnitude_2026_05_28",
    RESULTS / "slate_incremental_T2_democount_2026_05_28",
    RESULTS / "slate_incremental_T2_lambda_2026_05_28",
    RESULTS / "slate_incremental_T2_square_hybrid_2026_05_28",
]

# Adaptation wall-clock per method, derived from the existing
# T2_pareto_data.csv on the main T2 ladder.
_ADAPT_WALL_S = {
    "hybrid_5": 1.0,
    "hybrid_20": 1.0,
    "hybrid_other": 1.0,
    "hybrid_canonical": 0.0,
    "bc_unadapted": 0.0,
    "bc_retrained": 80.0,
    "square_hybrid": 1.0,
}


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _classify_method(prior: str, tag: str, params: dict) -> str:
    if prior == "bc":
        if "retrain" in tag.lower() or params.get("bc_checkpoint"):
            return "bc_retrained"
        return "bc_unadapted"
    # demo_1nn variants
    if "square" in tag.lower():
        return "square_hybrid"
    if "5demo" in tag.lower() or "_5demo_" in tag:
        return "hybrid_5"
    if "20demo" in tag.lower():
        return "hybrid_20"
    if "canonical_bank" in tag.lower():
        return "hybrid_canonical"
    return "hybrid_other"


def _walk_slate(d: Path) -> Iterable[dict]:
    if not d.exists():
        return
    for sj in sorted(d.glob("**/unified_summary_*.json")):
        try:
            summary = json.loads(sj.read_text())
        except Exception:
            continue
        prior = summary.get("prior") or "demo_1nn"
        # tag = filename's trailing _T2_X_... section
        m = re.match(r"unified_summary_[a-z0-9_]+_K-\d+_R-\d+_(.+)\.json$", sj.name)
        tag = m.group(1) if m else sj.stem
        for cell in summary.get("per_lambda", []):
            # Look for matching unified_*.json with full params
            params = _find_params(sj.parent, prior, tag, cell.get("lambda_prior"))
            method = _classify_method(prior, tag, params)
            k = int(cell["n_success"])
            n = int(cell["n_total"])
            lo, hi = _wilson(k, n)
            yield {
                "slate":       d.name,
                "cell_tag":    tag,
                "prior":       prior,
                "method":      method,
                "lambda_prior": cell.get("lambda_prior"),
                "n_success":   k,
                "n_total":     n,
                "succ_rate":   k / n if n else 0.0,
                "wilson_lo":   lo,
                "wilson_hi":   hi,
                "wall_s_cell": float(cell.get("wall_s", 0.0)),
                "adapt_wall_s": _ADAPT_WALL_S.get(method, 1.0),
                "n_perturbed": params.get("prior_bank_size"),
                "mass_scale":  params.get("mass_scale"),
                "seed":        params.get("seed"),
                "task":        params.get("task"),
                "source_json": str(sj.relative_to(THIS.parents[1])),
            }


def _find_params(d: Path, prior: str, tag: str, lam: float | None) -> dict:
    """Pull the params block from the matching per-cell JSON."""
    if lam is None:
        return {}
    cand = sorted(d.glob(f"unified_{prior}_lam-{lam:.1f}_K-*_R-*_{tag}.json"))
    if not cand:
        # fallback: try the lambda-omitted form (some old files have it
        # baked into the tag instead of the lam- prefix).
        cand = sorted(d.glob(f"unified_{prior}_*_{tag}.json"))
    if not cand:
        return {}
    try:
        return json.loads(cand[0].read_text()).get("params", {}) or {}
    except Exception:
        return {}


def collect() -> list[dict]:
    rows: list[dict] = []
    for d in SLATE_DIRS:
        rows.extend(list(_walk_slate(d)))
    return rows


def pareto_front(rows: list[dict]) -> list[dict]:
    """Upper-left Pareto in (adapt_wall_s, succ_rate) space."""
    front: list[dict] = []
    for r in rows:
        dominated = False
        for s in rows:
            if s is r:
                continue
            # s dominates r iff s is at least as cheap and at least as
            # successful, and strictly better on one axis.
            cheaper_eq = s["adapt_wall_s"] <= r["adapt_wall_s"]
            better_eq = s["succ_rate"] >= r["succ_rate"]
            strict = (
                s["adapt_wall_s"] < r["adapt_wall_s"]
                or s["succ_rate"] > r["succ_rate"]
            )
            if cheaper_eq and better_eq and strict:
                dominated = True
                break
        if not dominated:
            front.append(r)
    front.sort(key=lambda r: (r["adapt_wall_s"], -r["succ_rate"]))
    return front


def _stability_check(rows: list[dict]) -> list[str]:
    """Return narrative bullets about T2.1 stability across knobs."""
    # T2.1 anchor = mass=1.3, n_perturbed=205, lambda=1, hybrid_5,
    # seed=0 cell on the original T2 slate.
    anchor = next(
        (r for r in rows
         if r["method"] == "hybrid_5"
         and r["lambda_prior"] == 1.0
         and (r["mass_scale"] in (1.3, None))
         and "mass1p3" in r["cell_tag"]
         and "5demo" in r["cell_tag"]
         and "multiseed" not in r["slate"]
         and "magnitude" not in r["slate"]
         and "democount" not in r["slate"]
         and "lambda" not in r["slate"]),
        None,
    )
    lines = []
    if anchor is None:
        lines.append("- (anchor T2.1 cell not located -- stability check skipped)")
        return lines
    a_lo, a_hi = anchor["wilson_lo"], anchor["wilson_hi"]
    a_rate = anchor["succ_rate"]
    lines.append(
        f"- **Anchor T2.1 (mass=1.3, 5 perturbed demos, lambda=1, seed=0):**  "
        f"{anchor['n_success']}/{anchor['n_total']} = {100*a_rate:.1f}%  "
        f"[{100*a_lo:.1f}, {100*a_hi:.1f}]."
    )

    # Count how many other "T2.1-like" cells fall within anchor's CI.
    knob_buckets = {
        "seed (multiseed)":          [r for r in rows if "multiseed" in r["slate"] and "5demo_adapted" in r["cell_tag"]],
        "mass (magnitude sweep)":    [r for r in rows if "magnitude" in r["slate"] and "5demo_adapted" in r["cell_tag"]],
        "n_perturbed (democount)":   [r for r in rows if "democount" in r["slate"] and r["method"] == "hybrid_other"],
        "lambda (lambda sweep)":     [r for r in rows if "lambda" in r["slate"]],
    }
    for name, cells in knob_buckets.items():
        if not cells:
            continue
        in_ci = [c for c in cells if a_lo <= c["succ_rate"] <= a_hi]
        outside = [c for c in cells if not (a_lo <= c["succ_rate"] <= a_hi)]
        msg = f"- **{name}:**  {len(in_ci)} of {len(cells)} cells inside T2.1's Wilson CI."
        if outside:
            outs = ", ".join(
                f"{c['cell_tag']} ({100*c['succ_rate']:.1f}%)"
                for c in outside[:4]
            )
            msg += f"  Outside-CI cells: {outs}{'...' if len(outside)>4 else ''}"
        lines.append(msg)
    return lines


def write_outputs(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 1. master CSV
    master = OUT_DIR / "master_cells.csv"
    fields = [
        "slate", "cell_tag", "prior", "method",
        "lambda_prior", "n_perturbed", "mass_scale", "seed", "task",
        "n_success", "n_total", "succ_rate", "wilson_lo", "wilson_hi",
        "adapt_wall_s", "wall_s_cell", "source_json",
    ]
    with open(master, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    # 2. pareto front
    front = pareto_front(rows)
    pareto = OUT_DIR / "pareto_front.csv"
    with open(pareto, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in front:
            w.writerow({k: r.get(k, "") for k in fields})

    # 3. headline.md
    bullets = _stability_check(rows)
    md = OUT_DIR / "headline.md"
    with open(md, "w") as f:
        f.write("# Post-hoc consolidated Pareto / stability analysis\n\n")
        f.write(f"Generated 2026-05-29 by `aggregate_pareto_master.py` over "
                f"{len(rows)} measured cells across {len(SLATE_DIRS)} slate "
                f"directories on `main` (post merge cascade).\n\n")
        f.write("## Cells on the (adapt_wall_s, succ_rate) Pareto front\n\n")
        f.write(f"{len(front)} cells survive the upper-left Pareto filter:\n\n")
        f.write("| method | tag | succ | Wilson 95 % CI | adapt_wall_s |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for r in front:
            f.write(
                f"| {r['method']} | {r['cell_tag']} | "
                f"{r['n_success']}/{r['n_total']} = {100*r['succ_rate']:.1f}% | "
                f"[{100*r['wilson_lo']:.1f}, {100*r['wilson_hi']:.1f}] | "
                f"{r['adapt_wall_s']:.1f} |\n"
            )
        f.write("\n## T2.1 stability across perturbation knobs\n\n")
        for b in bullets:
            f.write(b + "\n")
        f.write(
            "\n*Interpretation.*  A knob \"passes\" the stability check "
            "when most of its cells land inside the anchor's Wilson 95 % CI; "
            "this means the T2.1 headline is not specific to a particular "
            "choice of that knob and would survive minor design changes.\n"
        )
    return master, pareto, md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="Persist master_cells.csv, pareto_front.csv, headline.md.")
    args = ap.parse_args()

    rows = collect()
    print(f"Collected {len(rows)} cells from {len(SLATE_DIRS)} slates.")
    front = pareto_front(rows)
    print(f"Pareto front in (adapt_wall_s, succ_rate): {len(front)} cells.")
    for b in _stability_check(rows):
        print(b)

    if args.write:
        m, p, md = write_outputs(rows)
        print(f"\n[wrote] {m}")
        print(f"[wrote] {p}")
        print(f"[wrote] {md}")


if __name__ == "__main__":
    main()
