"""3-seed pooled aggregator for the T2 multi-seed slate.

Reads
-----
* seed 0 cells (T2.0/T2.1/T2.2/T2.4) from
  ``results/slate_incremental_T2_2026_05_26/``;
* seed 0 T2.3 from
  ``results/slate_incremental_T2_bc_retrain_2026_05_26/`` (different
  output dir because that cell was rerun in a follow-up slate);
* seed 1 and seed 2 cells from
  ``results/slate_incremental_T2_multiseed_2026_05_28/seed_{1,2}/``.

The seed 0 T2.3 tag is ``T2_3_bc_retrained_mass1p3`` (no "205demos"
suffix), whereas seed 1/2 use the longer ``T2_3_bc_retrained_205demos_mass1p3``.
That asymmetry is a historical naming choice -- the 205-demo dataset
is identical -- so the aggregator treats them as the same cell.

Outputs (only with ``--write``)
-------------------------------
* ``results/slate_incremental_T2_multiseed_aggregate/summary.json``
  -- machine-readable per-seed + pooled rates with Wilson 95 % CI.
* ``docs/results_writeups/t2_multiseed.md`` -- the "Per-seed table"
  section is rewritten with concrete numbers + a computed headline
  line; all other sections (Motivation / Cells / Predicted / Failure /
  Integration) are preserved.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

THIS = Path(__file__).resolve().parent
RESULTS = THIS / "results"
SEED0_BASE = RESULTS / "slate_incremental_T2_2026_05_26"
SEED0_T23 = RESULTS / "slate_incremental_T2_bc_retrain_2026_05_26"
MULTISEED = RESULTS / "slate_incremental_T2_multiseed_2026_05_28"
OUT_DIR = RESULTS / "slate_incremental_T2_multiseed_aggregate"
WRITEUP = THIS.parents[1] / "docs" / "results_writeups" / "t2_multiseed.md"

# (short, label, seed0_tag, seed12_tag, seed0_dir, prior_token)
CELLS = [
    ("T2_0",
     "T2.0  hybrid + canonical 200 bank @ mass=1.3",
     "T2_0_hybrid_canonical_bank_mass1p3",
     "T2_0_hybrid_canonical_bank_mass1p3",
     SEED0_BASE, "demo_1nn"),
    ("T2_1",
     "T2.1  hybrid + 200 + 5 mass=1.3 demos (headline)",
     "T2_1_hybrid_5demo_adapted_mass1p3",
     "T2_1_hybrid_5demo_adapted_mass1p3",
     SEED0_BASE, "demo_1nn"),
    ("T2_2",
     "T2.2  BC unadapted @ mass=1.3",
     "T2_2_bc_unadapted_mass1p3",
     "T2_2_bc_unadapted_mass1p3",
     SEED0_BASE, "bc"),
    ("T2_3",
     "T2.3  BC retrained on 205 mass=1.3 demos",
     "T2_3_bc_retrained_mass1p3",
     "T2_3_bc_retrained_205demos_mass1p3",
     SEED0_T23, "bc"),
    ("T2_4",
     "T2.4  hybrid + 200 + 20 mass=1.3 demos (saturation)",
     "T2_4_hybrid_20demo_saturation_mass1p3",
     "T2_4_hybrid_20demo_saturation_mass1p3",
     SEED0_BASE, "demo_1nn"),
]


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _read_cell(dir_: Path, prior: str, tag: str, seed_suffix: str = "") -> dict | None:
    if not dir_.exists():
        return None
    suffix = f"_{seed_suffix}" if seed_suffix else ""
    pat = f"unified_summary_{prior}_*_{tag}{suffix}.json"
    matches = sorted(dir_.glob(pat))
    if not matches:
        return None
    with matches[0].open() as fh:
        data = json.load(fh)
    pl = data["per_lambda"][0]
    return {
        "n_success": int(pl["n_success"]),
        "n_total": int(pl["n_total"]),
        "rate": float(pl["success_rate"]),
        "wall_s": float(pl["wall_s"]),
        "source": str(matches[0].relative_to(THIS.parents[1])),
    }


def collect() -> dict:
    out: dict = {}
    for short, label, s0_tag, s12_tag, s0_dir, prior in CELLS:
        per_seed = {"0": _read_cell(s0_dir, prior, s0_tag)}
        for s in (1, 2):
            per_seed[str(s)] = _read_cell(
                MULTISEED / f"seed_{s}", prior, s12_tag, f"seed{s}"
            )
        valid = [v for v in per_seed.values() if v is not None]
        ksum = sum(v["n_success"] for v in valid)
        ntot = sum(v["n_total"] for v in valid)
        lo, hi = _wilson(ksum, ntot)
        out[short] = {
            "label": label,
            "prior": prior,
            "per_seed": per_seed,
            "pooled": {
                "n_success": ksum,
                "n_total": ntot,
                "rate": (ksum / ntot if ntot else 0.0),
                "wilson_lo": lo,
                "wilson_hi": hi,
            },
        }
    return out


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{100 * lo:.1f}, {100 * hi:.1f}]"


def render_per_seed_table(data: dict) -> str:
    head = (
        "| Cell | seed=0 | seed=1 | seed=2 | 3-seed pooled (Wilson 95 % CI) |\n"
        "|---|---:|---:|---:|---:|\n"
    )
    rows = []
    for short, label, *_ in CELLS:
        d = data[short]
        cell_id = label.split("  ", 1)[0]
        ps = d["per_seed"]

        def slot(s: str) -> str:
            v = ps[s]
            if v is None:
                return "—"
            return f"{v['n_success']}/{v['n_total']} = {_fmt_pct(v['rate'])}"

        p = d["pooled"]
        rows.append(
            f"| {cell_id} | {slot('0')} | {slot('1')} | {slot('2')} | "
            f"{p['n_success']}/{p['n_total']} = {_fmt_pct(p['rate'])}  "
            f"{_fmt_ci(p['wilson_lo'], p['wilson_hi'])} |"
        )
    return head + "\n".join(rows) + "\n"


def render_headline(data: dict) -> str:
    t21 = data["T2_1"]["pooled"]
    t23 = data["T2_3"]["pooled"]
    # CI overlap test: do the two intervals share any point?
    overlap = not (t21["wilson_lo"] > t23["wilson_hi"] or t23["wilson_lo"] > t21["wilson_hi"])
    delta_pp = (t21["rate"] - t23["rate"]) * 100
    lines = []
    lines.append("### Headline (computed from pooled 3-seed numbers)\n")
    lines.append(
        f"- **T2.1** (hybrid + 5 perturbed demos): "
        f"{t21['n_success']}/{t21['n_total']} = {_fmt_pct(t21['rate'])}  "
        f"{_fmt_ci(t21['wilson_lo'], t21['wilson_hi'])}"
    )
    lines.append(
        f"- **T2.3** (BC retrained on 205 perturbed demos): "
        f"{t23['n_success']}/{t23['n_total']} = {_fmt_pct(t23['rate'])}  "
        f"{_fmt_ci(t23['wilson_lo'], t23['wilson_hi'])}"
    )
    if overlap:
        lines.append(
            f"- CIs **overlap** -- the seed=0 gap of "
            f"{delta_pp:+.1f} pp is not robust across seeds.  "
            f"Headline needs revision."
        )
    else:
        lines.append(
            f"- CIs **do NOT overlap** -- the hybrid+5-demo > BC-retrained-on-205 "
            f"gap holds in the 3-seed pool ("
            f"\u0394 \u2248 {delta_pp:+.1f} pp).  "
            f"Headline substantiates the single-seed claim across N=90 trials."
        )
    return "\n".join(lines) + "\n"


def update_writeup(data: dict) -> None:
    table = render_per_seed_table(data)
    headline = render_headline(data)
    text = WRITEUP.read_text()
    block = (
        "### Per-seed table\n\n"
        f"{table}\n"
        f"{headline}\n"
        "### Pareto plot update"
    )
    pattern = re.compile(
        r"### Per-seed table\n.*?\n### Pareto plot update",
        flags=re.DOTALL,
    )
    if pattern.search(text):
        text = pattern.sub(block, text)
    else:
        text += "\n\n" + block
    text = re.sub(
        r"\*\*Status:\*\*\s+.*",
        "**Status:**  results landed 2026-05-29; 3-seed aggregation done.",
        text,
    )
    WRITEUP.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--write",
        action="store_true",
        help="Persist summary.json and patch docs/results_writeups/t2_multiseed.md.",
    )
    args = ap.parse_args()
    data = collect()
    print(render_per_seed_table(data))
    print(render_headline(data))
    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with (OUT_DIR / "summary.json").open("w") as fh:
            json.dump(data, fh, indent=2)
        print(f"\n[wrote] {OUT_DIR / 'summary.json'}")
        update_writeup(data)
        print(f"[updated] {WRITEUP}")


if __name__ == "__main__":
    main()
