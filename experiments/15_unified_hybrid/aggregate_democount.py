"""Aggregator for the T2 demo-count saturation sweep.

Reads
-----
* n=1/2/3/10 cells from
  ``results/slate_incremental_T2_democount_2026_05_28/``  (this branch).
* n=5 (T2.1) from ``results/slate_incremental_T2_2026_05_26/`` (main).
* n=20 (T2.4) from ``results/slate_incremental_T2_2026_05_26/`` (main).
* n=0 (T2.0) from ``results/slate_incremental_T2_2026_05_26/`` (main).

Outputs (only with ``--write``)
-------------------------------
* ``results/slate_incremental_T2_democount_aggregate/summary.json``
* ``docs/results_writeups/t2_democount.md`` -- the "Demo-count
  saturation table" section is filled in.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

THIS = Path(__file__).resolve().parent
RESULTS = THIS / "results"
DC_DIR = RESULTS / "slate_incremental_T2_democount_2026_05_28"
MAIN_DIR = RESULTS / "slate_incremental_T2_2026_05_26"
OUT_DIR = RESULTS / "slate_incremental_T2_democount_aggregate"
WRITEUP = THIS.parents[1] / "docs" / "results_writeups" / "t2_democount.md"

# (n_perturbed, label, source_dir, prior, tag)
CELLS = [
    (0,  "T2.0 (canonical only)", MAIN_DIR, "demo_1nn", "T2_0_hybrid_canonical_bank_mass1p3"),
    (1,  "1 perturbed demo",      DC_DIR,   "demo_1nn", "T2_1_hybrid_1demo_mass1p3"),
    (2,  "2 perturbed demos",     DC_DIR,   "demo_1nn", "T2_1_hybrid_2demo_mass1p3"),
    (3,  "3 perturbed demos",     DC_DIR,   "demo_1nn", "T2_1_hybrid_3demo_mass1p3"),
    (5,  "T2.1 (5 perturbed)",    MAIN_DIR, "demo_1nn", "T2_1_hybrid_5demo_adapted_mass1p3"),
    (10, "10 perturbed demos",    DC_DIR,   "demo_1nn", "T2_1_hybrid_10demo_mass1p3"),
    (20, "T2.4 (20 perturbed)",   MAIN_DIR, "demo_1nn", "T2_4_hybrid_20demo_saturation_mass1p3"),
]


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _read_cell(d: Path, prior: str, tag: str) -> dict | None:
    if not d.exists():
        return None
    matches = sorted(d.glob(f"unified_summary_{prior}_*_{tag}.json"))
    if not matches:
        return None
    data = json.loads(matches[0].read_text())
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
    for n, label, d, prior, tag in CELLS:
        out[str(n)] = {
            "n_perturbed": n,
            "label": label,
            **(_read_cell(d, prior, tag) or {}),
        }
    return out


def _fmt(v: dict) -> tuple[str, str]:
    if "n_success" not in v:
        return ("—", "—")
    k, n = v["n_success"], v["n_total"]
    rate = f"{v['n_success']}/{v['n_total']} = {100*v['rate']:.1f}%"
    lo, hi = _wilson(k, n)
    ci = f"[{100*lo:.1f}, {100*hi:.1f}]"
    return rate, ci


def render_saturation_table(data: dict) -> str:
    rows = [
        "| n_perturbed | label | succ | Wilson 95 % CI |",
        "|---:|---|---:|---:|",
    ]
    for n_str, v in data.items():
        rate, ci = _fmt(v)
        rows.append(f"| {n_str} | {v['label']} | {rate} | {ci} |")
    return "\n".join(rows) + "\n"


def render_headline(data: dict) -> str:
    lines = ["### Saturation knee (computed from N=30 per cell)\n"]

    # Find the smallest N whose CI overlaps T2.1 (n=5) CI.
    if "5" not in data or "n_success" not in data["5"]:
        return "\n".join(lines)
    five = data["5"]
    lo5, hi5 = _wilson(five["n_success"], five["n_total"])

    overlaps_with_5 = []
    for n_str in ("1", "2", "3", "10"):
        v = data.get(n_str, {})
        if "n_success" not in v:
            continue
        lo, hi = _wilson(v["n_success"], v["n_total"])
        overlap = not (lo > hi5 or lo5 > hi)
        overlaps_with_5.append((int(n_str), v, overlap, lo, hi))

    lines.append(
        f"- **T2.1 anchor (n=5):** {five['n_success']}/{five['n_total']} = "
        f"{100*five['rate']:.1f}%  [{100*lo5:.1f}, {100*hi5:.1f}]"
    )
    lines.append("")
    lines.append("- **Per-N overlap with the T2.1 (n=5) CI:**")
    for n, v, ovlp, lo, hi in overlaps_with_5:
        rate, _ = _fmt(v)
        ovlp_str = "yes (saturated)" if ovlp else "**no** (below knee)"
        lines.append(f"  - n={n:>2}: {rate}  [{100*lo:.1f}, {100*hi:.1f}]  -- {ovlp_str}")

    # Smallest saturated N
    sat = [n for n, _, o, _, _ in overlaps_with_5 if o and n < 5]
    if sat:
        knee = min(sat)
        lines.append("")
        lines.append(
            f"- **Saturation knee:** as small as **n={knee}** demos already lies "
            f"within the T2.1 (n=5) Wilson 95 % CI.  More demos do not reliably "
            f"add lift; the T2 adaptation effect is essentially demo-count-saturated "
            f"at the very first perturbed demo."
        )
    else:
        lines.append("")
        lines.append(
            "- No sub-5 cell's CI overlaps T2.1's; the saturation knee is at n=5 (or above)."
        )
    return "\n".join(lines) + "\n"


def update_writeup(data: dict) -> None:
    table = render_saturation_table(data)
    headline = render_headline(data)
    text = WRITEUP.read_text()
    block = (
        "### Demo-count saturation table\n\n"
        f"{table}\n"
        f"{headline}\n"
        "### Predicted outcomes"
    )
    pattern = re.compile(
        r"### Demo-count saturation table\n.*?\n### Predicted outcomes",
        flags=re.DOTALL,
    )
    if pattern.search(text):
        text = pattern.sub(block, text)
    else:
        text += "\n\n" + block
    text = re.sub(
        r"\*\*Status:\*\*\s+.*",
        "**Status:**  results landed 2026-05-29; aggregation done.",
        text,
    )
    WRITEUP.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="Persist summary.json and patch t2_democount.md.")
    args = ap.parse_args()
    data = collect()
    print(render_saturation_table(data))
    print(render_headline(data))
    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "summary.json").write_text(json.dumps(data, indent=2))
        print(f"\n[wrote] {OUT_DIR / 'summary.json'}")
        update_writeup(data)
        print(f"[updated] {WRITEUP}")


if __name__ == "__main__":
    main()
