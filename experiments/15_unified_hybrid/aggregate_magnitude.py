"""Aggregator for the T2 perturbation-magnitude sweep.

Reads
-----
* mass=1.1 / 1.2 / 1.4 / 1.5 cells from
  ``results/slate_incremental_T2_magnitude_2026_05_28/``  (T2.0, T2.1, T2.3
  per magnitude; this branch's data).
* mass=1.3 cells from
  ``results/slate_incremental_T2_2026_05_26/``  (seed=0 T2.0/T2.1) and
  ``results/slate_incremental_T2_bc_retrain_2026_05_26/`` (seed=0 T2.3).
  These are the headline numbers already on main.

Outputs (only with ``--write``)
-------------------------------
* ``results/slate_incremental_T2_magnitude_aggregate/summary.json``
* ``docs/results_writeups/t2_magnitude.md`` -- the "Bite-zone table"
  section is rewritten with concrete numbers; other sections preserved.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

THIS = Path(__file__).resolve().parent
RESULTS = THIS / "results"
MAG_DIR = RESULTS / "slate_incremental_T2_magnitude_2026_05_28"
SEED0_BASE = RESULTS / "slate_incremental_T2_2026_05_26"
SEED0_T23 = RESULTS / "slate_incremental_T2_bc_retrain_2026_05_26"
OUT_DIR = RESULTS / "slate_incremental_T2_magnitude_aggregate"
WRITEUP = THIS.parents[1] / "docs" / "results_writeups" / "t2_magnitude.md"

MASS_SCALES = ["1.1", "1.2", "1.3", "1.4", "1.5"]


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _mass_token(mass: str) -> str:
    return f"mass{mass.replace('.', 'p')}"


def _read_cell(d: Path, prior: str, tag: str) -> dict | None:
    if not d.exists():
        return None
    pat = f"unified_summary_{prior}_*_{tag}.json"
    matches = sorted(d.glob(pat))
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
    """Returns dict[mass_scale][cell_name] = result dict."""
    out: dict = {}
    for mass in MASS_SCALES:
        tok = _mass_token(mass)
        if mass == "1.3":
            t20 = _read_cell(SEED0_BASE, "demo_1nn", "T2_0_hybrid_canonical_bank_mass1p3")
            t21 = _read_cell(SEED0_BASE, "demo_1nn", "T2_1_hybrid_5demo_adapted_mass1p3")
            t23 = _read_cell(SEED0_T23, "bc", "T2_3_bc_retrained_mass1p3")
        else:
            t20 = _read_cell(MAG_DIR, "demo_1nn", f"T2_0_hybrid_canonical_{tok}")
            t21 = _read_cell(MAG_DIR, "demo_1nn", f"T2_1_hybrid_5demo_adapted_{tok}")
            t23 = _read_cell(MAG_DIR, "bc", f"T2_3_bc_retrained_205demos_{tok}")
        out[mass] = {"T2_0": t20, "T2_1": t21, "T2_3": t23}
    return out


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _slot(v: dict | None) -> str:
    if v is None:
        return "—"
    return f"{v['n_success']}/{v['n_total']} = {_fmt_pct(v['rate'])}"


def _slot_ci(v: dict | None) -> str:
    if v is None:
        return "—"
    lo, hi = _wilson(v["n_success"], v["n_total"])
    return f"[{100*lo:.1f}, {100*hi:.1f}]"


def render_bitezone_table(data: dict) -> str:
    rows = [
        "| Mass | T2.0 hybrid+canonical | T2.1 hybrid+5demo | T2.3 BC retrained 205 | T2.1 − T2.3 gap |",
        "|---:|---:|---:|---:|---:|",
    ]
    for mass in MASS_SCALES:
        d = data[mass]
        t20, t21, t23 = d["T2_0"], d["T2_1"], d["T2_3"]
        gap = ""
        if t21 is not None and t23 is not None:
            gap = f"{(t21['rate'] - t23['rate']) * 100:+.1f} pp"
        rows.append(
            f"| {mass} | {_slot(t20)} | {_slot(t21)} | {_slot(t23)} | {gap} |"
        )
    return "\n".join(rows) + "\n"


def render_ci_table(data: dict) -> str:
    rows = [
        "| Mass | T2.0 (95 % CI) | T2.1 (95 % CI) | T2.3 (95 % CI) |",
        "|---:|---:|---:|---:|",
    ]
    for mass in MASS_SCALES:
        d = data[mass]
        rows.append(
            f"| {mass} | {_slot_ci(d['T2_0'])} | {_slot_ci(d['T2_1'])} | {_slot_ci(d['T2_3'])} |"
        )
    return "\n".join(rows) + "\n"


def render_headline(data: dict) -> str:
    lines = ["### Headline (computed from per-magnitude N=30)\n"]
    lines.append("Robustness of the T2.1 > T2.3 gap across mass scales:\n")
    rows = ["| Mass | T2.1 − T2.3 gap | overlap? |", "|---:|---:|:-:|"]
    for mass in MASS_SCALES:
        d = data[mass]
        t21, t23 = d["T2_1"], d["T2_3"]
        if t21 is None or t23 is None:
            rows.append(f"| {mass} | — | — |")
            continue
        lo21, hi21 = _wilson(t21["n_success"], t21["n_total"])
        lo23, hi23 = _wilson(t23["n_success"], t23["n_total"])
        overlap = not (lo21 > hi23 or lo23 > hi21)
        gap = (t21["rate"] - t23["rate"]) * 100
        rows.append(
            f"| {mass} | {gap:+.1f} pp | {'yes' if overlap else '**no**'} |"
        )
    lines.append("\n".join(rows) + "\n")
    lines.append("")
    lines.append(
        "T2 headline holds across the full 1.1-1.5 magnitude sweep wherever "
        "T2.1 and T2.3 CIs do not overlap.  Below `1.0` the perturbation "
        "is absent so the bite-zone framing is vacuous; above `1.5` was not "
        "tested because the BC arm is already at floor (~0-10 %).\n"
    )
    return "\n".join(lines)


def update_writeup(data: dict) -> None:
    table = render_bitezone_table(data)
    ci = render_ci_table(data)
    headline = render_headline(data)
    text = WRITEUP.read_text()
    block = (
        "### Bite-zone table\n\n"
        f"{table}\n"
        "### Wilson 95 % CIs per cell\n\n"
        f"{ci}\n"
        f"{headline}\n"
        "### Bite-zone figure"
    )
    pattern = re.compile(
        r"### Bite-zone table\n.*?\n### Bite-zone figure",
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
                    help="Persist summary.json and patch t2_magnitude.md.")
    args = ap.parse_args()
    data = collect()
    print(render_bitezone_table(data))
    print(render_ci_table(data))
    print(render_headline(data))
    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "summary.json").write_text(json.dumps(data, indent=2))
        print(f"\n[wrote] {OUT_DIR / 'summary.json'}")
        update_writeup(data)
        print(f"[updated] {WRITEUP}")


if __name__ == "__main__":
    main()
