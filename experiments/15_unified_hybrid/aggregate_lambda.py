"""Aggregator for the T2 lambda_prior sweep.

Reads
-----
* lam in {0.5, 2.0, 5.0} cells from
  ``results/slate_incremental_T2_lambda_2026_05_28/``  (this branch).
* lam = 1.0 (T2.1 on main) from
  ``results/slate_incremental_T2_2026_05_26/``.

Outputs (only with ``--write``)
-------------------------------
* ``results/slate_incremental_T2_lambda_aggregate/summary.json``
* ``docs/results_writeups/t2_lambda.md`` -- "lambda sweep table"
  section is filled in.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

THIS = Path(__file__).resolve().parent
RESULTS = THIS / "results"
LAM_DIR = RESULTS / "slate_incremental_T2_lambda_2026_05_28"
MAIN_DIR = RESULTS / "slate_incremental_T2_2026_05_26"
OUT_DIR = RESULTS / "slate_incremental_T2_lambda_aggregate"
WRITEUP = THIS.parents[1] / "docs" / "results_writeups" / "t2_lambda.md"

# (lambda, source_dir, prior, tag)
CELLS = [
    (0.5, LAM_DIR,  "demo_1nn", "T2_1_lambda_0p5"),
    (1.0, MAIN_DIR, "demo_1nn", "T2_1_hybrid_5demo_adapted_mass1p3"),
    (2.0, LAM_DIR,  "demo_1nn", "T2_1_lambda_2p0"),
    (5.0, LAM_DIR,  "demo_1nn", "T2_1_lambda_5p0"),
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
    for lam, d, prior, tag in CELLS:
        out[f"{lam}"] = {"lambda_prior": lam, **(_read_cell(d, prior, tag) or {})}
    return out


def _slot(v: dict) -> tuple[str, str]:
    if "n_success" not in v:
        return ("—", "—")
    k, n = v["n_success"], v["n_total"]
    rate = f"{k}/{n} = {100*v['rate']:.1f}%"
    lo, hi = _wilson(k, n)
    ci = f"[{100*lo:.1f}, {100*hi:.1f}]"
    return rate, ci


def render_table(data: dict) -> str:
    rows = [
        "| lambda_prior | succ | Wilson 95 % CI | wall_s/cell |",
        "|---:|---:|---:|---:|",
    ]
    for lam_str in ("0.5", "1.0", "2.0", "5.0"):
        v = data.get(lam_str, {})
        rate, ci = _slot(v)
        wall = f"{v.get('wall_s', 0):.0f}" if "wall_s" in v else "—"
        rows.append(f"| {lam_str} | {rate} | {ci} | {wall} |")
    return "\n".join(rows) + "\n"


def render_headline(data: dict) -> str:
    lines = ["### Headline (computed from per-lambda N=30)\n"]

    def get(k):
        return data.get(k, {})

    lam_pts = []
    for ls in ("0.5", "1.0", "2.0", "5.0"):
        v = get(ls)
        if "n_success" not in v:
            continue
        lo, hi = _wilson(v["n_success"], v["n_total"])
        lam_pts.append((float(ls), v["rate"], lo, hi))

    if not lam_pts:
        return "\n".join(lines)

    rates = [r for _, r, _, _ in lam_pts]
    best = max(rates)
    worst = min(rates)
    spread = (best - worst) * 100

    # Specific findings
    half = next((p for p in lam_pts if p[0] == 0.5), None)
    one = next((p for p in lam_pts if p[0] == 1.0), None)
    two = next((p for p in lam_pts if p[0] == 2.0), None)
    five = next((p for p in lam_pts if p[0] == 5.0), None)

    lines.append(
        f"- **Performance spread across lambda in {{0.5, 1, 2, 5}}:** {spread:.1f} pp"
        f"  (worst {worst*100:.1f} %, best {best*100:.1f} %)."
    )

    if half and one:
        lo05, hi05 = half[2], half[3]
        lo1, hi1 = one[2], one[3]
        overlap = not (lo05 > hi1 or lo1 > hi05)
        gap = (one[1] - half[1]) * 100
        if not overlap:
            lines.append(
                f"- **lambda = 0.5 collapses** ({half[1]*100:.1f} %) below lambda = 1 "
                f"({one[1]*100:.1f} %).  Wilson CIs do not overlap; gap = {gap:+.1f} pp.  "
                f"Below lambda ~ 1 the prior pull is too weak to dominate MBD's "
                f"score corrections."
            )
        else:
            lines.append(
                f"- lambda = 0.5 ({half[1]*100:.1f} %) is statistically tied with "
                f"lambda = 1 ({one[1]*100:.1f} %) at N=30."
            )

    if one and two and five:
        # Are 1, 2, 5 mutually CI-overlapping?
        pts = [one, two, five]
        all_overlap = all(
            not (a[2] > b[3] or b[2] > a[3])
            for a in pts for b in pts if a is not b
        )
        if all_overlap:
            lines.append(
                f"- **lambda in [1, 5] is a robust plateau:** "
                f"lam=1 -> {one[1]*100:.1f} %, lam=2 -> {two[1]*100:.1f} %, "
                f"lam=5 -> {five[1]*100:.1f} %.  Pairwise Wilson CIs all overlap.  "
                f"T2.1 headline at lambda = 1 is not specific to that choice; "
                f"any lambda in [1, 5] would have produced the same headline."
            )
        else:
            lines.append(
                f"- lambda in [1, 5] is NOT a clean plateau: at least one "
                f"pairwise comparison shows non-overlapping CIs."
            )

    return "\n".join(lines) + "\n"


def update_writeup(data: dict) -> None:
    table = render_table(data)
    headline = render_headline(data)
    text = WRITEUP.read_text()
    # Look for the "## To fill in after slate completes" or the placeholder
    # table block ending with "### Predicted outcomes".
    block = (
        "### lambda sweep table\n\n"
        f"{table}\n"
        f"{headline}\n"
        "### Predicted outcomes"
    )
    pattern = re.compile(
        r"(?:### lambda sweep table\n.*?\n)?### Predicted outcomes",
        flags=re.DOTALL,
    )
    if pattern.search(text):
        text = pattern.sub(block, text, count=1)
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
                    help="Persist summary.json and patch t2_lambda.md.")
    args = ap.parse_args()
    data = collect()
    print(render_table(data))
    print(render_headline(data))
    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "summary.json").write_text(json.dumps(data, indent=2))
        print(f"\n[wrote] {OUT_DIR / 'summary.json'}")
        update_writeup(data)
        print(f"[updated] {WRITEUP}")


if __name__ == "__main__":
    main()
