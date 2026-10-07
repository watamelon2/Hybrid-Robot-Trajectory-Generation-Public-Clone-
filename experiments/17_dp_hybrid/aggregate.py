"""Aggregate Phase 1 DP-hybrid cells into a single table.

Reads:
  - experiments/17_dp_hybrid/results/baseline_standalone_dp/standalone_dp_repro.json
  - experiments/17_dp_hybrid/results/cell_dp_pullmbd/unified_summary_*.json
  - (optional ablations) experiments/17_dp_hybrid/results/abl_*/unified_summary_*.json

Writes:
  - experiments/17_dp_hybrid/results/summary.csv
  - experiments/17_dp_hybrid/results/summary.md

Computes Wilson 95% CIs for each cell and a paired-cell pp delta vs the
DP-standalone baseline, with a Wilson-interval-style CI on the delta.

Run from repo root:
    python experiments/17_dp_hybrid/aggregate.py
"""
from __future__ import annotations

import csv
import glob
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "17_dp_hybrid" / "results"


# ---------------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson 95% CI for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def delta_ci(k1: int, n1: int, k2: int, n2: int, z: float = 1.96) -> tuple[float, float, float]:
    """Approx CI for p1 - p2 (independent-cell Wald with Wilson-shrunk se).

    Returns (delta, lo, hi) in fractional units.
    """
    if n1 == 0 or n2 == 0:
        return (0.0, -1.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    delta = p1 - p2
    # Use Wilson-derived variance proxy: max(p,(1-p))/n is a conservative
    # upper bound; here we use the unbiased SE which is fine for n>=20.
    var1 = p1 * (1 - p1) / n1
    var2 = p2 * (1 - p2) / n2
    se = math.sqrt(var1 + var2)
    return (delta, delta - z * se, delta + z * se)


# ---------------------------------------------------------------------------
# cell loader
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    name: str
    description: str
    n_total: int = 0
    n_success: int = 0
    success_rate: float = 0.0
    wall_s: float = 0.0
    status: str = "missing"
    source: str = ""
    extras: dict = field(default_factory=dict)


def _load_standalone(path: Path) -> Cell:
    if not path.exists():
        return Cell(name="baseline_standalone_dp",
                    description="DP standalone (K=100, n_act=8, H=16)",
                    status="missing")
    d = json.loads(path.read_text())
    return Cell(
        name="baseline_standalone_dp",
        description="DP standalone (K=100, n_act=8, H=16)",
        n_total=int(d["n"]),
        n_success=int(d["n_succ"]),
        success_rate=float(d["succ_rate"]),
        wall_s=float(d["wall_s"]),
        status="done",
        source=str(path.relative_to(REPO)),
        extras={"prior_ckpt": d.get("prior_checkpoint", "")},
    )


def _load_unified(path: Path, name: str, description: str) -> Cell:
    if not path.exists():
        return Cell(name=name, description=description, status="missing")
    d = json.loads(path.read_text())
    return Cell(
        name=name,
        description=description,
        n_total=int(d.get("n_total", 0)),
        n_success=int(d.get("n_success", 0)),
        success_rate=float(d.get("success_rate", 0.0)),
        wall_s=float(d.get("wall_s", 0.0)),
        status="done",
        source=str(path.relative_to(REPO)),
    )


def _find_unified(cell_dir: Path) -> Path | None:
    cands = sorted(cell_dir.glob("unified_summary_*.json"))
    return cands[0] if cands else None


def gather_cells() -> list[Cell]:
    cells: list[Cell] = []

    cells.append(_load_standalone(
        RESULTS / "baseline_standalone_dp" / "standalone_dp_repro.json"
    ))

    hyb_summary = _find_unified(RESULTS / "cell_dp_pullmbd")
    cells.append(
        _load_unified(
            hyb_summary if hyb_summary else RESULTS / "cell_dp_pullmbd" / "MISSING.json",
            name="cell_dp_pullmbd",
            description="DP + pull-mode MBD (K=10, M=16, λ=1, H=16, R=29)",
        )
    )

    # Any ablations matching abl_* / unified_summary_*.json.
    for abl_dir in sorted(RESULTS.glob("abl_*")):
        if not abl_dir.is_dir():
            continue
        summ = _find_unified(abl_dir)
        if summ is None:
            cells.append(Cell(name=abl_dir.name, description="(ablation, no summary yet)", status="running"))
            continue
        cells.append(_load_unified(summ, name=abl_dir.name, description="(ablation)"))

    return cells


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def write_csv(cells: list[Cell], baseline: Cell, path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "cell", "description", "status",
            "n_total", "n_success", "success_rate",
            "wilson_lo", "wilson_hi",
            "delta_vs_standalone_pp",
            "delta_lo_pp", "delta_hi_pp",
            "wall_s", "wall_min",
        ])
        for c in cells:
            if c.status != "done":
                w.writerow([c.name, c.description, c.status,
                            "", "", "", "", "", "", "", "", "", ""])
                continue
            lo, hi = wilson_ci(c.n_success, c.n_total)
            if baseline.status == "done" and c.name != baseline.name:
                d, dlo, dhi = delta_ci(c.n_success, c.n_total,
                                       baseline.n_success, baseline.n_total)
            else:
                d = dlo = dhi = 0.0
            w.writerow([
                c.name, c.description, c.status,
                c.n_total, c.n_success, f"{c.success_rate:.4f}",
                f"{lo:.4f}", f"{hi:.4f}",
                f"{100*d:+.2f}", f"{100*dlo:+.2f}", f"{100*dhi:+.2f}",
                f"{c.wall_s:.1f}", f"{c.wall_s/60:.2f}",
            ])


def render_md(cells: list[Cell], baseline: Cell) -> str:
    lines: list[str] = []
    lines.append("# Experiment 17 — DP-hybrid Phase 1 summary\n")
    lines.append("Generated by `experiments/17_dp_hybrid/aggregate.py`. "
                 "Each row is a single-seed eval cell of 30 trials "
                 "(10 demos × 3 rollouts). Wilson 95% CI on success "
                 "rate; delta column compares to DP-standalone.\n")

    lines.append("\n## Headline table\n")
    lines.append("| Cell | Description | n_done | succ | rate (95% CI) | Δ vs standalone (pp, 95% CI) | Wall |")
    lines.append("|---|---|---:|---:|---|---|---:|")
    for c in cells:
        if c.status != "done":
            lines.append(f"| `{c.name}` | {c.description} | – | – | _{c.status}_ | – | – |")
            continue
        lo, hi = wilson_ci(c.n_success, c.n_total)
        rate_str = f"{100*c.success_rate:.1f} % [{100*lo:.1f}, {100*hi:.1f}]"
        if baseline.status == "done" and c.name != baseline.name:
            d, dlo, dhi = delta_ci(c.n_success, c.n_total,
                                   baseline.n_success, baseline.n_total)
            delta_str = f"{100*d:+.1f} [{100*dlo:+.1f}, {100*dhi:+.1f}]"
        else:
            delta_str = "—"
        wall_str = f"{c.wall_s/60:.1f} min"
        lines.append(
            f"| `{c.name}` | {c.description} | {c.n_total} | {c.n_success} | "
            f"{rate_str} | {delta_str} | {wall_str} |"
        )

    lines.append("\n## Interpretation\n")
    if baseline.status != "done":
        lines.append("- Baseline (DP standalone) not yet measured — skipping interpretation.")
    else:
        hyb = next((c for c in cells if c.name == "cell_dp_pullmbd"), None)
        if hyb is None or hyb.status != "done":
            lines.append("- Hybrid cell not yet measured — skipping interpretation.")
        else:
            d, dlo, dhi = delta_ci(hyb.n_success, hyb.n_total,
                                   baseline.n_success, baseline.n_total)
            if dlo > 0:
                verdict = "Hybrid significantly above standalone (CI excludes 0)."
            elif dhi < 0:
                verdict = "Hybrid significantly below standalone (regression confirmed)."
            else:
                verdict = "Hybrid not separated from standalone at 95% (CI straddles 0)."
            lines.append(f"- **Δ(hybrid − standalone) = {100*d:+.1f} pp**, "
                         f"95% CI [{100*dlo:+.1f}, {100*dhi:+.1f}]. {verdict}")
            if d >= 0.13:
                lines.append("- **Phase 1 §2.3 falsifiable claim (Δ ≥ 13 pp) satisfied** "
                             "→ proceed to ablations and writeup.")
            elif d > 0:
                lines.append("- Marginal positive Δ < 13 pp — λ_prior / R sweep needed.")
            else:
                lines.append("- Negative or null Δ — debug pull-mode score on DP "
                             "(see §5 risks).")

    lines.append("\n## Files\n")
    for c in cells:
        if c.source:
            lines.append(f"- `{c.name}`: `{c.source}`")
    lines.append("\n*Updated automatically.*\n")
    return "\n".join(lines)


def main() -> None:
    cells = gather_cells()
    baseline = next((c for c in cells if c.name == "baseline_standalone_dp"),
                    Cell(name="baseline_standalone_dp", description="(missing)", status="missing"))

    csv_path = RESULTS / "summary.csv"
    md_path = RESULTS / "summary.md"
    write_csv(cells, baseline, csv_path)
    md_path.write_text(render_md(cells, baseline))
    print(f"[aggregate] wrote {csv_path.relative_to(REPO)}")
    print(f"[aggregate] wrote {md_path.relative_to(REPO)}")

    print("\n--- summary ---")
    for c in cells:
        if c.status == "done":
            lo, hi = wilson_ci(c.n_success, c.n_total)
            print(f"  {c.name:30s}  {c.n_success:2d}/{c.n_total:2d} = "
                  f"{100*c.success_rate:5.1f}%  [{100*lo:5.1f}, {100*hi:5.1f}]  "
                  f"wall={c.wall_s/60:5.1f} min")
        else:
            print(f"  {c.name:30s}  ({c.status})")


if __name__ == "__main__":
    sys.exit(main())
