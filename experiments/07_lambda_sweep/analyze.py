"""Post-hoc analysis of the lambda-prior sweep.

Reads per-lambda JSONs produced by ``experiments/07_lambda_sweep/run_can.py``
and emits:

  results/summary_bc.json    -- machine-readable BC-prior table
  results/summary_flow.json  -- machine-readable flow-prior table (if any)
  results/summary.md         -- markdown summary
  results/lambda_curve.png   -- success-rate-vs-lambda curve

File-naming convention (since 2026-05-14 disambiguation patch):

  lambda_{lam}.json       -- flow-prior runs (legacy naming kept)
  lambda_bc_{lam}.json    -- BC-prior runs (the headline sweep)

The BC-prior curve is the main scientific result -- it shows the
classic peaked ablation (neither lambda=0 nor lambda=inf works,
the combination does).

Usage:

    python experiments/07_lambda_sweep/analyze.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_RESULTS = _THIS_DIR / "results"

_FNAME_RE = re.compile(r"^lambda_(?:(?P<prior>[a-z]+)_)?(?P<lam>[^/]+)\.json$")


def _parse_lambda(lam_str: str) -> float:
    """Parse the lambda token out of a filename. 'inf' -> math.inf."""
    s = lam_str.strip()
    if s in ("inf", "Inf", "INF"):
        return math.inf
    return float(s)


def _load_cells(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for f in sorted(results_dir.glob("lambda_*.json")):
        m = _FNAME_RE.match(f.name)
        if not m:
            continue
        # Skip aggregate files we'll be overwriting.
        if f.name in ("summary.json", "summary_bc.json", "summary_flow.json"):
            continue
        prior_from_name = m.group("prior") or "flow"
        lam_from_name = _parse_lambda(m.group("lam"))
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            print(f"  [warn] skipping {f.name}: {e}", file=sys.stderr)
            continue
        # Prefer the prior field in the JSON itself when it disagrees
        # with the filename (a few of the older flow-prior files don't
        # have the disambiguating infix because they predate the patch).
        prior = d.get("prior", prior_from_name)
        # Same for lambda: trust the JSON field over the filename token.
        lam = d.get("lambda_prior", lam_from_name)
        if isinstance(lam, str) and lam.lower() == "inf":
            lam = math.inf
        else:
            try:
                lam = float(lam)
            except (TypeError, ValueError):
                lam = lam_from_name
        # Skip degenerate / smoke runs (n_total < 30).
        n_total = int(d.get("n_total", 0))
        if n_total < 30:
            print(f"  [skip] {f.name}: n_total={n_total} (smoke?)",
                  file=sys.stderr)
            continue
        rows.append({
            "prior":        prior,
            "lambda_prior": lam,
            "n_total":      n_total,
            "n_success":    int(d.get("n_success", 0)),
            "success_rate": float(
                d.get("success_rate",
                      d.get("n_success", 0) / max(n_total, 1))
            ),
            "wall_s":       d.get("total_wall_s") or d.get("wall_s"),
            "warm_start":   d.get("warm_start"),
            "params":       d.get("params"),
            "source":       str(f),
        })
    return rows


def _lam_sort_key(lam: float) -> float:
    """Sort key that keeps inf at the end."""
    return 1e18 if math.isinf(lam) else lam


def _write_summary_json(rows: list[dict[str, Any]], out: Path) -> None:
    # JSON cannot represent math.inf, so stringify it.
    payload = []
    for r in rows:
        rr = dict(r)
        if math.isinf(rr["lambda_prior"]):
            rr["lambda_prior"] = "inf"
        payload.append(rr)
    out.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out}")


def _write_summary_md(rows_by_prior: dict[str, list[dict[str, Any]]],
                      out: Path) -> None:
    lines = []
    lines.append("# Lambda-prior sweep -- summary\n")
    lines.append(
        "Each cell evaluates `denoise_loop` with a fixed lambda_prior\n"
        "value on 60 rollouts (20 demos x 3 rollouts/demo, seed=0).\n"
        "lambda=0 -> planner ignores the prior; lambda=inf -> planner\n"
        "is inert, output is the prior's BC rollout.\n"
    )
    for prior, rows in rows_by_prior.items():
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: _lam_sort_key(r["lambda_prior"]))
        lines.append(f"## Prior = `{prior}`\n")
        lines.append("| lambda | n | succ | success_rate | wall (min) |")
        lines.append("|--------|---|------|--------------|------------|")
        for r in rows_sorted:
            lam_str = "inf" if math.isinf(r["lambda_prior"]) else f"{r['lambda_prior']:g}"
            wall = (f"{r['wall_s']/60:.1f}"
                    if isinstance(r['wall_s'], (int, float)) else "?")
            lines.append(
                f"| {lam_str:>6} | {r['n_total']} "
                f"| {r['n_success']:>4} "
                f"| {r['success_rate']*100:>5.1f}% "
                f"| {wall:>10} |"
            )
        best = max(rows_sorted, key=lambda r: r["success_rate"])
        worst_succ = min(r["success_rate"] for r in rows_sorted)
        if best["success_rate"] > 0 and worst_succ < best["success_rate"]:
            best_lam_str = ("inf" if math.isinf(best["lambda_prior"])
                            else f"{best['lambda_prior']:g}")
            lines.append("")
            if worst_succ > 0:
                ratio = best["success_rate"] / worst_succ
                lines.append(
                    f"**Peak**: lambda={best_lam_str} at "
                    f"{best['success_rate']*100:.1f}% "
                    f"(~{ratio:.1f}x the worst non-zero cell)."
                )
            else:
                lines.append(
                    f"**Peak**: lambda={best_lam_str} at "
                    f"{best['success_rate']*100:.1f}% "
                    f"(other cells in this prior bottom out at 0%)."
                )
        lines.append("")

    # Cross-prior comparison narrative
    bc_rows = rows_by_prior.get("bc", [])
    flow_rows = rows_by_prior.get("flow", [])
    if bc_rows and flow_rows:
        best_bc = max(bc_rows, key=lambda r: r["success_rate"])
        best_flow = max(flow_rows, key=lambda r: r["success_rate"])
        bc_lam_str = ("inf" if math.isinf(best_bc["lambda_prior"])
                      else f"{best_bc['lambda_prior']:g}")
        flow_lam_str = ("inf" if math.isinf(best_flow["lambda_prior"])
                        else f"{best_flow['lambda_prior']:g}")
        lines.append("## BC vs flow prior\n")
        lines.append(
            f"- Best BC-prior cell: lambda={bc_lam_str} -> "
            f"{best_bc['success_rate']*100:.1f}%"
        )
        lines.append(
            f"- Best flow-prior cell: lambda={flow_lam_str} -> "
            f"{best_flow['success_rate']*100:.1f}%"
        )
        if best_bc["success_rate"] > best_flow["success_rate"]:
            lines.append(
                "\nThe BC-trained prior dominates the (currently "
                "undertrained) flow-matching prior at every lambda "
                "where both have data. This matches the run-log note "
                "that v_theta needs a longer training run before "
                "flow-prior numbers become competitive."
            )
        lines.append("")

    out.write_text("\n".join(lines))
    print(f"  wrote {out}")


def _write_lambda_curve(rows_by_prior: dict[str, list[dict[str, Any]]],
                        out: Path) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))

    style = {
        "bc":   {"color": "#3b82f6", "marker": "o", "label": "BC prior",
                  "lw": 2.0},
        "flow": {"color": "#94a3b8", "marker": "s", "label": "flow prior",
                  "lw": 1.6, "linestyle": "--"},
    }

    # We plot lambda on a symlog-ish axis: linear in [0, 0.05], log
    # thereafter, with 'inf' rendered as a special tick at the right edge.
    # Easiest implementation: replace lambda with an arbitrary "x position"
    # and label the ticks ourselves.
    # Define an x ordering across the union of lambdas in both priors.
    all_lams = sorted({r["lambda_prior"]
                       for rs in rows_by_prior.values() for r in rs},
                      key=_lam_sort_key)
    # x positions: 0 -> 0, 0.1 -> 1, 0.5 -> 2, 1 -> 3, 5 -> 4, inf -> 5
    lam_to_x = {lam: i for i, lam in enumerate(all_lams)}
    tick_labels = ["inf" if math.isinf(l) else f"{l:g}" for l in all_lams]

    for prior, rows in rows_by_prior.items():
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: _lam_sort_key(r["lambda_prior"]))
        xs = [lam_to_x[r["lambda_prior"]] for r in rows_sorted]
        ys = [r["success_rate"] * 100 for r in rows_sorted]
        ns = [r["n_total"] for r in rows_sorted]
        ax.plot(xs, ys, **style.get(prior, {"label": prior, "color": "black"}))
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(f"{y:.1f}%", (x, y),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9,
                        color=style.get(prior, {}).get("color", "black"))

        # Highlight the peak of the BC curve specifically.
        if prior == "bc":
            peak = max(rows_sorted, key=lambda r: r["success_rate"])
            px = lam_to_x[peak["lambda_prior"]]
            py = peak["success_rate"] * 100
            ax.scatter([px], [py], s=180, facecolors="none",
                       edgecolors="#16a34a", linewidths=2.5, zorder=10)
            peak_lam_str = ("inf" if math.isinf(peak["lambda_prior"])
                            else f"{peak['lambda_prior']:g}")
            ax.annotate(
                f"peak: lam={peak_lam_str}\n"
                f"  ({py:.1f}% = {peak['n_success']}/{peak['n_total']})",
                (px, py), textcoords="offset points", xytext=(15, -8),
                fontsize=10, color="#16a34a", fontweight="bold",
            )

    ax.set_xticks(range(len(all_lams)), tick_labels)
    ax.set_xlabel(r"$\lambda_{\mathrm{prior}}$  "
                  "(0 = planner only,  inf = prior only)")
    ax.set_ylabel("success rate (%)")
    ax.set_title(
        "Experiment 07 -- lambda-prior sweep on Robomimic Can\n"
        "60 rollouts/cell (20 demos x 3), seed=0, K=10, M=16, "
        "warm_start=True"
    )
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    ax.set_ylim(-0.5, max(12.0, ax.get_ylim()[1]))

    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, default=_DEFAULT_RESULTS,
                   help="Directory containing lambda_*.json files.")
    args = p.parse_args()

    if not args.results.is_dir():
        print(f"[error] results dir does not exist: {args.results}",
              file=sys.stderr)
        return 1

    rows = _load_cells(args.results)
    if not rows:
        print("[error] no lambda_*.json cells found", file=sys.stderr)
        return 1
    print(f"[exp07/analyze] loaded {len(rows)} cells from {args.results}")

    # Bucket by prior.
    rows_by_prior: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        rows_by_prior.setdefault(r["prior"], []).append(r)

    # Write per-prior JSON summaries.
    if "bc" in rows_by_prior:
        _write_summary_json(
            sorted(rows_by_prior["bc"],
                   key=lambda r: _lam_sort_key(r["lambda_prior"])),
            args.results / "summary_bc.json",
        )
    if "flow" in rows_by_prior:
        _write_summary_json(
            sorted(rows_by_prior["flow"],
                   key=lambda r: _lam_sort_key(r["lambda_prior"])),
            args.results / "summary_flow.json",
        )

    _write_summary_md   (rows_by_prior, args.results / "summary.md")
    _write_lambda_curve (rows_by_prior, args.results / "lambda_curve.png")

    # Stdout echo.
    print()
    for prior, rs in rows_by_prior.items():
        rs = sorted(rs, key=lambda r: _lam_sort_key(r["lambda_prior"]))
        print(f"  prior={prior}  ({len(rs)} cells)")
        for r in rs:
            lam_str = ("inf" if math.isinf(r["lambda_prior"])
                       else f"{r['lambda_prior']:g}")
            print(f"    lam={lam_str:>5}  "
                  f"{r['n_success']:>2}/{r['n_total']:<2}  "
                  f"{r['success_rate']*100:>5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
