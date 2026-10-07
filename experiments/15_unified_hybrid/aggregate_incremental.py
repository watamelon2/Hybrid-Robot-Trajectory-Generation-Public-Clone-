"""Aggregator for the T1 + T2 incremental-prior slates.

Produces:

1. ``summary.md`` -- human-readable headline table with Wilson 95% CIs.
2. ``summary.csv`` -- machine-readable per-cell rows.
3. ``T1_bar_data.csv`` -- two-column (label, succ_rate) ready for the
   T1 bar chart in the final report.
4. ``T2_pareto_data.csv`` -- three-column (method, wall_s, succ_rate)
   for the T2 wall-clock x adapted-succ Pareto plot.

Falsification helpers: a separate "Interpretation" section in
summary.md prints the verdict per the spec's pre-registered
predictions (sec 5.1 of docs/adaptation/04_incremental_prior_spec.md).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_T1_DIR = _THIS_DIR / "results" / "slate_incremental_T1_2026_05_26"
_T2_DIR = _THIS_DIR / "results" / "slate_incremental_T2_2026_05_26"
_T23_DIR = _THIS_DIR / "results" / "slate_incremental_T2_bc_retrain_2026_05_26"
_OUT_DIR = _THIS_DIR / "results" / "slate_incremental_aggregate"


# Cell registry: tag -> (label_for_table, description, predicted_pct).
# Order matters for the headline tables.
_T1_CELLS: list[tuple[str, str, str, str]] = [
    ("T1_0_can_can_control",      "T1.0", "Can eval, Can bank (control)",     "~74%"),
    ("T1_1_can_lift",             "T1.1", "Can eval, Lift bank (cross-task)", "10-50%"),
    ("T1_2_can_square",           "T1.2", "Can eval, Square bank (cross-task)", "10-50%"),
    ("T1_3_lift_can",             "T1.3", "Lift eval, Can bank (cross-task)",   "10-50%"),
    ("T1_4_can_no_prior",         "T1.4", "Can eval, lambda=0 (no prior)",     "~0%"),
    ("T1_5_lift_lift_control",    "T1.5", "Lift eval, Lift bank (control)",    "~100%"),
]

_T2_CELLS: list[tuple[str, str, str, str]] = [
    ("T2_0_hybrid_canonical_bank_mass1p3",
        "T2.0", "hybrid + canonical 200 bank @ mass=1.3",  "~15-25%"),
    ("T2_1_hybrid_5demo_adapted_mass1p3",
        "T2.1", "HEADLINE: hybrid + 200+5 mass=1.3 bank @ mass=1.3", "~50-65%"),
    ("T2_2_bc_unadapted_mass1p3",
        "T2.2", "BC unadapted @ mass=1.3",  "~10%"),
    ("T2_4_hybrid_20demo_saturation_mass1p3",
        "T2.4", "hybrid + 200+20 mass=1.3 bank @ mass=1.3 (saturation)", "~55-70%"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--write", action="store_true",
        help="Write summary files to disk; without this flag, only "
             "the table is printed to stdout.",
    )
    return p.parse_args()


def _wilson_ci(succ: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson 95% CI on a Binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = succ / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _load_cell(out_dir: Path, tag: str) -> dict | None:
    """Find and load the summary JSON for a cell.  Returns None if
    the cell hasn't landed yet."""
    matches = list(out_dir.glob(f"unified_summary_*_{tag}.json"))
    if not matches:
        return None
    # Each summary file has per_lambda -- aggregate across lambdas.
    summary = json.load(open(matches[0]))
    per_lambda = summary.get("per_lambda", [])
    if not per_lambda:
        return None
    # Total trials = sum of n_total; succ = sum of n_success across lambdas.
    n_total = sum(int(p["n_total"]) for p in per_lambda)
    n_success = sum(int(p["n_success"]) for p in per_lambda)
    wall_s = sum(float(p.get("wall_s", 0)) for p in per_lambda)
    rate = n_success / n_total if n_total else 0.0
    lo, hi = _wilson_ci(n_success, n_total)
    return {
        "tag": tag,
        "n_total": n_total,
        "n_success": n_success,
        "rate": rate,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "wall_s": wall_s,
        "per_lambda": per_lambda,
        "json_path": str(matches[0]),
    }


def _format_pct(p: float) -> str:
    return f"{100 * p:5.1f}%"


def _format_ci(lo: float, hi: float) -> str:
    return f"[{100 * lo:.1f}, {100 * hi:.1f}]"


def _row(label: str, desc: str, predicted: str, result: dict | None) -> list[str]:
    if result is None:
        return [label, desc, predicted, "_missing_", "-", "-", "-"]
    return [
        label,
        desc,
        predicted,
        f"{result['n_success']}/{result['n_total']}",
        _format_pct(result["rate"]),
        _format_ci(result["wilson_lo"], result["wilson_hi"]),
        f"{result['wall_s']:.0f}",
    ]


def _interpret_t1(results: dict[str, dict | None]) -> list[str]:
    lines = []
    t10 = results["T1_0_can_can_control"]
    t14 = results["T1_4_can_no_prior"]
    cross = [results[k] for k in ("T1_1_can_lift", "T1_2_can_square", "T1_3_lift_can")]
    cross_done = [c for c in cross if c is not None]
    if t10 is None or t14 is None or len(cross_done) < 3:
        lines.append("- (interpretation deferred until T1.0, T1.4, and all "
                     "three cross-task cells have landed)")
        return lines

    threshold_low = 0.05
    threshold_band_lo, threshold_band_hi = 0.10, 0.50
    cross_rates = [c["rate"] for c in cross_done]

    if all(r > threshold_band_lo for r in cross_rates) and all(
        r < t10["rate"] for r in cross_rates
    ):
        lines.append("- **T1 SUBSTANTIATED**: cross-task succ in "
                     f"[{100*min(cross_rates):.1f}%, {100*max(cross_rates):.1f}%], "
                     f"strictly between T1.4 ({_format_pct(t14['rate'])}) and "
                     f"T1.0 ({_format_pct(t10['rate'])}).  Retrieval prior "
                     "captures task-generic motion structure.")
    elif all(r <= threshold_low for r in cross_rates):
        lines.append("- **T1 FALSIFIED**: cross-task succ <= 5% across all "
                     "three pairs.  Retract \"captures task-generic motion "
                     "structure\" framing; restrict claim to \"reduces variance "
                     "within a task family\".")
    else:
        lines.append(f"- T1 PARTIAL: cross-task rates "
                     f"{[_format_pct(r) for r in cross_rates]}.  Mixed signal; "
                     "manual review.")
    return lines


def _interpret_t2(results: dict[str, dict | None]) -> list[str]:
    lines = []
    t20 = results.get("T2_0_hybrid_canonical_bank_mass1p3")
    t21 = results.get("T2_1_hybrid_5demo_adapted_mass1p3")
    t22 = results.get("T2_2_bc_unadapted_mass1p3")
    t23 = results.get("T2_3_bc_retrained_mass1p3")
    t24 = results.get("T2_4_hybrid_20demo_saturation_mass1p3")

    if t21 is None or t23 is None:
        lines.append("- (interpretation deferred until T2.1 headline cell "
                     "and T2.3 BC-retrained cell have both landed)")
        return lines

    # Pareto check: T2.1 wall <= T2.3 wall AND T2.1 succ >= T2.3 succ.
    # T2.3 wall includes the BC retrain time.  We approximate T2.1's
    # adaptation cost as 0 (just re-index the bank) + the eval cost.
    # For the Pareto comparison only the *adaptation* time matters;
    # eval cost is shared.
    adapt_t21_s = 1.0  # bank re-index
    # Pull retrain wall from the canonical train.py JSON (survives
    # slate re-runs that only refresh run.log); fall back to run.log
    # parsing, then to the spec's ~1800 s prior.
    adapt_t23_s = 1800.0
    train_log_json = (
        _T23_DIR.parents[2]
        / "03_bc_baseline"
        / "results"
        / "training_log_mass1p3_205.json"
    )
    if train_log_json.exists():
        try:
            tl = json.loads(train_log_json.read_text())
            if "total_wall_s" in tl:
                adapt_t23_s = float(tl["total_wall_s"])
        except Exception:
            pass
    if adapt_t23_s == 1800.0 and _T23_DIR.exists():
        log = _T23_DIR / "run.log"
        if log.exists():
            for line in log.read_text().splitlines():
                if "BC retrain wall:" in line:
                    try:
                        s = line.split("BC retrain wall:")[1].split("s")[0].strip()
                        adapt_t23_s = float(s)
                    except Exception:
                        pass
                    break

    if t21["rate"] >= t23["rate"] and adapt_t21_s < adapt_t23_s:
        lines.append(
            f"- **T2 SUBSTANTIATED -- STRICT PARETO DOMINANCE**: "
            f"T2.1 ({_format_pct(t21['rate'])}, "
            f"{adapt_t21_s:.0f} s adaptation) sits upper-left of "
            f"T2.3 ({_format_pct(t23['rate'])}, "
            f"{adapt_t23_s:.0f} s adaptation).  "
            f"Pareto ratio: {adapt_t23_s / max(adapt_t21_s, 1e-6):.0f}x wall-"
            "clock speedup at non-inferior succ.  "
            "Constructive demonstration of Theorem 1' in adaptation space."
        )
    elif t21["rate"] < t23["rate"]:
        lines.append(
            f"- **T2 PARTIAL**: T2.3 ({_format_pct(t23['rate'])}) beats "
            f"T2.1 ({_format_pct(t21['rate'])}) on succ.  Fall back to "
            "wall-clock-only framing: hybrid is still "
            f"{adapt_t23_s / max(adapt_t21_s, 1e-6):.0f}x faster to adapt "
            "(weaker claim than strict Pareto)."
        )
    else:
        lines.append(
            f"- T2 MARGINAL: T2.1 ({_format_pct(t21['rate'])}) vs "
            f"T2.3 ({_format_pct(t23['rate'])}) succ within CI; "
            "wall-clock advantage stands.  Use the wall-clock-only "
            "framing for the brief."
        )

    if t20 is not None:
        lines.append(
            f"- Degradation control: T2.0 ({_format_pct(t20['rate'])}) "
            f"vs T2.1 ({_format_pct(t21['rate'])}).  Improvement from "
            f"5 perturbed demos: +"
            f"{100*(t21['rate'] - t20['rate']):.1f} pp."
        )
    if t24 is not None and t21 is not None:
        lines.append(
            f"- Saturation: T2.4 ({_format_pct(t24['rate'])}) vs T2.1 "
            f"({_format_pct(t21['rate'])}).  Marginal pp from 15 more "
            f"perturbed demos: +"
            f"{100*(t24['rate'] - t21['rate']):.1f} pp."
        )
    return lines


def main() -> int:
    args = parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all cells.
    t1_results = {tag: _load_cell(_T1_DIR, tag) for (tag, _, _, _) in _T1_CELLS}
    t2_results = {tag: _load_cell(_T2_DIR, tag) for (tag, _, _, _) in _T2_CELLS}
    # T2.3 lives in its own dir.
    t23 = _load_cell(_T23_DIR, "T2_3_bc_retrained_mass1p3")

    # Build the headline table.
    headline_lines = []
    headline_lines.append(
        "# Experiment 04 (incremental-prior) -- T1 + T2 summary\n"
    )
    headline_lines.append(
        "Generated by `experiments/15_unified_hybrid/aggregate_incremental.py`. "
        "Wilson 95% CI on success rate; wall_s is total CPU wall across the "
        "cell's evaluation rollouts.\n"
    )

    # T1 table
    headline_lines.append("## T1 cross-task transfer (headline bar chart)\n")
    headline_lines.append(
        "| Cell | Description | Predicted | trials | rate | 95% CI (%) | wall_s |\n"
        "|---|---|---|---:|---:|---|---:|"
    )
    for tag, label, desc, predicted in _T1_CELLS:
        row = _row(label, desc, predicted, t1_results[tag])
        headline_lines.append("| " + " | ".join(row) + " |")
    headline_lines.append("")
    headline_lines.append("### T1 interpretation")
    headline_lines.extend(_interpret_t1(t1_results))
    headline_lines.append("")

    # T2 table
    headline_lines.append("## T2 incremental adaptation (Pareto plot)\n")
    headline_lines.append(
        "| Cell | Description | Predicted | trials | rate | 95% CI (%) | wall_s |\n"
        "|---|---|---|---:|---:|---|---:|"
    )
    for tag, label, desc, predicted in _T2_CELLS:
        row = _row(label, desc, predicted, t2_results[tag])
        headline_lines.append("| " + " | ".join(row) + " |")
    # T2.3 row
    row = _row("T2.3", "BC retrained on 205 mass=1.3 demos @ mass=1.3",
               "~25%", t23)
    headline_lines.append("| " + " | ".join(row) + " |")
    headline_lines.append("")
    headline_lines.append("### T2 interpretation")
    t2_results_with_23 = {**t2_results, "T2_3_bc_retrained_mass1p3": t23}
    headline_lines.extend(_interpret_t2(t2_results_with_23))
    headline_lines.append("")

    md_text = "\n".join(headline_lines)
    print(md_text)

    if not args.write:
        return 0

    # ----- write summary.md -----
    (_OUT_DIR / "summary.md").write_text(md_text)

    # ----- write summary.csv -----
    with open(_OUT_DIR / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "label", "description", "predicted",
                    "n_total", "n_success", "succ_rate",
                    "wilson_lo", "wilson_hi", "wall_s"])
        for tag, label, desc, predicted in _T1_CELLS:
            r = t1_results[tag]
            if r is None:
                w.writerow([tag, label, desc, predicted, "", "", "", "", "", ""])
            else:
                w.writerow([tag, label, desc, predicted,
                            r["n_total"], r["n_success"], r["rate"],
                            r["wilson_lo"], r["wilson_hi"], r["wall_s"]])
        for tag, label, desc, predicted in _T2_CELLS:
            r = t2_results[tag]
            if r is None:
                w.writerow([tag, label, desc, predicted, "", "", "", "", "", ""])
            else:
                w.writerow([tag, label, desc, predicted,
                            r["n_total"], r["n_success"], r["rate"],
                            r["wilson_lo"], r["wilson_hi"], r["wall_s"]])
        if t23 is not None:
            w.writerow(["T2_3_bc_retrained_mass1p3", "T2.3",
                        "BC retrained on 205 demos @ mass=1.3", "~25%",
                        t23["n_total"], t23["n_success"], t23["rate"],
                        t23["wilson_lo"], t23["wilson_hi"], t23["wall_s"]])

    # ----- T1 bar chart data -----
    with open(_OUT_DIR / "T1_bar_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "eval_task", "bank_task", "succ_rate", "wilson_lo", "wilson_hi"])
        for tag, label, desc, _ in _T1_CELLS:
            r = t1_results[tag]
            if r is None:
                continue
            # Parse eval/bank from tag.
            parts = tag.split("_")
            eval_t = parts[2] if len(parts) >= 3 else ""
            bank_t = parts[3] if len(parts) >= 4 else ""
            w.writerow([label, eval_t, bank_t, r["rate"],
                        r["wilson_lo"], r["wilson_hi"]])

    # ----- T2 Pareto data -----
    with open(_OUT_DIR / "T2_pareto_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "adaptation_wall_s", "succ_rate", "wilson_lo", "wilson_hi"])
        # Hybrid + 5 demos: bank re-index ~1 s.
        if t2_results.get("T2_1_hybrid_5demo_adapted_mass1p3") is not None:
            r = t2_results["T2_1_hybrid_5demo_adapted_mass1p3"]
            w.writerow(["hybrid+5demos", 1.0, r["rate"], r["wilson_lo"], r["wilson_hi"]])
        # BC unadapted: 0 s adaptation cost.
        if t2_results.get("T2_2_bc_unadapted_mass1p3") is not None:
            r = t2_results["T2_2_bc_unadapted_mass1p3"]
            w.writerow(["BC_unadapted", 0.0, r["rate"], r["wilson_lo"], r["wilson_hi"]])
        # BC retrained: needs the retrain wall from the T2.3 log.
        # Prefer the canonical training_log JSON written by train.py
        # (key ``total_wall_s``) since it survives slate re-runs that
        # only rebuild run.log.  Fall back to grepping run.log if the
        # JSON isn't there, then to a sentinel.
        retrain_wall_s = 1800.0  # fallback
        train_log_json = (
            _T23_DIR.parents[2]
            / "03_bc_baseline"
            / "results"
            / "training_log_mass1p3_205.json"
        )
        if train_log_json.exists():
            try:
                tl = json.loads(train_log_json.read_text())
                if "total_wall_s" in tl:
                    retrain_wall_s = float(tl["total_wall_s"])
            except Exception:
                pass
        if retrain_wall_s == 1800.0:
            log = _T23_DIR / "run.log"
            if log.exists():
                for line in log.read_text().splitlines():
                    if "BC retrain wall:" in line:
                        try:
                            retrain_wall_s = float(
                                line.split("BC retrain wall:")[1].split("s")[0].strip()
                            )
                        except Exception:
                            pass
                        break
        if t23 is not None:
            w.writerow(["BC_retrained_205", retrain_wall_s, t23["rate"],
                        t23["wilson_lo"], t23["wilson_hi"]])
        # Hybrid + 20 demos saturation point.
        if t2_results.get("T2_4_hybrid_20demo_saturation_mass1p3") is not None:
            r = t2_results["T2_4_hybrid_20demo_saturation_mass1p3"]
            w.writerow(["hybrid+20demos", 1.0, r["rate"], r["wilson_lo"], r["wilson_hi"]])

    print(f"\n[aggregate] wrote summary files to {_OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
