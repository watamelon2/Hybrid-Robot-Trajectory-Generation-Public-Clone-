"""Print the SOTA reference table in copy-pasteable Markdown.

Doc-only follow-up B8 (2026-06-01).  Single source of truth for the
numbers lives in ``docs/references/sota_reference_numbers.md``; this
script reproduces the per-task SOTA breakdown as a Markdown table
ready to paste into:

  - presentation/SLIDE_SPEC.md  Slide 5 cross-task panel
  - presentation/DECK_GUIDE.md  Slide 5 callout
  - docs/results.md             section 7 cross-task table

Numbers are transcribed from the cited papers; this script does no
computation.  If the source numbers change (e.g. a new DP variant
publishes a higher rate on can/ph), update both files in lock-step.

Usage::

    python presentation/scripts/print_sota_reference.py
    python presentation/scripts/print_sota_reference.py --format csv
    python presentation/scripts/print_sota_reference.py --format json

Provenance for each entry is in
``docs/references/sota_reference_numbers.md``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from typing import Iterable


# Single source of truth.  Keep in lock-step with
# ``docs/references/sota_reference_numbers.md``.
# (task, sota_pct, method, source_short, obs_modality)
PER_TASK_SOTA: list[tuple[str, float, str, str, str]] = [
    ("Lift",      99.3, "BC-RNN-GMM", "Mandlekar 2021 Table 4", "low-dim, ph"),
    ("Can",       92.3, "BC-RNN-GMM", "Mandlekar 2021 Table 4", "low-dim, ph"),
    ("Square",    74.7, "BC-RNN-GMM", "Mandlekar 2021 Table 4", "low-dim, ph"),
    ("Transport", 65.3, "BC-RNN-GMM", "Mandlekar 2021 Table 4", "low-dim, ph"),
    ("ToolHang",  17.3, "BC-RNN-GMM", "Mandlekar 2021 Table 4", "low-dim, ph"),
]


def emit_markdown(rows: Iterable[tuple]) -> str:
    out = io.StringIO()
    out.write("| Task | SOTA | Method | Source | Obs |\n")
    out.write("|---|---:|---|---|---|\n")
    for task, sota, method, source, obs in rows:
        out.write(f"| {task} | **{sota:.1f} %** | {method} | {source} | {obs} |\n")
    return out.getvalue()


def emit_csv(rows: Iterable[tuple]) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["task", "sota_pct", "method", "source", "obs"])
    for row in rows:
        w.writerow(row)
    return out.getvalue()


def emit_json(rows: Iterable[tuple]) -> str:
    payload = [
        {"task": t, "sota_pct": s, "method": m, "source": src, "obs": o}
        for (t, s, m, src, o) in rows
    ]
    return json.dumps(payload, indent=2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--format", choices=("markdown", "csv", "json"), default="markdown",
        help="Output format (default markdown for paste-into-deck).",
    )
    args = p.parse_args()

    emit = {"markdown": emit_markdown, "csv": emit_csv, "json": emit_json}[args.format]
    sys.stdout.write(emit(PER_TASK_SOTA))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
