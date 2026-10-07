"""Aggregator for the T2 Square cross-task ladder.

NOTE -- this is a NULL RESULT.  The Square hybrid pipeline does not
work at mass = 1.3 (or at mass = 1.0; Slate G found Square hybrid
already at 0-8 % under every perturbation knob).  Two issues are
recorded as diagnosed:

  1. T2.0_square slate bug: the cell launcher did NOT pass
     ``--prior-bank-dataset``, so the demo_1nn prior fell back to
     an empty/default bank.  The 3.3 % result is therefore not a
     valid T2.0 (canonical 200 Square bank) measurement.

  2. T2.1_square / T2.4_square ran with the correct
     ``square_mass1p3_205demos.hdf5`` / ``_220demos.hdf5`` banks
     (workhorse built them ~02:00 PDT, pushed to data/banks LFS).
     Both still got 0-3.3 %, consistent with the canonical
     mass=1.0 Square hybrid floor measured in Slate G.

Reads
-----
* T2.0/T2.1/T2.4 Square cells from
  ``results/slate_incremental_T2_square_hybrid_2026_05_28/``.

Outputs (only with ``--write``)
-------------------------------
* ``results/slate_incremental_T2_square_aggregate/summary.json``
* ``docs/results_writeups/t2_square.md`` -- the "Square T2 table"
  section is filled in and the diagnostic narrative is appended.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

THIS = Path(__file__).resolve().parent
RESULTS = THIS / "results"
SQ_DIR = RESULTS / "slate_incremental_T2_square_hybrid_2026_05_28"
OUT_DIR = RESULTS / "slate_incremental_T2_square_aggregate"
WRITEUP = THIS.parents[1] / "docs" / "results_writeups" / "t2_square.md"

# (short, label, tag)
CELLS = [
    ("T2_0_sq",  "T2.0_square (canonical bank, slate bug: no --prior-bank-dataset passed)",
                 "T2_0_square_hybrid_canonical_bank_mass1p3"),
    ("T2_1_sq",  "T2.1_square (5 perturbed mass=1.3 Square demos)",
                 "T2_1_square_hybrid_5demo_adapted_mass1p3"),
    ("T2_4_sq",  "T2.4_square (20 perturbed mass=1.3 Square demos, saturation)",
                 "T2_4_square_hybrid_20demo_saturation_mass1p3"),
]


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _read_cell(prior: str, tag: str) -> dict | None:
    if not SQ_DIR.exists():
        return None
    matches = sorted(SQ_DIR.glob(f"unified_summary_{prior}_*_{tag}.json"))
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


def _read_rollout_params(tag: str) -> dict | None:
    """Return prior_bank_* fields from the underlying rollout JSON."""
    matches = sorted(SQ_DIR.glob(f"unified_demo_1nn_lam-1.0_*_{tag}.json"))
    if not matches:
        return None
    d = json.loads(matches[0].read_text())
    p = d.get("params", {})
    return {
        "prior_bank_dataset": p.get("prior_bank_dataset"),
        "prior_bank_size": p.get("prior_bank_size"),
        "prior_bank_task": p.get("prior_bank_task"),
        "task": p.get("task"),
        "mass_scale": p.get("mass_scale"),
    }


def collect() -> dict:
    out: dict = {}
    for short, label, tag in CELLS:
        cell = _read_cell("demo_1nn", tag)
        params = _read_rollout_params(tag)
        out[short] = {
            "label": label,
            "tag": tag,
            **(cell or {}),
            "rollout_params": params,
        }
    return out


def _slot(v: dict) -> tuple[str, str]:
    if "n_success" not in v:
        return ("—", "—")
    k, n = v["n_success"], v["n_total"]
    rate = f"{k}/{n} = {100*v['rate']:.1f}%"
    lo, hi = _wilson(k, n)
    ci = f"[{100*lo:.1f}, {100*hi:.1f}]"
    return rate, ci


def render_square_table(data: dict) -> str:
    rows = [
        "| Cell | bank arg used | succ | Wilson 95 % CI |",
        "|---|---|---:|---:|",
    ]
    for short, _, _ in CELLS:
        v = data[short]
        rate, ci = _slot(v)
        bank = (v.get("rollout_params") or {}).get("prior_bank_dataset")
        bank_str = "None (slate bug)" if bank is None else Path(bank).name
        label = v["label"].split(" (", 1)[0]
        rows.append(f"| {label} | {bank_str} | {rate} | {ci} |")
    return "\n".join(rows) + "\n"


def render_diagnostic(data: dict) -> str:
    lines = ["### Diagnostic narrative (computed 2026-05-29)\n"]
    t20 = data["T2_0_sq"]
    t21 = data["T2_1_sq"]
    t24 = data["T2_4_sq"]

    lines.append("**Two distinct failure modes confirmed:**\n")

    lines.append(
        "1. **T2.0_square slate bug.**  The launcher run "
        "`experiments/15_unified_hybrid/slate_incremental_T2_square_hybrid_2026_05_28.sh` "
        "for the T2.0 cell does NOT pass `--prior-bank-dataset`:"
    )
    lines.append("")
    lines.append("   ```bash")
    lines.append("   # T2.0_square: hybrid + canonical 200 Square bank @ mass=1.3.")
    lines.append("   run \"T2_0_square_hybrid_canonical_bank_mass1p3\" \\")
    lines.append("     --prior demo_1nn --lambda-prior 1")
    lines.append("   ```")
    lines.append("")
    bank20 = (t20.get("rollout_params") or {}).get("prior_bank_dataset")
    lines.append(
        f"   At runtime, `prior_bank_dataset = {bank20!r}` was recorded in the "
        f"rollout JSON.  The demo_1nn prior fell back to a default/empty bank; "
        f"the resulting {t20['n_success']}/{t20['n_total']} = "
        f"{100*t20['rate']:.1f}% is **not a valid measurement** of "
        f"T2.0_square (canonical 200 Square bank).  Future re-run needs the "
        f"missing `--prior-bank-dataset \"$REPO/.../square_canonical_200.hdf5\"` arg."
    )
    lines.append("")

    bank21 = Path((t21.get("rollout_params") or {}).get("prior_bank_dataset") or "?").name
    bank24 = Path((t24.get("rollout_params") or {}).get("prior_bank_dataset") or "?").name
    lines.append(
        "2. **T2.1_square / T2.4_square ran with the right banks but the "
        "upstream Square hybrid pipeline is broken.**  Both cells loaded the "
        f"workhorse-built banks (`{bank21}` / `{bank24}`), produced "
        f"{t21['n_success']}/{t21['n_total']} = {100*t21['rate']:.1f}% / "
        f"{t24['n_success']}/{t24['n_total']} = {100*t24['rate']:.1f}% "
        "success respectively.  These are consistent with the canonical-mass "
        "Square hybrid floor that Slate G (2026-05-21) already measured at "
        "0-8 % under every perturbation knob (sigma sweep, soft-NN beta "
        "sweep, action history).  The mass=1.3 perturbation is therefore "
        "**not the killer here** -- the Square hybrid pipeline was already "
        "broken at canonical mass=1.0."
    )
    lines.append("")
    lines.append(
        "**Net verdict.**  T2 transfer to Square cannot be demonstrated until "
        "the upstream Square hybrid pipeline is fixed (suspect obs schema / "
        "coordinate-frame issue, similar to the DP Phase-A finding for "
        "Can at the legacy 46-D `can` schema).  Future work, not blocking "
        "the Can T2 headline."
    )
    return "\n".join(lines) + "\n"


def update_writeup(data: dict) -> None:
    table = render_square_table(data)
    diagnostic = render_diagnostic(data)
    text = WRITEUP.read_text()
    block = (
        "### Square T2 table\n\n"
        f"{table}\n"
        f"{diagnostic}\n"
        "### Cross-task confirmation table (Can vs Square)"
    )
    pattern = re.compile(
        r"### Square T2 table\n.*?\n### Cross-task confirmation table \(Can vs Square\)",
        flags=re.DOTALL,
    )
    if pattern.search(text):
        text = pattern.sub(block, text)
    else:
        text += "\n\n" + block
    text = re.sub(
        r"\*\*Status:\*\*\s+.*",
        "**Status:**  results landed 2026-05-29; aggregation done; NULL RESULT diagnosed.",
        text,
    )
    WRITEUP.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="Persist summary.json and patch t2_square.md.")
    args = ap.parse_args()
    data = collect()
    print(render_square_table(data))
    print(render_diagnostic(data))
    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "summary.json").write_text(json.dumps(data, indent=2))
        print(f"\n[wrote] {OUT_DIR / 'summary.json'}")
        update_writeup(data)
        print(f"[updated] {WRITEUP}")


if __name__ == "__main__":
    main()
