#!/usr/bin/env python3
"""Aggregate the pooled nwfix DP grid (Phase 1 + Phase 2 + Phase 3) into the
report's "calibration of test-time guidance" tables.

Reads ``results/vast_safe_dp_nwfix_2026_06_12`` (merged from both machines) and
emits, with Wilson 95% CIs and pooled seed counts:

  1. Flagship over-anchoring curves: success vs lambda for
       Can DP ep0199 sigma0.10, Can DP ep0199 sigma0.25, Square DP ep0199.
  2. Cross-prior-quality ladder: Square ep0049/0099/0149/0199 vs lambda.
  3. Robustness sub-study: nominal vs shifted dynamics (mass/friction), lambda
     0 vs 1, with the guidance robustness delta per shift.

Cell metadata is parsed from the run-dir name; per_lambda rows are pooled over
seed shards within each condition.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = (Path(__file__).resolve().parent
        / "results" / "vast_safe_dp_nwfix_2026_06_12")

CELL_RE = re.compile(
    r"(?P<task>can|square)_dp_ep(?P<ep>\d+)"
    r"(?:_smax(?P<smax>[0-9p]+))?"
    r"_lam(?P<lam>[0-9pm]+)"
    r"(?:_m(?P<mass>[0-9p]+)_f(?P<fric>[0-9p]+))?"
    r"_seed(?P<seed>\d+)_n(?P<n>\d+)"
)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - h), min(1.0, c + h))


def unp(s: str | None) -> str:
    return s.replace("p", ".") if s else "-"


def load() -> dict[tuple, list]:
    """key=(task,ep,smax,mass,fric,lam) -> [k, n, wall, nseeds]."""
    agg: dict[tuple, list] = defaultdict(lambda: [0, 0, 0.0, 0])
    files = sorted(ROOT.rglob("unified_summary_*.json"))
    unparsed = 0
    for f in files:
        m = CELL_RE.search(f.parent.name)
        if not m:
            unparsed += 1
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            unparsed += 1
            continue
        task = m["task"]
        ep = m["ep"]
        smax = unp(m["smax"])
        mass = unp(m["mass"])
        fric = unp(m["fric"])
        pl = d.get("per_lambda", [])
        if isinstance(pl, dict):
            pl = [pl]
        for e in pl:
            lam = float(e.get("lambda_prior", 0.0))
            a = agg[(task, ep, smax, mass, fric, lam)]
            a[0] += int(e.get("n_success", 0))
            a[1] += int(e.get("n_total", 0))
            a[2] += float(e.get("wall_s", 0.0) or 0.0)
            a[3] += 1
    print(f"[load] parsed {len(files)-unparsed}/{len(files)} summaries "
          f"({unparsed} unparsed)  root={ROOT.name}")
    return agg


def is_nominal(mass: str, fric: str) -> bool:
    return mass in ("-", "1.0") and fric in ("-", "1.0")


def row(lam: float, k: int, n: int, ns: int) -> str:
    lo, hi = wilson(k, n)
    rate = (k / n * 100) if n else 0.0
    return (f"  lam {lam:>5.2f}  {k:>3}/{n:<3}  {rate:>5.1f}%  "
            f"[{lo*100:>4.1f},{hi*100:>5.1f}]  (seeds={ns})")


def ci_sep(k1: int, n1: int, k2: int, n2: int) -> bool:
    a = wilson(k1, n1)
    b = wilson(k2, n2)
    return a[1] < b[0] or b[1] < a[0]


def main() -> None:
    agg = load()

    print("\n" + "=" * 70)
    print("1. FLAGSHIP OVER-ANCHORING CURVES (nominal dynamics)")
    print("=" * 70)
    groups: dict[tuple, list] = defaultdict(list)
    for (task, ep, smax, mass, fric, lam), v in agg.items():
        if is_nominal(mass, fric):
            groups[(task, ep, smax)].append((lam, *v))
    for key in sorted(groups):
        task, ep, smax = key
        rows = sorted(groups[key])
        tag = f"{task.upper()} ep{ep}" + (f" smax={smax}" if smax != "-" else "")
        print(f"\n[{tag}]")
        best = max(rows, key=lambda r: (r[1] / r[2]) if r[2] else 0)
        base = next((r for r in rows if r[0] == 0.0), None)
        for lam, k, n, wall, ns in rows:
            mark = "  <- peak" if (lam, k, n, wall, ns) == best else ""
            if base and lam == best[0] and base is not best:
                mark += (" *CI-separated from lam0*"
                         if ci_sep(base[1], base[2], k, n) else " (CI overlaps lam0)")
            print(row(lam, k, n, ns) + mark)

    print("\n" + "=" * 70)
    print("2. CROSS-PRIOR-QUALITY LADDER (Square, nominal)")
    print("=" * 70)
    for ep in ("0049", "0099", "0149", "0199"):
        rows = sorted((lam, *v) for (t, e, sm, ms, fr, lam), v in agg.items()
                      if t == "square" and e == ep and is_nominal(ms, fr))
        if rows:
            print(f"\n[Square ep{ep}]")
            for lam, k, n, wall, ns in rows:
                print(row(lam, k, n, ns))

    print("\n" + "=" * 70)
    print("3. ROBUSTNESS SUB-STUDY (shifted dynamics; guidance delta)")
    print("=" * 70)
    for task in ("can", "square"):
        shifts = sorted({(ms, fr) for (t, e, sm, ms, fr, lam), v in agg.items()
                         if t == task and not is_nominal(ms, fr)})
        if not shifts:
            continue
        print(f"\n[{task.upper()} ep0199 robustness]")
        # nominal reference (lam0, lam1)
        for (ms, fr) in [("1.0", "1.0"), *shifts]:
            def get(lam):
                # robustness study is ep0199 only; for the nominal baseline use
                # the matched BLOCK0 nominal cell (smax0.10 for can).
                best = None
                for (t, e, sm, m2, f2, l2), v in agg.items():
                    if not (t == task and e == "0199" and abs(l2 - lam) < 1e-9):
                        continue
                    if ms == "1.0" and fr == "1.0":
                        if is_nominal(m2, f2):
                            if task == "can" and sm not in ("0.10",):
                                continue
                            best = v
                    elif m2 == ms and f2 == fr:
                        best = v
                return best
            v0, v1 = get(0.0), get(1.0)
            if not v0 or not v1:
                continue
            r0 = v0[0] / v0[1] * 100 if v0[1] else 0
            r1 = v1[0] / v1[1] * 100 if v1[1] else 0
            label = "nominal       " if (ms, fr) == ("1.0", "1.0") else f"mass={ms} fric={fr}"
            print(f"  {label}:  lam0 {v0[0]}/{v0[1]} {r0:>5.1f}%   "
                  f"lam1 {v1[0]}/{v1[1]} {r1:>5.1f}%   "
                  f"delta(lam1-lam0)={r1-r0:>+6.1f}pp")


if __name__ == "__main__":
    main()
