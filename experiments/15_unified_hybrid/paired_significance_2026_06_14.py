#!/usr/bin/env python3
"""Matched-seed paired significance for the nwfix DP guidance grid.

The flagship cells are seed-blocked: lambda0 and lambda* are evaluated on the
*same* RNG seeds (same 6 task instances per block).  Comparing two independent
Wilson CIs throws away that pairing and is badly under-powered.  Here we instead
test the WITHIN-seed difference d_s = rate(lambda*, s) - rate(lambda0, s):

  * per-family paired bootstrap 95% CI on mean(d_s) + one-sided sign test;
  * a POOLED test that stacks the matched per-seed diffs across the three
    independent flagship curves (the report's actual claim: a consistent
    over-anchoring lift), which is the highest-powered defensible statistic.

Uses only numpy.  Reads the same root as analyze_nwfix_grid_2026_06_13.py.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = (Path(__file__).resolve().parent
        / "results" / "vast_safe_dp_nwfix_2026_06_12")

CELL_RE = re.compile(
    r"(?P<task>can|square)_dp_ep(?P<ep>\d+)"
    r"(?:_smax(?P<smax>[0-9p]+))?"
    r"_lam(?P<lam>[0-9pm]+)"
    r"(?:_m(?P<mass>[0-9p]+)_f(?P<fric>[0-9p]+))?"
    r"_seed(?P<seed>\d+)_n(?P<n>\d+)"
)

RNG = np.random.default_rng(0)


def unp(s: str | None) -> str:
    return s.replace("p", ".") if s else "-"


def is_nominal(mass: str, fric: str) -> bool:
    return mass in ("-", "1.0") and fric in ("-", "1.0")


def load_per_seed() -> dict[tuple, tuple[int, int]]:
    """key=(task,ep,smax,mass,fric,lam,seed) -> (k, n)."""
    rows: dict[tuple, tuple[int, int]] = {}
    for f in ROOT.rglob("unified_summary_*.json"):
        m = CELL_RE.search(f.parent.name)
        if not m:
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        pl = d.get("per_lambda", [])
        if isinstance(pl, dict):
            pl = [pl]
        for e in pl:
            lam = float(e.get("lambda_prior", 0.0))
            key = (m["task"], m["ep"], unp(m["smax"]),
                   unp(m["mass"]), unp(m["fric"]), lam, m["seed"])
            k, n = rows.get(key, (0, 0))
            rows[key] = (k + int(e.get("n_success", 0)),
                         n + int(e.get("n_total", 0)))
    return rows


def matched_diffs(rows, task, ep, smax, lam0, lam1, shift=None):
    """Per-seed rate differences rate(lam1) - rate(lam0) on common seeds."""
    def pick(lam):
        out = {}
        for (t, e, sm, ms, fr, l, seed), (k, n) in rows.items():
            if not (t == task and e == ep and sm == smax and abs(l - lam) < 1e-9):
                continue
            ok = (is_nominal(ms, fr) if shift is None else (ms, fr) == shift)
            if ok and n:
                out[seed] = (k, n)
        return out
    a, b = pick(lam0), pick(lam1)
    seeds = sorted(set(a) & set(b))
    d = np.array([b[s][0] / b[s][1] - a[s][0] / a[s][1] for s in seeds])
    return seeds, d


def boot_ci(d: np.ndarray, B: int = 50000):
    if len(d) == 0:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def sign_p(d: np.ndarray) -> float:
    """Two-sided sign test p-value (exact binomial, ties dropped)."""
    pos = int((d > 0).sum())
    neg = int((d < 0).sum())
    n = pos + neg
    if n == 0:
        return 1.0
    from math import comb
    k = max(pos, neg)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def report(label, d):
    if len(d) == 0:
        print(f"  {label:<34} no matched seeds")
        return
    lo, hi = boot_ci(d)
    mean = d.mean()
    star = "  *** CI excludes 0" if (lo > 0 or hi < 0) else "  (CI includes 0)"
    print(f"  {label:<34} n={len(d):>2}  Delta={mean*100:>+5.1f}pp  "
          f"95%CI[{lo*100:>+5.1f},{hi*100:>+5.1f}]  sign-p={sign_p(d):.3f}{star}")


def main():
    rows = load_per_seed()
    n_cells = len(rows)
    print(f"[load] {n_cells} (family,seed,lambda) cells from {ROOT.name}\n")

    # Flagship families: (task, ep, smax, lam0, lam_peak, lam_canonical)
    fams = [
        ("can", "0199", "0.10", 0.0, 0.75, 1.0),
        ("can", "0199", "0.25", 0.0, 10.0, 1.0),
        ("square", "0199", "-", 0.0, 1.0, 1.0),
    ]
    print("=" * 74)
    print("FLAGSHIP over-anchoring: matched-seed paired lift vs lambda0")
    print("=" * 74)
    pooled_peak, pooled_canon = [], []
    for task, ep, smax, l0, lpk, lcn in fams:
        tag = f"{task} ep{ep}" + (f" s{smax}" if smax != "-" else "")
        _, dpk = matched_diffs(rows, task, ep, smax, l0, lpk)
        _, dcn = matched_diffs(rows, task, ep, smax, l0, lcn)
        print(f"\n[{tag}]")
        report(f"lam0 -> lam{lcn:g} (canonical)", dcn)
        report(f"lam0 -> lam{lpk:g} (empirical peak)", dpk)
        pooled_canon.append(dcn)
        pooled_peak.append(dpk)

    print("\n" + "-" * 74)
    print("POOLED across the 3 independent flagship curves (the report's claim)")
    print("-" * 74)
    report("lam0 -> canonical lam1", np.concatenate(pooled_canon))
    report("lam0 -> empirical peak", np.concatenate(pooled_peak))

    # Robustness: paired lam1 vs lam0 within each shift cell (ep0199).
    print("\n" + "=" * 74)
    print("ROBUSTNESS: matched-seed paired lift (lam1 vs lam0) per shift")
    print("=" * 74)
    for task in ("can", "square"):
        smax = "0.10" if task == "can" else "-"
        shifts = sorted({(ms, fr) for (t, e, sm, ms, fr, l, s) in rows
                         if t == task and e == "0199" and not is_nominal(ms, fr)})
        print(f"\n[{task} ep0199]")
        _, dn = matched_diffs(rows, task, "0199", smax, 0.0, 1.0)
        report("nominal", dn)
        for ms, fr in shifts:
            _, d = matched_diffs(rows, task, "0199", smax, 0.0, 1.0, shift=(ms, fr))
            report(f"mass={ms} fric={fr}", d)


if __name__ == "__main__":
    main()
