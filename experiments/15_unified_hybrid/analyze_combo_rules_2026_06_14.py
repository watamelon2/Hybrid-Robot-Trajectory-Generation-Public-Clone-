#!/usr/bin/env python3
"""Additive-vs-multiplicative combination-rule contrast analysis.

Reads results/combo_rules_2026_06_14 (override with arg1) where each cell is
  {task}_dp_ep0199[_smax0p10]_lam1_comb{add,prod,temp,gate}_seed{S}_n6
evaluated on MATCHED seeds.  Reports, per task:

  * success rate per fusion rule (additive bridge vs product / temper / max-gate),
    pooled over seeds with Wilson CIs;
  * matched-seed paired lift of additive over each multiplicative rule
    (paired-bootstrap CI + sign test) -- the report's "additive avoids the
    over-optimization that product/gate/temper suffer" claim;
  * the over-optimization proxy aggregated from the per-rollout iter_curve
    (gate_frac, ldem_minus_lmod) for each rule.

numpy only.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parent / "results" / "combo_rules_2026_06_14")

CELL_RE = re.compile(
    r"(?P<task>can|square)_dp_ep(?P<ep>\d+)(?:_smax[0-9p]+)?"
    r"_lam(?P<lam>[0-9pm]+)_comb(?P<rule>add|prod|temp|gate)_seed(?P<seed>\d+)_n(?P<n>\d+)"
)
RULE_ORDER = ["add", "prod", "temp", "gate"]
RULE_NAME = {"add": "additive(score)", "prod": "product(p=0)",
             "temp": "temper(p=1)", "gate": "max-gate(p=inf)"}
RNG = np.random.default_rng(0)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot_ci(d, B=50000):
    if len(d) == 0:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, len(d), size=(B, len(d)))
    return tuple(np.percentile(d[idx].mean(axis=1), [2.5, 97.5]))


def sign_p(d):
    pos = int((d > 0).sum()); neg = int((d < 0).sum()); n = pos + neg
    if n == 0:
        return 1.0
    from math import comb
    k = max(pos, neg)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def proxy_from_rollout(path):
    """Mean gate_frac & ldem_minus_lmod over all iter_curve entries in a file."""
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    gf, lm = [], []
    for r in d.get("per_rollout", []):
        for ch in r.get("chunks", []):
            for it in ch.get("iter_curve", []):
                if "gate_frac" in it:
                    gf.append(it["gate_frac"])
                if "ldem_minus_lmod" in it:
                    lm.append(it["ldem_minus_lmod"])
    return (np.mean(gf) if gf else float("nan"),
            np.mean(lm) if lm else float("nan"))


def load():
    """task -> rule -> {seed:(k,n)} ; task->rule->[proxy gf], [proxy lm]."""
    succ = defaultdict(lambda: defaultdict(dict))
    prox = defaultdict(lambda: defaultdict(lambda: ([], [])))
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
        if not pl:
            continue
        e = pl[0]
        task, rule, seed = m["task"], m["rule"], m["seed"]
        succ[task][rule][seed] = (int(e.get("n_success", 0)), int(e.get("n_total", 0)))
        # proxy from the matching per-rollout file
        roll = sorted(f.parent.glob("unified_flow_pull_lam-*.json"))
        if roll:
            pr = proxy_from_rollout(roll[0])
            if pr:
                prox[task][rule][0].append(pr[0])
                prox[task][rule][1].append(pr[1])
    return succ, prox


def matched(succ_task, rule_a, rule_b):
    a, b = succ_task.get(rule_a, {}), succ_task.get(rule_b, {})
    seeds = sorted(set(a) & set(b))
    d = np.array([b[s][0] / b[s][1] - a[s][0] / a[s][1]
                  for s in seeds if a[s][1] and b[s][1]])
    return d


def main():
    succ, prox = load()
    if not succ:
        print(f"[combo] no cells yet under {ROOT}")
        return
    print(f"[combo] root={ROOT.name}\n")
    for task in sorted(succ):
        print("=" * 78)
        print(f"TASK = {task}  (ep0199, nominal, lambda=1, matched seeds)")
        print("=" * 78)
        print(f"  {'rule':<20}{'succ':>10}{'rate':>8}{'  Wilson95':>16}"
              f"{'gate_frac':>11}{'ldem-lmod':>11}")
        for rule in RULE_ORDER:
            if rule not in succ[task]:
                continue
            ks = sum(v[0] for v in succ[task][rule].values())
            ns = sum(v[1] for v in succ[task][rule].values())
            lo, hi = wilson(ks, ns)
            gf = np.nanmean(prox[task][rule][0]) if prox[task][rule][0] else float("nan")
            lm = np.nanmean(prox[task][rule][1]) if prox[task][rule][1] else float("nan")
            print(f"  {RULE_NAME[rule]:<20}{ks:>4}/{ns:<5}{ks/ns*100 if ns else 0:>7.1f}%"
                  f"  [{lo*100:>4.1f},{hi*100:>5.1f}]{gf:>11.3f}{lm:>11.3f}")
        print(f"\n  matched-seed paired lift of ADDITIVE over each multiplicative rule:")
        for rule in ("prod", "temp", "gate"):
            if rule not in succ[task]:
                continue
            d = matched(succ[task], rule, "add")  # add - rule
            if len(d) == 0:
                continue
            lo, hi = boot_ci(d)
            star = " ***" if (lo > 0 or hi < 0) else ""
            print(f"    additive - {RULE_NAME[rule]:<18} n={len(d):>2} "
                  f"Delta={d.mean()*100:>+5.1f}pp 95%CI[{lo*100:>+5.1f},{hi*100:>+5.1f}] "
                  f"sign-p={sign_p(d):.3f}{star}")
        print()


if __name__ == "__main__":
    main()
