#!/usr/bin/env python3
"""Predictive guidance decision-rule from the existing nwfix grid (ZERO new compute).

Turns the descriptive calibration map into a PREDICTIVE rule: from a single cheap,
a-priori-measurable proxy -- the *standalone* (lambda0) success rate, which you can
measure before deciding whether to guide -- predict (a) whether test-time guidance
will materially help, and (b) the lift it will buy.  Validated leave-one-CONDITION-out
so the accuracy is honest (each held-out curve is predicted by a rule fit on the rest).

A "condition" = one (task, ep, smax, shift) curve; we pair its lambda0 cell against its
guided cell (lambda1 for the canonical operating point) over the pooled grid.  Reuses
analyze_nwfix_grid_2026_06_13's loader conventions.  numpy only.
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


def unp(s):
    return s.replace("p", ".") if s else "-"


def load():
    """key=(task,ep,smax,mass,fric,lam) -> [k,n]; skips comb* (own root anyway)."""
    agg = defaultdict(lambda: [0, 0])
    for f in ROOT.rglob("unified_summary_*.json"):
        name = f.parent.name
        if "comb" in name:
            continue
        m = CELL_RE.search(name)
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
            key = (m["task"], m["ep"], unp(m["smax"]), unp(m["mass"]), unp(m["fric"]), lam)
            a = agg[key]
            a[0] += int(e.get("n_success", 0))
            a[1] += int(e.get("n_total", 0))
    return agg


def shift_mag(mass, fric):
    """A-priori shift magnitude: max |log| deviation from nominal (0 = nominal)."""
    m = 1.0 if mass == "-" else float(mass)
    f = 1.0 if fric == "-" else float(fric)
    return max(abs(np.log(m)), abs(np.log(f)))


def build_conditions(agg):
    """One row per (task,ep,smax,shift): standalone lam0 vs guided lam1."""
    rows = []
    cond = defaultdict(dict)  # (task,ep,smax,mass,fric) -> {lam:(k,n)}
    for (task, ep, smax, mass, fric, lam), (k, n) in agg.items():
        cond[(task, ep, smax, mass, fric)][lam] = (k, n)
    for (task, ep, smax, mass, fric), lams in cond.items():
        if 0.0 not in lams or 1.0 not in lams:
            continue
        k0, n0 = lams[0.0]
        k1, n1 = lams[1.0]
        if n0 < 12 or n1 < 12:   # need a minimally-pooled estimate
            continue
        p0, p1 = k0 / n0, k1 / n1
        # empirical argmax-lambda among available lambdas (shape claim)
        best_lam = max(lams, key=lambda L: (lams[L][0] / lams[L][1]) if lams[L][1] else 0)
        rows.append(dict(
            task=task, ep=ep, smax=smax, mass=mass, fric=fric,
            shift=shift_mag(mass, fric), p0=p0, p1=p1, lift=p1 - p0,
            n0=n0, n1=n1, lam_star=best_lam,
        ))
    return rows


def loo_linear(x, y):
    """Leave-one-out predictions for y ~ 1 + x (closed-form refit each holdout)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    preds = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        A = np.c_[np.ones(m.sum()), x[m]]
        beta, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        preds[i] = beta[0] + beta[1] * x[i]
    return preds


def loo_threshold_help(p0, lift, help_thresh=0.05):
    """1-D leave-one-out: predict 'guidance materially helps' (lift>=thresh) from p0.

    Rule learned on train: helps iff p0 in [lo, hi] (the band where lift>=thresh).
    """
    p0 = np.asarray(p0, float)
    helps = np.asarray(lift, float) >= help_thresh
    n = len(p0)
    correct = 0
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        pos = p0[m][helps[m]]
        if len(pos) == 0:
            pred = False
        else:
            lo, hi = pos.min(), pos.max()
            pred = (p0[i] >= lo) and (p0[i] <= hi)
        correct += int(pred == helps[i])
    return correct / n, helps


def main():
    agg = load()
    rows = build_conditions(agg)
    rows.sort(key=lambda r: (r["task"], r["ep"], r["smax"], r["shift"]))
    print(f"[predictive] {len(rows)} conditions (task,ep,smax,shift) with lam0 & lam1, n>=12\n")

    print("=" * 86)
    print(f"{'condition':<40}{'lam0%':>7}{'lam1%':>7}{'lift_pp':>9}{'shift':>7}{'lam*':>7}")
    print("=" * 86)
    for r in rows:
        sh = "" if (r["mass"] == "-" and r["fric"] == "-") else f" m{r['mass']} f{r['fric']}"
        tag = f"{r['task']} ep{r['ep']}" + (f" s{r['smax']}" if r['smax'] != '-' else "") + sh
        print(f"{tag:<40}{r['p0']*100:>7.1f}{r['p1']*100:>7.1f}{r['lift']*100:>+9.1f}"
              f"{r['shift']:>7.2f}{r['lam_star']:>7g}")

    p0 = [r["p0"] for r in rows]
    lift = [r["lift"] for r in rows]
    shift = [r["shift"] for r in rows]

    print("\n" + "-" * 86)
    print("RULE 1 (regression): lift ~ standalone success p0   [leave-one-condition-out]")
    print("-" * 86)
    pred = loo_linear(p0, lift)
    mae = np.mean(np.abs(pred - np.array(lift))) * 100
    sign_acc = np.mean((pred > 0) == (np.array(lift) > 0))
    r_pear = np.corrcoef(p0, lift)[0, 1]
    print(f"  Pearson r(p0, lift) = {r_pear:+.2f}   LOO sign-accuracy = {sign_acc*100:.0f}%"
          f"   LOO MAE(lift) = {mae:.1f} pp")

    print("\n" + "-" * 86)
    print("RULE 2 (decision): 'guidance materially helps' (lift>=+5pp) from p0 band")
    print("-" * 86)
    acc, helps = loo_threshold_help(p0, lift)
    band = [r["p0"] for r, h in zip(rows, helps) if h]
    print(f"  helps in {sum(helps)}/{len(rows)} conditions; learned p0-band ~"
          f" [{min(band)*100:.0f}%, {max(band)*100:.0f}%]   LOO accuracy = {acc*100:.0f}%")
    futile = [(r['task'], r['mass'], r['fric'], round(r['p0'] * 100)) for r, h in zip(rows, helps) if not h]
    print(f"  guidance-futile conditions (lift<+5pp): {futile}")

    print("\n" + "-" * 86)
    print("RULE 3 (shape): empirical lam* across SOLVABLE conditions (p0>=15%)")
    print("-" * 86)
    solv = [r for r in rows if r["p0"] >= 0.15]
    interior = [r for r in solv if 0.5 <= r["lam_star"] <= 2.0]
    print(f"  lam* in [0.5, 2] for {len(interior)}/{len(solv)} solvable conditions"
          f"  (lam* values: {sorted({r['lam_star'] for r in solv})})")


if __name__ == "__main__":
    main()
