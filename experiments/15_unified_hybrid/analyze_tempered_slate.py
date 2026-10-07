"""Aggregate + interpret the MBD gated/tempered slate results.

Reads the ``unified_*.json`` files written by ``run_can.py`` for the
``slate_mbd_gated_tempered_2026_06_04.sh`` slate and prints the three
analyses the slate is designed to answer (docs/theory/09 §6.1/§6.2/§6.3):

  * Section D -- sigma_demo calibration curve: where does the demo
    branch become competitive (gate_frac 0->1, ldem_minus_lmod crosses
    ~0)?  This must be read FIRST; it sets the sigma_demo for A/C.
  * Section A -- tempered-p Pareto: success vs p at the canonical point.
    Look for a finite p >= max(product p=0, max-gate p=inf).
  * Section C -- off-distribution product vs max-gate: does the gate
    rescue the cases where the product collapses?

Usage:
    python experiments/15_unified_hybrid/analyze_tempered_slate.py \
        [--results-dir DIR]

Pure-stdlib + numpy; safe to run while the slate is still in progress
(only reads completed JSONs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_DEFAULT_DIR = (
    Path(__file__).resolve().parent
    / "results" / "slate_mbd_gated_tempered_2026_06_04"
)


def _suffix_tag(stem: str) -> str:
    """Recover the --output-name-suffix tag from a unified_* filename."""
    # Filenames look like: unified_<prior>_lam-<L>_K-<K>_R-<R>_<TAG>
    if "_R-" in stem:
        after = stem.split("_R-", 1)[1]
        # after = "<R><TAG>" e.g. "1_A_temper_p2"; drop the leading R int
        parts = after.split("_", 1)
        return parts[1] if len(parts) > 1 else stem
    return stem


def _mean_diag(d: dict) -> tuple[float, float]:
    """Mean gate_frac and ldem_minus_lmod over all logged iters."""
    gfs, dls = [], []
    for r in d.get("per_rollout", []):
        for ch in r.get("chunks", []):
            for it in ch.get("iter_curve", []):
                gf, dl = it.get("gate_frac"), it.get("ldem_minus_lmod")
                if gf is not None and np.isfinite(gf):
                    gfs.append(gf)
                if dl is not None and np.isfinite(dl):
                    dls.append(dl)
    gf = float(np.mean(gfs)) if gfs else float("nan")
    dl = float(np.mean(dls)) if dls else float("nan")
    return gf, dl


def _load(results_dir: Path) -> list[dict]:
    rows = []
    for js in sorted(results_dir.glob("unified_*.json")):
        if "summary" in js.name:
            continue
        try:
            d = json.load(open(js))
        except (json.JSONDecodeError, OSError):
            continue  # partial file from an in-flight run; skip
        p = d.get("params", {})
        gf, dl = _mean_diag(d)
        rwd = [r.get("final_reward") for r in d.get("per_rollout", [])
               if r.get("final_reward") is not None]
        rows.append({
            "tag": _suffix_tag(js.stem),
            "reward": float(np.mean(rwd)) if rwd else float("nan"),
            "task": p.get("task", "?"),
            "raw_sparse": bool(p.get("raw_sparse_reward", False)),
            "n": d.get("n_total", 0),
            "succ": d.get("n_success", 0),
            "sr": (d.get("n_success", 0) / d["n_total"] * 100)
            if d.get("n_total") else 0.0,
            "combine_mode": p.get("combine_mode", "?"),
            "temper_p": p.get("temper_p", float("nan")),
            "sigma_demo": p.get("sigma_demo", float("nan")),
            "mass": float(d.get("mass_scale", p.get("mass_scale", 1.0))),
            "fric": float(p.get("friction_scale", 1.0)),
            "gate_frac": gf,
            "ldem_minus_lmod": dl,
        })
    return rows


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "inf"
    return f"{p:g}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=_DEFAULT_DIR)
    args = ap.parse_args()

    if not args.results_dir.exists():
        print(f"[analyze] results dir not found: {args.results_dir}")
        return 1
    rows = _load(args.results_dir)
    if not rows:
        print(f"[analyze] no completed unified_*.json yet in "
              f"{args.results_dir}")
        return 0

    print(f"\n=== MBD gated/tempered slate -- {len(rows)} completed cells ===")
    print(f"    {args.results_dir}\n")

    # ---- Section D: sigma_demo calibration ----
    dsec = sorted(
        [r for r in rows if r["tag"].startswith("D_")],
        key=lambda r: r["sigma_demo"],
    )
    if dsec:
        print("Section D -- sigma_demo calibration (max-gate, canonical)")
        print(f"  {'sigma_demo':>11}{'gate_frac':>11}{'ldem-lmod':>12}"
              f"{'succ %':>9}")
        print("  " + "-" * 41)
        for r in dsec:
            print(f"  {r['sigma_demo']:>11g}{r['gate_frac']:>11.3f}"
                  f"{r['ldem_minus_lmod']:>+12.2f}{r['sr']:>8.1f}%")
        print("  -> calibrated sigma_demo ~ where ldem-lmod crosses 0 / "
              "gate_frac leaves 0.\n")

    # ---- Section A: tempered-p Pareto (canonical) ----
    asec = sorted(
        [r for r in rows if r["tag"].startswith(("A_", "A2_"))],
        key=lambda r: (r["sigma_demo"], r["combine_mode"], r["temper_p"]),
    )
    if asec:
        print("Section A -- tempered-p rule selection (canonical Can)")
        print(f"  {'tag':<24}{'mode':>9}{'sig_d':>7}{'p':>6}{'succ %':>9}"
              f"{'gate_frac':>11}")
        print("  " + "-" * 66)
        best = max(asec, key=lambda r: r["sr"])
        for r in asec:
            star = "  <- best" if r is best else ""
            print(f"  {r['tag']:<24}{r['combine_mode']:>9}"
                  f"{r['sigma_demo']:>7g}"
                  f"{_fmt_p(r['temper_p']):>6}{r['sr']:>8.1f}%"
                  f"{r['gate_frac']:>11.3f}{star}")
        print("  -> a FINITE p beating both p=0 (product) and p=inf "
              "(max-gate) is the headline result.\n")

    # ---- Section B: sigma-mode ablation ----
    bsec = [r for r in rows if r["tag"].startswith("B_")]
    if bsec:
        print("Section B -- sigma-mode ablation (#4, legacy score path)")
        print(f"  {'tag':<24}{'succ %':>9}")
        print("  " + "-" * 33)
        for r in sorted(bsec, key=lambda r: r["tag"]):
            print(f"  {r['tag']:<24}{r['sr']:>8.1f}%")
        print()

    # ---- Section C: off-distribution product vs max-gate ----
    csec = [r for r in rows if r["tag"].startswith(("C_", "C2_"))]
    if csec:
        print("Section C -- off-distribution product vs max-gate (#8/§6.1)")
        print(f"  {'tag':<24}{'mass':>6}{'fric':>6}{'p':>6}{'succ %':>9}"
              f"{'gate_frac':>11}")
        print("  " + "-" * 62)
        for r in sorted(csec, key=lambda r: r["tag"]):
            print(f"  {r['tag']:<24}{r['mass']:>6g}{r['fric']:>6g}"
                  f"{_fmt_p(r['temper_p']):>6}{r['sr']:>8.1f}%"
                  f"{r['gate_frac']:>11.3f}")
        print("  -> the §5 prediction: product collapses, max-gate holds; "
              "gate_frac->1 where the prior is off-distribution.\n")

    # ---- #6 demo-quality gate-ceiling (DQ_<band>_{gate,product}) ----
    dqsec = [r for r in rows if r["tag"].startswith("DQ_")]
    if dqsec:
        byband: dict[str, dict[str, float]] = {}
        for r in dqsec:
            # tag = DQ_<band>_<gate|product>
            parts = r["tag"].split("_")
            if len(parts) >= 3:
                band, arm = parts[1], parts[2]
                byband.setdefault(band, {})[arm] = r["sr"]
        print("Section DQ -- demo-quality gate-ceiling (#6, §4.5b-2)")
        print(f"  {'band':<10}{'gate %':>9}{'product %':>11}{'gap':>8}")
        print("  " + "-" * 38)
        for band in ("optimal", "mid", "worst"):
            if band not in byband:
                continue
            g = byband[band].get("gate"); p = byband[band].get("product")
            gs = f"{g:.1f}" if g is not None else "--"
            ps = f"{p:.1f}" if p is not None else "--"
            gap = f"{p - g:+.1f}" if (g is not None and p is not None) else "--"
            print(f"  {band:<10}{gs:>9}{ps:>11}{gap:>8}")
        print("  -> #6 prediction: product >= gate every band, gap WIDENS "
              "with cost.\n")

    # ---- Section E: lambda x p operating surface (E_lam<L>_p<P>) ----
    esec = [r for r in rows if r["tag"].startswith("E_")]
    if esec:
        print("Section E -- lambda x p operating surface (canonical Can)")
        print(f"  {'tag':<20}{'lambda':>8}{'p':>6}{'succ %':>9}"
              f"{'gate_frac':>11}")
        print("  " + "-" * 54)
        best = max(esec, key=lambda r: r["sr"])
        for r in sorted(esec, key=lambda r: r["tag"]):
            lam = r["tag"].split("_lam")[-1].split("_")[0].replace("p", ".")
            star = "  <- best" if r is best else ""
            print(f"  {r['tag']:<20}{lam:>8}{_fmt_p(r['temper_p']):>6}"
                  f"{r['sr']:>8.1f}%{r['gate_frac']:>11.3f}{star}")
        print("  -> an INTERIOR best (finite p, moderate lambda) is the "
              "strongest 'tuned product beats both endpoints' claim.\n")

    # ---- Section G: cross-task generality (G_<task>_<cell>) ----
    gsec = [r for r in rows if r["tag"].startswith("G_")]
    if gsec:
        print("Section G -- cross-task generality (Lift / Square)")
        print(f"  {'tag':<22}{'task':>8}{'sig_d':>7}{'p':>6}{'succ %':>9}"
              f"{'gate_frac':>11}{'ldem-lmod':>11}")
        print("  " + "-" * 74)
        for r in sorted(gsec, key=lambda r: r["tag"]):
            t = r["task"] if r["task"] != "?" else r["tag"].split("_")[1]
            print(f"  {r['tag']:<22}{t:>8}{r['sigma_demo']:>7g}"
                  f"{_fmt_p(r['temper_p']):>6}{r['sr']:>8.1f}%"
                  f"{r['gate_frac']:>11.3f}{r['ldem_minus_lmod']:>+11.2f}")
        print("  -> Dg* cells calibrate sigma_demo per task (ldem-lmod~0); "
              "the Can product<->gate finding should transfer.\n")

    # ---- Section S: sparse-reward revival (S_<raw|dense>_<arm>) ----
    ssec = [r for r in rows if r["tag"].startswith("S_")]
    if ssec:
        print("Section S -- sparse-reward revival (§7.1; mean REWARD matters)")
        print(f"  {'tag':<20}{'cost':>7}{'arm':>9}{'succ %':>9}"
              f"{'mean_rwd':>10}{'gate_frac':>11}")
        print("  " + "-" * 66)
        for r in sorted(ssec, key=lambda r: r["tag"]):
            parts = r["tag"].split("_")
            cost = parts[1] if len(parts) > 1 else "?"
            arm = parts[2] if len(parts) > 2 else "?"
            print(f"  {r['tag']:<20}{cost:>7}{arm:>9}{r['sr']:>8.1f}%"
                  f"{r['reward']:>10.3f}{r['gate_frac']:>11.3f}")
        print("  -> prediction: raw -> product~gate~score (flat J); dense -> "
              "product/score reward climbs > max-gate (gate is threshold-lagged).\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
