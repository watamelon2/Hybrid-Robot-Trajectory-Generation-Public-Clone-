"""Pick the calibrated ``sigma_demo`` from a completed Section-D sweep.

The Section-D cells of ``slate_mbd_gated_tempered`` run the max-gate at a
range of ``sigma_demo`` and log the per-iter likelihood-ratio diagnostic
(``ldem_minus_lmod`` = mean ``ℓ_dem − ℓ_mod``).  The two branches are
"properly scaled" (MBD §4.3) where that crosses ~0 -- below it the demo
Gaussian's ``‖τ−Y_demo‖²/(2σ²)`` term dwarfs the cost term and the gate
dies (``gate_frac→0``); above it the demo branch dominates.

This script reads those JSONs and prints the calibrated ``sigma_demo``
(the candidate with the smallest ``|ldem_minus_lmod|`` among those whose
mean ``gate_frac`` is in a healthy band) to stdout, so a slate can do::

    SIGMA=$(python pick_calibrated_sigma.py --results-dir DIR)

Prints ONLY the number on stdout (diagnostics go to stderr) so it is
safe to capture in a shell variable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Reuse the slate loader (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_tempered_slate import _DEFAULT_DIR, _load  # noqa: E402


def pick(rows: list[dict], gate_lo: float, gate_hi: float,
         fallback: float) -> float:
    """Calibrated sigma_demo = argmin |ldem-lmod| over healthy-gate cells."""
    dsec = [r for r in rows if r["tag"].startswith("D_")
            and r["sigma_demo"] == r["sigma_demo"]]  # drop NaN sigma
    if not dsec:
        print(f"[pick] no Section-D cells; falling back to {fallback}",
              file=sys.stderr)
        return fallback
    # Healthy gate band: gate is alive but not saturated on the demo.
    healthy = [r for r in dsec
               if gate_lo <= r["gate_frac"] <= gate_hi
               and r["ldem_minus_lmod"] == r["ldem_minus_lmod"]]
    pool = healthy or [r for r in dsec
                       if r["ldem_minus_lmod"] == r["ldem_minus_lmod"]]
    if not pool:
        print(f"[pick] no finite diagnostics; falling back to {fallback}",
              file=sys.stderr)
        return fallback
    # Smallest |ldem-lmod|; tie-break toward the smaller sigma (tighter
    # demo pull, closer to the canonical 0.25 operating point).
    best = min(pool, key=lambda r: (abs(r["ldem_minus_lmod"]),
                                    r["sigma_demo"]))
    for r in sorted(dsec, key=lambda r: r["sigma_demo"]):
        mark = "  <-- picked" if r is best else ""
        print(f"[pick] sigma_demo={r['sigma_demo']:>6g}  "
              f"gate_frac={r['gate_frac']:.3f}  "
              f"ldem-lmod={r['ldem_minus_lmod']:+.2f}{mark}", file=sys.stderr)
    return float(best["sigma_demo"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, default=_DEFAULT_DIR)
    ap.add_argument("--gate-lo", type=float, default=0.2,
                    help="Min healthy mean gate_frac (default 0.2).")
    ap.add_argument("--gate-hi", type=float, default=0.65,
                    help="Max healthy mean gate_frac (default 0.65).")
    ap.add_argument("--fallback", type=float, default=1.0,
                    help="Value to print if Section D is unreadable.")
    args = ap.parse_args()
    rows = _load(args.results_dir) if args.results_dir.exists() else []
    sigma = pick(rows, args.gate_lo, args.gate_hi, args.fallback)
    print(f"{sigma:g}")  # stdout: the number only
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
