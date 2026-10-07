from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "slate_bank_size_scaling_2026_06_07"
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "docs" / "results_writeups" / "bank_size_saturation.pdf"


def wilson_ci(succ: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = succ / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return centre - half, centre + half


def load_points(results_dir: Path) -> list[tuple[int, int, int]]:
    points: list[tuple[int, int, int]] = []
    for path in sorted(results_dir.glob("unified_demo_1nn_*bank_size_*.json")):
        match = re.search(r"bank_size_(\d+)\.json$", path.name)
        if match is None:
            continue
        data = json.loads(path.read_text())
        bank_size = int(match.group(1))
        points.append((bank_size, int(data["n_success"]), int(data["n_total"])))
    return sorted(points)


def plot(points: list[tuple[int, int, int]], out: Path) -> None:
    if not points:
        raise ValueError("no bank-size result JSONs found")
    banks = np.asarray([p[0] for p in points], dtype=float)
    rates = np.asarray([p[1] / p[2] for p in points], dtype=float)
    lo = []
    hi = []
    for _, succ, n in points:
        l, h = wilson_ci(succ, n)
        r = succ / n
        lo.append(max(0.0, r - l))
        hi.append(max(0.0, h - r))

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.errorbar(banks, rates, yerr=[lo, hi], fmt="o-", color="#1a7f3c", lw=2.2, capsize=4)
    ax.set_xscale("log")
    ax.set_xticks(banks)
    ax.set_xticklabels([str(int(b)) for b in banks])
    ax.set_xlabel("retrieval bank size")
    ax.set_ylabel("success rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0.0, 0.85)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.5)
    ax.axvspan(20, 200, color="#1a7f3c", alpha=0.07)
    ax.text(22, 0.08, "coverage knee\nthen saturation", fontsize=8.5, color="#1a7f3c")
    ax.set_title("Bank-size saturation supports finite mode coverage", fontsize=10.5)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot(load_points(args.results_dir), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
