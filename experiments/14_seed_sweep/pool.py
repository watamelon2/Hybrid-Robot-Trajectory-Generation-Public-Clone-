"""Pool seed-sweep JSONs and report Wilson 95% CIs.

Reads every ``exp{04,05,05b}_seed*.json`` under ``--in`` and emits
a per-method row with absolute counts, success rate, and the
Wilson confidence interval. Pure stdlib + math; runs in
milliseconds.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson 95% confidence interval for a binomial."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(centre - half, 0.0), min(centre + half, 1.0))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    pat = re.compile(r"exp(0?[45]b?)_seed(\d+)\.json$")
    pools: dict[str, dict] = defaultdict(
        lambda: {"k": 0, "n": 0, "seeds": [], "files": []}
    )
    for f in sorted(args.in_dir.iterdir()):
        m = pat.match(f.name)
        if not m:
            continue
        method = m.group(1)
        blob = json.loads(f.read_text())
        pools[method]["k"] += int(blob.get("n_success", 0))
        pools[method]["n"] += int(blob.get("n_total", 0))
        pools[method]["seeds"].append(int(m.group(2)))
        pools[method]["files"].append(str(f))

    rows = []
    for method, agg in sorted(pools.items()):
        lo, hi = wilson_ci(agg["k"], agg["n"])
        rows.append({
            "method": f"exp{method}",
            "n_success": agg["k"],
            "n_total": agg["n"],
            "success_rate": (agg["k"] / agg["n"]) if agg["n"] else 0.0,
            "wilson_95_lo": lo,
            "wilson_95_hi": hi,
            "seeds": agg["seeds"],
            "files": agg["files"],
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))

    print("| method | succ | n | rate | 95% Wilson |")
    print("| ------ | ---: | -:| ----:| ---------- |")
    for r in rows:
        print(
            f"| {r['method']} | {r['n_success']} | {r['n_total']} | "
            f"{r['success_rate']:.1%} | "
            f"[{r['wilson_95_lo']:.1%}, {r['wilson_95_hi']:.1%}] |"
        )
    print(f"\n[exp14] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
