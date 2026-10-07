"""Post-hoc failure taxonomy over experiments 04 / 05 / 05b JSONs.

The Exp 05 README narrates the failure modes informally
("no reach" / "partial reach" / "strong reach" / "success"). This
script promotes that narration to a column so every later experiment
can just point at the same buckets.

Buckets are defined purely from the per-rollout
``final_exec_reward`` and ``success`` fields that already live in the
results JSONs, so this is a *no-env*, *no-MuJoCo* analysis -- it
runs in <1 s on a laptop and depends only on the standard library.

Bucket boundaries were chosen to match the published Week-4 thresholds
(see Exp 05 README, section "Behaviour breakdown"):

    success       : any rollout with success=True
    no_reach      : exec_reward <= +0.5
    partial_reach : +0.5 <  exec_reward <= +3.0    and not success
    strong_reach  : +3.0 <  exec_reward             and not success

Usage:

    python experiments/06_failure_taxonomy/run.py \
        --inputs experiments/04_mbd_baseline/results/_archive/can_ph.json \
                 experiments/05_hybrid_method/results/_archive/can_ph.json \
                 experiments/05_hybrid_method/results/_archive/can_ph_mpc.json \
        --output experiments/06_failure_taxonomy/results/taxonomy.json

The output is one row per input JSON with absolute counts and shares
plus a Markdown table (printed to stdout and written next to the
JSON) suitable for paste into a README.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NO_REACH_MAX = 0.5
PARTIAL_MAX = 3.0


def classify(reward: float, success: bool) -> str:
    if success:
        return "success"
    if reward <= NO_REACH_MAX:
        return "no_reach"
    if reward <= PARTIAL_MAX:
        return "partial_reach"
    return "strong_reach"


def summarise(path: Path) -> dict:
    blob = json.loads(path.read_text())
    rollouts = blob.get("per_rollout", [])
    n = len(rollouts)
    counts = {"success": 0, "strong_reach": 0, "partial_reach": 0, "no_reach": 0}
    bc_success = 0
    mbd_beats_bc = 0
    n_bc = 0
    for r in rollouts:
        bucket = classify(
            float(r.get("final_exec_reward", 0.0)),
            bool(r.get("success", False)),
        )
        counts[bucket] += 1
        bc_r = r.get("bc_exec_reward")
        if bc_r is not None:
            n_bc += 1
            if float(r["final_exec_reward"]) > float(bc_r):
                mbd_beats_bc += 1
            if r.get("bc_success"):
                bc_success += 1

    # Strong-reach -> success conversion (interpretable mechanism stat
    # used in Exp 05b README). Numerator: successes; denominator: any
    # rollout that *would* have been classified strong_reach if it
    # had failed, plus the actual successes (since we don't have the
    # pre-success reward distribution, we approximate with successes
    # + strong_reach).
    strong_reach_pool = counts["success"] + counts["strong_reach"]
    sr_to_success = (
        counts["success"] / strong_reach_pool
        if strong_reach_pool
        else float("nan")
    )

    return {
        "input": str(path),
        "experiment": blob.get("experiment"),
        "params": blob.get("params"),
        "n_total": n,
        "n_success": counts["success"],
        "success_rate": counts["success"] / n if n else 0.0,
        "counts": counts,
        "shares": {
            k: (v / n if n else 0.0) for k, v in counts.items()
        },
        "bc_success": bc_success if n_bc else None,
        "bc_success_rate": (bc_success / n_bc) if n_bc else None,
        "mbd_beats_bc_count": mbd_beats_bc if n_bc else None,
        "mbd_beats_bc_share": (mbd_beats_bc / n_bc) if n_bc else None,
        "strong_reach_to_success_share": sr_to_success,
    }


def render_markdown(rows: list[dict]) -> str:
    header = (
        "| run | n | succ | strong | partial | no-reach | "
        "MBD>BC | SR->succ |\n"
        "| --- | -:| -----:| -----:| -------:| ------:| -----:| -------:|\n"
    )
    body = []
    for r in rows:
        c = r["counts"]
        body.append(
            f"| {Path(r['input']).parts[-2]}/{Path(r['input']).name} | "
            f"{r['n_total']} | "
            f"{c['success']} ({r['success_rate']:.1%}) | "
            f"{c['strong_reach']} ({r['shares']['strong_reach']:.1%}) | "
            f"{c['partial_reach']} ({r['shares']['partial_reach']:.1%}) | "
            f"{c['no_reach']} ({r['shares']['no_reach']:.1%}) | "
            f"{(r['mbd_beats_bc_share'] or 0):.0%} | "
            f"{(r['strong_reach_to_success_share'] or 0):.1%} |"
        )
    return header + "\n".join(body) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_inputs = [
        Path("experiments/04_mbd_baseline/results/_archive/can_ph.json"),
        Path("experiments/05_hybrid_method/results/_archive/can_ph.json"),
        Path("experiments/05_hybrid_method/results/_archive/can_ph_mpc.json"),
    ]
    p.add_argument("--inputs", type=Path, nargs="+", default=default_inputs)
    p.add_argument(
        "--output", type=Path,
        default=Path("experiments/06_failure_taxonomy/results/taxonomy.json"),
    )
    args = p.parse_args()

    rows = []
    for path in args.inputs:
        if not path.exists():
            print(f"[exp06] skipping missing input: {path}")
            continue
        rows.append(summarise(path))

    if not rows:
        raise SystemExit("[exp06] no input JSONs found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))

    md = render_markdown(rows)
    md_path = args.output.with_suffix(".md")
    md_path.write_text(md)

    print(md, end="")
    print(f"\n[exp06] wrote {args.output}\n[exp06] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
