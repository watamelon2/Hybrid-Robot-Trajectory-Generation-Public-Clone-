from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _vec(obs: dict[str, Any], key: str) -> list[float] | None:
    value = obs.get(key)
    if value is None:
        return None
    return [float(x) for x in value]


def _dist(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None or len(a) != len(b):
        return None
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _min_dist(trace: list[dict[str, Any]], a_key: str, b_key: str) -> float | None:
    values = []
    for step in trace:
        obs = step.get("obs", {})
        d = _dist(_vec(obs, a_key), _vec(obs, b_key))
        if d is not None:
            values.append(d)
    return min(values) if values else None


def _max_z(trace: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for step in trace:
        pos = _vec(step.get("obs", {}), key)
        if pos is not None and len(pos) >= 3:
            values.append(pos[2])
    return max(values) if values else None


def _flatten_step_trace(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for chunk in rollout.get("chunks", []):
        out.extend(chunk.get("step_trace", []))
    return out


def classify_square_rollout(rollout: dict[str, Any]) -> dict[str, Any]:
    trace = _flatten_step_trace(rollout)
    min_nut_eef = _min_dist(trace, "SquareNut_pos", "robot0_eef_pos")
    min_rel_eef = None
    rel_values = []
    for step in trace:
        rel = _vec(step.get("obs", {}), "SquareNut_to_eef_pos")
        if rel is not None:
            rel_values.append(math.sqrt(sum(x * x for x in rel)))
    if rel_values:
        min_rel_eef = min(rel_values)
    approach_dist = min_rel_eef if min_rel_eef is not None else min_nut_eef
    max_nut_z = _max_z(trace, "SquareNut_pos")
    final_reward = float(rollout.get("final_reward", 0.0))

    if bool(rollout.get("success", False)):
        phase = "success"
    elif not trace:
        phase = "missing_trace"
    elif approach_dist is None:
        phase = "unclassified_missing_obs"
    elif approach_dist > 0.08:
        phase = "approach"
    elif max_nut_z is not None and max_nut_z < 0.84:
        phase = "grasp_or_lift"
    elif final_reward > 100.0:
        phase = "align_or_insert"
    else:
        phase = "post_contact_unclear"

    return {
        "demo": rollout.get("demo"),
        "rollout_idx": rollout.get("rollout_idx"),
        "rollout_kind": rollout.get("rollout_kind"),
        "init_seed": rollout.get("init_seed"),
        "success": bool(rollout.get("success", False)),
        "phase": phase,
        "final_reward": final_reward,
        "n_steps_traced": len(trace),
        "min_square_nut_to_eef_m": approach_dist,
        "max_square_nut_z_m": max_nut_z,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.json_path.read_text())
    rows = [classify_square_rollout(r) for r in data.get("per_rollout", [])]
    counts = Counter(r["phase"] for r in rows)
    payload = {
        "source": str(args.json_path),
        "n_rollouts": len(rows),
        "phase_counts": dict(sorted(counts.items())),
        "rows": rows,
    }

    text = json.dumps(payload, indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
