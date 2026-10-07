from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .certifier import DryRunRobosuiteCertifier
from .level1_contract import run_level1_trial
from .planner_adapter import MJXPlannerConfig, make_planner
from .task_scaffolds import iter_task_scaffolds
from .task_registry import get_task_spec, iter_task_specs
from .validation import default_validation_gates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run the MJX Level-1 scaffold without MJX or robosuite.")
    parser.add_argument("--task", default="can_level1")
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--list-scaffolds", action="store_true")
    parser.add_argument("--list-gates", action="store_true")
    return parser.parse_args()


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def main() -> int:
    args = parse_args()
    if args.list_tasks:
        rows = [spec.__dict__ for spec in iter_task_specs()]
        print(json.dumps(rows, indent=2, default=_json_default))
        return 0
    if args.list_gates:
        rows = [gate.__dict__ for gate in default_validation_gates()]
        print(json.dumps(rows, indent=2, default=_json_default))
        return 0
    if args.list_scaffolds:
        rows = [item.__dict__ for item in iter_task_scaffolds()]
        print(json.dumps(rows, indent=2, default=_json_default))
        return 0
    spec = get_task_spec(args.task)
    action_dim = int(spec.action_dim if args.action_dim is None else args.action_dim)
    planner = make_planner(
        MJXPlannerConfig(
            task_name=spec.name,
            horizon=int(args.H),
            action_dim=action_dim,
            scene_path=spec.scene_path,
            dry_run=True,
        )
    )
    certifier = DryRunRobosuiteCertifier(task_name=spec.name, action_dim=action_dim)
    result = run_level1_trial(
        planner=planner,
        certifier=certifier,
        task=spec.name,
        init_reference=np.zeros(1, dtype=np.float32),
        obs_reference=np.zeros(1, dtype=np.float32),
        prior_context={},
    )
    result["task_phase"] = int(spec.phase)
    result["validation_gate"] = spec.validation_gate
    result["implementation_status"] = spec.implementation_status
    text = json.dumps(result, indent=2, default=_json_default)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
