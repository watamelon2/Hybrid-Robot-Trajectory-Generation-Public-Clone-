from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .level1_contract import MJXPlanCandidate
from .planner_adapter import MJXPlannerConfig, make_planner
from .task_registry import get_task_spec


@dataclass(frozen=True)
class Exp15MJXSeamConfig:
    task_name: str = "can_level1"
    dry_run: bool = True
    horizon: int = 120
    action_dim: int = 7
    output_dir: Path | None = None


@dataclass(frozen=True)
class Exp15MJXPlanResult:
    candidate: MJXPlanCandidate
    backend_name: str
    task_name: str
    requires_robosuite_certification: bool


def plan_candidate_for_exp15(
    *,
    config: Exp15MJXSeamConfig,
    init_state: Any,
    obs_vector: np.ndarray,
    initial_mean: np.ndarray | None,
    prior_context: dict[str, Any] | None = None,
) -> Exp15MJXPlanResult:
    spec = get_task_spec(config.task_name)
    context = dict(prior_context or {})
    if initial_mean is not None:
        context["initial_mean"] = np.asarray(initial_mean, dtype=np.float32)
    planner = make_planner(
        MJXPlannerConfig(
            task_name=spec.name,
            horizon=int(config.horizon),
            action_dim=int(config.action_dim),
            scene_path=spec.scene_path,
            dry_run=bool(config.dry_run),
        )
    )
    candidate = planner.plan(
        task=spec.name,
        init_reference=init_state,
        obs_reference=np.asarray(obs_vector, dtype=np.float32),
        prior_context=context,
    )
    return Exp15MJXPlanResult(
        candidate=candidate,
        backend_name="mjx_level1_dry_run" if config.dry_run else "mjx_level1",
        task_name=spec.name,
        requires_robosuite_certification=True,
    )
