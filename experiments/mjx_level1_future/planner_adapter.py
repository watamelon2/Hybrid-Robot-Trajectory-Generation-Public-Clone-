from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .level1_contract import MJXPlanCandidate, MJXPlannerBackend
from .obs_bridge import ObservationBridge, PlaceholderObservationBridge
from .state_bridge import PlaceholderStateBridge, StateBridge
from .task_registry import MJXTaskSpec, get_task_spec


@dataclass(frozen=True)
class MJXPlannerConfig:
    task_name: str = "can_level1"
    horizon: int = 120
    action_dim: int = 7
    scene_path: Path | None = None
    n_substeps: int = 5
    dry_run: bool = True


class DryRunMJXPlanner(MJXPlannerBackend):
    def __init__(self, config: MJXPlannerConfig | None = None) -> None:
        self.config = config or MJXPlannerConfig()
        self.task = get_task_spec(self.config.task_name)

    def plan(
        self,
        *,
        task: str,
        init_reference: Any,
        obs_reference: np.ndarray,
        prior_context: dict[str, Any],
    ) -> MJXPlanCandidate:
        del init_reference, obs_reference
        if task != self.task.name and task != self.task.family:
            raise ValueError(f"planner configured for {self.task.name!r}, got task={task!r}")
        initial_mean = prior_context.get("initial_mean")
        if initial_mean is None:
            actions = np.zeros((self.config.horizon, self.config.action_dim), dtype=np.float32)
        else:
            actions = np.asarray(initial_mean, dtype=np.float32)
        return MJXPlanCandidate(
            actions=actions,
            planner_cost=0.0,
            planner_metadata={
                "mode": "dry_run",
                "task_name": self.task.name,
                "phase": self.task.phase,
                "validation_gate": self.task.validation_gate,
            },
        )


class Level1MJXPlannerAdapter(MJXPlannerBackend):
    def __init__(
        self,
        *,
        config: MJXPlannerConfig,
        state_bridge: StateBridge | None = None,
        obs_bridge: ObservationBridge | None = None,
    ) -> None:
        self.config = config
        self.task: MJXTaskSpec = get_task_spec(config.task_name)
        self.state_bridge = state_bridge or PlaceholderStateBridge()
        self.obs_bridge = obs_bridge or PlaceholderObservationBridge()

    def plan(
        self,
        *,
        task: str,
        init_reference: Any,
        obs_reference: np.ndarray,
        prior_context: dict[str, Any],
    ) -> MJXPlanCandidate:
        del init_reference, obs_reference, prior_context
        if task != self.task.name and task != self.task.family:
            raise ValueError(f"planner configured for {self.task.name!r}, got task={task!r}")
        raise NotImplementedError(
            "Level1MJXPlannerAdapter is scaffolded. Implement Phase-1 by "
            "building an exp15-compatible MJX rollout backend and returning "
            "an OSC_POSE 7-D action sequence for robosuite certification."
        )


def make_planner(config: MJXPlannerConfig | None = None) -> MJXPlannerBackend:
    cfg = config or MJXPlannerConfig()
    if cfg.dry_run:
        return DryRunMJXPlanner(cfg)
    return Level1MJXPlannerAdapter(config=cfg)
