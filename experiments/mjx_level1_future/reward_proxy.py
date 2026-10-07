from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .task_registry import MJXTaskSpec


@dataclass(frozen=True)
class RewardProxyResult:
    cost: float
    success_proxy: bool
    components: dict[str, float]
    metadata: dict[str, Any]


class RewardProxy(Protocol):
    def evaluate(
        self,
        *,
        task: MJXTaskSpec,
        trajectory: Any,
        actions: np.ndarray,
    ) -> RewardProxyResult:
        ...


class PlaceholderRewardProxy:
    def evaluate(
        self,
        *,
        task: MJXTaskSpec,
        trajectory: Any,
        actions: np.ndarray,
    ) -> RewardProxyResult:
        del trajectory, actions
        raise NotImplementedError(
            f"reward proxy for {task.name!r} is scaffolded but not implemented"
        )


class DryRunRewardProxy:
    def evaluate(
        self,
        *,
        task: MJXTaskSpec,
        trajectory: Any,
        actions: np.ndarray,
    ) -> RewardProxyResult:
        del trajectory
        actions = np.asarray(actions, dtype=np.float32)
        return RewardProxyResult(
            cost=float(np.sum(actions * actions)),
            success_proxy=False,
            components={"action_l2": float(np.sum(actions * actions))},
            metadata={"mode": "dry_run", "task_name": task.name},
        )
