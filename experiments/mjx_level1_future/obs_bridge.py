from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .task_registry import MJXTaskSpec


@dataclass(frozen=True)
class MJXObservation:
    vector: np.ndarray
    schema: str
    task_name: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector)
        if vector.ndim != 1:
            raise ValueError(f"observation vector must be 1-D, got {vector.shape}")


class ObservationBridge(Protocol):
    def from_mjx_data(
        self,
        *,
        task: MJXTaskSpec,
        mjx_data: Any,
        action_history: np.ndarray | None = None,
    ) -> MJXObservation:
        ...

    def from_robosuite_env(
        self,
        *,
        task: MJXTaskSpec,
        env: Any,
        action_history: np.ndarray | None = None,
    ) -> MJXObservation:
        ...


class PlaceholderObservationBridge:
    def from_mjx_data(
        self,
        *,
        task: MJXTaskSpec,
        mjx_data: Any,
        action_history: np.ndarray | None = None,
    ) -> MJXObservation:
        raise NotImplementedError(
            f"MJX observation bridge for {task.name!r} is scaffolded but not implemented"
        )

    def from_robosuite_env(
        self,
        *,
        task: MJXTaskSpec,
        env: Any,
        action_history: np.ndarray | None = None,
    ) -> MJXObservation:
        raise NotImplementedError(
            f"robosuite observation bridge for {task.name!r} is scaffolded but not implemented"
        )


def append_action_history(obs: np.ndarray, action_history: np.ndarray | None) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if action_history is None:
        return obs
    history = np.asarray(action_history, dtype=np.float32).reshape(-1)
    return np.concatenate([obs, history], axis=0).astype(np.float32)
