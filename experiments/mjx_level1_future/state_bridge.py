from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .task_registry import MJXTaskSpec


@dataclass(frozen=True)
class MJXInitialState:
    qpos: np.ndarray
    qvel: np.ndarray
    source_backend: str
    task_name: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        qpos = np.asarray(self.qpos)
        qvel = np.asarray(self.qvel)
        if qpos.ndim != 1:
            raise ValueError(f"qpos must be 1-D, got {qpos.shape}")
        if qvel.ndim != 1:
            raise ValueError(f"qvel must be 1-D, got {qvel.shape}")


class StateBridge(Protocol):
    def from_robosuite_state(
        self,
        *,
        task: MJXTaskSpec,
        init_reference: Any,
        env: Any | None = None,
    ) -> MJXInitialState:
        ...


class PlaceholderStateBridge:
    def from_robosuite_state(
        self,
        *,
        task: MJXTaskSpec,
        init_reference: Any,
        env: Any | None = None,
    ) -> MJXInitialState:
        raise NotImplementedError(
            f"state bridge for {task.name!r} is scaffolded but not implemented"
        )


def validate_state_shapes(state: MJXInitialState, *, nq: int | None = None, nv: int | None = None) -> None:
    if nq is not None and int(state.qpos.shape[0]) != int(nq):
        raise ValueError(f"qpos length {state.qpos.shape[0]} != expected nq={nq}")
    if nv is not None and int(state.qvel.shape[0]) != int(nv):
        raise ValueError(f"qvel length {state.qvel.shape[0]} != expected nv={nv}")
