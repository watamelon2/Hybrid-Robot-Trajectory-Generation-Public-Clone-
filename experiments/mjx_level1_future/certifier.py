from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .level1_contract import (
    VALID_OSC_ACTION_DIMS,
    action_space_label,
    RobosuiteCertification,
    RobosuiteCertifier,
)

CheckSuccessFn = Callable[[Any], bool]
SetStateFn = Callable[[Any, Any], None]


@dataclass(frozen=True)
class RobosuiteCertifierConfig:
    task_name: str
    max_steps: int | None = None
    allowed_action_dims: tuple[int, ...] = VALID_OSC_ACTION_DIMS


class ReplayRobosuiteCertifier(RobosuiteCertifier):
    def __init__(
        self,
        *,
        env: Any,
        check_success_fn: CheckSuccessFn,
        set_state_fn: SetStateFn,
        config: RobosuiteCertifierConfig,
    ) -> None:
        self.env = env
        self.check_success_fn = check_success_fn
        self.set_state_fn = set_state_fn
        self.config = config

    def certify(
        self,
        *,
        task: str,
        init_reference: Any,
        actions: np.ndarray,
        output_trace: Path | None = None,
    ) -> RobosuiteCertification:
        del output_trace
        if task != self.config.task_name:
            raise ValueError(f"certifier configured for {self.config.task_name!r}, got task={task!r}")
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"actions must have shape (H, A), got {actions.shape}")
        if actions.shape[1] not in self.config.allowed_action_dims:
            raise ValueError(
                f"expected OSC_POSE action dim in {self.config.allowed_action_dims}, "
                f"got {actions.shape[1]}"
            )
        max_steps = actions.shape[0] if self.config.max_steps is None else min(actions.shape[0], self.config.max_steps)
        t0 = time.perf_counter()
        self.env.reset()
        self.set_state_fn(self.env, init_reference)
        reward_total = 0.0
        success_ever = False
        for action in actions[:max_steps]:
            _obs, reward, _done, _info = self.env.step(action)
            reward_total += float(reward)
            if not success_ever and self.check_success_fn(self.env):
                success_ever = True
        elapsed_s = time.perf_counter() - t0
        success = bool(success_ever or self.check_success_fn(self.env))
        return RobosuiteCertification(
            success=success,
            reward=float(reward_total),
            n_steps=int(max_steps),
            certification_metadata={
                "task_name": self.config.task_name,
                "action_space": action_space_label(actions.shape[1]),
                "elapsed_s": float(elapsed_s),
                "backend": "robosuite",
            },
        )


class DryRunRobosuiteCertifier(RobosuiteCertifier):
    def __init__(self, *, task_name: str = "can_level1", action_dim: int = 7) -> None:
        self.task_name = task_name
        self.action_dim = int(action_dim)

    def certify(
        self,
        *,
        task: str,
        init_reference: Any,
        actions: np.ndarray,
        output_trace: Path | None = None,
    ) -> RobosuiteCertification:
        del init_reference, output_trace
        actions = np.asarray(actions, dtype=np.float32)
        if task != self.task_name and task != "can":
            raise ValueError(f"dry-run certifier configured for {self.task_name!r}, got task={task!r}")
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(
                f"expected OSC_POSE action matrix (H, {self.action_dim}), got {actions.shape}"
            )
        return RobosuiteCertification(
            success=False,
            reward=0.0,
            n_steps=int(actions.shape[0]),
            certification_metadata={
                "mode": "dry_run",
                "backend": "robosuite",
                "action_space": action_space_label(actions.shape[1]),
                "canonical_success_available": False,
            },
        )
