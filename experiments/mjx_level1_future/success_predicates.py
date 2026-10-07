from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class SuccessPredicateResult:
    success: bool
    signals: dict[str, bool]
    metadata: dict[str, Any]


class SuccessPredicate(Protocol):
    def evaluate(self, *, task_name: str, state_or_obs: Any) -> SuccessPredicateResult:
        ...


class PlaceholderSuccessPredicate:
    def evaluate(self, *, task_name: str, state_or_obs: Any) -> SuccessPredicateResult:
        del state_or_obs
        raise NotImplementedError(
            f"MJX success predicate for {task_name!r} is scaffolded but not implemented"
        )


class DryRunSuccessPredicate:
    def evaluate(self, *, task_name: str, state_or_obs: Any) -> SuccessPredicateResult:
        del state_or_obs
        return SuccessPredicateResult(
            success=False,
            signals={},
            metadata={"mode": "dry_run", "task_name": task_name},
        )
