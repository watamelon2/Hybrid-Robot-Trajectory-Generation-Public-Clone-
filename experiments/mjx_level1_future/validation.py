from __future__ import annotations

from dataclasses import dataclass

from .task_registry import MJXTaskSpec, iter_task_specs


@dataclass(frozen=True)
class ValidationGate:
    task_name: str
    phase: int
    required_before_report: bool
    paired_rollouts_min: int
    gate_name: str
    evidence_artifacts: tuple[str, ...]


def default_validation_gates() -> tuple[ValidationGate, ...]:
    gates: list[ValidationGate] = []
    for spec in iter_task_specs():
        gates.append(
            ValidationGate(
                task_name=spec.name,
                phase=spec.phase,
                required_before_report=spec.phase <= 3,
                paired_rollouts_min=60 if spec.phase <= 3 else 10,
                gate_name=spec.validation_gate,
                evidence_artifacts=(
                    "mjx_planner_json",
                    "robosuite_certification_json",
                    "paired_summary_json",
                ),
            )
        )
    return tuple(gates)


def missing_components(spec: MJXTaskSpec, available_components: set[str]) -> tuple[str, ...]:
    return tuple(component for component in spec.required_bridges if component not in available_components)


def assert_task_gate_ready(spec: MJXTaskSpec, available_components: set[str]) -> None:
    missing = missing_components(spec, available_components)
    if missing:
        raise RuntimeError(
            f"MJX task {spec.name!r} is not gate-ready; missing components: {', '.join(missing)}"
        )
