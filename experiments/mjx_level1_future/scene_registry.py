from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .task_registry import MJXTaskSpec, get_task_spec, iter_task_specs


@dataclass(frozen=True)
class MJXSceneSpec:
    task_name: str
    phase: int
    xml_path: Path | None
    source: str
    requires_authoring: bool
    parity_notes: tuple[str, ...]


def scene_spec_for_task(task: MJXTaskSpec | str) -> MJXSceneSpec:
    spec = get_task_spec(task) if isinstance(task, str) else task
    requires_authoring = spec.scene_path is None
    if spec.name in {"can_level1", "can_retrieval", "can_dp", "can_adaptation"}:
        source = "existing_exp04_simplified_can"
        parity_notes = (
            "simplified_scene",
            "robosuite_certification_required",
        )
    elif spec.name == "square_dp":
        source = "future_square_mjcf"
        parity_notes = (
            "contact_geometry_sensitive",
            "cost_rank_correlation_required",
            "robosuite_certification_required",
        )
    elif spec.name == "tool_hang":
        source = "future_tool_hang_mjcf"
        parity_notes = (
            "multi_object_contact_required",
            "long_horizon_required",
            "frame_and_tool_success_predicates_required",
            "subgoal_or_value_proxy_required",
            "robosuite_certification_required",
        )
    elif spec.name == "transport":
        source = "future_transport_mjcf"
        parity_notes = (
            "dual_arm_osc_required",
            "action_dim_14_required",
            "multi_object_contact_required",
            "success_predicate_required",
            "subgoal_or_value_proxy_required",
            "robosuite_certification_required",
        )
    else:
        source = "unknown"
        parity_notes = ("robosuite_certification_required",)
    return MJXSceneSpec(
        task_name=spec.name,
        phase=spec.phase,
        xml_path=spec.scene_path,
        source=source,
        requires_authoring=requires_authoring,
        parity_notes=parity_notes,
    )


def iter_scene_specs() -> tuple[MJXSceneSpec, ...]:
    return tuple(scene_spec_for_task(spec) for spec in iter_task_specs())


def assert_scene_available(spec: MJXSceneSpec, *, project_root: Path) -> None:
    if spec.xml_path is None:
        raise FileNotFoundError(f"MJX scene for {spec.task_name!r} has not been authored")
    path = project_root / spec.xml_path
    if not path.exists():
        raise FileNotFoundError(f"MJX scene for {spec.task_name!r} not found: {path}")
