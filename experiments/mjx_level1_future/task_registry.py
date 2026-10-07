from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TaskFamily = Literal[
    "can",
    "can_dp",
    "square_dp",
    "can_adaptation",
    "tool_hang",
    "transport",
]


@dataclass(frozen=True)
class MJXTaskSpec:
    name: str
    family: TaskFamily
    phase: int
    scene_path: Path | None
    obs_schema: str | None
    action_dim: int
    supported_priors: tuple[str, ...]
    required_bridges: tuple[str, ...]
    validation_gate: str
    report_role: str
    implementation_status: str


_TASK_SPECS: dict[str, MJXTaskSpec] = {
    "can_level1": MJXTaskSpec(
        name="can_level1",
        family="can",
        phase=1,
        scene_path=Path("experiments/04_mbd_baseline/scenes/simplified_can.xml"),
        obs_schema="can",
        action_dim=7,
        supported_priors=("none", "demo_1nn", "flow"),
        required_bridges=("state", "obs", "certifier"),
        validation_gate="paired_can_mjx_planned_robosuite_certified_trials",
        report_role="future direction until gate passes",
        implementation_status="scaffolded",
    ),
    "can_retrieval": MJXTaskSpec(
        name="can_retrieval",
        family="can",
        phase=2,
        scene_path=Path("experiments/04_mbd_baseline/scenes/simplified_can.xml"),
        obs_schema="can",
        action_dim=7,
        supported_priors=("demo_1nn", "demo_knn", "demo_lwr"),
        required_bridges=("state", "obs", "retrieval_features", "certifier"),
        validation_gate="fresh_seed_bank_size_trend_matches_robosuite",
        report_role="validated acceleration study after phase 1",
        implementation_status="planned",
    ),
    "can_dp": MJXTaskSpec(
        name="can_dp",
        family="can_dp",
        phase=3,
        scene_path=Path("experiments/04_mbd_baseline/scenes/simplified_can.xml"),
        obs_schema="can_dp",
        action_dim=7,
        supported_priors=("flow",),
        required_bridges=("state", "obs", "obs_history", "dp_prior", "certifier"),
        validation_gate="dp_checkpoint_headroom_conclusion_preserved",
        report_role="selected expensive-cell acceleration",
        implementation_status="planned",
    ),
    "square_dp": MJXTaskSpec(
        name="square_dp",
        family="square_dp",
        phase=4,
        scene_path=None,
        obs_schema="square_dp",
        action_dim=7,
        supported_priors=("flow",),
        required_bridges=("scene", "state", "obs", "reward_proxy", "certifier"),
        validation_gate="mjx_cost_ranking_correlates_with_robosuite_outcome",
        report_role="future work",
        implementation_status="deferred_high_risk",
    ),
    "can_adaptation": MJXTaskSpec(
        name="can_adaptation",
        family="can_adaptation",
        phase=5,
        scene_path=Path("experiments/04_mbd_baseline/scenes/simplified_can.xml"),
        obs_schema="can",
        action_dim=7,
        supported_priors=("demo_1nn", "flow"),
        required_bridges=("state", "obs", "mass_friction_mapping", "certifier"),
        validation_gate="perturbation_semantics_match_robosuite_trends",
        report_role="future acceleration study",
        implementation_status="planned_after_can",
    ),
    "tool_hang": MJXTaskSpec(
        name="tool_hang",
        family="tool_hang",
        phase=6,
        scene_path=None,
        obs_schema="tool_hang",
        action_dim=7,
        supported_priors=("demo_1nn", "flow"),
        required_bridges=("scene", "state", "obs", "success_predicate", "subgoal_reward", "certifier"),
        validation_gate="tool_hang_phase_behavior_survives_robosuite_certification",
        report_role="longer-term future work",
        implementation_status="deferred_research_infrastructure",
    ),
    "transport": MJXTaskSpec(
        name="transport",
        family="transport",
        phase=6,
        scene_path=None,
        obs_schema="transport",
        action_dim=14,
        supported_priors=("demo_1nn", "flow"),
        required_bridges=("scene", "state", "dual_arm_obs", "success_predicate", "subgoal_reward", "certifier"),
        validation_gate="transport_dual_arm_behavior_survives_robosuite_certification",
        report_role="longer-term future work",
        implementation_status="deferred_research_infrastructure",
    ),
}


def get_task_spec(name: str) -> MJXTaskSpec:
    try:
        return _TASK_SPECS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(_TASK_SPECS))
        raise KeyError(f"unknown MJX task spec {name!r}; valid: {valid}") from exc


def iter_task_specs() -> tuple[MJXTaskSpec, ...]:
    return tuple(sorted(_TASK_SPECS.values(), key=lambda spec: (spec.phase, spec.name)))


def promotable_task_names(max_phase: int = 3) -> tuple[str, ...]:
    return tuple(spec.name for spec in iter_task_specs() if spec.phase <= int(max_phase))
