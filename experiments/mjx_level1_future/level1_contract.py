from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

VALID_OSC_ACTION_DIMS = (7, 14)


def action_space_label(action_dim: int) -> str:
    if int(action_dim) == 7:
        return "OSC_POSE_7D"
    if int(action_dim) == 14:
        return "OSC_POSE_DUAL_14D"
    raise ValueError(f"unsupported OSC action_dim={action_dim}; valid: {VALID_OSC_ACTION_DIMS}")


@dataclass(frozen=True)
class MJXPlanCandidate:
    """Action sequence selected by an MJX candidate-rollout planner."""

    actions: np.ndarray
    planner_cost: float
    planner_metadata: dict[str, Any]

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions)
        if actions.ndim != 2:
            raise ValueError(f"actions must have shape (H, A), got {actions.shape}")
        if actions.shape[1] not in VALID_OSC_ACTION_DIMS:
            raise ValueError(
                "Level-1 MJX must preserve robosuite OSC_POSE action semantics; "
                f"got action_dim={actions.shape[1]}"
            )


@dataclass(frozen=True)
class RobosuiteCertification:
    """Ground-truth replay result for an MJX-selected action sequence."""

    success: bool
    reward: float
    n_steps: int
    certification_metadata: dict[str, Any]


@dataclass(frozen=True)
class MJXScopeItem:
    """A gated task/phase entry for the expanded Level-1 roadmap."""

    phase: int
    name: str
    task_family: str
    implementation_status: str
    required_components: tuple[str, ...]
    validation_gate: str
    report_role: str


LEVEL1_SCOPE: tuple[MJXScopeItem, ...] = (
    MJXScopeItem(
        phase=0,
        name="existing_can_gate4",
        task_family="can",
        implementation_status="existing evidence",
        required_components=(
            "simplified_can_mjx_backend",
            "jax_osc_controller",
            "gpu_speed_measurement",
        ),
        validation_gate="appendix_only_no_canonical_success",
        report_role="engineering appendix",
    ),
    MJXScopeItem(
        phase=1,
        name="can_level1_planner_certifier",
        task_family="can",
        implementation_status="first implementation target",
        required_components=(
            "exp15_backend_seam",
            "can_state_bridge",
            "can_obs_bridge",
            "robosuite_certification_replay",
        ),
        validation_gate="paired_can_mjx_planned_robosuite_certified_trials",
        report_role="future direction until gate passes",
    ),
    MJXScopeItem(
        phase=2,
        name="can_retrieval_coverage",
        task_family="can",
        implementation_status="planned extension",
        required_components=(
            "retrieval_feature_bridge",
            "fresh_seed_state_bridge",
            "bank_size_wrapper",
        ),
        validation_gate="fresh_seed_bank_size_trend_matches_robosuite",
        report_role="validated acceleration study after phase 1",
    ),
    MJXScopeItem(
        phase=3,
        name="can_dp_headroom",
        task_family="can_dp",
        implementation_status="planned extension",
        required_components=(
            "can_dp_obs_schema",
            "obs_history_buffer",
            "dp_prior_bridge",
            "robosuite_certification_replay",
        ),
        validation_gate="dp_checkpoint_headroom_conclusion_preserved",
        report_role="selected expensive-cell acceleration",
    ),
    MJXScopeItem(
        phase=4,
        name="square_dp_prototype",
        task_family="square_dp",
        implementation_status="deferred high-risk prototype",
        required_components=(
            "square_mjx_scene",
            "square_obs_schema",
            "square_reward_proxy",
            "paired_robosuite_certification",
        ),
        validation_gate="mjx_cost_ranking_correlates_with_robosuite_outcome",
        report_role="future work",
    ),
    MJXScopeItem(
        phase=5,
        name="can_adaptation_mass_friction",
        task_family="can_adaptation",
        implementation_status="deferred extension",
        required_components=(
            "mjx_mass_friction_controls",
            "robosuite_to_mjx_perturbation_mapping",
            "paired_perturbation_validation",
        ),
        validation_gate="perturbation_semantics_match_robosuite_trends",
        report_role="future acceleration study",
    ),
    MJXScopeItem(
        phase=6,
        name="sparse_task_prototypes",
        task_family="tool_hang_transport",
        implementation_status="deferred research infrastructure",
        required_components=(
            "new_mjx_scenes",
            "multi_object_contact_setup",
            "task_success_predicates",
            "subgoal_or_value_reward_proxy",
        ),
        validation_gate="small_paired_runs_survive_robosuite_certification",
        report_role="longer-term future work",
    ),
)


class MJXPlannerBackend(Protocol):
    """Future interface for MJX candidate planning.

    Implementations should use MJX only to rank candidate MBD rollouts.
    They must keep OSC_POSE action semantics and must not report MJX-only
    success as the final task success.
    """

    def plan(
        self,
        *,
        task: str,
        init_reference: Any,
        obs_reference: np.ndarray,
        prior_context: dict[str, Any],
    ) -> MJXPlanCandidate:
        """Return an OSC_POSE action sequence chosen by MJX planning."""


class RobosuiteCertifier(Protocol):
    """Future interface for robosuite final-success certification."""

    def certify(
        self,
        *,
        task: str,
        init_reference: Any,
        actions: np.ndarray,
        output_trace: Path | None = None,
    ) -> RobosuiteCertification:
        """Replay actions in robosuite and return canonical success/reward."""


def run_level1_trial(
    *,
    planner: MJXPlannerBackend,
    certifier: RobosuiteCertifier,
    task: str,
    init_reference: Any,
    obs_reference: np.ndarray,
    prior_context: dict[str, Any],
    output_trace: Path | None = None,
) -> dict[str, Any]:
    """Contract-level orchestration for Level-1 MJX.

    This function intentionally contains no MJX or robosuite imports. It is a
    minimal future-direction seam: MJX proposes an OSC_POSE action sequence;
    robosuite certifies that exact sequence.
    """

    candidate = planner.plan(
        task=task,
        init_reference=init_reference,
        obs_reference=obs_reference,
        prior_context=prior_context,
    )
    certification = certifier.certify(
        task=task,
        init_reference=init_reference,
        actions=candidate.actions,
        output_trace=output_trace,
    )
    return {
        "task": task,
        "action_space": action_space_label(candidate.actions.shape[1]),
        "planner_backend": "mjx",
        "certifier_backend": "robosuite",
        "planner_cost": float(candidate.planner_cost),
        "success": bool(certification.success),
        "reward": float(certification.reward),
        "n_steps": int(certification.n_steps),
        "planner_metadata": dict(candidate.planner_metadata),
        "certification_metadata": dict(certification.certification_metadata),
    }
