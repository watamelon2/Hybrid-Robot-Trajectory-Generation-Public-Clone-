from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MJXTaskScaffold:
    task_name: str
    phase: int
    action_dim: int
    obs_keys: tuple[str, ...]
    synthetic_keys: tuple[str, ...]
    scene_requirements: tuple[str, ...]
    reward_components: tuple[str, ...]
    success_signals: tuple[str, ...]
    validation_gate: str
    horizon_note: str


SQUARE_DP_SCAFFOLD = MJXTaskScaffold(
    task_name="square_dp",
    phase=4,
    action_dim=7,
    obs_keys=(
        "SquareNut_pos",
        "SquareNut_quat",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "SquareNut_to_eef_pos_synth",
        "SquareNut_to_eef_quat_synth",
        "robot0_gripper_qpos",
    ),
    synthetic_keys=(
        "SquareNut_to_eef_pos_synth",
        "SquareNut_to_eef_quat_synth",
    ),
    scene_requirements=(
        "panda_single_arm_osc",
        "square_nut_body",
        "peg_or_socket_goal_geometry",
        "contact_parameters_matching_robosuite_intent",
    ),
    reward_components=(
        "eef_to_square",
        "square_lift",
        "square_to_peg_or_socket",
        "orientation_alignment",
        "robosuite_certified_success",
    ),
    success_signals=("robosuite_check_success",),
    validation_gate="mjx_cost_ranking_correlates_with_robosuite_outcome",
    horizon_note="Square-DP should not be promoted until paired robosuite certification confirms insertion/contact behavior.",
)


TOOL_HANG_SCAFFOLD = MJXTaskScaffold(
    task_name="tool_hang",
    phase=6,
    action_dim=7,
    obs_keys=(
        "frame_pos",
        "frame_quat",
        "tool_pos",
        "tool_quat",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot0_gripper_qvel",
        "robot0_joint_pos",
        "robot0_joint_pos_cos",
        "robot0_joint_pos_sin",
        "robot0_joint_vel",
    ),
    synthetic_keys=(),
    scene_requirements=(
        "panda_single_arm_osc",
        "stand_base_body",
        "frame_body",
        "tool_body",
        "hook_contact_geometry",
        "frame_assembly_constraint_or_contact_proxy",
    ),
    reward_components=(
        "eef_near_frame",
        "frame_assembled",
        "eef_near_tool",
        "tool_near_frame_hook",
        "tool_on_frame",
        "robosuite_certified_success",
    ),
    success_signals=("frame_is_assembled", "tool_on_frame", "robosuite_check_success"),
    validation_gate="tool_hang_phase_behavior_survives_robosuite_certification",
    horizon_note="ToolHang PH horizons are long; MJX prototypes should assume H>=300 before interpreting failure.",
)


TRANSPORT_SCAFFOLD = MJXTaskScaffold(
    task_name="transport",
    phase=6,
    action_dim=14,
    obs_keys=(
        "payload_pos",
        "payload_quat",
        "trash_pos",
        "trash_quat",
        "lid_handle_pos",
        "lid_handle_quat",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot0_gripper_qvel",
        "robot0_joint_pos",
        "robot0_joint_vel",
        "robot1_eef_pos",
        "robot1_eef_quat",
        "robot1_gripper_qpos",
        "robot1_gripper_qvel",
        "robot1_joint_pos",
        "robot1_joint_vel",
    ),
    synthetic_keys=(),
    scene_requirements=(
        "dual_panda_osc",
        "payload_body",
        "trash_body",
        "lid_handle_body",
        "target_bin_geometry",
        "trash_bin_geometry",
        "dual_arm_collision_and_contact_setup",
    ),
    reward_components=(
        "gripper0_to_payload",
        "payload_in_target_bin",
        "gripper1_to_trash",
        "trash_in_trash_bin",
        "joint_success_bonus",
        "robosuite_certified_success",
    ),
    success_signals=("payload_in_target_bin", "trash_in_trash_bin", "robosuite_check_success"),
    validation_gate="transport_dual_arm_behavior_survives_robosuite_certification",
    horizon_note="Transport uses two concatenated 7-D OSC_POSE commands and long PH horizons; use action_dim=14 and avoid single-arm assumptions.",
)


_TASK_SCAFFOLDS = {
    item.task_name: item
    for item in (SQUARE_DP_SCAFFOLD, TOOL_HANG_SCAFFOLD, TRANSPORT_SCAFFOLD)
}


def get_task_scaffold(task_name: str) -> MJXTaskScaffold:
    try:
        return _TASK_SCAFFOLDS[task_name]
    except KeyError as exc:
        valid = ", ".join(sorted(_TASK_SCAFFOLDS))
        raise KeyError(f"unknown task scaffold {task_name!r}; valid: {valid}") from exc


def iter_task_scaffolds() -> tuple[MJXTaskScaffold, ...]:
    return tuple(_TASK_SCAFFOLDS[name] for name in sorted(_TASK_SCAFFOLDS))
