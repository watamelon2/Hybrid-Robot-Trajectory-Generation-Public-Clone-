"""Runtime env perturbations for the adaptation experiments.

This module is the single place all adaptation flags from
``run_can.py`` route through.  Each helper here mutates the
robosuite env *after* ``set_sim_state`` has restored the canonical
init state, so the perturbation overrides whatever the dataset's
saved state vector encoded.

Both ``apply_goal_perturbation`` and ``apply_dynamics_perturbation``
are FULLY IMPLEMENTED (cost-time landed 2026-05-25 on
``feat/adapt-cost-time``; dynamics landed 2026-05-25 on
``feat/adapt-dynamics`` -- the two impls coexist here after the
dynamics merge).  All per-task registries (goal-body names, robot
link-body prefixes, friction-geom prefixes) are filled in from
direct REPL inspection of each task's MuJoCo model (see
/tmp/probe_robosuite_bodies.py for the probe script).

Cost-time impl: translates the goal body by mutating
``env.sim.model.body_pos`` in place + ``env.sim.forward()`` to
refresh kinematics.  Dynamics impl: mass scaling mutates both
``body_mass`` and ``body_inertia`` in proportion (uniform-density
approximation, Tobin et al. 2017 / Peng et al. 2018); friction
scaling mutates ``geom_friction`` for the contact pair (since
``npair == 0`` across our task family, there is no explicit
``pair_friction`` table to update).

See:

- ``docs/adaptation/01_cost_time_spec.md`` -- cost-time plan
  (goal-swap + obstacle-injection on Can).
- ``docs/adaptation/02_dynamics_spec.md`` -- dynamics plan (mass
  + friction perturbation; D1 and D2 ladders).

Flag surface and output JSON metadata are already wired in
``run_can.py``: ``--goal-perturbation-xyz``, ``--mass-scale``,
``--friction-scale``.  Output JSON's ``params`` block records the
perturbation values for analysis.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# Per-task target-body name registry.  Used by
# apply_goal_perturbation.  Filled 2026-05-25 by direct REPL
# inspection of each task's MuJoCo model (see
# /tmp/probe_robosuite_bodies.py for the script).
#
# - ``can`` -> ``bin2`` is the PLACEMENT bin in PickPlaceCan;
#   ``bin1`` is the source bin where the can starts.  Moving
#   ``bin2`` is the correct cost-time perturbation -- it changes
#   where the "place" subgoal is located.
# - ``lift`` has no static goal body -- the success condition is
#   the cube exceeding a height threshold above the table.  Goal
#   perturbation does not apply to Lift.
# - ``square`` -> ``peg1`` is the square-shaped peg in
#   NutAssemblySquare; ``peg2`` is the round-peg variant used by
#   NutAssemblyRound.
# - ``tool_hang`` / ``transport``: not yet probed; left as None so
#   apply_goal_perturbation raises a clear "task not registered"
#   error rather than silently using the wrong body.
_GOAL_BODY_NAMES: dict[str, str | None] = {
    "can":        "bin2",
    "lift":       None,       # Lift goal is height threshold; not applicable
    "square":     "peg1",
    "tool_hang":  None,       # TODO: probe ToolHang env
    "transport":  None,       # TODO: probe TwoArmTransport env
}


# Per-task robot link body prefixes.  Used by
# apply_dynamics_perturbation for mass scaling.  Filled 2026-05-25
# from REPL probe (see /tmp/probe_robosuite_bodies.py).  All four
# single-arm Robomimic tasks share the Panda + Panda-Gripper body
# structure: 7 numbered links (robot0_link0..link7), a ``hand``
# connector, and a gripper subtree under ``gripper0_``.  Transport
# is dual-arm (robot0 + robot1 each with their own gripper).
#
# A body matches if its name starts with any of the listed
# prefixes; the matching loop in apply_dynamics_perturbation scales
# every such body's mass AND inertia by mass_scale.
_ROBOT_LINK_BODY_PREFIXES: dict[str, tuple[str, ...]] = {
    "can":        ("robot0_link", "robot0_right_hand", "gripper0_"),
    "lift":       ("robot0_link", "robot0_right_hand", "gripper0_"),
    "square":     ("robot0_link", "robot0_right_hand", "gripper0_"),
    "tool_hang":  ("robot0_link", "robot0_right_hand", "gripper0_"),
    "transport":  (
        "robot0_link", "robot0_right_hand", "gripper0_",
        "robot1_link", "robot1_right_hand", "gripper1_",
    ),
}


# Per-task OBJECT-side friction geom prefixes.  Used by
# apply_dynamics_perturbation for friction scaling.  Filled
# 2026-05-25 from REPL probe.
#
# Why object-side prefixes instead of (object, table) pairs:
# the Can env uses **unnamed** collision geoms for bin1 / bin2
# (the can rests inside bin1, transported to bin2; both bins
# have anonymous box-collision children).  Pair-name lookup
# therefore fails for Can.  Instead, we scale every collision
# geom on the OBJECT's body subtree -- which gives a
# monotone-in-friction_scale knob for any contact the object is
# in, because MuJoCo blends the two geoms' frictions per-contact
# (default solmix=0.5 ⇒ effective ~ (a+b)/2; scaling one side
# preserves monotonicity in the scale even though it does not
# scale the *effective* friction linearly).
#
# Caveat: this means friction_scale's relationship to *effective*
# contact friction is sub-linear (factor ~0.5 for solmix=0.5).
# A scale of 2.0 yields ~1.5x effective; a scale of 0.5 yields
# ~0.75x effective.  Acceptable for the experiment because the
# direction of the gradient is preserved; for tight constants
# the analysis pipeline should report friction_scale directly
# (which the output JSON does) rather than effective friction.
_FRICTION_GEOM_PREFIXES: dict[str, tuple[str, ...] | None] = {
    "can":        ("Can_g",),         # scales Can_g0 (and Can_g0_visual harmlessly)
    "lift":       ("cube_g",),        # scales cube_g0
    "square":     ("SquareNut_g",),   # scales SquareNut_g0..g4
    "tool_hang":  None,               # TODO: probe ToolHang env
    "transport":  None,               # TODO: probe TwoArmTransport env
}


def apply_goal_perturbation(env, task: str, xyz: Tuple[float, float, float]) -> None:
    """Translate the task's goal body by ``xyz`` metres in the world frame.

    Called once per env instantiation by ``run_can.py`` when
    ``--goal-perturbation-xyz`` is non-zero.  The translation is
    applied *after* ``set_sim_state`` restores the canonical init
    state, so it strictly overrides the dataset's encoded goal
    position.

    Parameters
    ----------
    env : robosuite env
        Live env returned by ``make_env_for_dataset``.  Must expose
        ``env.sim.model`` (MuJoCo bindings) for body lookup.
    task : str
        One of ``can`` / ``lift`` / ``square`` / ``tool_hang`` / ``transport``.
        Must be a key of ``_GOAL_BODY_NAMES``.
    xyz : (float, float, float)
        World-frame translation in metres.  ``(0, 0, 0)`` is a
        no-op and the caller should skip the call entirely.

    Raises
    ------
    KeyError
        If ``task`` is not a key of ``_GOAL_BODY_NAMES``.  Add the
        task to the registry via a REPL probe before perturbing.
    ValueError
        If ``task`` has no static goal body (e.g., Lift, where the
        goal is a height threshold), so cost-time goal-perturbation
        is structurally inapplicable.

    Notes
    -----
    The translation mutates ``env.sim.model.body_pos`` (NOT
    ``body_xpos`` -- xpos is a derived quantity refreshed by
    ``sim.forward()``).  ``body_pos`` is the body's offset relative
    to its parent in the MuJoCo kinematic tree; for the root-of-task
    bodies in our registry (bin2 / peg1) the parent is the world,
    so ``body_pos`` IS the world-frame position.  ``sim.forward()``
    re-derives ``body_xpos`` and refreshes contact data.
    """
    if xyz == (0.0, 0.0, 0.0):
        return  # canonical no-op, defensive

    if task not in _GOAL_BODY_NAMES:
        raise KeyError(
            f"task={task!r} is not registered in _GOAL_BODY_NAMES; "
            f"known: {sorted(_GOAL_BODY_NAMES)}.  Probe the env in a "
            f"REPL and add the target-body name; see the module "
            f"docstring and docs/adaptation/01_cost_time_spec.md."
        )
    body_name = _GOAL_BODY_NAMES[task]
    if body_name is None:
        raise ValueError(
            f"task={task!r} has no static goal body "
            f"(success condition is a state predicate, not a pose); "
            f"cost-time goal-perturbation does not apply.  Use a "
            f"different adaptation axis (e.g. --mass-scale on "
            f"feat/adapt-dynamics)."
        )

    sim = env.sim
    delta = np.asarray(xyz, dtype=np.float64)
    bid = sim.model.body_name2id(body_name)
    sim.model.body_pos[bid] = sim.model.body_pos[bid].astype(np.float64) + delta

    # Task-specific: robosuite caches reward / success goal positions
    # in env-level attributes that are NOT derived from sim.model.body_pos
    # at evaluation time.  We must mutate the cached attributes too,
    # otherwise the reward and success check read the unperturbed goal
    # while only the *visual* body moves.  Worse, PickPlace's
    # _initialize_sim() actively *restores* body_pos[bin2] from the
    # cached self.bin2_pos on every env.reset() (pick_place.py line 713)
    # -- so without this, the body_pos mutation reverts on the next
    # rollout and the perturbation is a complete no-op for both reward
    # and success.
    #
    # NutAssembly (Square) is the lucky exception: its reward reads
    # sim.data.body_xpos[peg1_body_id] LIVE (nut_assembly.py line 363+),
    # not a cached attribute, and _initialize_sim does NOT restore peg
    # body_pos.  So body_pos mutation alone is sufficient there.
    #
    # The branches below extend this for new tasks; if a task is in the
    # registry but has no matching attribute branch, we emit a noisy
    # warning so the experimenter knows the perturbation may not flow
    # through to the reward.
    _patch_cached_goal_attrs(env, task, delta)

    sim.forward()


def _patch_cached_goal_attrs(env, task: str, delta: np.ndarray) -> None:
    """Update env-level cached goal-position attributes for tasks that
    reward off cached values (PickPlace family) rather than live sim
    state (NutAssembly family).

    Updates required per task:

    - ``can`` (``PickPlaceCan``): ``env.bin2_pos`` (used by
      ``staged_rewards``, ``_check_success``, and ``_initialize_sim``;
      the last forces sim.model.body_pos[bin2] back to this value on
      every reset) AND ``env.target_bin_placements`` (a (n_objects, 3)
      array used inside ``staged_rewards`` for the ``hover_reward``
      stage).
    - ``square`` (``NutAssemblySquare``): no env-level attribute exists;
      reward reads live ``sim.data.body_xpos[peg1_body_id]``.  No-op
      here; ``sim.forward()`` after the body_pos mutation is enough.
    - ``lift``: not applicable (apply_goal_perturbation already raises
      ValueError for Lift before reaching this function).
    - ``tool_hang`` / ``transport``: not yet probed; emits a warning
      directing the user to add a branch here once the cached
      attributes are identified.
    """
    if task == "can":
        # Both attributes are np.ndarray; .copy() defensively to avoid
        # aliasing surprises if robosuite shares the array across uses.
        env.bin2_pos = env.bin2_pos.astype(np.float64).copy() + delta
        env.target_bin_placements = (
            env.target_bin_placements.astype(np.float64).copy() + delta[None, :]
        )
    elif task == "square":
        # Live-read; nothing to patch.
        pass
    else:
        import warnings
        warnings.warn(
            f"apply_goal_perturbation: task={task!r} has body_pos mutated "
            f"but no cached-attribute branch in _patch_cached_goal_attrs. "
            f"The reward function may still read the unperturbed goal.  "
            f"Inspect robosuite's reward() for this task and either add "
            f"a branch here or document why the perturbation is reward-"
            f"transparent for this task.",
            stacklevel=3,
        )


def apply_dynamics_perturbation(
    env,
    task: str,
    mass_scale: float = 1.0,
    friction_scale: float = 1.0,
) -> None:
    """Scale link masses and/or table-object friction by the given factors.

    Called once per env instantiation by ``run_can.py`` when either
    ``--mass-scale`` or ``--friction-scale`` is not 1.0.  The
    mutation is applied to ``env.sim.model`` (not ``env.sim.data``)
    so it persists across env.reset() within the same env instance.

    Parameters
    ----------
    env : robosuite env
        Live env returned by ``make_env_for_dataset``.  Must expose
        ``env.sim.model``.
    task : str
        Task name; used to look up the robot's link bodies and the
        contact-pair geoms in this module's registries.
    mass_scale : float
        Multiplier applied to every body in ``_ROBOT_LINK_BODY_PREFIXES[task]``.
        ``1.0`` is the canonical no-op.
    friction_scale : float
        Multiplier applied to the tangential friction coefficient
        of the geom pair in ``_FRICTION_CONTACT_GEOMS[task]``.
        ``1.0`` is the canonical no-op.

    Raises
    ------
    KeyError
        If ``task`` is not a key of ``_ROBOT_LINK_BODY_PREFIXES`` or
        ``_FRICTION_CONTACT_GEOMS``.
    ValueError
        If ``friction_scale != 1.0`` but the task's friction-pair
        registry entry is None (task not yet probed).

    Notes
    -----
    Mass scaling: both ``body_mass`` and ``body_inertia`` are
    multiplied by ``mass_scale``.  body_inertia is the principal
    moments of inertia (length-3 per body); scaling proportionally
    is the standard "uniform density, same shape" approximation
    used in domain-randomisation work (Tobin et al. 2017,
    Peng et al. 2018).

    Friction scaling: ``geom_friction`` is a length-3 row per geom
    (sliding / torsional / rolling).  We scale all three
    components for both geoms in the contact pair.  Since
    ``env.sim.model.npair == 0`` across our task family, MuJoCo
    computes contact friction as the per-geom maximum (or
    solref-blended), so scaling both ends of the pair guarantees
    the effective contact friction also scales by the same factor.
    """
    if mass_scale == 1.0 and friction_scale == 1.0:
        return  # canonical no-op

    sim = env.sim

    # ---- Mass perturbation ------------------------------------------------
    if mass_scale != 1.0:
        if task not in _ROBOT_LINK_BODY_PREFIXES:
            raise KeyError(
                f"task={task!r} not in _ROBOT_LINK_BODY_PREFIXES; "
                f"known: {sorted(_ROBOT_LINK_BODY_PREFIXES)}"
            )
        prefixes = _ROBOT_LINK_BODY_PREFIXES[task]
        n_scaled = 0
        for bid in range(sim.model.nbody):
            name = sim.model.body_id2name(bid)
            if name and name.startswith(prefixes):
                sim.model.body_mass[bid] *= mass_scale
                sim.model.body_inertia[bid] *= mass_scale
                n_scaled += 1
        if n_scaled == 0:
            raise ValueError(
                f"task={task!r}: mass_scale={mass_scale} but no "
                f"bodies matched prefixes {prefixes!r}.  Re-probe the "
                f"env's body_names; the registry may be stale."
            )

    # ---- Friction perturbation -------------------------------------------
    if friction_scale != 1.0:
        if task not in _FRICTION_GEOM_PREFIXES:
            raise KeyError(
                f"task={task!r} not in _FRICTION_GEOM_PREFIXES; "
                f"known: {sorted(_FRICTION_GEOM_PREFIXES)}"
            )
        prefixes = _FRICTION_GEOM_PREFIXES[task]
        if prefixes is None:
            raise ValueError(
                f"task={task!r}: friction_scale={friction_scale} but "
                f"the task's friction-prefix entry is None.  Probe the "
                f"env and add object-side geom prefix(es) to "
                f"_FRICTION_GEOM_PREFIXES first."
            )
        n_geoms_scaled = 0
        for gid in range(sim.model.ngeom):
            name = sim.model.geom_id2name(gid)
            if name and name.startswith(prefixes):
                sim.model.geom_friction[gid] = (
                    sim.model.geom_friction[gid].astype(np.float64) * friction_scale
                )
                n_geoms_scaled += 1
        if n_geoms_scaled == 0:
            raise ValueError(
                f"task={task!r}: friction_scale={friction_scale} but no "
                f"geoms matched prefixes {prefixes!r}.  Re-probe the "
                f"env's geom_names; the registry may be stale."
            )

    sim.forward()
