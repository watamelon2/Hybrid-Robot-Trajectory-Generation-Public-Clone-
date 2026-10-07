"""Stage 1 of experiment 04: MBD on a 2-D linear-quadratic regulator.

Validates the MBD denoising loop in isolation against the closed-form
LQ solution from the discrete algebraic Riccati equation
(``scipy.linalg.solve_discrete_are``). No mujoco, no robosuite.

Why this test exists: if the denoising schedule is wrong, the softmax
temperature mis-set, or the weighted average flipped in sign, the bug
will be visible here without spending minutes per debug iteration on
mjx rollouts. Pass criterion is documented in README.md ("Stage 1").
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Make sibling modules importable without dotted-package syntax (this
# directory's name starts with a digit, see exp 03's policy.py).
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from mbd import (  # noqa: E402
    MBDParams,
    cosine_sigma_schedule,
    denoise_loop,
)


# ---------------------------------------------------------------------------
# 2-D linear-quadratic system (constants chosen for a non-trivial regulator)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LQSystem:
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray

    @property
    def state_dim(self) -> int:
        return int(self.A.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.B.shape[1])


def default_system() -> LQSystem:
    """A small 2-D point-mass discretised at dt=0.1 with friction.

    State: ``[x, v]``. Control: scalar force on ``v``. Quadratic cost
    on staying near the origin.
    """
    dt = 0.1
    A = np.array([[1.0, dt], [0.0, 1.0 - 0.05 * dt]], dtype=np.float32)
    B = np.array([[0.0], [dt]], dtype=np.float32)
    Q = np.diag([1.0, 0.1]).astype(np.float32)
    R = np.array([[0.01]], dtype=np.float32)
    return LQSystem(A=A, B=B, Q=Q, R=R)


def lq_optimal_gain(system: LQSystem) -> np.ndarray:
    """Discrete-time LQR feedback gain ``K*`` via DARE.

    ``u_t = -K x_t`` is the cost-minimising policy.
    """
    # Imported lazily so module import doesn't hard-require scipy.
    from scipy.linalg import solve_discrete_are

    P = solve_discrete_are(system.A, system.B, system.Q, system.R)
    K = np.linalg.solve(system.B.T @ P @ system.B + system.R,
                        system.B.T @ P @ system.A)
    return K


def lq_rollout_cost(
    system: LQSystem,
    x0: np.ndarray,
    actions: np.ndarray,
) -> float:
    """Closed-form total cost of a single horizon-H action sequence."""
    x = x0.astype(np.float32).copy()
    cost = 0.0
    for u in actions:
        cost += float(x @ system.Q @ x + u @ system.R @ u)
        x = system.A @ x + system.B @ u
    cost += float(x @ system.Q @ x)  # terminal
    return cost


def lq_optimal_action_sequence(
    system: LQSystem,
    x0: np.ndarray,
    H: int,
) -> np.ndarray:
    """Roll the DARE-optimal feedback ``u = -K* x`` open-loop from x0.

    The returned action sequence is what MBD must approximate.
    Note this is the *infinite-horizon* gain applied to a finite
    horizon, which is slightly suboptimal for small H but standard for
    sanity-check purposes (gap shrinks fast in H).
    """
    K_star = lq_optimal_gain(system)
    x = x0.astype(np.float32).copy()
    actions = np.zeros((H, system.action_dim), dtype=np.float32)
    for t in range(H):
        u = (-K_star @ x).astype(np.float32)
        actions[t] = u
        x = system.A @ x + system.B @ u
    return actions


def make_rollout_fn(system: LQSystem, x0: np.ndarray):
    """Closure: ``(traj_batch) -> states_batch`` for the given x0."""
    A = system.A
    B = system.B
    state_dim = system.state_dim

    def rollout_fn(traj_batch: np.ndarray) -> np.ndarray:
        # traj_batch: (M, H, action_dim)
        M, H, _ = traj_batch.shape
        states = np.empty((M, H + 1, state_dim), dtype=np.float32)
        states[:, 0, :] = x0[None, :]
        for t in range(H):
            # state_t @ A.T + action_t @ B.T (per-sample matmul)
            states[:, t + 1, :] = (
                states[:, t, :] @ A.T + traj_batch[:, t, :] @ B.T
            )
        return states

    return rollout_fn


def make_cost_fn(system: LQSystem):
    """Closure: ``(states_batch, traj_batch) -> (M,)`` quadratic cost."""
    Q = system.Q
    R = system.R

    def cost_fn(
        states_batch: np.ndarray, traj_batch: np.ndarray
    ) -> np.ndarray:
        # states_batch: (M, H+1, state_dim), traj_batch: (M, H, action_dim)
        x_cost = np.einsum("mti,ij,mtj->m", states_batch, Q, states_batch)
        u_cost = np.einsum("mti,ij,mtj->m", traj_batch, R, traj_batch)
        return x_cost + u_cost

    return cost_fn


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--H", type=int, default=30,
                   help="LQ horizon (default 30 steps).")
    p.add_argument("--M", type=int, default=128,
                   help="MBD perturbed copies per iteration.")
    p.add_argument("--K", type=int, default=100,
                   help="MBD denoising iterations. Default 100 gives "
                        "~5-8% above LQ optimum across seeds; K=50 sits "
                        "right on the 10% pass threshold and fails for "
                        "some seeds.")
    p.add_argument("--lambda_", type=float, default=5.0,
                   help="MBD softmax temperature. Cost differences "
                        "between random and optimal trajectories on the "
                        "default system are O(20-30); lambda ~ 5 gives "
                        "non-degenerate ESS through the schedule.")
    p.add_argument("--sigma-max", type=float, default=2.0,
                   help="Initial perturbation std (cosine schedule top).")
    p.add_argument("--sigma-min", type=float, default=0.05,
                   help="Final perturbation std (cosine schedule end).")
    p.add_argument("--pass-ratio", type=float, default=1.10,
                   help="Pass criterion: MBD cost / LQ optimum <= this.")
    p.add_argument(
        "--output",
        type=Path,
        default=_THIS_DIR / "results" / "lq_sanity.json",
        help="Where to write the JSON cost-vs-iteration log.",
    )
    return p.parse_args()


def main() -> int:
    """Run the LQ sanity check end-to-end and return 0 on pass / 1 on fail."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # 1. Build the system + analytical optimum.
    # ------------------------------------------------------------------
    system = default_system()
    x0 = np.array([1.0, 0.0], dtype=np.float32)

    optimal_actions = lq_optimal_action_sequence(system, x0, args.H)
    optimal_cost = lq_rollout_cost(system, x0, optimal_actions)
    zero_cost = lq_rollout_cost(
        system, x0,
        np.zeros((args.H, system.action_dim), dtype=np.float32),
    )

    print(f"[lq   ] H={args.H}  M={args.M}  K={args.K}  "
          f"lambda={args.lambda_}  seed={args.seed}")
    print(f"[lq   ] LQ optimal (-K* feedback) cost: {optimal_cost:.4f}")
    print(f"[lq   ] zero-action cost (baseline)   : {zero_cost:.4f}")

    # ------------------------------------------------------------------
    # 2. Hook the LQ system into the MBD callable interface.
    # ------------------------------------------------------------------
    rollout_fn = make_rollout_fn(system, x0)
    cost_fn = make_cost_fn(system)
    schedule = cosine_sigma_schedule(
        args.K, sigma_max=args.sigma_max, sigma_min=args.sigma_min
    )
    params = MBDParams(
        H=args.H,
        M=args.M,
        K=args.K,
        action_dim=system.action_dim,
        lambda_=args.lambda_,
        sigma_schedule=schedule,
        clip_action=None,
    )

    # ------------------------------------------------------------------
    # 3. Run MBD from a zero initial mean.
    # ------------------------------------------------------------------
    mu_initial = np.zeros((args.H, system.action_dim), dtype=np.float32)
    mu_final, log = denoise_loop(
        mu_initial, rollout_fn, cost_fn, params, rng
    )
    mbd_cost = lq_rollout_cost(system, x0, mu_final)

    # ------------------------------------------------------------------
    # 4. Compare + report.
    # ------------------------------------------------------------------
    ratio = mbd_cost / optimal_cost
    passed = ratio <= args.pass_ratio
    print(f"[lq   ] MBD final mean cost           : {mbd_cost:.4f}")
    print(f"[lq   ] ratio MBD / optimum           : {ratio:.3f}")
    print(f"[lq   ] pass criterion ratio <= {args.pass_ratio}: "
          f"{'PASS' if passed else 'FAIL'}")

    summary = {
        "experiment": "04_mbd_baseline/lq_sanity",
        "system": {
            "A": system.A.tolist(),
            "B": system.B.tolist(),
            "Q": system.Q.tolist(),
            "R": system.R.tolist(),
            "x0": x0.tolist(),
        },
        "params": {
            "H": int(args.H),
            "M": int(args.M),
            "K": int(args.K),
            "lambda": float(args.lambda_),
            "sigma_max": float(args.sigma_max),
            "sigma_min": float(args.sigma_min),
            "seed": int(args.seed),
            "sigma_schedule": [float(s) for s in schedule.tolist()],
        },
        "result": {
            "lq_optimal_cost": float(optimal_cost),
            "zero_action_cost": float(zero_cost),
            "mbd_final_cost": float(mbd_cost),
            "ratio_to_optimum": float(ratio),
            "pass_threshold": float(args.pass_ratio),
            "pass": bool(passed),
        },
        "per_iter": log["per_iter"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"[lq   ] wrote {args.output}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
