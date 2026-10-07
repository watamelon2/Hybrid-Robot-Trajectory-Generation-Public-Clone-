from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path("/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup")
PYTHON = Path("/Users/szh/anaconda3/envs/hybrid-traj/bin/python")
RUN_CAN = REPO / "experiments/15_unified_hybrid/run_can.py"
ROOT_OUT = REPO / "experiments/15_unified_hybrid/results/research_queue_2026_06_10"
SHARED = Path("/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data")
CAN_DP = REPO / "experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt"
SQUARE_DP_DIR = REPO / "experiments/08_flow_matching_prior/results/square_dp_prior_2026_06_07"
SQUARE_CKPTS = {
    "ep0049": SQUARE_DP_DIR / "v_theta_dp_square_tobs2_h16_ep0199_ep0049.pt",
    "ep0099": SQUARE_DP_DIR / "v_theta_dp_square_tobs2_h16_ep0199_ep0099.pt",
    "ep0149": SQUARE_DP_DIR / "v_theta_dp_square_tobs2_h16_ep0199_ep0149.pt",
    "ep0199": SQUARE_DP_DIR / "v_theta_dp_square_tobs2_h16_ep0199.pt",
}


@dataclass(frozen=True, slots=True)
class Experiment:
    name: str
    args: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)

    def command(self) -> list[str]:
        return [str(PYTHON), str(RUN_CAN), *self.args]

    def shell_command(self) -> str:
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(self.env.items()))
        cmd = " ".join(shlex.quote(part) for part in self.command())
        return f"{env_prefix} {cmd}".strip()


@dataclass(frozen=True, slots=True)
class Lane:
    name: str
    n_workers: int
    experiments: tuple[Experiment, ...]


def base_common(outdir: Path, tag: str, n_workers: int) -> list[str]:
    return [
        "--n-workers", str(n_workers),
        "--seed", "0",
        "--output-dir", str(outdir),
        "--output-name-suffix", f"_{tag}",
    ]


def can_retrieval_args(outdir: Path, tag: str, n_workers: int) -> list[str]:
    return [
        "--task", "can",
        "--prior", "demo_1nn",
        "--lambda-prior", "1",
        "--combine-mode", "score",
        "--H", "120",
        "--mpc-replans", "1",
        "--mpc-chunk-steps", "60",
        "--max-env-steps", "120",
        "--K", "10",
        "--M", "16",
        "--sigma-max", "0.25",
        "--sigma-min", "0.05",
        "--n-demos", "0",
        "--n-rollouts", "0",
        "--n-test-rollouts", "30",
        "--test-start-seed", "100000",
        *base_common(outdir, tag, n_workers),
    ]


def build_coverage_lane() -> Lane:
    outdir = ROOT_OUT / "coverage_fresh"
    experiments: list[Experiment] = []
    for bank_size in (1, 2, 5, 10, 20, 50, 100, 200):
        for seed0 in (100000, 200000):
            tag = f"can_fresh_bank{bank_size}_seed{seed0}"
            args = can_retrieval_args(outdir, tag, 2)
            args.extend(["--prior-bank-size", str(bank_size)])
            args[args.index("--test-start-seed") + 1] = str(seed0)
            experiments.append(Experiment(tag, tuple(args)))
    return Lane("coverage_fresh", 2, tuple(experiments))


def build_retrieval_variant_lane() -> Lane:
    outdir = ROOT_OUT / "retrieval_variants"
    experiments: list[Experiment] = []
    variants: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("demo_1nn", ()),
        ("demo_knn", ("--knn-k", "5")),
        ("demo_lwr", ("--knn-k", "5")),
        ("demo_topk_random", ("--knn-k", "5")),
        ("demo_softnn", ("--knn-k", "5", "--retrieval-beta", "0.5")),
        ("demo_softnn", ("--knn-k", "5", "--retrieval-beta", "2.0")),
        ("demo_1nn_proprio", ()),
        ("demo_1nn_object", ()),
    )
    for prior, extra in variants:
        suffix = prior
        if "--retrieval-beta" in extra:
            suffix = f"{prior}_beta{extra[extra.index('--retrieval-beta') + 1].replace('.', 'p')}"
        for bank_size in (20, 200):
            tag = f"can_{suffix}_bank{bank_size}_fresh"
            args = can_retrieval_args(outdir, tag, 2)
            args[args.index("--prior") + 1] = prior
            args.extend(["--prior-bank-size", str(bank_size), *extra])
            experiments.append(Experiment(tag, tuple(args)))
    return Lane("retrieval_variants", 2, tuple(experiments))


def build_can_dp_lane() -> Lane:
    outdir = ROOT_OUT / "can_dp_calibration"
    experiments: list[Experiment] = []
    for sigma_max in (0.10, 0.25):
        tag = f"can_dp_ep0199_smax{str(sigma_max).replace('.', 'p')}_lamgrid"
        args = [
            "--task", "can",
            "--prior", "flow",
            "--score-mode", "pull",
            "--prior-checkpoint", str(CAN_DP),
            "--H", "16",
            "--mpc-chunk-steps", "8",
            "--mpc-replans", "29",
            "--max-env-steps", "240",
            "--K", "10",
            "--M", "16",
            "--lambda-prior", "0", "0.5", "1", "2", "5", "10",
            "--sigma-max", str(sigma_max),
            "--sigma-min", "0.05",
            "--prior-n-steps", "100",
            "--deterministic-prior-warmstart",
            "--n-demos", "0",
            "--n-rollouts", "0",
            "--n-test-rollouts", "30",
            "--test-start-seed", "100000",
            "--device", "cpu",
            *base_common(outdir, tag, 2),
        ]
        experiments.append(Experiment(tag, tuple(args)))
    return Lane("can_dp_calibration", 2, tuple(experiments))


def build_square_dp_lane() -> Lane:
    outdir = ROOT_OUT / "square_dp_checkpoint_calibration"
    experiments: list[Experiment] = []
    for ckpt_name, ckpt in SQUARE_CKPTS.items():
        tag = f"square_dp_{ckpt_name}_lamgrid_trace"
        args = [
            "--task", "square",
            "--prior", "flow",
            "--score-mode", "pull",
            "--prior-checkpoint", str(ckpt),
            "--H", "16",
            "--mpc-chunk-steps", "8",
            "--mpc-replans", "49",
            "--max-env-steps", "400",
            "--K", "10",
            "--M", "16",
            "--lambda-prior", "0", "0.5", "1", "2",
            "--sigma-max", "0.10",
            "--sigma-min", "0.05",
            "--prior-n-steps", "100",
            "--deterministic-prior-warmstart",
            "--n-demos", "0",
            "--n-rollouts", "0",
            "--n-test-rollouts", "30",
            "--test-start-seed", "100000",
            "--device", "cpu",
            "--save-step-trace",
            *base_common(outdir, tag, 2),
        ]
        experiments.append(Experiment(tag, tuple(args)))
    return Lane("square_dp_checkpoint_calibration", 2, tuple(experiments))


def bank_path(kind: str, value_tag: str, n: int) -> Path:
    return SHARED / f"can_{kind}{value_tag}_{n}demos.hdf5"


def build_adaptation_lane() -> Lane:
    outdir = ROOT_OUT / "adaptation_mass_friction"
    experiments: list[Experiment] = []
    mass_items = (("0p5", "0.5"), ("1p3", "1.3"), ("1p6", "1.6"), ("2p5", "2.5"))
    friction_items = (("0p3", "0.3"), ("0p5", "0.5"), ("0p7", "0.7"), ("1p5", "1.5"), ("2p0", "2.0"), ("3p0", "3.0"))
    for label, value in mass_items:
        for bank_n in (0, 205, 220):
            tag = f"can_mass{label}_{'canonicalbank' if bank_n == 0 else f'bank{bank_n}'}_fresh"
            args = can_retrieval_args(outdir, tag, 2)
            args.extend(["--mass-scale", value])
            if bank_n:
                args.extend(["--prior-bank-dataset", str(bank_path("mass", label, bank_n)), "--prior-bank-task", "can"])
            experiments.append(Experiment(tag, tuple(args)))
    for label, value in friction_items:
        for bank_n in (0, 205, 220):
            tag = f"can_fric{label}_{'canonicalbank' if bank_n == 0 else f'bank{bank_n}'}_fresh"
            args = can_retrieval_args(outdir, tag, 2)
            args.extend(["--friction-scale", value])
            if bank_n:
                args.extend(["--prior-bank-dataset", str(bank_path("fric", label, bank_n)), "--prior-bank-task", "can"])
            experiments.append(Experiment(tag, tuple(args)))
    return Lane("adaptation_mass_friction", 2, tuple(experiments))


def build_sparse_extra_lane() -> Lane:
    outdir = ROOT_OUT / "sparse_task_extra"
    experiments: list[Experiment] = []
    cells: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        ("tool_hang", "th_lam_ablate_trace", ("--lambda-prior", "0", "1", "inf")),
        ("tool_hang", "th_raw_sparse_trace", ("--lambda-prior", "1", "--raw-sparse-reward")),
        ("tool_hang", "th_demo_phase_trace", ("--lambda-prior", "1", "--demo-phase-reward", "--demo-phase-scale", "1000")),
        ("transport", "tr_lam_ablate_trace", ("--lambda-prior", "0", "1", "inf")),
        ("transport", "tr_demo_phase_trace", ("--lambda-prior", "1", "--demo-phase-reward", "--demo-phase-scale", "1000")),
    )
    for task, tag, extra in cells:
        args = [
            "--task", task,
            "--prior", "demo_1nn",
            "--combine-mode", "score",
            "--H", "300",
            "--mpc-replans", "3",
            "--mpc-chunk-steps", "75",
            "--max-env-steps", "300",
            "--K", "10",
            "--M", "16",
            "--sigma-max", "0.25",
            "--sigma-min", "0.05",
            "--n-demos", "5",
            "--n-rollouts", "1",
            "--save-step-trace",
            "--save-executed-actions",
            "--log-gate-diagnostic",
            *base_common(outdir, tag, 1),
            *extra,
        ]
        experiments.append(Experiment(tag, tuple(args)))
    return Lane("sparse_task_extra", 1, tuple(experiments))


def build_lanes() -> tuple[Lane, ...]:
    return (
        build_coverage_lane(),
        build_retrieval_variant_lane(),
        build_can_dp_lane(),
        build_square_dp_lane(),
        build_adaptation_lane(),
        build_sparse_extra_lane(),
    )


def validate_paths(lanes: tuple[Lane, ...]) -> list[str]:
    missing: list[str] = []
    required = [PYTHON, RUN_CAN, CAN_DP, *SQUARE_CKPTS.values()]
    for path in required:
        if not path.exists():
            missing.append(str(path))
    for lane in lanes:
        for experiment in lane.experiments:
            args = list(experiment.args)
            for flag in ("--prior-bank-dataset", "--prior-checkpoint"):
                for index, value in enumerate(args):
                    if value == flag and index + 1 < len(args):
                        path = Path(args[index + 1])
                        if not str(path).startswith("<") and not path.exists():
                            missing.append(str(path))
    return sorted(set(missing))


def write_lane_script(lane: Lane) -> Path:
    lane_dir = ROOT_OUT / "launchers"
    lane_dir.mkdir(parents=True, exist_ok=True)
    script = lane_dir / f"run_{lane.name}.sh"
    log = ROOT_OUT / lane.name / "queue.log"
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        f"LOG={shlex.quote(str(log))}",
        "mkdir -p \"$(dirname \"$LOG\")\"",
        "exec > >(tee -a \"$LOG\") 2>&1",
        f"echo '[lane {lane.name}] start '$(date)",
        f"echo '[lane {lane.name}] repo={REPO}'",
        f"echo '[lane {lane.name}] worker_budget={lane.n_workers}'",
    ]
    for experiment in lane.experiments:
        lines.extend([
            "",
            f"echo '[lane {lane.name}] cell {experiment.name} start '$(date)",
            experiment.shell_command(),
            "rc=$?",
            f"echo '[lane {lane.name}] cell {experiment.name} done rc='$rc' '$(date)",
            "if [ \"$rc\" -ne 0 ]; then exit \"$rc\"; fi",
        ])
    lines.extend(["", f"echo '[lane {lane.name}] all done '$(date)"])
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o755)
    return script


def write_manifest(lanes: tuple[Lane, ...], scripts: dict[str, Path]) -> Path:
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "root_out": str(ROOT_OUT),
        "lanes": [
            {
                "name": lane.name,
                "n_workers": lane.n_workers,
                "script": str(scripts[lane.name]),
                "n_experiments": len(lane.experiments),
                "experiments": [
                    {"name": experiment.name, "command": experiment.command()}
                    for experiment in lane.experiments
                ],
            }
            for lane in lanes
        ],
    }
    path = ROOT_OUT / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def launch_lane(lane: Lane, script: Path, dry_run: bool) -> None:
    screen_name = f"rq_{lane.name}"
    if dry_run:
        print(f"DRY screen -dmS {screen_name} bash {script}")
        return
    subprocess.run(["screen", "-dmS", screen_name, "bash", str(script)], check=True)
    print(f"launched {screen_name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lanes = build_lanes()
    missing = validate_paths(lanes)
    if missing:
        print("missing required paths:")
        for item in missing:
            print(item)
        return 2

    scripts: dict[str, Path] = {}
    if args.write or args.launch or args.dry_run:
        for lane in lanes:
            scripts[lane.name] = write_lane_script(lane)
        manifest = write_manifest(lanes, scripts)
        print(f"manifest={manifest}")

    total_experiments = sum(len(lane.experiments) for lane in lanes)
    total_workers = sum(lane.n_workers for lane in lanes)
    print(f"lanes={len(lanes)} experiments={total_experiments} worker_budget={total_workers}")
    for lane in lanes:
        print(f"{lane.name}: experiments={len(lane.experiments)} workers={lane.n_workers}")

    if args.launch or args.dry_run:
        for lane in lanes:
            launch_lane(lane, scripts[lane.name], dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
