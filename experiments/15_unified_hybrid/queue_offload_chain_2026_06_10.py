from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path("/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup")
PYTHON = Path("/Users/szh/anaconda3/envs/hybrid-traj/bin/python")
RUN15 = REPO / "experiments/15_unified_hybrid/run_can.py"
RUN05 = REPO / "experiments/05_hybrid_method/run_can.py"
RUN06 = REPO / "experiments/06_failure_taxonomy/run.py"
THIS_SCRIPT = REPO / "experiments/15_unified_hybrid/queue_offload_chain_2026_06_10.py"
ROOT_OUT = REPO / "experiments/15_unified_hybrid/results/offload_chain_2026_06_10"
WAIT_SCREENS = (
    "toolhang_phase_probe",
    "rq_coverage_fresh",
    "rq_retrieval_variants",
    "rq_can_dp_calibration",
    "rq_square_dp_checkpoint_calibration",
    "rq_adaptation_mass_friction",
    "rq_sparse_task_extra",
)
N_WORKERS = 10


@dataclass(frozen=True, slots=True)
class Cell:
    name: str
    command: tuple[str, ...]
    marker: Path


def q(parts: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def write_launch_script() -> Path:
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    script = ROOT_OUT / "run_offload_chain.sh"
    log = ROOT_OUT / "queue.log"
    cmd = (str(PYTHON), str(THIS_SCRIPT), "--offload-chain-run")
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        f"LOG={shlex.quote(str(log))}",
        "mkdir -p \"$(dirname \"$LOG\")\"",
        "exec > >(tee -a \"$LOG\") 2>&1",
        q(cmd),
        "exit $?",
    ]
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o755)
    return script


def screen_names() -> list[str]:
    result = subprocess.run(["screen", "-ls"], text=True, capture_output=True)
    lines = result.stdout.splitlines() + result.stderr.splitlines()
    names: list[str] = []
    for line in lines:
        stripped = line.strip()
        if "." not in stripped:
            continue
        first = stripped.split()[0]
        if "." in first:
            names.append(first.split(".", 1)[1])
    return names


def wait_for_current_queues(poll_seconds: int) -> None:
    while True:
        active = sorted(name for name in screen_names() if name in WAIT_SCREENS)
        if not active:
            print("[offload_chain] dependency screens drained")
            return
        print(f"[offload_chain] waiting for screens: {', '.join(active)}")
        sys.stdout.flush()
        time.sleep(poll_seconds)


def run_cell(cell: Cell) -> None:
    cell.marker.parent.mkdir(parents=True, exist_ok=True)
    if cell.marker.exists():
        print(f"[offload_chain] skip {cell.name}: marker exists")
        return
    print(f"[offload_chain] start {cell.name}: {q(cell.command)}")
    sys.stdout.flush()
    started = time.time()
    result = subprocess.run(cell.command)
    elapsed = time.time() - started
    print(f"[offload_chain] done {cell.name}: rc={result.returncode} elapsed_s={elapsed:.1f}")
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    cell.marker.write_text(json.dumps({"name": cell.name, "elapsed_s": elapsed}, indent=2), encoding="utf-8")


def run15_cell(name: str, outdir: Path, args: tuple[str, ...]) -> Cell:
    return Cell(
        name=name,
        command=(str(PYTHON), str(RUN15), *args),
        marker=outdir / ".markers" / f"{name}.done.json",
    )


def run05_cell(name: str, outdir: Path, output: Path, args: tuple[str, ...]) -> Cell:
    return Cell(
        name=name,
        command=(str(PYTHON), str(RUN05), *args, "--output", str(output)),
        marker=outdir / ".markers" / f"{name}.done.json",
    )


def common15(outdir: Path, suffix: str, seed: int = 0) -> tuple[str, ...]:
    return (
        "--task", "can",
        "--prior", "bc",
        "--H", "120",
        "--M", "16",
        "--sigma-max", "0.25",
        "--sigma-min", "0.05",
        "--n-demos", "20",
        "--n-rollouts", "3",
        "--n-workers", str(N_WORKERS),
        "--seed", str(seed),
        "--output-dir", str(outdir),
        "--output-name-suffix", suffix,
    )


def build_static_cells() -> list[Cell]:
    cells: list[Cell] = []
    kxr = ROOT_OUT / "bc_kxr_grid"
    for k in (5, 10, 20):
        for r in (0, 1, 2):
            name = f"bc_k{k}_r{r}_lam5"
            cells.append(run15_cell(
                name,
                kxr,
                (*common15(kxr, f"_{name}"), "--lambda-prior", "5", "--K", str(k), "--mpc-replans", str(r)),
            ))

    lams = ROOT_OUT / "bc_lambda_sweep"
    cells.append(run15_cell(
        "bc_k10_r2_lam_sweep_no5",
        lams,
        (*common15(lams, "_bc_k10_r2_lam_sweep_no5"), "--lambda-prior", "0", "1", "2", "10", "20", "--K", "10", "--mpc-replans", "2"),
    ))

    seeds = ROOT_OUT / "sigma_seed_pool"
    for seed in (1, 2):
        output = seeds / f"sigma0p05_seed{seed}.json"
        cells.append(run05_cell(
            f"exp05_sigma0p05_seed{seed}",
            seeds,
            output,
            (
                "--n-demos", "20",
                "--n-rollouts", "3",
                "--seed", str(seed),
                "--no-live-view",
                "--video-dir", "",
                "--n-workers", str(N_WORKERS),
                "--sigma-max", "0.05",
                "--deterministic-bc-reset",
            ),
        ))
    return cells


def success_rate(path: Path) -> float:
    blob = json.loads(path.read_text())
    return float(blob.get("success_rate", 0.0))


def find_best_kxr() -> tuple[int, int, Path, float]:
    outdir = ROOT_OUT / "bc_kxr_grid"
    best: tuple[int, int, Path, float] | None = None
    for path in sorted(outdir.glob("unified_bc_lam-5.0_K-*_R-*_bc_k*_r*_lam5.json")):
        blob = json.loads(path.read_text())
        k = int(blob["params"]["K"])
        r = int(blob["mpc_replans"])
        rate = float(blob.get("success_rate", 0.0))
        if best is None or rate > best[3]:
            best = (k, r, path, rate)
    if best is None:
        raise SystemExit("[offload_chain] no KxR outputs found")
    return best


def run_best_cell_replications() -> None:
    k, r, path, rate = find_best_kxr()
    meta_path = ROOT_OUT / "bc_kxr_best.json"
    meta_path.write_text(json.dumps({"K": k, "R": r, "source": str(path), "success_rate": rate}, indent=2), encoding="utf-8")
    outdir = ROOT_OUT / "bc_kxr_best_seed_pool"
    for seed in (1, 2):
        name = f"bc_best_k{k}_r{r}_lam5_seed{seed}"
        run_cell(run15_cell(
            name,
            outdir,
            (*common15(outdir, f"_{name}", seed=seed), "--lambda-prior", "5", "--K", str(k), "--mpc-replans", str(r)),
        ))


def run_particle_ablation() -> None:
    outdir = ROOT_OUT / "particle_ablation"
    rates: dict[int, float] = {}
    for m in (16, 64):
        output = outdir / f"sigma0p05_M{m}.json"
        run_cell(run05_cell(
            f"exp05_sigma0p05_M{m}",
            outdir,
            output,
            (
                "--n-demos", "20",
                "--n-rollouts", "3",
                "--seed", "0",
                "--no-live-view",
                "--video-dir", "",
                "--n-workers", str(N_WORKERS),
                "--sigma-max", "0.05",
                "--M", str(m),
                "--deterministic-bc-reset",
            ),
        ))
        rates[m] = success_rate(output)

    if rates[64] - rates[16] >= 0.02:
        output = outdir / "sigma0p05_M256.json"
        run_cell(run05_cell(
            "exp05_sigma0p05_M256",
            outdir,
            output,
            (
                "--n-demos", "20",
                "--n-rollouts", "3",
                "--seed", "0",
                "--no-live-view",
                "--video-dir", "",
                "--n-workers", str(N_WORKERS),
                "--sigma-max", "0.05",
                "--M", "256",
                "--deterministic-bc-reset",
            ),
        ))
        rates[256] = success_rate(output)
        if rates[256] - rates[16] > 0.05:
            output = outdir / "sigma0p05_M1024.json"
            run_cell(run05_cell(
                "exp05_sigma0p05_M1024",
                outdir,
                output,
                (
                    "--n-demos", "20",
                    "--n-rollouts", "3",
                    "--seed", "0",
                    "--no-live-view",
                    "--video-dir", "",
                    "--n-workers", str(N_WORKERS),
                    "--sigma-max", "0.05",
                    "--M", "1024",
                    "--deterministic-bc-reset",
                ),
            ))
            rates[1024] = success_rate(output)
    else:
        print("[offload_chain] skipping M256/M1024: M64 did not clear +2pp gate")

    (outdir / "particle_gate_summary.json").write_text(json.dumps(rates, indent=2), encoding="utf-8")


def run_taxonomy() -> None:
    raw_inputs = sorted((ROOT_OUT / "bc_kxr_grid").glob("*.json"))
    raw_inputs.extend(sorted((ROOT_OUT / "bc_kxr_best_seed_pool").glob("*.json")))
    raw_inputs.extend(sorted((ROOT_OUT / "bc_lambda_sweep").glob("*.json")))
    inputs: list[Path] = []
    for path in raw_inputs:
        blob = json.loads(path.read_text())
        if isinstance(blob.get("per_rollout"), list):
            inputs.append(path)
    converted = ROOT_OUT / "taxonomy" / "inputs"
    converted.mkdir(parents=True, exist_ok=True)
    taxonomy_inputs: list[Path] = []
    for path in inputs:
        blob = json.loads(path.read_text())
        rows = []
        for row in blob.get("per_rollout", []):
            next_row = dict(row)
            if "final_exec_reward" not in next_row and "final_reward" in next_row:
                next_row["final_exec_reward"] = next_row["final_reward"]
            rows.append(next_row)
        blob["per_rollout"] = rows
        out = converted / path.name
        out.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        taxonomy_inputs.append(out)
    output = ROOT_OUT / "taxonomy" / "taxonomy.json"
    command = (str(PYTHON), str(RUN06), "--inputs", *(str(path) for path in taxonomy_inputs), "--output", str(output))
    run_cell(Cell("failure_taxonomy_refresh", command, output.parent / ".markers" / "failure_taxonomy_refresh.done.json"))


def write_manifest() -> Path:
    cells = build_static_cells()
    manifest = {
        "root_out": str(ROOT_OUT),
        "wait_screens": list(WAIT_SCREENS),
        "n_workers": N_WORKERS,
        "static_cells": [{"name": cell.name, "command": list(cell.command)} for cell in cells],
        "dynamic_stages": ["best_kxr_seed_replication", "particle_ablation_gated", "failure_taxonomy_refresh"],
    }
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    path = ROOT_OUT / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def run_chain(poll_seconds: int) -> int:
    write_manifest()
    wait_for_current_queues(poll_seconds)
    for cell in build_static_cells():
        run_cell(cell)
    run_best_cell_replications()
    run_particle_ablation()
    run_taxonomy()
    print("[offload_chain] all done")
    return 0


def launch() -> int:
    script = write_launch_script()
    write_manifest()
    subprocess.run(["screen", "-dmS", "rq_offload_chain", "bash", str(script)], check=True)
    print(f"launched rq_offload_chain with {script}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--offload-chain-run", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()

    if args.offload_chain_run:
        return run_chain(args.poll_seconds)

    manifest = write_manifest()
    script = write_launch_script()
    print(f"manifest={manifest}")
    print(f"script={script}")
    cells = build_static_cells()
    print(f"static_cells={len(cells)} dynamic_stages=3 n_workers={N_WORKERS}")
    if args.dry_run:
        print(f"DRY screen -dmS rq_offload_chain bash {script}")
    if args.launch:
        return launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
