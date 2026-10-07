from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path

REPO = Path(os.environ.get("HRTG_REMOTE_REPO", "/workspace/hrtg-cloud/HRTG-square-dp-followup"))
PYTHON = Path(os.environ.get("HRTG_REMOTE_PYTHON", "/workspace/hrtg-venv/bin/python"))
SHARED = Path(os.environ.get("HRTG_SHARED_DATA", "/workspace/hrtg-cloud/HRTG-shared-data"))
RUN_CAN = REPO / "experiments/15_unified_hybrid/run_can.py"
ROOT = REPO / "experiments/15_unified_hybrid/results/research_queue_remote_remainder_2026_06_10"
MAX_TOTAL = int(os.environ.get("HRTG_REMAINDER_MAX_TOTAL", "10"))
LAUNCH_DELAY_SECONDS = int(os.environ.get("HRTG_REMAINDER_LAUNCH_DELAY_SECONDS", "10"))
POLL_SECONDS = int(os.environ.get("HRTG_REMAINDER_POLL_SECONDS", "20"))
MAX_RETRIES = int(os.environ.get("HRTG_REMAINDER_MAX_RETRIES", "1"))


@dataclass(frozen=True, slots=True)
class Job:
    name: str
    command: tuple[str, ...]
    outdir: Path
    attempt: int = 0


def q(parts: tuple[str, ...]) -> str:
    return " ".join(parts)


def summary_exists(job: Job) -> bool:
    return any(job.outdir.glob(f"unified_summary_*_{job.name}.json"))


def marker(job: Job) -> Path:
    return job.outdir / ".markers" / f"{job.name}.done.json"


def job_done(job: Job) -> bool:
    return marker(job).exists() or summary_exists(job)


def common_can(outdir: Path, tag: str, n_workers: int) -> list[str]:
    return [
        str(PYTHON), str(RUN_CAN),
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
        "--n-workers", str(n_workers),
        "--seed", "0",
        "--output-dir", str(outdir),
        "--output-name-suffix", f"_{tag}",
    ]


def bank_path(label: str, n: int) -> Path:
    return SHARED / f"can_fric{label}_{n}demos.hdf5"


def adaptation_jobs() -> list[Job]:
    outdir = ROOT / "adaptation_mass_friction"
    jobs: list[Job] = []
    friction_items = (("0p3", "0.3"), ("0p5", "0.5"), ("0p7", "0.7"), ("1p5", "1.5"), ("2p0", "2.0"), ("3p0", "3.0"))
    for label, value in friction_items:
        for bank_n in (0, 205, 220):
            tag = f"can_fric{label}_{'canonicalbank' if bank_n == 0 else f'bank{bank_n}'}_fresh"
            command = common_can(outdir, tag, 2)
            command.extend(["--friction-scale", value])
            if bank_n:
                command.extend(["--prior-bank-dataset", str(bank_path(label, bank_n)), "--prior-bank-task", "can"])
            jobs.append(Job(tag, tuple(command), outdir))
    return jobs


def sparse_jobs() -> list[Job]:
    outdir = ROOT / "sparse_task_extra"
    cells = (
        ("transport", "tr_lam_ablate_trace", ("--lambda-prior", "0", "1", "inf")),
        ("transport", "tr_demo_phase_trace", ("--lambda-prior", "1", "--demo-phase-reward", "--demo-phase-scale", "1000")),
    )
    jobs: list[Job] = []
    for task, tag, extra in cells:
        command = [
            str(PYTHON), str(RUN_CAN),
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
            "--n-workers", "1",
            "--seed", "0",
            "--output-dir", str(outdir),
            "--output-name-suffix", f"_{tag}",
            *extra,
        ]
        jobs.append(Job(tag, tuple(command), outdir))
    return jobs


def make_jobs() -> list[Job]:
    return [*sparse_jobs(), *adaptation_jobs()]


def env_for_job() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "MUJOCO_GL": "egl",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def launch(job: Job) -> subprocess.Popen:
    job.outdir.mkdir(parents=True, exist_ok=True)
    marker(job).parent.mkdir(parents=True, exist_ok=True)
    log_path = job.outdir / f"queue_{job.name}_attempt{job.attempt}.log"
    f = log_path.open("ab", buffering=0)
    f.write(f"[remote-remainder] start job={job.name} attempt={job.attempt} {time.ctime()}\n".encode())
    proc = subprocess.Popen(job.command, cwd=str(REPO), env=env_for_job(), stdout=f, stderr=subprocess.STDOUT)
    proc._log_file = f  # type: ignore[attr-defined]
    return proc


def finish(proc: subprocess.Popen, job: Job, rc: int) -> None:
    log_file = getattr(proc, "_log_file", None)
    if log_file is not None:
        log_file.write(f"[remote-remainder] done job={job.name} attempt={job.attempt} rc={rc} {time.ctime()}\n".encode())
        log_file.close()
    if rc == 0:
        marker(job).write_text(json.dumps({"name": job.name, "time": time.time()}, indent=2), encoding="utf-8")


def validate_paths(jobs: list[Job]) -> list[str]:
    missing = [path for path in (PYTHON, RUN_CAN) if not path.exists()]
    for job in jobs:
        cmd = list(job.command)
        for i, token in enumerate(cmd):
            if token == "--prior-bank-dataset" and i + 1 < len(cmd):
                path = Path(cmd[i + 1])
                if not path.exists():
                    missing.append(path)
    return sorted({str(path) for path in missing})


def write_manifest(jobs: list[Job]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "root": str(ROOT),
        "max_total": MAX_TOTAL,
        "jobs": [{"name": job.name, "outdir": str(job.outdir), "command": list(job.command)} for job in jobs],
    }
    path = ROOT / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(path), "jobs": len(jobs), "max_total": MAX_TOTAL}, indent=2), flush=True)


def run() -> int:
    jobs = make_jobs()
    missing = validate_paths(jobs)
    if missing:
        print("missing required paths:", flush=True)
        for item in missing:
            print(item, flush=True)
        return 2
    write_manifest(jobs)
    pending = [job for job in jobs if not job_done(job)]
    running: dict[subprocess.Popen, Job] = {}
    failures: list[tuple[Job, int]] = []
    print(f"[remote-remainder] queue_start total={len(jobs)} pending={len(pending)} max_total={MAX_TOTAL} {time.ctime()}", flush=True)
    while pending or running:
        for proc, job in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            running.pop(proc)
            finish(proc, job, int(rc))
            print(f"[remote-remainder] done {job.name} attempt={job.attempt} rc={rc}", flush=True)
            if rc != 0:
                if job.attempt < MAX_RETRIES:
                    pending.append(replace(job, attempt=job.attempt + 1))
                    print(f"[remote-remainder] retry {job.name}", flush=True)
                else:
                    failures.append((job, int(rc)))
        while pending and len(running) < MAX_TOTAL:
            job = pending.pop(0)
            proc = launch(job)
            running[proc] = job
            print(f"[remote-remainder] launched {job.name} attempt={job.attempt} running={len(running)} pending={len(pending)}", flush=True)
            time.sleep(LAUNCH_DELAY_SECONDS)
        time.sleep(POLL_SECONDS)
    if failures:
        (ROOT / "failures.json").write_text(json.dumps([{"name": job.name, "rc": rc} for job, rc in failures], indent=2), encoding="utf-8")
        return 1
    print(f"[remote-remainder] queue_completed_success {time.ctime()}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    jobs = make_jobs()
    if args.manifest:
        missing = validate_paths(jobs)
        if missing:
            print("missing required paths:", flush=True)
            for item in missing:
                print(item, flush=True)
            return 2
        write_manifest(jobs)
        return 0
    if args.run:
        return run()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
