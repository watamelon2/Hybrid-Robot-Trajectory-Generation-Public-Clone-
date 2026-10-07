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
RUN15 = REPO / "experiments/15_unified_hybrid/run_can.py"
RUN05 = REPO / "experiments/05_hybrid_method/run_can.py"
CAN_DP = REPO / "experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt"
SQUARE_DP_DIR = REPO / "experiments/08_flow_matching_prior/results/square_dp_prior_2026_06_07"
SQUARE_CKPTS = {
    "ep0199": SQUARE_DP_DIR / "v_theta_dp_square_tobs2_h16_ep0199.pt",
    "ep0149": SQUARE_DP_DIR / "v_theta_dp_square_tobs2_h16_ep0199_ep0149.pt",
    "ep0099": SQUARE_DP_DIR / "v_theta_dp_square_tobs2_h16_ep0199_ep0099.pt",
    "ep0049": SQUARE_DP_DIR / "v_theta_dp_square_tobs2_h16_ep0199_ep0049.pt",
}
VALIDATION_ROOT = REPO / "experiments/15_unified_hybrid/results/vast_safe_validation_2026_06_10"
DP_ROOT = REPO / "experiments/15_unified_hybrid/results/vast_safe_dp_2026_06_10"
STATIC_ROOT = REPO / "experiments/15_unified_hybrid/results/offload_chain_2026_06_10"
DP_MAX_TOTAL = int(os.environ.get("HRTG_DP_MAX_TOTAL", "36"))
DP_MAX_PER_GPU = int(os.environ.get("HRTG_DP_MAX_PER_GPU", "18"))
DP_LAUNCH_DELAY_SECONDS = int(os.environ.get("HRTG_DP_LAUNCH_DELAY_SECONDS", "6"))
DP_POLL_SECONDS = int(os.environ.get("HRTG_DP_POLL_SECONDS", "10"))
DP_MAX_RETRIES = int(os.environ.get("HRTG_DP_MAX_RETRIES", "1"))
STATIC_MAX_TOTAL = int(os.environ.get("HRTG_STATIC_MAX_TOTAL", "4"))
STATIC_LAUNCH_DELAY_SECONDS = int(os.environ.get("HRTG_STATIC_LAUNCH_DELAY_SECONDS", "20"))
STATIC_POLL_SECONDS = int(os.environ.get("HRTG_STATIC_POLL_SECONDS", "15"))
STATIC_MAX_RETRIES = int(os.environ.get("HRTG_STATIC_MAX_RETRIES", "1"))
SKIP_CLAIMED_LOGS = os.environ.get("HRTG_SKIP_CLAIMED_LOGS", "") not in ("", "0", "false", "False")
COUNT_EXTERNAL_DP = os.environ.get("HRTG_COUNT_EXTERNAL_DP", "") not in ("", "0", "false", "False")


@dataclass(frozen=True, slots=True)
class Job:
    name: str
    command: tuple[str, ...]
    outdir: Path
    gpu: int | None = None
    attempt: int = 0
    marker: Path | None = None


def replace_flag(command: list[str], flag: str, values: list[str]) -> list[str]:
    if flag not in command:
        return [*command, flag, *values]
    i = command.index(flag)
    j = i + 1
    while j < len(command) and not command[j].startswith("--"):
        j += 1
    return [*command[: i + 1], *values, *command[j:]]


def tag_value(value: str) -> str:
    return value.replace(".", "p").replace("-", "m")


def has_unified_outputs(outdir: Path) -> bool:
    return any(outdir.glob("unified_*.json")) and any(outdir.glob("unified_summary_*.json"))


def job_done(job: Job) -> bool:
    if job.marker is not None and job.marker.exists():
        return True
    return has_unified_outputs(job.outdir)


def job_claimed(job: Job) -> bool:
    return any(job.outdir.glob(f"queue_{job.name}_attempt*.log"))


def base_can_dp_command(outdir: Path, suffix: str, sigma_max: str, lambdas: tuple[str, ...], n_rollouts: int, seed0: int) -> list[str]:
    return [
        str(PYTHON), str(RUN15),
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
        "--lambda-prior", *lambdas,
        "--sigma-max", sigma_max,
        "--sigma-min", "0.05",
        "--prior-n-steps", "100",
        "--deterministic-prior-warmstart",
        "--n-demos", "0",
        "--n-rollouts", "0",
        "--n-test-rollouts", str(n_rollouts),
        "--test-start-seed", str(seed0),
        "--device", "cuda:0",
        "--n-workers", "1",
        "--seed", "0",
        "--output-dir", str(outdir),
        "--output-name-suffix", f"_{suffix}",
    ]


def base_square_dp_command(outdir: Path, suffix: str, ckpt_name: str, lambdas: tuple[str, ...], n_rollouts: int, seed0: int) -> list[str]:
    return [
        str(PYTHON), str(RUN15),
        "--task", "square",
        "--prior", "flow",
        "--score-mode", "pull",
        "--prior-checkpoint", str(SQUARE_CKPTS[ckpt_name]),
        "--H", "16",
        "--mpc-chunk-steps", "8",
        "--mpc-replans", "49",
        "--max-env-steps", "400",
        "--K", "10",
        "--M", "16",
        "--lambda-prior", *lambdas,
        "--sigma-max", "0.10",
        "--sigma-min", "0.05",
        "--prior-n-steps", "100",
        "--deterministic-prior-warmstart",
        "--n-demos", "0",
        "--n-rollouts", "0",
        "--n-test-rollouts", str(n_rollouts),
        "--test-start-seed", str(seed0),
        "--device", "cuda:0",
        "--save-step-trace",
        "--n-workers", "1",
        "--seed", "0",
        "--output-dir", str(outdir),
        "--output-name-suffix", f"_{suffix}",
    ]


def make_validation_jobs() -> list[Job]:
    can_dir = VALIDATION_ROOT / "can_dp_validation_lam1_smax0p25"
    square_dir = VALIDATION_ROOT / "square_dp_validation_ep0199_lam2"
    return [
        Job(
            name="validate_can_dp_smax0p25_lam1",
            command=tuple(base_can_dp_command(can_dir, "validate_can_dp_smax0p25_lam1", "0.25", ("1",), 1, 910000)),
            outdir=can_dir,
            gpu=0,
        ),
        Job(
            name="validate_square_dp_ep0199_lam2",
            command=tuple(base_square_dp_command(square_dir, "validate_square_dp_ep0199_lam2", "ep0199", ("2",), 1, 910000)),
            outdir=square_dir,
            gpu=1,
        ),
    ]


def make_dp_jobs() -> list[Job]:
    jobs: list[Job] = []
    shard_size = 6
    counter = 0
    can_lambdas = ("1", "2", "5", "10", "0.5", "0")
    for sigma_max in ("0.25", "0.10"):
        for lam in can_lambdas:
            for offset in range(0, 30, shard_size):
                n_rollouts = min(shard_size, 30 - offset)
                seed0 = 100000 + offset
                name = f"can_dp_ep0199_smax{tag_value(sigma_max)}_lam{tag_value(lam)}_seed{seed0}_n{n_rollouts}"
                outdir = DP_ROOT / name
                jobs.append(Job(
                    name=name,
                    command=tuple(base_can_dp_command(outdir, name, sigma_max, (lam,), n_rollouts, seed0)),
                    outdir=outdir,
                    gpu=counter % 2,
                ))
                counter += 1
    for ckpt_name in ("ep0199", "ep0149", "ep0099", "ep0049"):
        for lam in ("2", "1", "0.5", "0"):
            for offset in range(0, 30, shard_size):
                n_rollouts = min(shard_size, 30 - offset)
                seed0 = 100000 + offset
                name = f"square_dp_{ckpt_name}_lam{tag_value(lam)}_seed{seed0}_n{n_rollouts}"
                outdir = DP_ROOT / name
                jobs.append(Job(
                    name=name,
                    command=tuple(base_square_dp_command(outdir, name, ckpt_name, (lam,), n_rollouts, seed0)),
                    outdir=outdir,
                    gpu=counter % 2,
                ))
                counter += 1
    return jobs


def static_base15(outdir: Path, suffix: str, seed: int = 0) -> tuple[str, ...]:
    return (
        str(PYTHON), str(RUN15),
        "--task", "can",
        "--prior", "bc",
        "--H", "120",
        "--M", "16",
        "--sigma-max", "0.25",
        "--sigma-min", "0.05",
        "--n-demos", "20",
        "--n-rollouts", "3",
        "--n-workers", "4",
        "--seed", str(seed),
        "--output-dir", str(outdir),
        "--output-name-suffix", suffix,
    )


def make_static_jobs() -> list[Job]:
    jobs: list[Job] = []
    kxr = STATIC_ROOT / "bc_kxr_grid"
    for k in (5, 10, 20):
        for r in (0, 1, 2):
            name = f"bc_k{k}_r{r}_lam5"
            command = (*static_base15(kxr, f"_{name}"), "--lambda-prior", "5", "--K", str(k), "--mpc-replans", str(r))
            jobs.append(Job(name=name, command=command, outdir=kxr, marker=kxr / ".markers" / f"{name}.done.json"))
    lams = STATIC_ROOT / "bc_lambda_sweep"
    name = "bc_k10_r2_lam_sweep_no5"
    command = (*static_base15(lams, f"_{name}"), "--lambda-prior", "0", "1", "2", "10", "20", "--K", "10", "--mpc-replans", "2")
    jobs.append(Job(name=name, command=command, outdir=lams, marker=lams / ".markers" / f"{name}.done.json"))
    seeds = STATIC_ROOT / "sigma_seed_pool"
    for seed in (1, 2):
        name = f"exp05_sigma0p05_seed{seed}"
        output = seeds / f"sigma0p05_seed{seed}.json"
        command = (
            str(PYTHON), str(RUN05),
            "--n-demos", "20",
            "--n-rollouts", "3",
            "--seed", str(seed),
            "--no-live-view",
            "--video-dir", "",
            "--n-workers", "4",
            "--sigma-max", "0.05",
            "--deterministic-bc-reset",
            "--output", str(output),
        )
        jobs.append(Job(name=name, command=command, outdir=seeds, marker=seeds / ".markers" / f"{name}.done.json"))
    return jobs


def env_for_job(job: Job) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "MUJOCO_GL": "egl",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    if job.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
    return env


def launch(job: Job) -> subprocess.Popen:
    job.outdir.mkdir(parents=True, exist_ok=True)
    if job.marker is not None:
        job.marker.parent.mkdir(parents=True, exist_ok=True)
    log_path = job.outdir / f"queue_{job.name}_attempt{job.attempt}.log"
    f = log_path.open("ab", buffering=0)
    f.write(f"[vast-safe] start job={job.name} attempt={job.attempt} gpu={job.gpu} {time.ctime()}\n".encode())
    proc = subprocess.Popen(job.command, cwd=str(REPO), env=env_for_job(job), stdout=f, stderr=subprocess.STDOUT)
    proc._log_file = f  # type: ignore[attr-defined]
    return proc


def running_counts(running: dict[subprocess.Popen, Job]) -> dict[int, int]:
    counts = {0: 0, 1: 0}
    for job in running.values():
        if job.gpu is not None:
            counts[job.gpu] = counts.get(job.gpu, 0) + 1
    return counts


def external_dp_counts(root: Path, exclude_pids: set[int]) -> tuple[int, dict[int, int]]:
    counts = {0: 0, 1: 0}
    total = 0
    proc = Path("/proc")
    if not proc.exists():
        return total, counts
    for pid_dir in proc.iterdir():
        if not pid_dir.name.isdigit() or int(pid_dir.name) == os.getpid() or int(pid_dir.name) in exclude_pids:
            continue
        try:
            cmdline = (pid_dir / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore")
        except OSError:
            continue
        if "run_can.py" not in cmdline or str(root) not in cmdline:
            continue
        gpu: int | None = None
        try:
            environ = (pid_dir / "environ").read_bytes().split(b"\x00")
        except OSError:
            environ = []
        for item in environ:
            if item.startswith(b"CUDA_VISIBLE_DEVICES="):
                value = item.split(b"=", 1)[1].decode("utf-8", "ignore").split(",", 1)[0]
                try:
                    gpu = int(value)
                except ValueError:
                    gpu = None
                break
        total += 1
        if gpu is not None:
            counts[gpu] = counts.get(gpu, 0) + 1
    return total, counts


def finish_proc(proc: subprocess.Popen, job: Job, rc: int) -> None:
    log_file = getattr(proc, "_log_file", None)
    if log_file is not None:
        log_file.write(f"[vast-safe] done job={job.name} attempt={job.attempt} rc={rc} {time.ctime()}\n".encode())
        log_file.close()
    if rc == 0 and job.marker is not None:
        job.marker.parent.mkdir(parents=True, exist_ok=True)
        job.marker.write_text(json.dumps({"name": job.name, "elapsed_marker_time": time.time()}, indent=2), encoding="utf-8")


def run_queue(jobs: list[Job], root: Path, max_total: int, max_per_gpu: int | None, launch_delay: int, poll_seconds: int, max_retries: int) -> int:
    root.mkdir(parents=True, exist_ok=True)
    pending = [job for job in jobs if not job_done(job) and not (SKIP_CLAIMED_LOGS and job_claimed(job))]
    running: dict[subprocess.Popen, Job] = {}
    failures: list[tuple[Job, int]] = []
    print(f"[vast-safe] queue_start root={root} total={len(jobs)} pending={len(pending)} max_total={max_total} max_per_gpu={max_per_gpu} {time.ctime()}", flush=True)
    while pending or running:
        for proc, job in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            running.pop(proc, None)
            finish_proc(proc, job, int(rc))
            print(f"[vast-safe] done {job.name} attempt={job.attempt} rc={rc}", flush=True)
            if rc != 0:
                if job.attempt < max_retries:
                    retry = replace(job, attempt=job.attempt + 1)
                    pending.append(retry)
                    print(f"[vast-safe] retry {job.name} next_attempt={retry.attempt}", flush=True)
                else:
                    failures.append((job, int(rc)))
                    print(f"[vast-safe] permanent_failure {job.name} rc={rc}", flush=True)
        external_total, external_counts = external_dp_counts(root, {proc.pid for proc in running}) if COUNT_EXTERNAL_DP else (0, {0: 0, 1: 0})
        counts = running_counts(running)
        for gpu, count in external_counts.items():
            counts[gpu] = counts.get(gpu, 0) + count
        launched = False
        i = 0
        while i < len(pending) and len(running) + external_total < max_total:
            job = pending[i]
            if max_per_gpu is not None and job.gpu is not None and counts.get(job.gpu, 0) >= max_per_gpu:
                i += 1
                continue
            pending.pop(i)
            proc = launch(job)
            running[proc] = job
            if job.gpu is not None:
                counts[job.gpu] = counts.get(job.gpu, 0) + 1
            launched = True
            print(f"[vast-safe] launched {job.name} attempt={job.attempt} gpu={job.gpu} running={len(running)} pending={len(pending)}", flush=True)
            time.sleep(launch_delay)
        if not launched:
            time.sleep(poll_seconds)
    if failures:
        (root / "failures.json").write_text(json.dumps([{"name": job.name, "rc": rc} for job, rc in failures], indent=2), encoding="utf-8")
        print(f"[vast-safe] queue_completed_with_failures n={len(failures)}", flush=True)
        return 1
    print(f"[vast-safe] queue_completed_success {time.ctime()}", flush=True)
    return 0


def validate_paths() -> list[str]:
    required = [PYTHON, RUN15, RUN05, CAN_DP, *SQUARE_CKPTS.values()]
    return [str(path) for path in required if not path.exists()]


def run_validation() -> int:
    missing = validate_paths()
    if missing:
        print("[vast-safe] missing paths:", flush=True)
        for item in missing:
            print(item, flush=True)
        return 2
    rc = run_queue(make_validation_jobs(), VALIDATION_ROOT, max_total=2, max_per_gpu=1, launch_delay=10, poll_seconds=5, max_retries=0)
    if rc != 0:
        return rc
    bad = [job.name for job in make_validation_jobs() if not has_unified_outputs(job.outdir)]
    if bad:
        print(f"[vast-safe] validation_missing_outputs {bad}", flush=True)
        return 3
    print("[vast-safe] validation_passed", flush=True)
    return 0


def launch_detached(session: str, command: str, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tmux", "kill-session", "-t", session], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["tmux", "new-session", "-d", "-s", session, f"{command} > {log} 2>&1"], check=True)


def launch_all() -> int:
    rc = run_validation()
    if rc != 0:
        print(f"[vast-safe] not_launching_after_validation_rc={rc}", flush=True)
        return rc
    this_script = REPO / "experiments/15_unified_hybrid/queue_vast_safe_remote_2026_06_10.py"
    launch_detached("vast_safe_dp", f"{PYTHON} {this_script} --run-dp", DP_ROOT / "orchestrator.log")
    launch_detached("vast_static_offload", f"{PYTHON} {this_script} --run-static", STATIC_ROOT / "remote_static_orchestrator.log")
    subprocess.run(["tmux", "ls"], check=False)
    return 0


def write_manifest() -> None:
    manifest = {
        "validation_root": str(VALIDATION_ROOT),
        "dp_root": str(DP_ROOT),
        "static_root": str(STATIC_ROOT),
        "dp_jobs": len(make_dp_jobs()),
        "static_jobs": len(make_static_jobs()),
        "dp_max_total": DP_MAX_TOTAL,
        "dp_max_per_gpu": DP_MAX_PER_GPU,
        "static_max_total": STATIC_MAX_TOTAL,
    }
    out = REPO / "experiments/15_unified_hybrid/results/vast_safe_remote_2026_06_10_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--run-dp", action="store_true")
    parser.add_argument("--run-static", action="store_true")
    parser.add_argument("--validate-then-launch", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    args = parser.parse_args()
    if args.manifest:
        write_manifest()
        return 0
    if args.validate:
        return run_validation()
    if args.run_dp:
        return run_queue(make_dp_jobs(), DP_ROOT, DP_MAX_TOTAL, DP_MAX_PER_GPU, DP_LAUNCH_DELAY_SECONDS, DP_POLL_SECONDS, DP_MAX_RETRIES)
    if args.run_static:
        return run_queue(make_static_jobs(), STATIC_ROOT, STATIC_MAX_TOTAL, None, STATIC_LAUNCH_DELAY_SECONDS, STATIC_POLL_SECONDS, STATIC_MAX_RETRIES)
    if args.validate_then_launch:
        return launch_all()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
