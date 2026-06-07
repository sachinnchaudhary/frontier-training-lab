from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


MLA_RUNS = ("small", "medium", "large", "mha")
KIMI_STATE_DIMS = (32, 64, 128, 256)
KIMI_CHUNK_SIZES = (8, 16, 32, 64, 128)
KIMI_GATE_TYPES = ("nogate", "scalar", "vector")
SPARSE_RUNS = ("topk4", "topk8", "topk16", "topk32")
CSA_RATES = (4, 8, 16)
HCA_RATES = (32, 64, 128)
MHC_RUNS = (
    "ordinary_l4",
    "ordinary_l8",
    "ordinary_l12",
    "mhc_l4",
    "mhc_l8",
    "mhc_l12",
)


@dataclass(frozen=True)
class Job:
    job_id: int
    name: str
    module: str
    args: tuple[str, ...]
    log_dir: str

    def command(self, python_exe: str) -> list[str]:
        return [python_exe, "-m", self.module, *self.args]


def kimi_runs() -> tuple[str, ...]:
    return tuple(
        f"state{state_dim}_chunk{chunk_size}_{gate_type}"
        for state_dim in KIMI_STATE_DIMS
        for chunk_size in KIMI_CHUNK_SIZES
        for gate_type in KIMI_GATE_TYPES
    )


def csa_hca_runs() -> tuple[str, ...]:
    return tuple(f"csa{csa}_hca{hca}" for csa in CSA_RATES for hca in HCA_RATES)


def build_jobs() -> list[Job]:
    jobs: list[Job] = []

    def add(name: str, module: str, log_dir: str, *args: str) -> None:
        jobs.append(Job(len(jobs) + 1, name, module, tuple(args), log_dir))

    add(
        "jax_training_default_mhla",
        "jax_training.train",
        "experiment/deepseek_mla_latent_sweep/medium_latent",
    )

    for run_id in MLA_RUNS:
        add(
            f"mla_{run_id}",
            "experiment.deepseek_mla_latent_sweep.run",
            f"experiment/deepseek_mla_latent_sweep/{run_id}",
            "--runs",
            run_id,
        )

    for run_id in kimi_runs():
        add(
            f"kimi_pilot_{run_id}",
            "experiment.kimi_deltanet_memory_sweep.run",
            f"experiment/kimi_deltanet_memory_sweep/pilot/{run_id}",
            "--mode",
            "pilot",
            "--runs",
            run_id,
        )

    for run_id in SPARSE_RUNS:
        add(
            f"sparse_full_{run_id}",
            "experiment.deepseek_sparse_topk_sweep.run",
            f"experiment/deepseek_sparse_topk_sweep/full/{run_id}",
            "--mode",
            "full",
            "--runs",
            run_id,
        )

    for run_id in csa_hca_runs():
        add(
            f"csa_hca_full_{run_id}",
            "experiment.csa_hca_compression_sweep.run",
            f"experiment/csa_hca_compression_sweep/full/{run_id}",
            "--mode",
            "full",
            "--runs",
            run_id,
        )

    for run_id in MHC_RUNS:
        add(
            f"mhc_full_{run_id}",
            "experiment.mhc_depth_scaling_sweep.run",
            f"experiment/mhc_depth_scaling_sweep/full/{run_id}",
            "--mode",
            "full",
            "--runs",
            run_id,
        )

    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full experiment plan as one independent job per GPU."
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=[str(i) for i in range(8)],
        help="GPU ids to use. Default: 0 1 2 3 4 5 6 7.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for child jobs. Default: current Python.",
    )
    parser.add_argument(
        "--log-dir",
        default="experiment/h100_parallel_logs",
        help="Directory for per-job logs and status.jsonl.",
    )
    parser.add_argument(
        "--xla-preallocate",
        default="false",
        choices=("true", "false"),
        help="Value for XLA_PYTHON_CLIENT_PREALLOCATE in child jobs.",
    )
    parser.add_argument(
        "--mem-fraction",
        default=None,
        help="Optional XLA_PYTHON_CLIENT_MEM_FRACTION for child jobs.",
    )
    parser.add_argument(
        "--tf-gpu-allocator",
        default=None,
        help="Optional TF_GPU_ALLOCATOR value, for example cuda_malloc_async.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the job plan without launching child processes.",
    )
    return parser.parse_args()


def write_status(status_path: Path, lock: threading.Lock, payload: dict) -> None:
    payload = {"time": datetime.now().isoformat(timespec="seconds"), **payload}
    line = json.dumps(payload, sort_keys=True)
    with lock:
        with status_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def run_worker(
    gpu_id: str,
    jobs_queue: queue.Queue[Job],
    args: argparse.Namespace,
    log_dir: Path,
    status_path: Path,
    status_lock: threading.Lock,
    failures: list[dict],
) -> None:
    while True:
        try:
            job = jobs_queue.get_nowait()
        except queue.Empty:
            return

        job_log_dir = Path(job.log_dir) / "launcher_logs"
        job_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = job_log_dir / f"{job.job_id:03d}_{job.name}_gpu{gpu_id}.log"
        command = job.command(args.python)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = args.xla_preallocate
        if args.mem_fraction is not None:
            env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = args.mem_fraction
        if args.tf_gpu_allocator is not None:
            env["TF_GPU_ALLOCATOR"] = args.tf_gpu_allocator

        start = time.time()
        write_status(
            status_path,
            status_lock,
            {
                "event": "start",
                "gpu": gpu_id,
                "job": asdict(job),
                "log_path": str(log_path),
                "command": command,
            },
        )
        print(f"[gpu {gpu_id}] start {job.job_id:03d} {job.name}", flush=True)

        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"gpu={gpu_id}\n")
            log_file.write("command=" + " ".join(command) + "\n\n")
            log_file.flush()
            result = subprocess.run(
                command,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )

        elapsed = time.time() - start
        payload = {
            "event": "finish",
            "gpu": gpu_id,
            "job": asdict(job),
            "returncode": result.returncode,
            "elapsed_sec": elapsed,
            "log_path": str(log_path),
        }
        write_status(status_path, status_lock, payload)
        if result.returncode != 0:
            failures.append(payload)
            print(
                f"[gpu {gpu_id}] fail {job.job_id:03d} {job.name} "
                f"rc={result.returncode} log={log_path}",
                flush=True,
            )
        else:
            print(
                f"[gpu {gpu_id}] done {job.job_id:03d} {job.name} "
                f"elapsed={elapsed / 3600:.2f}h",
                flush=True,
            )

        jobs_queue.task_done()


def main() -> int:
    args = parse_args()
    jobs = build_jobs()

    print(f"jobs={len(jobs)}")
    print("gpus=" + ", ".join(args.gpus))

    if args.dry_run:
        for job in jobs:
            print(
                f"{job.job_id:03d} {job.name}: "
                f"log_dir={job.log_dir}/launcher_logs "
                f"cmd={' '.join(job.command(args.python))}"
            )
        return 0

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.jsonl"
    status_lock = threading.Lock()
    failures: list[dict] = []

    jobs_queue: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        jobs_queue.put(job)

    write_status(
        status_path,
        status_lock,
        {
            "event": "launcher_start",
            "jobs": len(jobs),
            "gpus": args.gpus,
            "python": args.python,
            "log_dir": str(log_dir),
        },
    )

    threads = [
        threading.Thread(
            target=run_worker,
            args=(
                gpu_id,
                jobs_queue,
                args,
                log_dir,
                status_path,
                status_lock,
                failures,
            ),
            daemon=False,
        )
        for gpu_id in args.gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    write_status(
        status_path,
        status_lock,
        {
            "event": "launcher_finish",
            "jobs": len(jobs),
            "failures": len(failures),
        },
    )

    if failures:
        print(f"finished with failures={len(failures)}")
        print(f"see {status_path}")
        return 1

    print("finished all jobs successfully")
    print(f"status={status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
