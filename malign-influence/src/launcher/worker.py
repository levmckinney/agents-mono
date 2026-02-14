"""Worker implementation for executing jobs from queue"""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from launcher.jobs import JobState

logger = logging.getLogger(__name__)


def setup_file_logging(job_queue_path: Path, worker_id: str) -> None:
    """Setup file handler to save logs to disk in the job queue directory.
    All workers log to a single shared file in append mode.
    """
    log_dir = job_queue_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"worker_{worker_id}.log"

    # Create file handler in append mode
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.DEBUG)

    # Create formatter with worker_id baked in
    formatter = logging.Formatter(
        f"%(asctime)s - %(name)s - %(levelname)s - [worker-{worker_id}] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)

    # Add handler to root logger to capture all logs
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.DEBUG)

    # Also add to module logger
    logger.addHandler(file_handler)

    logger.info(f"Worker initialized, logging to: {log_file}")


TORCH_DISTRIBUTED_BASE_PORT = 29500


class Worker:
    """Worker that executes jobs from queue using subprocess"""

    def __init__(self, gpu_devices: list[int] | None = None):
        self.gpu_devices = gpu_devices or []
        if self.gpu_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_devices))
            logger.info(f"Configured GPU devices: {self.gpu_devices}")
        else:
            logger.info("No GPU devices configured, running on CPU")

    def run(self, job_queue_path: Path, worker_id: str, parallel_jobs: int = 1) -> None:
        """Run worker with parallel job execution"""
        from launcher.jobs import JobQueue

        logger.info(f"Starting worker {worker_id} with {parallel_jobs} parallel job slots")
        logger.info(f"Job queue path: {job_queue_path}")

        running_jobs: dict[str, subprocess.Popen[str]] = {}
        next_port = TORCH_DISTRIBUTED_BASE_PORT

        file_queue = JobQueue(job_array_path=job_queue_path)
        logger.info(f"Connected to job queue at {job_queue_path}")

        def cleanup():
            logger.info(f"Cleanup initiated for {len(running_jobs)} running jobs")
            while running_jobs:
                job_id, p = running_jobs.popitem()
                logger.info(f"Terminating job {job_id}")
                p.terminate()
                p.wait()
                logger.debug(f"Job {job_id} terminated")
            logger.info("Cleanup completed")

        def suspend():
            logger.warning(f"Worker {worker_id} suspended with {len(running_jobs)} running jobs")
            for job_id in running_jobs:
                logger.warning(f"Worker {worker_id} interrupted. Setting job {job_id} to PENDING.")
                file_queue.update_job_state(job_id, JobState.PENDING)

        def signal_handler(signum: int, _: Any) -> None:
            logger.info(f"Worker {worker_id} received signal {signum}")
            suspend()
            exit(1)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.info("Signal handlers registered for SIGINT and SIGTERM")

        try:
            logger.info("Entering main job processing loop")
            # Feed jobs to workers
            while True:
                while len(running_jobs) < parallel_jobs:
                    job = file_queue.get_next_job()

                    if job is None:
                        logger.debug("No pending jobs available in queue")
                        break

                    logger.info(f"Acquired job {job.id} from queue")

                    # Serialize the job to JSON
                    job_json = job.model_dump_json()

                    # Build command to run job
                    if job.resources.use_torch_distributed:
                        # Use torch distributed launch with dedicated port
                        nproc_per_node = job.resources.gpu if job.resources.gpu > 0 else 1
                        port = next_port
                        next_port += 1

                        cmd = [
                            sys.executable,
                            "-m",
                            "torch.distributed.run",
                            f"--nproc_per_node={nproc_per_node}",
                            f"--master_port={port}",
                            "--standalone",
                            "--nnodes=1",
                            "-m",
                            "launcher.jobs",
                            "--debug-on-crash",
                            "600",
                            "--job",
                            job_json,
                        ]
                        logger.info(
                            f"Starting torch distributed job {job.id} with {nproc_per_node} processes on port {port}"
                        )
                    else:
                        # Standard subprocess launch
                        cmd = [
                            sys.executable,
                            "-m",
                            "launcher.jobs",
                            "--debug-on-crash",
                            "600",
                            "--job",
                            job_json
                        ]
                        logger.info(f"Starting subprocess for job {job.id}")

                    p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True)

                    running_jobs[job.id] = p
                    logger.info(
                        f"Job {job.id} subprocess started (PID: {p.pid}), {len(running_jobs)}/{parallel_jobs} slots filled"
                    )

                if not running_jobs:
                    logger.info("No jobs running and no pending jobs in queue, worker shutting down")
                    break

                time.sleep(0.1)  # Prevent busy waiting

                for job_id, proc in list(running_jobs.items()):
                    ret_code = proc.poll()
                    if ret_code is not None:
                        # Process has finished
                        running_jobs.pop(job_id)
                        logger.info(f"Job {job_id} process completed with return code {ret_code}")

                        # Update job state based on return code
                        if ret_code == 0:
                            file_queue.update_job_state(job_id, JobState.COMPLETED)
                            logger.info(
                                f"Job {job_id} marked as COMPLETED, {len(running_jobs)}/{parallel_jobs} slots filled"
                            )
                        else:
                            file_queue.update_job_state(job_id, JobState.FAILED)
                            logger.error(
                                f"Job {job_id} marked as FAILED with return code {ret_code}, {len(running_jobs)}/{parallel_jobs} slots filled"
                            )

            logger.info("Exiting main job processing loop")
        finally:
            logger.info("Worker shutdown initiated")
            cleanup()
            logger.info("Worker shutdown complete")


# Entry point for worker processes
def worker_main() -> None:
    """Entry point for worker processes"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--job-queue-path", type=Path, required=True)
    parser.add_argument("--worker-id", type=str, required=True)
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument("--gpu-devices", type=str, default=None)
    args = parser.parse_args()

    # Setup file logging before anything else
    setup_file_logging(args.job_queue_path, args.worker_id)

    gpu_devices = [int(x) for x in args.gpu_devices.split(",") if x.strip()] if args.gpu_devices else None

    worker = Worker(gpu_devices=gpu_devices)
    worker.run(args.job_queue_path, args.worker_id, args.parallel_jobs)


if __name__ == "__main__":
    worker_main()
