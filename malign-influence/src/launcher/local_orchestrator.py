"""Local orchestrator for running jobs as subprocesses"""

import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from .orchestrator import SweepOrchestrator

if TYPE_CHECKING:
    from .jobs import JobQueue, ResourceRequest


logger = logging.getLogger(__name__)


class LocalConfig(BaseModel):
    """Configuration for local sweep execution

    Worker count is automatically determined based on available resources:
    - For GPU jobs: uses available GPUs divided by GPUs per job
    - For CPU jobs: uses number of CPU cores
    """

    log_dir: Path = Path("./logs/local")
    venv_activate_script: Path = Path("./.venv/bin/activate")


class LocalSweepOrchestrator(SweepOrchestrator):
    """Launches local subprocess workers"""

    def __init__(self, config: "LocalConfig"):
        self.config = config
        self.processes: list[subprocess.Popen[bytes]] = []

    def launch_workers(self, job_queue: "JobQueue", resource_request: "ResourceRequest", sweep_id: str) -> Any:
        """Launch local subprocess workers"""
        # Import here to avoid circular imports
        from launcher.jobs import get_available_gpus

        # Get available GPUs
        available_gpus = get_available_gpus()

        # Calculate optimal number of workers based on resources
        if resource_request.gpu > 0 and available_gpus:
            # Use all available GPUs efficiently
            num_workers = len(available_gpus) // resource_request.gpu
        else:
            # For CPU-only jobs, use number of CPU cores
            num_workers = os.cpu_count() or 4

        logger.info(f"Launching {num_workers} local workers with {resource_request.gpu} GPUs each")

        for worker_id in range(num_workers):
            # Allocate GPUs to this worker
            if resource_request.gpu > 0 and available_gpus:
                start_gpu = worker_id * resource_request.gpu
                worker_gpus = available_gpus[start_gpu : start_gpu + resource_request.gpu]
            else:
                worker_gpus = []

            # Launch worker subprocess
            process = self._launch_worker_process(
                job_queue.job_array_path, f"local_{worker_id}", worker_gpus, resource_request
            )
            self.processes.append(process)

        logger.info(f"Launched {len(self.processes)} local workers")

    def _launch_worker_process(
        self, job_queue_path: Path, worker_id: str, gpu_devices: list[int], resource_request: "ResourceRequest"
    ) -> subprocess.Popen[bytes]:
        """Launch single worker subprocess"""
        # Create environment
        env = os.environ.copy()
        env["WANDB_START_METHOD"] = "thread"
        # Pass through SWEEP_DIR if set
        if "SWEEP_DIR" in os.environ:
            env["SWEEP_DIR"] = os.environ["SWEEP_DIR"]

        # Build command
        cmd = [
            "bash",
            "-c",
            f"""
source {self.config.venv_activate_script}
python -c "
from launcher.worker import Worker
from pathlib import Path
worker = Worker({gpu_devices})
worker.run(Path('{job_queue_path}'), '{worker_id}', {resource_request.parallel_jobs})
"
""",
        ]

        # Set up logging in sweep directory
        log_dir = job_queue_path.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_file = open(log_dir / f"{worker_id}.out", "w")
        stderr_file = open(log_dir / f"{worker_id}.err", "w")

        # Launch process
        process = subprocess.Popen(cmd, env=env, stdout=stdout_file, stderr=stderr_file)

        logger.info(f"Launched worker {worker_id} (PID {process.pid}) with GPUs {gpu_devices}")
        return process

    def wait_for_completion(self) -> None:
        """Wait for all workers to complete"""
        for process in self.processes:
            process.wait()

        # Check for failures
        failed_workers = [p for p in self.processes if p.returncode != 0]
        if failed_workers:
            logger.warning(f"{len(failed_workers)} workers failed")

        logger.info("All workers completed")
