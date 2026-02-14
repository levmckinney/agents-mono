"""Base orchestrator for sweep execution"""

import os
import random
import string
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .jobs import JobArray, JobQueue, ResourceRequest


class SweepOrchestrator(ABC):
    """Orchestrates sweep execution by launching workers"""

    @abstractmethod
    def launch_workers(self, job_queue: "JobQueue", resource_request: "ResourceRequest", sweep_id: str) -> Any:
        """Launch workers to process the job queue"""
        pass

    @abstractmethod
    def wait_for_completion(self) -> None:
        """Wait for all workers to complete"""
        pass

    @staticmethod
    def generate_sweep_id(sweep_name: str = "sweep") -> str:
        """Generate a unique sweep ID with random suffix to prevent collisions"""
        random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        return f"{sweep_name}-{random_suffix}"

    def get_sweep_dir(self, sweep_id: str) -> Path:
        """Get the sweep directory path based on environment variable or default"""
        sweep_base = os.environ.get("SWEEP_DIR", "/tmp/sweeps")
        return Path(sweep_base) / sweep_id

    def run_sweep(
        self,
        job_array: "JobArray",
        resource_request: "ResourceRequest",
        sweep_name: str,
    ) -> None:
        """Main sweep execution logic"""
        # Import here to avoid circular imports
        from launcher.jobs import JobQueue

        # Generate unique sweep ID if not provided

        # Use environment-based sweep directory
        sweep_id = self.generate_sweep_id(sweep_name)
        sweep_dir = self.get_sweep_dir(sweep_id)
        sweep_dir.mkdir(parents=True, exist_ok=True)

        queue_path = sweep_dir / "sweep_queue.json"
        job_queue = JobQueue(job_array_path=queue_path)

        # Populate with jobs
        job_queue.initialize_with_jobs(job_array.jobs)

        # Launch workers
        self.launch_workers(job_queue, resource_request, sweep_id)

        # Wait for completion
        self.wait_for_completion()
