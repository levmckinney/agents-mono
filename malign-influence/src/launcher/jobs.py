import base64
import logging
import os
import pickle
import subprocess
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generator

from filelock import FileLock
from pydantic import BaseModel, Field, field_serializer, field_validator

from launcher.crash_debugger import debug_on_crash

logger = logging.getLogger(__name__)


class ResourceRequest(BaseModel):
    """Resources required for a job"""

    cpu: float = Field(..., description="Number of CPU cores required")
    memory: float = Field(..., description="Amount of memory in GB required")
    gpu: int = Field(default=0, description="Number of GPUs required, default is 0")
    parallel_jobs: int = Field(default=1, description="Number of parallel jobs a worker can run")
    use_torch_distributed: bool = Field(default=False, description="Whether to launch job using torch distributed")


def get_available_gpus() -> list[int]:
    """Get list of available GPU IDs from CUDA_VISIBLE_DEVICES or nvidia-smi"""
    # First check CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible:
        try:
            return [int(x) for x in cuda_visible.split(",") if x.strip()]
        except ValueError:
            logger.warning(f"Invalid CUDA_VISIBLE_DEVICES: {cuda_visible}, falling back to nvidia-smi")

    # Fall back to nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, check=True
        )
        return [int(line.strip()) for line in result.stdout.strip().split("\n") if line.strip()]
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        logger.warning("Could not detect GPUs, assuming no GPUs available")
        return []


class JobState(Enum):
    """Possible states for a job"""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Job(BaseModel):
    """A single job that needs to be launched"""

    id: str = Field(..., description="Unique identifier for the job")
    args: BaseModel = Field(..., description="The arguments for the entrypoint function")
    state: JobState = Field(..., description="Current state of the job")
    resources: ResourceRequest = Field(..., description="Resources required for the job")
    entrypoint: Callable[[BaseModel], None] = Field(..., description="The function to be executed for the job")

    def _serialize_with_pickle(self, obj: Any) -> str:
        """Helper to serialize any object using pickle and base64 encoding"""
        pickled_data = pickle.dumps(obj)
        return base64.b64encode(pickled_data).decode("utf-8")

    def _deserialize_with_pickle(self, value: Any) -> Any:
        """Helper to deserialize from base64 encoded pickle data"""
        if isinstance(value, str):
            pickled_data = base64.b64decode(value.encode("utf-8"))
            return pickle.loads(pickled_data)
        return value

    @field_serializer("args")
    def serialize_args(self, args: BaseModel) -> str:
        """Serialize args using pickle and base64 encoding"""
        return self._serialize_with_pickle(args)

    @field_validator("args", mode="before")
    @classmethod
    def deserialize_args(cls, value: Any) -> BaseModel:
        """Deserialize args from base64 encoded pickle data"""
        if isinstance(value, str):
            pickled_data = base64.b64decode(value.encode("utf-8"))
            return pickle.loads(pickled_data)
        return value

    @field_serializer("entrypoint")
    def serialize_entrypoint(self, entrypoint: Callable[[BaseModel], None]) -> str:
        """Serialize callable using pickle and base64 encoding"""
        return self._serialize_with_pickle(entrypoint)

    @field_validator("entrypoint", mode="before")
    @classmethod
    def deserialize_entrypoint(cls, value: Any) -> Callable[[BaseModel], None]:
        """Deserialize callable from base64 encoded pickle data"""
        if isinstance(value, str):
            pickled_data = base64.b64decode(value.encode("utf-8"))
            return pickle.loads(pickled_data)
        return value


class JobArray(BaseModel):
    """A set of jobs that need to be launched"""

    jobs: list[Job] = Field(..., description="List of jobs to be launched")


class JobQueue(BaseModel):
    """A queue of jobs to be executed"""

    job_array_path: Path = Field(..., description="Path to the job array file")

    @contextmanager
    def _job_array(self) -> Generator[JobArray, None, None]:
        """Context manager to read/write job array from/to file"""
        lock_path = self.job_array_path.with_suffix(".lock")
        with FileLock(str(lock_path)):
            if self.job_array_path.exists():
                with open(self.job_array_path, "r") as f:
                    data = f.read()
                    job_array = JobArray.model_validate_json(data)
            else:
                job_array = JobArray(jobs=[])

            yield job_array

            with open(self.job_array_path, "w") as f:
                f.write(job_array.model_dump_json(indent=2))

    def initialize_with_jobs(self, jobs: list[Job]) -> None:
        """Initialize queue with jobs"""
        self.job_array_path.parent.mkdir(parents=True, exist_ok=True)
        with self._job_array() as job_array:
            job_array.jobs = jobs

    def add_job(self, job: Job) -> None:
        """Add a job to the queue"""
        with self._job_array() as job_array:
            job_array.jobs.append(job)

    def get_next_job(self) -> Job | None:
        """Gets next pending job and marks it as running"""
        with self._job_array() as job_array:
            for job in job_array.jobs:
                if job.state == JobState.PENDING:
                    job.state = JobState.RUNNING
                    return job
            return None

    def update_job_state(self, job_id: str, state: JobState) -> None:
        """Update the state of a job"""
        with self._job_array() as job_array:
            for job in job_array.jobs:
                if job.id == job_id:
                    job.state = state
                    break

    def _count_jobs_by_state(self, state: JobState) -> int:
        """Helper to count jobs in a specific state"""
        with self._job_array() as job_array:
            return sum(1 for job in job_array.jobs if job.state == state)

    def count_pending_jobs(self) -> int:
        """Count jobs in pending state"""
        return self._count_jobs_by_state(JobState.PENDING)

    def count_running_jobs(self) -> int:
        """Count jobs in running state"""
        return self._count_jobs_by_state(JobState.RUNNING)

    def count_completed_jobs(self) -> int:
        """Count jobs in completed state"""
        return self._count_jobs_by_state(JobState.COMPLETED)

    def count_failed_jobs(self) -> int:
        """Count jobs in failed state"""
        return self._count_jobs_by_state(JobState.FAILED)


def create_job_array_from_sweep(
    target_args_model: type[BaseModel],
    target_entrypoint: Callable[[BaseModel], None],
    arguments: list[dict[str, Any]],
    resource_request: "ResourceRequest",
    sweep_id: str,
) -> JobArray:
    """Convert sweep arguments into Job objects"""
    jobs = []

    for i, arg_dict in enumerate(arguments):
        # Validate and create args instance
        validated_args = target_args_model.model_validate(arg_dict)

        # Create job with validated args instance
        job = Job(
            id=f"{sweep_id}_{i}",
            args=validated_args,
            state=JobState.PENDING,
            resources=resource_request,
            entrypoint=target_entrypoint,
        )
        jobs.append(job)

    return JobArray(jobs=jobs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Execute a serialized job")
    parser.add_argument("--job", required=True, help="Base64-encoded serialized job JSON")
    parser.add_argument(
        "--debug-on-crash",
        type=float,
        nargs="?",
        const=600,
        default=None,
        metavar="TIMEOUT",
        help="Launch remote PDB server on crash (optional timeout in seconds, default: 600)",
    )

    args = parser.parse_args()

    # Deserialize the job from command line argument
    job = Job.model_validate_json(args.job)

    # Execute the job's entrypoint with its arguments
    logger.info(f"Executing job {job.id}")

    if args.debug_on_crash is not None:
        with debug_on_crash(timeout=args.debug_on_crash):
            job.entrypoint(job.args)
    else:
        job.entrypoint(job.args)
