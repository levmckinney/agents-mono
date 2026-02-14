"""Kubernetes orchestrator for running jobs on cluster with Kueue"""

import logging
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .orchestrator import SweepOrchestrator

if TYPE_CHECKING:
    from .jobs import JobQueue, ResourceRequest

logger = logging.getLogger(__name__)


def is_clean_git_repo() -> bool:
    """Check if the current git repository is clean (no uncommitted changes)"""
    return subprocess.run(["git", "diff", "--quiet"]).returncode == 0


def get_current_commit_hash() -> str:
    """Get the current git commit hash"""
    # Fail if not clean
    if not is_clean_git_repo():
        raise RuntimeError("Git working directory is not clean. Please commit or stash changes.")

    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    return commit_hash


def get_current_upstream_repo() -> str:
    """Get the current git upstream repository URL"""
    repo_url = subprocess.check_output(
        ["git", "config", "--get", "remote.origin.url"], text=True, stderr=subprocess.DEVNULL
    ).strip()
    return repo_url


def get_current_branch() -> str:
    """Get the current git branch name"""
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True, stderr=subprocess.DEVNULL).strip()
    return branch


class KubernetesConfig(BaseModel):
    """Configuration for Kubernetes sweep execution from devbox

    Assumes the orchestrator runs from a devbox pod with /home/dev mounted
    from the same PVC that will be used by worker pods.
    """

    # PVC configuration (must match devbox)
    project_pvc: str = Field(..., description="PVC name (e.g., 'lev-colab')")

    # Paths - using /home/dev since it's shared between devbox and workers
    sweep_base_dir: str = "/home/dev/sweeps"  # Where to store sweep data

    # Kubernetes settings
    queue_name: str = "farai"
    priority_class: str = "interactive"
    namespace: str = "default"

    # Template path (relative to working directory)
    template_path: Path = Path("k8s/sweep-worker.template.yaml")
    parallel_workers: int = 1

    # Image settings - auto-detect if not provided
    image: str | None = None
    registry: str = "ghcr.io/alignmentresearch"
    image_name: str = "malign-influence"
    target: str = "devbox"

    def get_image_tag(self) -> str:
        """Generate image tag from git state if not explicitly set"""
        if self.image:
            return self.image

        branch = get_current_branch()
        return f"{self.registry}/{self.image_name}:{self.target}-{branch}"


class KubernetesSweepOrchestrator(SweepOrchestrator):
    """Launches workers as Kubernetes Jobs managed by Kueue

    Designed to run from within a devbox pod that has access to:
    - kubectl via service account
    - Shared PVC mounted at /home/dev
    """

    def __init__(self, config: KubernetesConfig):
        self.config = config
        self.job_names: list[str] = []

    def get_sweep_dir(self, sweep_id: str) -> Path:
        """Override to use Kubernetes config sweep base directory"""
        # Use config sweep_base_dir instead of environment variable
        return Path(self.config.sweep_base_dir) / sweep_id

    def launch_workers(self, job_queue: "JobQueue", resource_request: "ResourceRequest", sweep_id: str) -> None:
        """Launch Kubernetes Jobs for workers"""
        logger.info(f"Launching {self.config.parallel_workers} Kubernetes workers for sweep {sweep_id}")

        # Generate and submit Job manifests
        for worker_id in range(self.config.parallel_workers):
            manifest = self.generate_job_manifest(
                sweep_id=sweep_id,
                worker_id=worker_id,
                job_queue_path=str(job_queue.job_array_path),
                resource_request=resource_request,
                git_commit_hash=get_current_commit_hash(),
                git_upstream_repo=get_current_upstream_repo(),
            )

            # Write manifest to temp file
            with TemporaryDirectory() as temp_dir:
                temp_manifest = Path(temp_dir) / f"job-{sweep_id}-worker-{worker_id}.yaml"
                with open(temp_manifest, "w") as temp_file:
                    temp_file.write(manifest)
                # Submit job using kubectl
                try:
                    result = subprocess.run(
                        ["kubectl", "apply", "-f", str(temp_manifest)], capture_output=True, text=True, check=True
                    )
                    # Extract job name from kubectl output
                    # Format: "job.batch/sweep-abc123-worker-0-xyz created"
                    job_name = (
                        result.stdout.split()[0].split("/")[1]
                        if "/" in result.stdout
                        else f"sweep-{sweep_id}-worker-{worker_id}"
                    )
                    self.job_names.append(job_name)
                    logger.info(f"Submitted worker {worker_id} as job {job_name}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to submit worker {worker_id}: {e.stderr}")
                    raise
                finally:
                    # Clean up temp file
                    if temp_manifest.exists():
                        temp_manifest.unlink()

        logger.info(f"Successfully submitted {len(self.job_names)} worker Jobs to Kubernetes")
        logger.info(f"Monitor with: kubectl get jobs -l sweep-id={sweep_id}")

    def wait_for_completion(self) -> None:
        """No-op - jobs run independently in Kubernetes"""
        if self.job_names:
            logger.info("Jobs submitted to Kubernetes. They will run independently.")
            logger.info(f"Job names: {', '.join(self.job_names)}")
        pass

    def generate_job_manifest(
        self,
        sweep_id: str,
        worker_id: int,
        job_queue_path: str,
        resource_request: "ResourceRequest",
        git_commit_hash: str,
        git_upstream_repo: str,
    ) -> str:
        """Generate Kubernetes Job manifest from template"""

        # Read template
        template_content = self.config.template_path.read_text()

        # Prepare replacements
        replacements = {
            "__SWEEP_ID__": f"{sweep_id}",
            "__WORKER_ID__": f"w{worker_id}",
            "__QUEUE_NAME__": self.config.queue_name,
            "__PRIORITY_CLASS__": self.config.priority_class,
            "__IMAGE__": self.config.get_image_tag(),
            "__JOB_QUEUE_PATH__": job_queue_path,
            "__PROJECT_PVC__": self.config.project_pvc,
            "__CPU__": str(resource_request.cpu),
            "__MEMORY__": f"{resource_request.memory}G",
            "__UPSTREAM_REPO__": git_upstream_repo,
            "__COMMIT_HASH__": git_commit_hash,
            "__GPU__": str(resource_request.gpu),
            "__PARALLEL_JOBS__": str(resource_request.parallel_jobs),
            "__SWEEP_BASE_DIR__": self.config.sweep_base_dir,
        }

        # Replace all placeholders
        manifest = template_content
        for key, value in replacements.items():
            manifest = manifest.replace(key, value)

        return manifest
