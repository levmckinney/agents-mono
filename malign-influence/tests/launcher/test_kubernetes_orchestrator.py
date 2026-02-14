"""Tests for Kubernetes orchestrator"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from launcher.jobs import Job, JobArray, JobState, ResourceRequest
from launcher.kubernetes_orchestrator import KubernetesConfig, KubernetesSweepOrchestrator


class DummyArgs(BaseModel):
    """Dummy arguments for testing"""

    value: int = 42


def dummy_entrypoint(args: BaseModel) -> None:
    """Dummy entrypoint function"""
    pass


class TestKubernetesConfig:
    """Test KubernetesConfig model"""

    def test_basic_config(self):
        """Test basic configuration creation"""
        config = KubernetesConfig(project_pvc="test-pvc")
        assert config.project_pvc == "test-pvc"
        assert config.queue_name == "farai"
        assert config.priority_class == "interactive"
        assert config.sweep_base_dir == "/home/dev/sweeps"

    def test_custom_config(self):
        """Test custom configuration values"""
        config = KubernetesConfig(
            project_pvc="custom-pvc",
            queue_name="custom-queue",
            priority_class="batch",
            sweep_base_dir="/custom/path",
        )
        assert config.project_pvc == "custom-pvc"
        assert config.queue_name == "custom-queue"
        assert config.priority_class == "batch"
        assert config.sweep_base_dir == "/custom/path"

    @patch("subprocess.check_output")
    def test_image_tag_generation(self, mock_subprocess: MagicMock):
        """Test automatic image tag generation from git"""
        mock_subprocess.side_effect = [
            "main",  # git branch
        ]

        config = KubernetesConfig(project_pvc="test-pvc")
        image_tag = config.get_image_tag()

        assert image_tag == "ghcr.io/alignmentresearch/malign-influence:devbox-main"

    def test_explicit_image(self):
        """Test explicit image specification"""
        config = KubernetesConfig(project_pvc="test-pvc", image="custom-image:v1.0")
        assert config.get_image_tag() == "custom-image:v1.0"


class TestKubernetesSweepOrchestrator:
    """Test KubernetesSweepOrchestrator"""

    def test_orchestrator_creation(self):
        """Test orchestrator creation"""
        config = KubernetesConfig(project_pvc="test-pvc")
        orchestrator = KubernetesSweepOrchestrator(config)
        assert orchestrator.config == config
        assert orchestrator.job_names == []

    def test_sweep_dir_override(self):
        """Test that sweep_dir uses config instead of environment"""
        config = KubernetesConfig(project_pvc="test-pvc", sweep_base_dir="/custom/sweeps")
        orchestrator = KubernetesSweepOrchestrator(config)

        sweep_dir = orchestrator.get_sweep_dir("test-sweep")
        assert sweep_dir == Path("/custom/sweeps/test-sweep")

    @patch("subprocess.run")
    def test_manifest_generation(self, mock_run: MagicMock):
        """Test Kubernetes manifest generation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create template file
            template_path = Path(tmpdir) / "template.yaml"
            template_path.write_text("""
apiVersion: batch/v1
kind: Job
metadata:
  generateName: sweep-__SWEEP_ID__-worker-__WORKER_ID__-
spec:
  suspend: true
  template:
    spec:
      containers:
      - name: worker
        image: __IMAGE__
        resources:
          requests:
            cpu: __CPU__
            memory: __MEMORY__
          limits:
            nvidia.com/gpu: __GPU__
""")

            config = KubernetesConfig(project_pvc="test-pvc", template_path=template_path, image="test-image:v1")
            orchestrator = KubernetesSweepOrchestrator(config)

            resource_request = ResourceRequest(cpu=4, memory=16, gpu=1, parallel_jobs=2)

            manifest = orchestrator.generate_job_manifest(
                sweep_id="test-sweep",
                worker_id=0,
                job_queue_path="/home/dev/sweeps/test-sweep/queue.json",
                resource_request=resource_request,
                git_commit_hash="abc123",
                git_upstream_repo="https://github.com/user/repo.git",
            )

            assert "sweep-test-sweep-worker-0-" in manifest
            assert "image: test-image:v1" in manifest
            assert "cpu: 4" in manifest
            assert "memory: 16.0G" in manifest
            assert "nvidia.com/gpu: 1" in manifest

    @patch("launcher.kubernetes_orchestrator.is_clean_git_repo")
    @patch("subprocess.run")
    def test_job_submission(self, mock_run: MagicMock, mock_is_clean: MagicMock):
        mock_is_clean.return_value = True
        """Test job submission with kubectl"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup
            template_path = Path(tmpdir) / "template.yaml"
            template_path.write_text("apiVersion: batch/v1\nkind: Job\n__SWEEP_ID__ __WORKER_ID__")

            sweep_dir = Path(tmpdir) / "sweeps" / "test-sweep"
            sweep_dir.mkdir(parents=True)

            config = KubernetesConfig(
                project_pvc="test-pvc", template_path=template_path, sweep_base_dir=str(Path(tmpdir) / "sweeps")
            )
            orchestrator = KubernetesSweepOrchestrator(config)

            # Create a simple job array
            job = Job(
                id="test-job",
                args=DummyArgs(),
                state=JobState.PENDING,
                resources=ResourceRequest(cpu=1, memory=4, gpu=0, parallel_jobs=1),
                entrypoint=dummy_entrypoint,
            )
            job_array = JobArray(jobs=[job])

            # Mock kubectl response
            mock_run.return_value = MagicMock(stdout="job.batch/sweep-test-sweep-worker-0-xyz created", returncode=0)

            # Run sweep
            orchestrator.run_sweep(
                job_array=job_array,
                resource_request=ResourceRequest(cpu=1, memory=4, gpu=0, parallel_jobs=1),
                sweep_name="test-sweep",
            )

            # Verify kubectl was called
            assert mock_run.called
            call_args = mock_run.call_args[0][0]
            assert call_args[0] == "kubectl"
            assert call_args[1] == "apply"
            assert call_args[2] == "-f"
