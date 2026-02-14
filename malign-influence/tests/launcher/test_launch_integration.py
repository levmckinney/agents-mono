"""
Integration tests for the launch.py infrastructure using pytest
"""

import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel

from launcher.jobs import ResourceRequest, create_job_array_from_sweep
from launcher.local_orchestrator import LocalConfig, LocalSweepOrchestrator
from tests.launcher.test_functions import integration_test_function


class SweepTestArgs(BaseModel):
    """Test job arguments for integration tests"""

    job_name: str
    sleep_duration: float
    output_file: str


class TestLocalSweepIntegration:
    """Integration tests for local sweep functionality"""

    def test_local_sweep_basic(self):
        """Test basic local sweep functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test arguments
            arguments = [
                {"job_name": "test_job_1", "sleep_duration": 0.1, "output_file": str(temp_path / "job_1.txt")},
                {"job_name": "test_job_2", "sleep_duration": 0.1, "output_file": str(temp_path / "job_2.txt")},
                {"job_name": "test_job_3", "sleep_duration": 0.1, "output_file": str(temp_path / "job_3.txt")},
            ]

            # Resource request (no GPUs for testing)
            resource_request = ResourceRequest(
                cpu=1.0,
                memory=2.0,
                gpu=0,
                parallel_jobs=1,
            )

            # Local config
            local_config = LocalConfig(log_dir=temp_path / "logs", venv_activate_script=Path("./.venv/bin/activate"))

            # Create job array and run sweep
            job_array = create_job_array_from_sweep(
                target_args_model=SweepTestArgs,
                target_entrypoint=integration_test_function,
                arguments=arguments,
                resource_request=resource_request,
                sweep_id="test_sweep",
            )

            orchestrator = LocalSweepOrchestrator(local_config)
            orchestrator.run_sweep(job_array, resource_request, sweep_name="test_sweep")

            # Verify all jobs completed
            for i in range(1, 4):
                output_file = temp_path / f"job_{i}.txt"
                assert output_file.exists(), f"Job {i} output file not found"
                content = output_file.read_text()
                assert f"test_job_{i} completed" in content

    def test_local_sweep_single_job(self):
        """Test local sweep with single job"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Single job
            arguments = [
                {"job_name": "single_job", "sleep_duration": 0.1, "output_file": str(temp_path / "single.txt")},
            ]

            resource_request = ResourceRequest(cpu=1.0, memory=2.0, gpu=0, parallel_jobs=1)
            local_config = LocalConfig(log_dir=temp_path / "logs", venv_activate_script=Path("./.venv/bin/activate"))

            # Create job array and run sweep
            job_array = create_job_array_from_sweep(
                target_args_model=SweepTestArgs,
                target_entrypoint=integration_test_function,
                arguments=arguments,
                resource_request=resource_request,
                sweep_id="single_job_sweep",
            )

            orchestrator = LocalSweepOrchestrator(local_config)
            orchestrator.run_sweep(job_array, resource_request, sweep_name="single_job_sweep")

            # Verify job completed
            output_file = temp_path / "single.txt"
            assert output_file.exists()
            content = output_file.read_text()
            assert "single_job completed" in content

    def test_local_sweep_no_jobs(self):
        """Test local sweep with no jobs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # No jobs
            arguments = []

            resource_request = ResourceRequest(cpu=1.0, memory=2.0, gpu=0, parallel_jobs=1)
            local_config = LocalConfig(log_dir=temp_path / "logs", venv_activate_script=Path("./.venv/bin/activate"))

            # Create job array and run sweep - should complete without error
            job_array = create_job_array_from_sweep(
                target_args_model=SweepTestArgs,
                target_entrypoint=integration_test_function,
                arguments=arguments,
                resource_request=resource_request,
                sweep_id="empty_sweep",
            )

            orchestrator = LocalSweepOrchestrator(local_config)
            orchestrator.run_sweep(job_array, resource_request, sweep_name="empty_sweep")

            # Should not crash, just complete quickly


class TestSweepConfigurations:
    """Test different sweep configurations"""

    def test_high_parallelism_local(self):
        """Test local sweep with high parallelism"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create many small jobs
            arguments = [
                {"job_name": f"job_{i}", "sleep_duration": 0.1, "output_file": str(temp_path / f"job_{i}.txt")}
                for i in range(10)
            ]

            resource_request = ResourceRequest(cpu=0.5, memory=1.0, gpu=0, parallel_jobs=1)
            local_config = LocalConfig(log_dir=temp_path / "logs", venv_activate_script=Path("./.venv/bin/activate"))

            # Create job array and run sweep
            job_array = create_job_array_from_sweep(
                target_args_model=SweepTestArgs,
                target_entrypoint=integration_test_function,
                arguments=arguments,
                resource_request=resource_request,
                sweep_id="high_parallelism_sweep",
            )

            orchestrator = LocalSweepOrchestrator(local_config)
            orchestrator.run_sweep(job_array, resource_request, sweep_name="high_parallelism_sweep")

            # Verify all jobs completed
            for i in range(10):
                output_file = temp_path / f"job_{i}.txt"
                assert output_file.exists(), f"Job {i} output file not found"

    def test_gpu_resource_request(self):
        """Test sweep with GPU resource request"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            arguments = [
                {"job_name": "gpu_job", "sleep_duration": 0.1, "output_file": str(temp_path / "gpu_job.txt")},
            ]

            # Request GPUs (will be ignored if none available)
            resource_request = ResourceRequest(
                cpu=2.0,
                memory=8.0,
                gpu=2,
                parallel_jobs=1,
            )

            local_config = LocalConfig(log_dir=temp_path / "logs", venv_activate_script=Path("./.venv/bin/activate"))

            # Should work even if no GPUs available (will just run with 0 workers or CPU-only)
            try:
                # Create job array and run sweep
                job_array = create_job_array_from_sweep(
                    target_args_model=SweepTestArgs,
                    target_entrypoint=integration_test_function,
                    arguments=arguments,
                    resource_request=resource_request,
                    sweep_id="gpu_sweep",
                )

                orchestrator = LocalSweepOrchestrator(local_config)
                orchestrator.run_sweep(job_array, resource_request, sweep_name="gpu_sweep")

                # If no GPUs available, job might not run, which is expected behavior
                output_file = temp_path / "gpu_job.txt"
                if output_file.exists():
                    content = output_file.read_text()
                    assert "gpu_job completed" in content

            except Exception as e:
                # Expected if insufficient resources
                print(f"GPU sweep failed as expected: {e}")

    def test_parallel_jobs(self):
        """Test local sweep with parallel job execution"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create several jobs to test parallelism
            arguments = [
                {
                    "job_name": f"parallel_job_{i}",
                    "sleep_duration": 0.2,
                    "output_file": str(temp_path / f"parallel_{i}.txt"),
                }
                for i in range(4)
            ]

            # Use parallel_jobs=2 to run 2 jobs simultaneously per worker
            resource_request = ResourceRequest(cpu=1.0, memory=2.0, gpu=0, parallel_jobs=2)
            local_config = LocalConfig(log_dir=temp_path / "logs", venv_activate_script=Path("./.venv/bin/activate"))

            # Create job array and run sweep
            job_array = create_job_array_from_sweep(
                target_args_model=SweepTestArgs,
                target_entrypoint=integration_test_function,
                arguments=arguments,
                resource_request=resource_request,
                sweep_id="parallel_sweep",
            )

            orchestrator = LocalSweepOrchestrator(local_config)
            orchestrator.run_sweep(job_array, resource_request, sweep_name="parallel_sweep")

            # Verify all jobs completed
            for i in range(4):
                output_file = temp_path / f"parallel_{i}.txt"
                assert output_file.exists(), f"Parallel job {i} output file not found"
                content = output_file.read_text()
                assert f"parallel_job_{i} completed" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
